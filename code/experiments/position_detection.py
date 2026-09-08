#!/usr/bin/env python3
"""Matched two-sentence localization with downstream capture and restoration."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import warnings

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code" / "utils"))
from all_prompts import LOCALIZATION_SENTENCES
from experiment_progress import RunProgress
from position_detection_utils import (
    ALPHAS, CONCEPTS, LAYERS, SCHEMA_VERSION, condition_metrics, digest,
    forward_counts, select_pairs,
)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_tensor(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def file_digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def vector_inventory(args):
    inventory = []
    for concept in args.concepts:
        for layer in args.layers:
            path = args.vector_dir / f"{concept}_{layer}_{args.vec_type}.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Required vector is missing: {path}")
            data = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(data, dict) or "vector" not in data:
                raise ValueError(f"Vector file lacks provenance metadata: {path}")
            inventory.append({
                "concept": concept, "layer": layer, "path": str(path.resolve()),
                "sha256": file_digest(path),
                "saved_metadata": {k: v for k, v in data.items() if k != "vector"},
                "source_hidden_states_index": layer,
                "source_location": "embedding output" if layer == 0 else f"block {layer - 1} output",
                "injection_location": f"model.model.layers.{layer} output (before final model norm)",
                "location_mismatch": True,
                "provenance_basis": "repository compute_vector_single_prompt; file metadata does not encode hook location",
            })
    return inventory


def load_vector(info, args, hidden_size):
    data = torch.load(info["path"], map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or "vector" not in data:
        raise ValueError(f"Vector file must contain vector and provenance metadata: {info['path']}")
    # Existing artifacts use the older repository spelling for the same checkpoint.
    known_model_names = {"meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct"}
    if data.get("model_name") != args.model:
        if {data.get("model_name"), args.model} <= known_model_names:
            warnings.warn("Vector metadata uses the legacy Meta-Llama-3.1-8B-Instruct repository name; "
                          "accepting it for Llama-3.1-8B-Instruct. Original metadata is saved in the manifest.")
        else:
            raise ValueError(f"Vector model {data.get('model_name')!r} differs from {args.model!r}")
    for field, expected in (("layer", info["layer"]),
                            ("concept_name", info["concept"]), ("vec_type", args.vec_type)):
        if data.get(field) != expected:
            raise ValueError(f"Vector {info['path']}: {field}={data.get(field)!r}, expected {expected!r}")
    vector = torch.as_tensor(data["vector"]).float().reshape(-1)
    if vector.numel() != hidden_size or not torch.isfinite(vector).all() or vector.norm() == 0:
        raise ValueError(f"Invalid vector shape, values or norm: {info['path']}")
    return vector / vector.norm()


def forward(model, prompt, *, injection=None, capture_layers=(), restoration=None):
    """Each call starts with a fresh full prompt and no KV cache.

    injection: (block index, physical target position 1/2, unit vector, alpha).
    restoration: (block index, clean second-sentence tensor).
    Captures are raw block outputs, after any modification at that block.
    """
    captures, perturbation = {}, {}
    handles = []
    layers = model.model.layers
    device = model.get_input_embeddings().weight.device
    input_ids = torch.tensor([prompt["input_ids"]], device=device)
    capture_layers = set(capture_layers)
    active = capture_layers.copy()
    if injection is not None:
        active.add(injection[0])
    if restoration is not None:
        active.add(restoration[0])
    second_start, second_end = prompt["sentences"][1]["token_span"]

    def make_hook(layer):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.shape[:2] != input_ids.shape:
                raise ValueError("Unexpected hidden-state shape; full-prompt hooks only.")
            modified = False
            if injection is not None and layer == injection[0]:
                _, position, vector, alpha = injection
                start, end = prompt["sentences"][position - 1]["token_span"]
                before = h[:, start:end, :].clone()
                delta = (alpha * vector).to(device=h.device, dtype=h.dtype)
                h = h.clone()
                h[:, start:end, :] = before + delta
                realized = h[:, start:end, :].float() - before.float()
                perturbation.update(
                    alpha=alpha,
                    intended_per_token_l2=alpha,
                    realized_per_token_l2=realized.norm(dim=-1)[0].cpu().tolist(),
                    realized_frobenius_norm=realized.norm().item(),
                    stochastic_seed=None,
                )
                modified = True
            if restoration is not None and layer == restoration[0]:
                clean = restoration[1].to(device=h.device, dtype=h.dtype)
                if clean.shape != h[0, second_start:second_end, :].shape:
                    raise ValueError("Control activation shape does not match second sentence.")
                if not modified:
                    h = h.clone()
                h[:, second_start:second_end, :] = clean
                modified = True
            if layer in capture_layers:
                captures[layer] = h[0, second_start:second_end, :].detach().cpu().clone()
            if modified:
                return (h,) + output[1:] if isinstance(output, tuple) else h
            return None
        return hook

    try:
        for layer in sorted(active):
            handles.append(layers[layer].register_forward_hook(make_hook(layer)))
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                            use_cache=False, logits_to_keep=1)
            logits = outputs.logits[0, -1].float()
            a = logits[prompt["answer_token_ids"]["A"]].item()
            b = logits[prompt["answer_token_ids"]["B"]].item()
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError("Non-finite answer logits.")
        return {"logit_A": a, "logit_B": b, "L": a - b}, captures, perturbation
    finally:
        for handle in handles:
            handle.remove()


def run_condition(model, prompt, control, clean, vector, concept, layer, alpha, epsilon,
                  on_forward=None):
    injections = []
    first_activations = None
    for position in (1, 2):
        score, captured, perturbation = forward(
            model, prompt, injection=(layer, position, vector, alpha),
            capture_layers=range(layer, len(model.model.layers)) if position == 1 else (),
        )
        if on_forward is not None:
            on_forward(f"injected sentence {position}")
        target = prompt["sentences"][position - 1]
        injections.append({
            **score, **perturbation, "target_position": position,
            "target_label": target["label"], "target_token_indices": target["token_indices"],
            "target_token_ids": target["token_ids"],
            "injection_site": f"model.model.layers.{layer}:output",
        })
        if position == 1:
            first_activations = captured
    metrics = condition_metrics(control["L"], injections, prompt["sentences"][0]["label"])
    diagnostics = []
    for restore_layer, perturbed in first_activations.items():
        original = clean[restore_layer]
        numerator = torch.linalg.vector_norm(perturbed.float() - original.float()).item()
        denominator = torch.linalg.vector_norm(original.float()).item()
        restored, _, _ = forward(
            model, prompt, injection=(layer, 1, vector, alpha),
            restoration=(restore_layer, original),
        )
        if on_forward is not None:
            on_forward(f"restored block {restore_layer}")
        diagnostics.append({
            "layer": restore_layer, "P": numerator / (denominator + epsilon),
            "difference_frobenius_norm": numerator, "control_frobenius_norm": denominator,
            "L_restoration": restored["L"], "E": injections[0]["L"] - restored["L"],
        })
    return {
        "schema_version": SCHEMA_VERSION,
        **{key: prompt[key] for key in ("prompt_id", "pair_id", "content_order", "label_mapping")},
        "concept": concept, "layer": layer, "alpha": alpha,
        "control_L": control["L"], "injections": injections, "diagnostics": diagnostics, **metrics,
    }, first_activations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--tokenizer-model", help="Optional tokenizer source, recorded in manifest")
    parser.add_argument("--concepts", nargs="+", default=["all"])
    parser.add_argument("--layers", nargs="+", type=int, default=LAYERS)
    parser.add_argument("--alphas", "--strengths", nargs="+", type=float, default=ALPHAS)
    parser.add_argument("--num-pairs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vec-type", choices=["avg", "last"], default="avg")
    parser.add_argument("--vector-dir", type=Path, default=REPO / "data" / "saved_vectors" / "llama")
    parser.add_argument("--corpus", type=Path, help="Optional JSON list of sentence strings")
    parser.add_argument("--output-dir", type=Path, default=REPO / "results" / "position_detection")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--epsilon", type=float, default=1e-8, help="Only for the contamination ratio P")
    parser.add_argument("--prepare-only", action="store_true", help="Save pair/prompt manifest without loading weights")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-interval", type=float, default=30,
                        help="Seconds between progress.json and log updates")
    parser.add_argument("--no-progress", action="store_true", help="Disable the interactive tqdm bar")
    args = parser.parse_args()
    args.concepts = CONCEPTS.copy() if args.concepts == ["all"] else args.concepts
    if any(c not in CONCEPTS for c in args.concepts):
        parser.error("Unknown concept; use existing concept names or 'all'.")
    if not args.layers or min(args.layers) < 0 or max(args.layers) >= 32:
        parser.error("Llama-3.1-8B injection layers must be in [0, 31].")
    if any(not math.isfinite(a) or a < 0 for a in args.alphas):
        parser.error("Alphas must be finite and nonnegative.")
    if not math.isfinite(args.epsilon) or args.epsilon <= 0:
        parser.error("Epsilon must be finite and positive.")
    if not math.isfinite(args.log_interval) or args.log_interval <= 0:
        parser.error("Log interval must be finite and positive.")
    for values in (args.layers, args.alphas, args.concepts):
        if len(values) != len(set(values)):
            parser.error("Duplicate layers, alphas or concepts would duplicate conditions.")
    sentences = json.loads(args.corpus.read_text()) if args.corpus else LOCALIZATION_SENTENCES
    if not isinstance(sentences, list) or not all(isinstance(s, str) for s in sentences):
        parser.error("Corpus must be a JSON list of sentence strings.")
    inventory = vector_inventory(args)
    out = args.output_dir
    manifest_path = out / "manifest.json"
    if out.exists() and any(out.iterdir()) and not args.resume:
        parser.error("Output directory is not empty. Use --resume or a new directory.")
    if args.resume and not manifest_path.exists():
        parser.error("--resume requires an existing manifest.json.")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model or args.model, revision=args.revision)
    config = {
        "model": args.model, "revision": args.revision,
        "tokenizer_model": args.tokenizer_model or args.model,
        "tokenizer_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        "chat_template": tokenizer.chat_template,
        "concepts": args.concepts, "layers": args.layers, "alphas": args.alphas,
        "num_pairs": args.num_pairs, "seed": args.seed, "vec_type": args.vec_type,
        "dtype": args.dtype, "device": args.device, "epsilon": args.epsilon,
        "corpus_sha256": digest(sentences), "vectors": inventory,
        "implementation_sha256": digest({
            "runner": file_digest(Path(__file__)),
            "utils": file_digest(REPO / "code" / "utils" / "position_detection_utils.py"),
        }),
        "label_mappings": ["AB", "BA"], "restoration": "every_block_output_from_injection",
        "strength_definition": "absolute per-token L2 alpha; z not implemented",
    }
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        if manifest["schema_version"] != SCHEMA_VERSION or manifest["config"] != config:
            raise ValueError("Resume configuration, corpus, tokenizer or vector hashes changed.")
    else:
        print("Selecting exact-length pairs in all four prompt configurations...", flush=True)
        selection = select_pairs(tokenizer, sentences, args.num_pairs, args.seed)
        counts = forward_counts(args.num_pairs, args.layers, len(args.concepts), len(args.alphas))
        manifest = {"schema_version": SCHEMA_VERSION, "config": config, **selection,
                    "forward_passes": {**counts, "total": sum(counts.values())}}
        atomic_json(manifest_path, manifest)
    print(f"Pairs: {len(manifest['pairs'])} / {manifest['eligible_pair_count']} eligible; "
          f"forward evaluations: {manifest['forward_passes']['total']:,}", flush=True)
    warnings.warn(
        "Preserving legacy vector indexing: saved hidden_states[k] is the embedding output "
        "at k=0, otherwise block k-1 output; injection is at block k output. "
        "This one-block location mismatch is recorded for every vector in manifest.json."
    )
    if args.prepare_only:
        print(f"Prepared {manifest_path}; no model weights loaded.", flush=True)
        return
    with RunProgress(out, manifest, args.log_interval, not args.no_progress) as progress:
        execute(args, manifest, inventory, progress)


def execute(args, manifest, inventory, progress):
    out = args.output_dir
    torch.manual_seed(args.seed)
    model_kwargs = {"revision": args.revision, "torch_dtype": getattr(torch, args.dtype),
                    "attn_implementation": "eager"}
    if args.device == "auto":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.device != "auto":
        model.to(args.device)
    model.eval()
    if len(model.model.layers) != 32:
        raise ValueError("This runner expects the 32-block model used by the original experiment.")
    runtime = {"torch": torch.__version__, "transformers": transformers.__version__,
               "model_commit": getattr(model.config, "_commit_hash", None),
               "model_config_sha256": digest(model.config.to_dict()),
               "attention_implementation": "eager", "dtype": str(model.dtype),
               "device_map": {str(k): str(v) for k, v in getattr(model, "hf_device_map", {}).items()},
               "input_device": str(model.get_input_embeddings().weight.device)}
    runtime_path = out / "runtime.json"
    if runtime_path.exists() and json.loads(runtime_path.read_text()) != runtime:
        raise ValueError("Resume model/runtime changed; use a new output directory.")
    vectors = {(i["concept"], i["layer"]): load_vector(i, args, model.config.hidden_size) for i in inventory}
    atomic_json(runtime_path, runtime)
    progress.start_running()
    expected = len(manifest["prompts"]) * len(args.concepts) * len(args.layers) * len(args.alphas)
    completed = 0
    for prompt in manifest["prompts"]:
        pid = prompt["prompt_id"]
        control_path = out / "controls" / f"{pid}.json"
        clean_path = out / "activations" / pid / "control.pt"
        if control_path.exists() and clean_path.exists():
            control = json.loads(control_path.read_text())
            clean = torch.load(clean_path, map_location="cpu", weights_only=True)["second_sentence"]
        else:
            progress.set_condition(prompt_id=pid, stage="control")
            control, clean, _ = forward(model, prompt, capture_layers=range(min(args.layers), 32))
            control.update(prompt_id=pid, activation_file=str(clean_path.relative_to(out)))
            atomic_tensor(clean_path, {"prompt_id": pid, "second_sentence": clean})
            atomic_json(control_path, control)
            progress.control_saved()
        for concept in args.concepts:
            for layer in args.layers:
                for alpha_index, alpha in enumerate(args.alphas):
                    name = f"{concept}_layer{layer}_alpha{alpha_index}"
                    record_path = out / "conditions" / pid / f"{name}.json"
                    activation_path = out / "activations" / pid / f"{name}.pt"
                    if record_path.exists() and activation_path.exists():
                        completed += 1
                        continue
                    progress.set_condition(prompt_id=pid, concept=concept, layer=layer, alpha=alpha)
                    row, captured = run_condition(
                        model, prompt, control, clean, vectors[concept, layer],
                        concept, layer, alpha, args.epsilon, on_forward=progress.advance,
                    )
                    row.update(control_file=str(control_path.relative_to(out)),
                               activation_file=str(activation_path.relative_to(out)),
                               vector_sha256=next(i["sha256"] for i in inventory
                                                  if i["concept"] == concept and i["layer"] == layer))
                    atomic_tensor(activation_path, {"prompt_id": pid, "injected_position": 1,
                                                   "injection_layer": layer, "second_sentence": captured})
                    atomic_json(record_path, row)
                    completed += 1
                    progress.condition_saved()
    atomic_json(out / "completed.json", {"conditions": completed, "expected": expected})
    print(f"Finished. Analyze with code/analysis/compute_position_detection_accuracy.py --input-dir {out}")


if __name__ == "__main__":
    main()
