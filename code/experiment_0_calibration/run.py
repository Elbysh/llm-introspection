"""Run Experiment 0 exactly as enumerated in its prepared plan.

Protocol mapping:
  step 1: natural forward passes without intervention;
  step 2: extract h(layer, token) at explicit admissible positions;
  step 3: project those activations onto every planned direction;
  step 4: compute mean, SD, median, corrected MAD and quantiles;
  step 6: phrase-cluster bootstrap SD and MAD.

Dose construction (protocol step 8) is intentionally outside this module.
"""

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from . import EXPERIMENT_ID
from .config import Experiment0Config, load_config, repo_path
from .plan import materialize_direction, read_jsonl
from .statistics import build_bootstrap_plan, summarize_projection_matrix


def decoder_layer(model: Any, layer: int) -> Any:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or layer < 0 or layer >= len(layers):
        raise ValueError("model does not expose decoder block {}".format(layer))
    return layers[layer]


def disable_optional_triton_native_ops() -> bool:
    """Use standard CUDA kernels on Ruche when optional Triton cannot link."""
    python_native = getattr(torch.backends, "python_native", None)
    triton_backend = getattr(python_native, "triton", None)
    if triton_backend is None:
        return False
    triton_backend.enabled = False
    return True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def validate_prepared_plan(
    config: Experiment0Config,
    config_path: Path,
    plan_manifest: Dict[str, Any],
    contexts: List[Dict[str, Any]],
    observations: List[Dict[str, Any]],
    directions: List[Dict[str, Any]],
) -> None:
    """Fail before model loading if plan and configuration are inconsistent."""
    if plan_manifest.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("prepared plan belongs to another experiment")
    if plan_manifest.get("config_sha256") != sha256(config_path):
        raise ValueError("prepared plan was built from a different configuration")
    if [row["observation_index"] for row in observations] != list(range(len(observations))):
        raise ValueError("observation indices must be contiguous and ordered")
    context_ids = {row["context_id"] for row in contexts}
    if any(row["context_id"] not in context_ids for row in observations):
        raise ValueError("an observation references an unknown context")
    observation_ids = {row["observation_id"] for row in observations}
    direction_ids = [row["direction_id"] for row in directions]
    if len(direction_ids) != len(set(direction_ids)):
        raise ValueError("direction IDs must be unique")
    for layer in config.layers:
        counts = Counter(
            row["direction_family"]
            for row in directions
            if int(row["decoder_block_index"]) == layer
        )
        expected = {
            "concept": len(config.concepts),
            "fixed_random": config.fixed_random_count_per_layer,
            "renewed_noise": len(observations) * config.noise_repetitions_per_position,
        }
        if dict(counts) != expected:
            raise ValueError(
                "direction counts at block {} are {}, expected {}".format(
                    layer, dict(counts), expected
                )
            )
    if any(
        row["direction_family"] == "renewed_noise"
        and row["target_observation_id"] not in observation_ids
        for row in directions
    ):
        raise ValueError("a renewed-noise direction references an unknown observation")


@torch.inference_mode()
def collect_activations(
    model: Any,
    tokenizer: Any,
    contexts: Sequence[Dict[str, Any]],
    observations: Sequence[Dict[str, Any]],
    layers: Sequence[int],
    batch_size: int,
) -> Dict[int, torch.Tensor]:
    """Experiment 0 steps 1-2, preserving global observation order."""
    observations_by_context = defaultdict(list)
    for observation in observations:
        observations_by_context[observation["context_id"]].append(observation)
    for rows in observations_by_context.values():
        rows.sort(key=lambda row: row["observation_index"])

    captured: Dict[int, torch.Tensor] = {}
    chunks = {layer: [] for layer in layers}
    handles = []

    def capture(layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer] = hidden.detach()
        return hook

    for layer in layers:
        handles.append(decoder_layer(model, layer).register_forward_hook(capture(layer)))
    try:
        input_device = next(model.parameters()).device
        tokenizer.padding_side = "right"
        for start in range(0, len(contexts), batch_size):
            batch_contexts = contexts[start : start + batch_size]
            texts = [row["rendered_text"] for row in batch_contexts]
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
            )
            for batch_index, context in enumerate(batch_contexts):
                actual = encoded["input_ids"][batch_index, : len(context["input_ids"])].tolist()
                if actual != context["input_ids"]:
                    raise RuntimeError("tokenization drift for {}".format(context["context_id"]))
            captured.clear()
            model(
                **{key: value.to(input_device) for key, value in encoded.items()},
                use_cache=False
            )
            for layer in layers:
                selected = []
                for batch_index, context in enumerate(batch_contexts):
                    positions = [
                        row["token_index"]
                        for row in observations_by_context[context["context_id"]]
                    ]
                    selected.append(captured[layer][batch_index, positions, :].float().cpu())
                chunks[layer].append(torch.cat(selected, dim=0))
            print(
                "Experiment 0 steps 1-2: {}/{} contexts".format(
                    min(start + len(batch_contexts), len(contexts)), len(contexts)
                ),
                flush=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    result = {layer: torch.cat(chunks[layer], dim=0) for layer in layers}
    for layer, matrix in result.items():
        if matrix.shape[0] != len(observations):
            raise RuntimeError("observation count mismatch at block {}".format(layer))
    return result


def calibrate_layer(
    layer: int,
    activations: torch.Tensor,
    direction_rows: List[Dict[str, Any]],
    observations: List[Dict[str, Any]],
    config: Experiment0Config,
) -> tuple:
    """Experiment 0 steps 3, 4 and 6 for one decoder block."""
    hidden_size = activations.shape[1]
    matrices = []
    original_norms = []
    for start in range(0, len(direction_rows), config.direction_chunk_size):
        rows = direction_rows[start : start + config.direction_chunk_size]
        materialized = [materialize_direction(row, config, hidden_size) for row in rows]
        matrix = torch.stack([item[0] for item in materialized])
        matrices.append((activations @ matrix.T).numpy().astype(np.float32, copy=False))
        original_norms.extend(item[1] for item in materialized)
    projections = np.concatenate(matrices, axis=1)
    bootstrap = build_bootstrap_plan(
        [row["sentence_id"] for row in observations],
        config.bootstrap_resamples_sd,
        config.bootstrap_resamples_mad,
        config.bootstrap_seed,
    )
    statistics = summarize_projection_matrix(projections, bootstrap, config.ci_level)
    records = []
    for column, (direction, original_norm) in enumerate(zip(direction_rows, original_norms)):
        record = dict(direction)
        record.update(
            {
                "experiment_id": EXPERIMENT_ID,
                "activation_site": config.activation_site,
                "presentation_context": config.context_id,
                "position_policy": config.position_policy,
                "point_weighting": config.point_weighting,
                "bootstrap_unit": config.bootstrap_unit,
                "bootstrap_resamples_sd": config.bootstrap_resamples_sd,
                "bootstrap_resamples_mad": config.bootstrap_resamples_mad,
                "ci_level": config.ci_level,
                "n_sentences": len({row["sentence_id"] for row in observations}),
                "n_positions": len(observations),
                "original_direction_norm": original_norm,
                "valid_for_sd_normalization": bool(statistics["sd"][column] > 0.0),
            }
        )
        for name, values in statistics.items():
            record[name] = float(values[column])
        records.append(record)
    print(
        "Experiment 0 steps 3-6: block {} ({} directions)".format(
            layer, len(direction_rows)
        ),
        flush=True,
    )
    return records, projections


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 0 calibration")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = repo_path(args.config)
    config = load_config(args.config)
    if config.output_dir.exists():
        raise FileExistsError(
            "refusing to overwrite Experiment 0 output: {}".format(config.output_dir)
        )

    contexts_path = config.plan_dir / "contexts.jsonl"
    observations_path = config.plan_dir / "observations.jsonl"
    directions_path = config.plan_dir / "directions.jsonl"
    plan_manifest_path = config.plan_dir / "manifest.json"
    contexts = read_jsonl(contexts_path)
    observations = read_jsonl(observations_path)
    directions = read_jsonl(directions_path)
    with plan_manifest_path.open("r", encoding="utf-8") as handle:
        plan_manifest = json.load(handle)
    validate_prepared_plan(
        config,
        config_path,
        plan_manifest,
        contexts,
        observations,
        directions,
    )
    directions_by_layer = defaultdict(list)
    for direction in directions:
        directions_by_layer[int(direction["decoder_block_index"])].append(direction)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if disable_optional_triton_native_ops():
        print("Disabled optional PyTorch Triton-native kernels", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, revision=config.tokenizer_revision, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    activations = collect_activations(
        model, tokenizer, contexts, observations, config.layers, config.batch_size
    )
    all_records = []
    projection_archive = {}
    for layer in config.layers:
        records, projections = calibrate_layer(
            layer,
            activations.pop(layer),
            directions_by_layer[layer],
            observations,
            config,
        )
        all_records.extend(records)
        for column, record in enumerate(records):
            projection_archive[record["direction_id"]] = projections[:, column]

    config.output_dir.mkdir(parents=True, exist_ok=False)
    statistics_path = config.output_dir / "directional_scales.json"
    with statistics_path.open("w", encoding="utf-8") as handle:
        json.dump(all_records, handle, indent=2, ensure_ascii=False, sort_keys=True)
    projections_path = config.output_dir / "projections.npz"
    np.savez_compressed(str(projections_path), **projection_archive)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": config.protocol_version,
        "protocol_status": config.protocol_status,
        "git_commit": current_commit(),
        "config_sha256": sha256(config_path),
        "plan_manifest_sha256": sha256(plan_manifest_path),
        "contexts_sha256": sha256(contexts_path),
        "observations_sha256": sha256(observations_path),
        "directions_sha256": sha256(directions_path),
        "model_name": config.model_name,
        "model_revision": config.model_revision,
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "tokenizer_revision": config.tokenizer_revision,
        "resolved_tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "torch_version": torch.__version__,
        "model_dtype": "bfloat16",
        "projection_dtype": "float32",
        "n_contexts": len(contexts),
        "n_observations": len(observations),
        "n_directions": len(directions),
        "n_records": len(all_records),
        "statistics_file": statistics_path.name,
        "projections_file": projections_path.name,
        "dose_conversion_included": False,
    }
    with (config.output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)

    from .plots import create_all_plots

    create_all_plots(statistics_path, projections_path, config.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
