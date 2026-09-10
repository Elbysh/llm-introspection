"""Compute natural projection scales s(layer, direction)."""

import argparse
import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .common import disable_optional_triton_native_ops, load_repository_corpus
from .config import load_config
from .directions import load_direction_bank


def decoder_layer(model: Any, layer: int) -> Any:
    """Return a Llama decoder block using the protocol's layer convention."""
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError("expected a model exposing model.layers")
    if layer < 0 or layer >= len(layers):
        raise IndexError("layer {} is outside model.layers".format(layer))
    return layers[layer]


def robust_mad(values: np.ndarray) -> float:
    """Gaussian-consistent median absolute deviation."""
    center = np.median(values)
    return float(1.4826 * np.median(np.abs(values - center)))


@torch.inference_mode()
def compute_scales(
    model: Any,
    tokenizer: Any,
    sentences: List[str],
    directions: Dict[str, torch.Tensor],
    metadata: List[Dict[str, Any]],
    layers: List[int],
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray], List[Dict[str, Any]]]:
    """Collect q=v^T h for every valid token and summarize each direction.

    Only scalar projections are retained. Full hidden-state matrices would be
    much heavier and are unnecessary for SD, MAD, quantiles or diagnostic plots.

    Per-layer activation norms are collected alongside the projections. They are
    the scale a direction-free perturbation such as dropout is matched against,
    since its amplitude is set by a rate rather than by a chosen direction.
    """
    metadata_by_id = {row["direction_id"]: row for row in metadata}
    directions_by_layer = defaultdict(list)
    for direction_id, vector in directions.items():
        layer = int(metadata_by_id[direction_id]["layer"])
        directions_by_layer[layer].append((direction_id, vector.float()))

    matrices = {}
    for layer in layers:
        entries = directions_by_layer[layer]
        if not entries:
            raise ValueError("no directions configured at layer {}".format(layer))
        matrices[layer] = (
            [entry[0] for entry in entries],
            torch.stack([entry[1] for entry in entries], dim=0),
        )

    captured = {}
    handles = []

    def capture_hook(layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer] = hidden.detach()
        return hook

    for layer in layers:
        handles.append(decoder_layer(model, layer).register_forward_hook(capture_hook(layer)))

    projection_chunks = defaultdict(list)
    norm_chunks = defaultdict(list)
    try:
        input_device = next(model.parameters()).device
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start : start + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
            )
            encoded = {key: value.to(input_device) for key, value in encoded.items()}
            valid_tokens = encoded["attention_mask"].bool()
            captured.clear()
            model(**encoded, use_cache=False)

            for layer, (direction_ids, matrix_cpu) in matrices.items():
                hidden = captured[layer].float()
                token_mask = valid_tokens.to(hidden.device)
                norms = torch.linalg.vector_norm(hidden, dim=-1)[token_mask]
                norm_chunks[layer].append(norms.cpu().numpy())
                direction_matrix = matrix_cpu.to(hidden.device)
                # Shape: [batch, tokens, directions]. Padding is removed below.
                projected = hidden @ direction_matrix.T
                projected = projected[valid_tokens.to(projected.device)].cpu().numpy()
                for column, direction_id in enumerate(direction_ids):
                    projection_chunks[direction_id].append(projected[:, column])

            completed = min(start + len(batch), len(sentences))
            if completed % 100 == 0 or completed == len(sentences):
                print("Processed {}/{} sentences".format(completed, len(sentences)), flush=True)
    finally:
        for handle in handles:
            handle.remove()

    records = []
    saved_projections = {}
    for direction_id, chunks in projection_chunks.items():
        stored = np.concatenate(chunks).astype(np.float32, copy=False)
        saved_projections[direction_id] = stored
        values = stored.astype(np.float64, copy=False)
        source = metadata_by_id[direction_id]
        records.append(
            {
                "direction_id": direction_id,
                "kind": source["kind"],
                "concept": source.get("concept"),
                "layer": int(source["layer"]),
                "corpus_source": "repository:LOCALIZATION_SENTENCES",
                "n_sentences": len(sentences),
                "n_token_projections": int(values.size),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "mad": robust_mad(values),
                "p01": float(np.quantile(values, 0.01)),
                "p05": float(np.quantile(values, 0.05)),
                "p50": float(np.quantile(values, 0.50)),
                "p95": float(np.quantile(values, 0.95)),
                "p99": float(np.quantile(values, 0.99)),
            }
        )

    layer_records = []
    for layer in layers:
        norms = np.concatenate(norm_chunks[layer]).astype(np.float64, copy=False)
        layer_records.append(
            {
                "layer": int(layer),
                "corpus_source": "repository:LOCALIZATION_SENTENCES",
                "n_sentences": len(sentences),
                "n_token_activations": int(norms.size),
                # Root mean square of the token norms: the h-bar(layer) used to
                # convert a target amplitude into a dropout rate.
                "rms_norm": float(np.sqrt(np.mean(norms ** 2))),
                "mean_norm": float(norms.mean()),
                "median_norm": float(np.median(norms)),
                "p05_norm": float(np.quantile(norms, 0.05)),
                "p95_norm": float(np.quantile(norms, 0.95)),
            }
        )
    return records, saved_projections, layer_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute s(layer, direction)")
    parser.add_argument("--config", default="configs/calibration/pilot.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    sentences = load_repository_corpus()
    if len(sentences) < config.minimum_sentences:
        raise ValueError(
            "repository corpus contains {} sentences; {} required".format(
                len(sentences), config.minimum_sentences
            )
        )
    directions, metadata = load_direction_bank(config.directions_path)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if disable_optional_triton_native_ops():
        print("Disabled optional PyTorch Triton-native kernels", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    records, projections, layer_records = compute_scales(
        model,
        tokenizer,
        sentences,
        directions,
        metadata,
        config.layers,
        config.batch_size,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    statistics_path = config.output_dir / "calibration.json"
    projections_path = config.output_dir / "calibration_projections.npz"
    layer_norms_path = config.output_dir / "calibration_layer_norms.json"
    with statistics_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False, sort_keys=True)
    with layer_norms_path.open("w", encoding="utf-8") as handle:
        json.dump(layer_records, handle, indent=2, ensure_ascii=False, sort_keys=True)
    np.savez_compressed(str(projections_path), **projections)
    print("Saved statistics to {}".format(statistics_path))
    print("Saved layer activation norms to {}".format(layer_norms_path))
    print("Saved projections to {}".format(projections_path))

    from .plot_scales import create_plots

    create_plots(statistics_path, projections_path, config.output_dir)


if __name__ == "__main__":
    main()
