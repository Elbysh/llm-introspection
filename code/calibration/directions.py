"""Prepare unit concept, random-control and Gaussian-noise directions."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from safetensors.torch import load_file, save_file

from .config import CalibrationConfig, load_config


def unit(vector: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Return a float32 CPU unit vector and its original L2 norm."""
    vector = vector.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    norm = float(torch.linalg.vector_norm(vector).item())
    if not torch.isfinite(vector).all():
        raise ValueError("direction must be finite")
    if not norm > 0.0:
        raise ValueError("direction must be a non-zero vector")
    return vector / norm, norm


def load_concept_direction(
    vector_dir: Path, concept: str, hidden_state_index: int, vector_type: str
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Load one existing vector artifact without changing its on-disk format."""
    path = vector_dir / "{}_{}_{}.pt".format(concept, hidden_state_index, vector_type)
    if not path.exists():
        raise FileNotFoundError("missing concept vector: {}".format(path))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    vector, original_norm = unit(payload["vector"])
    return vector, {
        "source_path": str(path),
        "source_model": payload.get("model_name"),
        "original_norm": original_norm,
    }


def build_direction_bank(
    config: CalibrationConfig,
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
    """Build all directions whose natural projection scales will be measured."""
    tensors = {}
    records = []
    hidden_size = None

    for layer in config.layers:
        # Existing artifacts use outputs.hidden_states[i]. At decoder-block
        # output l, the matching Llama hidden-state index is l + 1.
        hidden_state_index = layer + config.vector_hidden_state_offset
        for concept in config.concepts:
            vector, source = load_concept_direction(
                config.vector_dir, concept, hidden_state_index, config.vector_type
            )
            hidden_size = vector.numel() if hidden_size is None else hidden_size
            if vector.numel() != hidden_size:
                raise ValueError("inconsistent hidden sizes in concept vectors")
            direction_id = "concept__l{}__{}".format(layer, concept)
            tensors[direction_id] = vector
            records.append(
                {
                    "direction_id": direction_id,
                    "kind": "concept",
                    "concept": concept,
                    "layer": layer,
                    "seed": None,
                    "source_hidden_state_index": hidden_state_index,
                    **source,
                }
            )

    if hidden_size is None:
        raise ValueError("no concept direction was loaded")

    # Both controls are isotropic Gaussian directions. They are stored as
    # separate named banks because they serve different later experimental
    # roles, but calibration treats every normalized direction identically.
    specifications = (
        ("random", config.random_directions_per_layer, 100_000),
        ("noise", config.noise_directions_per_layer, 200_000),
    )
    for layer in config.layers:
        for kind, count, seed_offset in specifications:
            for index in range(count):
                seed = config.seed + seed_offset + layer * 10_000 + index
                generator = torch.Generator(device="cpu").manual_seed(seed)
                vector, gaussian_norm = unit(torch.randn(hidden_size, generator=generator))
                direction_id = "{}__l{}__{:04d}".format(kind, layer, index)
                tensors[direction_id] = vector
                records.append(
                    {
                        "direction_id": direction_id,
                        "kind": kind,
                        "concept": None,
                        "layer": layer,
                        "seed": seed,
                        "sampled_gaussian_norm": gaussian_norm,
                    }
                )
    return tensors, records


def save_direction_bank(
    path: Path, tensors: Dict[str, torch.Tensor], records: List[Dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))
    with path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False, sort_keys=True)


def load_direction_bank(
    path: Path,
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
    tensors = load_file(str(path), device="cpu")
    with path.with_suffix(".json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return tensors, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare directions for calibration")
    parser.add_argument("--config", default="configs/calibration/pilot.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    tensors, metadata = build_direction_bank(config)
    save_direction_bank(config.directions_path, tensors, metadata)
    print("Saved {} directions to {}".format(len(metadata), config.directions_path))


if __name__ == "__main__":
    main()
