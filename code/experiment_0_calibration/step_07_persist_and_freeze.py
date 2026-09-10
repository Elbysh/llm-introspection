"""Experiment 0 step 7: validate provenance and persist immutable scales."""

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from . import EXPERIMENT_ID
from .protocol_config import Experiment0Config


PROTECTED_OUTPUT_NAMES = (
    "directional_scales.json",
    "directional_scales.csv",
    "projections.npz",
    "run_manifest.json",
    "figures",
)


def ensure_output_artifacts_absent(output_dir: Path) -> None:
    """Allow a documented result root, but never overwrite run artifacts."""
    existing_outputs = [
        output_dir / name
        for name in PROTECTED_OUTPUT_NAMES
        if (output_dir / name).exists()
    ]
    if existing_outputs:
        raise FileExistsError(
            "refusing to overwrite Experiment 0 artifacts: {}".format(
                ", ".join(str(path) for path in existing_outputs)
            )
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
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
    """Refuse stale, incomplete or internally inconsistent material plans."""
    if plan_manifest.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("prepared plan belongs to another experiment")
    if plan_manifest.get("config_sha256") != file_sha256(config_path):
        raise ValueError("prepared plan was built from a different configuration")
    if [row["observation_index"] for row in observations] != list(
        range(len(observations))
    ):
        raise ValueError("observation indices must be contiguous and ordered")
    context_ids = {row["context_id"] for row in contexts}
    if any(row["context_id"] not in context_ids for row in observations):
        raise ValueError("an observation references an unknown context")
    observation_ids = {row["observation_id"] for row in observations}
    direction_ids = [row["direction_id"] for row in directions]
    if len(direction_ids) != len(set(direction_ids)):
        raise ValueError("direction IDs must be unique")
    stochastic_seeds = [
        int(row["seed"])
        for row in directions
        if row["direction_family"] != "concept"
    ]
    if len(stochastic_seeds) != len(set(stochastic_seeds)):
        raise ValueError(
            "fixed-random and renewed-noise seeds must be globally unique"
        )
    for decoder_block_index in config.layers:
        counts = Counter(
            row["direction_family"]
            for row in directions
            if int(row["decoder_block_index"]) == decoder_block_index
        )
        expected = {
            "concept": len(config.concepts),
            "fixed_random": config.fixed_random_count_per_layer,
            "renewed_noise": len(observations)
            * config.noise_repetitions_per_position,
        }
        if dict(counts) != expected:
            raise ValueError(
                "direction counts at block {} are {}, expected {}".format(
                    decoder_block_index, dict(counts), expected
                )
            )
    if any(
        row["direction_family"] == "renewed_noise"
        and row["target_observation_id"] not in observation_ids
        for row in directions
    ):
        raise ValueError("a renewed-noise direction references an unknown observation")


def persist_calibration(
    config: Experiment0Config,
    config_path: Path,
    plan_manifest_path: Path,
    contexts_path: Path,
    observations_path: Path,
    directions_path: Path,
    records: List[Dict[str, Any]],
    projection_archive: Dict[str, np.ndarray],
    model: Any,
    tokenizer: Any,
    n_contexts: int,
    n_observations: int,
    n_directions: int,
) -> Dict[str, Any]:
    """Write a development result or freeze an approved calibration once.

    The output-directory guard is the concrete no-overwrite rule. A frozen
    configuration has additional validation in `protocol_config.py`.
    """
    ensure_output_artifacts_absent(config.output_dir)
    # The result root may already contain its tracked README. Scientific
    # artifacts themselves remain immutable and are never overwritten.
    config.output_dir.mkdir(parents=True, exist_ok=True)
    statistics_path = config.output_dir / "directional_scales.json"
    with statistics_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False, sort_keys=True)
    projections_path = config.output_dir / "projections.npz"
    np.savez_compressed(str(projections_path), **projection_archive)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": config.protocol_version,
        "protocol_status": config.protocol_status,
        "git_commit": current_git_commit(),
        "config_sha256": file_sha256(config_path),
        "plan_manifest_sha256": file_sha256(plan_manifest_path),
        "contexts_sha256": file_sha256(contexts_path),
        "observations_sha256": file_sha256(observations_path),
        "directions_sha256": file_sha256(directions_path),
        "model_name": config.model_name,
        "requested_model_revision": config.model_revision,
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "requested_tokenizer_revision": config.tokenizer_revision,
        "resolved_tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "torch_version": torch.__version__,
        "model_dtype": "bfloat16",
        "projection_dtype": "float32",
        "n_contexts": n_contexts,
        "n_observations": n_observations,
        "n_directions": n_directions,
        "n_records": len(records),
        "statistics_file": statistics_path.name,
        "projections_file": projections_path.name,
        "dose_conversion_included": False,
    }
    with (config.output_dir / "run_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
    return manifest

