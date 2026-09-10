"""Orchestrate Experiment 0 without hiding its protocol steps.

Each responsibility lives in the correspondingly numbered module. This file
only wires material, model and artifacts together. Dose construction
(`alpha`/`z`) is intentionally absent.
"""

import argparse
import json
from collections import defaultdict

import torch

from . import EXPERIMENT_ID
from .prepare_material import read_jsonl
from .protocol_config import load_config, repo_path
from .step_01_02_collect_natural_activations import collect_natural_activations
from .step_03_project_activations import project_activations
from .step_04_estimate_scales import estimate_point_scales
from .step_05_plot_distributions import create_all_plots
from .step_06_bootstrap_stability import (
    build_phrase_bootstrap_plan,
    estimate_bootstrap_stability,
)
from .step_07_persist_and_freeze import persist_calibration, validate_prepared_plan


def disable_optional_triton_native_ops() -> bool:
    """Ruche compatibility: use standard CUDA kernels if Triton cannot link."""
    python_native = getattr(torch.backends, "python_native", None)
    triton_backend = getattr(python_native, "triton", None)
    if triton_backend is None:
        return False
    triton_backend.enabled = False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 0 calibration")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = repo_path(args.config)
    config = load_config(args.config)

    # Material prepared before protocol step 1.
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
    if config.output_dir.exists():
        raise FileExistsError(
            "refusing to overwrite Experiment 0 output: {}".format(config.output_dir)
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if disable_optional_triton_native_ops():
        print("Disabled optional PyTorch Triton-native kernels", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        revision=config.tokenizer_revision,
        use_fast=True,
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

    # Steps 1-2: natural forward passes and exact admissible activations.
    activations_by_block = collect_natural_activations(
        model,
        tokenizer,
        contexts,
        observations,
        config.layers,
        config.batch_size,
    )
    directions_by_block = defaultdict(list)
    for direction in directions:
        directions_by_block[int(direction["decoder_block_index"])].append(direction)
    bootstrap_plan = build_phrase_bootstrap_plan(
        [observation["sentence_id"] for observation in observations],
        config.bootstrap_resamples_sd,
        config.bootstrap_resamples_mad,
        config.bootstrap_seed,
    )

    records = []
    projection_archive = {}
    sentence_count = len(
        {observation["sentence_id"] for observation in observations}
    )
    for decoder_block_index in config.layers:
        block_directions = directions_by_block[decoder_block_index]

        # Step 3: p(context, token, block, direction) = <h, v>.
        projections, original_norms = project_activations(
            activations_by_block.pop(decoder_block_index),
            block_directions,
            config,
        )

        # Step 4: point estimates. Step 6: phrase-bootstrap stability.
        estimates = estimate_point_scales(projections)
        estimates.update(
            estimate_bootstrap_stability(
                projections, bootstrap_plan, config.ci_level
            )
        )
        for column, (direction, original_norm) in enumerate(
            zip(block_directions, original_norms)
        ):
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
                    "n_sentences": sentence_count,
                    "n_positions": len(observations),
                    "original_direction_norm": original_norm,
                    "valid_for_sd_normalization": bool(
                        estimates["sd"][column] > 0.0
                    ),
                }
            )
            for name, values in estimates.items():
                record[name] = float(values[column])
            records.append(record)
            projection_archive[direction["direction_id"]] = projections[:, column]
        print(
            "Steps 3, 4 and 6 — decoder block {}: {} directions".format(
                decoder_block_index, len(block_directions)
            ),
            flush=True,
        )

    # Step 7: write once, with provenance sufficient for protocol freezing.
    run_manifest = persist_calibration(
        config,
        config_path,
        plan_manifest_path,
        contexts_path,
        observations_path,
        directions_path,
        records,
        projection_archive,
        model,
        tokenizer,
        len(contexts),
        len(observations),
        len(directions),
    )

    # Step 5 needs persisted projections; numbering follows responsibility,
    # although rendering occurs after the immutable data files are written.
    create_all_plots(
        config.output_dir / "directional_scales.json",
        config.output_dir / "projections.npz",
        config.output_dir,
    )
    print(json.dumps(run_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
