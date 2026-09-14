"""Prepare Experiment 4 inputs and validate their Experiment 0 calibration provenance."""

import argparse
import math

from experiment_0_calibration.prepare_material import file_sha256, write_jsonl
from experiment_0_calibration.protocol_config import repo_path

from . import EXPERIMENT_ID
from .protocol_config import load_config, read_json, validate_conditions, write_json
from .step_01_select_examples import build_classification_prompt, select_examples
from .step_02_check_concepts import check_concept_separation


def load_inputs(config):
    dataset = read_json(config.path("complex_data"))
    conditions = validate_conditions(config)
    records = read_json(config.path("directional_scales"))
    scales = {r["direction_id"]: r for r in records}
    if len(scales) != len(records):
        raise ValueError("Duplicate calibration direction IDs")
    calibration = read_json(config.path("calibration_manifest"))
    model = config.raw["model"]
    if calibration["model_name"] != model["name"]:
        raise ValueError("Calibration model does not match the experiment")
    revision = calibration.get("resolved_model_revision") or calibration["requested_model_revision"]
    if model["revision"] not in (revision, calibration["requested_model_revision"]):
        raise ValueError("Calibration model revision mismatch")
    if config.raw["protocol"]["status"] == "frozen" and calibration["protocol_status"] != "frozen":
        raise ValueError("A principal run needs frozen calibration; development results remain development")
    return dataset, conditions, scales, calibration


def build_plan(config, tokenizer):
    dataset, conditions, scales, _calibration = load_inputs(config)
    examples = select_examples(dataset, config.raw["examples"])
    prompts = [build_classification_prompt(tokenizer, e, mapping)
               for e in examples for mapping in ("X", "Y")]
    selected_js = set(config.raw["diagnostics"]["js_example_ids"])
    if not selected_js <= {e["example_id"] for e in examples}:
        raise ValueError("JS subset contains unselected examples")
    trials, source_hashes, noise_owners = [], {}, {}
    noise_banks = {}
    for layer in {c["layer"] for c in conditions}:
        noise_banks[layer] = iter(sorted(r["direction_id"] for r in scales.values()
            if r["decoder_block_index"] == layer and r["direction_family"] == "renewed_noise"))
    fixed_seeds = {int(r["seed"]) for r in scales.values() if r["direction_family"] == "fixed_random"}
    for condition in conditions:
        for prompt in prompts:
            if prompt["probed_concept"] not in condition["probed_concepts"]:
                continue
            if condition["family"] == "renewed_noise":
                # The assignment is frozen BEFORE execution. Independent token draws
                # have their own Experiment 0 scale, never a median approximation.
                if "noise_assignments" in condition:
                    ids = condition["noise_assignments"][prompt["prompt_id"]]
                else:
                    # Existing Experiment 0 directions are independent draws. Consume
                    # their stable ID order without a new, uncalibrated random draw.
                    ids = []
                    while len(ids) < len(prompt["target_token_indices"]):
                        candidate = next(noise_banks[condition["layer"]], None)
                        if candidate is None:
                            raise ValueError("Insufficient calibrated per-token noise directions; extend Experiment 0 material before this run")
                        if int(scales[candidate]["seed"]) not in noise_owners:
                            ids.append(candidate)
                if len(ids) != len(prompt["target_token_indices"]) or len(set(ids)) != len(ids):
                    raise ValueError("Renewed noise needs one distinct calibrated direction per token")
            else:
                ids = [condition["direction_id"]]
            for token_slot, direction_id in enumerate(ids):
                record = scales[direction_id]
                if record["decoder_block_index"] != condition["layer"] or record["direction_family"] != condition["family"]:
                    raise ValueError("Calibration direction does not match the condition")
                scale = float(record[condition["scale_statistic"]])
                if not math.isfinite(scale) or scale <= 0:
                    raise ValueError("Each used direction needs a finite positive calibration scale")
                check_concept_separation(prompt, record, dataset)
                if record["direction_family"] == "renewed_noise":
                    # Section 3.5 specifies a fresh draw for every trial/token.
                    owner = (condition["condition_id"], prompt["prompt_id"], token_slot)
                    seed = int(record["seed"])
                    if seed in fixed_seeds or (seed in noise_owners and noise_owners[seed] != owner):
                        raise ValueError("Noise draws are shared by distinct planned realizations")
                    noise_owners[seed] = owner
                if record.get("source_path"):
                    source = repo_path(record["source_path"])
                    actual_hash = file_sha256(source)
                    if actual_hash != record["source_sha256"]:
                        raise ValueError(f"Calibrated vector changed: {source}")
                    source_hashes[str(source)] = actual_hash
            trials.append({"trial_id": f"{condition['condition_id']}__{prompt['prompt_id']}",
                           "condition_id": condition["condition_id"], "prompt_id": prompt["prompt_id"],
                           "direction_ids": ids, "measure_js": prompt["example_id"] in selected_js})
    expected_sources = {str(config.source_path): file_sha256(config.source_path)}
    for key in ("complex_data", "directional_scales", "calibration_manifest"):
        source = config.path(key)
        expected_sources[str(source)] = file_sha256(source)
    return prompts, conditions, trials, {**expected_sources, **source_hashes}


def prepare(config, tokenizer):
    if config.plan_dir.exists():
        raise FileExistsError(f"Plan already exists: {config.plan_dir}")
    prompts, conditions, trials, sources = build_plan(config, tokenizer)
    config.plan_dir.mkdir(parents=True)
    for name, rows in (("prompts", prompts), ("conditions", conditions), ("trials", trials)):
        write_jsonl(config.plan_dir / f"{name}.jsonl", rows)
    hashes = {name: file_sha256(config.plan_dir / name)
              for name in ("prompts.jsonl", "conditions.jsonl", "trials.jsonl")}
    sham_keys = {(t["prompt_id"], next(c["layer"] for c in conditions if c["condition_id"] == t["condition_id"])) for t in trials}
    manifest = {"experiment_id": EXPERIMENT_ID, "schema_version": 1,
                "protocol": config.raw["protocol"], "config": config.raw,
                "source_hashes": sources, "artifact_hashes": hashes,
                "n_prompts": len(prompts), "n_interventions": len(trials),
                "n_sham": len(sham_keys), "n_js": sum(t["measure_js"] for t in trials),
                "response_rule": "greedy_first_token_full_vocabulary; invalid_counts_as_incorrect"}
    write_json(config.plan_dir / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.raw["model"]["name"],
                                             revision=config.raw["model"]["tokenizer_revision"], use_fast=True)
    result = prepare(config, tokenizer)
    print(f"Prepared {result['n_interventions']} interventions and {result['n_sham']} sham forwards")


if __name__ == "__main__":
    main()
