"""Freeze text variants, counterbalancing and concept coefficients before inference."""

import argparse

from experiment_0_calibration.prepare_material import (
    file_sha256,
    load_protocol_corpus,
    write_jsonl,
)
from experiment_0_calibration.protocol_config import repo_path

from . import EXPERIMENT_ID
from .protocol_config import load_config, read_json, write_json
from .step_01_select_text_pairs import select_text_pairs
from .step_03_fix_intervention import fix_intervention
from .step_04_counterbalance_prompts import counterbalanced_prompts


def load_calibration(config):
    records = read_json(config.path("directional_scales"))
    scales = {r["direction_id"]: r for r in records}
    if len(scales) != len(records):
        raise ValueError("Duplicate calibration direction IDs")
    manifest = read_json(config.path("calibration_manifest"))
    model = config.raw["model"]
    if manifest["model_name"] != model["name"]:
        raise ValueError("Calibration and experiment models differ")
    if model["revision"] not in (manifest["requested_model_revision"], manifest.get("resolved_model_revision")):
        raise ValueError("Calibration and experiment revisions differ")
    if config.raw["protocol"]["status"] == "frozen" and manifest["protocol_status"] != "frozen":
        raise ValueError("A principal experiment needs frozen calibration")
    return scales, manifest


def prepare(config, tokenizer):
    plan_dir = config.path("plan_dir")
    if plan_dir.exists():
        raise FileExistsError(f"Plan already exists: {plan_dir}")
    corpus = load_protocol_corpus()
    pairs = select_text_pairs(read_json(config.path("text_pairs")), corpus)
    scales, _ = load_calibration(config)
    conditions = [fix_intervention(c, scales) for c in config.raw["conditions"]]
    if {p["concept"] for p in pairs} != {c["concept"] for c in conditions}:
        raise ValueError("Every selected concept needs both text pairs and intervention conditions")
    prompts = counterbalanced_prompts(tokenizer, pairs, config.raw["texts"]["max_token_length_difference"])
    trials = []
    for condition in conditions:
        for prompt in prompts:
            if prompt["concept"] != condition["concept"]:
                continue
            for injected in (False, True):
                trials.append({"trial_id": f"{condition['condition_id']}__{prompt['prompt_id']}__{'injected' if injected else 'sham'}",
                    "prompt_id": prompt["prompt_id"], "condition_id": condition["condition_id"], "injected": injected})
    # Exactly 2 text versions × 2 intervention states × 2 positions × 2 mappings.
    n_cells = sum(p["concept"] == c["concept"] for p in pairs for c in conditions)
    if len(trials) != 16*n_cells:
        raise ValueError("Incomplete factorial counterbalancing")
    sources = {str(config.source_path): file_sha256(config.source_path)}
    for path in [config.path(key) for key in ("text_pairs", "directional_scales", "calibration_manifest")]:
        sources[str(path)] = file_sha256(path)
    corpus_source = repo_path("code/utils/all_prompts.py")
    sources[str(corpus_source)] = file_sha256(corpus_source)
    for c in conditions:
        source = repo_path(scales[c["direction_id"]]["source_path"])
        sources[str(source)] = file_sha256(source)
    plan_dir.mkdir(parents=True)
    for name, rows in (("pairs", pairs), ("prompts", prompts), ("conditions", conditions), ("trials", trials)):
        write_jsonl(plan_dir / f"{name}.jsonl", rows)
    artifacts = {name: file_sha256(plan_dir / name) for name in
                 ("pairs.jsonl", "prompts.jsonl", "conditions.jsonl", "trials.jsonl")}
    manifest = {"experiment_id": EXPERIMENT_ID, "schema_version": 1, "config": config.raw,
                "source_hashes": sources, "artifact_hashes": artifacts,
                "n_pairs": len(pairs), "n_concepts": len({p["concept"] for p in pairs}),
                "n_trials": len(trials), "n_trials_per_text_intervention_cell": len(trials)//4,
                "n_sham": len(trials)//2, "n_injected": len(trials)//2}
    write_json(plan_dir / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    config = load_config(parser.parse_args().config)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.raw["model"]["name"],
        revision=config.raw["model"]["tokenizer_revision"], use_fast=True)
    manifest = prepare(config, tokenizer)
    print(f"Prepared {manifest['n_trials']} trials, exactly half sham and half injected")


if __name__ == "__main__":
    main()
