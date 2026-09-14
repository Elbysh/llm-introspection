"""Orchestrate Experiment 4 without mixing preparation, inference and analysis."""

import argparse
import platform
from pathlib import Path

import torch
from experiment_0_calibration.prepare_material import file_sha256, read_jsonl
from experiment_0_calibration.run_experiment_0 import disable_optional_triton_native_ops
from experiment_0_calibration.step_07_persist_and_freeze import current_git_commit

from . import EXPERIMENT_ID
from .prepare_material_plan import load_inputs
from .protocol_config import load_config, read_json, write_json
from .step_06_run_conditions import execute_conditions


def validate_plan(config):
    manifest = read_json(config.plan_dir / "manifest.json")
    if manifest["experiment_id"] != EXPERIMENT_ID or manifest["config"] != config.raw:
        raise ValueError("Prepared plan and execution configuration differ")
    for name, expected in manifest["artifact_hashes"].items():
        if file_sha256(config.plan_dir / name) != expected:
            raise ValueError(f"Prepared artifact changed: {name}")
    for path, expected in manifest["source_hashes"].items():
        if file_sha256(Path(path)) != expected:
            raise ValueError(f"Prepared input changed: {path}")
    return manifest


def run(config, model, tokenizer):
    manifest = validate_plan(config)
    if config.output_dir.exists():
        raise FileExistsError(f"Results already exist: {config.output_dir}")
    _, _, scales, calibration = load_inputs(config)
    calibrated_revision = calibration.get("resolved_model_revision")
    actual_revision = getattr(model.config, "_commit_hash", None)
    if calibrated_revision and actual_revision and calibrated_revision != actual_revision:
        raise ValueError("Loaded model differs from the model actually used for calibration")
    if (
        config.raw["protocol"]["status"] == "frozen"
        and actual_revision != config.raw["model"]["revision"]
    ):
        raise ValueError("Cannot verify the frozen model revision")
    prompts = read_jsonl(config.plan_dir / "prompts.jsonl")
    conditions = read_jsonl(config.plan_dir / "conditions.jsonl")
    trials = read_jsonl(config.plan_dir / "trials.jsonl")
    for prompt in prompts:
        ids = tokenizer(prompt["rendered_text"], add_special_tokens=False)["input_ids"]
        if ids != prompt["input_ids"]:
            raise ValueError("Tokenizer changed since preparation")
    config.output_dir.mkdir(parents=True)
    run_metadata = {"experiment_id": EXPERIMENT_ID, "plan_manifest": manifest,
                    "git_commit": current_git_commit(), "torch_version": str(torch.__version__),
                    "python_version": platform.python_version(),
                    "resolved_model_revision": getattr(model.config, "_commit_hash", None),
                    "resolved_tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
                    "dtype": str(next(model.parameters()).dtype)}
    write_json(config.output_dir / "run_manifest.json", run_metadata)
    # Save exact prompts and intervention definitions next to their results.
    write_json(config.output_dir / "prompts.json", prompts)
    write_json(config.output_dir / "conditions.json", conditions)
    model.eval()
    try:
        with (config.output_dir / "trials.jsonl").open("x", encoding="utf-8") as stream:
            count = execute_conditions(model, tokenizer, prompts, conditions, trials, scales, stream)
    except Exception as error:
        write_json(config.output_dir / "failure.json", {"error_type": type(error).__name__, "message": str(error)})
        raise
    write_json(config.output_dir / "complete.json", {"n_rows": count,
               "trials_sha256": file_sha256(config.output_dir / "trials.jsonl")})
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    validate_plan(config)
    if config.output_dir.exists():
        raise FileExistsError(f"Results already exist: {config.output_dir}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    settings = config.raw["model"]
    tokenizer = AutoTokenizer.from_pretrained(settings["name"], revision=settings["tokenizer_revision"], use_fast=True)
    disable_optional_triton_native_ops()
    model = AutoModelForCausalLM.from_pretrained(
        settings["name"], revision=settings["revision"],
        torch_dtype=getattr(torch, config.raw["execution"]["dtype"]), device_map="auto")
    run(config, model, tokenizer)


if __name__ == "__main__":
    main()
