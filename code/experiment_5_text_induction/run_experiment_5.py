"""Run the prepared textual-induction experiment; existing result directories are immutable."""

import argparse
import platform
from pathlib import Path

import torch
from experiment_0_calibration.prepare_material import file_sha256, read_jsonl
from experiment_0_calibration.step_07_persist_and_freeze import current_git_commit

from . import EXPERIMENT_ID
from .prepare_material_plan import load_calibration
from .protocol_config import load_config, read_json, write_json
from .step_06_run_conditions import execute_conditions


def validate_plan(config):
    plan_dir = config.path("plan_dir")
    manifest = read_json(plan_dir / "manifest.json")
    if manifest["experiment_id"] != EXPERIMENT_ID or manifest["config"] != config.raw:
        raise ValueError("Plan/configuration mismatch")
    for name, expected in manifest["artifact_hashes"].items():
        if file_sha256(plan_dir / name) != expected:
            raise ValueError(f"Prepared artifact changed: {name}")
    for path, expected in manifest["source_hashes"].items():
        if file_sha256(Path(path)) != expected:
            raise ValueError(f"Source changed: {path}")
    return manifest


def run(config, model, tokenizer):
    manifest = validate_plan(config)
    output = config.path("output_dir")
    if output.exists():
        raise FileExistsError(f"Results already exist: {output}")
    scales, calibration = load_calibration(config)
    revision = getattr(model.config, "_commit_hash", None)
    if revision and calibration.get("resolved_model_revision") and revision != calibration["resolved_model_revision"]:
        raise ValueError("Loaded model differs from calibration model")
    if config.raw["protocol"]["status"] == "frozen" and revision != config.raw["model"]["revision"]:
        raise ValueError("Cannot verify frozen model revision")
    prepared = {name: read_jsonl(config.path("plan_dir") / f"{name}.jsonl")
                for name in ("pairs", "prompts", "conditions", "trials")}
    for prompt in prepared["prompts"]:
        if tokenizer(prompt["rendered_text"], add_special_tokens=False)["input_ids"] != prompt["input_ids"]:
            raise ValueError("Tokenizer changed since preparation")
    output.mkdir(parents=True)
    write_json(output / "run_manifest.json", {"experiment_id": EXPERIMENT_ID,
        "plan_manifest": manifest, "git_commit": current_git_commit(),
        "torch_version": str(torch.__version__), "python_version": platform.python_version(),
        "resolved_model_revision": revision,
        "resolved_tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "dtype": str(next(model.parameters()).dtype)})
    for name in ("pairs", "prompts", "conditions"):
        write_json(output / f"{name}.json", prepared[name])
    model.eval()
    try:
        with (output / "trials.jsonl").open("x", encoding="utf-8") as stream:
            count = execute_conditions(model, tokenizer, prepared["prompts"], prepared["conditions"], prepared["trials"], scales, stream)
    except Exception as error:
        write_json(output / "failure.json", {"error_type": type(error).__name__, "message": str(error)})
        raise
    write_json(output / "complete.json", {"n_trials": count, "trials_sha256": file_sha256(output / "trials.jsonl")})
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    config = load_config(parser.parse_args().config)
    validate_plan(config)
    if config.path("output_dir").exists():
        raise FileExistsError("Use a new result directory")
    from experiment_0_calibration.run_experiment_0 import (
        disable_optional_triton_native_ops,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer
    settings = config.raw["model"]
    disable_optional_triton_native_ops()
    tokenizer = AutoTokenizer.from_pretrained(settings["name"], revision=settings["tokenizer_revision"], use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(settings["name"], revision=settings["revision"],
        torch_dtype=getattr(torch, config.raw["execution"]["dtype"]), device_map="auto")
    run(config, model, tokenizer)


if __name__ == "__main__":
    main()
