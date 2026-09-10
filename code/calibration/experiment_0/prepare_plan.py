"""Prepare the explicit contexts, positions and directions for Experiment 0."""

import argparse
import hashlib
import json

import torch
from transformers import AutoTokenizer

from . import EXPERIMENT_ID
from .config import load_config, repo_path
from .plan import (
    build_context_and_observation_rows,
    build_direction_rows,
    load_protocol_corpus,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Experiment 0 calibration plan")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = repo_path(args.config)
    config = load_config(args.config)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        revision=config.tokenizer_revision,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise RuntimeError("Experiment 0 requires a fast tokenizer for exact offsets")
    sentences = load_protocol_corpus()
    contexts, observations = build_context_and_observation_rows(tokenizer, sentences, config)
    directions = build_direction_rows(observations, config)
    direction_ids = [row["direction_id"] for row in directions]
    if len(direction_ids) != len(set(direction_ids)):
        raise RuntimeError("direction IDs are not unique")

    config.plan_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(config.plan_dir / "contexts.jsonl", contexts)
    write_jsonl(config.plan_dir / "observations.jsonl", observations)
    write_jsonl(config.plan_dir / "directions.jsonl", directions)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": config.protocol_version,
        "protocol_status": config.protocol_status,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "model_name": config.model_name,
        "model_revision": config.model_revision,
        "tokenizer_revision": config.tokenizer_revision,
        "tokenizer_class": tokenizer.__class__.__name__,
        "torch_version": torch.__version__,
        "presentation_context": config.context_id,
        "position_policy": config.position_policy,
        "point_weighting": config.point_weighting,
        "n_sentences": len(sentences),
        "n_contexts": len(contexts),
        "n_observations": len(observations),
        "n_directions": len(directions),
    }
    with (config.plan_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
