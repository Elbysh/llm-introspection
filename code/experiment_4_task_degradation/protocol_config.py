"""Explicit inputs for Experiment 4; scientific selections have no hidden defaults."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import yaml
from experiment_0_calibration.protocol_config import repo_path


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class Experiment4Config:
    raw: dict
    source_path: Path

    @property
    def output_dir(self):
        return repo_path(self.raw["paths"]["output_dir"])

    @property
    def plan_dir(self):
        return repo_path(self.raw["paths"]["plan_dir"])

    def path(self, key):
        return repo_path(self.raw["paths"][key])

    def validate(self):
        r = self.raw
        if r["protocol"]["status"] not in ("development", "frozen"):
            raise ValueError("protocol.status must be development or frozen")
        if not r.get("examples"):
            raise ValueError("Select labelled examples explicitly before preparation")
        for spec in r["examples"]:
            positive, negative = spec["positive_indices"], spec["negative_indices"]
            if not positive or len(positive) != len(negative):
                raise ValueError("Each probed concept needs equally many positive and negative examples")
            for indices in (positive, negative):
                if len(set(indices)) != len(indices) or any(type(i) is not int or i < 0 for i in indices):
                    raise ValueError("Example indices must be distinct nonnegative integers")
        if len({s["probed_concept"] for s in r["examples"]}) != len(r["examples"]):
            raise ValueError("Duplicate probed concepts")
        if r["diagnostics"].get("js_example_ids") is None:
            raise ValueError("Specify the JS subset before execution (example IDs, or [] explicitly)")
        if r["execution"]["dtype"] not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unsupported execution dtype")
        if r["protocol"]["status"] == "frozen":
            if not r["diagnostics"]["js_example_ids"]:
                raise ValueError("A principal run must include the specified JS diagnostic subset")
            for field in ("revision", "tokenizer_revision"):
                revision = r["model"][field]
                if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
                    raise ValueError("Frozen model/tokenizer revisions must be immutable SHA-1 hashes")


def load_config(path):
    source = repo_path(path)
    with source.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    config = Experiment4Config(raw, source)
    config.validate()
    return config


def validate_conditions(config):
    """Validate the explicitly configured Experiment 4 intervention conditions."""
    rows = config.raw["conditions"]
    if not rows or len({r["condition_id"] for r in rows}) != len(rows):
        raise ValueError("Conditions must be nonempty and uniquely identified")
    for row in rows:
        if row["family"] not in ("concept", "fixed_random", "renewed_noise"):
            raise ValueError("Unknown perturbation family")
        if type(row["layer"]) is not int or not 0 <= row["layer"] < 32:
            raise ValueError("Expected a zero-based decoder output index in [0,31]")
        if row["dose_axis"] not in ("alpha", "z"):
            raise ValueError("Conditions must explicitly use alpha or z")
        if not math.isfinite(row["dose"]) or row["dose"] <= 0:
            raise ValueError("Intervention doses must be finite and positive; sham is separate")
        if row["scale_statistic"] not in ("sd", "mad_corrected"):
            raise ValueError("Use sd or mad_corrected, as in Experiment 0 records")
        if not row.get("intervention_id"):
            raise ValueError("A stable direction/noise-rule ID is required for reporting")
        if row["family"] == "renewed_noise" and row.get("noise_rule") != "independent_unit_gaussian_per_trial_token":
            raise ValueError("Noise must follow independent_unit_gaussian_per_trial_token")
        if not row.get("probed_concepts") or not set(row["probed_concepts"]) <= {
                s["probed_concept"] for s in config.raw["examples"]}:
            raise ValueError("Declare each condition's probed-concept associations explicitly")
    if config.raw["protocol"]["status"] == "frozen":
        if {r["layer"] for r in rows} != set(range(32)):
            raise ValueError("The principal experiment covers all 32 decoder outputs")
        if {r["family"] for r in rows} != {"concept", "fixed_random", "renewed_noise"}:
            raise ValueError("The principal experiment requires all three included intervention families")
        if {r["dose_axis"] for r in rows} != {"alpha", "z"}:
            raise ValueError("The principal experiment requires both dose parametrizations")
    return rows
