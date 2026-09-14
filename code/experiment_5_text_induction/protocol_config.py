"""Explicit text selections and intervention cells; no inferred semantic labels."""

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
class Experiment5Config:
    raw: dict
    source_path: object

    def path(self, key):
        return repo_path(self.raw["paths"][key])

    def validate(self):
        r = self.raw
        if r["protocol"]["status"] not in ("development", "frozen"):
            raise ValueError("protocol.status must be development or frozen")
        conditions = r["conditions"]
        if not conditions or len({c["condition_id"] for c in conditions}) != len(conditions):
            raise ValueError("Supply explicit, uniquely identified intervention conditions")
        for c in conditions:
            if c.get("family", "concept") != "concept":
                raise ValueError("Experiment 5 injects only the concept evoked by the text")
            if type(c["layer"]) is not int or not 0 <= c["layer"] < 32:
                raise ValueError("Layers are zero-based decoder outputs 0..31")
            if c["dose_axis"] not in ("alpha", "z") or not math.isfinite(c["dose"]) or c["dose"] <= 0:
                raise ValueError("Use a positive finite alpha or z dose; sham is separate")
            if c["dose_role"] not in ("near_threshold", "above_threshold"):
                raise ValueError("Label each retained dose as near_threshold or above_threshold")
            if c["scale_statistic"] not in ("sd", "mad_corrected"):
                raise ValueError("Use sd or mad_corrected from the calibration")
        statistics = r["statistics"]
        if type(statistics["bootstrap_resamples"]) is not int or statistics["bootstrap_resamples"] < 2:
            raise ValueError("Specify at least two bootstrap resamples")
        if type(statistics["bootstrap_seed"]) is not int or statistics["bootstrap_seed"] < 0:
            raise ValueError("Specify a nonnegative bootstrap seed")
        if not 0 < statistics["ci_level"] < 1:
            raise ValueError("ci_level must be between zero and one")
        if r["execution"]["dtype"] not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unsupported dtype")
        gap = r["texts"]["max_token_length_difference"]
        if gap is not None and (type(gap) is not int or gap < 0):
            raise ValueError("Token-length tolerance must be a nonnegative integer or null")
        if r["protocol"]["status"] == "frozen":
            for field in ("revision", "tokenizer_revision"):
                revision = r["model"][field]
                if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
                    raise ValueError("Frozen revisions must be immutable model/tokenizer hashes")
            if {c["layer"] for c in conditions} != set(range(32)):
                raise ValueError("The principal experiment explores all 32 layers")
            groups = {}
            for condition in conditions:
                key = (condition["concept"], condition["layer"], condition["dose_axis"], condition["scale_statistic"])
                groups.setdefault(key, set()).add(condition["dose_role"])
            if any(roles != {"near_threshold", "above_threshold"} for roles in groups.values()):
                raise ValueError("Each frozen concept/layer/axis/statistic needs near- and above-threshold doses")


def load_config(path):
    source = repo_path(path)
    with source.open(encoding="utf-8") as stream:
        config = Experiment5Config(yaml.safe_load(stream), source)
    config.validate()
    return config
