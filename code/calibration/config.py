"""Focused configuration for s(layer, direction) estimation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .common import resolve_repo_path


@dataclass
class CalibrationConfig:
    model: str
    seed: int
    layers: List[int]
    concepts: List[str]
    vector_type: str
    vector_hidden_state_offset: int
    random_directions_per_layer: int
    noise_directions_per_layer: int
    estimator: str
    corpus_source: str
    minimum_sentences: int
    batch_size: int
    vector_dir: Path
    directions_path: Path
    output_dir: Path

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "CalibrationConfig":
        directions = raw["directions"]
        corpus = raw["corpus"]
        paths = raw["paths"]
        config = cls(
            model=str(raw["model"]),
            seed=int(raw["seed"]),
            layers=[int(value) for value in raw["layers"]],
            concepts=[str(value) for value in raw["concepts"]],
            vector_type=str(raw.get("vector_type", "avg")),
            vector_hidden_state_offset=int(raw.get("vector_hidden_state_offset", 1)),
            random_directions_per_layer=int(directions.get("random_per_layer", 10)),
            noise_directions_per_layer=int(directions.get("noise_per_layer", 50)),
            estimator=str(raw.get("estimator", "sd")),
            corpus_source=str(corpus.get("source", "repository")),
            minimum_sentences=int(corpus.get("minimum_sentences", 100)),
            batch_size=int(raw.get("batch_size", 16)),
            vector_dir=resolve_repo_path(paths["vectors"]),
            directions_path=resolve_repo_path(paths["directions"]),
            output_dir=resolve_repo_path(paths["output_dir"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.layers or min(self.layers) < 0:
            raise ValueError("layers must contain non-negative indices")
        if not self.concepts:
            raise ValueError("at least one concept is required")
        if self.vector_type not in {"avg", "last"}:
            raise ValueError("vector_type must be avg or last")
        if self.vector_hidden_state_offset < 0:
            raise ValueError("vector_hidden_state_offset must be non-negative")
        if self.random_directions_per_layer < 1 or self.noise_directions_per_layer < 1:
            raise ValueError("direction counts must be positive")
        if self.estimator not in {"sd", "mad"}:
            raise ValueError("estimator must be sd or mad")
        if self.corpus_source != "repository":
            raise ValueError("only corpus.source=repository is currently supported")
        if self.minimum_sentences < 1 or self.batch_size < 1:
            raise ValueError("minimum_sentences and batch_size must be positive")


def load_config(path: str) -> CalibrationConfig:
    config_path = resolve_repo_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return CalibrationConfig.from_dict(yaml.safe_load(handle))
