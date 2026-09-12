"""Configuration schema for Experiment 0 of the experimental protocol."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


@dataclass(frozen=True)
class ConceptSpec:
    name: str
    dataset: str
    split: str


@dataclass(frozen=True)
class Experiment0Config:
    protocol_version: str
    protocol_status: str
    model_name: str
    model_revision: str
    tokenizer_revision: str
    num_decoder_blocks: Optional[int]
    layers: List[int]
    activation_site: str
    concepts: List[ConceptSpec]
    vector_type: str
    hidden_state_offset: int
    fixed_random_count_per_layer: int
    fixed_random_base_seed: int
    fixed_random_layer_stride: int
    noise_repetitions_per_position: int
    renewed_noise_base_seed: int
    presentation_mode: str
    context_id: str
    context_template: str
    context_manifest: Optional[Path]
    position_policy: str
    point_weighting: str
    bootstrap_unit: str
    bootstrap_resamples_sd: int
    bootstrap_resamples_mad: int
    bootstrap_seed: int
    ci_level: float
    batch_size: int
    direction_chunk_size: int
    vector_dir: Path
    plan_dir: Path
    output_dir: Path

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Experiment0Config":
        protocol = raw["protocol"]
        model = raw["model"]
        directions = raw["directions"]
        presentation = raw["presentation"]
        statistics = raw["statistics"]
        execution = raw["execution"]
        paths = raw["paths"]
        config = cls(
            protocol_version=str(protocol["version"]),
            protocol_status=str(protocol["status"]),
            model_name=str(model["name"]),
            model_revision=str(model["revision"]),
            tokenizer_revision=str(model["tokenizer_revision"]),
            # Optional, and checked against `layers` so that a config written for one
            # model cannot silently calibrate a different block count on another.
            num_decoder_blocks=(
                int(model["num_decoder_blocks"])
                if model.get("num_decoder_blocks") is not None
                else None
            ),
            layers=[int(layer) for layer in raw["layers"]],
            activation_site=str(raw["activation_site"]),
            concepts=[ConceptSpec(**entry) for entry in directions["concepts"]],
            vector_type=str(directions["concept_vector_type"]),
            hidden_state_offset=int(directions["hidden_state_offset"]),
            fixed_random_count_per_layer=int(
                directions["fixed_random"]["count_per_layer"]
            ),
            fixed_random_base_seed=int(
                directions["fixed_random"]["base_seed"]
            ),
            fixed_random_layer_stride=int(
                directions["fixed_random"]["layer_stride"]
            ),
            noise_repetitions_per_position=int(
                directions["renewed_noise"]["repetitions_per_position"]
            ),
            renewed_noise_base_seed=int(
                directions["renewed_noise"]["base_seed"]
            ),
            presentation_mode=str(presentation["mode"]),
            context_id=str(presentation["context_id"]),
            context_template=str(presentation.get("template", "")),
            context_manifest=(
                repo_path(presentation["manifest"])
                if presentation.get("manifest")
                else None
            ),
            position_policy=str(presentation["position_policy"]),
            point_weighting=str(statistics["point_weighting"]),
            bootstrap_unit=str(statistics["bootstrap_unit"]),
            bootstrap_resamples_sd=int(statistics["bootstrap_resamples_sd"]),
            bootstrap_resamples_mad=int(statistics["bootstrap_resamples_mad"]),
            bootstrap_seed=int(statistics["bootstrap_seed"]),
            ci_level=float(statistics["ci_level"]),
            batch_size=int(execution["batch_size"]),
            direction_chunk_size=int(execution["direction_chunk_size"]),
            vector_dir=repo_path(paths["concept_vectors"]),
            plan_dir=repo_path(paths["plan_dir"]),
            output_dir=repo_path(paths["output_dir"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.protocol_status not in {"development", "frozen"}:
            raise ValueError("protocol.status must be development or frozen")
        if not self.layers or self.layers != list(range(len(self.layers))):
            raise ValueError(
                "Experiment 0 must calibrate every decoder block, listed as the "
                "contiguous range 0 through N-1"
            )
        if self.num_decoder_blocks is not None and len(self.layers) != self.num_decoder_blocks:
            raise ValueError(
                "model.num_decoder_blocks is {} but {} layers are listed".format(
                    self.num_decoder_blocks, len(self.layers)
                )
            )
        if self.activation_site != "decoder_block_output":
            raise ValueError("Experiment 0 requires activation_site=decoder_block_output")
        if len({concept.name for concept in self.concepts}) != len(self.concepts):
            raise ValueError("concept names must be unique")
        if any(concept.dataset not in {"simple_data", "complex_data"} for concept in self.concepts):
            raise ValueError("concept dataset must be simple_data or complex_data")
        if any(
            concept.split not in {"development", "hold_out", "unassigned"}
            for concept in self.concepts
        ):
            raise ValueError("concept split must be development, hold_out or unassigned")
        if self.vector_type not in {"avg", "last"}:
            raise ValueError("concept_vector_type must be avg or last")
        if self.hidden_state_offset != 1:
            raise ValueError("decoder-block outputs require hidden_state_offset=1")
        if self.fixed_random_count_per_layer < 1 or self.noise_repetitions_per_position < 1:
            raise ValueError("direction counts must be positive")
        if min(self.fixed_random_base_seed, self.renewed_noise_base_seed) < 0:
            raise ValueError("direction base seeds must be non-negative")
        if self.fixed_random_layer_stride < self.fixed_random_count_per_layer:
            raise ValueError(
                "fixed-random layer_stride must cover every per-layer sample"
            )
        if self.presentation_mode not in {"template_per_sentence", "external_manifest"}:
            raise ValueError("unsupported presentation.mode")
        if (
            self.presentation_mode == "template_per_sentence"
            and self.context_template.count("{sentence}") != 1
        ):
            raise ValueError("presentation.template must contain {sentence} exactly once")
        if self.presentation_mode == "external_manifest" and self.context_manifest is None:
            raise ValueError("external_manifest presentation requires presentation.manifest")
        if self.position_policy != "all_sentence_tokens":
            raise ValueError("only the explicit all_sentence_tokens policy is implemented")
        if self.point_weighting != "equal_token":
            raise ValueError("only point_weighting=equal_token is implemented")
        if self.bootstrap_unit != "sentence":
            raise ValueError("Experiment 0 bootstrap_unit must be sentence")
        if min(self.bootstrap_resamples_sd, self.bootstrap_resamples_mad) < 2:
            raise ValueError("bootstrap resample counts must be at least two")
        if not 0.0 < self.ci_level < 1.0:
            raise ValueError("ci_level must lie strictly between zero and one")
        if min(self.batch_size, self.direction_chunk_size) < 1:
            raise ValueError("execution sizes must be positive")
        if self.protocol_status == "frozen" and (
            self.model_revision in {"", "main"}
            or self.tokenizer_revision in {"", "main"}
            or self.context_id.endswith("development")
            or self.presentation_mode != "external_manifest"
            or any(concept.split == "unassigned" for concept in self.concepts)
        ):
            raise ValueError(
                "a frozen calibration requires immutable model/tokenizer revisions "
                "a non-development context, and assigned concept splits"
            )


def load_config(path: str) -> Experiment0Config:
    config_path = repo_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Experiment0Config.from_dict(raw)
