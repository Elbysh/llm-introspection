"""Read side of the Experiment 0 calibration, for the experiments that consume it.

Experiment 0 persists one record per calibrated `(decoder block, direction)` pair:
the identity and provenance of the direction, and the natural scale `s(l, v)` of its
projections. A downstream experiment needs the direction and its scale together --
a `z = alpha / s(l, v)` dose is only meaningful for the exact direction the scale was
estimated on -- so this module hands out both from the persisted artifacts and never
re-derives either.

Directions are not stored as tensors. Concept directions are reloaded from the
`.pt` file recorded in `source_path`; fixed-random and renewed-noise directions are
regenerated from the recorded seed by the very function that created them,
`prepare_material.materialize_direction`. Every regenerated vector is checked against
the recorded `original_direction_norm`, so a drift in PyTorch's Gaussian generator or
an edited artifact is loud rather than silent.

Family names differ between the protocol and the behavioural experiments. Experiment 0
speaks of `concept`, `fixed_random` and `renewed_noise`; the intervention families of
section 3 are `concept`, `random` and `noise`. Both vocabularies are accepted here.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

from . import EXPERIMENT_ID
from .prepare_material import materialize_direction
from .protocol_config import REPO_ROOT, load_config, repo_path


# The committed development calibration. A fresh run writes to its own
# `paths.output_dir`; pass that directory explicitly to use it.
DEFAULT_CALIBRATION_DIR = REPO_ROOT / "results" / "experiment_0_calibration"

SCALES_FILE = "directional_scales.json"
MANIFEST_FILE = "run_manifest.json"

# Intervention family (doc section 3) -> Experiment 0 direction family.
FAMILY_ALIASES = {
    "concept": "concept",
    "random": "fixed_random",
    "noise": "renewed_noise",
    "scrambled": "scrambled_concept",
}

# Dose estimator -> the column of directional_scales.json that carries it.
# `sd` is the protocol's primary analysis, `mad` its robustness check.
ESTIMATOR_FIELDS = {"sd": "sd", "mad": "mad_corrected"}


def resolve_family(family: str) -> str:
    """Accept either vocabulary and return the Experiment 0 family name."""
    if family in FAMILY_ALIASES:
        return FAMILY_ALIASES[family]
    if family in set(FAMILY_ALIASES.values()):
        return family
    raise ValueError(
        "unknown direction family {!r}; expected one of {}".format(
            family, sorted(set(FAMILY_ALIASES) | set(FAMILY_ALIASES.values()))
        )
    )


def infer_hidden_size(records: Iterable[Dict[str, Any]]) -> int:
    """Hidden size of the calibrated model, read off a stored concept vector.

    The random and noise directions are Gaussian draws of that width, and the draw
    only reproduces if the width matches the one used at calibration time, so it is
    read from the data rather than assumed.
    """
    for record in records:
        if record.get("direction_family") != "concept":
            continue
        path = repo_path(record["source_path"])
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return int(payload["vector"].reshape(-1).numel())
    raise FileNotFoundError(
        "cannot infer the hidden size: no concept vector of the calibration is "
        "readable. Check paths.concept_vectors in the calibration config."
    )


class DirectionBank:
    """Calibrated directions and their natural scales, keyed by direction id."""

    def __init__(self, config, records, calibration_dir, estimator="sd",
                 hidden_size=None, manifest=None):
        if estimator not in ESTIMATOR_FIELDS:
            raise ValueError(
                "estimator must be one of {}".format(sorted(ESTIMATOR_FIELDS))
            )
        self.config = config
        self.calibration_dir = Path(calibration_dir)
        self.estimator = estimator
        self.manifest = manifest or {}
        self.records = {row["direction_id"]: row for row in records}
        if len(self.records) != len(records):
            raise ValueError("duplicate direction_id in {}".format(SCALES_FILE))
        self.hidden_size = int(hidden_size or infer_hidden_size(records))

        self._by_family_layer = defaultdict(list)
        for direction_id, row in self.records.items():
            key = (row["direction_family"], int(row["decoder_block_index"]))
            self._by_family_layer[key].append(direction_id)
        for ids in self._by_family_layer.values():
            ids.sort()
        self._vectors: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(cls, config_path, calibration_dir=None, estimator="sd", strict=True):
        """Load a calibration from its config and its output directory.

        `config_path` fixes what was calibrated (model, layers, concepts, vector
        directory); `calibration_dir` holds what came out of the run. They are
        separate arguments because the committed development results do not sit at
        the `paths.output_dir` of the config that produced them.
        """
        config = load_config(config_path)
        directory = Path(calibration_dir) if calibration_dir else DEFAULT_CALIBRATION_DIR
        if not directory.is_absolute():
            directory = REPO_ROOT / directory
        scales_path = directory / SCALES_FILE
        if not scales_path.exists():
            raise FileNotFoundError(
                "missing {}. Run experiment_0_calibration.run_experiment_0 for this "
                "config, or point --calibration_dir at the directory holding its "
                "results.".format(scales_path)
            )
        with scales_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)

        manifest = {}
        manifest_path = directory / MANIFEST_FILE
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            _check_manifest(manifest, config_path, strict=strict)

        bank = cls(config, records, directory, estimator=estimator, manifest=manifest)
        bank.validate_against_config()
        return bank

    def validate_against_config(self) -> None:
        """Refuse a calibration that does not cover what the config declares."""
        calibrated = set(self.layers)
        missing_layers = [layer for layer in self.config.layers if layer not in calibrated]
        if missing_layers:
            raise ValueError(
                "the calibration in {} does not cover decoder blocks {}".format(
                    self.calibration_dir, missing_layers
                )
            )
        for concept in self.config.concepts:
            for layer in self.config.layers:
                if not self.direction_ids("concept", layer, concept=concept.name):
                    raise ValueError(
                        "concept {!r} is not calibrated at decoder block {}".format(
                            concept.name, layer
                        )
                    )

    # ------------------------------------------------------------------ lookups

    @property
    def layers(self) -> List[int]:
        return sorted({int(row["decoder_block_index"]) for row in self.records.values()})

    @property
    def concepts(self) -> List[str]:
        return [concept.name for concept in self.config.concepts]

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def direction_ids(self, family: str, layer: int, concept: Optional[str] = None):
        ids = self._by_family_layer.get((resolve_family(family), int(layer)), [])
        if concept is not None:
            ids = [i for i in ids if self.records[i].get("concept") == concept]
        return list(ids)

    def record(self, direction_id: str) -> Dict[str, Any]:
        try:
            return self.records[direction_id]
        except KeyError:
            raise KeyError(
                "{!r} is not in the calibration at {}".format(
                    direction_id, self.calibration_dir
                )
            ) from None

    # ------------------------------------------------------------------ contents

    def vector(self, direction_id: str) -> torch.Tensor:
        """The unit direction the scale was estimated on, as float32 on the CPU."""
        cached = self._vectors.get(direction_id)
        if cached is not None:
            return cached
        record = self.record(direction_id)
        if record["direction_family"] == "concept":
            source = repo_path(record["source_path"])
            if not source.exists():
                raise FileNotFoundError(
                    "missing concept vector {} for {}. Regenerate it with\n"
                    "    python -m experiment_0_calibration.prepare_concept_vectors "
                    "--config <calibration config>".format(source, direction_id)
                )
        vector, original_norm = materialize_direction(record, self.config, self.hidden_size)
        expected = record.get("original_direction_norm")
        if expected is not None and not np.isclose(original_norm, float(expected),
                                                   rtol=1e-5, atol=1e-8):
            raise ValueError(
                "{} does not reproduce: norm {:.9g} but the calibration recorded "
                "{:.9g}. The direction the scale was estimated on is not the one "
                "being rebuilt.".format(direction_id, original_norm, float(expected))
            )
        self._vectors[direction_id] = vector
        return vector

    def scale(self, direction_id: str) -> float:
        """s(l, v) under the selected estimator."""
        record = self.record(direction_id)
        value = float(record[ESTIMATOR_FIELDS[self.estimator]])
        if not value > 0.0:
            raise ValueError(
                "non-positive {} scale for {}: {}".format(self.estimator, direction_id, value)
            )
        if self.estimator == "sd" and not record.get("valid_for_sd_normalization", True):
            raise ValueError(
                "{} is flagged invalid for SD normalization".format(direction_id)
            )
        return value

    def reference_scale(self, layer: int) -> float:
        """s_bar(l): the natural scale of a generic direction (doc 3.6).

        Dropout picks no direction before the injection, so its standardized dose is
        matched against the median scale of the fixed-random bank at that block.
        """
        ids = self.direction_ids("random", layer)
        if not ids:
            raise ValueError(
                "no fixed-random direction calibrated at decoder block {}".format(layer)
            )
        return float(np.median([self.scale(direction_id) for direction_id in ids]))

    def token_norm(self, layer: int) -> float:
        """h_bar(l): the RMS activation norm, to turn an amplitude into a rate.

        Experiment 0 records projections onto directions, not activation norms, so
        this quantity is not part of its artifacts. Measuring it on the trial's own
        targeted tokens is exact and is what the experiments do by default.
        """
        raise NotImplementedError(
            "Experiment 0 does not record per-layer activation norms; there is no "
            "h_bar({}) in {}. Measure it on the trial itself instead "
            "(--dropout_norm_source trial).".format(layer, self.calibration_dir)
        )

    def preflight(self, families, layers, concepts=None, num_random=None):
        """Materialize every fixed direction a plan needs, before any model is loaded.

        A missing concept vector or an unreproducible seed then fails in seconds
        rather than after the weights are on the GPU.
        """
        checked = 0
        for layer in layers:
            if "concept" in families:
                for concept in concepts or self.concepts:
                    ids = self.direction_ids("concept", layer, concept=concept)
                    if not ids:
                        raise ValueError(
                            "concept {!r} is not calibrated at decoder block {}".format(
                                concept, layer)
                        )
                    for direction_id in ids:
                        self.vector(direction_id)
                        self.scale(direction_id)
                        checked += 1
            if "scrambled" in families:
                for concept in concepts or self.concepts:
                    ids = self.direction_ids("scrambled", layer, concept=concept)
                    if not ids:
                        raise ValueError(
                            "concept {!r} has no scrambled counterpart at decoder block "
                            "{}; its calibration must enable "
                            "directions.scrambled_concept".format(concept, layer)
                        )
                    for direction_id in ids:
                        self.vector(direction_id)
                        self.scale(direction_id)
                        checked += 1
            if "random" in families or "dropout" in families:
                ids = self.direction_ids("random", layer)
                if not ids:
                    raise ValueError(
                        "no fixed-random direction calibrated at decoder block "
                        "{}".format(layer)
                    )
                for direction_id in ids[: num_random] if num_random else ids:
                    self.vector(direction_id)
                    self.scale(direction_id)
                    checked += 1
            if "noise" in families and not self.direction_ids("noise", layer):
                raise ValueError(
                    "no renewed-noise direction calibrated at decoder block "
                    "{}".format(layer)
                )
        return checked


def _check_manifest(manifest: Dict[str, Any], config_path, strict: bool = True) -> None:
    """Warn when the artifacts were not produced by the configuration being used."""
    from .step_07_persist_and_freeze import file_sha256

    if manifest.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("these calibration artifacts belong to another experiment")
    recorded = manifest.get("config_sha256")
    actual = file_sha256(repo_path(str(config_path)))
    if recorded and recorded != actual:
        message = (
            "the calibration in this directory was produced by a different version "
            "of {} (config sha256 {} vs {})".format(config_path, recorded, actual)
        )
        if strict:
            raise ValueError(message)
        print("WARNING: " + message, flush=True)

