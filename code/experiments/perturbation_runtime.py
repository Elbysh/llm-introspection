"""Shared calibrated perturbation mechanics. No experiment-specific task runner."""

import csv
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT / "code", REPO_ROOT / "code" / "utils"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiment_0_calibration.prepare_material import materialize_direction, unit
from experiment_0_calibration.protocol_config import (
    load_config as load_experiment0_config,
)
from experiment_0_calibration.protocol_config import repo_path
from gaussian_dropout_hooks import make_dropout_hook, make_vector_injection_hook
from save_random_vectors import derive_seed

MAPPINGS = ("XY", "YX")
FAMILIES = ("concept", "random", "noise", "dropout")
MATCHINGS = ("alpha", "z_sd", "z_mad")
LEGACY_MATCHING = "z"
EXPERIMENT1_ALPHA_DOSES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
EXPERIMENT1_Z_DOSES = (
    0.01,
    0.02,
    0.04,
    0.08,
    0.16,
    0.32,
    0.64,
    1.28,
    2.56,
    5.12,
    10.24,
    20.48,
)


# -----------------------------------------------------------------------------
# Common calibrated intervention mechanics for the separate experiment runners.


def log_grid(low, high, count):
    if not 0 < low < high:
        raise ValueError("dose bounds must satisfy 0 < low < high")
    if count < 2:
        raise ValueError("a grid needs at least two doses")
    step = (high / low) ** (1.0 / (count - 1))
    return [low * step**index for index in range(count)]


def parse_doses(explicit, bounds, count, default):
    if explicit:
        doses = [float(value) for value in explicit]
    elif bounds:
        doses = log_grid(float(bounds[0]), float(bounds[1]), count)
    else:
        doses = list(default)
    if any(dose <= 0 for dose in doses):
        raise ValueError(
            "doses must be strictly positive; the sham carries the null dose"
        )
    return sorted(doses)


def dropout_rate_for_amplitude(alpha, token_norm):
    if not token_norm > 0.0:
        raise ValueError("activation norm must be positive")
    rho = float(alpha) / float(token_norm)
    return rho * rho / (1.0 + rho * rho)


def estimator_for_matching(matching, default="sd"):
    """Resolve an explicit standardized arm to its calibration estimator."""
    if matching == "alpha":
        return None
    if matching == "z_sd":
        return "sd"
    if matching == "z_mad":
        return "mad"
    if matching == LEGACY_MATCHING:
        return default
    raise ValueError(f"unknown matching {matching!r}")


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


def make_token_varying_injection_hook(delta, token_range):
    def hook_fn(module, inputs, output):
        del module, inputs
        tensor = _hidden(output)
        start, end = token_range
        end = min(end, tensor.shape[1])
        if start >= end:
            return output
        rows = delta[: end - start].to(device=tensor.device, dtype=tensor.dtype)
        modified = tensor.clone()
        modified[:, start:end, :] = modified[:, start:end, :] + rows.unsqueeze(0)
        return (modified,) + output[1:] if isinstance(output, tuple) else modified

    return hook_fn


def make_capture_hook(store, key, token_range):
    def hook_fn(module, inputs, output):
        del module, inputs
        tensor = _hidden(output)
        start, end = token_range
        end = min(end, tensor.shape[1])
        store[key] = tensor[0, start:end, :].detach().to(torch.float32).clone()

    return hook_fn


def make_norm_capture_hook(store, layer, ranges):
    def hook_fn(module, inputs, output):
        del module, inputs
        tensor = _hidden(output)
        for index, (start, end) in enumerate(ranges):
            stop = min(end, tensor.shape[1])
            segment = tensor[0, start:stop, :].detach().to(torch.float32)
            store[(int(layer), index)] = float(
                segment.pow(2).sum(dim=-1).mean().sqrt().item()
            )

    return hook_fn


@contextmanager
def intervention(
    model,
    layer,
    token_range,
    family,
    vector=None,
    alpha=None,
    delta=None,
    rate=None,
    seed=0,
    measure=None,
):
    block = model.model.layers[layer]
    handles = []
    try:
        if measure is not None:
            handles.append(
                block.register_forward_hook(
                    make_capture_hook(measure, "clean", token_range)
                )
            )
        if family in ("concept", "random"):
            handles.append(
                block.register_forward_hook(
                    make_vector_injection_hook(vector, [(token_range, float(alpha))])
                )
            )
        elif family == "noise":
            handles.append(
                block.register_forward_hook(
                    make_token_varying_injection_hook(delta, token_range)
                )
            )
        elif family == "dropout":
            handles.append(
                block.register_forward_hook(
                    make_dropout_hook(float(rate), token_range, seed=seed)
                )
            )
        else:
            raise ValueError(f"unknown family {family!r}")
        if measure is not None:
            handles.append(
                block.register_forward_hook(
                    make_capture_hook(measure, "perturbed", token_range)
                )
            )
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def run_trial(
    model,
    encoding,
    layer,
    token_range,
    family,
    token_a,
    token_b,
    vector=None,
    alpha=None,
    delta=None,
    rate=None,
    seed=0,
    measure=True,
):
    captured = {} if measure else None
    with intervention(
        model,
        layer,
        token_range,
        family,
        vector=vector,
        alpha=alpha,
        delta=delta,
        rate=rate,
        seed=seed,
        measure=captured,
    ):
        logits = model(**encoding).logits[0, -1, :]
    realized = None
    clean_norm = None
    if captured and "clean" in captured and "perturbed" in captured:
        difference = captured["perturbed"] - captured["clean"]
        realized = float(difference.norm(dim=-1).mean().item())
        clean_norm = float(captured["clean"].pow(2).sum(dim=-1).mean().sqrt().item())
    logit_a = float(logits[token_a].item())
    logit_b = float(logits[token_b].item())
    finite = bool(torch.isfinite(logits).all().item())
    mass = (
        float(torch.softmax(logits.float(), dim=-1)[[token_a, token_b]].sum().item())
        if finite
        else None
    )
    return {
        "choice_probability_mass": mass,
        "logit_a": logit_a,
        "logit_b": logit_b,
        "contrast": logit_a - logit_b,
        "top_token": int(torch.argmax(logits).item()),
        "finite": bool(torch.isfinite(logits).all().item()),
        "realized_amplitude": realized,
        "clean_token_norm": clean_norm,
    }


class Experiment0Calibration:
    """Read the all-layer calibration format currently produced on ``main``.

    Direction vectors are loaded lazily from the exact concept or fixed-random vector
    file. Renewed-noise directions are reconstructed from the seed recorded by
    Experiment 0. Keeping scale rows and vectors behind the same object prevents a z
    dose from being paired with the wrong direction.
    """

    FAMILY_NAMES = {  # noqa: RUF012 - immutable class-level lookup table
        "concept": "concept",
        "fixed_random": "random",
        "renewed_noise": "noise",
    }
    ESTIMATOR_FIELDS = {  # noqa: RUF012 - immutable class-level lookup table
        "sd": "sd",
        "mad": "mad_corrected",
    }

    def __init__(self, config, records, run_manifest, results_dir, estimator):
        self.source_config = config
        self.records = {row["direction_id"]: row for row in records}
        self.run_manifest = run_manifest
        self.results_dir = Path(results_dir)
        self.estimator = estimator
        self.config_path = None
        self.config_sha256 = None
        self.config_matches_manifest = None
        self.config = type(
            "CalibrationView",
            (),
            {
                "model": config.model_name,
                "layers": list(config.layers),
                "concepts": [concept.name for concept in config.concepts],
                "output_dir": self.results_dir,
            },
        )()
        self._by_kind_layer = defaultdict(list)
        for direction_id, row in self.records.items():
            family = self.FAMILY_NAMES.get(row["direction_family"])
            if family is not None:
                self._by_kind_layer[(family, int(row["decoder_block_index"]))].append(
                    direction_id
                )
        for direction_ids in self._by_kind_layer.values():
            direction_ids.sort()
        concept_row = next(
            (row for row in records if row["direction_family"] == "concept"), None
        )
        if concept_row is None:
            raise ValueError("Experiment 0 calibration contains no concept direction")
        payload = torch.load(
            REPO_ROOT / concept_row["source_path"],
            map_location="cpu",
            weights_only=False,
        )
        self.hidden_size = int(payload["vector"].numel())
        self._vector_cache = {}
        self._fixed_random_files = {}
        fixed_random_pattern = re.compile(r"fixed_random__block_(\d+)__(\d+)$")
        for direction_id, row in self.records.items():
            if row["direction_family"] != "fixed_random":
                continue
            match = fixed_random_pattern.fullmatch(direction_id)
            if match is None:
                # This keeps the adapter usable with small synthetic fixtures. Every
                # production Experiment 0 row uses the canonical identifier.
                continue
            layer, sample = (int(value) for value in match.groups())
            path = config.vector_dir / (
                f"random_s{sample}_{layer}_{config.vector_type}.pt"
            )
            if not path.exists():
                raise FileNotFoundError(
                    f"missing committed fixed-random vector for {direction_id}: {path}"
                )
            self._fixed_random_files[direction_id] = path

    @classmethod
    def load(cls, config_path, results_dir, estimator="sd"):
        resolved_config_path = repo_path(str(config_path))
        config = load_experiment0_config(str(resolved_config_path))
        results_dir = Path(results_dir)
        if not results_dir.is_absolute():
            results_dir = REPO_ROOT / results_dir
        scales_path = results_dir / "directional_scales.json"
        manifest_path = results_dir / "run_manifest.json"
        if not scales_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(
                f"Experiment 0 results require {scales_path} and {manifest_path}"
            )
        with scales_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        layers = sorted({int(row["decoder_block_index"]) for row in records})
        if layers != list(config.layers):
            raise ValueError(
                f"calibration result layers {layers} do not match configuration "
                f"layers {config.layers}"
            )
        if any(row.get("activation_site") != config.activation_site for row in records):
            raise ValueError(
                "calibration result activation site does not match configuration"
            )
        expected_concepts = {concept.name for concept in config.concepts}
        expected_noise = (
            int(manifest["n_observations"]) * config.noise_repetitions_per_position
        )
        for layer in config.layers:
            layer_rows = [
                row for row in records if int(row["decoder_block_index"]) == layer
            ]
            concepts = {
                row["concept"]
                for row in layer_rows
                if row["direction_family"] == "concept"
            }
            random_count = sum(
                row["direction_family"] == "fixed_random" for row in layer_rows
            )
            noise_count = sum(
                row["direction_family"] == "renewed_noise" for row in layer_rows
            )
            if concepts != expected_concepts:
                raise ValueError(f"concept directions do not match at layer {layer}")
            if random_count != config.fixed_random_count_per_layer:
                raise ValueError(
                    f"fixed-random direction count does not match at layer {layer}"
                )
            if noise_count != expected_noise:
                raise ValueError(
                    f"renewed-noise direction count does not match at layer {layer}"
                )
            random_seeds = {
                int(row["seed"])
                for row in layer_rows
                if row["direction_family"] == "fixed_random"
            }
            expected_random_seeds = {
                config.fixed_random_base_seed
                + layer * config.fixed_random_layer_stride
                + index
                for index in range(config.fixed_random_count_per_layer)
            }
            if random_seeds != expected_random_seeds:
                raise ValueError(
                    f"fixed-random seeds do not match configuration at layer {layer}"
                )
            noise_seeds = {
                int(row["seed"])
                for row in layer_rows
                if row["direction_family"] == "renewed_noise"
            }
            expected_noise_seeds = {
                config.renewed_noise_base_seed + layer * expected_noise + index
                for index in range(expected_noise)
            }
            if noise_seeds != expected_noise_seeds:
                raise ValueError(
                    f"renewed-noise seeds do not match configuration at layer {layer}"
                )
        if manifest.get("model_name") != config.model_name:
            raise ValueError("calibration result model does not match configuration")
        if estimator not in cls.ESTIMATOR_FIELDS:
            raise ValueError(f"unknown calibration estimator {estimator!r}")
        invalid = [
            row["direction_id"]
            for row in records
            if not float(row[cls.ESTIMATOR_FIELDS[estimator]]) > 0.0
            or not row.get("valid_for_sd_normalization", True)
        ]
        if invalid:
            raise ValueError(
                f"{len(invalid)} calibration directions are invalid; first: {invalid[0]}"
            )
        instance = cls(config, records, manifest, results_dir, estimator)
        instance.config_path = resolved_config_path
        instance.config_sha256 = hashlib.sha256(
            resolved_config_path.read_bytes()
        ).hexdigest()
        instance.config_matches_manifest = instance.config_sha256 == manifest.get(
            "config_sha256"
        )
        return instance

    def direction_ids(self, kind, layer, concept=None):
        direction_ids = list(self._by_kind_layer.get((kind, int(layer)), []))
        if concept is not None:
            direction_ids = [
                direction_id
                for direction_id in direction_ids
                if self.records[direction_id].get("concept") == concept
            ]
        return direction_ids

    def vector(self, direction_id):
        if direction_id not in self._vector_cache:
            row = self.records[direction_id]
            fixed_path = self._fixed_random_files.get(direction_id)
            if fixed_path is None:
                if row["direction_family"] == "fixed_random":
                    generator = torch.Generator(device="cpu").manual_seed(
                        int(row["seed"])
                    )
                    vector, _ = unit(torch.randn(self.hidden_size, generator=generator))
                else:
                    vector, _ = materialize_direction(
                        row, self.source_config, self.hidden_size
                    )
            else:
                source_sha256 = row.get("source_sha256")
                if (
                    source_sha256
                    and hashlib.sha256(fixed_path.read_bytes()).hexdigest()
                    != source_sha256
                ):
                    raise ValueError(
                        f"fixed-random source hash mismatch for {direction_id}: "
                        f"{fixed_path}"
                    )
                payload = torch.load(fixed_path, map_location="cpu", weights_only=False)
                vector = payload.get("vector") if isinstance(payload, dict) else payload
                if not isinstance(vector, torch.Tensor):
                    raise ValueError(f"{fixed_path} does not contain a tensor vector")
                vector = vector.reshape(-1).to(torch.float32)
                match = re.fullmatch(r"fixed_random__block_(\d+)__(\d+)", direction_id)
                layer, sample = (int(value) for value in match.groups())
                expected_metadata = {
                    "experiment_0_direction_id": direction_id,
                    "decoder_block_index": layer,
                    "sample": sample,
                    "draw_seed": int(row["seed"]),
                    "seed_scheme": "linear",
                    "hidden_dim": self.hidden_size,
                    "model_name": self.source_config.model_name,
                }
                mismatches = {
                    key: (payload.get(key), expected)
                    for key, expected in expected_metadata.items()
                    if not isinstance(payload, dict) or payload.get(key) != expected
                }
                if mismatches:
                    raise ValueError(
                        f"fixed-random metadata mismatch for {direction_id}: {mismatches}"
                    )
                generator = torch.Generator(device="cpu").manual_seed(int(row["seed"]))
                regenerated, _ = unit(
                    torch.randn(self.hidden_size, generator=generator)
                )
                if vector.shape != regenerated.shape or not torch.equal(
                    vector, regenerated
                ):
                    raise ValueError(
                        f"committed vector {fixed_path} differs from the calibrated "
                        f"seed for {direction_id}"
                    )
            self._vector_cache[direction_id] = vector
        return self._vector_cache[direction_id]

    def scale(self, direction_id, estimator=None):
        estimator = estimator or self.estimator
        field = self.ESTIMATOR_FIELDS[estimator]
        value = float(self.records[direction_id][field])
        if not value > 0.0:
            raise ValueError(f"non-positive {field} for {direction_id}: {value}")
        return value

    def reference_scale(self, layer, estimator=None):
        direction_ids = self.direction_ids("random", layer)
        if not direction_ids:
            raise ValueError(f"no fixed-random scale at layer {layer}")
        return float(
            np.median(
                [self.scale(direction_id, estimator) for direction_id in direction_ids]
            )
        )

    def token_norm(self, layer):
        raise ValueError(
            "the current Experiment 0 output does not contain h_bar(layer); "
            "use --dropout_norm_source trial"
        )

    def provenance(self):
        return {
            "experiment_id": self.run_manifest.get("experiment_id"),
            "protocol_status": self.run_manifest.get("protocol_status"),
            "protocol_version": self.run_manifest.get("protocol_version"),
            "calibration_git_commit": self.run_manifest.get("git_commit"),
            "resolved_model_revision": self.run_manifest.get("resolved_model_revision"),
            "n_records": len(self.records),
            "results_dir": str(self.results_dir),
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
            "config_matches_manifest": self.config_matches_manifest,
            "fixed_random_vector_source": str(self.source_config.vector_dir),
            "fixed_random_vector_files": len(self._fixed_random_files),
            "fixed_random_validation": "metadata_and_exact_seed_regeneration_on_load",
        }


def build_conditions(calibration, args):
    """Enumerate the same four perturbation families at every selected layer."""
    conditions = defaultdict(list)
    for layer in args.layers:
        if "concept" in args.families:
            for concept in args.concepts:
                direction_ids = calibration.direction_ids("concept", layer, concept)
                if not direction_ids:
                    raise ValueError(f"concept {concept!r} is absent at layer {layer}")
                conditions[layer].extend(("concept", value) for value in direction_ids)
        if "random" in args.families:
            direction_ids = calibration.direction_ids("random", layer)[
                : args.num_random
            ]
            if len(direction_ids) != args.num_random:
                raise ValueError(
                    f"requested {args.num_random} random directions at layer {layer}, "
                    f"found {len(direction_ids)}"
                )
            conditions[layer].extend(("random", value) for value in direction_ids)
        if "noise" in args.families:
            if not calibration.direction_ids("noise", layer):
                raise ValueError(f"no renewed-noise directions at layer {layer}")
            conditions[layer].extend(
                ("noise", f"noise_realization_{index:02d}")
                for index in range(args.num_noise)
            )
        if "dropout" in args.families:
            conditions[layer].extend(
                ("dropout", f"dropout_realization_{index:02d}")
                for index in range(args.num_dropout)
            )
    return conditions


def noise_delta(calibration, layer, num_tokens, matching, dose, trial_seed):
    """Draw calibrated renewed-noise directions for every targeted token."""
    bank = calibration.direction_ids("noise", layer)
    rng = random.Random(derive_seed(trial_seed, "noise", layer))
    drawn = (
        rng.sample(bank, num_tokens)
        if len(bank) >= num_tokens
        else [bank[rng.randrange(len(bank))] for _ in range(num_tokens)]
    )
    rows = []
    amplitudes = []
    for direction_id in drawn:
        estimator = estimator_for_matching(matching, calibration.estimator)
        amplitude = (
            float(dose)
            if estimator is None
            else float(dose) * calibration.scale(direction_id, estimator)
        )
        rows.append(amplitude * calibration.vector(direction_id).to(torch.float32))
        amplitudes.append(amplitude)
    return torch.stack(rows), drawn, amplitudes


def perturbation_kwargs(
    calibration,
    args,
    family,
    direction_id,
    matching,
    dose,
    layer,
    token_range,
    sham,
    target_index,
    vectors,
    trial_seed,
):
    """Convert a raw or standardized dose into one family's hook parameters."""
    kwargs = {"seed": trial_seed}
    metadata = {"noise_direction_ids": None, "noise_token_amplitudes": None}
    rate = None
    estimator = estimator_for_matching(matching, calibration.estimator)
    if family in ("concept", "random"):
        alpha = (
            float(dose)
            if estimator is None
            else float(dose) * calibration.scale(direction_id, estimator)
        )
        kwargs.update(vector=vectors[direction_id], alpha=alpha)
    elif family == "noise":
        delta, used, amplitudes = noise_delta(
            calibration,
            layer,
            token_range[1] - token_range[0],
            matching,
            dose,
            trial_seed,
        )
        kwargs["delta"] = delta
        alpha = float(np.mean(amplitudes))
        metadata.update(noise_direction_ids=used, noise_token_amplitudes=amplitudes)
    else:
        token_norm = (
            sham["token_norms"][(layer, target_index)]
            if args.dropout_norm_source == "trial"
            else calibration.token_norm(layer)
        )
        alpha = (
            float(dose)
            if estimator is None
            else float(dose) * calibration.reference_scale(layer, estimator)
        )
        rate = dropout_rate_for_amplitude(alpha, token_norm)
        kwargs["rate"] = rate
    return kwargs, alpha, rate, metadata


def choice_token_ids(tokenizer, letters=("X", "Y")):
    """Require verified single-token answer continuations; never take a suffix token."""
    ids = []
    for letter in letters:
        candidates = [
            tokenizer.encode(" " + letter, add_special_tokens=False),
            tokenizer.encode(letter, add_special_tokens=False),
        ]
        valid = next((tokens for tokens in candidates if len(tokens) == 1), None)
        if valid is None:
            raise ValueError(f"No single-token continuation for {letter!r}")
        ids.append(valid[0])
    if len(set(ids)) != len(letters):
        raise ValueError("Answer letters map to the same token")
    return tuple(ids)


def recoded_score(logit_x, logit_y, mapping):
    """A positive score always favors the semantic response 'intervention'."""
    difference = float(logit_x) - float(logit_y)
    if mapping == "XY":
        return difference
    if mapping == "YX":
        return -difference
    raise ValueError(f"unknown response mapping {mapping!r}")


def binary_score(value):
    """Return 1 for intervention, 0 for no intervention, and 0.5 for a tie."""
    if value == 0.0:
        return 0.5
    return float(value > 0.0)


def response_labels(logit_x, logit_y, mapping):
    """Return the forced response token and its semantic interpretation."""
    difference = float(logit_x) - float(logit_y)
    if difference == 0.0:
        return "tie", "tie"
    letter = "X" if difference > 0.0 else "Y"
    intervention_letter = "X" if mapping == "XY" else "Y"
    semantic = "intervention" if letter == intervention_letter else "no_intervention"
    return letter, semantic


def _trial_id(payload):
    stable = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()[:20]


@torch.inference_mode()
def run_presence_sham(model, encoding, ranges, token_x, token_y, mapping, layers):
    norms = {}
    handles = []
    try:
        for layer in layers:
            handles.append(
                model.model.layers[layer].register_forward_hook(
                    make_norm_capture_hook(norms, layer, ranges)
                )
            )
        logits = model(**encoding).logits[0, -1, :]
    finally:
        for handle in handles:
            handle.remove()
    logit_x, logit_y = float(logits[token_x].item()), float(logits[token_y].item())
    return {
        "logit_x": logit_x,
        "logit_y": logit_y,
        "score": recoded_score(logit_x, logit_y, mapping),
        "top_token": int(torch.argmax(logits).item()),
        "finite": bool(torch.isfinite(logits).all().item()),
        "token_norms": norms,
    }


def write_trials_csv(path, trials):
    fields = []
    for row in trials:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trials)


def load_checkpoint(path):
    """Load completed perturbed trials from an append-only JSONL checkpoint."""
    if not path.exists():
        return []
    rows = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid checkpoint line {line_number} in {path}"
                ) from error
            trial_id = row.get("trial_id")
            if not trial_id or trial_id in seen:
                raise ValueError(
                    f"missing or duplicate trial_id at checkpoint line {line_number}"
                )
            seen.add(trial_id)
            rows.append(row)
    return rows
