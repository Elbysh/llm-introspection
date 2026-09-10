"""The reader that lets a behavioural experiment consume Experiment 0.

These tests run against the committed development calibration, because the point of
the module is that a direction and the scale estimated on it stay together: a
synthetic fixture would not catch a drift between the two.
"""

import copy
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from experiment_0_calibration.direction_bank import (  # noqa: E402
    DEFAULT_CALIBRATION_DIR,
    DirectionBank,
    resolve_family,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs" / "experiment_0_calibration" / "development_full.yaml"

pytestmark = pytest.mark.skipif(
    not (DEFAULT_CALIBRATION_DIR / "directional_scales.json").exists(),
    reason="the committed Experiment 0 calibration is not present",
)


@pytest.fixture(scope="module")
def bank():
    return DirectionBank.load(CONFIG)


def test_family_names_of_both_vocabularies_resolve():
    assert resolve_family("random") == "fixed_random"
    assert resolve_family("noise") == "renewed_noise"
    assert resolve_family("concept") == "concept"
    # The Experiment 0 spelling is accepted unchanged, so a record can be looked up
    # with the name it carries.
    assert resolve_family("renewed_noise") == "renewed_noise"
    with pytest.raises(ValueError):
        resolve_family("dropout")


def test_the_bank_covers_every_calibrated_block_and_concept(bank):
    assert bank.layers == list(range(32))
    assert bank.concepts == [concept.name for concept in bank.config.concepts]
    for layer in (0, 16, 31):
        assert len(bank.direction_ids("concept", layer)) == len(bank.concepts)
        assert len(bank.direction_ids("random", layer)) == 10
        assert bank.direction_ids("noise", layer)


def test_a_direction_is_rebuilt_at_unit_norm_and_matches_its_recorded_norm(bank):
    for family in ("random", "noise"):
        direction_id = bank.direction_ids(family, 16)[0]
        vector = bank.vector(direction_id)

        assert vector.shape == (bank.hidden_size,)
        assert vector.dtype is torch.float32
        assert float(vector.norm()) == pytest.approx(1.0, abs=1e-5)


def test_a_concept_direction_comes_from_its_recorded_file(bank):
    direction_id = bank.direction_ids("concept", 16, concept="Dust")[0]
    record = bank.record(direction_id)
    payload = torch.load(REPO_ROOT / record["source_path"], map_location="cpu",
                         weights_only=False)
    stored = payload["vector"].to(torch.float32).reshape(-1)

    rebuilt = bank.vector(direction_id)

    assert torch.allclose(rebuilt, stored / stored.norm(), atol=1e-6)


def test_a_direction_that_no_longer_reproduces_is_refused(bank):
    tampered = copy.deepcopy(bank.records)
    direction_id = bank.direction_ids("random", 5)[0]
    tampered[direction_id] = dict(tampered[direction_id], seed=1234567)
    broken = DirectionBank(bank.config, list(tampered.values()), bank.calibration_dir,
                           hidden_size=bank.hidden_size)

    with pytest.raises(ValueError, match="does not reproduce"):
        broken.vector(direction_id)


def test_the_estimator_selects_the_column_it_names(bank):
    direction_id = bank.direction_ids("random", 16)[0]
    record = bank.record(direction_id)
    mad_bank = DirectionBank(bank.config, list(bank.records.values()),
                             bank.calibration_dir, estimator="mad",
                             hidden_size=bank.hidden_size)

    assert bank.scale(direction_id) == pytest.approx(record["sd"])
    assert mad_bank.scale(direction_id) == pytest.approx(record["mad_corrected"])


def test_the_reference_scale_is_the_median_of_the_fixed_random_bank(bank):
    import numpy as np

    scales = [bank.scale(i) for i in bank.direction_ids("random", 16)]

    assert bank.reference_scale(16) == pytest.approx(float(np.median(scales)))


def test_every_calibrated_scale_is_usable_as_a_z_denominator(bank):
    # z = alpha / s only exists for a strictly positive s, and the sweep divides by
    # it for all 32 blocks, so a single non-positive entry would break a whole run.
    for layer in bank.layers:
        for family in ("concept", "random"):
            for direction_id in bank.direction_ids(family, layer):
                assert bank.scale(direction_id) > 0.0


def test_the_saved_random_vectors_are_the_ones_the_scales_were_estimated_on(bank):
    """save_random_vectors.py --seed_scheme linear writes the same bank to disk.

    Two ways of obtaining one direction is two ways of drifting apart, and a drift
    would silently attach a calibrated s(l, v) to a vector it was never estimated on.
    The saved files name the Experiment 0 direction they claim to be, so the claim is
    checkable, and this test checks it.
    """
    vector_dir = REPO_ROOT / bank.config.vector_dir
    saved = sorted(vector_dir.glob("random_s*_avg.pt"))
    if not saved:
        pytest.skip("the fixed-random bank has not been written to disk")

    for path in saved:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        direction_id = payload.get("experiment_0_direction_id")
        if direction_id is None:
            continue
        # The file is named by decoder block; the concept files next to it are named
        # by hidden state. Both conventions must survive the round trip.
        assert payload["hidden_state_index"] == payload["decoder_block_index"] + 1
        assert bank.record(direction_id)["decoder_block_index"] == (
            payload["decoder_block_index"])

        stored = payload["vector"].to(torch.float32).reshape(-1)
        assert torch.allclose(stored / stored.norm(), bank.vector(direction_id), atol=1e-6)


def test_an_unknown_direction_names_the_directory_it_was_looked_for_in(bank):
    with pytest.raises(KeyError, match="experiment_0_calibration"):
        bank.record("concept__block_99__Nothing")


def test_the_activation_norm_is_reported_as_absent_rather_than_invented(bank):
    # Experiment 0 records projections onto directions, not activation norms.
    with pytest.raises(NotImplementedError, match="dropout_norm_source trial"):
        bank.token_norm(16)


def test_artifacts_from_another_configuration_are_refused(tmp_path):
    directory = tmp_path / "calibration"
    directory.mkdir()
    with (DEFAULT_CALIBRATION_DIR / "directional_scales.json").open() as handle:
        records = json.load(handle)
    with (directory / "directional_scales.json").open("w") as handle:
        json.dump(records[:100], handle)
    with (DEFAULT_CALIBRATION_DIR / "run_manifest.json").open() as handle:
        manifest = json.load(handle)
    manifest["config_sha256"] = "0" * 64
    with (directory / "run_manifest.json").open("w") as handle:
        json.dump(manifest, handle)

    with pytest.raises(ValueError, match="different version"):
        DirectionBank.load(CONFIG, calibration_dir=directory)


def test_a_calibration_that_does_not_cover_the_config_is_refused(tmp_path):
    directory = tmp_path / "partial"
    directory.mkdir()
    with (DEFAULT_CALIBRATION_DIR / "directional_scales.json").open() as handle:
        records = json.load(handle)
    kept = [row for row in records if row["decoder_block_index"] < 4]
    with (directory / "directional_scales.json").open("w") as handle:
        json.dump(kept, handle)

    with pytest.raises(ValueError, match="does not cover decoder blocks"):
        DirectionBank.load(CONFIG, calibration_dir=directory)


def test_a_missing_calibration_points_at_the_command_that_produces_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_experiment_0"):
        DirectionBank.load(CONFIG, calibration_dir=tmp_path / "absent")
