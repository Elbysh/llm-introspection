"""The published CSV must recover the JSON direction_bank.py reads."""

from pathlib import Path

import pytest

from experiment_0_calibration.rebuild_scales_json import parse_cell, rebuild

REPO_ROOT = Path(__file__).resolve().parents[2]
# The two calibrations that publish both files, so the round trip is checkable.
PUBLISHED_BOTH = [
    "results/experiment_0_calibration",
    "results/experiment_0_calibration_qwen38_27b",
]


def test_a_sha256_of_digits_stays_a_string():
    # json.loads would turn it into an integer and lose its leading zeros.
    digits = "0" * 8 + "1" * 56
    assert parse_cell("source_sha256", digits) == digits
    assert parse_cell("concept", "12345") == "12345"


def test_types_come_back_from_a_typeless_format():
    assert parse_cell("decoder_block_index", "13") == 13
    assert parse_cell("sd", "0.2511") == pytest.approx(0.2511)
    assert parse_cell("valid_for_sd_normalization", "True") is True
    assert parse_cell("source_provenance_complete", "False") is False
    assert parse_cell("seed", "") is None


@pytest.mark.parametrize("name", PUBLISHED_BOTH)
def test_the_csv_recovers_every_value_the_json_records(name):
    """Value-exact on every key the JSON has, across the whole calibration.

    The CSV header is the union of all families' fields, so a rebuilt record also
    carries the keys a non-concept record omitted. They come back null, which is what
    `dict.get` already returned for them, so nothing downstream can tell the
    difference -- but they are the reason this is not a plain equality check.
    """
    json = pytest.importorskip("json")
    directory = REPO_ROOT / name
    scales = directory / "directional_scales.json"
    if not scales.exists():
        pytest.skip("{} is not published in this working tree".format(scales))

    with scales.open(encoding="utf-8") as handle:
        original = json.load(handle)
    rebuilt = rebuild(directory)

    assert len(rebuilt) == len(original)
    for restored, expected in zip(rebuilt, original):
        for field, value in expected.items():
            assert restored[field] == value, field
        assert all(restored[field] is None for field in set(restored) - set(expected))
