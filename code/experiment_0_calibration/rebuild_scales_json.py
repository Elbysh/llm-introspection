"""Rebuild directional_scales.json from the directional_scales.csv beside it.

`direction_bank.py` reads the JSON, so publishing a calibration means publishing it --
that is the convention 56df1f2 states, and the reason Experiment 1 is reproducible from
a clone. The behavioural-context calibrations broke it by size, not by choice: their
JSON is 117 MB for Llama and 233 MB for Qwen, past the 100 MiB file GitHub refuses
outright. Their CSV is tracked instead, at 50 MB and 98 MB.

The CSV is lossless: step_05 writes it from the same records, with the same fields and
no rounding, so the JSON is recoverable exactly. Only the types are gone, CSV having
none, and they come back by parsing each cell as JSON and falling back to the string.
The test in tests/experiment_0_calibration/ checks that round trip against a
calibration that publishes both.

    python -m experiment_0_calibration.rebuild_scales_json \
        --calibration_dir results/experiment_0_calibration_2afc
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from .protocol_config import repo_path

# Fields that are free text and must never be reinterpreted as numbers. A sha256 of
# nothing but digits is possible, and would silently become an integer.
TEXT_FIELDS = frozenset({
    "direction_id", "direction_family", "concept", "concept_dataset", "concept_split",
    "concept_vector_type", "source_path", "source_sha256", "source_model_name",
    "source_model_revision", "source_dataset", "target_observation_id",
    "experiment_id", "activation_site", "presentation_context", "position_policy",
    "point_weighting", "bootstrap_unit",
})


def parse_cell(field, value):
    """Recover the value's original type. Empty means null, as csv writes None."""
    if field in TEXT_FIELDS:
        return value if value != "" else None
    if value == "":
        return None
    # csv writes Python's repr of a bool, which JSON does not recognise.
    if value in ("True", "False"):
        return value == "True"
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def rebuild(calibration_dir):
    source = calibration_dir / "directional_scales.csv"
    if not source.exists():
        raise SystemExit("no directional_scales.csv in {}".format(calibration_dir))
    with source.open(encoding="utf-8", newline="") as handle:
        return [
            {field: parse_cell(field, value) for field, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration_dir", required=True)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing directional_scales.json")
    args = parser.parse_args()

    calibration_dir = repo_path(args.calibration_dir)
    records = rebuild(calibration_dir)
    target = calibration_dir / "directional_scales.json"
    if target.exists() and not args.force:
        raise SystemExit("{} already exists; pass --force to replace it".format(target))
    with target.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False, sort_keys=True)
    print("wrote {} with {} records".format(target, len(records)), file=sys.stderr)


if __name__ == "__main__":
    main()
