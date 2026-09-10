#!/usr/bin/env python3
"""Analyze saved matched A/B conditions without loading the model."""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from position_detection_utils import SCHEMA_VERSION, summarize


def analyze(input_dir):
    manifest = json.loads((input_dir / "manifest.json").read_text())
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported results schema; legacy YES/NO results are not compatible.")
    records = []
    for path in sorted((input_dir / "conditions").glob("*/*.json")):
        row = json.loads(path.read_text())
        if row["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema in {path}")
        if not (input_dir / row["activation_file"]).exists():
            raise ValueError(f"Missing activation artifact for {path}")
        records.append(row)
    config = manifest["config"]
    expected = len(manifest["prompts"]) * len(config["concepts"]) * len(config["layers"]) * len(config["alphas"])
    controls = [json.loads(p.read_text()) for p in sorted((input_dir / "controls").glob("*.json"))]
    result = {
        "schema_version": SCHEMA_VERSION, "complete": len(records) == expected,
        "completed_conditions": len(records), "expected_conditions": expected,
        "summary": summarize(records),
        "controls": {
            "n": len(controls),
            "A_preference_rate": sum(r["L"] > 0 for r in controls) / len(controls) if controls else None,
            "tie_rate": sum(r["L"] == 0 for r in controls) / len(controls) if controls else None,
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.input_dir)
    args.input_dir.joinpath("summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    scalar_rows = [{k: v for k, v in r.items() if k != "diagnostics"} for r in result["summary"]]
    if scalar_rows:
        with args.input_dir.joinpath("summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(scalar_rows[0]))
            writer.writeheader()
            writer.writerows(scalar_rows)
        diagnostic_rows = []
        for row in result["summary"]:
            condition = {key: row[key] for key in ("concept", "layer", "alpha", "content_order", "label_mapping")}
            condition["injection_layer"] = condition.pop("layer")
            diagnostic_rows.extend({**condition, **d} for d in row["diagnostics"])
        if diagnostic_rows:
            with args.input_dir.joinpath("diagnostics.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(diagnostic_rows[0]))
                writer.writeheader()
                writer.writerows(diagnostic_rows)
    status = "complete" if result["complete"] else "PARTIAL"
    print(f"{status}: {result['completed_conditions']}/{result['expected_conditions']} conditions; "
          f"{len(result['summary'])} summary groups. Results in {args.input_dir}")


if __name__ == "__main__":
    main()
