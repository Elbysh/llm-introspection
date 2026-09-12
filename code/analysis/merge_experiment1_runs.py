#!/usr/bin/env python3
"""Pool Experiment 1 runs that split one sweep across jobs by layer group.

experiment1_psychometrics.py writes trials.csv only after its final trial, so a
long sweep is split by layer group to bound what a crash costs. The pieces are one
experiment and the depth-resolved figures need them together. This concatenates the
trial rows and the per-dose and per-curve records, which are keyed by
(layer, family, matching) and so cannot collide between disjoint layer groups.

Nothing is recomputed: the merged summary carries each source's own records, plus a
`notes` line recording which run each block came from. Refuses to merge runs that
disagree on the model, the calibration or the dose grids, since pooling those would
be meaningless.

    python code/analysis/merge_experiment1_runs.py \
        --run_dir results/experiment1/qwen38-shallow \
        --run_dir results/experiment1/qwen38-deep \
        --out results/experiment1/qwen38_all
"""

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Pooling runs that disagree on any of these would produce a meaningless summary.
MUST_AGREE = ("model", "calibration_dir", "dose_grid", "families", "matchings",
              "concepts", "label_orders", "estimator", "seed")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run_dirs = [Path(d) if Path(d).is_absolute() else REPO_ROOT / d for d in args.run_dir]
    out_dir = Path(args.out) if Path(args.out).is_absolute() else REPO_ROOT / args.out

    summaries = []
    for run_dir in run_dirs:
        path = run_dir / "summary.json"
        if not path.exists():
            raise SystemExit(f"no summary.json in {run_dir}")
        with path.open(encoding="utf-8") as handle:
            summaries.append(json.load(handle))

    first = summaries[0]
    for field in MUST_AGREE:
        values = [json.dumps(s.get(field), sort_keys=True) for s in summaries]
        if len(set(values)) > 1:
            raise SystemExit(f"runs disagree on {field!r}; refusing to merge")

    layers, provenance = [], []
    for run_dir, summary in zip(run_dirs, summaries):
        blocks = sorted(summary["layers"])
        overlap = sorted(set(blocks) & set(layers))
        if overlap:
            raise SystemExit(f"blocks {overlap} appear in more than one run")
        layers.extend(blocks)
        provenance.append(f"blocks {blocks} from {run_dir.name}")

    merged = dict(first)
    merged["layers"] = sorted(layers)
    for field in ("per_dose", "curves"):
        merged[field] = [row for summary in summaries for row in summary[field]]
    merged["contamination"] = [row for summary in summaries
                               for row in summary.get("contamination", [])]
    merged["n_trials"] = sum(summary["n_trials"] for summary in summaries)
    merged["sham"] = {"per_run": [summary["sham"] for summary in summaries]}
    merged["notes"] = "Pooled by merge_experiment1_runs.py: " + "; ".join(provenance)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2, ensure_ascii=False, sort_keys=True)

    rows_written, header = 0, None
    with (out_dir / "trials.csv").open("w", encoding="utf-8", newline="") as out:
        writer = None
        for run_dir in run_dirs:
            with (run_dir / "trials.csv").open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if writer is None:
                    header = reader.fieldnames
                    writer = csv.DictWriter(out, fieldnames=header)
                    writer.writeheader()
                elif reader.fieldnames != header:
                    raise SystemExit(f"{run_dir} has different trials.csv columns")
                for row in reader:
                    writer.writerow(row)
                    rows_written += 1

    print(f"Merged {len(run_dirs)} runs into {out_dir}")
    print(f"  blocks:  {merged['layers']}")
    print(f"  trials:  {rows_written} rows, summary reports {merged['n_trials']}")
    print(f"  note:    {merged['notes']}")


if __name__ == "__main__":
    main()
