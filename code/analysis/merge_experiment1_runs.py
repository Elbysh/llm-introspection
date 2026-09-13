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
# dose_grid is checked separately, per matching and only among the runs contributing
# that matching, so a sweep whose two grids were extended in different jobs still
# merges into one depth map.
MUST_AGREE = ("model", "calibration_dir", "families",
              "concepts", "label_orders", "estimator", "seed")

MATCHINGS = ("alpha", "z")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", action="append", required=True,
                        help="run directory, optionally suffixed '=alpha' or '=z' to "
                             "take only that matching from it")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run_dirs, taken = [], []
    for spec in args.run_dir:
        path, _, matching = spec.partition("=")
        if matching and matching not in MATCHINGS:
            raise SystemExit(f"unknown matching {matching!r} in {spec!r}")
        directory = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
        run_dirs.append(directory)
        taken.append((matching,) if matching else MATCHINGS)
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

    # Each matching's grid need only agree among the runs that supply that matching.
    for matching in MATCHINGS:
        grids = {json.dumps(summary["dose_grid"][matching])
                 for summary, want in zip(summaries, taken) if matching in want}
        if len(grids) > 1:
            raise SystemExit(
                f"runs supplying the {matching} matching disagree on its dose grid; "
                "refusing to merge")

    # A block may be supplied once per matching, by different runs.
    seen, layers, provenance = {}, set(), []
    for run_dir, summary, want in zip(run_dirs, summaries, taken):
        blocks = sorted(summary["layers"])
        for matching in want:
            clash = sorted(b for b in blocks if (b, matching) in seen)
            if clash:
                raise SystemExit(
                    f"blocks {clash} are supplied for the {matching} matching by both "
                    f"{seen[(clash[0], matching)]} and {run_dir.name}")
            for block in blocks:
                seen[(block, matching)] = run_dir.name
        layers.update(blocks)
        provenance.append(
            "blocks {} from {}{}".format(
                blocks, run_dir.name,
                "" if len(want) == len(MATCHINGS) else f" ({want[0]} only)"))
    layers = sorted(layers)

    # A block carried by only one matching would render as a blank row in the other.
    ragged = sorted({b for b in layers
                     if any((b, m) not in seen for m in MATCHINGS)})
    if ragged:
        print(f"note: blocks {ragged} are present for only one matching")

    merged = dict(first)
    merged["layers"] = layers
    merged["matchings"] = list(MATCHINGS)
    merged["dose_grid"] = {
        matching: next(summary["dose_grid"][matching]
                       for summary, want in zip(summaries, taken) if matching in want)
        for matching in MATCHINGS
    }
    for field in ("per_dose", "curves"):
        merged[field] = [row for summary, want in zip(summaries, taken)
                         for row in summary[field] if row["matching"] in want]
    merged["contamination"] = [row for summary, want in zip(summaries, taken)
                               for row in summary.get("contamination", [])
                               if row.get("matching") in want]
    merged["sham"] = {"per_run": [summary["sham"] for summary in summaries]}
    merged["notes"] = "Pooled by merge_experiment1_runs.py: " + "; ".join(provenance)

    out_dir.mkdir(parents=True, exist_ok=True)

    rows_written, header, sham_written = 0, None, False
    with (out_dir / "trials.csv").open("w", encoding="utf-8", newline="") as out:
        writer = None
        for run_dir, want in zip(run_dirs, taken):
            with (run_dir / "trials.csv").open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if writer is None:
                    header = reader.fieldnames
                    writer = csv.DictWriter(out, fieldnames=header)
                    writer.writeheader()
                elif reader.fieldnames != header:
                    raise SystemExit(f"{run_dir} has different trials.csv columns")
                for row in reader:
                    if row["kind"] == "sham":
                        # Shams carry no layer or matching and are one clean pass per
                        # presentation, identical across runs of the same model and
                        # prompts. Keeping every run's copies would multiply them.
                        if sham_written:
                            continue
                    elif row["matching"] not in want:
                        continue
                    writer.writerow(row)
                    rows_written += 1
            sham_written = True
    merged["n_trials"] = rows_written

    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"Merged {len(run_dirs)} runs into {out_dir}")
    print(f"  blocks:  {merged['layers']}")
    print(f"  trials:  {rows_written} rows")
    for matching in MATCHINGS:
        grid = merged["dose_grid"][matching]
        print(f"  {matching:>6}:  {len(grid)} doses, {grid[0]:g} to {grid[-1]:g}")
    print(f"  note:    {merged['notes']}")


if __name__ == "__main__":
    main()
