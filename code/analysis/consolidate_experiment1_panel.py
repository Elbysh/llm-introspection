#!/usr/bin/env python3
"""Group the Experiment 1 runs of one model and one dose rule into a single run directory.

A model's panel is spread over several runs. Llama's ten concepts and ten fixed random
directions arrived in three pieces -- the original sweep plus the two panel-completion
jobs -- and each piece carries both dose arms, only one of which may be interpretable.
Reading the panel therefore meant repeating three --run_dir flags and remembering which
arm to keep, every time. This writes the pooled panel once, as an ordinary run directory.

Why not merge_experiment1_runs.py. That script pools runs that split ONE sweep by layer
group, and refuses anything else -- it rejects these with "runs disagree on 'families'"
because it keys per-curve records by (layer, family, matching) and disjoint layer groups
are what make those keys safe. Here the runs share every layer and differ by direction,
so the records collide by construction and the summary has to be recomputed rather than
concatenated. That is what this does: it re-runs experiment1_psychometrics.summarize over
the pooled trials, so summary.json carries real per-dose records and psychometric curves
fitted on the whole panel, not a stitched copy of its parts'.

--matching is required and never defaulted. The alpha arm of a pre-recalibration sweep is
valid, because under matching == "alpha" the amplitude is the grid value and
calibration.scale() is never consulted; its z arm inherited the position-0 defect of doc
8.5 and is not. Pooling a run whole would blend the two silently.

    python code/analysis/consolidate_experiment1_panel.py \
        --run_dir results/experiment1/z2afc_all \
        --run_dir results/experiment1/panel-concepts \
        --run_dir results/experiment1/panel-random \
        --matching z --out results/experiment1/llama_z_panel

Sham rows are kept once per source run: they carry no dose and serve both arms, and the
scores of each run were adjusted against its own sham.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

from experiments.experiment1_psychometrics import summarize  # noqa: E402

# summarize() reads these as numbers and booleans, not strings.
FLOAT_FIELDS = ("dose", "alpha_requested", "dropout_rate", "logit_a", "logit_b",
                "contrast", "sham_contrast", "adjusted_contrast", "correct_raw",
                "correct_adjusted", "realized_amplitude", "clean_token_norm")
BOOL_FIELDS = ("top_token_is_choice", "finite", "tie_adjusted")


def parse(row):
    """One CSV row with the types summarize expects.

    `correct_raw` is 1.0/0.0/0.5 in the current engine and a bare boolean in sweeps that
    predate the tie encoding, so both spellings are accepted.
    """
    out = dict(row)
    out["layer"] = int(row["layer"]) if row["layer"] else None
    for field in FLOAT_FIELDS:
        value = row.get(field)
        if value in (None, ""):
            out[field] = None
        elif value in ("True", "False"):
            out[field] = 1.0 if value == "True" else 0.0
        else:
            out[field] = float(value)
    for field in BOOL_FIELDS:
        out[field] = row.get(field) == "True"
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", action="append", required=True, type=Path)
    parser.add_argument("--matching", required=True, choices=("alpha", "z"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--exclude_layers", nargs="+", type=int, default=[])
    parser.add_argument("--no_trials", action="store_true",
                        help="write only summary.json, when the pooled trials.csv would "
                             "merely duplicate rows already versioned in the sources")
    args = parser.parse_args()

    run_dirs = [p if p.is_absolute() else REPO_ROOT / p for p in args.run_dir]
    excluded = set(args.exclude_layers)
    rows, fieldnames, sources = [], None, []
    for run_dir in run_dirs:
        path = run_dir / "trials.csv"
        if not path.exists():
            raise SystemExit(f"no trials.csv in {run_dir}")
        kept = 0
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = fieldnames or reader.fieldnames
            for raw in reader:
                if raw["kind"] == "perturbed":
                    if raw["matching"] != args.matching:
                        continue
                    if raw["layer"] and int(raw["layer"]) in excluded:
                        continue
                rows.append(parse(raw))
                kept += 1
        sources.append({"run": str(run_dir.relative_to(REPO_ROOT)), "rows": kept})
        print(f"  {run_dir.name}: {kept:,} rows", flush=True)

    meta = {}
    first = run_dirs[0] / "summary.json"
    if first.exists():
        meta = json.loads(first.read_text(encoding="utf-8"))

    per_dose, curves, sham = summarize(rows)
    out_dir = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    perturbed = [r for r in rows if r["kind"] == "perturbed"]
    concepts = sorted({r["direction_id"].rsplit("__", 1)[-1] for r in perturbed
                       if r["family"] == "concept"})
    randoms = sorted({r["direction_id"].rsplit("__", 1)[-1] for r in perturbed
                      if r["family"] == "random"})
    layers = sorted({r["layer"] for r in perturbed})

    summary = {
        "model": meta.get("model"),
        "matching": args.matching,
        "matchings": [args.matching],
        "estimator": meta.get("estimator"),
        "seed": meta.get("seed"),
        "pairs": meta.get("pairs"),
        "label_orders": meta.get("label_orders"),
        "dropout_norm_source": meta.get("dropout_norm_source"),
        "layers": layers,
        "excluded_layers": sorted(excluded),
        "families": sorted({r["family"] for r in perturbed}),
        "concepts": concepts,
        "random_directions": randoms,
        "dose_grid": {args.matching: sorted({r["dose"] for r in perturbed})},
        "n_trials": len(rows),
        "n_perturbed": len(perturbed),
        "sources": sources,
        "sham": sham,
        "per_dose": per_dose,
        "curves": curves,
        "notes": [
            f"Pooled {args.matching}-matched panel: {len(concepts)} concept and "
            f"{len(randoms)} fixed random directions over {len(layers)} blocks. "
            "per_dose and curves are recomputed over the pooled trials, not concatenated "
            "from the sources, since those records collide across runs sharing layers.",
        ],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, sort_keys=True)

    if not args.no_trials:
        with (out_dir / "trials.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    print(f"{len(concepts)} concepts x {len(randoms)} random, {len(layers)} blocks, "
          f"{len(rows):,} rows -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
