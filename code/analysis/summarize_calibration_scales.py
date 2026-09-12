#!/usr/bin/env python3
"""Summarize an Experiment 0 calibration and propose the Experiment 1 dose grids.

Section 5.5 of docs/livrables/cadrage-experiments.md keeps two independent grids.
The z grid is already expressed in units of s(l, v), so it transfers between models
unchanged. The alpha grid is a raw activation norm and does not: a direction's
natural scale at a given block depends on the model, so the bounds chosen for one
have no reason to bracket another's transition.

This reads directional_scales.csv and prints, per block and per family, the scale
s(l, v) the calibration estimated. It then proposes an alpha grid from those
numbers: the transition is expected where alpha is comparable to the scale, so the
grid is a factor-two ladder covering the families' scales with a margin on each
side. The proposal is a starting bracket for the pilot of doc 14.6, not a
substitute for it.

    python code/analysis/summarize_calibration_scales.py \
        --calibration_dir results/experiment_0_calibration_qwen38_27b \
        --layers 6 32 56
"""

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ESTIMATORS = {"sd": "sd", "mad": "mad_corrected"}


def load_rows(calibration_dir, estimator):
    path = calibration_dir / "directional_scales.csv"
    if not path.exists():
        raise SystemExit(f"no directional_scales.csv in {calibration_dir}")
    field = ESTIMATORS[estimator]
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            value = row.get(field)
            if not value:
                continue
            rows.append(
                {
                    "layer": int(row["decoder_block_index"]),
                    "family": row["direction_family"],
                    "concept": row["concept"],
                    "scale": float(value),
                }
            )
    if not rows:
        raise SystemExit(f"no {estimator} scales recorded in {path}")
    return rows


def ladder(low, high, factor=2.0):
    """The factor-two dose ladder of doc 5.5 covering [low, high]."""
    doses = []
    dose = factor ** math.floor(math.log(low, factor))
    while dose <= high * factor:
        # Keep the printed grid readable: powers of two are exact, and the rounding
        # below only trims the floating-point tail of e.g. 0.12500000000000003.
        doses.append(round(dose, 6))
        dose *= factor
    return doses


def trim_to_budget(doses, centre, budget):
    """Keep at most `budget` doses, the ones closest to `centre` in log space.

    Concept directions and generic ones sit decades apart, so the ladder that
    brackets both ends of a calibration is longer than a single job can afford. The
    window is kept contiguous, since a psychometric curve needs neighbouring doses.
    """
    if len(doses) <= budget:
        return doses
    # Slide a window of `budget` doses and keep the one whose geometric midpoint is
    # closest to the centre scale.
    best, best_distance = 0, None
    for start in range(len(doses) - budget + 1):
        window = doses[start : start + budget]
        midpoint = math.sqrt(window[0] * window[-1])
        distance = abs(math.log(midpoint) - math.log(centre))
        if best_distance is None or distance < best_distance:
            best, best_distance = start, distance
    return doses[best : best + budget]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration_dir", required=True)
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="decoder blocks to report; default every calibrated block")
    parser.add_argument("--estimator", choices=sorted(ESTIMATORS), default="sd")
    parser.add_argument("--margin", type=float, default=4.0,
                        help="factor of headroom below the smallest and above the "
                             "largest family median when proposing the alpha grid")
    parser.add_argument("--num_doses", type=int, default=10,
                        help="dose budget for the proposed alpha grid; the full "
                             "bracket is reported too when it does not fit")
    args = parser.parse_args()

    calibration_dir = Path(args.calibration_dir)
    if not calibration_dir.is_absolute():
        calibration_dir = REPO_ROOT / calibration_dir
    rows = load_rows(calibration_dir, args.estimator)

    layers = args.layers or sorted({row["layer"] for row in rows})
    selected = [row for row in rows if row["layer"] in set(layers)]
    if not selected:
        raise SystemExit(f"no calibrated directions at blocks {layers}")

    by_key = defaultdict(list)
    for row in selected:
        by_key[(row["layer"], row["family"])].append(row["scale"])

    print(f"Calibration: {calibration_dir}")
    print(f"Estimator:   {args.estimator}\n")
    print(f"{'block':>6} {'family':>9} {'n':>6} {'min':>12} {'median':>12} {'max':>12}")
    for layer in layers:
        for family in ("concept", "fixed_random", "renewed_noise", "random", "noise"):
            scales = by_key.get((layer, family))
            if not scales:
                continue
            print(f"{layer:>6} {family:>9} {len(scales):>6} {min(scales):>12.4g} "
                  f"{statistics.median(scales):>12.4g} {max(scales):>12.4g}")

    # A family's transition is expected where alpha is comparable to that family's
    # typical scale, so the bracket is built from the per-family medians rather than
    # from the extreme single directions, which are outliers by construction.
    medians = {key: statistics.median(scales) for key, scales in by_key.items()}
    low = min(medians.values()) / args.margin
    high = max(medians.values()) * args.margin
    full = ladder(low, high)
    centre = math.sqrt(min(medians.values()) * max(medians.values()))
    alpha_doses = trim_to_budget(full, centre, args.num_doses)

    print(f"\nFamily medians span {min(medians.values()):.4g} to "
          f"{max(medians.values()):.4g} over blocks {layers}.")
    print(f"Full bracket at margin x{args.margin:g} each side: {len(full)} doses "
          f"({full[0]:g} to {full[-1]:g}).")
    if len(full) > args.num_doses:
        print(f"That exceeds the {args.num_doses}-dose budget. The window below is "
              f"centred on the geometric mean of the family medians ({centre:.4g}); "
              f"it under-covers the families whose scales fall outside it, which is "
              f"a choice to make deliberately.")
    print(f"Proposed ALPHA_DOSES ({len(alpha_doses)} doses):")
    print("  " + " ".join(f"{dose:g}" for dose in alpha_doses))
    print("\nThe z grid needs no such rescaling: z is already in units of s(l, v).")


if __name__ == "__main__":
    main()
