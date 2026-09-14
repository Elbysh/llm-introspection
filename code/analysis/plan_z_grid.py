#!/usr/bin/env python3
"""Build the z grid of Experiment 1, and check it reaches every family's threshold.

Section 8.5 of docs/livrables/calibration-contexte-probleme-et-correctif.md states the
rule that was missing when the first z sweeps were run:

> Check that the maximum dose delivers, for EVERY family, an amplitude above that
> family's own threshold measured on the alpha grid.

A z dose delivers `alpha = z * s(l, v)`, so a grid fixed in z hands each family an
amplitude proportional to its own scale. When the concept scale was 54x the generic
one, a single z grid could not bracket both: it overshot concept by 6-14x and
undershot every control by 5-20x, and the flat control curves were that artefact and
nothing else. After recalibration in the behavioural context the ratio is 1.2-2.4x
and one grid shared by the four families becomes possible -- but only if its extent
is rebuilt, since the old `z in [0.01, 20.48]` tops out around alpha = 5.

This reads the recalibrated scales and an existing alpha-matched sweep, then proposes
a factor-two z ladder and prints, per block and family, the alpha the extreme doses
actually deliver against that family's alpha-grid threshold.

    python code/analysis/plan_z_grid.py \
        --calibration_dir results/experiment_0_calibration_2afc \
        --alpha_trials results/experiment1/full32_all/trials.csv
"""

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Experiment 0 family -> the intervention family of doc section 3 that consumes it.
# Dropout picks no direction and is matched against the median fixed-random scale
# (doc 3.6), so it shares the generic scale rather than owning one.
SCALE_FAMILY = {
    "concept": "concept",
    "fixed_random": "random",
    "renewed_noise": "noise",
    "scrambled_concept": "scrambled",
}
GENERIC_FAMILY = "fixed_random"


def load_scales(calibration_dir):
    """Median s(l, v) per (block, intervention family), plus dropout's reference."""
    path = calibration_dir / "directional_scales.csv"
    if not path.exists():
        raise SystemExit(f"no directional_scales.csv in {calibration_dir}")
    grouped = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("sd"):
                continue
            family = SCALE_FAMILY.get(row["direction_family"])
            if family is None:
                continue
            grouped[(int(row["decoder_block_index"]), family)].append(float(row["sd"]))
    scales = {key: statistics.median(values) for key, values in grouped.items()}
    for (layer, family), value in list(scales.items()):
        if family == SCALE_FAMILY[GENERIC_FAMILY]:
            scales[(layer, "dropout")] = value
    if not scales:
        raise SystemExit(f"no usable scales in {path}")
    return scales


def load_alpha_curves(trials_path):
    """Adjusted-score accuracy per (block, family, alpha dose) from an alpha sweep."""
    totals = defaultdict(lambda: [0.0, 0])
    with trials_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["kind"] != "perturbed" or row["matching"] != "alpha":
                continue
            if not row["correct_adjusted"]:
                continue
            key = (int(row["layer"]), row["family"], float(row["dose"]))
            totals[key][0] += float(row["correct_adjusted"])
            totals[key][1] += 1
    curves = defaultdict(dict)
    for (layer, family, dose), (correct, count) in totals.items():
        curves[(layer, family)][dose] = correct / count
    return curves


def threshold(curve, level=0.75):
    """Lowest tested alpha reaching `level`, or None when the sweep never did."""
    for dose in sorted(curve):
        if curve[dose] >= level:
            return dose
    return None


def ladder(low, high, factor=2.0):
    doses = []
    dose = factor ** math.floor(math.log(low, factor))
    while dose < high * factor:
        doses.append(dose)
        dose *= factor
    return doses


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration_dir", required=True)
    parser.add_argument("--alpha_trials", required=True,
                        help="trials.csv of an alpha-matched sweep of the same model; "
                             "its thresholds are what the z grid has to reach")
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--alpha_low", type=float, default=0.25,
                        help="bottom of the alpha grid the z grid must also reach, so "
                             "that every family is tested near chance as well")
    parser.add_argument("--alpha_high", type=float, default=128.0,
                        help="top of the alpha grid. A family whose alpha threshold was "
                             "never reached still needs the z grid to deliver this much, "
                             "or its flat z curve says nothing.")
    args = parser.parse_args()

    calibration_dir = Path(args.calibration_dir)
    if not calibration_dir.is_absolute():
        calibration_dir = REPO_ROOT / calibration_dir
    trials_path = Path(args.alpha_trials)
    if not trials_path.is_absolute():
        trials_path = REPO_ROOT / trials_path

    scales = load_scales(calibration_dir)
    curves = load_alpha_curves(trials_path)
    layers = args.layers or sorted({layer for layer, _ in scales})
    families = sorted({family for _, family in scales})

    print(f"{'block':>6} " + " ".join(f"{family:>10}" for family in families)
          + "    s(l, v), median per family")
    for layer in layers:
        cells = " ".join(f"{scales.get((layer, family), float('nan')):>10.4g}"
                         for family in families)
        print(f"{layer:>6} {cells}")

    # The grid has to deliver alpha_low to the family with the LARGEST scale and
    # alpha_high to the one with the SMALLEST, at every block it covers.
    selected = {key: value for key, value in scales.items() if key[0] in set(layers)}
    z_low = args.alpha_low / max(selected.values())
    z_high = args.alpha_high / min(selected.values())
    doses = ladder(z_low, z_high)
    print(f"\nScales over blocks {layers[0]}-{layers[-1]} span "
          f"{min(selected.values()):.4g} to {max(selected.values()):.4g}.")
    print(f"Covering alpha in [{args.alpha_low:g}, {args.alpha_high:g}] for every "
          f"family needs z from {z_low:.4g} to {z_high:.4g}: "
          f"{len(doses)} doses at a factor of two.")
    print("Proposed Z_DOSES:")
    print("  " + " ".join(f"{dose:g}" for dose in doses))

    print(f"\nSection 8.5 coverage check. 'alpha at z_max' is what the top dose "
          f"delivers; it must exceed the family's own alpha-grid threshold.")
    print(f"\n{'block':>6} {'family':>9} {'s':>9} {'alpha@z_min':>12} "
          f"{'alpha@z_max':>12} {'alpha75':>9} {'margin':>8}")
    failures = []
    for layer in layers:
        for family in families:
            scale = scales.get((layer, family))
            if scale is None:
                continue
            low, high = doses[0] * scale, doses[-1] * scale
            reached = threshold(curves.get((layer, family), {}))
            # A family the alpha sweep never brought to 75% still has to be handed the
            # top of the alpha grid, or its flat z curve is uninformative either way.
            target = reached if reached is not None else args.alpha_high
            margin = high / target
            if margin < 1.0 or low > args.alpha_low:
                failures.append((layer, family))
            label = f"{reached:g}" if reached is not None else f">{args.alpha_high:g}"
            print(f"{layer:>6} {family:>9} {scale:>9.4g} {low:>12.3g} "
                  f"{high:>12.3g} {label:>9} {margin:>8.2f}")
    print(f"\n{len(failures)} (block, family) cells are not covered by this grid"
          + (": " + ", ".join(f"{l}/{f}" for l, f in failures) if failures else "."))


if __name__ == "__main__":
    main()
