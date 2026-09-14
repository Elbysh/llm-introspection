#!/usr/bin/env python3
"""Did the recalibration actually move s(l, v) into the behavioural context?

Section 8.6 of docs/livrables/calibration-contexte-probleme-et-correctif.md fixes two
acceptance criteria, both computable from `directional_scales.csv` alone:

1. `s_SD / s_MAD ~ 1` for every family. In the isolated context the sink token gave
   ratios of 12 to 2500; in the behavioural context the projection distribution is
   essentially Gaussian and the two estimators agree.
2. The tail mass, `f = |mean - median| / |extreme - median|`, times the number of
   tokens per rendered sequence, well under 1. That product estimates how many extreme
   tokens each sequence contributes, and it was ~1 in the isolated context: the
   position-0 token, exactly one per sequence. The denominator is per SEQUENCE and not
   per corpus sentence, since the 2AFC manifest presents each sentence four times and
   dividing the positions by 100 would inflate the product fourfold for a reason that
   has nothing to do with tails.

   Read it against its own floor: `f` does not vanish on a Gaussian sample, because
   `mean - median` fluctuates by about `sigma / sqrt(n)` while `p99 - median` is about
   `2.33 sigma`, so a clean sample of n positions still shows `f ~ 0.43 / sqrt(n)`.
   The report prints that floor beside the measurement.

A failure of either is a reason to look, not a verdict: both thresholds were set
against Llama, where the defect is violent (`s_SD / s_MAD` up to 2360). On a model
whose position-0 token is less extreme, a family can sit just outside them for
reasons that have nothing to do with the manifest. `--plan_dir` settles it on
evidence instead: it reads `projections.npz` beside the plan's `observations.jsonl`
and reports WHERE the extreme projections actually sit. A sink artefact puts them at
prompt position 0, at a magnitude orders above everything else; an ordinary
distribution spreads them over ordinary positions at a few standard deviations. That
is the question the criteria are proxies for, and the note's own statement of what a
failure would mean: a manifest targeting the wrong tokens, most likely through
shifted character bounds.

The concept/random scale ratio is reported alongside, because that is the quantity
the division by s is meant to remove and the one the defect inflated from 1.2-2.4 to
56-95.

    python code/analysis/check_calibration_context.py \
        --calibration_dir results/experiment_0_calibration_2afc \
        --baseline_dir results/experiment_0_calibration
"""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# The two control families whose ratio the note tracks. `concept` is the numerator.
CONCEPT_FAMILY = "concept"
GENERIC_FAMILY = "fixed_random"


def positional_diagnostic(calibration_dir, plan_dir, direction_ids):
    """Where do the extreme projections sit, and how extreme are they?

    The sink signature is unmistakable: every extreme at prompt position 0, orders of
    magnitude above the rest. Anything else -- extremes spread over ordinary positions
    at a handful of standard deviations -- is a distribution with a tail, not a token
    that should never have been in the sample.
    """
    import numpy as np

    observations = [
        json.loads(line)
        for line in (plan_dir / "observations.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    positions = [int(row["token_index"]) for row in observations]
    archive = np.load(calibration_dir / "projections.npz")

    print(f"\nWhere the extreme projections sit ({len(positions)} admissible positions, "
          f"prompt positions {min(positions)}-{max(positions)}, "
          f"{positions.count(0)} at position 0)")
    print(f"\n{'direction':>34} {'sd':>10} {'max|proj|':>11} {'ratio':>7} "
          f"{'argmax at':>10} {'top-8 positions':>28}")
    for direction_id in direction_ids:
        if direction_id not in archive:
            continue
        values = archive[direction_id]
        sd = float(values.std(ddof=1))
        magnitudes = np.abs(values)
        order = np.argsort(-magnitudes)[:8]
        peak = float(magnitudes[order[0]])
        spread = sorted({positions[index] for index in order})
        print(f"{direction_id:>34} {sd:>10.4g} {peak:>11.4g} "
              f"{peak / sd if sd else float('nan'):>7.1f} {positions[order[0]]:>10} "
              f"{str(spread):>28}")


def sequence_count(calibration_dir, rows):
    """Number of rendered sequences the calibration ran, for the tail-mass criterion.

    `n_positions / n_sentences` would count a sentence presented four times as a
    24-token sentence. The run manifest records how many sequences were actually
    rendered, which is what "extreme tokens per sequence" needs.
    """
    manifest_path = calibration_dir / "run_manifest.json"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            contexts = json.load(handle).get("n_contexts")
        if contexts:
            return int(contexts)
    # Older results predate the manifest; one sequence per sentence is what the
    # isolated-sentence context did.
    return rows[0]["n_sentences"]


def load(calibration_dir):
    path = calibration_dir / "directional_scales.csv"
    if not path.exists():
        raise SystemExit(f"no directional_scales.csv in {calibration_dir}")
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("sd") or not row.get("mad_corrected"):
                continue
            rows.append(
                {
                    "layer": int(row["decoder_block_index"]),
                    "family": row["direction_family"],
                    "sd": float(row["sd"]),
                    "mad": float(row["mad_corrected"]),
                    "mean": float(row["mean"]),
                    "median": float(row["median"]),
                    "p01": float(row["p01"]),
                    "p99": float(row["p99"]),
                    "n_sentences": int(row["n_sentences"]),
                    "n_positions": int(row["n_positions"]),
                    "context": row.get("presentation_context", ""),
                    "policy": row.get("position_policy", ""),
                }
            )
    if not rows:
        raise SystemExit(f"no usable scales in {path}")
    return rows


def tail_mass(row):
    """|mean - median| / |extreme - median|: the weight the mean gives the tail.

    The extreme is taken as the further of the two recorded 1% quantiles, which is
    the most extreme value the persisted summary exposes. Under a symmetric
    distribution the numerator vanishes; a single huge token drags the mean and not
    the median, and the ratio approaches that token's share of the sample.
    """
    spread = max(abs(row["p99"] - row["median"]), abs(row["p01"] - row["median"]))
    if spread <= 0.0:
        return float("nan")
    return abs(row["mean"] - row["median"]) / spread


def report(name, rows, sequences, layers):
    by_key = defaultdict(list)
    for row in rows:
        by_key[(row["layer"], row["family"])].append(row)
    families = sorted({key[1] for key in by_key})
    reference = rows[0]
    tokens_per_sequence = reference["n_positions"] / sequences
    # |mean - median| ~ sigma / sqrt(n) and |p99 - median| ~ 2.33 sigma on a Gaussian
    # sample, so f floors at about 0.43 / sqrt(n) even with no tail at all.
    floor = 0.43 / math.sqrt(reference["n_positions"])

    print(f"\n=== {name} ===")
    print(f"context {reference['context']!r}, policy {reference['policy']!r}, "
          f"{reference['n_positions']} positions over {sequences} sequences and "
          f"{reference['n_sentences']} sentences ({tokens_per_sequence:.2f} "
          f"tokens/sequence)")
    print(f"Gaussian floor of the tail mass at n = {reference['n_positions']}: "
          f"{floor:.4f}")
    print(f"\n{'block':>6} {'family':>16} {'s_SD':>11} {'s_SD/s_MAD':>11} "
          f"{'tail mass':>10} {'x tokens/seq':>13}")
    worst_ratio = 0.0
    worst_product = 0.0
    for layer in layers:
        for family in families:
            group = by_key.get((layer, family))
            if not group:
                continue
            sd = statistics.median(row["sd"] for row in group)
            ratio = statistics.median(row["sd"] / row["mad"] for row in group)
            mass = statistics.median(tail_mass(row) for row in group)
            product = mass * tokens_per_sequence
            worst_ratio = max(worst_ratio, abs(ratio - 1.0))
            worst_product = max(worst_product, product)
            print(f"{layer:>6} {family:>16} {sd:>11.4g} {ratio:>11.3g} "
                  f"{mass:>10.3g} {product:>13.3g}")

    print(f"\n{'block':>6} {'concept/random, SD':>20} {'concept/random, MAD':>21}")
    for layer in layers:
        concepts = by_key.get((layer, CONCEPT_FAMILY))
        generics = by_key.get((layer, GENERIC_FAMILY))
        if not concepts or not generics:
            continue
        sd_ratio = (statistics.median(row["sd"] for row in concepts)
                    / statistics.median(row["sd"] for row in generics))
        mad_ratio = (statistics.median(row["mad"] for row in concepts)
                     / statistics.median(row["mad"] for row in generics))
        print(f"{layer:>6} {sd_ratio:>20.3g} {mad_ratio:>21.3g}")

    criterion_1 = worst_ratio <= 0.25
    criterion_2 = worst_product < 1.0
    print(f"\ncriterion 1, |s_SD/s_MAD - 1| <= 0.25 for every family: "
          f"{'PASS' if criterion_1 else 'FAIL'} (worst deviation {worst_ratio:.3g})")
    print(f"criterion 2, tail mass x tokens/sequence < 1:            "
          f"{'PASS' if criterion_2 else 'FAIL'} (worst {worst_product:.3g}, "
          f"floor {floor * tokens_per_sequence:.3g})")
    return criterion_1 and criterion_2


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration_dir", required=True,
                        help="the recalibrated directory, which must pass both criteria")
    parser.add_argument("--baseline_dir", default=None,
                        help="an earlier calibration to print beside it, typically the "
                             "isolated-sentence one the fix replaces. It is reported "
                             "for contrast and is not expected to pass.")
    parser.add_argument("--layers", type=int, nargs="*", default=[1, 5, 9, 13],
                        help="decoder blocks to report; the note's table uses 1 5 9 13")
    parser.add_argument("--plan_dir", default=None,
                        help="the recalibration's paths.plan_dir. Adds the positional "
                             "diagnostic, which decides on evidence what the two "
                             "thresholds only approximate.")
    parser.add_argument("--diagnostic_directions", nargs="*", default=None,
                        help="direction IDs to show it for; default is one concept and "
                             "one fixed-random direction at the first reported block.")
    args = parser.parse_args()

    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else REPO_ROOT / path

    if args.baseline_dir:
        directory = resolve(args.baseline_dir)
        rows = load(directory)
        report("baseline: " + args.baseline_dir, rows,
               sequence_count(directory, rows), args.layers)
    directory = resolve(args.calibration_dir)
    rows = load(directory)
    passed = report("recalibrated: " + args.calibration_dir, rows,
                    sequence_count(directory, rows), args.layers)
    if args.plan_dir:
        block = args.layers[0]
        ids = args.diagnostic_directions or [
            "concept__block_{:02d}__recursion".format(block),
            "concept__block_{:02d}__Dust".format(block),
            "fixed_random__block_{:02d}__0000".format(block),
        ]
        positional_diagnostic(directory, resolve(args.plan_dir), ids)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
