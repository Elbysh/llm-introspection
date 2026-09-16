#!/usr/bin/env python3
"""Report an Experiment 1 sweep on the paired localization contrast S.

Section 1 of docs/livrables/experiment1-full32-findings.md establishes which metric
an Experiment 1 run can be read on. Raw 2AFC accuracy is pinned near chance by the
model's standing answer bias. `accuracy_adjusted` subtracts the sham contrast, but
that does not remove the bias: a perturbation of any kind tends to pull the logit
gap away from the default answer, which scores as a hit on every trial targeting the
other letter and a miss on the rest, and averages to something that looks like
localization. The estimator that survives is the paired one of doc 5.8,

    S = (contrast when A is targeted - contrast when B is targeted) / 2

over the two trials sharing everything but which sentence was hit. Any shift that
does not depend on the target cancels exactly. `summary.json` stores its mean per
curve as `mean_localization_contrast`; this script recomputes it from `trials.csv`
so it can also be split by label order, by target letter and by dose, and tested
against a permutation null.

The pairing key is the one experiment1_psychometrics.py:summarize uses, so the
per-curve means here reproduce that field exactly.

    python code/analysis/experiment1_localization_report.py \
        --run_dir results/experiment1/qwen38-shallow \
        --run_dir results/experiment1/qwen38-deep

Several --run_dir are pooled, which is how a sweep split across jobs by layer group
is read back as one run.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# Everything but the targeted sentence, as in experiment1_psychometrics.py:summarize.
PAIR_KEY = ("layer", "family", "matching", "dose", "direction_id",
            "pair_id", "order", "label_order")


def load_trials(run_dirs, matching=None):
    """Perturbed and sham rows from every run directory, with numbers parsed.

    `matching` keeps only that arm's perturbed rows. It matters when a panel is
    assembled from runs of different vintage: the alpha arm of a pre-recalibration
    sweep is valid, because under matching == "alpha" the amplitude is the grid value
    and no calibrated scale is consulted, while its z arm is not. Pooling such a run
    whole would blend its unusable z rows into the z figures without any sign of it.
    Sham rows are kept either way -- they carry no dose and serve both arms.
    """
    perturbed, shams, models = [], [], set()
    for run_dir in run_dirs:
        path = run_dir / "trials.csv"
        if not path.exists():
            raise SystemExit(f"no trials.csv in {run_dir}")
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            with summary_path.open(encoding="utf-8") as handle:
                models.add(json.load(handle).get("model"))
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                # Sham rows carry no layer, dose or target: they are one clean pass
                # per presentation, shared by every condition.
                row["layer"] = int(row["layer"]) if row["layer"] else None
                row["contrast"] = float(row["contrast"])
                row["dose"] = float(row["dose"]) if row["dose"] else 0.0
                # The two scores are 1.0, 0.0 or 0.5, the half marking a tie; the tie
                # flag is a boolean. Both are blank on sham rows, which are not
                # scored against a target.
                for field in ("correct_raw", "correct_adjusted"):
                    row[field] = float(row[field]) if row.get(field) else None
                row["tie_adjusted"] = row.get("tie_adjusted") == "True"
                if row["kind"] == "sham":
                    shams.append(row)
                elif matching is None or row["matching"] == matching:
                    perturbed.append(row)
    if len(models) > 1:
        raise SystemExit(f"run directories disagree on the model: {sorted(models)}")
    return perturbed, shams, (models.pop() if models else None)


def paired_contrasts(rows, extra_key=()):
    """S for every complete pair, grouped by (layer, family, matching) + extra_key."""
    paired = defaultdict(lambda: {"n": 0, "sum": 0.0})
    for row in rows:
        key = tuple(row[field] for field in PAIR_KEY) + tuple(row[f] for f in extra_key)
        entry = paired[key]
        entry["n"] += 1
        # A pair contributes +contrast from its A-targeted half and -contrast from its
        # B-targeted half, so a target-independent shift cancels in the sum.
        entry["sum"] += row["contrast"] if row["target_label"] == "A" else -row["contrast"]
    grouped = defaultdict(list)
    for key, entry in paired.items():
        if entry["n"] == 2:            # an incomplete pair cannot cancel anything
            grouped[key[:3] + key[len(PAIR_KEY):]].append(entry["sum"] / 2.0)
    return grouped


def describe(values):
    """Mean S, its t against zero, and how the pairs sit around zero.

    Logits are read in bfloat16, so a contrast is quantized to steps of about 0.125
    and a pair whose two halves land on the same step gives S exactly 0. Where the
    perturbation does little, most pairs are such ties, and `share_positive` then
    reads far below 0.5 while saying nothing about direction. `share_zero` exposes
    that, and `share_positive_of_moved` is the sign test restricted to the pairs
    that actually moved, which is the readable one in that regime.
    """
    array = np.asarray(values, dtype=float)
    n = len(array)
    mean = float(array.mean())
    sem = float(array.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    moved = array[array != 0.0]
    return {
        "n": n,
        "mean": mean,
        "t": mean / sem if sem and sem == sem and sem > 0 else float("nan"),
        "share_positive": float((array > 0).mean()),
        "share_zero": float((array == 0.0).mean()),
        "share_positive_of_moved": float((moved > 0).mean()) if len(moved) else float("nan"),
    }


def permutation_null(rows, draws, seed):
    """Shuffle which half of each pair counts as A-targeted, keeping every logit fixed.

    If the effect came from the answer bias rather than from the pairing between
    perturbation and target, it would survive this shuffle.
    """
    by_pair = defaultdict(list)
    for row in rows:
        by_pair[tuple(row[field] for field in PAIR_KEY)].append(row)
    pairs = [[row["contrast"] for row in group] for group in by_pair.values()
             if len(group) == 2]
    if not pairs:
        return None
    halves = np.asarray(pairs, dtype=float)
    differences = (halves[:, 0] - halves[:, 1]) / 2.0
    rng = np.random.default_rng(seed)
    means = [float((differences * rng.choice([-1.0, 1.0], size=len(differences))).mean())
             for _ in range(draws)]
    return {"mean": float(np.mean(means)), "sd": float(np.std(means)),
            "draws": draws, "max_abs": float(np.max(np.abs(means)))}


def print_table(title, header, rows):
    print(f"\n{title}")
    print("  " + "  ".join(header))
    for row in rows:
        print("  " + "  ".join(row))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", action="append", required=True,
                        help="Experiment 1 output directory; repeat to pool a split sweep")
    parser.add_argument("--matching", choices=("alpha", "z"), default=None,
                        help="keep only this dose arm; required when pooling runs whose "
                             "other arm is not interpretable")
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="restrict to these decoder blocks")
    parser.add_argument("--permutation_draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--by_dose", action="store_true",
                        help="print S against dose, the psychometric curve on the "
                             "paired estimator, beside accuracy_adjusted")
    parser.add_argument("--json_out", default=None,
                        help="also write the tables to this path as JSON")
    args = parser.parse_args()

    run_dirs = [Path(d) if Path(d).is_absolute() else REPO_ROOT / d for d in args.run_dir]
    perturbed, shams, model = load_trials(run_dirs, args.matching)
    if args.layers:
        keep = set(args.layers)
        perturbed = [row for row in perturbed if row["layer"] in keep]
        # Sham rows have no layer of their own, so they are never filtered out.

    print("=" * 78)
    print("EXPERIMENT 1: PAIRED LOCALIZATION CONTRAST")
    print("=" * 78)
    print(f"Model:  {model}")
    print(f"Runs:   {', '.join(str(d.relative_to(REPO_ROOT)) for d in run_dirs)}")
    print(f"Trials: {len(perturbed)} perturbed, {len(shams)} sham")

    report = {"model": model, "runs": [str(d) for d in run_dirs],
              "n_perturbed": len(perturbed), "n_sham": len(shams)}

    # The answer bias, which is what makes the naive metrics unreadable.
    if shams:
        contrasts = np.asarray([row["contrast"] for row in shams], dtype=float)
        report["sham"] = {"mean_contrast": float(contrasts.mean()),
                          "share_prefers_a": float((contrasts > 0).mean()),
                          "min": float(contrasts.min()), "max": float(contrasts.max())}
        print(f"\nSham A/B margin: mean {contrasts.mean():+.3f} logits "
              f"(range {contrasts.min():+.3f} to {contrasts.max():+.3f}), "
              f"prefers A in {(contrasts > 0).mean():.1%} of {len(contrasts)} conditions")

    for row in perturbed:
        row["logit_a"] = float(row["logit_a"])
        row["logit_b"] = float(row["logit_b"])
    raw = np.mean([row["correct_raw"] for row in perturbed])
    adjusted = np.mean([row["correct_adjusted"] for row in perturbed])
    chose_a = np.mean([row["logit_a"] > row["logit_b"] for row in perturbed])
    report["accuracy"] = {"raw": float(raw), "adjusted": float(adjusted),
                          "share_prefers_a": float(chose_a)}
    print(f"Raw accuracy {raw:.4f}; adjusted accuracy {adjusted:.4f}; "
          f"logit A > logit B in {chose_a:.1%} of perturbed trials")

    # S by layer and family, pooled over doses and both matchings.
    by_layer_family = paired_contrasts(perturbed, extra_key=())
    pooled = defaultdict(list)
    for (layer, family, _matching), values in by_layer_family.items():
        pooled[(layer, family)].extend(values)
    rows = []
    for layer, family in sorted(pooled):
        stat = describe(pooled[(layer, family)])
        rows.append([f"{layer:>5}", f"{family:>9}", f"{stat['n']:>7}",
                     f"{stat['mean']:>+9.4f}", f"{stat['t']:>8.1f}",
                     f"{stat['share_zero']:>8.3f}",
                     f"{stat['share_positive_of_moved']:>9.3f}"])
    print_table("Paired localization contrast S, by block and family "
                "(pooled over doses and matchings):",
                [f"{'block':>5}", f"{'family':>9}", f"{'pairs':>7}",
                 f"{'mean S':>9}", f"{'t':>8}", f"{'S==0':>8}", f"{"+|moved":>9}"], rows)
    report["by_layer_family"] = {f"{layer}|{family}": describe(values)
                                 for (layer, family), values in pooled.items()}

    # Doc 5.5 asks whether the family ordering changes between the two matchings.
    for matching in sorted({row["matching"] for row in perturbed}):
        subset = [row for row in perturbed if row["matching"] == matching]
        grouped = paired_contrasts(subset)
        by_family = defaultdict(list)
        for (_layer, family, _m), values in grouped.items():
            by_family[family].extend(values)
        rows = [[f"{family:>9}", f"{describe(v)['n']:>7}",
                 f"{describe(v)['mean']:>+9.4f}", f"{describe(v)['t']:>8.1f}",
                 f"{describe(v)['share_zero']:>8.3f}",
                 f"{describe(v)['share_positive_of_moved']:>9.3f}"]
                for family, v in sorted(by_family.items(),
                                        key=lambda kv: -describe(kv[1])["mean"])]
        print_table(f"Family ordering at matched {matching} (all blocks pooled):",
                    [f"{'family':>9}", f"{'pairs':>7}", f"{'mean S':>9}",
                     f"{'t':>8}", f"{'S==0':>8}", f"{"+|moved":>9}"], rows)
        report[f"by_family_{matching}"] = {f: describe(v) for f, v in by_family.items()}

    # Doc 5.3's label-order control: a cosmetic relabelling must not change the effect.
    grouped = paired_contrasts(perturbed, extra_key=("label_order",))
    by_order = defaultdict(list)
    for key, values in grouped.items():
        by_order[key[3]].extend(values)
    rows = [[f"{order:>11}", f"{describe(v)['n']:>7}", f"{describe(v)['mean']:>+9.4f}",
             f"{describe(v)['share_zero']:>8.3f}",
             f"{describe(v)['share_positive_of_moved']:>9.3f}"]
            for order, v in sorted(by_order.items())]
    print_table("By label order (AB prints the first sentence as \"A\"):",
                [f"{'label_order':>11}", f"{'pairs':>7}", f"{'mean S':>9}",
                 f"{'S==0':>8}", f"{"+|moved":>9}"], rows)
    report["by_label_order"] = {o: describe(v) for o, v in by_order.items()}

    # The psychometric curve itself, on the estimator that survives section 1. Reading
    # accuracy_adjusted instead can show a clean rise that S says is not localization:
    # a perturbation that merely pulls the gap away from the default answer scores as
    # a hit on every trial targeting the other letter.
    if args.by_dose:
        grouped = paired_contrasts(perturbed, extra_key=("dose",))
        adjusted = {(row["layer"], row["family"], row["matching"], row["dose"]):
                    row for row in []}
        summaries = {}
        for run_dir in run_dirs:
            summary_path = run_dir / "summary.json"
            if summary_path.exists():
                with summary_path.open(encoding="utf-8") as handle:
                    for row in json.load(handle).get("per_dose", []):
                        summaries[(row["layer"], row["family"],
                                   row["matching"], row["dose"])] = row
        report["by_dose"] = {}
        for layer, family, matching in sorted({k[:3] for k in grouped}):
            rows = []
            for key in sorted((k for k in grouped if k[:3] == (layer, family, matching)),
                              key=lambda k: k[3]):
                stat = describe(grouped[key])
                cell = summaries.get((layer, family, matching, key[3]))
                rows.append([
                    f"{key[3]:>9g}", f"{stat['n']:>7}", f"{stat['mean']:>+9.4f}",
                    f"{stat['t']:>7.1f}", f"{stat['share_zero']:>7.3f}",
                    f"{stat['share_positive_of_moved']:>8.3f}",
                    f"{cell['accuracy_adjusted']:>9.3f}" if cell else f"{'':>9}",
                ])
                report["by_dose"][f"{layer}|{family}|{matching}|{key[3]:g}"] = stat
            print_table(f"S vs dose - block {layer}, {family}, matched {matching}:",
                        [f"{'dose':>9}", f"{'pairs':>7}", f"{'mean S':>9}",
                         f"{'t':>7}", f"{'S==0':>7}", f"{'+|moved':>8}",
                         f"{'acc_adj':>9}"], rows)

    null = permutation_null(perturbed, args.permutation_draws, args.seed)
    if null:
        observed = describe([v for values in pooled.values() for v in values])["mean"]
        sigma = (observed - null["mean"]) / null["sd"] if null["sd"] > 0 else float("inf")
        report["permutation_null"] = dict(null, observed=observed, sigma=sigma)
        print(f"\nPermutation null ({null['draws']} draws, target assignment shuffled):")
        print(f"  observed mean S {observed:+.4f}; null {null['mean']:+.4f} "
              f"+/- {null['sd']:.4f}; largest null draw {null['max_abs']:.4f} "
              f"-> {sigma:.1f} sd")

    if args.json_out:
        out = Path(args.json_out)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
