#!/usr/bin/env python3
"""
Analysis for Experience 9 counting (multi_detection.py's saved trials,
section 14):
  - Confusion matrix: reported vs. actual k (count-report prompt style)
  - False-positive rate: detection rate at k=0 (the plan's actual sham)
  - real-vs-random comparison: detection rate by condition, at matched k
    ("random" is the plan's active control, section 14.4 step 6 -- not its
    sham, which is k=0)
  - Rephrasing check: mean 0-10 scale rating by condition/k, to confirm
    results aren't an artifact of yes-biased phrasing in the count-report prompt
  - Logit contrast: logit(true k's digit) - logit("1") at the first response
    token, per trial -- isolates whether the true count gets more support
    than the "always 1" default bias, beyond what the reported digit alone
    shows. Excludes k_actual==1 (trivially 0 there). Older trial files saved
    before this measure was added are handled gracefully.

Every table is broken down by dose, since multi_detection.py sweeps a dose
range in one file rather than one dose per run: "alpha" in the individual
regime, "z_total" in the budget regime (the per-injection alpha there
depends on k too, so z_total is the more meaningful sweep axis).

--input defaults to the "individual" dose-regime output; pass the "budget"
regime's file separately to compare (section 14.2's two dosing regimes are
saved to separate files by multi_detection.py, since dose_regime is fixed
per run).
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ttest_1samp


def load_trials(input_path):
    data = torch.load(input_path, weights_only=False)
    return data["trials"]


def build_confusion_matrix(trials, k_values, condition="real"):
    """Reported-count vs. actual-k confusion matrix, count-report prompt style,
    coherent trials only, for the given condition."""
    matrix = defaultdict(Counter)
    for t in trials:
        if t["prompt_style"] != "count" or t["condition"] != condition or not t["is_coherent"]:
            continue
        matrix[t["k_actual"]][t["reported_count"]] += 1
    return matrix


def print_confusion_matrix(matrix, k_values):
    columns = [*k_values, None]
    header = "actual\\reported |" + "".join(f" {str(r):>5} |" for r in columns)
    print(header)
    print("-" * len(header))
    for actual in k_values:
        counts = matrix.get(actual, Counter())
        total = sum(counts.values())
        row = f"{actual:>15} |"
        for reported in columns:
            n = counts.get(reported, 0)
            frac = n / total if total else float("nan")
            row += f" {frac:5.0%} |" if frac == frac else "     - |"
        print(row + f"   (n={total})")


def detection_rate_by_condition_and_k(trials, prompt_style="count"):
    """Fraction of trials where the model claims to detect >0 injections
    (count style) or rates processing as >0 different from normal (scale
    style), keyed by (condition, k_actual)."""
    counts = defaultdict(lambda: [0, 0])  # [affirmative, total]
    for t in trials:
        if t["prompt_style"] != prompt_style or not t["is_coherent"]:
            continue
        key = (t["condition"], t["k_actual"])
        counts[key][1] += 1
        value = t["reported_count"] if prompt_style == "count" else t["reported_scale"]
        if value is not None and value > 0:
            counts[key][0] += 1
    return {key: (aff / total if total else float("nan")) for key, (aff, total) in counts.items()}


def rephrasing_check(trials):
    """Mean 0-10 scale rating, per (condition, k) -- the C1.3 rephrasing check."""
    scale_values = defaultdict(list)
    for t in trials:
        if t["prompt_style"] != "scale" or not t["is_coherent"] or t.get("reported_scale") is None:
            continue
        scale_values[(t["condition"], t["k_actual"])].append(t["reported_scale"])
    return {key: float(np.mean(v)) for key, v in scale_values.items()}


def logit_contrast_check(trials, condition):
    """Mean logit(true k) - logit("1") for count-style trials with k != 1
    (trivially 0 there), for the given condition -- available for every
    trial regardless of is_coherent, since it doesn't depend on parsing."""
    vals = [t["logit_contrast_adjusted"] for t in trials
            if t["prompt_style"] == "count" and t["condition"] == condition
            and t["k_actual"] != 1 and "logit_contrast_adjusted" in t]
    if not vals:
        return None
    result = {"n": len(vals), "mean": float(np.mean(vals))}
    if len(vals) > 1:
        result["t"] = float(ttest_1samp(vals, 0.0).statistic)
        result["p"] = float(ttest_1samp(vals, 0.0).pvalue)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("plots/multi_detection_trials_llama_individual.pt"),
                         help="multi_detection.py now tags its output filename with --model, e.g. "
                              "multi_detection_trials_qwen_individual.pt for a Qwen run")
    args = parser.parse_args()

    data = torch.load(args.input, weights_only=False)
    trials = data["trials"]
    dose_regime = data.get("dose_regime", "unknown")
    dose_field = "alpha" if dose_regime == "individual" else "z_total"
    k_values = sorted(set(t["k_actual"] for t in trials))
    conditions = sorted(set(t["condition"] for t in trials))
    doses = sorted(set(t[dose_field] for t in trials))

    print("=" * 80)
    print(f"EXPERIENCE 9 COUNTING ANALYSIS: {args.input.name} "
          f"(n_trials={len(trials)}, dose_regime={dose_regime}, {dose_field}s={doses})")
    print("=" * 80)

    for dose in doses:
        by_dose = [t for t in trials if t[dose_field] == dose]
        print(f"\n{'#'*60}\n### {dose_field}={dose}\n{'#'*60}")

        for condition in conditions:
            print(f"\n--- confusion matrix (condition={condition}) ---")
            matrix = build_confusion_matrix(by_dose, k_values, condition=condition)
            print_confusion_matrix(matrix, k_values)

        det_rate = detection_rate_by_condition_and_k(by_dose, prompt_style="count")

        print("\n--- false-positive rate (k=0, 'detects >0 injections' rate) ---")
        for condition in conditions:
            fpr = det_rate.get((condition, 0), float("nan"))
            print(f"  condition={condition}: {fpr:.0%}" if fpr == fpr else f"  condition={condition}: n/a")

        print("\n--- real-vs-random detection rate by k ---")
        for k in k_values:
            row = f"  k={k}: "
            for condition in conditions:
                rate = det_rate.get((condition, k), float("nan"))
                row += f"{condition}={rate:.0%}  " if rate == rate else f"{condition}=n/a  "
            print(row)

        if any(t["prompt_style"] == "scale" for t in by_dose):
            print("\n--- rephrasing check: mean 0-10 'different from normal' rating ---")
            scale_means = rephrasing_check(by_dose)
            for condition in conditions:
                for k in k_values:
                    val = scale_means.get((condition, k), float("nan"))
                    print(f"  condition={condition}, k={k}: {val:.2f}"
                          if val == val else f"  condition={condition}, k={k}: n/a")

        print("\n--- logit contrast: logit(true k) - logit('1'), k != 1 ---")
        for condition in conditions:
            result = logit_contrast_check(by_dose, condition)
            if result is None:
                print(f"  condition={condition}: not present in this file")
            elif "t" in result:
                print(f"  condition={condition}: mean={result['mean']:+.3f} (n={result['n']}), "
                      f"t={result['t']:.3f}, p={result['p']:.3g}")
            else:
                print(f"  condition={condition}: mean={result['mean']:+.3f} (n={result['n']})")


if __name__ == "__main__":
    main()
