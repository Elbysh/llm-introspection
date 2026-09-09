#!/usr/bin/env python3
"""
Analysis for Block 1 detection (multi_detection.py's saved trials):
  - C1.2 confusion matrix: reported vs. actual k (count-report prompt style)
  - C1.1 false-positive rate: detection rate at k=0
  - C1.4 real-vs-sham comparison: detection rate by condition, at matched k
  - C1.3 rephrasing check: mean 0-10 scale rating by condition/k, to confirm
    results aren't an artifact of yes-biased phrasing in the count-report prompt
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("plots/multi_detection_trials.pt"))
    args = parser.parse_args()

    trials = load_trials(args.input)
    k_values = sorted(set(t["k_actual"] for t in trials))
    conditions = sorted(set(t["condition"] for t in trials))

    print("=" * 80)
    print(f"BLOCK 1 DETECTION ANALYSIS (n_trials={len(trials)})")
    print("=" * 80)

    for condition in conditions:
        print(f"\n--- C1.2 confusion matrix (condition={condition}) ---")
        matrix = build_confusion_matrix(trials, k_values, condition=condition)
        print_confusion_matrix(matrix, k_values)

    det_rate = detection_rate_by_condition_and_k(trials, prompt_style="count")

    print("\n--- C1.1 false-positive rate (k=0, 'detects >0 injections' rate) ---")
    for condition in conditions:
        fpr = det_rate.get((condition, 0), float("nan"))
        print(f"  condition={condition}: {fpr:.0%}" if fpr == fpr else f"  condition={condition}: n/a")

    print("\n--- C1.4 real-vs-sham detection rate by k ---")
    for k in k_values:
        row = f"  k={k}: "
        for condition in conditions:
            rate = det_rate.get((condition, k), float("nan"))
            row += f"{condition}={rate:.0%}  " if rate == rate else f"{condition}=n/a  "
        print(row)

    if any(t["prompt_style"] == "scale" for t in trials):
        print("\n--- C1.3 rephrasing check: mean 0-10 'different from normal' rating ---")
        scale_means = rephrasing_check(trials)
        for condition in conditions:
            for k in k_values:
                val = scale_means.get((condition, k), float("nan"))
                print(f"  condition={condition}, k={k}: {val:.2f}" if val == val else f"  condition={condition}, k={k}: n/a")


if __name__ == "__main__":
    main()
