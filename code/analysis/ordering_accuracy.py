#!/usr/bin/env python3
"""
Analysis for Block 3 layer-depth ordering (layer_ordering.py's saved trials):
  - Accuracy: fraction correctly reporting the shallower-layer concept.
  - C3.2 chance baseline: scipy.stats.binomtest against 50%.
  - C3.1 presentation-order bias: accuracy and "picked named_concept_1" rate,
    split by presentation_order, to detect primacy/recency independent of the
    real signal.
  - Accuracy by layer distance, as a preview of E4's distance sweep.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest


def load_trials(input_path):
    data = torch.load(input_path, weights_only=False)
    return data["trials"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("plots/layer_ordering_trials.pt"))
    args = parser.parse_args()

    trials = load_trials(args.input)
    coherent = [t for t in trials if t["is_coherent"] and t["reported_concept"] is not None]
    excluded = len(trials) - len(coherent)

    print("=" * 80)
    print(f"BLOCK 3 ORDERING ANALYSIS (n_trials={len(trials)}, excluded_incoherent_or_ambiguous={excluded})")
    print("=" * 80)

    correct_flags = [t["correct"] for t in coherent]
    n_correct = sum(correct_flags)
    n = len(correct_flags)
    accuracy = n_correct / n if n else float("nan")
    print(f"\nOverall accuracy: {accuracy:.1%} ({n_correct}/{n})")

    if n:
        result = binomtest(n_correct, n, p=0.5, alternative="two-sided")
        print(f"C3.2 binomial test vs. chance (50%): p={result.pvalue:.4g}")

    print("\n--- C3.1 presentation-order bias ---")
    by_order = defaultdict(list)
    for t in coherent:
        by_order[t["presentation_order"]].append(t)
    for order, group in sorted(by_order.items()):
        acc = float(np.mean([t["correct"] for t in group]))
        picked_first = float(np.mean([t["reported_concept"] == t["named_concept_1"] for t in group]))
        print(f"  presentation_order={order}: accuracy={acc:.1%}, "
              f"picked_named_concept_1_rate={picked_first:.1%} (n={len(group)})")

    print("\n--- Accuracy by layer distance ---")
    by_distance = defaultdict(list)
    for t in coherent:
        by_distance[t["layer_distance"]].append(t["correct"])
    for distance in sorted(by_distance):
        vals = by_distance[distance]
        print(f"  distance={distance}: accuracy={np.mean(vals):.1%} (n={len(vals)})")


if __name__ == "__main__":
    main()
