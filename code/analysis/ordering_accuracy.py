#!/usr/bin/env python3
"""
Analysis for Experience 11 layer-depth ordering (layer_ordering.py's saved
trials, section 16):
  - Accuracy: fraction correctly reporting the letter assigned to the
    shallower-layer concept.
  - Chance baseline: scipy.stats.binomtest against 50%.
  - Label-assignment bias: accuracy and "picked A" rate, split by
    label_assignment, to detect an A/B position bias independent of the
    real signal.
  - Accuracy by layer distance, as a preview of the E4 distance-sweep
    modulator.
  - Accuracy by alpha, since layer_ordering.py sweeps a dose range
    (--alphas) in one file rather than one alpha per run.
  - Logit contrast (section 16.6): logit(correct letter) - logit(incorrect
    letter) at the first response token, sign-adjusted per trial so a
    positive value always means evidence toward the correct answer,
    canceling out a fixed A/B letter preference (see the label-assignment
    bias above). Available for every trial, including incoherent/unparsed
    ones, since it doesn't depend on parsing the decoded text -- a
    one-sample t-test against 0 uses the full n, not just the answered
    subset.
  - Double-adjusted logit contrast: the above minus the same contrast
    measured on a matched sham generation (same prompt, no injection at
    all) -- the plan's L_adjusted pattern (section 5.7) applied here,
    isolating the injection's own contribution from whatever baseline A/B
    preference the prompt has with no injection. Requires layer_ordering.py
    to have been run with the paired-sham update.
  Older trial files saved before either measure was added are handled
  gracefully (reported as unavailable).
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest, ttest_1samp


def load_trials(input_path):
    data = torch.load(input_path, weights_only=False)
    return data["trials"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("plots/layer_ordering_trials.pt"))
    args = parser.parse_args()

    trials = load_trials(args.input)
    coherent = [t for t in trials if t["is_coherent"] and t["reported_letter"] is not None]
    excluded = len(trials) - len(coherent)

    print("=" * 80)
    print(f"EXPERIENCE 11 ORDERING ANALYSIS (n_trials={len(trials)}, excluded_incoherent_or_ambiguous={excluded})")
    print("=" * 80)

    correct_flags = [t["correct"] for t in coherent]
    n_correct = sum(correct_flags)
    n = len(correct_flags)
    accuracy = n_correct / n if n else float("nan")
    print(f"\nOverall accuracy: {accuracy:.1%} ({n_correct}/{n})")

    if n:
        result = binomtest(n_correct, n, p=0.5, alternative="two-sided")
        print(f"Binomial test vs. chance (50%): p={result.pvalue:.4g}")

    print("\n--- Label-assignment bias ---")
    by_assignment = defaultdict(list)
    for t in coherent:
        by_assignment[t["label_assignment"]].append(t)
    for assignment, group in sorted(by_assignment.items()):
        acc = float(np.mean([t["correct"] for t in group]))
        picked_a = float(np.mean([t["reported_letter"] == "A" for t in group]))
        print(f"  label_assignment={assignment}: accuracy={acc:.1%}, "
              f"picked_a_rate={picked_a:.1%} (n={len(group)})")

    print("\n--- Accuracy by layer distance ---")
    by_distance = defaultdict(list)
    for t in coherent:
        by_distance[t["layer_distance"]].append(t["correct"])
    for distance in sorted(by_distance):
        vals = by_distance[distance]
        print(f"  distance={distance}: accuracy={np.mean(vals):.1%} (n={len(vals)})")

    print("\n--- Accuracy by alpha ---")
    by_alpha = defaultdict(list)
    for t in coherent:
        by_alpha[t["alpha"]].append(t["correct"])
    for alpha in sorted(by_alpha):
        vals = by_alpha[alpha]
        print(f"  alpha={alpha}: accuracy={np.mean(vals):.1%} (n={len(vals)})")

    print("\n--- Logit contrast (adjusted for which letter was correct) ---")
    if not all("logit_contrast_adjusted" in t for t in trials):
        print("  not present in this file -- rerun layer_ordering.py to get this measure")
    else:
        all_vals = [t["logit_contrast_adjusted"] for t in trials]
        print(f"  mean, all {len(all_vals)} trials (incl. incoherent/unparsed): {np.mean(all_vals):+.3f}")
        if len(all_vals) > 1:
            result = ttest_1samp(all_vals, 0.0)
            print(f"  one-sample t-test vs. 0: t={result.statistic:.3f}, p={result.pvalue:.4g}")

        coherent_vals = [t["logit_contrast_adjusted"] for t in coherent]
        if coherent_vals:
            print(f"  mean, coherent & answered only (n={len(coherent_vals)}): {np.mean(coherent_vals):+.3f}")

        print("\n  mean logit contrast by alpha (all trials):")
        by_alpha_logit = defaultdict(list)
        for t in trials:
            by_alpha_logit[t["alpha"]].append(t["logit_contrast_adjusted"])
        for alpha in sorted(by_alpha_logit):
            vals = by_alpha_logit[alpha]
            print(f"    alpha={alpha}: {np.mean(vals):+.3f} (n={len(vals)})")

    print("\n--- Double-adjusted logit contrast (minus matched sham baseline) ---")
    if not all("logit_contrast_double_adjusted" in t for t in trials):
        print("  not present in this file -- rerun layer_ordering.py with the paired-sham update")
    else:
        double_vals = [t["logit_contrast_double_adjusted"] for t in trials]
        print(f"  mean, all {len(double_vals)} trials: {np.mean(double_vals):+.3f}")
        if len(double_vals) > 1:
            result = ttest_1samp(double_vals, 0.0)
            print(f"  one-sample t-test vs. 0: t={result.statistic:.3f}, p={result.pvalue:.4g}")

        sham_vals = [t["logit_contrast_adjusted_sham"] for t in trials]
        print(f"  (for reference) mean sham-only contrast: {np.mean(sham_vals):+.3f} -- "
              "the prompt's own baseline A/B preference with no injection at all")

        print("\n  mean double-adjusted contrast by alpha (all trials):")
        by_alpha_double = defaultdict(list)
        for t in trials:
            by_alpha_double[t["alpha"]].append(t["logit_contrast_double_adjusted"])
        for alpha in sorted(by_alpha_double):
            vals = by_alpha_double[alpha]
            print(f"    alpha={alpha}: {np.mean(vals):+.3f} (n={len(vals)})")


if __name__ == "__main__":
    main()
