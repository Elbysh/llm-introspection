#!/usr/bin/env python3
"""Generate a baseline-adjusted accuracy table from position-detection results."""

import argparse
import ast
from pathlib import Path

import torch


def load_trial_logit_diffs(results_file):
    data = torch.load(results_file, map_location="cpu", weights_only=False)
    trial_diffs = {}

    for concept_results in data["all_results"].values():
        for key, trials in concept_results.items():
            layer, strength = ast.literal_eval(key)
            trial_diffs.setdefault((int(layer), float(strength)), [])
            trial_diffs[(int(layer), float(strength))].extend(
                trial["logit_diff"] for trial in trials
            )

    return trial_diffs


def print_table(trial_diffs, baseline):
    layers = sorted({layer for layer, _ in trial_diffs})
    strengths = sorted({strength for _, strength in trial_diffs})

    print("| Layer | " + " | ".join(f"α={strength:g}" for strength in strengths) + " |")
    print("|---:|" + "|".join("---:" for _ in strengths) + "|")

    for layer in layers:
        values = []
        for strength in strengths:
            diffs = trial_diffs[(layer, strength)]
            adjusted_accuracy = 100 * sum(
                logit_diff - baseline > 0 for logit_diff in diffs
            ) / len(diffs)
            values.append(f"{adjusted_accuracy:.1f}")
        print(f"| L{layer} | " + " | ".join(values) + " |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("plots/position_detection_aggregated.pt"),
    )
    parser.add_argument("--baseline", type=float, default=-3.317)
    args = parser.parse_args()

    trial_diffs = load_trial_logit_diffs(args.results_file)
    print(f"Baseline LD: {args.baseline:+.3f}\n")
    print_table(trial_diffs, args.baseline)


if __name__ == "__main__":
    main()