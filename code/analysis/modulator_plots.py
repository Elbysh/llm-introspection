#!/usr/bin/env python3
"""
Plots for Block 4 modulators (modulators.py's saved trials):
  E4 (H4): accuracy vs. |layer_i - layer_j|, identification and ordering, split
           by placement (C4.1: fixed distance at early vs. late absolute position).
  E5 (H5): identification accuracy vs. alpha_A/alpha_B ratio.
  E6 (H6): identification accuracy vs. injected concept-pair cosine similarity.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_trials(path):
    data = torch.load(path, weights_only=False)
    return data["trials"]


def identification_correct(trial, threshold):
    sims = trial.get("sim_to_injected")
    if not sims:
        return None
    return all(s > threshold for s in sims)


def plot_e4(path, output_path, threshold):
    trials = [t for t in load_trials(path) if t["is_coherent"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, task in zip(axes, ["identification", "ordering"]):
        task_trials = [t for t in trials if t["task"] == task]
        for placement in sorted(set(t["placement"] for t in task_trials)):
            by_distance = defaultdict(list)
            for t in task_trials:
                if t["placement"] != placement:
                    continue
                correct = identification_correct(t, threshold) if task == "identification" else t["correct"]
                if correct is not None:
                    by_distance[t["distance"]].append(correct)
            distances = sorted(by_distance)
            if not distances:
                continue
            accuracies = [np.mean(by_distance[d]) for d in distances]
            ax.plot(distances, accuracies, marker="o", label=placement)

        ax.axhline(0.5, color="gray", linestyle=":", label="chance")
        ax.set_xlabel("layer distance |i-j|")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1)
        ax.set_title(f"E4: {task} accuracy vs. layer distance")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved E4 plot to {output_path}")


def plot_e5(path, output_path, threshold):
    trials = [t for t in load_trials(path) if t["is_coherent"]]
    by_ratio = defaultdict(list)
    for t in trials:
        correct = identification_correct(t, threshold)
        if correct is not None:
            by_ratio[t["ratio"]].append(correct)

    ratios = sorted(by_ratio)
    if not ratios:
        print(f"No usable trials in {path}, skipping E5 plot")
        return
    accuracies = [np.mean(by_ratio[r]) for r in ratios]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ratios, accuracies, marker="o")
    ax.axhline(0.5, color="gray", linestyle=":", label="chance")
    ax.set_xscale("log")
    ax.set_xlabel("alpha_A / alpha_B")
    ax.set_ylabel("identification accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("E5: identification accuracy vs. relative injection strength")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved E5 plot to {output_path}")


def plot_e6(path, output_path, threshold):
    trials = [t for t in load_trials(path) if t["is_coherent"]]
    by_sim = defaultdict(list)
    for t in trials:
        correct = identification_correct(t, threshold)
        if correct is not None:
            by_sim[round(t["concept_pair_similarity"], 3)].append(correct)

    sims = sorted(by_sim)
    if not sims:
        print(f"No usable trials in {path}, skipping E6 plot")
        return
    accuracies = [np.mean(by_sim[s]) for s in sims]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sims, accuracies, marker="o")
    ax.axhline(0.5, color="gray", linestyle=":", label="chance")
    ax.set_xlabel("concept-pair cosine similarity")
    ax.set_ylabel("identification accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("E6: identification accuracy vs. concept-pair similarity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved E6 plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e4_input", type=Path, default=Path("plots/modulators_trials_e4_distance.pt"))
    parser.add_argument("--e5_input", type=Path, default=Path("plots/modulators_trials_e5_alpha_ratio.pt"))
    parser.add_argument("--e6_input", type=Path, default=Path("plots/modulators_trials_e6_similarity.pt"))
    parser.add_argument("--output_dir", type=Path, default=Path("plots"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.e4_input.exists():
        plot_e4(args.e4_input, args.output_dir / "modulators_e4_distance.png", args.threshold)
    if args.e5_input.exists():
        plot_e5(args.e5_input, args.output_dir / "modulators_e5_alpha_ratio.png", args.threshold)
    if args.e6_input.exists():
        plot_e6(args.e6_input, args.output_dir / "modulators_e6_similarity.png", args.threshold)


if __name__ == "__main__":
    main()
