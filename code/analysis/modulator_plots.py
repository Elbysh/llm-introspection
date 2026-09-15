#!/usr/bin/env python3
"""
Plots for Experience 11 modulators (modulators.py's saved trials, section
16.5), all scored on the ordering task's A/B letter accuracy. Each sweeps a
dose range (--alphas) crossed with its own modulator, so every plot draws
one line per alpha:
  E4: accuracy vs. |layer_i - layer_j|, one subplot per placement (early vs.
      late absolute position at a fixed distance), one line per alpha.
  E5: accuracy vs. dose ratio z_A / z_B, one line per base alpha.
  E6: accuracy vs. injected concept-pair cosine similarity, one line per alpha.
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


def _usable_trials(path):
    return [t for t in load_trials(path) if t["is_coherent"] and t["correct"] is not None]


def _alpha_colors(alphas):
    cmap = plt.get_cmap("viridis")
    if len(alphas) == 1:
        return {alphas[0]: cmap(0.5)}
    return {alpha: cmap(i / (len(alphas) - 1)) for i, alpha in enumerate(alphas)}


def plot_e4(path, output_path):
    trials = _usable_trials(path)
    placements = sorted(set(t["placement"] for t in trials))
    alphas = sorted(set(t["alpha"] for t in trials))
    colors = _alpha_colors(alphas)

    fig, axes = plt.subplots(1, len(placements), figsize=(7 * len(placements), 6), squeeze=False)
    axes = axes[0]
    for ax, placement in zip(axes, placements):
        for alpha in alphas:
            by_distance = defaultdict(list)
            for t in trials:
                if t["placement"] == placement and t["alpha"] == alpha:
                    by_distance[t["distance"]].append(t["correct"])
            distances = sorted(by_distance)
            if not distances:
                continue
            accuracies = [np.mean(by_distance[d]) for d in distances]
            ax.plot(distances, accuracies, marker="o", color=colors[alpha], label=f"alpha={alpha}")

        ax.axhline(0.5, color="gray", linestyle=":", label="chance")
        ax.set_xlabel("layer distance |i-j|")
        ax.set_ylabel("ordering accuracy")
        ax.set_ylim(0, 1)
        ax.set_title(f"E4: ordering accuracy vs. distance ({placement})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved E4 plot to {output_path}")


def plot_e5(path, output_path):
    trials = _usable_trials(path)
    alphas = sorted(set(t["base_alpha"] for t in trials))
    colors = _alpha_colors(alphas)

    fig, ax = plt.subplots(figsize=(8, 6))
    any_line = False
    for alpha in alphas:
        by_ratio = defaultdict(list)
        for t in trials:
            if t["base_alpha"] == alpha:
                by_ratio[t["ratio"]].append(t["correct"])
        ratios = sorted(by_ratio)
        if not ratios:
            continue
        any_line = True
        accuracies = [np.mean(by_ratio[r]) for r in ratios]
        ax.plot(ratios, accuracies, marker="o", color=colors[alpha], label=f"base_alpha={alpha}")

    if not any_line:
        print(f"No usable trials in {path}, skipping E5 plot")
        plt.close()
        return

    ax.axhline(0.5, color="gray", linestyle=":", label="chance")
    ax.set_xscale("log")
    ax.set_xlabel("dose ratio z_A / z_B")
    ax.set_ylabel("ordering accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("E5: ordering accuracy vs. relative injection strength")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved E5 plot to {output_path}")


def plot_e6(path, output_path):
    trials = _usable_trials(path)
    alphas = sorted(set(t["alpha"] for t in trials))
    colors = _alpha_colors(alphas)

    fig, ax = plt.subplots(figsize=(8, 6))
    any_line = False
    for alpha in alphas:
        by_sim = defaultdict(list)
        for t in trials:
            if t["alpha"] == alpha:
                by_sim[round(t["concept_pair_similarity"], 3)].append(t["correct"])
        sims = sorted(by_sim)
        if not sims:
            continue
        any_line = True
        accuracies = [np.mean(by_sim[s]) for s in sims]
        ax.plot(sims, accuracies, marker="o", color=colors[alpha], label=f"alpha={alpha}")

    if not any_line:
        print(f"No usable trials in {path}, skipping E6 plot")
        plt.close()
        return

    ax.axhline(0.5, color="gray", linestyle=":", label="chance")
    ax.set_xlabel("concept-pair cosine similarity")
    ax.set_ylabel("ordering accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("E6: ordering accuracy vs. concept-pair similarity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved E6 plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e4_input", type=Path, default=Path("plots/modulators_trials_llama_e4_distance.pt"),
                         help="modulators.py now tags output filenames with --model, e.g. "
                              "modulators_trials_qwen_e4_distance.pt for a Qwen run")
    parser.add_argument("--e5_input", type=Path, default=Path("plots/modulators_trials_llama_e5_alpha_ratio.pt"))
    parser.add_argument("--e6_input", type=Path, default=Path("plots/modulators_trials_llama_e6_similarity.pt"))
    parser.add_argument("--output_dir", type=Path, default=Path("plots"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.e4_input.exists():
        plot_e4(args.e4_input, args.output_dir / "modulators_e4_distance.png")
    if args.e5_input.exists():
        plot_e5(args.e5_input, args.output_dir / "modulators_e5_alpha_ratio.png")
    if args.e6_input.exists():
        plot_e6(args.e6_input, args.output_dir / "modulators_e6_similarity.png")


if __name__ == "__main__":
    main()
