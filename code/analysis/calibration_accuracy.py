#!/usr/bin/env python3
"""
Loads calibration.py's saved trial records and produces:
  - a detection-rate / false-positive-rate table by (layer, alpha)
  - a similarity-vs-baseline check (mean concept_similarity, alpha=0 vs alpha>0)
  - a detection-rate heatmap (layer x alpha)

This is "select operating window for Blocks 1-4" from the plan: cells where
detection rate is well above the alpha=0 false-positive rate, at high
coherence, mark the usable (layer, alpha) region for Blocks 1-4.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_trials(input_path):
    data = torch.load(input_path, weights_only=False)
    return data["trials"], data["layers"], data["alphas"]


def compute_tables(trials):
    """
    Returns, keyed by (layer, alpha):
      detection_rate: fraction of coherent trials with claims_noticing == True
      mean_similarity: mean concept_similarity among coherent trials
      coherence_rate: fraction of all trials that were coherent
      n: number of coherent trials the rates above are computed over
    """
    coherent_by_cell = defaultdict(list)
    coherence_counts = defaultdict(lambda: [0, 0])  # [coherent, total]

    for t in trials:
        key = (t["layer"], t["alpha"])
        coherence_counts[key][1] += 1
        if not t["is_coherent"]:
            continue
        coherence_counts[key][0] += 1
        coherent_by_cell[key].append(t)

    detection_rate, mean_similarity, n_counted = {}, {}, {}
    for key, cell_trials in coherent_by_cell.items():
        claims = [t["claims_noticing"] for t in cell_trials if t["claims_noticing"] is not None]
        detection_rate[key] = float(np.mean(claims)) if claims else float("nan")
        sims = [t["concept_similarity"] for t in cell_trials]
        mean_similarity[key] = float(np.mean(sims)) if sims else float("nan")
        n_counted[key] = len(cell_trials)

    coherence_rate = {
        key: (coh / total if total else float("nan"))
        for key, (coh, total) in coherence_counts.items()
    }

    return detection_rate, mean_similarity, coherence_rate, n_counted


def print_tables(layers, alphas, detection_rate, mean_similarity, coherence_rate, n_counted):
    header = "Layer |" + "".join(f"  a={a:<4} |" for a in alphas)
    print("DETECTION RATE (fraction of coherent trials claiming to notice something)")
    print(header)
    print("-" * len(header))
    for layer in layers:
        row = f"  {layer:3d} |"
        for alpha in alphas:
            rate = detection_rate.get((layer, alpha), float("nan"))
            row += f"  {rate:5.0%} |" if rate == rate else "    -   |"
        print(row)

    print("\nMEAN CONCEPT SIMILARITY (embedding cosine sim, response vs. true injected concept)")
    print(header)
    print("-" * len(header))
    for layer in layers:
        row = f"  {layer:3d} |"
        for alpha in alphas:
            sim = mean_similarity.get((layer, alpha), float("nan"))
            row += f" {sim:6.3f} |" if sim == sim else "    -   |"
        print(row)

    print("\nCOHERENCE RATE (fraction of trials passing is_coherent)")
    print(header)
    print("-" * len(header))
    for layer in layers:
        row = f"  {layer:3d} |"
        for alpha in alphas:
            coh = coherence_rate.get((layer, alpha), float("nan"))
            row += f"  {coh:5.0%} |" if coh == coh else "    -   |"
        print(row)

    if 0 in alphas:
        print("\nFALSE-POSITIVE RATE (alpha=0 detection rate, per layer):")
        for layer in layers:
            fpr = detection_rate.get((layer, 0), float("nan"))
            print(f"  layer {layer:3d}: {fpr:.0%}" if fpr == fpr else f"  layer {layer:3d}: n/a")

    print("\nSIMILARITY-VS-BASELINE CHECK (mean concept_similarity, alpha=0 vs. alpha>0):")
    if 0 in alphas:
        baseline_sims = [mean_similarity[(layer, 0)] for layer in layers if (layer, 0) in mean_similarity]
        injected_sims = [
            mean_similarity[(layer, alpha)]
            for layer in layers for alpha in alphas
            if alpha != 0 and (layer, alpha) in mean_similarity
        ]
        baseline_mean = float(np.mean(baseline_sims)) if baseline_sims else float("nan")
        injected_mean = float(np.mean(injected_sims)) if injected_sims else float("nan")
        print(f"  alpha=0 mean similarity: {baseline_mean:.3f}")
        print(f"  alpha>0 mean similarity: {injected_mean:.3f}")
        print(f"  delta: {injected_mean - baseline_mean:+.3f}")
    else:
        print("  (no alpha=0 cell present in this sweep)")


def plot_heatmap(layers, alphas, detection_rate, output_path):
    grid = np.full((len(layers), len(alphas)), np.nan)
    for i, layer in enumerate(layers):
        for j, alpha in enumerate(alphas):
            grid[i, j] = detection_rate.get((layer, alpha), np.nan)

    fig, ax = plt.subplots(figsize=(1.2 * len(alphas) + 2, 0.4 * len(layers) + 2))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels(alphas)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    ax.set_xlabel("alpha")
    ax.set_ylabel("layer")
    ax.set_title("Block 0 calibration: detection rate by (layer, alpha)")
    fig.colorbar(im, ax=ax, label="detection rate")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved heatmap to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("plots/calibration_trials_llama.pt"),
                         help="calibration.py now tags its output filename with --model, e.g. "
                              "calibration_trials_qwen.pt for a Qwen run")
    parser.add_argument("--output", type=Path, default=Path("plots/calibration_heatmap.png"))
    args = parser.parse_args()

    trials, layers, alphas = load_trials(args.input)
    detection_rate, mean_similarity, coherence_rate, n_counted = compute_tables(trials)

    print("=" * 80)
    print(f"CALIBRATION ACCURACY (n_trials={len(trials)}, layers={layers}, alphas={alphas})")
    print("=" * 80)
    print_tables(layers, alphas, detection_rate, mean_similarity, coherence_rate, n_counted)
    plot_heatmap(layers, alphas, detection_rate, args.output)


if __name__ == "__main__":
    main()
