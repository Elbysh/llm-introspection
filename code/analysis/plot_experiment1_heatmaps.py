"""Block x dose heatmaps of the paired contrast S, for report.tex.

The layer_maps heatmaps plot `accuracy_adjusted`, which the Llama analysis shows is
inflated by the letter-A bias. These plot S (doc 5.8) instead, which cancels any shift
that does not depend on which sentence was targeted.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_ROOT = REPO_ROOT / "docs/livrables/figures"
FAMILIES = ["concept", "random", "noise", "dropout"]
LABELS = {"concept": "concept", "random": "aléatoire fixe", "noise": "bruit",
          "dropout": "dropout"}
KEY = ["layer", "family", "matching", "dose", "direction_id", "pair_id", "order",
       "label_order"]
plt.rcParams.update({"font.size": 8, "figure.dpi": 160, "savefig.bbox": "tight"})


def paired_contrast(runs):
    """S per (layer, family, matching, dose), pooled over the given runs."""
    frames = []
    for run in runs:
        trials = pd.read_csv(REPO_ROOT / f"results/experiment1/{run}/trials.csv",
                             low_memory=False)
        frames.append(trials[trials.kind == "perturbed"])
    p = pd.concat(frames, ignore_index=True)
    p["signed"] = np.where(p.target_label == "A", p.contrast, -p.contrast)
    g = (p.groupby(KEY, sort=False)
           .agg(total=("signed", "sum"), n=("signed", "size")).reset_index())
    g = g[g.n == 2].copy()
    g["S"] = g.total / 2.0
    g["layer"] = g.layer.astype(int)
    return g


def heatmap(cells, matching, path, title, gate=None, dose_label=r"$\alpha$ par token ciblé"):
    sub = cells[cells.matching == matching]
    layers = sorted(sub.layer.unique())
    doses = sorted(sub.dose.unique())
    grids = {}
    for family in FAMILIES:
        grid = np.full((len(layers), len(doses)), np.nan)
        cell = (sub[sub.family == family]
                .groupby(["layer", "dose"]).S.mean())
        for (layer, dose), value in cell.items():
            grid[layers.index(layer), doses.index(dose)] = value
        grids[family] = grid

    finite = np.concatenate([g[np.isfinite(g)] for g in grids.values()])
    limit = float(np.percentile(np.abs(finite), 99))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    figure, axes = plt.subplots(1, 4, figsize=(7.6, 3.4), sharey=True)
    for column, family in enumerate(FAMILIES):
        axis = axes[column]
        mesh = axis.imshow(np.ma.masked_invalid(grids[family]), aspect="auto",
                           origin="lower", cmap="RdBu_r", norm=norm,
                           interpolation="nearest")
        axis.set_xticks(range(len(doses)))
        axis.set_xticklabels([f"{d:g}" for d in doses], rotation=90, fontsize=6)
        axis.set_yticks(range(len(layers)))
        axis.set_yticklabels([str(l) for l in layers], fontsize=6)
        axis.set_title(LABELS[family], fontsize=8.5)
        axis.set_xlabel(dose_label, fontsize=7)
        if gate is not None and gate in layers:
            axis.axhline(layers.index(gate) - 0.5, color="k", lw=1.0)
    axes[0].set_ylabel("bloc décodeur perturbé", fontsize=7.5)
    figure.suptitle(title, fontsize=9.5, y=1.01)
    bar = figure.colorbar(mesh, ax=axes, fraction=0.022, pad=0.02)
    bar.set_label("$S$ moyen (logits)", fontsize=7.5)
    bar.ax.tick_params(labelsize=6)
    figure.savefig(path)
    plt.close(figure)


def main():
    llama = paired_contrast(["full32_all"])
    out = FIG_ROOT / "experiment1-llama"; out.mkdir(parents=True, exist_ok=True)
    heatmap(llama, "alpha", out / "heatmap_alpha.pdf",
            "Llama-3.1-8B : contraste apparié $S$ par bloc et par dose, $\\alpha$ apparié",
            gate=14)

    qwen = paired_contrast(["qwen38_all", "qwen38-mid"])
    out = FIG_ROOT / "experiment1-qwen"; out.mkdir(parents=True, exist_ok=True)
    heatmap(qwen, "alpha", out / "heatmap_alpha.pdf",
            "Qwen3.8-27B : contraste apparié $S$ par bloc et par dose, $\\alpha$ apparié")
    print("wrote heatmap_alpha.pdf for both models")


if __name__ == "__main__":
    main()
