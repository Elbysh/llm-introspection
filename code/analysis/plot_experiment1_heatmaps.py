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
# report.tex moved to docs/misc and sets \graphicspath{{figures/}}, so this is where it
# looks. docs/livrables/figures no longer exists.
FIG_ROOT = REPO_ROOT / "docs/misc/figures"
FAMILIES = ["concept", "random", "noise", "dropout"]
LABELS = {"concept": "concept", "random": "aléatoire fixe", "noise": "bruit",
          "dropout": "dropout"}
# paper.tex is in English and report.tex is in French, so both label sets are kept.
# The French files stay where report.tex expects them; the English ones go to figures/paper/.
LABELS_EN = {"concept": "concept", "random": "fixed random", "noise": "renewed noise",
             "dropout": "dropout"}
STRINGS = {
    "fr": {"labels": LABELS, "dose": r"$\alpha$ par token ciblé",
           "ylabel": "bloc décodeur perturbé", "cbar": "$S$ moyen (logits)",
           "title": "{model} : contraste apparié $S$ par bloc et par dose, $\\alpha$ apparié"},
    "en": {"labels": LABELS_EN, "dose": r"$\alpha$ per targeted token",
           "ylabel": "perturbed decoder block", "cbar": "mean $S$ (logits)",
           "title": "{model}: paired contrast $S$ by block and dose, matched $\\alpha$"},
}
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


def heatmap(cells, matching, path, title, gate=None, lang="fr"):
    strings = STRINGS[lang]
    labels, dose_label = strings["labels"], strings["dose"]
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
        axis.set_title(labels[family], fontsize=8.5)
        axis.set_xlabel(dose_label, fontsize=7)
        if gate is not None and gate in layers:
            axis.axhline(layers.index(gate) - 0.5, color="k", lw=1.0)
    axes[0].set_ylabel(strings["ylabel"], fontsize=7.5)
    figure.suptitle(title, fontsize=9.5, y=1.01)
    bar = figure.colorbar(mesh, ax=axes, fraction=0.022, pad=0.02)
    bar.set_label(strings["cbar"], fontsize=7.5)
    bar.ax.tick_params(labelsize=6)
    figure.savefig(path)
    plt.close(figure)


def main():
    # The grouped alpha panels, not the source sweeps: those are no longer tracked, so
    # naming them made this script unrunnable from a clone. Llama is the ten-by-ten panel
    # of doc 5.7 and replaces full32_all's four concepts and three random directions --
    # a sample picked with the answers in hand, which flattered the concept/control gap by
    # about 20%. It reaches block 30, not 31: full32_all never swept block 31 for concept,
    # so that row was partial. qwen_alpha_panel is exactly qwen38_all plus qwen38-mid on
    # the alpha arm, 24,200 + 14,520 = 38,720 rows over the same blocks, families and
    # doses, so Qwen's figure is unchanged. Qwen was never completed and stays at 4 x 3.
    llama = paired_contrast(["llama_alpha_panel"])
    qwen = paired_contrast(["qwen_alpha_panel"])
    runs = [("Llama-3.1-8B", llama, "experiment1-llama", "llama", 14),
            ("Qwen3.8-27B", qwen, "experiment1-qwen", "qwen", None)]

    for model, cells, folder, stem, gate in runs:
        out = FIG_ROOT / folder; out.mkdir(parents=True, exist_ok=True)
        heatmap(cells, "alpha", out / "heatmap_alpha.pdf",
                STRINGS["fr"]["title"].format(model=model), gate=gate, lang="fr")

    out = FIG_ROOT / "paper"; out.mkdir(parents=True, exist_ok=True)
    for model, cells, _folder, stem, gate in runs:
        heatmap(cells, "alpha", out / f"heatmap_alpha_{stem}.pdf",
                STRINGS["en"]["title"].format(model=model), gate=gate, lang="en")
    print("wrote French heatmaps for report.tex and English heatmaps for paper.tex")


if __name__ == "__main__":
    main()
