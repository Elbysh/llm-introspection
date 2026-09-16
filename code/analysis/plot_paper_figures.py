"""Build the figures the manuscript uses, in English, into docs/figures.

The report deliverable keeps its own French figures under docs/livrables/figures.
This writes only what the manuscript cites, so the paper reads from one directory.

Every standardized (z) panel comes from a sweep calibrated in the behavioural 2AFC
context, results/experiment_0_calibration_2afc for Llama and
results/experiment_0_calibration_qwen38_27b_2afc for Qwen. The earlier z sweeps, which
divided by scales estimated on isolated sentences, are superseded and are not plotted.

The degradation figures read results/experiment4/main/summary.json, the per-cell
summary of the task-degradation sweep, which is the only file of that run the
manuscript needs; its trials stay in the experiment_4 tree.

Run: python code/analysis/plot_paper_figures.py
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "docs/figures"
LEGACY_FIGURES = REPO_ROOT / "docs/livrables/figures"
FAMILIES = ["concept", "random", "noise", "dropout"]
LABELS = {"concept": "concept", "random": "fixed random", "noise": "renewed noise",
          "dropout": "dropout", "scrambled": "scrambled concept"}
COLOURS = {"concept": "#1b6ca8", "random": "#c8553d", "noise": "#3c896d",
           "dropout": "#8a6bbe", "scrambled": "#d98b28"}
KEY = ["layer", "family", "matching", "dose", "direction_id", "pair_id", "order",
       "label_order"]
EXP4_SUMMARY = REPO_ROOT / "results/experiment4/main/summary.json"
# One ICML column. Figures are drawn at this width so LaTeX never rescales them,
# and the drawn width is a little under it because savefig trims to a tight box.
COLUMN = 3.25
DETECT = 0.70   # experiment 1 adjusted accuracy at or above which a cell counts as detected
DAMAGE = 0.05   # accuracy lost against sham at or above which a cell counts as damaged

# One row of the map figure: label, runs to pool, dose rule, block boundary to mark.
# panel-concepts and panel-random add the six concepts and the seven fixed-random
# directions the first sweeps left out, on both rules, so the Llama rows are the full
# ten-by-ten panel of doc 5.7. Qwen was not completed and stays at four by three.
ROWS = [
    ("Llama, matched $\\alpha$", ["full32_all", "panel-concepts", "panel-random"], "alpha", 14),
    ("Llama, matched $z$", ["z2afc_all", "panel-concepts", "panel-random"], "z", 14),
    ("Qwen, matched $\\alpha$", ["qwen38_all", "qwen38-mid"], "alpha", None),
    ("Qwen, matched $z$", ["z2afc_qwen38"], "z", None),
]
plt.rcParams.update({"font.size": 8, "figure.dpi": 200, "savefig.bbox": "tight"})


def trials(runs):
    frames = []
    for run in runs:
        table = pd.read_csv(REPO_ROOT / f"results/experiment1/{run}/trials.csv",
                            low_memory=False)
        frames.append(table[table.kind == "perturbed"])
    return pd.concat(frames, ignore_index=True)


def paired_contrast(runs):
    """Paired contrast S per (layer, family, matching, dose), pooled over runs."""
    p = trials(runs).copy()
    p["signed"] = np.where(p.target_label == "A", p.contrast, -p.contrast)
    g = (p.groupby(KEY, sort=False)
          .agg(total=("signed", "sum"), n=("signed", "size")).reset_index())
    g = g[g.n == 2].copy()
    g["S"] = g.total / 2.0
    g["layer"] = g.layer.astype(int)
    return g


def localization_maps():
    """Block by dose maps of S: both models, both dose rules, one column per family."""
    figure, axes = plt.subplots(len(ROWS), len(FAMILIES), figsize=(7.3, 4.7))
    for r, (row_label, runs, matching, gate) in enumerate(ROWS):
        cells = paired_contrast(runs)
        sub = cells[cells.matching == matching]
        layers, doses = sorted(sub.layer.unique()), sorted(sub.dose.unique())
        grids = {}
        for family in FAMILIES:
            grid = np.full((len(layers), len(doses)), np.nan)
            for (layer, dose), value in (sub[sub.family == family]
                                         .groupby(["layer", "dose"]).S.mean().items()):
                grid[layers.index(layer), doses.index(dose)] = value
            grids[family] = grid
        finite = np.concatenate([g[np.isfinite(g)] for g in grids.values()])
        limit = float(np.percentile(np.abs(finite), 99))
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        # Every grid is geometric with ratio 2, so powers of two keep the ticks short.
        exponents = [int(round(np.log2(d))) for d in doses]
        ticks = list(range(0, len(doses), max(1, int(np.ceil(len(doses) / 5)))))
        for c, family in enumerate(FAMILIES):
            axis = axes[r, c]
            mesh = axis.imshow(np.ma.masked_invalid(grids[family]), aspect="auto",
                               origin="lower", cmap="RdBu_r", norm=norm,
                               interpolation="nearest")
            axis.grid(False)
            axis.set_xticks(ticks)
            axis.set_xticklabels([f"$2^{{{exponents[i]}}}$" for i in ticks], fontsize=6)
            if c == 0:
                marks = list(range(0, len(layers), max(1, len(layers) // 4)))
                axis.set_yticks(marks)
                axis.set_yticklabels([str(layers[i]) for i in marks], fontsize=6)
                axis.set_ylabel(row_label, fontsize=6.5)
            else:
                axis.set_yticks([])
            if r == 0:
                axis.set_title(LABELS[family], fontsize=8)
            if gate is not None and gate in layers:
                axis.axhline(layers.index(gate) - 0.5, color="k", lw=0.9)
        bar = figure.colorbar(mesh, ax=axes[r, :].tolist(), fraction=0.018, pad=0.012)
        bar.ax.tick_params(labelsize=5.5)
    figure.supxlabel("dose per targeted token", fontsize=8, y=0.02)
    figure.supylabel("perturbed decoder block", fontsize=8, x=0.055)
    figure.savefig(OUT / "localization_maps.pdf")
    plt.close(figure)


def localization_curves():
    """Psychometric curves at one block, and the depth profile on both models."""
    figure, (curves, depth) = plt.subplots(2, 1, figsize=(3.4, 4.0))

    block3 = trials(["qwen38_all", "scram-shallow"])
    block3 = block3[(block3.layer == 3) & (block3.matching == "alpha")]
    for family in ["concept", "scrambled", "random", "noise", "dropout"]:
        curve = (block3[block3.family == family]
                 .groupby("dose").correct_adjusted.mean())
        if curve.empty:
            continue
        curves.plot(curve.index, curve.values, "o-", ms=3, lw=1.3,
                    color=COLOURS[family], label=LABELS[family])
    curves.axhline(0.75, color="k", ls=":", lw=0.9)
    curves.axhline(0.5, color="0.6", lw=0.8)
    curves.text(0.97, 0.79, "75%", fontsize=6, color="0.25",
                transform=curves.transAxes, ha="right")
    curves.set_xscale("log", base=2)
    curves.set_xlabel("$\\alpha$ per targeted token", fontsize=8)
    curves.set_ylabel("localization accuracy", fontsize=8)
    curves.set_title("Qwen, block 3, matched $\\alpha$", fontsize=8)
    curves.legend(fontsize=5.8, loc="upper left", handlelength=1.6, borderpad=0.3)
    curves.tick_params(labelsize=6.5)

    # Llama reads the completed ten-concept panel; Qwen was not completed and keeps
    # the four the first sweep named.
    for name, runs, blocks in [("Llama (32 blocks)",
                                ["full32_all", "panel-concepts"], 32),
                               ("Qwen (64 blocks)", ["qwen38_all_layers"], 64)]:
        cells = paired_contrast(runs)
        cells = cells[(cells.matching == "alpha") & (cells.family == "concept")]
        profile = cells.groupby("layer").S.mean()
        depth.plot(100.0 * profile.index / (blocks - 1), profile.values, "o-",
                   ms=3, lw=1.3, label=name)
    depth.axhline(0.0, color="0.6", lw=0.8)
    depth.set_xlabel("relative depth (% of the stack)", fontsize=8)
    depth.set_ylabel("paired contrast $S$", fontsize=8)
    depth.set_title("concept family, matched $\\alpha$", fontsize=8)
    depth.legend(fontsize=6.5, handlelength=1.6)
    depth.tick_params(labelsize=6.5)
    depth.grid(alpha=0.25)
    curves.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(OUT / "localization_curves.pdf")
    plt.close(figure)


def _map_panel(axis, grid, layers, doses, norm, cmap, dose_step, layer_step, ticks=6):
    """One block-by-dose panel of a map figure, with its own tick policy."""
    mesh = axis.imshow(np.ma.masked_invalid(grid), aspect="auto", origin="lower",
                       cmap=cmap, norm=norm, interpolation="nearest")
    axis.grid(False)
    exponents = [int(round(np.log2(d))) for d in doses]
    tick_at = list(range(0, len(doses), dose_step))
    axis.set_xticks(tick_at)
    axis.set_xticklabels([f"$2^{{{exponents[i]}}}$" for i in tick_at], fontsize=ticks)
    marks = list(range(0, len(layers), layer_step))
    axis.set_yticks(marks)
    axis.set_yticklabels([str(layers[i]) for i in marks], fontsize=ticks)
    return mesh


def appendix_localization_maps():
    """The maps of the main figure again, one model per figure, one family per row."""
    for name, columns, height, block_step in [("llama", ROWS[:2], 3.7, 6),
                                              ("qwen", ROWS[2:], 3.0, 2)]:
        figure, axes = plt.subplots(len(FAMILIES), len(columns), figsize=(3.06, height),
                                    squeeze=False)
        panels = {}
        for c, (column_label, runs, matching, gate) in enumerate(columns):
            cells = paired_contrast(runs)
            sub = cells[cells.matching == matching]
            layers, doses = sorted(sub.layer.unique()), sorted(sub.dose.unique())
            for family in FAMILIES:
                grid = np.full((len(layers), len(doses)), np.nan)
                for (layer, dose), value in (sub[sub.family == family]
                                             .groupby(["layer", "dose"]).S.mean().items()):
                    grid[layers.index(layer), doses.index(dose)] = value
                panels[(c, family)] = (grid, layers, doses, gate)
        finite = np.concatenate([g[np.isfinite(g)] for g, _, _, _ in panels.values()])
        # One scale for the whole figure: the two dose rules measure the same contrast.
        limit = float(np.percentile(np.abs(finite), 99))
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        for c, (column_label, _, _, _) in enumerate(columns):
            for r, family in enumerate(FAMILIES):
                grid, layers, doses, gate = panels[(c, family)]
                axis = axes[r, c]
                mesh = _map_panel(axis, grid, layers, doses, norm, "RdBu_r",
                                  max(1, int(np.ceil(len(doses) / 4))), block_step, ticks=5)
                if c == 0:
                    axis.set_ylabel(LABELS[family], fontsize=6)
                else:
                    axis.set_yticklabels([])
                if r == 0:
                    axis.set_title(column_label, fontsize=7)
                if r < len(FAMILIES) - 1:
                    axis.set_xticklabels([])
                if gate is not None and gate in layers:
                    axis.axhline(layers.index(gate) - 0.5, color="k", lw=0.8)
        bar = figure.colorbar(mesh, ax=axes.ravel().tolist(), fraction=0.035, pad=0.03)
        bar.set_label("paired contrast $S$", fontsize=6)
        bar.ax.tick_params(labelsize=5)
        figure.supxlabel("dose per targeted token", fontsize=7, y=0.01)
        figure.savefig(OUT / f"localization_maps_{name}.pdf")
        plt.close(figure)


def degradation_cells():
    """Per-cell classification measures and the detection value paired with each cell."""
    summary = json.loads(EXP4_SUMMARY.read_text())
    return pd.DataFrame(summary["per_dose"]), pd.DataFrame(summary["detection_vs_performance"])


def degradation_maps():
    """Classification accuracy minus sham, block by dose, one family per row."""
    per_dose, _ = degradation_cells()
    sub = per_dose[per_dose.matching == "alpha"]
    layers, doses = sorted(sub.layer.unique()), sorted(sub.dose.unique())
    grids = {}
    for family in FAMILIES:
        grid = np.full((len(layers), len(doses)), np.nan)
        for (layer, dose), value in (sub[sub.family == family]
                                     .groupby(["layer", "dose"]).delta_accuracy.mean().items()):
            grid[layers.index(layer), doses.index(dose)] = value
        grids[family] = grid
    limit = float(np.nanmax(np.abs(np.concatenate([g.ravel() for g in grids.values()]))))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    figure, axes = plt.subplots(len(FAMILIES), 1, figsize=(2.96, 3.8))
    for r, family in enumerate(FAMILIES):
        axis = axes[r]
        # RdBu, not RdBu_r: a loss is negative here, and red stays the colour of an effect.
        mesh = _map_panel(axis, grids[family], layers, doses, norm, "RdBu", 2, 6, ticks=5)
        axis.set_ylabel(LABELS[family], fontsize=6)
        if r < len(FAMILIES) - 1:
            axis.set_xticklabels([])
        if 14 in layers:
            axis.axhline(layers.index(14) - 0.5, color="k", lw=0.8)
    bar = figure.colorbar(mesh, ax=axes.ravel().tolist(), fraction=0.045, pad=0.03)
    bar.set_label("accuracy minus sham", fontsize=6)
    bar.ax.tick_params(labelsize=5)
    figure.supxlabel("$\\alpha$ per targeted token", fontsize=7, y=0.01)
    figure.savefig(OUT / "degradation_maps.pdf")
    plt.close(figure)


def degradation_regimes():
    """Where detection and task damage occur together, and where they come apart."""
    _, paired = degradation_cells()
    sub = paired[paired.matching == "alpha"].copy()
    detected = sub.detection_accuracy >= DETECT
    damaged = sub.classification_delta_accuracy <= -DAMAGE
    # 0 quiet, 1 detected and intact, 2 detected and damaged, 3 damaged and not detected.
    sub["regime"] = np.where(detected & ~damaged, 1,
                             np.where(detected & damaged, 2, np.where(damaged, 3, 0)))
    names = ["neither", "detected, task intact", "detected, task damaged",
             "task damaged, no detection"]
    colours = ListedColormap(["#e8e8e8", "#1b6ca8", "#c8553d", "#e8a33d"])
    layers, doses = sorted(sub.layer.unique()), sorted(sub.dose.unique())
    figure, axes = plt.subplots(len(FAMILIES), 1, figsize=(3.35, 4.0))
    for r, family in enumerate(FAMILIES):
        grid = np.full((len(layers), len(doses)), np.nan)
        for (layer, dose), value in (sub[sub.family == family]
                                     .groupby(["layer", "dose"]).regime.max().items()):
            grid[layers.index(layer), doses.index(dose)] = value
        axis = axes[r]
        _map_panel(axis, grid, layers, doses, Normalize(vmin=-0.5, vmax=3.5), colours,
                   2, 6, ticks=5)
        axis.set_ylabel(LABELS[family], fontsize=6)
        if r < len(FAMILIES) - 1:
            axis.set_xticklabels([])
    counts = sub.regime.value_counts()
    handles = [Patch(facecolor=colours(i), edgecolor="0.4", lw=0.4,
                     label=f"{names[i]} ({int(counts.get(i, 0))} cells)") for i in range(4)]
    figure.legend(handles=handles, ncol=1, fontsize=5.5, loc="lower center",
                  frameon=False, bbox_to_anchor=(0.5, -0.13))
    figure.supxlabel("$\\alpha$ per targeted token", fontsize=7, y=0.01)
    figure.savefig(OUT / "degradation_regimes.pdf")
    plt.close(figure)


def degradation_depth():
    """The two measures by block, on the same amplitudes: both die at the same place."""
    top = [16.0, 32.0, 64.0, 128.0]
    per_dose, _ = degradation_cells()
    damage = per_dose[(per_dose.matching == "alpha") & (per_dose.dose.isin(top))]
    # Cells hold unequal trial counts by family, so the pooled loss is trial-weighted.
    damage = damage.assign(weighted=damage.delta_accuracy * damage.n_trials)
    damage = (damage.groupby(["layer", "family"])
              .apply(lambda g: g.weighted.sum() / g.n_trials.sum(), include_groups=False)
              .rename("delta").reset_index())

    cells = paired_contrast(["full32_all"])
    located = cells[(cells.matching == "alpha") & (cells.dose.isin(top))]
    located = located.groupby(["layer", "family"]).S.mean().rename("S").reset_index()

    figure, (task, loc) = plt.subplots(2, 1, figsize=(COLUMN, 3.3), sharex=True)
    for family in FAMILIES:
        line = damage[damage.family == family].sort_values("layer")
        task.plot(line.layer, line.delta, "o-", ms=2.5, lw=1.2,
                  color=COLOURS[family], label=LABELS[family])
        line = located[located.family == family].sort_values("layer")
        loc.plot(line.layer, line.S, "o-", ms=2.5, lw=1.2, color=COLOURS[family])
    for axis in (task, loc):
        axis.axhline(0.0, color="0.6", lw=0.8)
        axis.axvline(13.5, color="k", lw=0.9, ls="--")
        axis.grid(alpha=0.25)
        axis.tick_params(labelsize=6.5)
    task.set_ylabel("classification accuracy\nminus sham", fontsize=7.5)
    loc.set_ylabel("localization contrast $S$", fontsize=7.5)
    loc.set_xlabel("perturbed decoder block", fontsize=8)
    task.legend(fontsize=6, ncol=2, loc="lower right", handlelength=1.6, borderpad=0.3)
    task.annotate("block 14", xy=(13.5, 1.03), xycoords=("data", "axes fraction"),
                  ha="center", fontsize=6.5, color="0.25")
    figure.tight_layout()
    figure.savefig(OUT / "degradation_depth.pdf")
    plt.close(figure)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    localization_maps()
    localization_curves()
    appendix_localization_maps()
    degradation_maps()
    degradation_regimes()
    degradation_depth()
    # The calibration figure is already in English and unchanged.
    shutil.copyfile(LEGACY_FIGURES / "experiment_0_scales_sd.png",
                    OUT / "experiment_0_scales_sd.png")
    print("wrote", ", ".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
