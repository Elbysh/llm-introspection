#!/usr/bin/env python3
"""
Layer-resolved views of an Experiment 1 sweep: heatmaps, 3D surfaces, profiles.

experiment1_psychometrics.py already writes one accuracy-vs-dose figure per layer,
which answers "what happens at layer L" but hides the depth structure: the reader has
to flip through 32 files to see that detectability collapses somewhere in the middle
of the stack. These figures put the layer on an axis instead.

Four views, each produced for both matchings and, on request, both metrics:

  heatmap_<matching>_<metric>.png     layer x dose accuracy, one panel per family
  surface3d_<matching>_<metric>.png   the same grid as a 3D surface
  profile_<matching>_<metric>.png     accuracy vs layer at the top doses, and the
                                      concept-minus-controls gap
  localization_<matching>.png         the paired contrast S by block - the metric to
                                      lead with, see plot_localization
  effect_<matching>.png               layer x dose mean |adjusted contrast|, the
                                      control for "is a flat band just signal decay?"

The first three read summary.json. The last one needs the per-trial contrasts and so
reads trials.csv; it is skipped when that file is absent.

    python code/analysis/plot_experiment1_layer_maps.py \
        --run_dir results/experiment1/full32_all

Accuracy is diverging around the 0.5 forced-choice guessing rate of doc 5.10, so it
gets a two-hue scale with a neutral midpoint pinned there. The mean absolute contrast
is a magnitude with no meaningful midpoint, so it gets a one-hue ramp.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize

REPO_ROOT = Path(__file__).resolve().parents[2]

FAMILIES = ("concept", "random", "noise", "dropout")
MATCHINGS = ("alpha", "z")
CHANCE = 0.5

FAMILY_LABELS = {
    "concept": "concept direction",
    "random": "fixed random direction",
    "noise": "renewed noise",
    "dropout": "gaussian dropout",
}
MATCHING_LABELS = {
    "alpha": r"$\alpha$ per targeted token",
    "z": r"$z = \alpha\,/\,s(\mathrm{layer},\,\mathrm{direction})$",
}

# Palette slots 1-4 of the reference categorical theme, taken in order. Concept is the
# family under test and holds slot 1; the three controls follow.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "page": "#f9f9f7",
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "diverging": ("#0d366b", "#2a78d6", "#f0efec", "#d03b3b", "#6d1c1c"),
        "sequential": ("#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"),
        "series": {"concept": "#2a78d6", "random": "#eb6834",
                   "noise": "#1baf7a", "dropout": "#eda100"},
        "missing": "#e1e0d9",
    },
    "dark": {
        "surface": "#1a1a19",
        "page": "#0d0d0d",
        "ink": "#ffffff",
        "ink_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "diverging": ("#9ec5f4", "#3987e5", "#383835", "#d03b3b", "#e87b7b"),
        "sequential": ("#184f95", "#256abf", "#3987e5", "#86b6ef", "#cde2fb"),
        "series": {"concept": "#3987e5", "random": "#d95926",
                   "noise": "#199e70", "dropout": "#c98500"},
        "missing": "#2c2c2a",
    },
}


def make_colormaps(theme):
    """Diverging blue<->red with a neutral midpoint, and a one-hue blue ramp."""
    diverging = LinearSegmentedColormap.from_list("acc_diverging", theme["diverging"])
    diverging.set_bad(theme["missing"])
    sequential = LinearSegmentedColormap.from_list("effect_sequential", theme["sequential"])
    sequential.set_bad(theme["missing"])
    return diverging, sequential


def style_axes(axis, theme, grid=True):
    axis.set_facecolor(theme["surface"])
    for spine in axis.spines.values():
        spine.set_color(theme["axis"])
        spine.set_linewidth(0.8)
    axis.tick_params(colors=theme["muted"], labelcolor=theme["ink_secondary"], width=0.8)
    if grid:
        axis.grid(True, color=theme["grid"], linewidth=0.8, alpha=0.9)
        axis.set_axisbelow(True)


# --------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------

def load_summary(run_dir):
    path = run_dir / "summary.json"
    if not path.exists():
        raise SystemExit(f"No summary.json in {run_dir}")
    with path.open() as handle:
        return json.load(handle)


def grid_from_per_dose(per_dose, family, matching, metric, layers, doses):
    """A (layer, dose) accuracy matrix, NaN where the run holds no cell.

    full32_all is stitched from two jobs and has no concept cell at block 31, so a
    missing cell is a normal outcome and must stay visibly empty rather than be
    interpolated or silently read as chance.
    """
    index = {(row["layer"], row["dose"]): row[f"accuracy_{metric}"]
             for row in per_dose
             if row["family"] == family and row["matching"] == matching}
    grid = np.full((len(layers), len(doses)), np.nan)
    for i, layer in enumerate(layers):
        for j, dose in enumerate(doses):
            value = index.get((layer, dose))
            if value is not None:
                grid[i, j] = value
    return grid


def axis_extents(summary, matching):
    layers = sorted(summary["layers"])
    doses = sorted(summary["dose_grid"][matching])
    return layers, doses


def symmetric_norm(grids, center=CHANCE):
    """A TwoSlopeNorm pinned at `center` and spanning every panel of one figure."""
    finite = np.concatenate([g[np.isfinite(g)].ravel() for g in grids])
    low = float(min(finite.min(), center - 1e-3))
    high = float(max(finite.max(), center + 1e-3))
    return TwoSlopeNorm(vmin=low, vcenter=center, vmax=high)


def dose_tick_labels(doses):
    return [f"{d:g}" for d in doses]


def thin(labels, keep_every):
    """Blank all but every `keep_every`-th label; 3D axes have no room for all of them."""
    return [label if index % keep_every == 0 else "" for index, label in enumerate(labels)]


def detect_gate(summary, matching, metric, layers, threshold=0.55, n_top_doses=4,
                n_sigma=3.0):
    """The block where the contiguous band of localizable depths ends.

    A block counts as live when some family beats `threshold` *and* clears chance by
    `n_sigma` binomial standard errors. Two guards, because each catches a different
    way of being fooled: without the significance test a cell of the flat band that
    reaches 0.555 by luck (2.4 SE on the trials behind it) sets the gate ten blocks too
    deep, and without contiguity an isolated significant excursion well inside the flat
    band does the same. What the figures mark is where the run of live blocks stops.

    The collapse depth is a result, not a constant of the model, so it is read off the
    data instead of being written into the figure. Returns None when the band never
    stops, which is what a run with signal all the way up should give.
    """
    doses = sorted(summary["dose_grid"][matching])[-n_top_doses:]
    live = np.zeros(len(layers), dtype=bool)
    for family in FAMILIES:
        values, counts = pooled_by_layer(summary["per_dose"], family, matching, metric,
                                         layers, doses)
        with np.errstate(invalid="ignore", divide="ignore"):
            error = np.sqrt(values * (1.0 - values) / counts)
            significant = (values - CHANCE) >= n_sigma * error
        beats = np.nan_to_num(values, nan=0.0) >= threshold
        live |= beats & np.nan_to_num(significant, nan=False)
    dead = np.flatnonzero(~live)
    if dead.size == 0 or dead[0] == 0:
        return None
    return layers[dead[0]]


def draw_gate(axis, layers, gate, theme, orientation="y", as_index=True):
    """Mark the collapse depth, in data or index coordinates depending on the axes."""
    if gate is None:
        return
    position = layers.index(gate) - 0.5 if as_index else gate - 0.5
    line = axis.axhline if orientation == "y" else axis.axvline
    line(position, color=theme["ink"], linewidth=1.2, zorder=5)


# --------------------------------------------------------------------------------
# Heatmaps
# --------------------------------------------------------------------------------

def plot_heatmaps(summary, matching, metric, theme_name, output_dir, gate_threshold=0.55):
    theme = THEMES[theme_name]
    diverging, _ = make_colormaps(theme)
    layers, doses = axis_extents(summary, matching)
    grids = {family: grid_from_per_dose(summary["per_dose"], family, matching,
                                        metric, layers, doses)
             for family in FAMILIES}
    norm = symmetric_norm(list(grids.values()))
    gate = detect_gate(summary, matching, metric, layers, gate_threshold)

    figure, axes = plt.subplots(1, 4, figsize=(17, 8.5), sharey=True)
    figure.patch.set_facecolor(theme["page"])
    for column, family in enumerate(FAMILIES):
        axis = axes[column]
        style_axes(axis, theme, grid=False)
        mesh = axis.imshow(np.ma.masked_invalid(grids[family]), aspect="auto",
                           origin="lower", cmap=diverging, norm=norm,
                           interpolation="nearest")
        axis.set_xticks(range(len(doses)))
        axis.set_xticklabels(dose_tick_labels(doses), rotation=90, fontsize=8)
        draw_gate(axis, layers, gate, theme)
        axis.set_title(FAMILY_LABELS[family], color=theme["ink"], fontsize=11, pad=8)
        axis.set_xlabel(MATCHING_LABELS[matching], color=theme["ink_secondary"], fontsize=9)
        if column == 0 and gate is not None:
            axis.annotate(f"block {gate}: the localizable band stops here",
                          (0.0, layers.index(gate) - 0.5), xytext=(3, 4),
                          textcoords="offset points", color=theme["ink"], fontsize=8)
        if column == 0:
            axis.set_yticks(range(0, len(layers), 2))
            axis.set_yticklabels([str(layers[i]) for i in range(0, len(layers), 2)],
                                 fontsize=8)
            axis.set_ylabel("decoder block the perturbation is applied at",
                            color=theme["ink_secondary"], fontsize=10)

    bar = figure.colorbar(mesh, ax=axes, fraction=0.02, pad=0.02, extend="both")
    bar.set_label(f"2AFC localization accuracy ({metric})", color=theme["ink_secondary"])
    bar.ax.tick_params(colors=theme["muted"], labelcolor=theme["ink_secondary"])
    bar.outline.set_edgecolor(theme["axis"])
    bar.ax.axhline(norm(CHANCE), color=theme["ink"], linewidth=1.0)

    figure.suptitle(
        f"Experiment 1 - localization accuracy by block and dose, {matching}-matched",
        color=theme["ink"], fontsize=14)
    path = output_dir / f"heatmap_{matching}_{metric}_{theme_name}.png"
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor=theme["page"])
    plt.close(figure)
    return path


# --------------------------------------------------------------------------------
# 3D surfaces
# --------------------------------------------------------------------------------

def plot_surfaces(summary, matching, metric, theme_name, output_dir, elev=28, azim=-58):
    theme = THEMES[theme_name]
    diverging, _ = make_colormaps(theme)
    layers, doses = axis_extents(summary, matching)
    grids = {family: grid_from_per_dose(summary["per_dose"], family, matching,
                                        metric, layers, doses)
             for family in FAMILIES}
    norm = symmetric_norm(list(grids.values()))

    # One z range for all four panels, or the eye reads panel height as accuracy.
    finite = np.concatenate([g[np.isfinite(g)].ravel() for g in grids.values()])
    z_low = float(np.floor(finite.min() * 20) / 20)
    z_high = float(np.ceil(finite.max() * 20) / 20)

    dose_index, layer_index = np.meshgrid(np.arange(len(doses)), np.array(layers, float))
    figure = plt.figure(figsize=(15, 10))
    figure.patch.set_facecolor(theme["page"])
    for position, family in enumerate(FAMILIES, start=1):
        axis = figure.add_subplot(2, 2, position, projection="3d")
        grid = grids[family]
        # plot_surface drops a whole quad on a NaN corner; filling missing cells with
        # chance keeps the mesh closed, and the flat patch reads as "no signal here"
        # rather than as a hole. The heatmap is the view that shows the gap honestly.
        filled = np.where(np.isfinite(grid), grid, CHANCE)
        colors = diverging(norm(filled))
        # With facecolors set, plot_surface forwards one colour array as both the face
        # and the edge colour, so the separating hairline has to be applied afterwards.
        surface = axis.plot_surface(dose_index, layer_index, filled, facecolors=colors,
                                    rstride=1, cstride=1, antialiased=True, shade=False)
        surface.set_edgecolor(theme["surface"])
        surface.set_linewidth(0.2)
        chance_plane = np.full_like(filled, CHANCE)
        axis.plot_surface(dose_index, layer_index, chance_plane, color=theme["muted"],
                          alpha=0.18, linewidth=0, antialiased=True, shade=False)

        axis.set_xticks(range(len(doses)))
        axis.set_xticklabels(thin(dose_tick_labels(doses), 2), fontsize=8)
        axis.set_yticks([layer for layer in layers if layer % 4 == 0])
        axis.set_xlabel(MATCHING_LABELS[matching], color=theme["ink_secondary"],
                        fontsize=9, labelpad=10)
        axis.set_ylabel("decoder block", color=theme["ink_secondary"], fontsize=9,
                        labelpad=8)
        axis.set_zlabel(f"accuracy ({metric})", color=theme["ink_secondary"],
                        fontsize=9, labelpad=4)
        axis.set_zlim(z_low, z_high)
        axis.set_title(FAMILY_LABELS[family], color=theme["ink"], fontsize=11, y=0.93)
        axis.set_box_aspect((1.5, 1.5, 0.85), zoom=1.05)
        axis.view_init(elev=elev, azim=azim)
        axis.set_facecolor(theme["page"])
        for pane_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
            pane_axis.set_pane_color((0, 0, 0, 0))
            pane_axis._axinfo["grid"]["color"] = theme["grid"]
        axis.tick_params(colors=theme["muted"], labelcolor=theme["ink_secondary"])

    # subplots_adjust would not move a colorbar attached to the panels, so the panel
    # box is fixed first and the bar then placed in the strip left free beside it.
    figure.subplots_adjust(left=0.0, right=0.86, top=0.93, bottom=0.03,
                           wspace=0.0, hspace=0.05)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=diverging)
    bar_axes = figure.add_axes((0.90, 0.22, 0.014, 0.56))
    bar = figure.colorbar(mappable, cax=bar_axes, extend="both")
    bar.set_label(f"2AFC localization accuracy ({metric}); the flat plane is chance",
                  color=theme["ink_secondary"])
    bar.ax.tick_params(colors=theme["muted"], labelcolor=theme["ink_secondary"])
    bar.outline.set_edgecolor(theme["axis"])

    figure.suptitle(
        f"Experiment 1 - accuracy surface over block and dose, {matching}-matched",
        color=theme["ink"], fontsize=14, y=0.97)
    path = output_dir / f"surface3d_{matching}_{metric}_{theme_name}.png"
    figure.savefig(path, dpi=150, facecolor=theme["page"])
    plt.close(figure)
    return path


# --------------------------------------------------------------------------------
# Layer profiles
# --------------------------------------------------------------------------------

def pooled_by_layer(per_dose, family, matching, metric, layers, top_doses):
    """Trial-weighted accuracy over the `top_doses` largest doses, per layer."""
    keep = set(top_doses)
    sums = {layer: [0.0, 0] for layer in layers}
    for row in per_dose:
        if (row["family"], row["matching"]) != (family, matching):
            continue
        if row["dose"] not in keep:
            continue
        total = row["n_trials"]
        sums[row["layer"]][0] += row[f"accuracy_{metric}"] * total
        sums[row["layer"]][1] += total
    values, counts = [], []
    for layer in layers:
        accumulated, total = sums[layer]
        values.append(accumulated / total if total else np.nan)
        counts.append(total)
    return np.array(values), np.array(counts, dtype=float)


def plot_profiles(summary, matching, metric, theme_name, output_dir, n_top_doses=4):
    theme = THEMES[theme_name]
    layers, doses = axis_extents(summary, matching)
    top_doses = doses[-n_top_doses:]

    pooled = {family: pooled_by_layer(summary["per_dose"], family, matching, metric,
                                      layers, top_doses)
              for family in FAMILIES}

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                                         gridspec_kw={"height_ratios": [2, 1]})
    figure.patch.set_facecolor(theme["page"])
    for axis in (top, bottom):
        style_axes(axis, theme)

    peaks = []
    for family in FAMILIES:
        values, counts = pooled[family]
        color = theme["series"][family]
        # Binomial standard error on the pooled cell; the bands overlap enough that
        # the eye needs them to tell a real family gap from sampling noise.
        with np.errstate(invalid="ignore", divide="ignore"):
            error = np.sqrt(values * (1.0 - values) / counts)
        top.plot(layers, values, color=color, linewidth=2.0, marker="o",
                 markersize=4, markeredgecolor=theme["surface"], markeredgewidth=1.0,
                 label=FAMILY_LABELS[family], zorder=3)
        top.fill_between(layers, values - error, values + error, color=color,
                         alpha=0.15, linewidth=0, zorder=2)
        peaks.append((family, color, values))

    # Direct-label at each series peak, not at the endpoint: past the collapse the four
    # curves sit on top of each other and four labels there are one illegible smear.
    placed = []
    for family, color, values in sorted(peaks, key=lambda item: -np.nanmax(item[2])):
        if not np.isfinite(values).any():
            continue
        index = int(np.nanargmax(values))
        x, y = layers[index], values[index]
        if any(abs(x - other_x) < 4 and abs(y - other_y) < 0.035
               for other_x, other_y in placed):
            continue
        placed.append((x, y))
        top.annotate(FAMILY_LABELS[family], (x, y), xytext=(0, 9),
                     textcoords="offset points", color=color, fontsize=8,
                     ha="center", va="bottom")
    top.axhline(CHANCE, color=theme["ink_secondary"], linewidth=1.0)
    top.annotate("chance", (layers[0], CHANCE), xytext=(2, -11),
                 textcoords="offset points", color=theme["ink_secondary"], fontsize=8,
                 ha="left", va="top")
    top.set_ylabel(f"accuracy ({metric}), pooled over the {n_top_doses} largest doses",
                   color=theme["ink_secondary"], fontsize=10)
    top.set_ylim(0.35, 1.0)
    legend = top.legend(loc="upper right", fontsize=9, framealpha=0.0)
    for text in legend.get_texts():
        text.set_color(theme["ink_secondary"])

    concept, concept_n = pooled["concept"]
    controls = np.nanmean(np.vstack([pooled[f][0] for f in ("random", "noise", "dropout")]),
                          axis=0)
    gap = concept - controls
    bottom.bar(layers, gap, color=theme["series"]["concept"], width=0.7, zorder=3)
    bottom.axhline(0.0, color=theme["ink_secondary"], linewidth=1.0)
    bottom.set_ylabel("concept minus\nmean of the three controls",
                      color=theme["ink_secondary"], fontsize=10)
    bottom.set_xlabel("decoder block the perturbation is applied at",
                      color=theme["ink_secondary"], fontsize=10)
    bottom.set_xticks(layers)
    bottom.set_xticklabels([str(layer) for layer in layers], fontsize=8)
    gate = detect_gate(summary, matching, metric, layers)
    for axis in (top, bottom):
        draw_gate(axis, layers, gate, theme, orientation="x", as_index=False)
    if gate is not None:
        top.annotate(f"block {gate}", (gate - 0.5, 0.97), xytext=(4, 0),
                     textcoords="offset points", color=theme["ink"], fontsize=9,
                     va="top")

    figure.suptitle(
        f"Experiment 1 - depth profile at the top doses, {matching}-matched",
        color=theme["ink"], fontsize=14)
    figure.tight_layout()
    path = output_dir / f"profile_{matching}_{metric}_{theme_name}.png"
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor=theme["page"])
    plt.close(figure)
    return path


# --------------------------------------------------------------------------------
# Paired localization contrast
# --------------------------------------------------------------------------------

def plot_localization(summary, matching, theme_name, output_dir):
    """S = (contrast when A is targeted - contrast when B is targeted) / 2, by block.

    The metric to lead with. `accuracy_adjusted` inherits the model's letter-A bias:
    the perturbation pulls the A/B gap down whatever it hits, which scores as a hit on
    every B-targeted trial and a miss on every A-targeted one (0.753 against 0.481 in
    the live band). S differences the two targets of an otherwise identical pair, so
    any shift that does not depend on which sentence was hit cancels exactly. It is
    already in summary.json as `mean_localization_contrast`.
    """
    theme = THEMES[theme_name]
    layers = sorted(summary["layers"])
    series = {}
    for family in FAMILIES:
        index = {row["layer"]: row["mean_localization_contrast"]
                 for row in summary["curves"]
                 if (row["family"], row["matching"], row["metric"]) == (family, matching,
                                                                       "adjusted")}
        series[family] = np.array([index.get(layer, np.nan) for layer in layers],
                                  dtype=float)

    figure, (linear, log) = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                                         gridspec_kw={"height_ratios": [2, 1]})
    figure.patch.set_facecolor(theme["page"])
    for axis in (linear, log):
        style_axes(axis, theme)
        for family in FAMILIES:
            axis.plot(layers, series[family], color=theme["series"][family],
                      linewidth=2.0, marker="o", markersize=4,
                      markeredgecolor=theme["surface"], markeredgewidth=1.0,
                      label=FAMILY_LABELS[family], zorder=3)
        axis.axhline(0.0, color=theme["ink_secondary"], linewidth=1.0)
        draw_gate(axis, layers, detect_gate(summary, matching, "adjusted", layers),
                  theme, orientation="x", as_index=False)

    # The flat band is three orders of magnitude below the live one; a symmetric log
    # scale is the only way to show both without a broken axis.
    log.set_yscale("symlog", linthresh=0.01)
    log.set_ylabel("same, symlog", color=theme["ink_secondary"], fontsize=10)
    log.set_xlabel("decoder block the perturbation is applied at",
                   color=theme["ink_secondary"], fontsize=10)
    log.set_xticks(layers)
    log.set_xticklabels([str(layer) for layer in layers], fontsize=8)

    linear.set_ylabel("paired localization contrast S (logits)",
                      color=theme["ink_secondary"], fontsize=10)
    legend = linear.legend(loc="upper right", fontsize=9, framealpha=0.0)
    for text in legend.get_texts():
        text.set_color(theme["ink_secondary"])
    for family in ("concept", "random"):
        values = series[family]
        if np.isfinite(values).any():
            index = int(np.nanargmax(values))
            linear.annotate(FAMILY_LABELS[family], (layers[index], values[index]),
                            xytext=(0, 9), textcoords="offset points",
                            color=theme["series"][family], fontsize=8,
                            ha="center", va="bottom")

    figure.suptitle(
        f"Experiment 1 - paired localization contrast by block, {matching}-matched",
        color=theme["ink"], fontsize=14)
    figure.tight_layout()
    path = output_dir / f"localization_{matching}_{theme_name}.png"
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor=theme["page"])
    plt.close(figure)
    return path


# --------------------------------------------------------------------------------
# Effect size, from the per-trial contrasts
# --------------------------------------------------------------------------------

def effect_grid(trials_path, matching, layers, doses):
    """Mean |adjusted contrast| per (layer, dose), pooled over families.

    Read with the csv module rather than pandas: the file is tens of megabytes and
    only four columns are needed, so there is no reason to materialise a frame.
    """
    import csv

    sums = {}
    with trials_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["kind"] != "perturbed" or row["matching"] != matching:
                continue
            adjusted = row["adjusted_contrast"]
            if not adjusted:
                continue
            key = (int(row["layer"]), float(row["dose"]))
            entry = sums.setdefault(key, [0.0, 0])
            entry[0] += abs(float(adjusted))
            entry[1] += 1
    grid = np.full((len(layers), len(doses)), np.nan)
    for i, layer in enumerate(layers):
        for j, dose in enumerate(doses):
            entry = sums.get((layer, dose))
            if entry and entry[1]:
                grid[i, j] = entry[0] / entry[1]
    return grid


def plot_effect(summary, trials_path, matching, theme_name, output_dir):
    theme = THEMES[theme_name]
    _, sequential = make_colormaps(theme)
    layers, doses = axis_extents(summary, matching)
    grid = effect_grid(trials_path, matching, layers, doses)

    figure, (heat, profile) = plt.subplots(1, 2, figsize=(13, 8.5),
                                           gridspec_kw={"width_ratios": [2, 1]},
                                           sharey=True)
    figure.patch.set_facecolor(theme["page"])
    style_axes(heat, theme, grid=False)
    style_axes(profile, theme)

    finite = grid[np.isfinite(grid)]
    norm = Normalize(vmin=0.0, vmax=float(finite.max()) if finite.size else 1.0)
    mesh = heat.imshow(np.ma.masked_invalid(grid), aspect="auto", origin="lower",
                       cmap=sequential, norm=norm, interpolation="nearest")
    heat.set_xticks(range(len(doses)))
    heat.set_xticklabels(dose_tick_labels(doses), rotation=90, fontsize=8)
    heat.set_yticks(range(0, len(layers), 2))
    heat.set_yticklabels([str(layers[i]) for i in range(0, len(layers), 2)], fontsize=8)
    heat.set_xlabel(MATCHING_LABELS[matching], color=theme["ink_secondary"], fontsize=10)
    heat.set_ylabel("decoder block the perturbation is applied at",
                    color=theme["ink_secondary"], fontsize=10)
    heat.set_title("mean |sham-adjusted logit contrast|", color=theme["ink"], fontsize=11)
    gate = detect_gate(summary, matching, "adjusted", layers)
    draw_gate(heat, layers, gate, theme)
    draw_gate(profile, layers, gate, theme)

    bar = figure.colorbar(mesh, ax=heat, fraction=0.04, pad=0.02)
    bar.set_label("logit units", color=theme["ink_secondary"])
    bar.ax.tick_params(colors=theme["muted"], labelcolor=theme["ink_secondary"])
    bar.outline.set_edgecolor(theme["axis"])

    # The two panels share the y axis, and imshow puts it in row indices, so the line
    # is drawn against indices too rather than against the block numbers themselves.
    largest = np.array([grid[i, -1] for i in range(len(layers))])
    profile.plot(largest, np.arange(len(layers)), color=theme["series"]["concept"],
                 linewidth=2.0,
                 marker="o", markersize=4, markeredgecolor=theme["surface"],
                 markeredgewidth=1.0)
    profile.set_xlabel("at the largest dose", color=theme["ink_secondary"], fontsize=10)
    profile.set_title("how far the perturbation carries",
                      color=theme["ink"], fontsize=11)

    figure.suptitle(
        f"Experiment 1 - size of the effect on the answer logits, {matching}-matched",
        color=theme["ink"], fontsize=14)
    figure.tight_layout()
    path = output_dir / f"effect_{matching}_{theme_name}.png"
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor=theme["page"])
    plt.close(figure)
    return path


# --------------------------------------------------------------------------------
# Text report
# --------------------------------------------------------------------------------

def write_report(summary, matching, metric, layers, handle, n_top_doses=4):
    doses = sorted(summary["dose_grid"][matching])
    top_doses = doses[-n_top_doses:]
    pooled = {family: pooled_by_layer(summary["per_dose"], family, matching, metric,
                                      layers, top_doses)[0]
              for family in FAMILIES}
    controls = np.nanmean(np.vstack([pooled[f] for f in ("random", "noise", "dropout")]),
                          axis=0)
    print(f"\n{matching}-matched, {metric} accuracy pooled over doses "
          f"{', '.join(f'{d:g}' for d in top_doses)}", file=handle)
    header = f"{'block':>5} " + " ".join(f"{f:>8}" for f in FAMILIES) + f" {'gap':>8}"
    print(header, file=handle)
    for index, layer in enumerate(layers):
        cells = " ".join(
            "     n/a" if not np.isfinite(pooled[f][index]) else f"{pooled[f][index]:8.3f}"
            for f in FAMILIES)
        gap = pooled["concept"][index] - controls[index]
        gap_cell = "     n/a" if not np.isfinite(gap) else f"{gap:8.3f}"
        print(f"{layer:>5} {cells} {gap_cell}", file=handle)


# --------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Layer-resolved heatmaps, 3D surfaces and depth profiles for an "
                    "Experiment 1 sweep.")
    parser.add_argument("--run_dir", type=Path,
                        default=REPO_ROOT / "results" / "experiment1" / "full32_all",
                        help="Directory holding summary.json and trials.csv")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Where the figures go (default: <run_dir>/layer_maps)")
    parser.add_argument("--matchings", nargs="+", default=list(MATCHINGS),
                        choices=list(MATCHINGS))
    parser.add_argument("--metrics", nargs="+", default=["adjusted"],
                        choices=["raw", "adjusted"],
                        help="The raw metric is pinned by the model's A/B bias; the "
                             "sham-adjusted one is what isolates the intervention.")
    parser.add_argument("--themes", nargs="+", default=["light"],
                        choices=list(THEMES))
    parser.add_argument("--n_top_doses", type=int, default=4,
                        help="How many of the largest doses the depth profile pools")
    parser.add_argument("--elev", type=float, default=28.0,
                        help="3D elevation angle")
    parser.add_argument("--azim", type=float, default=-58.0,
                        help="3D azimuth angle")
    parser.add_argument("--skip_effect", action="store_true",
                        help="Skip the figure that reads trials.csv")
    parser.add_argument("--report", type=Path, default=None,
                        help="Also write the pooled depth table here (default: stdout)")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    summary = load_summary(run_dir)
    output_dir = (args.output_dir or run_dir / "layer_maps").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run      {run_dir}")
    print(f"Model    {summary['model']}")
    print(f"Trials   {summary['n_trials']}")
    print(f"Blocks   {min(summary['layers'])}-{max(summary['layers'])}")
    if summary.get("notes"):
        print(f"Notes    {summary['notes']}")

    trials_path = run_dir / "trials.csv"
    for theme_name in args.themes:
        for matching in args.matchings:
            for metric in args.metrics:
                print(f"Saved {plot_heatmaps(summary, matching, metric, theme_name, output_dir)}")
                print(f"Saved {plot_surfaces(summary, matching, metric, theme_name, output_dir, args.elev, args.azim)}")
                print(f"Saved {plot_profiles(summary, matching, metric, theme_name, output_dir, args.n_top_doses)}")
            print(f"Saved {plot_localization(summary, matching, theme_name, output_dir)}")
            if not args.skip_effect:
                if trials_path.exists():
                    print(f"Saved {plot_effect(summary, trials_path, matching, theme_name, output_dir)}")
                else:
                    print(f"No trials.csv in {run_dir}; skipping the effect-size figure")

    layers = sorted(summary["layers"])
    if args.report:
        with args.report.open("w") as handle:
            for matching in args.matchings:
                for metric in args.metrics:
                    write_report(summary, matching, metric, layers, handle, args.n_top_doses)
        print(f"Saved {args.report}")
    else:
        for matching in args.matchings:
            for metric in args.metrics:
                write_report(summary, matching, metric, layers, sys.stdout, args.n_top_doses)


if __name__ == "__main__":
    main()
