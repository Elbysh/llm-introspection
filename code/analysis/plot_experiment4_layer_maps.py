#!/usr/bin/env python3
"""
Layer-resolved views of an Experiment 4 sweep: heatmaps, 3D surfaces, profiles, and
the joint detection/degradation map.

experiment4_task_degradation.py writes one accuracy-vs-dose figure per block, which
answers "what happens at block L" but hides the depth structure: the reader has to
flip through 31 files to see that the task survives untouched above the middle of the
stack. These figures put the block on an axis instead.

The theme, the colour ramps and the axis styling are imported from
plot_experiment1_layer_maps rather than restated, so the two experiments' figures are
comparable at a glance and a palette change lands on both.

  heatmap_<matching>_<field>.png    block x dose, one panel per family
  surface3d_<matching>_<field>.png  the same grid as a 3D surface
  profile_<matching>_<field>.png    the field against block at the top doses, and the
                                    concept-minus-controls gap
  regimes_<matching>.png            block x dose, four-way: is the perturbation
                                    detected, is the task damaged, or both

What differs from Experiment 1 is the quantity being mapped. There, accuracy diverges
around the 0.5 forced-choice guessing rate. Here the primary measures of doc 8.6 are
*changes* from the sham and diverge around zero, and their damaging direction is
negative, so those fields get the reversed ramp: red is loss.

The regimes figure is the one that answers H4, and it needs a run whose summary.json
carries `detection_vs_performance` -- that is, one that has been through

    python code/experiments/experiment4_task_degradation.py \
        --reanalyze <run_dir> --experiment1_summary <an Experiment 1 run>

It is skipped, with a message, when that join is absent.

    python code/analysis/plot_experiment4_layer_maps.py \
        --run_dir results/experiment4/main
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Imports matplotlib and selects the Agg backend, which is what a batch figure needs.
from plot_experiment1_layer_maps import (  # noqa: E402
    FAMILY_LABELS,
    MATCHING_LABELS,
    THEMES,
    dose_tick_labels,
    make_colormaps,
    style_axes,
    thin,
)
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

FAMILIES = ("concept", "random", "noise", "dropout")
MATCHINGS = ("alpha", "z")

# Figure container. PNG is right for reading a sweep on screen; the LaTeX reports carry
# vector PDFs, and a heatmap's imshow raster and a surface's quads both embed cleanly.
FORMAT = "png"

# Geometry. The screen figures are drawn wide and read at full size. A manuscript is the
# other regime: the figure must be drawn at its *final* width, because \includegraphics
# shrinking a 17-inch canvas to a 6.75-inch column divides every font size by the same
# factor and turns 8 pt type into 3 pt. Drawing at final size means the type sizes tuned
# for the wide canvas are now proportionally too large, so FONT_SCALE goes below one
# rather than above it -- scaling it up instead makes the labels collide.
FIG_SCALE = 1.0
FONT_SCALE = 1.0


def fs(size):
    """A font size in points, scaled for the target medium."""
    return size * FONT_SCALE


def figsize(width, height):
    return (width * FIG_SCALE, height * FIG_SCALE)


# Per-field: what to call it, where its neutral value sits, and which ramp reads it.
#
# `center=0.0` means the field is a change from the sham and its sign is meaningful.
# `reverse=True` flips the imported ramp so that the damaging direction -- negative,
# for every change measure here -- is the red end. Without it a collapsing task would
# be painted in the same blue the Experiment 1 figures use for strong detection.
# `center=None` means a magnitude with no meaningful midpoint, which gets the one-hue
# sequential ramp instead; `center="sham"` pins the midpoint on the run's own sham
# accuracy, since that, not chance, is the level an undamaged cell sits at.
FIELDS = {
    "delta_accuracy": {
        "label": "classification accuracy, minus sham",
        "center": 0.0, "reverse": True, "kind": "diverging",
    },
    "mean_margin_delta": {
        "label": "margin of the correct letter, minus sham (logits)",
        "center": 0.0, "reverse": True, "kind": "diverging",
    },
    "accuracy": {
        "label": "classification accuracy",
        "center": "sham", "reverse": False, "kind": "diverging",
    },
    "invalid_rate": {
        "label": "invalid answer rate",
        "center": None, "reverse": False, "kind": "sequential",
    },
    "mean_js_divergence": {
        "label": "JS divergence from the sham output (bits)",
        "center": None, "reverse": False, "kind": "sequential",
    },
}

# Regimes of the joint map. Order is the drawing order of the legend.
REGIME_LABELS = (
    "quiet: no detection, task intact",
    "detected, task intact",
    "detected, task damaged",
    "task damaged, no detection",
)


# --------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------

def load_summary(run_dir):
    path = run_dir / "summary.json"
    if not path.exists():
        raise SystemExit(f"No summary.json in {run_dir}")
    with path.open() as handle:
        return json.load(handle)


def axis_extents(summary, matching):
    return sorted(summary["layers"]), sorted(summary["dose_grid"][matching])


def grid_from_per_dose(per_dose, family, matching, field, layers, doses):
    """A (block, dose) matrix of `field`, NaN where the run holds no cell.

    A missing cell stays visibly empty rather than being interpolated or read as zero:
    a block the sweep never reached and a block where nothing happened are different
    claims.
    """
    index = {(row["layer"], row["dose"]): row[field]
             for row in per_dose
             if row["family"] == family and row["matching"] == matching}
    grid = np.full((len(layers), len(doses)), np.nan)
    for i, layer in enumerate(layers):
        for j, dose in enumerate(doses):
            value = index.get((layer, dose))
            if value is not None:
                grid[i, j] = float(value)
    return grid


def field_norm(grids, spec, summary):
    """Colour normalization for one figure, shared by all four of its panels."""
    finite = np.concatenate([g[np.isfinite(g)].ravel() for g in grids])
    low, high = float(finite.min()), float(finite.max())
    center = spec["center"]
    if center == "sham":
        center = float(summary["sham"]["accuracy"])
    if center is None:
        return Normalize(vmin=low, vmax=high)
    center = float(center)
    # Symmetric about the centre, deliberately. These change measures are almost
    # entirely one-sided -- the perturbation damages the task and essentially never
    # improves it -- so a TwoSlopeNorm fitted to the data would stretch a +2% sampling
    # fluctuation across the whole positive half of the ramp and paint noise at the
    # deep blocks as vividly as real damage at the shallow ones. Equal half-ranges
    # cost nothing on the negative side, which still owns half the ramp either way.
    half = max(abs(low - center), abs(high - center), 1e-3)
    return TwoSlopeNorm(vmin=center - half, vcenter=center, vmax=center + half)


def field_colormap(theme, spec):
    diverging, sequential = make_colormaps(theme)
    cmap = diverging if spec["kind"] == "diverging" else sequential
    return cmap.reversed() if spec.get("reverse") else cmap


def pooled_by_layer(per_dose, family, matching, field, layers, top_doses):
    """Trial-weighted mean of `field` over the `top_doses` largest doses, per block."""
    keep = set(top_doses)
    sums = {layer: [0.0, 0] for layer in layers}
    for row in per_dose:
        if (row["family"], row["matching"]) != (family, matching):
            continue
        if row["dose"] not in keep or row[field] is None:
            continue
        total = row["n_trials"]
        sums[row["layer"]][0] += float(row[field]) * total
        sums[row["layer"]][1] += total
    values, counts = [], []
    for layer in layers:
        accumulated, total = sums[layer]
        values.append(accumulated / total if total else np.nan)
        counts.append(total)
    return np.array(values), np.array(counts, dtype=float)


def paired_standard_errors(trials_path, matching, field, layers, top_doses):
    """SE of the pooled per-block change, from the per-trial paired differences.

    The two accuracy measures being differenced are computed on the *same* items, so
    an independent-samples error bar would overstate the uncertainty. This reads the
    per-trial difference directly -- one pass over trials.csv, accumulating a count, a
    sum and a sum of squares per (block, family) -- and reports the standard error of
    its mean. Returns None when the file is absent; the profile then draws no band
    rather than an invented one.
    """
    column = {"delta_accuracy": ("correct", "sham_correct"),
              "mean_margin_delta": ("margin", "sham_margin")}.get(field)
    if column is None or not trials_path.exists():
        return None
    perturbed_key, sham_key = column
    keep = {f"{dose:.6g}" for dose in top_doses}
    stats = {}
    with trials_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["kind"] != "perturbed" or row["matching"] != matching:
                continue
            if f"{float(row['dose']):.6g}" not in keep:
                continue
            difference = float(row[perturbed_key]) - float(row[sham_key])
            entry = stats.setdefault((int(row["layer"]), row["family"]), [0, 0.0, 0.0])
            entry[0] += 1
            entry[1] += difference
            entry[2] += difference * difference
    errors = {}
    for family in FAMILIES:
        column_values = []
        for layer in layers:
            count, total, total_squares = stats.get((layer, family), (0, 0.0, 0.0))
            if count < 2:
                column_values.append(np.nan)
                continue
            variance = max(total_squares / count - (total / count) ** 2, 0.0)
            column_values.append(float(np.sqrt(variance / count)))
        errors[family] = np.array(column_values)
    return errors


def damage_gate(summary, matching, layers, threshold=0.05, n_top_doses=4):
    """The block where the contiguous band of damaged depths ends.

    A block counts as damaged when some family's pooled accuracy drop over the largest
    doses is at least `threshold`. Contiguity matters: past the collapse an isolated
    block can cross the line on its own, and what the figures mark is where the run of
    damaged blocks stops, not the deepest block that ever crosses.

    This is a magnitude criterion and is labelled as one. It carries no significance
    claim -- a pooled cell here holds several hundred trials, so a five-point drop is
    far outside sampling noise, but the figure says what it tested rather than
    implying a test it did not run.

    Returns None when the band never stops, which is what a run damaged all the way up
    would give.
    """
    doses = sorted(summary["dose_grid"][matching])[-n_top_doses:]
    damaged = np.zeros(len(layers), dtype=bool)
    for family in FAMILIES:
        values, _ = pooled_by_layer(summary["per_dose"], family, matching,
                                    "delta_accuracy", layers, doses)
        damaged |= np.nan_to_num(values, nan=0.0) <= -threshold
    intact = np.flatnonzero(~damaged)
    if intact.size == 0 or intact[0] == 0:
        return None
    return layers[intact[0]]


def draw_gate(axis, layers, gate, theme, orientation="y", as_index=True):
    """Mark the depth where the damage stops, in data or index coordinates."""
    if gate is None:
        return
    position = layers.index(gate) - 0.5 if as_index else gate - 0.5
    line = axis.axhline if orientation == "y" else axis.axvline
    line(position, color=theme["ink"], linewidth=1.2, zorder=5)


# --------------------------------------------------------------------------------
# Heatmaps
# --------------------------------------------------------------------------------

def plot_heatmaps(summary, matching, field, theme_name, output_dir, gate_threshold=0.05):
    theme = THEMES[theme_name]
    spec = FIELDS[field]
    cmap = field_colormap(theme, spec)
    layers, doses = axis_extents(summary, matching)
    grids = {family: grid_from_per_dose(summary["per_dose"], family, matching,
                                        field, layers, doses)
             for family in FAMILIES}
    norm = field_norm(list(grids.values()), spec, summary)
    gate = damage_gate(summary, matching, layers, gate_threshold)

    figure, axes = plt.subplots(1, 4, figsize=figsize(17, 8.5), sharey=True)
    figure.patch.set_facecolor(theme["page"])
    for column, family in enumerate(FAMILIES):
        axis = axes[column]
        style_axes(axis, theme, grid=False)
        mesh = axis.imshow(np.ma.masked_invalid(grids[family]), aspect="auto",
                           origin="lower", cmap=cmap, norm=norm,
                           interpolation="nearest")
        axis.set_xticks(range(len(doses)))
        axis.set_xticklabels(dose_tick_labels(doses), rotation=90, fontsize=fs(8))
        draw_gate(axis, layers, gate, theme)
        axis.set_title(FAMILY_LABELS[family], color=theme["ink"], fontsize=fs(11), pad=8)
        axis.set_xlabel(MATCHING_LABELS[matching], color=theme["ink_secondary"], fontsize=fs(9))
        if column == 0 and gate is not None:
            # The full sentence runs off a manuscript-width panel; at that size the
            # caption carries the explanation and the figure keeps only the number.
            note = (f"block {gate}: the damaged band stops here" if FIG_SCALE > 0.7
                    else f"block {gate}")
            axis.annotate(note, (0.0, layers.index(gate) - 0.5), xytext=(3, 4),
                          textcoords="offset points", color=theme["ink"], fontsize=fs(8))
        if column == 0:
            step = 2 if FIG_SCALE > 0.7 else 4
            axis.set_yticks(range(0, len(layers), step))
            axis.set_yticklabels([str(layers[i]) for i in range(0, len(layers), step)],
                                 fontsize=fs(8))
            axis.set_ylabel("decoder block the perturbation is applied at",
                            color=theme["ink_secondary"], fontsize=fs(10))

    bar = figure.colorbar(mesh, ax=axes, fraction=0.02, pad=0.02, extend="both")
    bar.set_label(spec["label"], color=theme["ink_secondary"])
    bar.ax.tick_params(colors=theme["muted"], labelcolor=theme["ink_secondary"])
    bar.outline.set_edgecolor(theme["axis"])
    if spec["center"] is not None:
        center = (float(summary["sham"]["accuracy"]) if spec["center"] == "sham"
                  else float(spec["center"]))
        bar.ax.axhline(norm(center), color=theme["ink"], linewidth=1.0)

    figure.suptitle(
        f"Experiment 4 - {spec['label']}, {matching}-matched"
        if FIG_SCALE <= 0.7 else
        f"Experiment 4 - {spec['label']}, by block and dose, {matching}-matched",
        color=theme["ink"], fontsize=fs(14))
    path = output_dir / f"heatmap_{matching}_{field}_{theme_name}.{FORMAT}"
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor=theme["page"])
    plt.close(figure)
    return path


# --------------------------------------------------------------------------------
# 3D surfaces
# --------------------------------------------------------------------------------

def plot_surfaces(summary, matching, field, theme_name, output_dir, elev=28, azim=-58):
    theme = THEMES[theme_name]
    spec = FIELDS[field]
    cmap = field_colormap(theme, spec)
    layers, doses = axis_extents(summary, matching)
    grids = {family: grid_from_per_dose(summary["per_dose"], family, matching,
                                        field, layers, doses)
             for family in FAMILIES}
    norm = field_norm(list(grids.values()), spec, summary)

    neutral = 0.0
    if spec["center"] == "sham":
        neutral = float(summary["sham"]["accuracy"])
    elif spec["center"] is None:
        finite = np.concatenate([g[np.isfinite(g)].ravel() for g in grids.values()])
        neutral = float(finite.min())

    # One z range for all four panels, or the eye reads panel height as effect size.
    finite = np.concatenate([g[np.isfinite(g)].ravel() for g in grids.values()])
    span = max(float(finite.max()) - float(finite.min()), 1e-6)
    z_low = float(finite.min()) - 0.05 * span
    z_high = float(finite.max()) + 0.05 * span

    dose_index, layer_index = np.meshgrid(np.arange(len(doses)), np.array(layers, float))
    figure = plt.figure(figsize=figsize(15, 10))
    figure.patch.set_facecolor(theme["page"])
    for position, family in enumerate(FAMILIES, start=1):
        axis = figure.add_subplot(2, 2, position, projection="3d")
        grid = grids[family]
        # plot_surface drops a whole quad on a NaN corner; filling missing cells with
        # the neutral level keeps the mesh closed, and the flat patch reads as "no
        # change here" rather than as a hole. The heatmap shows the gap honestly.
        filled = np.where(np.isfinite(grid), grid, neutral)
        colors = cmap(norm(filled))
        # With facecolors set, plot_surface forwards one colour array as both the face
        # and the edge colour, so the separating hairline is applied afterwards.
        surface = axis.plot_surface(dose_index, layer_index, filled, facecolors=colors,
                                    rstride=1, cstride=1, antialiased=True, shade=False)
        surface.set_edgecolor(theme["surface"])
        surface.set_linewidth(0.2)
        axis.plot_surface(dose_index, layer_index, np.full_like(filled, neutral),
                          color=theme["muted"], alpha=0.18, linewidth=0,
                          antialiased=True, shade=False)

        axis.set_xticks(range(len(doses)))
        axis.set_xticklabels(thin(dose_tick_labels(doses), 2), fontsize=fs(8))
        axis.set_yticks([layer for layer in layers if layer % 4 == 0])
        axis.set_xlabel(MATCHING_LABELS[matching], color=theme["ink_secondary"],
                        fontsize=fs(9), labelpad=10)
        axis.set_ylabel("decoder block", color=theme["ink_secondary"], fontsize=fs(9),
                        labelpad=8)
        axis.set_zlabel(spec["label"], color=theme["ink_secondary"],
                        fontsize=fs(8), labelpad=4)
        axis.set_zlim(z_low, z_high)
        axis.set_title(FAMILY_LABELS[family], color=theme["ink"], fontsize=fs(11), y=0.93)
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
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    bar_axes = figure.add_axes((0.90, 0.22, 0.014, 0.56))
    bar = figure.colorbar(mappable, cax=bar_axes, extend="both")
    bar.set_label(f"{spec['label']}; the flat plane is no change",
                  color=theme["ink_secondary"])
    bar.ax.tick_params(colors=theme["muted"], labelcolor=theme["ink_secondary"])
    bar.outline.set_edgecolor(theme["axis"])

    figure.suptitle(
        f"Experiment 4 - {spec['label']} over block and dose, {matching}-matched",
        color=theme["ink"], fontsize=fs(14), y=0.97)
    path = output_dir / f"surface3d_{matching}_{field}_{theme_name}.{FORMAT}"
    figure.savefig(path, dpi=150, facecolor=theme["page"])
    plt.close(figure)
    return path


# --------------------------------------------------------------------------------
# Depth profiles
# --------------------------------------------------------------------------------

def plot_profiles(summary, matching, field, theme_name, output_dir, n_top_doses=4,
                  trials_path=None):
    theme = THEMES[theme_name]
    spec = FIELDS[field]
    layers, doses = axis_extents(summary, matching)
    top_doses = doses[-n_top_doses:]

    pooled = {family: pooled_by_layer(summary["per_dose"], family, matching, field,
                                      layers, top_doses)
              for family in FAMILIES}
    errors = (paired_standard_errors(trials_path, matching, field, layers, top_doses)
              if trials_path is not None else None)

    figure, (top, bottom) = plt.subplots(2, 1, figsize=figsize(12, 9), sharex=True,
                                         gridspec_kw={"height_ratios": [2, 1]})
    figure.patch.set_facecolor(theme["page"])
    for axis in (top, bottom):
        style_axes(axis, theme)

    for family in FAMILIES:
        values, _ = pooled[family]
        color = theme["series"][family]
        top.plot(layers, values, color=color, linewidth=2.0, marker="o",
                 markersize=4, markeredgecolor=theme["surface"], markeredgewidth=1.0,
                 label=FAMILY_LABELS[family], zorder=3)
        if errors is not None and family in errors:
            band = errors[family]
            top.fill_between(layers, values - band, values + band, color=color,
                             alpha=0.15, linewidth=0, zorder=2)

    # No direct labels here, unlike the Experiment 1 profile. There the curves separate
    # at their peaks, so a label at each peak beats a legend. Here every series is most
    # extreme at the shallowest blocks, where all four sit on top of each other, so
    # direct labels would all land in the same place; the legend carries the names.
    if spec["center"] == 0.0:
        top.axhline(0.0, color=theme["ink_secondary"], linewidth=1.0)
        top.annotate("no change from sham", (layers[0], 0.0), xytext=(2, 5),
                     textcoords="offset points", color=theme["ink_secondary"],
                     fontsize=fs(8), ha="left", va="bottom")
    top.set_ylabel(f"{spec['label']},\npooled over the {n_top_doses} largest doses",
                   color=theme["ink_secondary"], fontsize=fs(10))
    legend = top.legend(loc="lower right", fontsize=fs(9), framealpha=0.0)
    for text in legend.get_texts():
        text.set_color(theme["ink_secondary"])

    concept = pooled["concept"][0]
    controls = np.nanmean(np.vstack([pooled[f][0] for f in ("random", "noise", "dropout")]),
                          axis=0)
    bottom.bar(layers, concept - controls, color=theme["series"]["concept"],
               width=0.7, zorder=3)
    bottom.axhline(0.0, color=theme["ink_secondary"], linewidth=1.0)
    bottom.set_ylabel("concept minus\nmean of the three controls",
                      color=theme["ink_secondary"], fontsize=fs(10))
    bottom.set_xlabel("decoder block the perturbation is applied at",
                      color=theme["ink_secondary"], fontsize=fs(10))
    bottom.set_xticks(layers)
    bottom.set_xticklabels([str(layer) for layer in layers], fontsize=fs(8))

    gate = damage_gate(summary, matching, layers)
    for axis in (top, bottom):
        draw_gate(axis, layers, gate, theme, orientation="x", as_index=False)
    if gate is not None:
        top.annotate(f"block {gate}", (gate - 0.5, top.get_ylim()[0]), xytext=(4, 6),
                     textcoords="offset points", color=theme["ink"], fontsize=fs(9),
                     va="bottom")

    figure.suptitle(
        f"Experiment 4 - depth profile at the top doses, {matching}-matched",
        color=theme["ink"], fontsize=fs(14))
    figure.tight_layout()
    path = output_dir / f"profile_{matching}_{field}_{theme_name}.{FORMAT}"
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor=theme["page"])
    plt.close(figure)
    return path


# --------------------------------------------------------------------------------
# The joint detection / degradation map
# --------------------------------------------------------------------------------

def regime_grid(joined, family, matching, layers, doses, detect_threshold,
                damage_threshold):
    """Four-way classification of each (block, dose) cell, NaN where unjoined.

    0 quiet, 1 detected with the task intact, 2 detected with the task damaged,
    3 damaged with no detection. Code 1 is the H4-compatible cell of doc 8.8: the
    intervention is reported above chance while the model still does the task.
    """
    index = {(row["layer"], row["dose"]): row for row in joined
             if row["family"] == family and row["matching"] == matching}
    grid = np.full((len(layers), len(doses)), np.nan)
    for i, layer in enumerate(layers):
        for j, dose in enumerate(doses):
            row = index.get((layer, dose))
            if row is None:
                continue
            detected = row["detection_accuracy"] >= detect_threshold
            damaged = row["classification_delta_accuracy"] < -damage_threshold
            grid[i, j] = (1 if detected and not damaged else
                          2 if detected and damaged else
                          3 if damaged else 0)
    return grid


def plot_regimes(summary, matching, theme_name, output_dir, detect_threshold=0.70,
                 damage_threshold=0.05):
    joined = summary.get("detection_vs_performance")
    if not joined:
        return None
    theme = THEMES[theme_name]
    layers, doses = axis_extents(summary, matching)
    # Grey for quiet, the concept hue for the H4 cells, red for detection bought with
    # damage, amber for damage that buys no detection.
    colors = [theme["grid"], theme["series"]["concept"], "#d03b3b", theme["series"]["dropout"]]
    cmap = ListedColormap(colors)
    cmap.set_bad(theme["missing"])
    norm = Normalize(vmin=-0.5, vmax=3.5)

    figure, axes = plt.subplots(1, 4, figsize=figsize(17, 8.5), sharey=True)
    figure.patch.set_facecolor(theme["page"])
    counts = {code: 0 for code in range(4)}
    for column, family in enumerate(FAMILIES):
        axis = axes[column]
        style_axes(axis, theme, grid=False)
        grid = regime_grid(joined, family, matching, layers, doses,
                           detect_threshold, damage_threshold)
        for code in range(4):
            counts[code] += int(np.sum(grid == code))
        axis.imshow(np.ma.masked_invalid(grid), aspect="auto", origin="lower",
                    cmap=cmap, norm=norm, interpolation="nearest")
        axis.set_xticks(range(len(doses)))
        axis.set_xticklabels(dose_tick_labels(doses), rotation=90, fontsize=fs(8))
        axis.set_title(FAMILY_LABELS[family], color=theme["ink"], fontsize=fs(11), pad=8)
        axis.set_xlabel(MATCHING_LABELS[matching], color=theme["ink_secondary"],
                        fontsize=fs(9))
        if column == 0:
            step = 2 if FIG_SCALE > 0.7 else 4
            axis.set_yticks(range(0, len(layers), step))
            axis.set_yticklabels([str(layers[i]) for i in range(0, len(layers), step)],
                                 fontsize=fs(8))
            axis.set_ylabel("decoder block the perturbation is applied at",
                            color=theme["ink_secondary"], fontsize=fs(10))

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[code], edgecolor="none")
               for code in range(4)]
    labels = [f"{REGIME_LABELS[code]}  ({counts[code]} cells)" for code in range(4)]
    # Reserve the strip the legend will sit in before placing it. Anchoring it just
    # below the figure and relying on a tight bounding box puts it on top of the
    # rotated dose labels, which occupy exactly that space.
    rows = 1 if FIG_SCALE > 0.7 else 2
    figure.subplots_adjust(bottom=0.16 + 0.06 * rows)
    legend = figure.legend(handles, labels, loc="lower center",
                           ncols=4 // rows, frameon=False,
                           fontsize=fs(9), bbox_to_anchor=(0.5, 0.0))
    for text in legend.get_texts():
        text.set_color(theme["ink_secondary"])

    title = f"Experiment 4 - detection against task damage, {matching}-matched"
    if FIG_SCALE > 0.7:
        title += (f" (detected: Experiment 1 accuracy >= {detect_threshold:.0%}; "
                  f"damaged: accuracy drop > {damage_threshold:.0%})")
    figure.suptitle(title, color=theme["ink"], fontsize=fs(13))
    path = output_dir / f"regimes_{matching}_{theme_name}.{FORMAT}"
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor=theme["page"])
    plt.close(figure)
    return path


# --------------------------------------------------------------------------------
# Text report
# --------------------------------------------------------------------------------

def write_report(summary, matching, field, layers, handle, n_top_doses=4):
    doses = sorted(summary["dose_grid"][matching])
    top_doses = doses[-n_top_doses:]
    pooled = {family: pooled_by_layer(summary["per_dose"], family, matching, field,
                                      layers, top_doses)[0]
              for family in FAMILIES}
    controls = np.nanmean(np.vstack([pooled[f] for f in ("random", "noise", "dropout")]),
                          axis=0)
    print(f"\n{matching}-matched, {FIELDS[field]['label']}, pooled over doses "
          f"{', '.join(f'{d:g}' for d in top_doses)}", file=handle)
    print(f"{'block':>5} " + " ".join(f"{f:>9}" for f in FAMILIES) + f" {'gap':>9}",
          file=handle)
    for index, layer in enumerate(layers):
        cells = " ".join(
            "      n/a" if not np.isfinite(pooled[f][index]) else f"{pooled[f][index]:9.3f}"
            for f in FAMILIES)
        gap = pooled["concept"][index] - controls[index]
        gap_cell = "      n/a" if not np.isfinite(gap) else f"{gap:9.3f}"
        print(f"{layer:>5} {cells} {gap_cell}", file=handle)


# --------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Layer-resolved heatmaps, 3D surfaces, depth profiles and the "
                    "joint detection/degradation map for an Experiment 4 sweep.")
    parser.add_argument("--run_dir", type=Path,
                        default=REPO_ROOT / "results" / "experiment4" / "main",
                        help="Directory holding summary.json (and optionally trials.csv)")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Where the figures go (default: <run_dir>/layer_maps)")
    parser.add_argument("--matchings", nargs="+", default=list(MATCHINGS),
                        choices=list(MATCHINGS))
    parser.add_argument("--fields", nargs="+",
                        default=["delta_accuracy", "mean_margin_delta"],
                        choices=list(FIELDS),
                        help="Which per-cell measure the block x dose views map.")
    parser.add_argument("--themes", nargs="+", default=["light"], choices=list(THEMES))
    parser.add_argument("--n_top_doses", type=int, default=4,
                        help="How many of the largest doses the depth profile pools")
    parser.add_argument("--damage_threshold", type=float, default=0.05,
                        help="Accuracy drop past which a cell counts as damaged, for "
                             "the gate line and the regimes map.")
    parser.add_argument("--detect_threshold", type=float, default=0.70,
                        help="Experiment 1 detection accuracy at which a cell counts "
                             "as detected, for the regimes map.")
    parser.add_argument("--elev", type=float, default=28.0, help="3D elevation angle")
    parser.add_argument("--azim", type=float, default=-58.0, help="3D azimuth angle")
    parser.add_argument("--no_error_bands", action="store_true",
                        help="Skip the pass over trials.csv that draws the paired "
                             "standard error on the depth profile.")
    parser.add_argument("--paper", action="store_true",
                        help="Draw at two-column manuscript width with type scaled to "
                             "match, instead of the wide on-screen geometry.")
    parser.add_argument("--format", choices=["png", "pdf"], default="png",
                        help="Figure container. Use pdf for the LaTeX reports.")
    parser.add_argument("--report", type=Path, default=None,
                        help="Also write the pooled depth table here (default: stdout)")
    args = parser.parse_args()

    global FORMAT, FIG_SCALE, FONT_SCALE
    FORMAT = args.format
    if args.paper:
        # 17 in -> 7.1 in, about the \textwidth of a two-column article, included at
        # width=\textwidth so the on-page scale factor is ~1. The wide-canvas type
        # sizes (8-14 pt) are then proportionally too big; 0.72 puts them at 6-10 pt.
        FIG_SCALE = 0.42
        FONT_SCALE = 0.72

    run_dir = args.run_dir.resolve()
    summary = load_summary(run_dir)
    output_dir = (args.output_dir or run_dir / "layer_maps").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run      {run_dir}")
    print(f"Model    {summary['model']}")
    print(f"Trials   {summary['n_trials']}")
    print(f"Blocks   {min(summary['layers'])}-{max(summary['layers'])}")
    print(f"Sham     {summary['sham']['accuracy']:.3f} classification accuracy")

    trials_path = run_dir / "trials.csv"
    if args.no_error_bands or not trials_path.exists():
        if not args.no_error_bands:
            print(f"No trials.csv in {run_dir}; the depth profile draws no error band")
        trials_path = None

    for theme_name in args.themes:
        for matching in args.matchings:
            for field in args.fields:
                print(f"Saved {plot_heatmaps(summary, matching, field, theme_name, output_dir, args.damage_threshold)}")
                print(f"Saved {plot_surfaces(summary, matching, field, theme_name, output_dir, args.elev, args.azim)}")
                print(f"Saved {plot_profiles(summary, matching, field, theme_name, output_dir, args.n_top_doses, trials_path)}")
            path = plot_regimes(summary, matching, theme_name, output_dir,
                                args.detect_threshold, args.damage_threshold)
            if path is None:
                print("No detection_vs_performance in summary.json; skipping the "
                      "regimes map. Rerun --reanalyze with --experiment1_summary.")
            else:
                print(f"Saved {path}")

    layers = sorted(summary["layers"])
    if args.report:
        with args.report.open("w") as handle:
            for matching in args.matchings:
                for field in args.fields:
                    write_report(summary, matching, field, layers, handle, args.n_top_doses)
        print(f"Saved {args.report}")
    else:
        for matching in args.matchings:
            for field in args.fields:
                write_report(summary, matching, field, layers, sys.stdout, args.n_top_doses)


if __name__ == "__main__":
    main()
