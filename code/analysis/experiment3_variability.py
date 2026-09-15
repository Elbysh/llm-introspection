#!/usr/bin/env python3
"""Experiment 3: variance of the localization effect over concepts, directions and layers.

Doc 7 (Experience 3) asks whether the mean effect of Experiment 1 is homogeneous, and
for the variance components behind it. The trials it needs are the Experiment 1 trials:
the sweep already varies direction_id inside each family, so a per-direction read of the
same rows is the experiment, and no new GPU pass is required.

What this adds over experiment1_psychometrics.py:summarize is the key. That function
pools every concept into one `concept` curve per (layer, matching) -- 256 curves for a
32-layer run -- so nothing in summary.json can speak to between-concept variance. Here
the cell is (layer, family, direction_id, matching), which is what doc 7.5 measures.

The measure. Doc 7.5 asks for accuracy, contrast and an individual 75% threshold. On
this task the threshold is mostly unavailable: raw 2AFC accuracy is pinned near chance
by the model's standing answer bias, so even the pooled curves cross 75% in only 0-5 of
32 layers, and a per-direction cell has a quarter of the trials. The threshold is still
reported wherever the curve genuinely brackets 75%, but the workhorse is the paired
localization contrast of doc 5.8,

    S = (contrast when A is targeted - contrast when B is targeted) / 2

over the two trials sharing everything but which sentence was hit. Any shift that does
not depend on the target cancels, which is what makes it readable where accuracy is not.
This is the same estimator, and the same pairing key, as
experiment1_localization_report.py.

The decomposition. Doc 7.6 asks for a model carrying effects proper to the sentences and
to the directions. For a design this balanced a two- and three-way mean decomposition is
the mixed model's fixed part and needs no fitting: over the cells of a family,

    S[d, l, p] = grand + a[d] + b[l] + c[p] + interaction + residual

and the variance of each term is reported as a share of the total. `direction x layer`
is the interaction doc 7.7 warns about -- where it dominates, a family difference cannot
be attributed to the architecture without naming the injection site.

    python code/analysis/experiment3_variability.py \
        --run_dir results/experiment1/z2afc_all --matching z \
        --out_dir results/experiment3/llama_z

Several --run_dir are pooled, as in experiment1_localization_report.py. --matching picks
one arm: the two are different dose variables and must never share a decomposition.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

from experiments.experiment1_psychometrics import (  # noqa: E402
    bracketing_doses,
    fit_psychometric,
    threshold_at_75,
)

# Everything but the targeted sentence, as in experiment1_psychometrics.py:summarize.
PAIR_KEY = ("layer", "family", "matching", "dose", "direction_id",
            "pair_id", "order", "label_order")

# Families whose direction_id names a thing worth taking a variance over. noise and
# dropout redraw their perturbation every trial, so their "directions" are realization
# indices and a spread across them is sampling noise, not between-direction variance.
IDENTIFIED_FAMILIES = ("concept", "random", "scrambled")


def direction_label(direction_id):
    """The direction's identity with the block stripped out.

    direction_id carries the injection site -- `concept__block_07__Dust`,
    `fixed_random__block_07__0001` -- because the vector is derived at the block it is
    injected into. Crossed with `layer` as it stands it would give a design that is
    empty off the diagonal, since a block-7 vector exists only at layer 7. The factor
    doc 7.5 takes a variance over is the identity that recurs at every depth: the
    concept, or the random direction's index.
    """
    return direction_id.rsplit("__", 1)[-1]


def load_trials(run_dirs, matching):
    """Perturbed rows of one matching arm, plus the model, from every run directory."""
    rows, models, seen_matchings = [], set(), set()
    for run_dir in run_dirs:
        path = run_dir / "trials.csv"
        if not path.exists():
            raise SystemExit(f"no trials.csv in {run_dir}")
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            with summary_path.open(encoding="utf-8") as handle:
                models.add(json.load(handle).get("model"))
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                # Sham rows carry no layer, dose or target: one clean pass per
                # presentation, shared by every condition and both matchings. They
                # score nothing here, since S is defined only on a targeted pair.
                if row["kind"] != "perturbed":
                    continue
                seen_matchings.add(row["matching"])
                if row["matching"] != matching:
                    continue
                row["layer"] = int(row["layer"])
                row["dose"] = float(row["dose"])
                row["contrast"] = float(row["contrast"])
                for field in ("correct_raw", "correct_adjusted"):
                    row[field] = float(row[field]) if row.get(field) else None
                rows.append(row)
    if len(models) > 1:
        raise SystemExit(f"run directories disagree on the model: {sorted(models)}")
    if not rows:
        raise SystemExit(
            f"no {matching}-matched rows in {[str(d) for d in run_dirs]}; "
            f"these runs carry {sorted(seen_matchings)}")
    return rows, (models.pop() if models else None)


def paired_s(rows, key_fields):
    """S for every complete pair, grouped by key_fields.

    A pair contributes +contrast from its A-targeted half and -contrast from its
    B-targeted half, so a target-independent shift cancels in the sum. An incomplete
    pair cannot cancel anything and is dropped.
    """
    paired = defaultdict(lambda: {"n": 0, "sum": 0.0, "row": None})
    for row in rows:
        entry = paired[tuple(row[field] for field in PAIR_KEY)]
        entry["n"] += 1
        entry["sum"] += row["contrast"] if row["target_label"] == "A" else -row["contrast"]
        entry["row"] = row
    grouped = defaultdict(list)
    for entry in paired.values():
        if entry["n"] == 2:
            grouped[field_values(entry["row"], key_fields)].append(entry["sum"] / 2.0)
    return grouped


def field_values(row, fields):
    """Row fields as a key, resolving the derived `direction_label`."""
    return tuple(direction_label(row["direction_id"]) if field == "direction_label"
                 else row[field] for field in fields)


def per_direction_curves(rows):
    """One accuracy-and-threshold record per (layer, family, direction_id, dose-set).

    The fit and the reportability gate are experiment1_psychometrics.py's, so a
    per-direction threshold means exactly what a pooled one means in summary.json.
    """
    cells = defaultdict(lambda: {"n": 0, "raw": 0.0, "adjusted": 0.0})
    for row in rows:
        cell = cells[(row["layer"], row["family"], row["direction_id"], row["dose"])]
        cell["n"] += 1
        cell["raw"] += row["correct_raw"]
        cell["adjusted"] += row["correct_adjusted"]

    s_by_cell = paired_s(rows, ("layer", "family", "direction_id", "dose"))

    by_curve = defaultdict(list)
    for (layer, family, direction_id, dose), cell in cells.items():
        s_values = s_by_cell.get((layer, family, direction_id, dose), [])
        by_curve[(layer, family, direction_id)].append({
            "dose": dose,
            "n_trials": cell["n"],
            "accuracy_raw": cell["raw"] / cell["n"],
            "accuracy_adjusted": cell["adjusted"] / cell["n"],
            "mean_s": float(np.mean(s_values)) if s_values else None,
            "n_pairs": len(s_values),
        })

    curves = []
    for (layer, family, direction_id), points in sorted(by_curve.items()):
        points.sort(key=lambda point: point["dose"])
        record = {"layer": layer, "family": family, "direction_id": direction_id,
                  "points": points}
        for metric in ("raw", "adjusted"):
            doses = [point["dose"] for point in points]
            totals = [point["n_trials"] for point in points]
            accuracies = [point[f"accuracy_{metric}"] for point in points]
            corrects = [int(round(a * n)) for a, n in zip(accuracies, totals)]
            beta0 = beta1 = None
            converged = False
            if len(doses) >= 3:
                beta0, beta1, converged = fit_psychometric(doses, corrects, totals)
            bracket = bracketing_doses(doses, accuracies)
            threshold = threshold_at_75(beta0, beta1) if converged else None
            # Doc 5.10: nothing is extrapolated outside the tested range, and no
            # threshold is reported without a bracketed transition.
            reportable = (threshold is not None and bracket is not None
                          and min(doses) <= threshold <= max(doses))
            record[f"threshold_75_{metric}"] = threshold if reportable else None
            record[f"threshold_reportable_{metric}"] = reportable
        curves.append(record)
    return curves


def select_doses(rows, pooled_family="concept", count=3):
    """The three doses of doc 7.3: below, near and above the global threshold.

    Doc 7.4 fixes them from the pooled curve before the per-direction analysis. Where
    that curve reports no threshold -- the usual case here, since accuracy is bias-pinned
    -- the anchor falls back to the tested dose at which pooled |S| is largest, which is
    where the perturbation demonstrably bites. The three returned doses are always tested
    grid values, never interpolated.
    """
    doses = sorted({row["dose"] for row in rows})
    pooled = {family: {} for family in {row["family"] for row in rows}}
    for (family, dose), values in paired_s(rows, ("family", "dose")).items():
        pooled[family][dose] = float(np.mean(values))

    cells = defaultdict(lambda: {"n": 0, "raw": 0.0})
    for row in rows:
        if row["family"] != pooled_family:
            continue
        cell = cells[row["dose"]]
        cell["n"] += 1
        cell["raw"] += row["correct_raw"]
    grid = sorted(cells)
    accuracies = [cells[dose]["raw"] / cells[dose]["n"] for dose in grid]
    corrects = [int(round(a * cells[dose]["n"])) for a, dose in zip(accuracies, grid)]
    totals = [cells[dose]["n"] for dose in grid]

    anchor, basis = None, None
    if len(grid) >= 3:
        beta0, beta1, converged = fit_psychometric(grid, corrects, totals)
        threshold = threshold_at_75(beta0, beta1) if converged else None
        if (threshold is not None and bracketing_doses(grid, accuracies)
                and min(grid) <= threshold <= max(grid)):
            anchor, basis = threshold, "pooled_threshold_75_raw"
    if anchor is None:
        curve = pooled.get(pooled_family, {})
        if not curve:
            raise SystemExit(f"no {pooled_family} rows to anchor the dose selection on")
        anchor = max(curve, key=lambda dose: abs(curve[dose]))
        basis = "max_abs_pooled_s"

    # The tested dose nearest the anchor, then its neighbours on the grid.
    index = min(range(len(doses)), key=lambda i: abs(np.log(doses[i]) - np.log(anchor)))
    low = max(0, min(index - 1, len(doses) - count))
    selected = doses[low:low + count]
    return {"doses": selected, "anchor": float(anchor), "basis": basis,
            "grid": doses, "pooled_s": pooled}


def decompose(matrix, factor_names):
    """Share of variance carried by each main effect and by the interaction.

    `matrix` is a dense array indexed by the factors in order. The design is balanced by
    construction -- every direction meets every layer and every phrase -- so the mean
    decomposition below is the fixed part of the mixed model of doc 7.6, and its terms
    are orthogonal, meaning the shares sum to 1 without a fitting step.
    """
    grand = float(matrix.mean())
    total = float(((matrix - grand) ** 2).mean())
    components = {}
    residual = matrix - grand
    for axis, name in enumerate(factor_names):
        other = tuple(i for i in range(matrix.ndim) if i != axis)
        effect = matrix.mean(axis=other) - grand
        components[name] = float((effect ** 2).mean())
        shape = [1] * matrix.ndim
        shape[axis] = matrix.shape[axis]
        residual = residual - effect.reshape(shape)
    components["interaction"] = float((residual ** 2).mean())
    return {
        "grand_mean": grand,
        "total_variance": total,
        "components": components,
        "shares": ({name: value / total for name, value in components.items()}
                   if total > 0 else None),
        "sd": {name: float(np.sqrt(value)) for name, value in components.items()},
    }


def variance_components(rows, doses):
    """Doc 7.5's variances, per family, over the three selected doses.

    Restricting to the selected doses is what makes the decomposition meaningful: a
    variance taken across the whole grid would be dominated by the dose effect, which is
    the psychometric curve and not the question doc 7 asks.
    """
    selected = set(doses)
    kept = [row for row in rows if row["dose"] in selected]
    report = {}
    for family in sorted({row["family"] for row in kept}):
        if family not in IDENTIFIED_FAMILIES:
            continue
        family_rows = [row for row in kept if row["family"] == family]
        s_cells = paired_s(family_rows, ("direction_label", "layer", "pair_id"))
        directions = sorted({key[0] for key in s_cells})
        layers = sorted({key[1] for key in s_cells})
        pairs = sorted({key[2] for key in s_cells})
        if len(directions) < 2:
            continue

        # A cell missing from a run that did not sweep every layer would make the
        # decomposition non-orthogonal, so the dense grid is required and checked.
        dense = np.full((len(directions), len(layers), len(pairs)), np.nan)
        for (direction, layer, pair), values in s_cells.items():
            dense[directions.index(direction), layers.index(layer),
                  pairs.index(pair)] = float(np.mean(values))
        if np.isnan(dense).any():
            missing = int(np.isnan(dense).sum())
            raise SystemExit(
                f"family {family}: {missing} of {dense.size} (direction, layer, phrase) "
                "cells are empty; the decomposition needs a complete design")

        by_direction = {direction: float(np.nanmean(dense[index]))
                        for index, direction in enumerate(directions)}
        report[family] = {
            "n_directions": len(directions),
            "n_layers": len(layers),
            "n_phrases": len(pairs),
            "directions": directions,
            "mean_s_by_direction": by_direction,
            "spread_between_directions": {
                "sd": float(np.std(list(by_direction.values()), ddof=1)),
                "min": min(by_direction, key=by_direction.get),
                "max": max(by_direction, key=by_direction.get),
                "ratio_max_over_min": (
                    float(max(abs(v) for v in by_direction.values())
                          / min(abs(v) for v in by_direction.values()))
                    if min(abs(v) for v in by_direction.values()) > 0 else None),
            },
            "decomposition": decompose(dense, ("direction", "layer", "phrase")),
            "decomposition_direction_layer": decompose(
                dense.mean(axis=2), ("direction", "layer")),
        }
    return report


# --------------------------------------------------------------------------------
# Figures
#
# Doc 7.6 asks for distributions and individual trajectories, not a mean with an error
# bar, so every direction is drawn as its own line or point.
# --------------------------------------------------------------------------------

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Palette and styling of plot_experiment1_layer_maps.py, so the Experiment 3 figures
# sit beside the Experiment 1 ones without a second visual language.
THEMES = {
    "light": {"surface": "#fcfcfb", "page": "#f9f9f7", "ink": "#0b0b0b",
              "ink_secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7",
              "series": {"concept": "#2a78d6", "random": "#eb6834",
                         "noise": "#1baf7a", "dropout": "#eda100",
                         "scrambled": "#8a63d2"}},
    "dark": {"surface": "#1a1a19", "page": "#0d0d0d", "ink": "#ffffff",
             "ink_secondary": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835",
             "series": {"concept": "#3987e5", "random": "#d95926",
                        "noise": "#199e70", "dropout": "#c98500",
                        "scrambled": "#9b7ae0"}},
}


def style_axes(axis, theme, grid=True):
    axis.set_facecolor(theme["surface"])
    for spine in axis.spines.values():
        spine.set_color(theme["axis"])
        spine.set_linewidth(0.8)
    axis.tick_params(colors=theme["muted"], labelcolor=theme["ink_secondary"], width=0.8)
    if grid:
        axis.grid(True, color=theme["grid"], linewidth=0.8, alpha=0.9)
        axis.set_axisbelow(True)


# Tint alone separates two or three lines but not four, so each direction also takes
# its own marker. Which concept is the outlier is the result these figures carry, so
# the lines have to be tellable apart without reading the legend order.
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")


def shade(base, index, count):
    """`count` distinguishable tints of one family colour, for the individual lines."""
    rgb = np.array(matplotlib.colors.to_rgb(base))
    if count < 2:
        return tuple(rgb)
    weight = 0.85 * (index / (count - 1)) - 0.35     # -0.35 darker .. +0.50 lighter
    mixed = rgb + weight * ((1.0 - rgb) if weight > 0 else rgb)
    return tuple(np.clip(mixed, 0.0, 1.0))


def plot_trajectories(rows, doses, out_path, theme, matching):
    """Mean S against layer, one line per direction, one panel per family."""
    selected = set(doses)
    kept = [row for row in rows if row["dose"] in selected
            and row["family"] in IDENTIFIED_FAMILIES]
    families = sorted({row["family"] for row in kept})
    if not families:
        return
    cells = paired_s(kept, ("family", "direction_label", "layer"))

    figure, axes = plt.subplots(1, len(families), figsize=(6.2 * len(families), 4.6),
                                squeeze=False, facecolor=theme["page"])
    for axis, family in zip(axes[0], families):
        style_axes(axis, theme)
        directions = sorted({key[1] for key in cells if key[0] == family})
        for index, direction in enumerate(directions):
            layers = sorted(key[2] for key in cells if key[:2] == (family, direction))
            values = [float(np.mean(cells[(family, direction, layer)])) for layer in layers]
            axis.plot(layers, values, marker=MARKERS[index % len(MARKERS)],
                      markersize=3.4, linewidth=1.3,
                      color=shade(theme["series"][family], index, len(directions)),
                      label=direction)
        axis.axhline(0.0, color=theme["axis"], linewidth=1.0, zorder=1)
        axis.set_title(f"{family} -- {len(directions)} directions", color=theme["ink"])
        axis.set_xlabel("injection block", color=theme["ink_secondary"])
        axis.set_ylabel("mean paired localization contrast S", color=theme["ink_secondary"])
        axis.legend(frameon=False, fontsize=7, ncol=2,
                    labelcolor=theme["ink_secondary"])
    figure.suptitle(f"Individual trajectories over depth, {matching}-matched, "
                    f"doses {', '.join(f'{d:g}' for d in doses)}", color=theme["ink"])
    figure.tight_layout()
    figure.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=theme["page"])
    plt.close(figure)


def plot_distributions(rows, doses, out_path, theme, matching):
    """Every (direction, layer) cell as a point, so the spread is visible, not summarised."""
    selected = set(doses)
    kept = [row for row in rows if row["dose"] in selected
            and row["family"] in IDENTIFIED_FAMILIES]
    families = sorted({row["family"] for row in kept})
    if not families:
        return
    cells = paired_s(kept, ("family", "direction_label", "layer"))

    figure, axis = plt.subplots(figsize=(1.35 * sum(
        len({key[1] for key in cells if key[0] == family}) for family in families) + 2.5,
        4.6), facecolor=theme["page"])
    style_axes(axis, theme)
    rng = np.random.default_rng(0)
    position, ticks, labels = 0, [], []
    for family in families:
        for direction in sorted({key[1] for key in cells if key[0] == family}):
            values = np.array([float(np.mean(cells[key])) for key in cells
                               if key[:2] == (family, direction)])
            colour = theme["series"][family]
            axis.scatter(position + rng.uniform(-0.17, 0.17, len(values)), values,
                         s=11, color=colour, alpha=0.55, linewidths=0)
            axis.plot([position - 0.32, position + 0.32], [values.mean()] * 2,
                      color=theme["ink"], linewidth=1.6, zorder=3)
            ticks.append(position)
            labels.append(direction)
            position += 1
        position += 0.6
    axis.axhline(0.0, color=theme["axis"], linewidth=1.0, zorder=1)
    axis.set_xticks(ticks)
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axis.set_ylabel("mean S per layer", color=theme["ink_secondary"])
    axis.set_title(f"Distribution over layers of each direction's effect, "
                   f"{matching}-matched (bar = direction mean)", color=theme["ink"])
    figure.tight_layout()
    figure.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=theme["page"])
    plt.close(figure)


def plot_variance_shares(report, out_path, theme, matching):
    """Share of variance on direction, layer, phrase and interaction, per family."""
    families = [family for family in sorted(report)
                if report[family]["decomposition"]["shares"]]
    if not families:
        return
    names = ("direction", "layer", "phrase", "interaction")
    figure, axis = plt.subplots(figsize=(1.9 * len(families) + 3.4, 4.2),
                                facecolor=theme["page"])
    style_axes(axis, theme)
    width = 0.2
    base = np.arange(len(families))
    greys = ("#2a78d6", "#eb6834", "#1baf7a", "#898781")
    for index, name in enumerate(names):
        values = [report[family]["decomposition"]["shares"][name] for family in families]
        bars = axis.bar(base + (index - 1.5) * width, values, width, label=name,
                        color=greys[index], linewidth=0)
        axis.bar_label(bars, fmt="%.2f", fontsize=7, color=theme["ink_secondary"],
                       padding=1)
    axis.set_xticks(base)
    axis.set_xticklabels(families, color=theme["ink_secondary"])
    axis.set_ylabel("share of variance in S", color=theme["ink_secondary"])
    axis.set_title(f"Variance components, {matching}-matched", color=theme["ink"])
    axis.legend(frameon=False, fontsize=8, labelcolor=theme["ink_secondary"])
    figure.tight_layout()
    figure.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=theme["page"])
    plt.close(figure)


# --------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------

CURVE_FIELDS = ("layer", "family", "direction_id", "direction_label",
                "dose", "n_trials", "n_pairs",
                "accuracy_raw", "accuracy_adjusted", "mean_s",
                "threshold_75_raw", "threshold_reportable_raw",
                "threshold_75_adjusted", "threshold_reportable_adjusted")


def write_curves_csv(curves, path):
    """One row per (layer, family, direction_id, dose); the threshold repeats per curve."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CURVE_FIELDS)
        writer.writeheader()
        for curve in curves:
            for point in curve["points"]:
                writer.writerow({
                    "layer": curve["layer"], "family": curve["family"],
                    "direction_id": curve["direction_id"],
                    "direction_label": direction_label(curve["direction_id"]),
                    "threshold_75_raw": curve["threshold_75_raw"],
                    "threshold_reportable_raw": curve["threshold_reportable_raw"],
                    "threshold_75_adjusted": curve["threshold_75_adjusted"],
                    "threshold_reportable_adjusted": curve["threshold_reportable_adjusted"],
                    **point})


def relative_to_repo(path):
    """Repo-relative where possible, absolute otherwise, so a run dir outside the
    working tree -- a re-read of an exported slice, say -- still records its source."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def export_run(rows, source_dirs, matching, out_dir):
    """The rows this run read, written as a run directory in their own right.

    `trials.csv` keeps the source file's column order and spelling, and the numeric
    fields round-trip through repr of the parsed value, so re-reading the export
    reproduces the analysis exactly. `summary.json` carries the fields load_trials
    consults -- the model -- plus the provenance, so the export is a --run_dir like any
    other rather than a loose CSV that only this script knows how to open.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with (source_dirs[0] / "trials.csv").open(encoding="utf-8", newline="") as handle:
        fieldnames = next(csv.reader(handle))
    with (out_dir / "trials.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    source = {}
    summary_path = source_dirs[0] / "summary.json"
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as handle:
            source = json.load(handle)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "model": source.get("model"),
            "calibration_dir": source.get("calibration_dir"),
            "concepts": source.get("concepts"),
            "matchings": [matching],
            "n_trials": len(rows),
            "notes": [
                f"The {matching}-matched rows of "
                f"{', '.join(relative_to_repo(d) for d in source_dirs)}, extracted by "
                "experiment3_variability.py --export_run_dir. The other matching arm of "
                "those runs is deliberately absent."],
        }, handle, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", action="append", required=True, type=Path,
                        help="an Experiment 1 run directory; repeat to pool")
    parser.add_argument("--matching", required=True, choices=("alpha", "z"),
                        help="which dose arm to read; the two are never pooled")
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--num_doses", type=int, default=3,
                        help="doc 7.3 fixes three, around the global threshold")
    parser.add_argument("--themes", nargs="+", default=["light"],
                        choices=sorted(THEMES))
    parser.add_argument("--export_run_dir", type=Path, default=None,
                        help="write the rows this run read as a run directory of their "
                             "own, so a source sweep that is not itself tracked still "
                             "reproduces from a clone")
    args = parser.parse_args()

    run_dirs = [path if path.is_absolute() else REPO_ROOT / path for path in args.run_dir]
    rows, model = load_trials(run_dirs, args.matching)
    print(f"{len(rows)} {args.matching}-matched trials from "
          f"{len(run_dirs)} run(s), model {model}", flush=True)

    selection = select_doses(rows, count=args.num_doses)
    print(f"doses {selection['doses']} around anchor {selection['anchor']:.4g} "
          f"({selection['basis']})", flush=True)

    curves = per_direction_curves(rows)
    components = variance_components(rows, selection["doses"])

    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_curves_csv(curves, out_dir / "per_direction_curves.csv")

    if args.export_run_dir is not None:
        export = (args.export_run_dir if args.export_run_dir.is_absolute()
                  else REPO_ROOT / args.export_run_dir)
        export_run(rows, run_dirs, args.matching, export)
        print(f"exported {len(rows)} rows to {export}", flush=True)

    reportable = sum(curve["threshold_reportable_adjusted"] for curve in curves)
    report = {
        "model": model,
        "matching": args.matching,
        "runs": [relative_to_repo(path) for path in run_dirs],
        "n_trials": len(rows),
        "n_curves": len(curves),
        "n_thresholds_reportable_adjusted": reportable,
        "dose_selection": selection,
        "variance_components": components,
    }
    with (out_dir / "variability_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    for name in args.themes:
        theme = THEMES[name]
        plot_trajectories(rows, selection["doses"],
                          out_dir / f"trajectories_{name}.png", theme, args.matching)
        plot_distributions(rows, selection["doses"],
                           out_dir / f"distributions_{name}.png", theme, args.matching)
        plot_variance_shares(components, out_dir / f"variance_shares_{name}.png",
                             theme, args.matching)

    for family, entry in sorted(components.items()):
        shares = entry["decomposition"]["shares"]
        spread = entry["spread_between_directions"]
        print(f"  {family}: {entry['n_directions']} directions x {entry['n_layers']} "
              f"layers x {entry['n_phrases']} phrases, sd between directions "
              f"{spread['sd']:.4g}")
        if shares:
            print("    shares " + ", ".join(f"{k} {v:.3f}" for k, v in shares.items()))
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
