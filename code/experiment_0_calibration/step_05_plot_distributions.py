"""Experiment 0, step 5: report-ready diagnostic figures.

Every aggregation is explicit: concepts keep their semantic identity, whereas
fixed-random and renewed-noise directions are summarized within each block.
Families are placed in separate panels whenever their sample sizes or scales
would otherwise make a shared axis misleading.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FAMILIES = ["concept", "fixed_random", "renewed_noise"]
FAMILY_COLORS = {
    "concept": "#3366cc",
    "fixed_random": "#dd4477",
    "renewed_noise": "#109618",
}


def load_records(path: Path) -> pd.DataFrame:
    """Load the direction-level statistics and validate plotting inputs."""
    with path.open("r", encoding="utf-8") as handle:
        frame = pd.DataFrame(json.load(handle))
    required = {
        "direction_id",
        "direction_family",
        "decoder_block_index",
        "sd",
        "mad_corrected",
        "sd_ci_low",
        "sd_ci_high",
        "sd_bootstrap_cv",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("missing Experiment 0 result columns: {}".format(sorted(missing)))
    return frame


def _plot_family_scale_panel(
    axis: plt.Axes, frame: pd.DataFrame, family: str, log_scale: bool
) -> None:
    """Plot one family without connecting unrelated controls across blocks."""
    subset = frame.loc[frame["direction_family"] == family]
    if family == "concept":
        # A concept retains its identity across blocks and can be followed as a
        # curve. Plotting the ten concepts avoids hiding semantic heterogeneity.
        for concept, group in subset.groupby("concept"):
            ordered = group.sort_values("decoder_block_index")
            axis.plot(
                ordered["decoder_block_index"],
                ordered["sd"],
                linewidth=1.5,
                alpha=0.82,
                label=str(concept),
            )
        axis.legend(fontsize=8, ncol=2)
    else:
        # Control directions are independently drawn for each block. Connecting
        # individual draws would create artificial trajectories, so show their
        # within-block median, IQR and 5th--95th percentile envelope instead.
        grouped = subset.groupby("decoder_block_index")["sd"]
        median = grouped.median()
        q25 = grouped.quantile(0.25)
        q75 = grouped.quantile(0.75)
        q05 = grouped.quantile(0.05)
        q95 = grouped.quantile(0.95)
        color = FAMILY_COLORS[family]
        axis.fill_between(
            median.index,
            q05.values,
            q95.values,
            color=color,
            alpha=0.10,
            label="5th--95th percentiles",
        )
        axis.fill_between(
            median.index,
            q25.values,
            q75.values,
            color=color,
            alpha=0.25,
            label="IQR",
        )
        axis.plot(
            median.index,
            median.values,
            linewidth=2.4,
            color=color,
            label="median",
        )
        axis.legend(fontsize=8)
    if log_scale:
        axis.set_yscale("log")
    axis.set_title("{} ({} direction-block records)".format(family, len(subset)))
    axis.set_ylabel("s_SD(block, direction)")
    axis.grid(alpha=0.25, which="both")


def plot_scales(frame: pd.DataFrame, path: Path, log_scale: bool) -> None:
    """Plot SD scales in one panel per family, in linear or logarithmic form."""
    figure, axes = plt.subplots(3, 1, figsize=(13, 13), sharex=True)
    for axis, family in zip(axes, FAMILIES):
        _plot_family_scale_panel(axis, frame, family, log_scale)
    axes[-1].set_xlabel("Decoder block index (output hook)")
    figure.suptitle(
        "Experiment 0 - natural directional scale{}".format(
            " (log scale)" if log_scale else ""
        )
    )
    figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_sd_mad_ratio(frame: pd.DataFrame, path: Path) -> None:
    """Summarize SD/MAD tail sensitivity with dispersion across directions."""
    prepared = frame.copy()
    prepared["sd_over_mad"] = prepared["sd"] / prepared["mad_corrected"].replace(
        0.0, np.nan
    )
    figure, axis = plt.subplots(figsize=(11, 6))
    for family in FAMILIES:
        grouped = prepared.loc[
            prepared["direction_family"] == family
        ].groupby("decoder_block_index")["sd_over_mad"]
        median = grouped.median()
        q25 = grouped.quantile(0.25)
        q75 = grouped.quantile(0.75)
        color = FAMILY_COLORS[family]
        axis.plot(median.index, median.values, color=color, label=family)
        axis.fill_between(
            median.index, q25.values, q75.values, color=color, alpha=0.14
        )
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set_yscale("log")
    axis.set_xlabel("Decoder block index (output hook)")
    axis.set_ylabel("Median s_SD / s_MAD_corrected (IQR ribbon)")
    axis.set_title("Experiment 0 - SD/MAD tail-sensitivity diagnostic")
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _positive_plot_limits(frame: pd.DataFrame) -> List[float]:
    values = np.concatenate(
        [
            frame.loc[frame["sd"] > 0.0, "sd"].to_numpy(dtype=float),
            frame.loc[
                frame["mad_corrected"] > 0.0, "mad_corrected"
            ].to_numpy(dtype=float),
        ]
    )
    low = float(values.min()) * 0.75
    high = float(values.max()) * 1.35
    return [low, high]


def plot_sd_vs_mad(frame: pd.DataFrame, path: Path) -> None:
    """Compare SD and MAD in three readable, independently scaled panels."""
    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    for axis, family in zip(axes, FAMILIES):
        subset = frame.loc[frame["direction_family"] == family]
        limits = _positive_plot_limits(subset)
        axis.scatter(
            subset["sd"],
            subset["mad_corrected"],
            s=13 if family != "renewed_noise" else 5,
            alpha=0.48 if family != "renewed_noise" else 0.16,
            color=FAMILY_COLORS[family],
            edgecolors="none",
            rasterized=True,
        )
        axis.plot(limits, limits, "k--", linewidth=1, label="SD = MAD")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_title("{} (n={})".format(family, len(subset)))
        axis.set_xlabel("s_SD")
        axis.grid(alpha=0.22, which="both")
        axis.legend(fontsize=8)
    axes[0].set_ylabel("s_MAD_corrected")
    figure.suptitle("Experiment 0 - direction-level SD versus corrected MAD")
    figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_bootstrap_stability(frame: pd.DataFrame, path: Path) -> None:
    """Show median and IQR of phrase-bootstrap SD uncertainty."""
    figure, axis = plt.subplots(figsize=(11, 6))
    line_handles = []
    for family in FAMILIES:
        grouped = frame.loc[
            frame["direction_family"] == family
        ].groupby("decoder_block_index")["sd_bootstrap_cv"]
        median = grouped.median()
        q25 = grouped.quantile(0.25)
        q75 = grouped.quantile(0.75)
        color = FAMILY_COLORS[family]
        line = axis.plot(median.index, median.values, color=color, label=family)[0]
        line_handles.append(line)
        axis.fill_between(
            median.index, q25.values, q75.values, color=color, alpha=0.14
        )
    axis.set_xlabel("Decoder block index (output hook)")
    axis.set_ylabel("Bootstrap CV of s_SD (median and IQR)")
    axis.set_title("Experiment 0 - phrase-bootstrap stability")
    axis.grid(alpha=0.25)
    # Explicit handles avoid a rendering bug in older Matplotlib versions when
    # several translucent fill_between artists are present.
    axis.legend(line_handles, FAMILIES, loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _representative_layers(layers: List[int]) -> List[int]:
    """Choose non-boundary early, middle and late blocks deterministically."""
    if len(layers) < 3:
        return layers
    early = layers[1] if len(layers) > 3 else layers[0]
    middle = layers[len(layers) // 2]
    late = layers[-2] if len(layers) > 3 else layers[-1]
    return list(dict.fromkeys([early, middle, late]))


def _evenly_spaced_direction_ids(subset: pd.DataFrame, maximum: int) -> List[str]:
    """Sample the ordered direction list across its full range, not its head."""
    identifiers = sorted(str(value) for value in subset["direction_id"])
    if len(identifiers) <= maximum:
        return identifiers
    indices = np.linspace(0, len(identifiers) - 1, maximum).astype(int)
    return [identifiers[index] for index in indices]


def _standardized_pool(archive: np.lib.npyio.NpzFile, ids: List[str]) -> np.ndarray:
    pooled = []
    for direction_id in ids:
        values = archive[direction_id].astype(np.float64)
        scale = values.std(ddof=1)
        if scale > 0.0:
            pooled.append((values - values.mean()) / scale)
    if not pooled:
        raise ValueError("no non-constant projection distribution to plot")
    return np.concatenate(pooled)


def plot_family_distributions(
    frame: pd.DataFrame, archive_path: Path, path: Path, max_directions: int = 25
) -> None:
    """Compare standardized shapes at deterministic early/middle/late blocks.

    Each cell pools equally sized, individually standardized direction samples.
    All concept and fixed-random directions are used (ten per block); renewed
    noise uses 25 IDs evenly spaced across its ordered list instead of the first
    25. Shared axes and a standard-normal reference make panels comparable.
    """
    archive = np.load(str(archive_path))
    layers = sorted(int(value) for value in frame["decoder_block_index"].unique())
    selected_layers = _representative_layers(layers)
    bins = np.linspace(-5.0, 5.0, 81)
    centers = 0.5 * (bins[:-1] + bins[1:])
    normal_density = np.exp(-0.5 * centers ** 2) / np.sqrt(2.0 * np.pi)
    figure, axes = plt.subplots(
        len(FAMILIES), len(selected_layers), figsize=(15, 11), sharex=True, sharey=True
    )
    axes = np.asarray(axes).reshape(len(FAMILIES), len(selected_layers))
    for row_index, family in enumerate(FAMILIES):
        for column_index, layer in enumerate(selected_layers):
            axis = axes[row_index, column_index]
            subset = frame.loc[
                (frame["direction_family"] == family)
                & (frame["decoder_block_index"] == layer)
            ]
            ids = _evenly_spaced_direction_ids(subset, max_directions)
            standardized = _standardized_pool(archive, ids)
            density, _ = np.histogram(standardized, bins=bins, density=True)
            tail_rate = float(np.mean(np.abs(standardized) > 3.0))
            axis.plot(
                centers,
                density,
                color=FAMILY_COLORS[family],
                linewidth=1.8,
                label="empirical",
            )
            axis.plot(
                centers,
                normal_density,
                "k--",
                linewidth=1.0,
                label="N(0,1)",
            )
            axis.set_title(
                "{} - block {}\n{} directions; P(|z|>3)={:.3f}".format(
                    family, layer, len(ids), tail_rate
                ),
                fontsize=10,
            )
            axis.grid(alpha=0.20)
            if row_index == len(FAMILIES) - 1:
                axis.set_xlabel("Standardized natural projection")
            if column_index == 0:
                axis.set_ylabel("Density")
            if row_index == 0 and column_index == 0:
                axis.legend(fontsize=8)
    archive.close()
    figure.suptitle(
        "Experiment 0 - projection shapes at early, middle and late blocks"
    )
    figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_concept_scale_heatmap(frame: pd.DataFrame, path: Path) -> None:
    """Expose concept-by-block heterogeneity hidden by family summaries."""
    concepts = frame.loc[frame["direction_family"] == "concept"].copy()
    matrix = concepts.pivot(
        index="concept", columns="decoder_block_index", values="sd"
    ).sort_index()
    log_matrix = np.log10(np.maximum(matrix.to_numpy(dtype=float), 1e-12))
    figure, axis = plt.subplots(figsize=(14, 6))
    image = axis.imshow(log_matrix, origin="upper", aspect="auto", cmap="viridis")
    axis.set_yticks(np.arange(len(matrix.index)))
    axis.set_yticklabels(matrix.index)
    axis.set_xticks(np.arange(len(matrix.columns)))
    axis.set_xticklabels(matrix.columns, fontsize=7)
    axis.set_xlabel("Decoder block index (output hook)")
    axis.set_ylabel("Concept")
    axis.set_title("Experiment 0 - concept directional scales")
    figure.colorbar(image, ax=axis, label="log10(s_SD)")
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_relative_scales(frame: pd.DataFrame, path: Path) -> None:
    """Compare concepts and renewed noise to the fixed-random scale baseline."""
    random_median = frame.loc[
        frame["direction_family"] == "fixed_random"
    ].groupby("decoder_block_index")["sd"].median()
    figure, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)

    concepts = frame.loc[frame["direction_family"] == "concept"]
    for concept, group in concepts.groupby("concept"):
        ordered = group.sort_values("decoder_block_index")
        blocks = ordered["decoder_block_index"].astype(int)
        baseline = blocks.map(random_median).to_numpy(dtype=float)
        axes[0].plot(
            blocks,
            ordered["sd"].to_numpy(dtype=float) / baseline,
            linewidth=1.4,
            label=str(concept),
        )
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("concept s_SD / fixed-random median s_SD")
    axes[0].set_title("Concept alignment relative to isotropic random directions")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(alpha=0.25, which="both")

    noise_grouped = frame.loc[
        frame["direction_family"] == "renewed_noise"
    ].groupby("decoder_block_index")["sd"]
    noise_median = noise_grouped.median() / random_median
    noise_q25 = noise_grouped.quantile(0.25) / random_median
    noise_q75 = noise_grouped.quantile(0.75) / random_median
    axes[1].fill_between(
        noise_median.index,
        noise_q25.values,
        noise_q75.values,
        color=FAMILY_COLORS["renewed_noise"],
        alpha=0.22,
        label="renewed-noise IQR",
    )
    axes[1].plot(
        noise_median.index,
        noise_median.values,
        color=FAMILY_COLORS["renewed_noise"],
        linewidth=2.2,
        label="renewed-noise median",
    )
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Decoder block index (output hook)")
    axes[1].set_ylabel("renewed-noise s_SD / fixed-random median s_SD")
    axes[1].set_title("Control-family agreement")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    figure.suptitle("Experiment 0 - scales relative to fixed-random baseline")
    figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    figure.savefig(path, dpi=220)
    plt.close(figure)


def create_all_plots(
    statistics_path: Path, projections_path: Path, output_dir: Path
) -> None:
    """Recreate every documented figure from persisted Experiment 0 outputs."""
    frame = load_records(statistics_path)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "directional_scales.csv"
    if not csv_path.exists():
        # A fresh run needs the convenient tabular export. Replotting an older
        # run must not rewrite that scientific artifact with another pandas
        # version merely because figures are being regenerated.
        frame.to_csv(csv_path, index=False)
    plot_scales(frame, figure_dir / "experiment_0_scales_sd.png", False)
    plot_scales(frame, figure_dir / "experiment_0_scales_sd_log.png", True)
    plot_sd_vs_mad(frame, figure_dir / "experiment_0_sd_vs_mad.png")
    plot_sd_mad_ratio(frame, figure_dir / "experiment_0_sd_over_mad.png")
    plot_bootstrap_stability(
        frame, figure_dir / "experiment_0_bootstrap_stability.png"
    )
    plot_family_distributions(
        frame,
        projections_path,
        figure_dir / "experiment_0_projection_distributions.png",
    )
    plot_concept_scale_heatmap(
        frame, figure_dir / "experiment_0_concept_scale_heatmap.png"
    )
    plot_relative_scales(frame, figure_dir / "experiment_0_relative_scales.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Experiment 0 results")
    parser.add_argument("--statistics", required=True)
    parser.add_argument("--projections", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    create_all_plots(
        Path(args.statistics), Path(args.projections), Path(args.output_dir)
    )


if __name__ == "__main__":
    main()
