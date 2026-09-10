"""Experiment 0, step 5 and analysis: protocol-labelled figures."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FAMILY_COLORS = {
    "concept": "#3366cc",
    "fixed_random": "#dd4477",
    "renewed_noise": "#109618",
}


def load_records(path: Path) -> pd.DataFrame:
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
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("missing Experiment 0 result columns: {}".format(sorted(missing)))
    return frame


def plot_scales(frame: pd.DataFrame, path: Path, log_scale: bool) -> None:
    """Primary SD curves: concepts individually, controls as median and IQR."""
    figure, axis = plt.subplots(figsize=(12, 7))
    concepts = frame.loc[frame["direction_family"] == "concept"]
    for concept, group in concepts.groupby("concept"):
        ordered = group.sort_values("decoder_block_index")
        axis.plot(
            ordered["decoder_block_index"],
            ordered["sd"],
            linewidth=1.4,
            alpha=0.75,
            label=str(concept),
        )
    for family in ("fixed_random", "renewed_noise"):
        grouped = frame.loc[frame["direction_family"] == family].groupby(
            "decoder_block_index"
        )["sd"]
        median = grouped.median()
        low = grouped.quantile(0.25)
        high = grouped.quantile(0.75)
        axis.plot(
            median.index,
            median.values,
            linewidth=2.4,
            color=FAMILY_COLORS[family],
            label="{} median".format(family),
        )
        axis.fill_between(
            median.index,
            low.values,
            high.values,
            color=FAMILY_COLORS[family],
            alpha=0.16,
            label="{} IQR".format(family),
        )
    if log_scale:
        axis.set_yscale("log")
    axis.set_xlabel("Decoder block index (output hook)")
    axis.set_ylabel("s_SD(block, direction)")
    axis.set_title(
        "Experiment 0 — natural directional scale{}".format(
            " (log scale)" if log_scale else ""
        )
    )
    axis.grid(alpha=0.25, which="both")
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_sd_mad_ratio(frame: pd.DataFrame, path: Path) -> None:
    """Sensitivity analysis requested in Experiment 0 section 4.6."""
    prepared = frame.copy()
    prepared["sd_over_mad"] = prepared["sd"] / prepared["mad_corrected"].replace(
        0.0, np.nan
    )
    figure, axis = plt.subplots(figsize=(11, 6))
    for family, group in prepared.groupby("direction_family"):
        summary = group.groupby("decoder_block_index")["sd_over_mad"].median()
        axis.plot(
            summary.index,
            summary.values,
            color=FAMILY_COLORS[family],
            label=family,
        )
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set_yscale("log")
    axis.set_xlabel("Decoder block index (output hook)")
    axis.set_ylabel("Median s_SD / s_MAD_corrected")
    axis.set_title("Experiment 0 — SD/MAD tail-sensitivity diagnostic")
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_bootstrap_stability(frame: pd.DataFrame, path: Path) -> None:
    """Phrase-bootstrap stability, summarized without hiding direction records."""
    figure, axis = plt.subplots(figsize=(11, 6))
    for family, group in frame.groupby("direction_family"):
        summary = group.groupby("decoder_block_index")["sd_bootstrap_cv"].median()
        axis.plot(
            summary.index,
            summary.values,
            color=FAMILY_COLORS[family],
            label=family,
        )
    axis.set_xlabel("Decoder block index (output hook)")
    axis.set_ylabel("Median bootstrap coefficient of variation of s_SD")
    axis.set_title("Experiment 0 — phrase-bootstrap stability")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_family_distributions(
    frame: pd.DataFrame, archive_path: Path, path: Path, max_directions: int = 25
) -> None:
    """Show standardized distributions for every layer and direction family.

    For readability and equal family weighting, at most `max_directions`
    deterministically ordered directions contribute at each layer. Each
    direction is standardized before pooling, so this plot diagnoses shape and
    tails rather than the scale already shown in the primary figure.
    """
    archive = np.load(str(archive_path))
    layers = sorted(int(value) for value in frame["decoder_block_index"].unique())
    bins = np.linspace(-5.0, 5.0, 101)
    families = ["concept", "fixed_random", "renewed_noise"]
    densities: Dict[str, List[np.ndarray]] = {family: [] for family in families}
    for family in families:
        for layer in layers:
            subset = frame.loc[
                (frame["direction_family"] == family)
                & (frame["decoder_block_index"] == layer)
            ].sort_values("direction_id")
            pooled = []
            for direction_id in subset["direction_id"].head(max_directions):
                values = archive[str(direction_id)].astype(np.float64)
                standardized = (values - values.mean()) / (values.std(ddof=1) + 1e-12)
                pooled.append(standardized)
            density, _ = np.histogram(
                np.concatenate(pooled), bins=bins, density=True
            )
            densities[family].append(density)
    archive.close()

    figure, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    extent = [bins[0], bins[-1], min(layers) - 0.5, max(layers) + 0.5]
    for axis, family in zip(axes, families):
        image = axis.imshow(
            np.stack(densities[family]),
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="magma",
        )
        axis.set_ylabel("Decoder block")
        axis.set_title("{} (up to {} directions/block)".format(family, max_directions))
        figure.colorbar(image, ax=axis, label="Density")
    axes[-1].set_xlabel("Standardized natural projection")
    figure.suptitle("Experiment 0 — projection distributions by block and family")
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def create_all_plots(statistics_path: Path, projections_path: Path, output_dir: Path) -> None:
    frame = load_records(statistics_path)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "directional_scales.csv", index=False)
    plot_scales(frame, figure_dir / "experiment_0_scales_sd.png", False)
    plot_scales(frame, figure_dir / "experiment_0_scales_sd_log.png", True)
    plot_sd_mad_ratio(frame, figure_dir / "experiment_0_sd_over_mad.png")
    plot_bootstrap_stability(frame, figure_dir / "experiment_0_bootstrap_stability.png")
    plot_family_distributions(
        frame, projections_path, figure_dir / "experiment_0_projection_distributions.png"
    )


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

