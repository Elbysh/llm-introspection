"""Create focused diagnostics for s(layer, direction)."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import resolve_repo_path


COLORS = {"concept": "#3366cc", "random": "#dd4477", "noise": "#109618"}


def load_table(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        frame = pd.DataFrame(json.load(handle))
    required = {"direction_id", "kind", "layer", "sd", "mad"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("missing calibration columns: {}".format(sorted(missing)))
    return frame


def plot_directional_scale(frame: pd.DataFrame, path: Path, log_scale: bool) -> None:
    """Plot concepts individually and Gaussian controls as median plus IQR."""
    figure, axis = plt.subplots(figsize=(10, 6))
    for concept, group in frame.loc[frame["kind"] == "concept"].groupby("concept"):
        ordered = group.sort_values("layer")
        axis.plot(ordered["layer"], ordered["sd"], marker="o", alpha=0.7, label=str(concept))

    for kind in ("random", "noise"):
        control = frame.loc[frame["kind"] == kind]
        grouped = control.groupby("layer")["sd"]
        # Build columns explicitly for compatibility with the repository's
        # older pandas environments as well as current versions.
        summary = pd.DataFrame(
            {
                "median": grouped.median(),
                "low": grouped.quantile(0.25),
                "high": grouped.quantile(0.75),
            }
        )
        x = summary.index.to_numpy(float)
        axis.plot(
            x,
            summary["median"].to_numpy(),
            marker="s",
            linewidth=2.5,
            color=COLORS[kind],
            label="{} median".format(kind),
        )
        axis.fill_between(
            x,
            summary["low"].to_numpy(),
            summary["high"].to_numpy(),
            color=COLORS[kind],
            alpha=0.15,
            label="{} IQR".format(kind),
        )

    if log_scale:
        axis.set_yscale("log")
    axis.set_xlabel("Decoder block output")
    axis.set_ylabel("s(layer, direction), estimated by SD")
    axis.set_title("Natural directional scale by layer{}".format(" (log scale)" if log_scale else ""))
    axis.grid(alpha=0.25, which="both")
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_sd_vs_mad(frame: pd.DataFrame, path: Path) -> None:
    """Diagnose whether SD is inflated relative to the robust MAD scale."""
    layers = sorted(frame["layer"].unique())
    figure, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 5), squeeze=False)
    for axis, layer in zip(axes[0], layers):
        subset = frame.loc[frame["layer"] == layer]
        for kind, group in subset.groupby("kind"):
            axis.scatter(group["sd"], group["mad"], color=COLORS[kind], alpha=0.75, label=kind)
        limit = float(max(subset["sd"].max(), subset["mad"].max()) * 1.05)
        axis.plot([0, limit], [0, limit], "k--", linewidth=1)
        axis.set_xlim(0, limit)
        axis.set_ylim(0, limit)
        axis.set_title("Layer {}".format(int(layer)))
        axis.set_xlabel("SD")
        axis.set_ylabel("MAD")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("SD versus robust MAD calibration")
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_sd_mad_ratio(frame: pd.DataFrame, path: Path) -> None:
    """Summarize tail sensitivity by direction family and layer."""
    prepared = frame.copy()
    prepared["sd_over_mad"] = prepared["sd"] / prepared["mad"].replace(0.0, np.nan)
    figure, axis = plt.subplots(figsize=(9, 5))
    for kind, group in prepared.groupby("kind"):
        summary = group.groupby("layer")["sd_over_mad"].median().sort_index()
        axis.plot(summary.index, summary.values, marker="o", color=COLORS[kind], label=kind)
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Decoder block output")
    axis.set_ylabel("Median SD / MAD")
    axis.set_title("Sensitivity of directional scale to projection tails")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_projection_distributions(frame: pd.DataFrame, archive_path: Path, path: Path) -> None:
    """Compare representative standardized projection distributions."""
    archive = np.load(str(archive_path))
    layers = sorted(frame["layer"].unique())
    figure, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 4), squeeze=False)
    bins = np.linspace(-5, 5, 81)
    for axis, layer in zip(axes[0], layers):
        subset = frame.loc[frame["layer"] == layer]
        for kind in ("concept", "random", "noise"):
            candidates = subset.loc[subset["kind"] == kind]
            if candidates.empty:
                continue
            direction_id = str(candidates.iloc[0]["direction_id"])
            values = archive[direction_id].astype(np.float64)
            standardized = (values - values.mean()) / (values.std(ddof=1) + 1e-12)
            axis.hist(
                standardized,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.7,
                color=COLORS[kind],
                label=kind,
            )
        axis.set_title("Layer {}".format(int(layer)))
        axis.set_xlabel("Standardized projection")
        axis.set_ylabel("Density")
        axis.set_xlim(-5, 5)
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("Representative natural projection distributions")
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)
    archive.close()


def create_plots(statistics_path: Path, projections_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    frame = load_table(statistics_path)
    frame.to_csv(output_dir / "calibration_table.csv", index=False)
    plot_directional_scale(frame, figure_dir / "directional_scale_by_layer.png", False)
    plot_directional_scale(frame, figure_dir / "directional_scale_by_layer_log.png", True)
    plot_sd_vs_mad(frame, figure_dir / "sd_vs_mad.png")
    plot_sd_mad_ratio(frame, figure_dir / "sd_over_mad_by_layer.png")
    plot_projection_distributions(frame, projections_path, figure_dir / "projection_distributions.png")
    print("Saved calibration table and figures to {}".format(output_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot directional calibration diagnostics")
    parser.add_argument("--statistics", required=True)
    parser.add_argument("--projections", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    create_plots(
        resolve_repo_path(args.statistics),
        resolve_repo_path(args.projections),
        resolve_repo_path(args.output_dir),
    )


if __name__ == "__main__":
    main()
