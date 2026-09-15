#!/usr/bin/env python3
"""Plot detection-only Experiment 2 summaries; never load capability data."""

import argparse
import json
import gzip
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def layer_root(root):
    candidate = root / "layers"
    return candidate if candidate.is_dir() else root


def plot(root, output=None):
    output = output or root
    output.mkdir(parents=True, exist_ok=True)
    source = layer_root(root)
    paths = list(source.glob("layer_*/summary.json")) or [source / "summary.json"]
    snapshot = root / "snapshot.json.gz"
    records = []
    if snapshot.exists():
        with gzip.open(snapshot, "rt") as stream:
            summaries = [json.load(stream)]
    else:
        summaries = [json.loads(path.read_text()) for path in paths]
    for summary in summaries:
        if summary.get("experiment") != 2:
            raise ValueError("Expected detection Experiment 2 summaries")
        for cell in summary["cells"]:
            record = {k: v for k, v in cell.items() if k not in ("ci95", "by_mapping")}
            interval = cell["ci95"].get("auroc")
            record["auroc_ci_low"], record["auroc_ci_high"] = (
                interval or [float("nan")] * 2
            )
            records.append(record)
    data = pd.DataFrame(records)
    data.to_csv(output / "detection_cells.csv", index=False)
    matchings = list(data.matching.unique())
    families = list(data.family.unique())
    fig, axes = plt.subplots(
        len(matchings),
        len(families),
        figsize=(4 * len(families), 4 * len(matchings)),
        squeeze=False,
        layout="constrained",
    )
    for i, matching in enumerate(matchings):
        for j, family in enumerate(families):
            part = data[data.matching.eq(matching) & data.family.eq(family)].pivot(
                index="layer", columns="dose", values="auroc"
            )
            image = axes[i, j].imshow(
                part, aspect="auto", vmin=0, vmax=1, cmap="RdBu_r"
            )
            axes[i, j].set(
                title=f"{family} · {matching}", xlabel="Dose", ylabel="Layer"
            )
            axes[i, j].set_xticks(
                range(len(part.columns)), [f"{d:g}" for d in part.columns], rotation=60
            )
            axes[i, j].set_yticks(range(len(part.index)), [str(v) for v in part.index])
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Detection AUROC")
    fig.suptitle("Experiment 2: intervention versus clean detection sham")
    fig.savefig(output / "detection_auroc.png", dpi=180)
    plt.close(fig)
    plot_d_prime(data, output)
    plot_details(data, output)
    return output


def plot_d_prime(data, output):
    columns = [
        "layer",
        "family",
        "matching",
        "dose",
        "d_prime",
        "criterion",
        "hit_rate",
        "false_alarm_rate",
    ]
    data[columns].to_csv(output / "d_prime.csv", index=False)
    matchings = list(data.matching.unique())
    families = list(data.family.unique())
    limit = max(float(data["d_prime"].abs().max()), 1.0)
    fig, axes = plt.subplots(
        len(matchings),
        len(families),
        figsize=(4 * len(families), 4 * len(matchings)),
        squeeze=False,
        layout="constrained",
    )
    for i, matching in enumerate(matchings):
        for j, family in enumerate(families):
            part = data[data.matching.eq(matching) & data.family.eq(family)].pivot(
                index="layer", columns="dose", values="d_prime"
            )
            image = axes[i, j].imshow(
                part,
                aspect="auto",
                vmin=-limit,
                vmax=limit,
                cmap="RdBu_r",
            )
            axes[i, j].set(
                title=f"{family} · {matching}",
                xlabel="Dose",
                ylabel="Layer",
            )
            axes[i, j].set_xticks(
                range(len(part.columns)), [f"{d:g}" for d in part.columns], rotation=60
            )
            axes[i, j].set_yticks(range(len(part.index)), [str(v) for v in part.index])
    fig.colorbar(image, ax=axes.ravel().tolist(), label="d'")
    fig.suptitle("Experiment 2: signal-detection sensitivity d'")
    fig.savefig(output / "d_prime.png", dpi=180)
    plt.close(fig)


def plot_details(data, output):
    selected = [layer for layer in (0, 12, 20) if layer in set(data.layer)]
    if not selected:
        selected = [int(data.layer.min())]
    matchings = list(data.matching.unique())
    fig, axes = plt.subplots(len(selected), len(matchings), squeeze=False,
                             figsize=(5 * len(matchings), 3.5 * len(selected)), layout='constrained')
    for i, layer in enumerate(selected):
        for j, matching in enumerate(matchings):
            ax = axes[i, j]
            for family in data.family.unique():
                part = data[data.layer.eq(layer) & data.matching.eq(matching) & data.family.eq(family)].sort_values('dose')
                line, = ax.plot(part.dose, part.auroc, label=family)
                ax.fill_between(part.dose, part.auroc_ci_low, part.auroc_ci_high, color=line.get_color(), alpha=.15)
            ax.axhline(.5, color='gray', ls=':')
            ax.set(xscale='log', ylim=(0,1), xlabel=matching, ylabel='AUROC', title=f'Layer {layer}')
            ax.grid(alpha=.15)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4)
    fig.suptitle('Detection dose response: stored pointwise 95% crossed-bootstrap intervals')
    fig.savefig(output / 'detection_uncertainty.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(selected), squeeze=False, figsize=(5*len(selected),4), layout='constrained')
    for ax, layer in zip(axes[0], selected):
        for family in data.family.unique():
            part = data[data.layer.eq(layer) & data.matching.eq('alpha') & data.family.eq(family)].sort_values('dose')
            line, = ax.plot(part.dose, part.hit_rate, label=f'{family}: hit')
            ax.plot(part.dose, part.false_alarm_rate, ls='--', color=line.get_color())
        ax.set(xscale='log', ylim=(-.02,1.02), xlabel='Raw alpha', ylabel='Rate', title=f'Layer {layer}')
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4)
    fig.suptitle('Detection at score > 0: solid hits, dashed false alarms (descriptive rates)')
    fig.savefig(output / 'detection_hit_rates.png', dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(plot(args.input_dir, args.output_dir))
