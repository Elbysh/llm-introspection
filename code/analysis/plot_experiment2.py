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


def plot(root, output=None):
    output = output or root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    paths = list(root.glob("layer_*/summary.json")) or [root / "summary.json"]
    records = []
    for path in paths:
        if (root / "snapshot.json.gz").exists():
            with gzip.open(root / "snapshot.json.gz", "rt") as stream:
                summary = json.load(stream)
        else:
            summary = json.loads(path.read_text())
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
    plot_details(data, root, output)
    return output


def plot_details(data, root, output):
    from experiment2_diagnostics import plot_conversions
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
    if (root / 'alpha_to_z.csv').exists():
        plot_conversions(pd.read_csv(root / 'alpha_to_z.csv'), output)
    if (root / 'edge_case_scores.csv.gz').exists():
        frame = pd.read_csv(root / 'edge_case_scores.csv.gz')
        fig, axes = plt.subplots(2, 3, figsize=(15,8), layout='constrained')
        for col, (layer, family) in enumerate([(0,'random'),(12,'noise'),(20,'concept')]):
            for row, mapping in enumerate(['XY','YX']):
                ax = axes[row,col]
                part = frame[frame.layer.eq(layer) & frame.mapping.eq(mapping)]
                ax.hist([part.score,part.sham_score], bins=25, density=True, label=['Perturbed','Matched clean'], alpha=.7)
                ax.axvline(0, color='black', ls='--')
                ax.set(title=f'L{layer}, {family}, alpha=128, {mapping}', xlabel='Recoded intervention score', ylabel='Density')
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=2)
        fig.suptitle('Hit-rate edge cases: matched clean scores retain intervention-reference weights')
        fig.savefig(output / 'detection_edge_cases.png', dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(plot(args.input_dir, args.output_dir))
