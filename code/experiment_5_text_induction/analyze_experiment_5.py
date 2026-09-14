"""Tables, paired false-positive differences, uncertainty and descriptive figures."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from experiment_0_calibration.prepare_material import file_sha256, read_jsonl

from .measure_presence import paired_blocks, paired_uncertainty, presence_metrics
from .protocol_config import read_json, write_json


def write_csv(path, rows, empty_fields=()):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]) if rows else list(empty_fields)
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, settings):
    blocks = paired_blocks(rows)
    main_groups, detail_groups = defaultdict(list), defaultdict(list)
    for row in rows:
        main_groups[(row["condition_id"], row["text_version"])].append(row)
        detail_groups[(row["condition_id"], row["text_version"], row["target_label"], row["present_label"])].append(row)
    fields = ("condition_id", "concept", "layer", "direction_id", "dose_axis", "dose", "dose_role", "scale_statistic")
    main, detail = [], []
    for (_, version), selected in main_groups.items():
        main.append({**{k: selected[0][k] for k in fields}, "text_version": version, **presence_metrics(selected)})
    for (_, version, target, mapping), selected in detail_groups.items():
        detail.append({**{k: selected[0][k] for k in fields}, "text_version": version,
            "target_label": target, "present_label": mapping, **presence_metrics(selected)})
    groups = defaultdict(list)
    for block in blocks:
        groups[("condition", block["condition_id"])].append(block)
        # Pool only on the same dose parametrization and natural-scale statistic.
        groups[("pooled", f"{block['dose_axis']}__{block['scale_statistic']}")].append(block)
    differences = []
    for (scope, key), selected in groups.items():
        differences.append({"scope": scope, "group_id": key,
            "dose_axis": selected[0]["dose_axis"], "scale_statistic": selected[0]["scale_statistic"],
            "n_paired_blocks": len(selected), "ci_level": settings["ci_level"],
            **paired_uncertainty(selected, settings)})
    return main, detail, differences


def plot_results(summary, differences, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = defaultdict(list)
    for row in summary:
        groups[(row["concept"], row["layer"], row["dose_axis"], row["scale_statistic"])].append(row)
    intervals = {r["group_id"]: r for r in differences if r["scope"] == "condition"}
    for index, ((concept, layer, axis, statistic), rows) in enumerate(sorted(groups.items())):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for version in ("neutral", "evocative"):
            values = sorted([r for r in rows if r["text_version"] == version], key=lambda r: r["dose"])
            for ax, metric in zip(axes[:2], ("false_positive_rate", "hit_rate")):
                ax.plot([r["dose"] for r in values], [r[metric] for r in values], "o-", label=version)
                ax.set(xlabel=axis, ylabel=metric, ylim=(0, 1))
                ax.legend()
        neutral = sorted([r for r in rows if r["text_version"] == "neutral"], key=lambda r: r["dose"])
        for row in neutral:
            estimate = intervals[row["condition_id"]]
            point = estimate["delta_false_positive_rate"]
            axes[2].scatter(row["dose"], point, color="tab:purple")
            low, high = estimate["delta_false_positive_rate_ci_low"], estimate["delta_false_positive_rate_ci_high"]
            if low is not None:
                # Percentile intervals need not contain the point estimate.
                axes[2].vlines(row["dose"], low, high, color="tab:purple")
        axes[2].set(xlabel=axis, ylabel="FP evocative - FP neutral", ylim=(-1, 1))
        axes[2].axhline(0, color="grey", linestyle="--")
        for ax in axes:
            ax.grid(alpha=.2)
        fig.suptitle(f"{concept}, decoder output {layer}, {axis}, {statistic}")
        fig.tight_layout()
        fig.savefig(output / f"presence_{index:03d}_block_{layer:02d}_{axis}_{statistic}.png", dpi=160)
        plt.close(fig)


def analyze(run_dir, output):
    run_dir, output = Path(run_dir), Path(output)
    if output.exists():
        raise FileExistsError(f"Analysis already exists: {output}")
    complete = read_json(run_dir / "complete.json")
    if file_sha256(run_dir / "trials.jsonl") != complete["trials_sha256"]:
        raise ValueError("Trial file changed or is incomplete")
    rows = read_jsonl(run_dir / "trials.jsonl")
    if not rows or len(rows) != complete["n_trials"]:
        raise ValueError("Completed trial count mismatch")
    manifest = read_json(run_dir / "run_manifest.json")
    settings = manifest["plan_manifest"]["config"]["statistics"]
    summary, detail, differences = summarize(rows, settings)
    output.mkdir(parents=True)
    write_csv(output / "presence_summary.csv", summary)
    write_csv(output / "mapping_and_position.csv", detail)
    write_csv(output / "paired_text_effects.csv", differences)
    plot_results(summary, differences, output)
    write_json(output / "analysis_manifest.json", {
        "source_trials_sha256": complete["trials_sha256"], "statistics": settings,
        "primary_effect": "false_positive_rate_evocative_minus_neutral_on_paired_sham",
        "uncertainty": "crossed concept/sentence cluster bootstrap, paired text/intervention/mapping/position/doses",
        "ci_type": "pointwise percentile; no multiple-comparison correction",
        "invalid_response_policy": "no affirmative or negative report; incorrect in balanced accuracy",
        "auroc_score": "presence-oriented next-token logit contrast, including invalid discrete answers",
    })
    return summary, differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", help="Use paths.output_dir and paths.analysis_dir from a run configuration")
    args = parser.parse_args()
    if args.config:
        if args.run or args.output:
            parser.error("Use either --config or both --run/--output")
        from .protocol_config import load_config
        config = load_config(args.config)
        analyze(config.path("output_dir"), config.path("analysis_dir"))
    elif args.run and args.output:
        analyze(args.run, args.output)
    else:
        parser.error("Provide --config or both --run and --output")


if __name__ == "__main__":
    main()
