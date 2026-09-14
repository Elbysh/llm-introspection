"""Direct classification analysis against sham; no Experiment 1 results required."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from experiment_0_calibration.prepare_material import file_sha256, read_jsonl

from .protocol_config import read_json, write_json

CONDITION_FIELDS = ("condition_id", "family", "intervention_id", "layer", "dose_axis", "dose", "scale_statistic")


def summarize_classification(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["family"] != "sham":
            groups[row["condition_id"]].append(row)
    result = []
    for condition_id, trials in groups.items():
        descriptors = {tuple(t[k] for k in CONDITION_FIELDS) for t in trials}
        if len(descriptors) != 1:
            raise ValueError(f"Inconsistent condition descriptors: {condition_id}")
        def average(field, trial_rows=trials):
            return sum(trial[field] for trial in trial_rows) / len(trial_rows)
        js = [t["js"] for t in trials if t["js"] is not None]
        result.append({**{k: trials[0][k] for k in CONDITION_FIELDS},
                       "evaluation_set_id": hashlib.sha256(json.dumps(sorted({t["example_id"] for t in trials})).encode()).hexdigest(),
                       "n_trials": len(trials), "accuracy": average("accuracy"),
                       "n_correct": sum(t["accuracy"] for t in trials),
                       "n_sham_correct": sum(t["sham_accuracy"] for t in trials),
                       "sham_accuracy": average("sham_accuracy"),
                       "delta_accuracy": average("delta_accuracy"),
                       "accuracy_loss": -average("delta_accuracy"),
                       "correct_margin": average("correct_margin"),
                       "delta_correct_margin": average("delta_correct_margin"),
                       "invalid_rate": 1-average("valid_response"),
                       "forced_label_accuracy": average("forced_label_accuracy"),
                       "js_mean": sum(js)/len(js) if js else None, "n_js": len(js)})
    return result



def write_csv(path, rows, empty_fields=()):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else list(empty_fields))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = defaultdict(list)
    for row in summary:
        groups[(row["layer"], row["dose_axis"], row["scale_statistic"])].append(row)
    for (layer, axis, statistic), rows in groups.items():
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        trajectories = defaultdict(list)
        for row in rows:
            trajectories[(row["family"], row["intervention_id"], row["evaluation_set_id"])].append(row)
        for (family, direction, cohort), values in sorted(trajectories.items()):
            values.sort(key=lambda r: r["dose"])
            for ax, metric in zip(axes, ("delta_accuracy", "delta_correct_margin", "invalid_rate", "js_mean")):
                ax.plot([r["dose"] for r in values], [r[metric] for r in values],
                        "o-", label=f"{family}: {direction} ({cohort[:6]})")
                ax.set_xlabel(axis)
                ax.set_ylabel(metric)
                ax.axhline(0, color="grey", linestyle="--", linewidth=.7)
                ax.grid(alpha=.2)
        axes[-1].legend(fontsize=6)
        fig.suptitle(f"Experiment 4 — decoder output {layer}, {axis}, {statistic}")
        fig.tight_layout()
        fig.savefig(output / f"classification__block_{layer:02d}__{axis}__{statistic}.png", dpi=160)
        plt.close(fig)

def analyze(run_dir, output):
    run_dir, output = Path(run_dir), Path(output)
    if output.exists():
        raise FileExistsError(f"Analysis directory already exists: {output}")
    complete = read_json(run_dir / "complete.json")
    if file_sha256(run_dir / "trials.jsonl") != complete["trials_sha256"]:
        raise ValueError("Trial file changed or is incomplete")
    rows = read_jsonl(run_dir / "trials.jsonl")
    if len(rows) != complete["n_rows"] or not rows:
        raise ValueError("Invalid completed trial count")
    summary = summarize_classification(rows)
    output.mkdir(parents=True)
    write_csv(output / "classification_summary.csv", summary)
    # Mapping-specific and concept-specific views expose averaging artifacts.
    subgroups = defaultdict(list)
    for row in rows:
        if row["family"] != "sham":
            subgroups[(row["probed_concept"], row["yes_label"])].append(row)
    by_mapping = []
    for (concept, yes_label), subset in subgroups.items():
        by_mapping.extend({**r, "probed_concept": concept, "yes_label": yes_label}
                          for r in summarize_classification(subset))
    write_csv(output / "classification_by_concept_and_mapping.csv", by_mapping)
    plot_results(summary, output)
    write_json(output / "analysis_manifest.json", {
        "source_trials_sha256": complete["trials_sha256"],
        "scope": "classification_only; comparison_to_experiment_1_excluded_by_user",
        "analysis": "descriptive; no regression or significance claim",
        "n_conditions": len(summary),
    })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    analyze(args.run, args.output)


if __name__ == "__main__":
    main()
