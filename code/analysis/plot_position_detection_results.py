#!/usr/bin/env python3
"""Create paper-ready figures for a completed position-detection experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

SCHEMA_VERSION = 1
CACHE_VERSION = 2
DEFAULT_BOOTSTRAP_REPLICATES = 5000
DEFAULT_BOOTSTRAP_SEED = 42
FIGURE_CAPTIONS = {
    "fig01_control_bias": (
        "Clean-prompt logit preferences and their counterbalanced factorial decomposition. "
        "Diamonds and bars show pair-bootstrap means and 95% percentile intervals."
    ),
    "fig02_accuracy_landscape": (
        "Raw and matched-control-adjusted detection accuracy across injection layers and "
        "strengths, alongside the adjusted exact-tie rate."
    ),
    "fig03_layer_profiles": (
        "Adjusted accuracy, ties, non-tie accuracy, and target-aligned intervention effects. "
        "Bands are 95% pair-bootstrap percentile intervals."
    ),
    "fig04_concept_heterogeneity": (
        "Adjusted detection accuracy by studied concept, injection layer, and strength."
    ),
    "fig05_causal_propagation": (
        "Representational contamination and position-aligned restoration effects at the "
        "predefined representative intervention strength."
    ),
}


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ValueError(f"Missing required result file: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_run(input_dir: Path, main_alpha: float):
    required = {
        name: input_dir / name
        for name in ("manifest.json", "completed.json", "summary.json")
    }
    manifest = read_json(required["manifest.json"])
    completed = read_json(required["completed.json"])
    summary = read_json(required["summary.json"])
    runtime_path = input_dir / "runtime.json"
    runtime = read_json(runtime_path) if runtime_path.exists() else {}

    for name, value in (("manifest", manifest), ("summary", summary)):
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported {name} schema {value.get('schema_version')!r}; "
                f"expected {SCHEMA_VERSION}."
            )
    config = manifest.get("config", {})
    for key in ("concepts", "layers", "alphas"):
        if not config.get(key):
            raise ValueError(f"Manifest is missing non-empty config.{key}.")
    if not math.isfinite(main_alpha) or main_alpha not in config["alphas"]:
        raise ValueError(
            f"Representative alpha {main_alpha:g} is unavailable; "
            f"choose one of {config['alphas']}."
        )

    prompts = manifest.get("prompts", [])
    if not prompts:
        raise ValueError("Manifest has no prompts.")
    configurations_by_pair = {}
    prompt_ids = set()
    for prompt in prompts:
        prompt_id = prompt.get("prompt_id")
        if not prompt_id or prompt_id in prompt_ids:
            raise ValueError(f"Manifest has a missing or duplicate prompt_id: {prompt_id!r}.")
        prompt_ids.add(prompt_id)
        configuration = (prompt.get("content_order"), prompt.get("label_mapping"))
        configurations_by_pair.setdefault(prompt.get("pair_id"), set()).add(configuration)
    expected_configurations = {(order, mapping) for order in ("xy", "yx") for mapping in ("AB", "BA")}
    incomplete_pairs = [
        pair_id
        for pair_id, configurations in configurations_by_pair.items()
        if configurations != expected_configurations
    ]
    if incomplete_pairs:
        raise ValueError(
            "Manifest prompts are not fully counterbalanced for pair(s): "
            + ", ".join(map(str, incomplete_pairs[:5]))
        )
    expected = len(prompts) * len(config["concepts"]) * len(config["layers"]) * len(config["alphas"])
    if not summary.get("complete"):
        raise ValueError("summary.json marks this run as partial; complete the run before plotting.")
    if completed.get("conditions") != expected or completed.get("expected") != expected:
        raise ValueError(
            "completed.json is inconsistent with the manifest: "
            f"found {completed.get('conditions')}/{completed.get('expected')}, expected {expected}."
        )
    if summary.get("completed_conditions") != expected or summary.get("expected_conditions") != expected:
        raise ValueError("summary.json condition counts are inconsistent with the manifest.")

    control_paths = sorted((input_dir / "controls").glob("*.json"))
    condition_paths = sorted((input_dir / "conditions").glob("*/*.json"))
    if len(control_paths) != len(prompts):
        raise ValueError(f"Expected {len(prompts)} controls, found {len(control_paths)}.")
    if len(condition_paths) != expected:
        raise ValueError(f"Expected {expected} condition records, found {len(condition_paths)}.")
    return manifest, completed, summary, runtime, control_paths, condition_paths


def cache_signature(input_dir: Path, condition_count: int, control_count: int) -> dict:
    return {
        "cache_version": CACHE_VERSION,
        "manifest_sha256": sha256(input_dir / "manifest.json"),
        "completed_sha256": sha256(input_dir / "completed.json"),
        "summary_sha256": sha256(input_dir / "summary.json"),
        "condition_count": condition_count,
        "control_count": control_count,
    }


def _mean_or_nan(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def scan_records(manifest: dict, control_paths: list[Path], condition_paths: list[Path]):
    prompts = {prompt["prompt_id"]: prompt for prompt in manifest["prompts"]}
    config = manifest["config"]
    controls = []
    control_values = {}
    for path in control_paths:
        row = read_json(path)
        prompt_id = row.get("prompt_id", path.stem)
        if prompt_id not in prompts:
            raise ValueError(f"Control {path} references unknown prompt {prompt_id!r}.")
        if prompt_id in control_values:
            raise ValueError(f"Duplicate control record for prompt {prompt_id!r}.")
        prompt = prompts[prompt_id]
        control_values[prompt_id] = float(row["L"])
        controls.append(
            (
                prompt_id,
                prompt["pair_id"],
                prompt["content_order"],
                prompt["label_mapping"],
                row["logit_A"],
                row["logit_B"],
                row["L"],
                prompt.get("tokens_between_sentences"),
                prompt["sentences"][0].get("token_count", len(prompt["sentences"][0]["token_indices"])),
                prompt["sentences"][1].get("token_count", len(prompt["sentences"][1]["token_indices"])),
            )
        )
    missing_controls = sorted(set(prompts) - set(control_values))
    if missing_controls:
        raise ValueError(f"Missing control record(s) for prompt(s): {missing_controls[:5]}.")

    condition_rows = []
    injection_rows = []
    diagnostic_totals = {}
    seen_conditions = set()
    allowed_concepts = set(config["concepts"])
    allowed_layers = {int(value) for value in config["layers"]}
    allowed_alphas = {float(value) for value in config["alphas"]}
    endpoint = {"P_at_injection": 0.0, "E_at_injection": 0.0, "E_at_final": 0.0}
    final_layer = 31
    for path in condition_paths:
        row = read_json(path)
        if row.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported condition schema in {path}.")
        prompt_id = row["prompt_id"]
        if prompt_id not in prompts:
            raise ValueError(f"Condition {path} references unknown prompt {prompt_id!r}.")
        prompt = prompts[prompt_id]
        if any(
            row.get(key) != prompt[key]
            for key in ("pair_id", "content_order", "label_mapping")
        ):
            raise ValueError(f"Condition metadata disagrees with manifest prompt in {path}.")
        condition_key = (
            prompt_id,
            row["concept"],
            int(row["layer"]),
            float(row["alpha"]),
        )
        if condition_key in seen_conditions:
            raise ValueError(f"Duplicate condition record for {condition_key!r}.")
        if (
            row["concept"] not in allowed_concepts
            or int(row["layer"]) not in allowed_layers
            or float(row["alpha"]) not in allowed_alphas
        ):
            raise ValueError(f"Condition setting is not declared in the manifest: {path}.")
        seen_conditions.add(condition_key)
        if not math.isclose(float(row["control_L"]), control_values[prompt_id], abs_tol=1e-12):
            raise ValueError(f"Condition control_L disagrees with its clean control in {path}.")
        base = (
            row["pair_id"],
            prompt_id,
            row["concept"],
            int(row["layer"]),
            float(row["alpha"]),
            row["content_order"],
            row["label_mapping"],
        )
        condition_rows.append(base + (row["control_L"], row["S"], row["S_position"]))
        injection_labels = [injection.get("target_label") for injection in row["injections"]]
        injection_positions = [injection.get("target_position") for injection in row["injections"]]
        if sorted(injection_labels) != ["A", "B"] or sorted(injection_positions) != [1, 2]:
            raise ValueError(f"Condition must contain one injection per label and position: {path}.")
        for injection in row["injections"]:
            target_label = injection["target_label"]
            delta = injection["L_adjusted"]
            token_count = len(injection["target_token_indices"])
            realized = injection.get("realized_per_token_l2", [])
            realized_mean = _mean_or_nan(realized)
            intended = float(injection.get("intended_per_token_l2", row["alpha"]))
            frobenius = float(injection.get("realized_frobenius_norm", float("nan")))
            norm_ratio = realized_mean / intended if intended else float("nan")
            frobenius_ratio = frobenius / (intended * math.sqrt(token_count)) if intended else float("nan")
            injection_rows.append(
                base
                + (
                    int(injection["target_position"]),
                    target_label,
                    token_count,
                    injection["L"],
                    row["control_L"],
                    delta,
                    injection["accuracy_raw"],
                    injection["accuracy_adjusted"],
                    float(injection["L"] == 0),
                    float(delta == 0),
                    delta if target_label == "A" else -delta,
                    intended,
                    realized_mean,
                    frobenius,
                    norm_ratio,
                    frobenius_ratio,
                )
            )
        sign = 1.0 if row["label_mapping"] == "AB" else -1.0
        expected_restoration_layers = set(range(int(row["layer"]), final_layer + 1))
        saved_restoration_layers = {int(diagnostic["layer"]) for diagnostic in row["diagnostics"]}
        if saved_restoration_layers != expected_restoration_layers or len(row["diagnostics"]) != len(
            expected_restoration_layers
        ):
            raise ValueError(f"Condition has incomplete or duplicate restoration diagnostics: {path}.")
        for diagnostic in row["diagnostics"]:
            restore_layer = int(diagnostic["layer"])
            p_value = float(diagnostic["P"])
            e_value = float(diagnostic["E"])
            diagnostic_key = (row["pair_id"], int(row["layer"]), float(row["alpha"]), restore_layer)
            totals = diagnostic_totals.setdefault(diagnostic_key, [0.0, 0.0, 0])
            totals[0] += p_value
            totals[1] += sign * e_value
            totals[2] += 1
            if restore_layer == row["layer"]:
                endpoint["P_at_injection"] = max(endpoint["P_at_injection"], abs(p_value))
                endpoint["E_at_injection"] = max(endpoint["E_at_injection"], abs(e_value))
            if restore_layer == final_layer:
                endpoint["E_at_final"] = max(endpoint["E_at_final"], abs(e_value))

    controls_frame = pd.DataFrame.from_records(
        controls,
        columns=[
            "prompt_id", "pair_id", "content_order", "label_mapping", "logit_A", "logit_B", "L",
            "tokens_between_sentences", "first_token_count", "second_token_count",
        ],
    )
    conditions_frame = pd.DataFrame.from_records(
        condition_rows,
        columns=[
            "pair_id", "prompt_id", "concept", "layer", "alpha", "content_order", "label_mapping",
            "control_L", "S", "S_position",
        ],
    )
    injections_frame = pd.DataFrame.from_records(
        injection_rows,
        columns=[
            "pair_id", "prompt_id", "concept", "layer", "alpha", "content_order", "label_mapping",
            "target_position", "target_label", "target_token_count", "L", "control_L", "L_adjusted",
            "accuracy_raw", "accuracy_adjusted", "tie_raw", "tie_adjusted", "target_aligned_delta",
            "intended_per_token_l2", "realized_per_token_l2_mean", "realized_frobenius_norm",
            "per_token_norm_ratio", "frobenius_norm_ratio",
        ],
    )
    diagnostic_rows = [
        key + (totals[0] / totals[2], totals[1] / totals[2], totals[2])
        for key, totals in diagnostic_totals.items()
    ]
    diagnostics_frame = pd.DataFrame.from_records(
        diagnostic_rows,
        columns=[
            "pair_id", "injection_layer", "alpha", "restoration_layer", "P", "E_position",
            "n_observations",
        ],
    )
    return controls_frame, conditions_frame, injections_frame, diagnostics_frame, endpoint


def load_or_build_tables(
    input_dir: Path,
    source_dir: Path,
    manifest: dict,
    control_paths: list[Path],
    condition_paths: list[Path],
    rebuild: bool,
):
    source_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = source_dir / "cache_metadata.json"
    files = {
        "controls": source_dir / "cache_controls.csv.gz",
        "conditions": source_dir / "cache_conditions.csv.gz",
        "injections": source_dir / "cache_injections.csv.gz",
        "diagnostics": source_dir / "cache_diagnostics.csv.gz",
    }
    signature = cache_signature(input_dir, len(condition_paths), len(control_paths))
    cached = read_json(metadata_path) if metadata_path.exists() else None
    if not rebuild and cached and cached.get("signature") == signature and all(path.exists() for path in files.values()):
        frames = {name: pd.read_csv(path) for name, path in files.items()}
        return frames["controls"], frames["conditions"], frames["injections"], frames["diagnostics"], cached["endpoint_checks"]

    controls, conditions, injections, diagnostics, endpoint = scan_records(
        manifest, control_paths, condition_paths
    )
    for name, frame in (
        ("controls", controls),
        ("conditions", conditions),
        ("injections", injections),
        ("diagnostics", diagnostics),
    ):
        frame.to_csv(files[name], index=False, compression="gzip")
    atomic_json(metadata_path, {"signature": signature, "endpoint_checks": endpoint})
    return controls, conditions, injections, diagnostics, endpoint


def _percentile_interval(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan")
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


def _bootstrap_weights(pair_ids: list[str], replicates: int, seed: int) -> np.ndarray:
    if replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive.")
    rng = np.random.default_rng(seed)
    count = len(pair_ids)
    return rng.multinomial(count, np.full(count, 1 / count), size=replicates).astype(float)


def bootstrap_injection_metrics(
    frame: pd.DataFrame, group_columns: list[str], replicates: int, seed: int
) -> pd.DataFrame:
    working = frame.copy()
    if not group_columns:
        working["_all"] = "all"
        group_columns = ["_all"]
    working["non_tie"] = 1.0 - working["tie_adjusted"]
    working["non_tie_correct"] = working["accuracy_adjusted"] * working["non_tie"]
    metrics = [
        "accuracy_raw", "accuracy_adjusted", "tie_adjusted", "non_tie_correct",
        "non_tie", "target_aligned_delta",
    ]
    pair_ids = sorted(working["pair_id"].unique())
    weights = _bootstrap_weights(pair_ids, replicates, seed)
    grouped = working.groupby(group_columns + ["pair_id"], observed=True)[metrics].agg(["sum", "count"])
    output = []
    for key, group in grouped.groupby(level=group_columns, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        group = group.droplevel(group_columns).reindex(pair_ids, fill_value=0)
        counts = group[("accuracy_raw", "count")].to_numpy(float)
        denominator = weights @ counts
        row = {column: value for column, value in zip(group_columns, key)}
        row["n_pairs"] = len(pair_ids)
        row["n_injections"] = int(counts.sum())
        bootstrap_values = {}
        for metric in ("accuracy_raw", "accuracy_adjusted", "tie_adjusted", "target_aligned_delta"):
            sums = group[(metric, "sum")].to_numpy(float)
            point = sums.sum() / counts.sum()
            draws = (weights @ sums) / denominator
            bootstrap_values[metric] = draws
            low, high = _percentile_interval(draws)
            name = "tie_rate_adjusted" if metric == "tie_adjusted" else metric
            row[name] = float(point)
            row[f"{name}_ci_low"] = low
            row[f"{name}_ci_high"] = high
        non_tie_sums = group[("non_tie_correct", "sum")].to_numpy(float)
        non_tie_counts = group[("non_tie", "sum")].to_numpy(float)
        point_denominator = non_tie_counts.sum()
        row["non_tie_accuracy"] = float(non_tie_sums.sum() / point_denominator) if point_denominator else float("nan")
        draws_denominator = weights @ non_tie_counts
        draws = np.divide(
            weights @ non_tie_sums,
            draws_denominator,
            out=np.full(replicates, np.nan),
            where=draws_denominator != 0,
        )
        row["non_tie_accuracy_ci_low"], row["non_tie_accuracy_ci_high"] = _percentile_interval(draws)
        output.append(row)
    result = pd.DataFrame(output)
    return result.drop(columns=["_all"], errors="ignore")


def bootstrap_means(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_columns: list[str],
    replicates: int,
    seed: int,
    weight_column: str | None = None,
) -> pd.DataFrame:
    working = frame.copy()
    if not group_columns:
        working["_all"] = "all"
        group_columns = ["_all"]
    pair_ids = sorted(working["pair_id"].unique())
    weights = _bootstrap_weights(pair_ids, replicates, seed)
    observation_weights = (
        working[weight_column].astype(float)
        if weight_column is not None
        else pd.Series(1.0, index=working.index)
    )
    aggregate_columns = []
    for value in value_columns:
        valid = working[value].notna()
        sum_column = f"__sum_{value}"
        count_column = f"__count_{value}"
        working[sum_column] = working[value].fillna(0.0) * observation_weights
        working[count_column] = observation_weights.where(valid, 0.0)
        aggregate_columns.extend([sum_column, count_column])
    grouped = working.groupby(group_columns + ["pair_id"], observed=True)[aggregate_columns].sum()
    output = []
    for key, group in grouped.groupby(level=group_columns, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        group = group.droplevel(group_columns).reindex(pair_ids, fill_value=0)
        row = {column: value for column, value in zip(group_columns, key)}
        row["n_pairs"] = len(pair_ids)
        for value in value_columns:
            sums = group[f"__sum_{value}"].to_numpy(float)
            counts = group[f"__count_{value}"].to_numpy(float)
            row[f"{value}_n"] = int(counts.sum())
            row[value] = float(sums.sum() / counts.sum())
            draw_counts = weights @ counts
            draws = np.divide(
                weights @ sums,
                draw_counts,
                out=np.full(replicates, np.nan),
                where=draw_counts != 0,
            )
            row[f"{value}_ci_low"], row[f"{value}_ci_high"] = _percentile_interval(draws)
        output.append(row)
    result = pd.DataFrame(output)
    return result.drop(columns=["_all"], errors="ignore")


def control_contrasts(controls: pd.DataFrame) -> pd.DataFrame:
    values = controls.copy()
    values["configuration"] = values["content_order"] + "_" + values["label_mapping"]
    pivot = values.pivot(index="pair_id", columns="configuration", values="L")
    required = ["xy_AB", "xy_BA", "yx_AB", "yx_BA"]
    missing = [column for column in required if column not in pivot or pivot[column].isna().any()]
    if missing:
        raise ValueError(f"Controls are not fully counterbalanced; missing {missing}.")
    a, b, c, d = (pivot[column] for column in required)
    return pd.DataFrame(
        {
            "pair_id": pivot.index,
            "A_label_intercept": (a + b + c + d) / 4,
            "first_position_preference": (a - b + c - d) / 4,
            "x_vs_y_content_preference": (a - b - c + d) / 4,
            "order_residual": (a + b - c - d) / 4,
        }
    ).reset_index(drop=True)


def bootstrap_contrast_summary(contrasts: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    metrics = [column for column in contrasts.columns if column != "pair_id"]
    weights = _bootstrap_weights(sorted(contrasts["pair_id"]), replicates, seed)
    rows = []
    ordered = contrasts.sort_values("pair_id")
    for metric in metrics:
        values = ordered[metric].to_numpy(float)
        draws = (weights @ values) / len(values)
        low, high = _percentile_interval(draws)
        rows.append(
            {
                "contrast": metric,
                "n_pairs": len(values),
                "mean": float(values.mean()),
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


def heatmap(
    ax: plt.Axes,
    frame: pd.DataFrame,
    row: str,
    column: str,
    value: str,
    rows: list,
    columns: list,
    title: str,
    cmap,
    norm=None,
    vmin=None,
    vmax=None,
    annotate: bool = True,
    formatter=lambda value: f"{value:.0%}",
):
    matrix = frame.pivot(index=row, columns=column, values=value).reindex(index=rows, columns=columns)
    image = ax.imshow(matrix.to_numpy(float), aspect="auto", cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(columns)), [f"{item:g}" for item in columns])
    ax.set_yticks(range(len(rows)), [str(item) for item in rows])
    ax.set_xlabel(column.replace("_", " ").title())
    ax.set_ylabel(row.replace("_", " ").title())
    ax.set_title(title)
    if annotate:
        for y in range(len(rows)):
            for x in range(len(columns)):
                number = matrix.iloc[y, x]
                if pd.notna(number):
                    ax.text(x, y, formatter(number), ha="center", va="center", fontsize=7)
    return image


def plot_control_bias(
    controls: pd.DataFrame,
    control_groups: pd.DataFrame,
    contrasts: pd.DataFrame,
    contrast_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    values = np.sort(controls["L"].to_numpy(float))
    axes[0].hist(values, bins=min(20, max(5, len(values) // 4)), color="#4477AA", alpha=0.75)
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set(title="Clean-control label preference", xlabel=r"$L_{control}=logit_A-logit_B$", ylabel="Count")
    twin = axes[0].twinx()
    twin.step(values, np.arange(1, len(values) + 1) / len(values), where="post", color="#CC6677")
    twin.set_ylabel("ECDF")
    panel_label(axes[0], "A")

    configurations = [("xy", "AB"), ("xy", "BA"), ("yx", "AB"), ("yx", "BA")]
    grouped_values = [
        controls.loc[(controls["content_order"] == order) & (controls["label_mapping"] == mapping), "L"].to_numpy()
        for order, mapping in configurations
    ]
    labels = [f"{order}/{mapping}" for order, mapping in configurations]
    axes[1].boxplot(grouped_values, tick_labels=labels, showfliers=False)
    for index, data in enumerate(grouped_values, start=1):
        jitter = np.linspace(-0.10, 0.10, len(data))
        axes[1].scatter(index + jitter, data, s=10, alpha=0.45, color="#4477AA")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set(title="Bias by counterbalancing cell", xlabel="Content order / label mapping", ylabel=r"$L_{control}$")
    panel_label(axes[1], "B")

    names = [column for column in contrasts.columns if column != "pair_id"]
    data = [contrasts[column].to_numpy() for column in names]
    readable = ["A-label\nintercept", "First-position\npreference", "x vs y\ncontent", "Order\nresidual"]
    axes[2].boxplot(data, tick_labels=readable, showfliers=False)
    for index, values_for_metric in enumerate(data, start=1):
        axes[2].scatter(index + np.linspace(-0.09, 0.09, len(values_for_metric)), values_for_metric, s=9, alpha=0.4)
        summary_row = contrast_summary.loc[contrast_summary["contrast"] == names[index - 1]].iloc[0]
        axes[2].errorbar(
            index,
            summary_row["mean"],
            yerr=[[summary_row["mean"] - summary_row["ci_low"]], [summary_row["ci_high"] - summary_row["mean"]]],
            color="black",
            marker="D",
            capsize=3,
        )
    axes[2].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[2].set(title="Per-pair factorial decomposition", ylabel="Logit-difference contrast")
    panel_label(axes[2], "C")
    preference = float((controls["L"] > 0).mean())
    fig.suptitle(f"Control diagnostics: A preferred in {preference:.1%} of clean prompts", fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    save_figure(fig, output_dir, "fig01_control_bias")


def plot_accuracy_landscape(stats: pd.DataFrame, layers: list[int], alphas: list[float], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    accuracy_norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    for index, (metric, title) in enumerate(
        (("accuracy_raw", "Raw accuracy"), ("accuracy_adjusted", "Matched-baseline adjusted accuracy"))
    ):
        image = heatmap(
            axes[index], stats, "alpha", "layer", metric, alphas, layers, title,
            cmap="RdBu", norm=accuracy_norm,
        )
        fig.colorbar(image, ax=axes[index], fraction=0.046, label="Accuracy")
        panel_label(axes[index], chr(ord("A") + index))
    image = heatmap(
        axes[2], stats, "alpha", "layer", "tie_rate_adjusted", alphas, layers,
        "Adjusted exact-tie rate", cmap="viridis", vmin=0, vmax=1,
    )
    fig.colorbar(image, ax=axes[2], fraction=0.046, label="Tie rate")
    panel_label(axes[2], "C")
    fig.suptitle("Position-detection performance landscape", fontsize=13)
    save_figure(fig, output_dir, "fig02_accuracy_landscape")


def _plot_lines_with_ci(ax, frame, metric, alphas, layers, ylabel, chance=None, zero=None):
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, len(alphas)))
    for color, alpha in zip(colors, alphas):
        group = frame.loc[frame["alpha"] == alpha].set_index("layer").reindex(layers)
        values = group[metric].to_numpy(float)
        low = group[f"{metric}_ci_low"].to_numpy(float)
        high = group[f"{metric}_ci_high"].to_numpy(float)
        ax.plot(layers, values, marker="o", color=color, label=rf"$\alpha={alpha:g}$")
        ax.fill_between(layers, low, high, color=color, alpha=0.16)
    if chance is not None:
        ax.axhline(chance, color="black", linestyle="--", linewidth=1, label="Chance")
    if zero is not None:
        ax.axhline(zero, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Injection layer", ylabel=ylabel)
    ax.set_xticks(layers)
    ax.grid(alpha=0.2)


def plot_layer_profiles(stats: pd.DataFrame, layers: list[int], alphas: list[float], output_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    _plot_lines_with_ci(axes[0], stats, "accuracy_adjusted", alphas, layers, "Adjusted accuracy", chance=0.5)
    axes[0].set_ylim(0.3, 1.0)
    axes[0].set_title("Detection accuracy")
    axes[0].legend(
        ncol=min(5, len(alphas) + 1),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.3 if len(alphas) > 8 else 1.18),
    )
    panel_label(axes[0], "A")

    _plot_lines_with_ci(axes[1], stats, "tie_rate_adjusted", alphas, layers, "Exact-tie rate")
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, len(alphas)))
    for color, alpha in zip(colors, alphas):
        group = stats.loc[stats["alpha"] == alpha].set_index("layer").reindex(layers)
        axes[1].plot(layers, group["non_tie_accuracy"], color=color, linestyle=":", marker="s", alpha=0.9)
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Solid: tie rate; dotted: accuracy conditional on a non-tie")
    panel_label(axes[1], "B")

    _plot_lines_with_ci(axes[2], stats, "target_aligned_delta", alphas, layers, r"Mean target-aligned $\Delta L$", zero=0)
    axes[2].set_title("Signed intervention effect")
    panel_label(axes[2], "C")
    fig.suptitle("Layer profiles with pair-bootstrap 95% intervals", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, output_dir, "fig03_layer_profiles")


def plot_concept_heterogeneity(
    stats: pd.DataFrame, concepts: list[str], layers: list[int], alphas: list[float], output_dir: Path
) -> None:
    column_count = min(4, len(alphas))
    row_count = math.ceil(len(alphas) / column_count)
    fig, axes = plt.subplots(
        row_count, column_count, figsize=(4.3 * column_count, 4.5 * row_count), constrained_layout=True
    )
    axes = np.asarray(axes, dtype=object).reshape(row_count, column_count)
    norm = TwoSlopeNorm(vmin=0.3, vcenter=0.5, vmax=1.0)
    for index, (ax, alpha) in enumerate(zip(axes.flat, alphas)):
        subset = stats.loc[stats["alpha"] == alpha]
        image = heatmap(
            ax, subset, "concept", "layer", "accuracy_adjusted", concepts, layers,
            rf"$\alpha={alpha:g}$", cmap="RdBu", norm=norm, annotate=True,
        )
        fig.colorbar(image, ax=ax, fraction=0.025, label="Adjusted accuracy")
        panel_label(ax, chr(ord("A") + index))
    for ax in list(axes.flat)[len(alphas):]:
        ax.set_visible(False)
    fig.suptitle("Concept heterogeneity in adjusted position-detection accuracy", fontsize=13)
    save_figure(fig, output_dir, "fig04_concept_heterogeneity")


def _propagation_matrix(frame, alpha, metric, layers, restoration_layers):
    subset = frame.loc[frame["alpha"] == alpha]
    return subset.pivot(index="injection_layer", columns="restoration_layer", values=metric).reindex(
        index=layers, columns=restoration_layers
    )


def _draw_propagation(ax, matrix, title, cmap, diverging=False):
    values = matrix.to_numpy(float)
    if diverging:
        bound = np.nanmax(np.abs(values))
        bound = bound if bound else 1e-12
        image = ax.imshow(values, aspect="auto", cmap=cmap, norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound))
    else:
        image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=np.nanmax(values) or 1e-12)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set(xlabel="Downstream/restoration layer", ylabel="Injection layer", title=title)
    return image


def plot_causal_propagation(
    stats: pd.DataFrame,
    layers: list[int],
    alphas: list[float],
    main_alpha: float,
    output_dir: Path,
) -> None:
    restoration_layers = sorted(stats["restoration_layer"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    p_matrix = _propagation_matrix(stats, main_alpha, "P", layers, restoration_layers)
    e_matrix = _propagation_matrix(stats, main_alpha, "E_position", layers, restoration_layers)
    for index, (ax, matrix, title, cmap, diverging) in enumerate(
        (
            (axes[0], p_matrix, "Representational contamination P", "viridis", False),
            (axes[1], e_matrix, "Position-aligned restoration effect E", "RdBu", True),
        )
    ):
        image = _draw_propagation(ax, matrix, title, cmap, diverging)
        fig.colorbar(image, ax=ax, fraction=0.025)
        panel_label(ax, chr(ord("A") + index))
    fig.suptitle(rf"Causal propagation at representative $\alpha={main_alpha:g}$", fontsize=13)
    save_figure(fig, output_dir, "fig05_causal_propagation")

    page_size = 4
    pages = [alphas[start:start + page_size] for start in range(0, len(alphas), page_size)]
    for page_index, page_alphas in enumerate(pages, start=1):
        appendix, axes = plt.subplots(
            len(page_alphas), 2, figsize=(15, 4 * len(page_alphas)), constrained_layout=True
        )
        axes = np.atleast_2d(axes)
        for row_index, alpha in enumerate(page_alphas):
            p_matrix = _propagation_matrix(stats, alpha, "P", layers, restoration_layers)
            e_matrix = _propagation_matrix(stats, alpha, "E_position", layers, restoration_layers)
            p_image = _draw_propagation(axes[row_index, 0], p_matrix, rf"P, $\alpha={alpha:g}$", "viridis")
            e_image = _draw_propagation(axes[row_index, 1], e_matrix, rf"E, $\alpha={alpha:g}$", "RdBu", True)
            appendix.colorbar(p_image, ax=axes[row_index, 0], fraction=0.025)
            appendix.colorbar(e_image, ax=axes[row_index, 1], fraction=0.025)
        appendix.suptitle("Causal propagation across intervention strengths", fontsize=13)
        stem = "appendix01_propagation_all_alphas"
        if len(pages) > 1:
            stem += f"_page{page_index:02d}"
        save_figure(appendix, output_dir, stem)


def plot_order_mapping(stats: pd.DataFrame, layers: list[int], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    colors = ["#4477AA", "#EE6677", "#228833", "#CCBB44"]
    for color, ((order, mapping), group) in zip(colors, stats.groupby(["content_order", "label_mapping"], sort=True)):
        group = group.set_index("layer").reindex(layers)
        label = f"{order}/{mapping}"
        for ax, metric in zip(axes, ("accuracy_adjusted", "target_aligned_delta")):
            ax.plot(layers, group[metric], marker="o", color=color, label=label)
            ax.fill_between(layers, group[f"{metric}_ci_low"], group[f"{metric}_ci_high"], color=color, alpha=0.14)
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set(title="Adjusted accuracy", ylabel="Accuracy")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set(title="Target-aligned intervention effect", ylabel=r"Mean $\Delta L$")
    for index, ax in enumerate(axes):
        ax.set(xlabel="Injection layer")
        ax.set_xticks(layers)
        ax.grid(alpha=0.2)
        panel_label(ax, chr(ord("A") + index))
    axes[0].legend(ncol=2)
    save_figure(fig, output_dir, "appendix02_order_mapping_robustness")


def plot_concept_profiles(
    stats: pd.DataFrame, concepts: list[str], layers: list[int], alphas: list[float], output_dir: Path
) -> None:
    column_count = 2
    row_count = math.ceil(len(concepts) / column_count)
    fig, axes = plt.subplots(
        row_count, column_count, figsize=(14, 3.6 * row_count), sharex=True, sharey=True, constrained_layout=True
    )
    axes = np.asarray(axes, dtype=object).reshape(row_count, column_count)
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, len(alphas)))
    for ax, concept in zip(axes.flat, concepts):
        subset = stats.loc[stats["concept"] == concept]
        for color, alpha in zip(colors, alphas):
            group = subset.loc[subset["alpha"] == alpha].set_index("layer").reindex(layers)
            ax.plot(layers, group["accuracy_adjusted"], marker="o", markersize=3, color=color, label=rf"$\alpha={alpha:g}$")
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(concept)
        ax.set_xticks(layers)
        ax.set_ylim(0.25, 1.02)
        ax.grid(alpha=0.15)
    for ax in list(axes.flat)[len(concepts):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Injection layer")
    for ax in axes[:, 0]:
        ax.set_ylabel("Adjusted accuracy")
    if len(alphas) <= 8:
        axes[0, 0].legend(ncol=len(alphas), loc="upper center", bbox_to_anchor=(1.05, 1.35))
    else:
        scalar = plt.cm.ScalarMappable(
            norm=plt.Normalize(vmin=min(alphas), vmax=max(alphas)), cmap="viridis"
        )
        fig.colorbar(scalar, ax=axes, location="right", label=r"$\alpha$", fraction=0.02)
    fig.suptitle("Exploratory per-concept layer and strength profiles", fontsize=13)
    save_figure(fig, output_dir, "appendix03_concept_profiles")


def plot_position_effects(stats: pd.DataFrame, layers: list[int], alphas: list[float], output_dir: Path) -> None:
    column_count = min(4, len(alphas))
    row_count = math.ceil(len(alphas) / column_count)
    fig, axes = plt.subplots(
        row_count, column_count, figsize=(4.3 * column_count, 4.2 * row_count), constrained_layout=True
    )
    axes = np.asarray(axes, dtype=object).reshape(row_count, column_count)
    bound = max(abs(stats["S_position"].min()), abs(stats["S_position"].max()), 1e-12)
    for index, (ax, alpha) in enumerate(zip(axes.flat, alphas)):
        subset = stats.loc[stats["alpha"] == alpha]
        concepts = sorted(subset["concept"].unique(), key=str.casefold)
        image = heatmap(
            ax, subset, "concept", "layer", "S_position", concepts, layers,
            rf"$\alpha={alpha:g}$", cmap="RdBu", norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
            annotate=False,
        )
        fig.colorbar(image, ax=ax, fraction=0.025, label=r"Mean $S_{position}$")
        panel_label(ax, chr(ord("A") + index))
    for ax in list(axes.flat)[len(alphas):]:
        ax.set_visible(False)
    fig.suptitle("Physical-position sensitivity", fontsize=13)
    save_figure(fig, output_dir, "appendix04_position_effects")


def plot_norm_validation(injections: pd.DataFrame, summary: pd.DataFrame, alphas: list[float], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    per_token = [injections.loc[injections["alpha"] == alpha, "per_token_norm_ratio"].dropna() for alpha in alphas]
    frobenius = [injections.loc[injections["alpha"] == alpha, "frobenius_norm_ratio"].dropna() for alpha in alphas]
    axes[0].boxplot(per_token, tick_labels=[f"{alpha:g}" for alpha in alphas], showfliers=False)
    axes[1].boxplot(frobenius, tick_labels=[f"{alpha:g}" for alpha in alphas], showfliers=False)
    for index, ax in enumerate(axes):
        ax.axhline(1, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel(r"Intended $\alpha$")
        ax.grid(alpha=0.2)
        panel_label(ax, chr(ord("A") + index))
    axes[0].set(title="Realized per-token norm / intended norm", ylabel="Ratio")
    axes[1].set(title=r"Realized Frobenius norm / $(\alpha\sqrt{n_{tokens}})$", ylabel="Ratio")
    fig.suptitle("Intervention-norm validation", fontsize=13)
    save_figure(fig, output_dir, "appendix05_norm_validation")


def plot_endpoint_checks(endpoint: dict, output_dir: Path) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "check": list(endpoint),
            "max_absolute_value": list(endpoint.values()),
            "ci_low": list(endpoint.values()),
            "ci_high": list(endpoint.values()),
        }
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    values = frame["max_absolute_value"].to_numpy(float)
    ax.bar(range(len(frame)), values, color=["#4477AA", "#228833", "#CC6677"])
    ax.set_xticks(range(len(frame)), ["P(k,k)", "E(k,k)", "E(k,31)"])
    ax.set_ylabel("Maximum absolute value")
    upper = max(values.max() * 1.2, 1e-12)
    ax.set_ylim(0, upper)
    for index, value in enumerate(values):
        ax.text(index, upper * 0.05, f"{value:.3g}", ha="center", va="bottom")
    ax.set_title("Structural endpoint validation across all saved conditions")
    fig.tight_layout()
    save_figure(fig, output_dir, "appendix06_endpoint_validation")
    return frame


def plot_effect_distributions(injections: pd.DataFrame, layers: list[int], alphas: list[float], output_dir: Path) -> pd.DataFrame:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    all_layers = [injections.loc[injections["layer"] == layer, "target_aligned_delta"].to_numpy() for layer in layers]
    axes[0].violinplot(all_layers, positions=layers, widths=1.8, showmedians=True, showextrema=False)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set(title="All strengths", xlabel="Injection layer", ylabel=r"Target-aligned $\Delta L$")
    layer15 = injections.loc[injections["layer"] == 15]
    if layer15.empty:
        layer15 = injections.loc[injections["layer"] == min(layers, key=lambda value: abs(value - 15))]
    distributions = [layer15.loc[layer15["alpha"] == alpha, "target_aligned_delta"].to_numpy() for alpha in alphas]
    axes[1].violinplot(distributions, positions=range(len(alphas)), widths=0.8, showmedians=True, showextrema=False)
    axes[1].set_xticks(range(len(alphas)), [f"{alpha:g}" for alpha in alphas])
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set(title=f"Transition layer {int(layer15['layer'].iloc[0])}", xlabel=r"$\alpha$", ylabel=r"Target-aligned $\Delta L$")
    panel_label(axes[0], "A")
    panel_label(axes[1], "B")
    fig.suptitle("Distribution of signed intervention effects", fontsize=13)
    save_figure(fig, output_dir, "appendix07_effect_distributions")

    rows = []
    for (layer, alpha), group in injections.groupby(["layer", "alpha"], observed=True):
        values = group["target_aligned_delta"].to_numpy(float)
        rows.append(
            {
                "layer": layer,
                "alpha": alpha,
                "n_injections": len(values),
                "mean": float(np.mean(values)),
                "q025": float(np.quantile(values, 0.025)),
                "median": float(np.median(values)),
                "q975": float(np.quantile(values, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def run_analysis(args) -> dict:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    source_dir = output_dir / "source_data"
    manifest, completed, _summary, runtime, control_paths, condition_paths = validate_run(input_dir, args.main_alpha)
    controls, conditions, injections, diagnostics, endpoint = load_or_build_tables(
        input_dir, source_dir, manifest, control_paths, condition_paths, args.rebuild_cache
    )
    config = manifest["config"]
    layers = sorted(int(value) for value in config["layers"])
    alphas = sorted(float(value) for value in config["alphas"])
    concepts = sorted((str(value) for value in config["concepts"]), key=str.casefold)
    configure_style()

    overall = bootstrap_injection_metrics(injections, [], args.bootstrap_replicates, args.bootstrap_seed)
    layer_alpha = bootstrap_injection_metrics(
        injections, ["layer", "alpha"], args.bootstrap_replicates, args.bootstrap_seed
    ).sort_values(["layer", "alpha"])
    concept_stats = bootstrap_injection_metrics(
        injections, ["concept", "layer", "alpha"], args.bootstrap_replicates, args.bootstrap_seed
    ).sort_values(["concept", "layer", "alpha"])
    order_mapping = bootstrap_injection_metrics(
        injections, ["content_order", "label_mapping", "layer"], args.bootstrap_replicates, args.bootstrap_seed
    ).sort_values(["content_order", "label_mapping", "layer"])
    position_stats = bootstrap_means(
        conditions, ["concept", "layer", "alpha"], ["S_position"], args.bootstrap_replicates, args.bootstrap_seed
    ).sort_values(["concept", "layer", "alpha"])
    propagation = bootstrap_means(
        diagnostics,
        ["injection_layer", "restoration_layer", "alpha"],
        ["P", "E_position"],
        args.bootstrap_replicates,
        args.bootstrap_seed,
        weight_column="n_observations",
    ).sort_values(["alpha", "injection_layer", "restoration_layer"])
    control_groups = bootstrap_means(
        controls, ["content_order", "label_mapping"], ["L"], args.bootstrap_replicates, args.bootstrap_seed
    ).sort_values(["content_order", "label_mapping"])
    contrasts = control_contrasts(controls)
    contrast_summary = bootstrap_contrast_summary(contrasts, args.bootstrap_replicates, args.bootstrap_seed)
    norm_summary = bootstrap_means(
        injections, ["alpha"], ["per_token_norm_ratio", "frobenius_norm_ratio"],
        args.bootstrap_replicates, args.bootstrap_seed,
    ).sort_values("alpha")

    source_tables = {
        "fig01_control_values.csv": controls,
        "fig01_control_groups.csv": control_groups,
        "fig01_control_contrasts.csv": contrasts,
        "fig01_control_contrast_summary.csv": contrast_summary,
        "fig02_accuracy_layer_alpha.csv": layer_alpha,
        "fig03_layer_profiles.csv": layer_alpha,
        "fig04_concept_layer_alpha.csv": concept_stats,
        "fig05_causal_propagation.csv": propagation,
        "appendix02_order_mapping.csv": order_mapping,
        "appendix03_concept_profiles.csv": concept_stats,
        "appendix04_position_effects.csv": position_stats,
        "appendix05_norm_summary.csv": norm_summary,
        "appendix05_norm_values.csv": injections[
            [
                "pair_id", "prompt_id", "concept", "layer", "alpha", "target_position",
                "target_token_count", "intended_per_token_l2", "realized_per_token_l2_mean",
                "realized_frobenius_norm", "per_token_norm_ratio", "frobenius_norm_ratio",
            ]
        ],
        "appendix07_effect_distribution_values.csv": injections[
            [
                "pair_id", "prompt_id", "concept", "layer", "alpha", "content_order",
                "label_mapping", "target_position", "target_label", "target_aligned_delta",
            ]
        ],
    }
    for name, frame in source_tables.items():
        frame.to_csv(source_dir / name, index=False)

    plot_control_bias(controls, control_groups, contrasts, contrast_summary, output_dir)
    plot_accuracy_landscape(layer_alpha, layers, alphas, output_dir)
    plot_layer_profiles(layer_alpha, layers, alphas, output_dir)
    plot_concept_heterogeneity(concept_stats, concepts, layers, alphas, output_dir)
    plot_causal_propagation(propagation, layers, alphas, args.main_alpha, output_dir)
    plot_order_mapping(order_mapping, layers, output_dir)
    plot_concept_profiles(concept_stats, concepts, layers, alphas, output_dir)
    plot_position_effects(position_stats, layers, alphas, output_dir)
    plot_norm_validation(injections, norm_summary, alphas, output_dir)
    endpoint_frame = plot_endpoint_checks(endpoint, output_dir)
    endpoint_frame.to_csv(source_dir / "appendix06_endpoint_validation.csv", index=False)
    distribution_summary = plot_effect_distributions(injections, layers, alphas, output_dir)
    distribution_summary.to_csv(source_dir / "appendix07_effect_distributions.csv", index=False)
    atomic_json(output_dir / "figure_captions.json", FIGURE_CAPTIONS)

    best = layer_alpha.loc[layer_alpha["accuracy_adjusted"].idxmax()]
    best_concepts = concept_stats.loc[concept_stats.groupby("concept")["accuracy_adjusted"].idxmax()].copy()
    best_concepts["exploratory"] = True
    best_concepts.to_csv(source_dir / "appendix03_exploratory_best_settings.csv", index=False)
    provenance = cache_signature(input_dir, len(condition_paths), len(control_paths))
    overall_row = overall.iloc[0]
    result = {
        "schema_version": SCHEMA_VERSION,
        "input_dir": str(input_dir),
        "input_hashes": provenance,
        "model_commit": runtime.get("model_commit"),
        "model": config.get("model"),
        "conditions": int(completed["conditions"]),
        "controls": len(controls),
        "bootstrap": {
            "unit": "sentence pair",
            "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
            "confidence_interval": "95% percentile",
            "concepts": "fixed studied set",
        },
        "representative_alpha": args.main_alpha,
        "figure_captions": "figure_captions.json",
        "headline": {
            "control_A_preference_rate": float((controls["L"] > 0).mean()),
            "overall_accuracy_raw": float(overall_row["accuracy_raw"]),
            "overall_accuracy_adjusted": float(overall_row["accuracy_adjusted"]),
            "overall_tie_rate_adjusted": float(overall_row["tie_rate_adjusted"]),
            "overall_non_tie_accuracy": float(overall_row["non_tie_accuracy"]),
            "best_layer": int(best["layer"]),
            "best_alpha": float(best["alpha"]),
            "best_accuracy_adjusted": float(best["accuracy_adjusted"]),
        },
        "endpoint_checks": {key: float(value) for key, value in endpoint.items()},
        "notes": [
            "Exact ties receive half credit and are reported separately.",
            "Best-setting summaries are exploratory because settings were selected after inspection.",
        ],
    }
    atomic_json(output_dir / "analysis_summary.json", result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Completed position-detection result directory")
    parser.add_argument("--output-dir", type=Path, default=Path("plots/position_detection"))
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--main-alpha", type=float, default=5.0)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = run_analysis(args)
    headline = result["headline"]
    print(
        f"Created position-detection figures in {args.output_dir}: "
        f"adjusted accuracy {headline['overall_accuracy_adjusted']:.1%}, "
        f"best layer {headline['best_layer']} / alpha {headline['best_alpha']:g} "
        f"({headline['best_accuracy_adjusted']:.1%})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
