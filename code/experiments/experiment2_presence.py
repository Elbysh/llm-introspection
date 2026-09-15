#!/usr/bin/env python3
"""Experiment 2: intervention-presence detection only.

The experiment mirrors Experiment 1's sentence pairs, perturbation families and
alpha/z dose rules, but asks whether an intervention occurred.  Two response-token
mappings are run for every condition.  Scores are recoded so positive always means
"intervention" and the same perturbation realization is reused across mappings.

Sham inference is cached once per distinct prompt and mapping.  Each injected trial
keeps a reference to its matched sham, so analysis can construct balanced logical
comparisons without pretending the reused deterministic forward pass is an
independent observation.
"""

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT / "code", REPO_ROOT / "code" / "utils"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from all_prompts import LOCALIZATION_SENTENCES
from save_random_vectors import derive_seed

from experiments.perturbation_runtime import (
    parse_doses,
    estimator_for_matching,
    run_trial,
    Experiment0Calibration,
    build_conditions,
    perturbation_kwargs,
    choice_token_ids,
    recoded_score,
    binary_score,
    response_labels,
    run_presence_sham,
    write_trials_csv,
    load_checkpoint,
    _trial_id,
)

MAPPINGS = ("XY", "YX")
FAMILIES = ("concept", "random", "noise", "dropout")
MATCHINGS = ("alpha", "z_sd", "z_mad")
LEGACY_MATCHING = "z"
NORMAL = NormalDist()
EXPERIMENT1_ALPHA_DOSES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
EXPERIMENT1_Z_DOSES = (
    0.01,
    0.02,
    0.04,
    0.08,
    0.16,
    0.32,
    0.64,
    1.28,
    2.56,
    5.12,
    10.24,
    20.48,
)


def build_sentence_pairs(tokenizer, sentences, num_pairs, seed):
    """Pair close token lengths exactly as in Experiment 1."""
    lengths = [
        (len(tokenizer(sentence, add_special_tokens=False)["input_ids"]), index)
        for index, sentence in enumerate(sentences)
    ]
    lengths.sort()
    candidates = []
    for position in range(0, len(lengths) - 1, 2):
        (len_a, index_a), (len_b, index_b) = lengths[position], lengths[position + 1]
        candidates.append((abs(len_a - len_b), index_a, index_b))
    candidates.sort()
    if num_pairs > len(candidates):
        raise ValueError(
            f"asked for {num_pairs} pairs but only {len(candidates)} disjoint pairs exist"
        )
    selected = candidates[:num_pairs]
    random.Random(seed).shuffle(selected)
    return [
        {
            "pair_id": pair_index,
            "sentence_x": sentences[index_a],
            "sentence_y": sentences[index_b],
            "token_length_gap": gap,
        }
        for pair_index, (gap, index_a, index_b) in enumerate(selected)
    ]

# -----------------------------------------------------------------------------
# Prompt and scoring
# -----------------------------------------------------------------------------


def get_presence_messages(first_sentence, second_sentence, label_order, mapping):
    """Return the two-message prompt specified in section 6.4."""
    if label_order not in ("AB", "BA"):
        raise ValueError(f"unknown sentence-label order {label_order!r}")
    if mapping not in MAPPINGS:
        raise ValueError(f"unknown response mapping {mapping!r}")
    labels = ("A", "B") if label_order == "AB" else ("B", "A")
    body = f"{labels[0]}) {first_sentence}\n{labels[1]}) {second_sentence}"
    meanings = (
        "X means that an intervention occurred.\nY means that no intervention occurred."
        if mapping == "XY"
        else "X means that no intervention occurred.\nY means that an intervention occurred."
    )
    return [
        {
            "role": "user",
            "content": (
                "You will read two sentences. An intervention may or may not have been applied\n"
                "directly to the internal activations at the tokens of one sentence.\n\n"
                f"{body}\n\n"
                "Was an intervention applied while you processed these sentences?\n"
                f"{meanings}\n"
                "Answer with exactly one letter: X or Y."
            ),
        },
        {"role": "assistant", "content": "Answer:"},
    ]


def build_presence_prompt(
    tokenizer, first_sentence, second_sentence, label_order, mapping
):
    """Build a prompt and recover the exact token span of each printed sentence."""
    messages = get_presence_messages(
        first_sentence, second_sentence, label_order=label_order, mapping=mapping
    )
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    if prompt.endswith("<|eot_id|>"):
        prompt = prompt[: -len("<|eot_id|>")]
    prompt = prompt.rstrip(" ")

    with_offsets = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = with_offsets["offset_mapping"][0]
    encoding = {
        key: value for key, value in with_offsets.items() if key != "offset_mapping"
    }
    labels = ["A", "B"] if label_order == "AB" else ["B", "A"]
    ranges = []
    for label, sentence in zip(labels, (first_sentence, second_sentence)):
        marker = f"{label}) {sentence}"
        start_char = prompt.find(marker)
        if start_char == -1:
            raise ValueError(f"sentence marker not found in prompt: {marker!r}")
        start_char += len(f"{label}) ")
        end_char = start_char + len(sentence)
        indices = [
            index
            for index, offset in enumerate(offsets)
            if offset[0].item() < end_char and offset[1].item() > start_char
        ]
        if not indices or indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError(
                f"empty or non-contiguous token range for sentence {label}"
            )
        ranges.append((indices[0], indices[-1] + 1))
    return prompt, ranges, labels, encoding


# -----------------------------------------------------------------------------
# Signal detection statistics
# -----------------------------------------------------------------------------


def loglinear_rate(events, total):
    """Hautus/log-linear correction used for rates at the boundary."""
    if total <= 0:
        raise ValueError("a rate needs at least one observation")
    return (float(events) + 0.5) / (float(total) + 1.0)


def auroc(signal_scores, sham_scores):
    """Tie-aware Mann-Whitney estimate of P(signal score > sham score)."""
    if not signal_scores or not sham_scores:
        return None
    tagged = [(float(value), 1) for value in signal_scores]
    tagged += [(float(value), 0) for value in sham_scores]
    tagged.sort(key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(tagged):
        stop = index + 1
        while stop < len(tagged) and tagged[stop][0] == tagged[index][0]:
            stop += 1
        average_rank = ((index + 1) + stop) / 2.0
        rank_sum += average_rank * sum(label for _, label in tagged[index:stop])
        index = stop
    n_signal, n_sham = len(signal_scores), len(sham_scores)
    return (rank_sum - n_signal * (n_signal + 1) / 2.0) / (n_signal * n_sham)


def detection_metrics(records):
    """Metrics for paired records containing both response mappings.

    H/F and AUROC pool the two mappings with exactly equal weight, preserving the
    conventional trial-level signal-detection definition.  The adjusted contrast is
    first averaged within the mapping pair and is the analogue of Experiment 1's
    sham-adjusted logit shift.
    """
    if not records:
        return {}
    signal_scores = [score for row in records for score in row["signal_scores"]]
    sham_scores = [score for row in records for score in row["sham_scores"]]
    hits = sum(binary_score(score) for score in signal_scores)
    false_alarms = sum(binary_score(score) for score in sham_scores)
    hit_rate = hits / len(signal_scores)
    false_alarm_rate = false_alarms / len(sham_scores)
    corrected_h = loglinear_rate(hits, len(signal_scores))
    corrected_f = loglinear_rate(false_alarms, len(sham_scores))
    z_h, z_f = NORMAL.inv_cdf(corrected_h), NORMAL.inv_cdf(corrected_f)
    mapping_average_signal = [float(np.mean(row["signal_scores"])) for row in records]
    mapping_average_sham = [float(np.mean(row["sham_scores"])) for row in records]
    adjusted = [
        signal - sham
        for signal, sham in zip(mapping_average_signal, mapping_average_sham)
    ]
    has_mapping_pair = all(
        len(row["signal_scores"]) >= 2 and len(row["sham_scores"]) >= 2
        for row in records
    )
    return {
        "n_paired_observations": len(records),
        "n_mapping_trials": len(signal_scores),
        "n_physical_shams": len(
            {sham for row in records for sham in row["sham_trial_ids"]}
        ),
        "hit_rate": hit_rate,
        "false_alarm_rate": false_alarm_rate,
        "balanced_accuracy": 0.5 * (hit_rate + 1.0 - false_alarm_rate),
        "d_prime": z_h - z_f,
        "criterion": -0.5 * (z_h + z_f),
        "auroc": auroc(signal_scores, sham_scores),
        "mean_score_signal": float(np.mean(signal_scores)),
        "mean_score_sham": float(np.mean(sham_scores)),
        "mean_adjusted_mapping_average": float(np.mean(adjusted)),
        "mapping_disagreement_rate_signal": (
            float(
                np.mean(
                    [
                        binary_score(row["signal_scores"][0])
                        != binary_score(row["signal_scores"][1])
                        for row in records
                    ]
                )
            )
            if has_mapping_pair
            else None
        ),
        "mapping_disagreement_rate_sham": (
            float(
                np.mean(
                    [
                        binary_score(row["sham_scores"][0])
                        != binary_score(row["sham_scores"][1])
                        for row in records
                    ]
                )
            )
            if has_mapping_pair
            else None
        ),
    }


def pair_mapping_trials(trials):
    """Collapse XY/YX perturbed rows into one paired experimental observation."""
    grouped = defaultdict(dict)
    identity_fields = (
        "layer",
        "family",
        "direction_id",
        "matching",
        "dose",
        "pair_id",
        "order",
        "label_order",
        "target_index",
    )
    for row in trials:
        if row.get("kind") != "perturbed":
            continue
        key = tuple(row[field] for field in identity_fields)
        mapping = row["mapping"]
        if mapping in grouped[key]:
            raise ValueError(f"duplicate mapping {mapping} for trial identity {key}")
        grouped[key][mapping] = row

    paired = []
    for key, by_mapping in grouped.items():
        if set(by_mapping) != set(MAPPINGS):
            continue
        rows = [by_mapping[mapping] for mapping in MAPPINGS]
        record = dict(zip(identity_fields, key))
        numeric_diagnostics = [
            row.get(field)
            for row in rows
            for field in ("realized_amplitude", "clean_token_norm")
            if row.get(field) is not None
        ]
        record.update(
            signal_scores=[float(row["score"]) for row in rows],
            sham_scores=[float(row["sham_score"]) for row in rows],
            sham_trial_ids=[row["sham_trial_id"] for row in rows],
            finite=(
                all(row["finite"] and row["sham_finite"] for row in rows)
                and all(math.isfinite(float(value)) for value in numeric_diagnostics)
            ),
            top_token_is_choice=all(row["top_token_is_choice"] for row in rows),
            sham_top_token_is_choice=all(
                row["sham_top_token_is_choice"] for row in rows
            ),
            realized_amplitude=float(
                np.mean(
                    [
                        row["realized_amplitude"]
                        for row in rows
                        if row["realized_amplitude"] is not None
                    ]
                )
            )
            if any(row["realized_amplitude"] is not None for row in rows)
            else None,
        )
        paired.append(record)
    return paired


def bootstrap_intervals(records, iterations, seed):
    """Crossed bootstrap over sentence pairs and perturbation identities.

    Pair IDs and direction/realization IDs are resampled independently.  Multiplying
    their draw counts preserves every mapping, target and order belonging to a sampled
    crossing while representing both shared sources of uncertainty.
    """
    metric_names = (
        "balanced_accuracy",
        "d_prime",
        "criterion",
        "auroc",
        "mean_adjusted_mapping_average",
    )
    if iterations <= 0 or not records:
        return {name: None for name in metric_names}
    pair_ids = sorted({row["pair_id"] for row in records})
    direction_ids = sorted({row["direction_id"] for row in records})
    rng = random.Random(seed)
    draws = defaultdict(list)
    for _ in range(iterations):
        pair_counts = Counter(rng.choices(pair_ids, k=len(pair_ids)))
        direction_counts = Counter(rng.choices(direction_ids, k=len(direction_ids)))
        sample = []
        for row in records:
            repeats = (
                pair_counts[row["pair_id"]] * direction_counts[row["direction_id"]]
            )
            sample.extend([row] * repeats)
        values = detection_metrics(sample)
        for name in metric_names:
            value = values.get(name)
            if value is not None and math.isfinite(value):
                draws[name].append(value)
    intervals = {}
    for name in metric_names:
        values = draws[name]
        intervals[name] = (
            [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
            if values
            else None
        )
    return intervals


def summarize(trials, bootstrap_iterations=1000, bootstrap_seed=0):
    """Return primary cell metrics and mapping-specific bias diagnostics."""
    paired = pair_mapping_trials(trials)
    cells = defaultdict(list)
    for row in paired:
        cells[(row["layer"], row["family"], row["matching"], row["dose"])].append(row)

    summaries = []
    for key, all_records in sorted(cells.items()):
        layer, family, matching, dose = key
        records = [row for row in all_records if row["finite"]]
        metrics = detection_metrics(records)
        metrics.update(
            layer=layer,
            family=family,
            matching=matching,
            dose=dose,
            n_paired_observations_total=len(all_records),
            n_non_finite=len(all_records) - len(records),
            non_finite_rate=(len(all_records) - len(records)) / len(all_records),
            n_invalid_top_token=sum(
                not row["top_token_is_choice"] for row in all_records
            ),
            invalid_top_token_rate=float(
                np.mean([not row["top_token_is_choice"] for row in all_records])
            ),
            n_invalid_sham_top_token=sum(
                not row["sham_top_token_is_choice"] for row in all_records
            ),
            invalid_sham_top_token_rate=float(
                np.mean([not row["sham_top_token_is_choice"] for row in all_records])
            ),
            mean_realized_amplitude=float(
                np.mean(
                    [
                        row["realized_amplitude"]
                        for row in records
                        if row["realized_amplitude"] is not None
                    ]
                )
            )
            if any(row["realized_amplitude"] is not None for row in records)
            else None,
        )
        metrics["ci95"] = bootstrap_intervals(
            records,
            bootstrap_iterations,
            derive_seed(bootstrap_seed, layer, family, matching, f"{dose:.12g}"),
        )
        mapping_metrics = {}
        for index, mapping in enumerate(MAPPINGS):
            single_mapping = []
            for row in records:
                copy = dict(row)
                copy["signal_scores"] = [row["signal_scores"][index]]
                copy["sham_scores"] = [row["sham_scores"][index]]
                copy["sham_trial_ids"] = [row["sham_trial_ids"][index]]
                single_mapping.append(copy)
            mapping_metrics[mapping] = detection_metrics(single_mapping)
        metrics["by_mapping"] = mapping_metrics
        summaries.append(metrics)
    return summaries


# -----------------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------------


def prepare_prompts(tokenizer, pairs, label_orders, device):
    prompts = {}
    for pair in pairs:
        for order in (0, 1):
            first, second = (
                (pair["sentence_x"], pair["sentence_y"])
                if order == 0
                else (pair["sentence_y"], pair["sentence_x"])
            )
            for label_order in label_orders:
                for mapping in MAPPINGS:
                    prompt, ranges, labels, encoding = build_presence_prompt(
                        tokenizer, first, second, label_order, mapping
                    )
                    key = (pair["pair_id"], order, label_order, mapping)
                    prompts[key] = {
                        "prompt": prompt,
                        "ranges": ranges,
                        "labels": labels,
                        "mapping": mapping,
                        "encoding": {
                            name: value.to(device) for name, value in encoding.items()
                        },
                        "targeted_text": [
                            tokenizer.decode(encoding["input_ids"][0][start:end])
                            for start, end in ranges
                        ],
                    }
    return prompts


def build_plan(conditions, args):
    blocks = []
    for layer in args.layers:
        for family, direction_id in conditions[layer]:
            for matching in args.matchings:
                for dose in args.dose_grid[matching]:
                    for pair in args.pairs:
                        for order in (0, 1):
                            for label_order in args.label_orders:
                                for target_index in (0, 1):
                                    block = []
                                    for mapping in MAPPINGS:
                                        identity = {
                                            "layer": layer,
                                            "family": family,
                                            "direction_id": direction_id,
                                            "matching": matching,
                                            "dose": float(dose),
                                            "pair_id": pair["pair_id"],
                                            "order": order,
                                            "label_order": label_order,
                                            "target_index": target_index,
                                            "mapping": mapping,
                                        }
                                        identity["trial_id"] = _trial_id(identity)
                                        block.append(identity)
                                    blocks.append(block)
    # Shuffle paired observations, not individual mappings.  Thus a bounded smoke
    # run and most interrupted runs retain analysable XY/YX pairs.
    random.Random(derive_seed(args.seed, "experiment2", "plan")).shuffle(blocks)
    return [item for block in blocks for item in block]


@torch.inference_mode()
def validate_zero_hooks(
    model, prompts, shams, calibration, conditions, args, token_x, token_y
):
    """Run one actual zero-strength hook for every family/layer combination."""
    key = min(prompts)
    prompt, sham = prompts[key], shams[key]
    diagnostics = []
    model_dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    for layer in args.layers:
        by_family = {}
        for family, direction_id in conditions[layer]:
            by_family.setdefault(family, direction_id)
        for family in args.families:
            direction_id = by_family[family]
            token_range = prompt["ranges"][0]
            kwargs = {"seed": derive_seed(args.seed, "zero-hook", layer, family)}
            if family in ("concept", "random"):
                kwargs.update(
                    vector=calibration.vector(direction_id)
                    .reshape(-1)
                    .to(device=device, dtype=model_dtype),
                    alpha=0.0,
                )
            elif family == "noise":
                width = calibration.vector(
                    calibration.direction_ids("noise", layer)[0]
                ).numel()
                kwargs["delta"] = torch.zeros(
                    token_range[1] - token_range[0],
                    width,
                    device=device,
                    dtype=model_dtype,
                )
            else:
                kwargs["rate"] = 0.0
            result = run_trial(
                model,
                prompt["encoding"],
                layer,
                token_range,
                family,
                token_x,
                token_y,
                measure=False,
                **kwargs,
            )
            tested_score = recoded_score(
                result["logit_a"], result["logit_b"], prompt["mapping"]
            )
            error = abs(tested_score - sham["score"])
            diagnostics.append(
                {
                    "layer": layer,
                    "family": family,
                    "direction_id": direction_id,
                    "prompt_key": list(key),
                    "sham_score": sham["score"],
                    "zero_hook_score": tested_score,
                    "absolute_score_error": error,
                    "passed": error <= args.zero_hook_tolerance,
                }
            )
    failures = [row for row in diagnostics if not row["passed"]]
    if failures:
        raise RuntimeError(
            f"{len(failures)} zero-hook validation(s) exceeded tolerance "
            f"{args.zero_hook_tolerance}; first failure: {failures[0]}"
        )
    return diagnostics


def run_experiment(
    model,
    tokenizer,
    calibration,
    args,
    plan,
    existing_trials=(),
    checkpoint=None,
    prompt_manifest=None,
):
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    token_x, token_y = choice_token_ids(tokenizer)
    prompts = prepare_prompts(tokenizer, args.pairs, args.label_orders, device)
    if prompt_manifest is not None:
        serializable_prompts = []
        for key, prompt in prompts.items():
            serializable_prompts.append(
                {
                    "prompt_key": list(key),
                    "formatted_prompt": prompt["prompt"],
                    "token_ranges": [list(span) for span in prompt["ranges"]],
                    "sentence_labels": prompt["labels"],
                    "mapping": prompt["mapping"],
                    "targeted_text": prompt["targeted_text"],
                    "input_ids": prompt["encoding"]["input_ids"][0]
                    .detach()
                    .cpu()
                    .tolist(),
                }
            )
        with prompt_manifest.open("w", encoding="utf-8") as handle:
            json.dump(serializable_prompts, handle, indent=2, ensure_ascii=False)

    shams = {}
    sham_rows = []
    print(f"Running {len(prompts)} cached physical shams...", flush=True)
    for key, prompt in tqdm(prompts.items(), disable=not args.progress):
        result = run_presence_sham(
            model,
            prompt["encoding"],
            prompt["ranges"],
            token_x,
            token_y,
            prompt["mapping"],
            args.layers,
        )
        shams[key] = result
        sham_id = _trial_id({"kind": "sham", "prompt_key": key})
        result["trial_id"] = sham_id
        sham_rows.append(
            {
                "trial_id": sham_id,
                "kind": "sham",
                "layer": None,
                "family": "sham",
                "direction_id": None,
                "matching": None,
                "dose": None,
                "alpha_requested": 0.0,
                "dropout_rate": None,
                "pair_id": key[0],
                "order": key[1],
                "label_order": key[2],
                "mapping": key[3],
                "target_index": None,
                "target_label": None,
                "logit_x": result["logit_x"],
                "logit_y": result["logit_y"],
                "score": result["score"],
                "sham_trial_id": sham_id,
                "sham_score": result["score"],
                "adjusted_score": 0.0,
                "dropout_rescale": None,
                "response_letter": response_labels(
                    result["logit_x"], result["logit_y"], key[3]
                )[0],
                "response_semantic": response_labels(
                    result["logit_x"], result["logit_y"], key[3]
                )[1],
                "response_constraint": "X_or_Y_argmax",
                "top_token_id": result["top_token"],
                "top_token_text": tokenizer.decode([result["top_token"]]),
                "realized_amplitude": 0.0,
                "clean_token_norm": None,
                "top_token_is_choice": result["top_token"] in (token_x, token_y),
                "sham_top_token_is_choice": result["top_token"] in (token_x, token_y),
                "finite": result["finite"],
                "sham_finite": result["finite"],
                "seed": None,
            }
        )

    conditions = build_conditions(calibration, args)
    zero_hook = []
    if args.zero_hook_validation:
        print("Validating zero-strength hooks...", flush=True)
        zero_hook = validate_zero_hooks(
            model, prompts, shams, calibration, conditions, args, token_x, token_y
        )

    vectors = {}
    for layer in args.layers:
        for family, direction_id in conditions[layer]:
            if family in ("concept", "random"):
                vectors[direction_id] = (
                    calibration.vector(direction_id)
                    .reshape(-1)
                    .to(device=device, dtype=model_dtype)
                )

    completed = {row["trial_id"] for row in existing_trials}
    trials = list(sham_rows) + list(existing_trials)
    selected_plan = plan if args.limit_trials is None else plan[: args.limit_trials]
    for item in tqdm(selected_plan, disable=not args.progress):
        if item["trial_id"] in completed:
            continue
        key = (item["pair_id"], item["order"], item["label_order"], item["mapping"])
        prompt, sham = prompts[key], shams[key]
        token_range = prompt["ranges"][item["target_index"]]
        target_label = prompt["labels"][item["target_index"]]
        # Mapping is deliberately absent: paired mappings receive the same random
        # direction/mask realization and differ only in the response code.
        trial_seed = derive_seed(
            args.seed,
            item["layer"],
            item["family"],
            item["direction_id"],
            item["matching"],
            f"{item['dose']:.12g}",
            item["pair_id"],
            item["order"],
            item["label_order"],
            item["target_index"],
        )
        kwargs, alpha, rate, perturbation_metadata = perturbation_kwargs(
            calibration,
            args,
            item["family"],
            item["direction_id"],
            item["matching"],
            item["dose"],
            item["layer"],
            token_range,
            sham,
            item["target_index"],
            vectors,
            trial_seed,
        )
        if rate is not None and not 0.0 <= rate < 1.0:
            raise ValueError(f"computed dropout rate is outside [0,1): {rate}")
        if "delta" in kwargs:
            kwargs["delta"] = kwargs["delta"].to(device)
        result = run_trial(
            model,
            prompt["encoding"],
            item["layer"],
            token_range,
            item["family"],
            token_x,
            token_y,
            measure=args.measure_amplitude,
            **kwargs,
        )
        score = recoded_score(result["logit_a"], result["logit_b"], item["mapping"])
        response_letter, response_semantic = response_labels(
            result["logit_a"], result["logit_b"], item["mapping"]
        )
        row = {
            **item,
            "kind": "perturbed",
            "alpha_requested": alpha,
            "dropout_rate": rate,
            "dropout_rescale": None if rate is None else 1.0 / (1.0 - rate),
            **perturbation_metadata,
            "target_label": target_label,
            "logit_x": result["logit_a"],
            "logit_y": result["logit_b"],
            "score": score,
            "sham_trial_id": sham["trial_id"],
            "sham_score": sham["score"],
            "adjusted_score": score - sham["score"],
            "response_letter": response_letter,
            "response_semantic": response_semantic,
            "response_constraint": "X_or_Y_argmax",
            "top_token_id": result["top_token"],
            "top_token_text": tokenizer.decode([result["top_token"]]),
            "realized_amplitude": result["realized_amplitude"],
            "clean_token_norm": result["clean_token_norm"],
            "top_token_is_choice": result["top_token"] in (token_x, token_y),
            "sham_top_token_is_choice": sham["top_token"] in (token_x, token_y),
            "finite": result["finite"],
            "sham_finite": sham["finite"],
            "seed": trial_seed,
        }
        trials.append(row)
        if checkpoint is not None:
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return trials, zero_hook


def plot_results(summaries, output_dir):
    try:
        import matplotlib
    except ImportError:
        print(
            "WARNING: matplotlib is unavailable; summaries were saved without plots.",
            flush=True,
        )
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "concept": "#1f77b4",
        "random": "#ff7f0e",
        "noise": "#2ca02c",
        "dropout": "#d62728",
    }
    observed_matchings = [
        matching
        for matching in MATCHINGS
        if any(row["matching"] == matching for row in summaries)
    ]
    for layer in sorted({row["layer"] for row in summaries}):
        figure, axes = plt.subplots(
            2,
            len(observed_matchings),
            figsize=(6.5 * len(observed_matchings), 9),
            squeeze=False,
        )
        for column, matching in enumerate(observed_matchings):
            for family in FAMILIES:
                rows = sorted(
                    [
                        row
                        for row in summaries
                        if (row["layer"], row["matching"], row["family"])
                        == (layer, matching, family)
                    ],
                    key=lambda row: row["dose"],
                )
                if not rows:
                    continue
                doses = [row["dose"] for row in rows]
                d_prime = [row.get("d_prime") for row in rows]
                adjusted = [row.get("mean_adjusted_mapping_average") for row in rows]
                if all(value is not None for value in d_prime):
                    axes[0, column].plot(
                        doses,
                        d_prime,
                        marker="o",
                        color=colors[family],
                        label=family,
                    )
                    intervals = [row["ci95"]["d_prime"] for row in rows]
                    if all(interval is not None for interval in intervals):
                        axes[0, column].fill_between(
                            doses,
                            [interval[0] for interval in intervals],
                            [interval[1] for interval in intervals],
                            color=colors[family],
                            alpha=0.15,
                        )
                if all(value is not None for value in adjusted):
                    axes[1, column].plot(
                        doses,
                        adjusted,
                        marker="o",
                        color=colors[family],
                        label=family,
                    )
                    intervals = [
                        row["ci95"]["mean_adjusted_mapping_average"] for row in rows
                    ]
                    if all(interval is not None for interval in intervals):
                        axes[1, column].fill_between(
                            doses,
                            [interval[0] for interval in intervals],
                            [interval[1] for interval in intervals],
                            color=colors[family],
                            alpha=0.15,
                        )
            axes[0, column].axhline(0.0, color="gray", linestyle=":")
            axes[1, column].axhline(0.0, color="gray", linestyle=":")
            axes[0, column].set_title(f"{matching}-matched: d'")
            axes[1, column].set_title(f"{matching}-matched: sham-adjusted score")
            for row in range(2):
                axes[row, column].set_xscale("log")
                axes[row, column].grid(True, alpha=0.3)
                handles, labels = axes[row, column].get_legend_handles_labels()
                if handles:
                    axes[row, column].legend(handles, labels, fontsize=8)
            axes[1, column].set_xlabel("alpha" if matching == "alpha" else "z")
        axes[0, 0].set_ylabel("d'")
        axes[1, 0].set_ylabel("mean adjusted recoded logit")
        figure.suptitle(f"Experiment 2 - layer {layer}")
        figure.tight_layout()
        path = output_dir / f"presence_layer{layer}.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Experiment 2: intervention-presence detection and false positives"
    )
    parser.add_argument(
        "--calibration_config",
        default="configs/experiment_0_calibration/development_full.yaml",
        help="Experiment 0 protocol configuration; defaults to all 32 layers.",
    )
    parser.add_argument(
        "--calibration_results",
        default="results/experiment_0_calibration",
        help="Directory containing directional_scales.json and run_manifest.json.",
    )
    parser.add_argument("--estimator", choices=["sd", "mad"], default=None)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument(
        "--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES)
    )
    parser.add_argument(
        "--matchings",
        nargs="+",
        default=list(MATCHINGS),
        choices=list(MATCHINGS) + [LEGACY_MATCHING],
    )
    parser.add_argument("--concepts", nargs="+", default=None)
    parser.add_argument("--num_random", type=int, default=2)
    parser.add_argument("--num_noise", type=int, default=1)
    parser.add_argument("--num_dropout", type=int, default=1)
    parser.add_argument("--num_pairs", type=int, default=5)
    parser.add_argument(
        "--label_orders", nargs="+", default=["AB"], choices=["AB", "BA"]
    )
    parser.add_argument("--alpha_doses", nargs="+", default=None)
    parser.add_argument("--alpha_range", nargs=2, default=None)
    parser.add_argument("--z_doses", nargs="+", default=None)
    parser.add_argument("--z_range", nargs=2, default=None)
    parser.add_argument("--num_doses", type=int, default=7)
    parser.add_argument(
        "--dropout_norm_source", choices=["trial", "calibration"], default="trial"
    )
    parser.add_argument("--bootstrap_iterations", type=int, default=1000)
    parser.add_argument("--zero_hook_validation", action="store_true", default=True)
    parser.add_argument(
        "--skip_zero_hook_validation", dest="zero_hook_validation", action="store_false"
    )
    parser.add_argument("--zero_hook_tolerance", type=float, default=0.0)
    parser.add_argument("--measure_amplitude", action="store_true", default=True)
    parser.add_argument(
        "--no_measure_amplitude", dest="measure_amplitude", action="store_false"
    )
    parser.add_argument("--limit_trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Model placement. Ruche jobs pass cuda explicitly; auto is portable.",
    )
    parser.add_argument(
        "--model_revision",
        default=None,
        help="Defaults to the immutable revision recorded by Experiment 0.",
    )
    parser.add_argument(
        "--tokenizer_revision",
        default=None,
        help="Defaults to Experiment 0's tokenizer revision, then the model revision.",
    )
    parser.add_argument("--output_dir", default="results/experiment2")
    parser.add_argument("--run_name", default="pilot")
    parser.add_argument(
        "--no_progress", dest="progress", action="store_false", default=True
    )
    parser.add_argument("--no_plots", dest="plots", action="store_false", default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the exact saved manifest from checkpoint.jsonl.",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.limit_trials is not None and (
        args.limit_trials <= 0 or args.limit_trials % 2
    ):
        parser.error(
            "--limit_trials must be a positive even number to preserve mapping pairs"
        )
    calibration = Experiment0Calibration.load(
        args.calibration_config,
        args.calibration_results,
        estimator=args.estimator or "sd",
    )
    if not calibration.config_matches_manifest:
        print(
            "WARNING: Experiment 0 results were produced from a different byte-level "
            "configuration. Core layers and direction metadata were validated, and the "
            "mismatch is recorded in the Experiment 2 manifest.",
            flush=True,
        )
    args.layers = args.layers or list(calibration.config.layers)
    missing = [layer for layer in args.layers if layer not in calibration.config.layers]
    if missing:
        parser.error(f"layers {missing} are not calibrated")
    args.concepts = args.concepts or list(calibration.config.concepts)
    alpha_doses = parse_doses(
        args.alpha_doses,
        args.alpha_range,
        args.num_doses,
        EXPERIMENT1_ALPHA_DOSES,
    )
    z_doses = parse_doses(
        args.z_doses, args.z_range, args.num_doses, EXPERIMENT1_Z_DOSES
    )
    args.dose_grid = {
        matching: (alpha_doses if matching == "alpha" else z_doses)
        for matching in args.matchings
    }
    requested_estimators = {
        estimator_for_matching(matching, calibration.estimator)
        for matching in args.matchings
    } - {None}
    for estimator in requested_estimators:
        field = calibration.ESTIMATOR_FIELDS[estimator]
        invalid = [
            direction_id
            for direction_id, row in calibration.records.items()
            if not float(row[field]) > 0.0
            or (estimator == "sd" and not row.get("valid_for_sd_normalization", True))
        ]
        if invalid:
            parser.error(
                f"{len(invalid)} directions are invalid for {estimator}; "
                f"first: {invalid[0]}"
            )
    model_name = args.model or calibration.config.model
    same_model_as_calibration = model_name == calibration.config.model
    model_revision = args.model_revision or (
        calibration.run_manifest.get("resolved_model_revision")
        if same_model_as_calibration
        else None
    )
    tokenizer_revision = args.tokenizer_revision or (
        calibration.run_manifest.get("resolved_tokenizer_revision") or model_revision
        if same_model_as_calibration
        else None
    )

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=tokenizer_revision)
    args.pairs = build_sentence_pairs(
        tokenizer, list(LOCALIZATION_SENTENCES), args.num_pairs, args.seed
    )
    conditions = build_conditions(calibration, args)
    plan = build_plan(conditions, args)
    full_plan_size = len(plan)
    if args.limit_trials is not None:
        plan = plan[: args.limit_trials]
    token_x, token_y = choice_token_ids(tokenizer)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir = output_dir / args.run_name
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume are mutually exclusive")
    existing_output = output_dir.exists() and any(output_dir.iterdir())
    if existing_output and not (args.overwrite or args.resume):
        parser.error(
            f"output directory is not empty: {output_dir}; use --resume, --overwrite, "
            "or a new run name"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": 2,
        "protocol_id": "experiment2-presence-v3",
        "protocol_locked": True,
        "shared_runtime_sha256": hashlib.sha256(
            (REPO_ROOT / "code/experiments/perturbation_runtime.py").read_bytes()
        ).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "model": model_name,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "device": args.device,
        "calibration_config": str(args.calibration_config),
        "calibration_results": str(args.calibration_results),
        "calibration_provenance": calibration.provenance(),
        "estimator": calibration.estimator,
        "seed": args.seed,
        "bootstrap_iterations": args.bootstrap_iterations,
        "layers": args.layers,
        "families": args.families,
        "matchings": args.matchings,
        "mappings": list(MAPPINGS),
        "response_protocol": {
            "assistant_prefix": "Answer:",
            "decision": "argmax restricted to the single-token X/Y choices",
            "unconstrained_global_argmax": "recorded as a diagnostic only",
        },
        "answer_token_ids": {"X": token_x, "Y": token_y},
        "concepts": args.concepts,
        "dose_grid": args.dose_grid,
        "label_orders": args.label_orders,
        "dropout_norm_source": args.dropout_norm_source,
        "pairs": args.pairs,
        "n_perturbed_forward_passes": len(plan),
        "n_full_perturbed_forward_passes": full_plan_size,
        "n_physical_sham_forward_passes": len(args.pairs)
        * 2
        * len(args.label_orders)
        * 2,
        "n_zero_hook_validation_passes": len(args.layers) * len(args.families),
        "plan": plan,
    }
    manifest_path = output_dir / "manifest.json"
    checkpoint_path = output_dir / "presence_checkpoint.jsonl"
    if args.resume:
        if not manifest_path.exists():
            parser.error(f"cannot resume without {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            previous_manifest = json.load(handle)
        if previous_manifest != manifest:
            parser.error(
                "the requested plan differs from the saved manifest; refusing to resume"
            )
    else:
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
        # --overwrite explicitly starts a new checkpoint while leaving unrelated run
        # artifacts intact until their replacements are successfully produced.
        checkpoint_path.write_text("", encoding="utf-8")

    print(
        f"Experiment 2 plan: {len(plan)} presence, "
        f"{manifest['n_physical_sham_forward_passes']} cached sham, "
        f"{manifest['n_zero_hook_validation_passes']} zero-hook validation passes",
        flush=True,
    )
    print(f"Output: {output_dir}", flush=True)
    if args.dry_run:
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    device_map = "auto" if args.device == "auto" else {"": args.device}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        dtype=torch.bfloat16,
        device_map=device_map,
    )
    model.eval()
    runtime = {
        "model": model_name,
        "model_class": type(model).__name__,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_revision": getattr(tokenizer, "_commit_hash", None),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "dtype": str(next(model.parameters()).dtype),
        "device": str(next(model.parameters()).device),
        "requested_device": args.device,
        "answer_token_ids": {"X": token_x, "Y": token_y},
        "hook": "output of model.model.layers[layer]",
    }
    with (output_dir / "runtime.json").open("w", encoding="utf-8") as handle:
        json.dump(runtime, handle, indent=2, ensure_ascii=False, sort_keys=True)
    existing_trials = load_checkpoint(checkpoint_path) if args.resume else []
    if existing_trials:
        print(
            f"Resuming after {len(existing_trials)} completed perturbed trials.",
            flush=True,
        )
    trials, zero_hook = run_experiment(
        model,
        tokenizer,
        calibration,
        args,
        plan,
        existing_trials=existing_trials,
        checkpoint=checkpoint_path,
        prompt_manifest=output_dir / "prompts.json",
    )
    summaries = summarize(trials, args.bootstrap_iterations, args.seed)
    write_trials_csv(output_dir / "trials.csv", trials)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **{key: value for key, value in manifest.items() if key != "plan"},
                "n_physical_trials": len(trials),
                "zero_hook_validation": zero_hook,
                "cells": summaries,
            },
            handle,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    if args.plots:
        plot_results(summaries, output_dir)
    print(
        f"Saved {len(trials)} presence trials and {len(summaries)} detection cells.",
        flush=True,
    )


if __name__ == "__main__":
    main()
