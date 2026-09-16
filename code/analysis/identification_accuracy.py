#!/usr/bin/env python3
"""
Analysis for Experience 10 identification (multi_identification.py's saved
trials, section 15):
  - Free response: fraction of injected-concept similarity scores above a
    pre-registered threshold, contrasted with the best-distractor
    similarity (embedding-judge grading).
  - Forced choice: exact match of the two reported labels (letters or NONE)
    against the two correct labels (order-insensitive, section 15.3).
  - NONE-slot false identification: for trials with at least one slot that
    should read NONE (single_concept, concept_plus_random, sham), the rate
    at which the free response's best match to ANY known concept exceeds the
    threshold.

Each metric is broken down by alpha, since multi_identification.py sweeps a
dose range (--alphas) in one file rather than one alpha per run.

Pass multiple --inputs (e.g. the four files from --run_all_conditions) to get
a breakdown per file/condition.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def load_trials(path):
    data = torch.load(path, weights_only=False)
    return data["trials"]


def free_response_accuracy(trials, threshold):
    coherent = [t for t in trials if t["mode"] == "free" and t["is_coherent"] and t.get("sim_to_injected")]
    if not coherent:
        return None
    injected_sims = [s for t in coherent for s in t["sim_to_injected"]]
    distractor_sims = [t["best_distractor_similarity"] for t in coherent if "best_distractor_similarity" in t]

    frac_above_threshold = float(np.mean([s > threshold for s in injected_sims]))
    frac_beats_distractor = float(np.mean([
        all(s > t.get("best_distractor_similarity", -1.0) for s in t["sim_to_injected"])
        for t in coherent
    ]))

    return {
        "n_trials": len(coherent),
        "n_concept_scores": len(injected_sims),
        "mean_sim_to_injected": float(np.mean(injected_sims)),
        "mean_best_distractor_similarity": float(np.mean(distractor_sims)) if distractor_sims else float("nan"),
        "frac_above_threshold": frac_above_threshold,
        "frac_beats_best_distractor": frac_beats_distractor,
    }


def forced_choice_accuracy(trials):
    coherent = [t for t in trials if t["mode"] == "forced_choice" and t["is_coherent"]]
    if not coherent:
        return None
    exact_match = [Counter(t["reported_labels"]) == Counter(t["correct_labels"]) for t in coherent]
    return {
        "n_trials": len(coherent),
        "exact_match_accuracy": float(np.mean(exact_match)),
    }


def none_slot_false_identification_rate(trials, threshold):
    """For free-response trials with at least one slot that should read NONE
    (best_match against ANY known concept, not just the injected ones)."""
    coherent = [
        t for t in trials
        if t["mode"] == "free" and t["is_coherent"] and "none_slot_best_match_similarity" in t
    ]
    if not coherent:
        return None
    false_id = float(np.mean([t["none_slot_best_match_similarity"] > threshold for t in coherent]))
    return {
        "n_trials": len(coherent),
        "false_identification_rate": false_id,
    }


def print_result(title, result):
    if not result:
        return
    print(f"\n--- {title} ---")
    for k, v in result.items():
        print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", default=[
        Path("plots/multi_identification_trials_llama_single_concept.pt"),
        Path("plots/multi_identification_trials_llama_two_concepts.pt"),
        Path("plots/multi_identification_trials_llama_concept_plus_random.pt"),
        Path("plots/multi_identification_trials_llama_sham.pt"),
    ], help="multi_identification.py now tags output filenames with --model, e.g. "
             "multi_identification_trials_qwen_two_concepts.pt for a Qwen run")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Pre-registered embedding-similarity threshold for a 'correct' identification")
    args = parser.parse_args()

    inputs = [p for p in args.inputs if p.exists()]
    if not inputs:
        raise FileNotFoundError(f"No input files found among: {args.inputs}")

    for path in inputs:
        trials = load_trials(path)
        alphas = sorted(set(t["alpha"] for t in trials))
        print("=" * 80)
        print(f"EXPERIENCE 10 IDENTIFICATION ANALYSIS: {path.name} "
              f"(n_trials={len(trials)}, threshold={args.threshold}, alphas={alphas})")
        print("=" * 80)

        print("\n### All alphas combined ###")
        print_result("Free response", free_response_accuracy(trials, args.threshold))
        print_result("Forced choice", forced_choice_accuracy(trials))
        print_result("NONE-slot false identification", none_slot_false_identification_rate(trials, args.threshold))

        for alpha in alphas:
            by_alpha = [t for t in trials if t["alpha"] == alpha]
            print(f"\n### alpha={alpha} ###")
            print_result("Free response", free_response_accuracy(by_alpha, args.threshold))
            print_result("Forced choice", forced_choice_accuracy(by_alpha))
            print_result("NONE-slot false identification",
                         none_slot_false_identification_rate(by_alpha, args.threshold))
        print()


if __name__ == "__main__":
    main()
