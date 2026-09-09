#!/usr/bin/env python3
"""
Analysis for Block 2 identification (multi_identification.py's saved trials):
  - Free response (E2, C2.1): fraction of injected-concept similarity scores
    above a pre-registered threshold, contrasted with the best-distractor
    similarity (embedding-judge grading, per the plan's E2 grading spec).
  - Forced choice (C2.2): exact-match accuracy against ground truth.
  - Sham slot (C2.3): false-identification rate -- fraction of sham trials
    where the model's best match to ANY known concept exceeds the threshold.

Pass multiple --inputs (e.g. the four files from --run_all_conditions) to get
a breakdown per file/condition.
"""

import argparse
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
    exact_match = [set(t["reported_letters"]) == set(t["correct_letters"]) for t in coherent]
    return {
        "n_trials": len(coherent),
        "exact_match_accuracy": float(np.mean(exact_match)),
    }


def sham_false_identification_rate(trials, threshold):
    coherent = [
        t for t in trials
        if t["is_coherent"] and t.get("sham_slots", 0) > 0 and "sham_best_match_similarity" in t
    ]
    if not coherent:
        return None
    false_id = float(np.mean([t["sham_best_match_similarity"] > threshold for t in coherent]))
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
    parser.add_argument("--inputs", type=Path, nargs="+",
                         default=[Path("plots/multi_identification_trials.pt")])
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Pre-registered embedding-similarity threshold for a 'correct' identification")
    args = parser.parse_args()

    inputs = [p for p in args.inputs if p.exists()]
    if not inputs:
        raise FileNotFoundError(f"No input files found among: {args.inputs}")

    for path in inputs:
        trials = load_trials(path)
        print("=" * 80)
        print(f"BLOCK 2 IDENTIFICATION ANALYSIS: {path.name} (n_trials={len(trials)}, threshold={args.threshold})")
        print("=" * 80)

        print_result("Free response (E2 / C2.1)", free_response_accuracy(trials, args.threshold))
        print_result("Forced choice (C2.2)", forced_choice_accuracy(trials))
        print_result("Sham slot (C2.3)", sham_false_identification_rate(trials, args.threshold))
        print()


if __name__ == "__main__":
    main()
