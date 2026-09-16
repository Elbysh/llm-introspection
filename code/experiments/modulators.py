#!/usr/bin/env python3
"""
Experience 11 modulators (plan_of_research.md section 16.5): layer distance,
absolute position, dose ratio, and concept-pair similarity, all applied to
the ordering task (section 16's A/B lettered 2AFC) -- the plan specifies no
modulator sweep for Experience 10 (identification), unlike the previous
version of this script, which ran E5/E6 against identification.

  E4 - layer distance |i-j|, at each requested distance.
    - each distance is tested at both an early and a late absolute position
      in the network (--placements), to confirm the effect tracks distance
      rather than absolute depth.
    - extreme distances |i-j|=1 and |i-j|=L-1=31, via --distances.
  E5 - dose ratio z_A/z_B, at a fixed layer pair (--layers). The ratio is
    tied to the A/B *labels* in the prompt, not to depth: which physical
    layer receives label A vs B is randomized per trial (independent of
    which is actually shallower), so the boosted dose lands on the shallow
    or deep layer with equal probability across trials.
  E6 - concept-pair cosine similarity, at a fixed layer pair, concept pairs
    selected via concepts.bucket_concept_pairs_by_similarity (reads saved
    vectors directly, no new precompute), including a near-identical pair as
    a limit case.

Every sweep answers via get_ordering_messages + parse_ab_choice and scores
against correct_letter (whichever label sits at the shallower layer).
alpha plays the role of the plan's z directly (no section-4 natural-scale
calibration in this pass; see multisteering_implementation_plan.md). Each
sweep also varies the base alpha over --alphas, with materials (concepts,
layers, A/B label draw) matched within a cell across that dose sweep --
in E5 this crosses with the ratio, so alpha_a/alpha_b become
alpha*ratio/alpha for each base alpha.

--experiments runs several of the three in one process, sharing one model load.
"""

import argparse
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from all_prompts import get_ordering_messages
from concepts import ALL_CONCEPTS, bucket_concept_pairs_by_similarity
from model_registry import build_inputs, load_model_and_tokenizer
from multi_inject import InjectionSpec, apply_multi_injection
from response_parsing import is_coherent, parse_ab_choice

DEFAULT_DISTANCES = [1, 2, 4, 8, 16, 31]
DEFAULT_ALPHAS = [1, 2, 3, 4, 5, 6, 7]
DEFAULT_ALPHA_RATIOS = [1.0, 2.0, 4.0, 8.0]


def load_vector(concept, layer, vector_dir, vec_type="avg"):
    """Load a concept vector for a specific layer."""
    vector_path = Path(f"{vector_dir}/{concept}_{layer}_{vec_type}.pt")
    if not vector_path.exists():
        raise FileNotFoundError(f"Vector not found: {vector_path}")
    data = torch.load(vector_path, weights_only=False)
    return data["vector"]


def _generate_with_first_token_logits(model, tokenizer, model_layers, messages, max_new_tokens, specs):
    """Like _generate, but also returns the full-vocabulary logits at the
    first generated position (section 16.6's logit-contrast measure). Reads
    them off the same generate() call (output_scores=True) rather than a
    second forward pass, so they reflect the injection exactly as it
    influenced the actual decoded response."""
    device = next(model.parameters()).device
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = build_inputs(tokenizer, formatted_prompt, device)
    with apply_multi_injection(model, specs, layers=model_layers):
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                              output_scores=True, return_dict_in_generate=True)
    generated_ids = out.sequences[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    first_token_logits = out.scores[0][0]
    return response, first_token_logits


def _sample_layer_pair_with_distance(rng, distance, placement, layer_max):
    """Samples (layer_low, layer_high) with layer_high - layer_low == distance,
    constrained to the early half (0..layer_max//2) or late half
    (layer_max//2..layer_max) of the network (C4.1). Falls back to the full
    range if the requested placement can't fit the distance (e.g. the C4.2
    extreme distance L-1, which only fits the full range)."""
    half = layer_max // 2
    lo_bound, hi_bound = (0, half) if placement == "early" else (half, layer_max)

    max_start = hi_bound - distance
    if max_start < lo_bound:
        lo_bound, hi_bound = 0, layer_max
        max_start = hi_bound - distance
    layer_low = rng.randint(lo_bound, max_start)
    return layer_low, layer_low + distance


def _run_ordering_item(model, tokenizer, model_layers, layer_a, concept_a, alpha_a,
                        layer_b, concept_b, alpha_b, vector_dir, vec_type, max_new_tokens):
    """One Exp 11 ordering trial with an explicit per-label (layer, concept,
    alpha) assignment. correct_letter is whichever label sits at the
    shallower (lower-index) layer.

    Records both the raw A-vs-B logit contrast (characterizes the model's
    raw letter preference, independent of ground truth) and the "adjusted"
    contrast -- logit(correct letter) - logit(incorrect letter), section
    16.6 -- which flips sign per trial so a positive value always means
    evidence toward the correct answer, canceling out a fixed A/B
    preference. Computed for every trial, including incoherent/unparsed
    ones, since it doesn't depend on parsing the decoded text.
    """
    specs = [
        InjectionSpec(layer=layer_a, vector=load_vector(concept_a, layer_a, vector_dir, vec_type), alpha=alpha_a),
        InjectionSpec(layer=layer_b, vector=load_vector(concept_b, layer_b, vector_dir, vec_type), alpha=alpha_b),
    ]
    correct_letter = "A" if layer_a < layer_b else "B"
    response, first_token_logits = _generate_with_first_token_logits(
        model, tokenizer, model_layers, get_ordering_messages(concept_a, concept_b), max_new_tokens, specs)
    reported = parse_ab_choice(response)

    token_id_a = tokenizer.encode("A", add_special_tokens=False)[0]
    token_id_b = tokenizer.encode("B", add_special_tokens=False)[0]
    logit_a = first_token_logits[token_id_a].item()
    logit_b = first_token_logits[token_id_b].item()
    logit_correct = logit_a if correct_letter == "A" else logit_b
    logit_incorrect = logit_b if correct_letter == "A" else logit_a

    return {
        "correct_letter": correct_letter,
        "response": response,
        # min_length=1: the format demands a single letter, A or B.
        "is_coherent": is_coherent(response, min_length=1),
        "reported_letter": reported,
        "correct": (reported == correct_letter) if reported is not None else None,
        "logit_a": logit_a,
        "logit_b": logit_b,
        "logit_contrast_ab": logit_a - logit_b,
        "logit_contrast_adjusted": logit_correct - logit_incorrect,
    }


@torch.inference_mode()
def run_e4_distance_sweep(model, tokenizer, spec, distances, placements, alphas, num_trials,
                           vec_type, max_new_tokens, seed):
    rng = random.Random(seed)
    model_layers = spec.get_layers(model)
    layer_max = spec.num_layers - 1
    trials = []
    total = len(distances) * len(placements) * num_trials * len(alphas)
    pbar = tqdm(total=total, desc="E4 distance sweep (ordering)", file=sys.stdout)

    for distance in distances:
        for placement in placements:
            for trial_idx in range(num_trials):
                layer_low, layer_high = _sample_layer_pair_with_distance(rng, distance, placement, layer_max)
                concept_low, concept_high = rng.sample(ALL_CONCEPTS, 2)

                if rng.random() < 0.5:
                    layer_a, concept_a, layer_b, concept_b = layer_low, concept_low, layer_high, concept_high
                else:
                    layer_a, concept_a, layer_b, concept_b = layer_high, concept_high, layer_low, concept_low

                for alpha in alphas:
                    result = _run_ordering_item(model, tokenizer, model_layers, layer_a, concept_a, alpha,
                                                 layer_b, concept_b, alpha, spec.vector_dir, vec_type, max_new_tokens)
                    trials.append({
                        "distance": distance, "placement": placement,
                        "layer_low": layer_low, "layer_high": layer_high,
                        "concept_low": concept_low, "concept_high": concept_high,
                        "alpha": alpha, **result,
                    })
                    pbar.update(1)

    pbar.close()
    return trials


@torch.inference_mode()
def run_e5_alpha_ratio_sweep(model, tokenizer, spec, layers, ratios, alphas, num_trials,
                              vec_type, max_new_tokens, seed):
    if len(layers) != 2:
        raise ValueError("E5 requires exactly 2 fixed layers (--layers l1 l2)")
    rng = random.Random(seed)
    model_layers = spec.get_layers(model)
    layer_low, layer_high = sorted(layers)
    trials = []
    total = len(ratios) * num_trials * len(alphas)
    pbar = tqdm(total=total, desc="E5 dose-ratio sweep (ordering)", file=sys.stdout)

    for ratio in ratios:
        for trial_idx in range(num_trials):
            concept_1, concept_2 = rng.sample(ALL_CONCEPTS, 2)
            # Which physical layer gets label A (and therefore the boosted
            # dose) is randomized per trial, independent of depth, so the
            # ratio's effect on the reported letter isn't confounded with
            # depth. Held fixed across the alpha sweep below (matched
            # materials): only the base alpha varies for a given trial.
            if rng.random() < 0.5:
                layer_a, concept_a, layer_b, concept_b = layer_low, concept_1, layer_high, concept_2
            else:
                layer_a, concept_a, layer_b, concept_b = layer_high, concept_1, layer_low, concept_2

            for base_alpha in alphas:
                alpha_a, alpha_b = base_alpha * ratio, base_alpha
                result = _run_ordering_item(model, tokenizer, model_layers, layer_a, concept_a, alpha_a,
                                             layer_b, concept_b, alpha_b, spec.vector_dir, vec_type, max_new_tokens)
                trials.append({
                    "ratio": ratio, "base_alpha": base_alpha,
                    "layer_a": layer_a, "layer_b": layer_b,
                    "alpha_a": alpha_a, "alpha_b": alpha_b,
                    "concept_a": concept_a, "concept_b": concept_b,
                    **result,
                })
                pbar.update(1)

    pbar.close()
    return trials


@torch.inference_mode()
def run_e6_similarity_sweep(model, tokenizer, spec, layers, similarity_reference_layer, num_buckets,
                             alphas, num_trials, vec_type, max_new_tokens, seed):
    if len(layers) != 2:
        raise ValueError("E6 requires exactly 2 fixed injection layers (--layers l1 l2), "
                          "held constant across buckets so only concept similarity varies")
    rng = random.Random(seed)
    model_layers = spec.get_layers(model)
    layer_low, layer_high = sorted(layers)
    representatives = bucket_concept_pairs_by_similarity(
        similarity_reference_layer, num_buckets=num_buckets, vec_type=vec_type,
        saved_vectors_dir=spec.vector_dir,
    )

    trials = []
    total = len(representatives) * num_trials * len(alphas)
    pbar = tqdm(total=total, desc="E6 similarity sweep (ordering)", file=sys.stdout)

    for rep in representatives:
        for trial_idx in range(num_trials):
            concept_1, concept_2 = rep["pair"]
            if rng.random() < 0.5:
                layer_a, concept_a, layer_b, concept_b = layer_low, concept_1, layer_high, concept_2
            else:
                layer_a, concept_a, layer_b, concept_b = layer_high, concept_1, layer_low, concept_2

            for alpha in alphas:
                result = _run_ordering_item(model, tokenizer, model_layers, layer_a, concept_a, alpha,
                                             layer_b, concept_b, alpha, spec.vector_dir, vec_type, max_new_tokens)
                trials.append({
                    "bucket": rep["bucket"], "concept_pair_similarity": rep["similarity"],
                    "similarity_reference_layer": similarity_reference_layer,
                    "layer_a": layer_a, "layer_b": layer_b,
                    "concept_a": concept_a, "concept_b": concept_b,
                    "alpha": alpha, **result,
                })
                pbar.update(1)

    pbar.close()
    return trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama", choices=["llama", "qwen"])
    parser.add_argument("--experiments", type=str, nargs="+",
                         default=["e4_distance", "e5_alpha_ratio", "e6_similarity"],
                         choices=["e4_distance", "e5_alpha_ratio", "e6_similarity"])
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS,
                         help="Sweep of base alpha (=z) values; materials are matched within a "
                              "cell across this sweep (E5 crosses it with --alpha_ratios)")
    parser.add_argument("--num_trials", type=int, default=20)
    parser.add_argument("--vec_type", type=str, default="avg", choices=["avg", "last"])
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="plots")

    # E4
    parser.add_argument("--distances", type=int, nargs="+", default=DEFAULT_DISTANCES)
    parser.add_argument("--placements", type=str, nargs="+", default=["early", "late"], choices=["early", "late"])

    # E5 / E6 (both use a fixed injection layer pair)
    parser.add_argument("--layers", type=int, nargs="+", default=[8, 24],
                         help="Fixed layer pair for E5/E6")
    parser.add_argument("--alpha_ratios", type=float, nargs="+", default=DEFAULT_ALPHA_RATIOS)

    # E6
    parser.add_argument("--similarity_reference_layer", type=int, default=16)
    parser.add_argument("--num_similarity_buckets", type=int, default=4)

    args = parser.parse_args()

    print("Loading model...", flush=True)
    model, tokenizer, spec = load_model_and_tokenizer(args.model)
    print("Model loaded!", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for experiment in args.experiments:
        print(f"\n{'='*60}\nEXPERIENCE 11 -- MODULATORS ({experiment}, {spec.key}: {spec.repo_id})\n{'='*60}",
              flush=True)

        if experiment == "e4_distance":
            trials = run_e4_distance_sweep(
                model, tokenizer, spec, args.distances, args.placements, args.alphas,
                args.num_trials, args.vec_type, args.max_new_tokens, args.seed,
            )
            meta = {"distances": args.distances, "placements": args.placements}
        elif experiment == "e5_alpha_ratio":
            trials = run_e5_alpha_ratio_sweep(
                model, tokenizer, spec, args.layers, args.alpha_ratios, args.alphas,
                args.num_trials, args.vec_type, args.max_new_tokens, args.seed,
            )
            meta = {"layers": args.layers, "alpha_ratios": args.alpha_ratios}
        else:
            trials = run_e6_similarity_sweep(
                model, tokenizer, spec, args.layers, args.similarity_reference_layer,
                args.num_similarity_buckets, args.alphas, args.num_trials,
                args.vec_type, args.max_new_tokens, args.seed,
            )
            meta = {
                "layers": args.layers,
                "similarity_reference_layer": args.similarity_reference_layer,
                "num_similarity_buckets": args.num_similarity_buckets,
            }

        output_path = output_dir / f"modulators_trials_{spec.key}_{experiment}.pt"
        torch.save({
            "trials": trials,
            "model": spec.key,
            "experiment": experiment,
            "alphas": args.alphas,
            "vec_type": args.vec_type,
            "num_trials": args.num_trials,
            "seed": args.seed,
            **meta,
        }, output_path)
        print(f"Saved {len(trials)} trials to {output_path}", flush=True)


if __name__ == "__main__":
    main()
