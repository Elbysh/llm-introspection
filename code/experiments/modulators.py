#!/usr/bin/env python3
"""
Block 4 - Modulators (H4, H5, H6)

Thin parametrized sweeps around Block 2 (identification, E2) and Block 3
(ordering, E3), reusing their prompts and grading:

  E4 (H4) - layer distance |i-j|: both identification and ordering, at each
    requested distance.
    C4.1 - each distance is tested at both an early and a late absolute
           position in the network (--placements), to confirm the effect
           tracks distance rather than absolute depth.
    C4.2 - extreme distances |i-j|=1 and |i-j|=L-1=31, via --distances.
  E5 (H5) - relative injection strength alpha_A/alpha_B, at a fixed layer pair
    (identification only, per the plan's "Repeat E2").
  E6 (H6, bonus) - concept-pair cosine similarity, at a fixed injection layer
    pair (identification only), concept pairs selected via
    concepts.bucket_concept_pairs_by_similarity (reads saved vectors directly,
    no new precompute), including a near-identical pair as a limit case.

--experiments runs several of the three in one process, sharing one model load.
"""

import argparse
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from all_prompts import get_identification_messages, get_ordering_messages
from concepts import ALL_CONCEPTS, bucket_concept_pairs_by_similarity, get_concept_description
from embedding_judge import cosine_similarity
from multi_inject import InjectionSpec, apply_multi_injection
from response_parsing import is_coherent, parse_concept_choice

ALL_LAYERS = list(range(0, 32))
DEFAULT_DISTANCES = [1, 2, 4, 8, 16, 31]
DEFAULT_ALPHA_RATIOS = [1.0, 2.0, 4.0, 8.0]


def load_vector(concept, layer, vec_type="avg"):
    """Load a concept vector for a specific layer."""
    vector_path = Path(f"saved_vectors/llama/{concept}_{layer}_{vec_type}.pt")
    if not vector_path.exists():
        raise FileNotFoundError(f"Vector not found: {vector_path}")
    data = torch.load(vector_path, weights_only=False)
    return data["vector"]


def _generate(model, tokenizer, messages, max_new_tokens, specs):
    device = next(model.parameters()).device
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with apply_multi_injection(model, specs):
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def _sample_layer_pair_with_distance(rng, distance, placement, layer_max=31):
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


@torch.inference_mode()
def run_e4_distance_sweep(model, tokenizer, distances, placements, alpha, num_trials,
                           vec_type, max_new_tokens, seed):
    rng = random.Random(seed)
    trials = []
    total = len(distances) * len(placements) * num_trials * 2  # x2: identification + ordering
    pbar = tqdm(total=total, desc="E4 distance sweep", file=sys.stdout)

    for distance in distances:
        for placement in placements:
            for trial_idx in range(num_trials):
                layer_low, layer_high = _sample_layer_pair_with_distance(rng, distance, placement)
                concept_low, concept_high = rng.sample(ALL_CONCEPTS, 2)
                specs = [
                    InjectionSpec(layer=layer_low, vector=load_vector(concept_low, layer_low, vec_type), alpha=alpha),
                    InjectionSpec(layer=layer_high, vector=load_vector(concept_high, layer_high, vec_type), alpha=alpha),
                ]

                # --- identification (E2) ---
                response = _generate(model, tokenizer, get_identification_messages(2), max_new_tokens, specs)
                sims = [cosine_similarity(response, get_concept_description(c)) for c in (concept_low, concept_high)] if response else [0.0, 0.0]
                trials.append({
                    "task": "identification", "distance": distance, "placement": placement,
                    "layer_low": layer_low, "layer_high": layer_high,
                    "concept_low": concept_low, "concept_high": concept_high,
                    "alpha": alpha, "response": response,
                    "is_coherent": is_coherent(response), "sim_to_injected": sims,
                })
                pbar.update(1)

                # --- ordering (E3) ---
                response = _generate(model, tokenizer, get_ordering_messages(concept_low, concept_high), max_new_tokens, specs)
                reported = parse_concept_choice(response, concept_low, concept_high)
                trials.append({
                    "task": "ordering", "distance": distance, "placement": placement,
                    "layer_low": layer_low, "layer_high": layer_high,
                    "concept_low": concept_low, "concept_high": concept_high,
                    "alpha": alpha, "response": response,
                    "is_coherent": is_coherent(response), "reported_concept": reported,
                    "correct": (reported == concept_low) if reported is not None else None,
                })
                pbar.update(1)

    pbar.close()
    return trials


@torch.inference_mode()
def run_e5_alpha_ratio_sweep(model, tokenizer, layers, ratios, base_alpha, num_trials,
                              vec_type, max_new_tokens, seed):
    if len(layers) != 2:
        raise ValueError("E5 requires exactly 2 fixed layers (--layers l1 l2)")
    rng = random.Random(seed)
    layer_a, layer_b = layers
    trials = []
    total = len(ratios) * num_trials
    pbar = tqdm(total=total, desc="E5 alpha-ratio sweep", file=sys.stdout)

    for ratio in ratios:
        alpha_a, alpha_b = base_alpha * ratio, base_alpha
        for trial_idx in range(num_trials):
            concept_a, concept_b = rng.sample(ALL_CONCEPTS, 2)
            specs = [
                InjectionSpec(layer=layer_a, vector=load_vector(concept_a, layer_a, vec_type), alpha=alpha_a),
                InjectionSpec(layer=layer_b, vector=load_vector(concept_b, layer_b, vec_type), alpha=alpha_b),
            ]

            response = _generate(model, tokenizer, get_identification_messages(2), max_new_tokens, specs)
            sims = [cosine_similarity(response, get_concept_description(c)) for c in (concept_a, concept_b)] if response else [0.0, 0.0]
            trials.append({
                "task": "identification", "ratio": ratio,
                "layer_a": layer_a, "layer_b": layer_b,
                "alpha_a": alpha_a, "alpha_b": alpha_b,
                "concept_a": concept_a, "concept_b": concept_b,
                "response": response, "is_coherent": is_coherent(response), "sim_to_injected": sims,
            })
            pbar.update(1)

    pbar.close()
    return trials


@torch.inference_mode()
def run_e6_similarity_sweep(model, tokenizer, layers, similarity_reference_layer, num_buckets,
                             alpha, num_trials, vec_type, max_new_tokens, seed):
    if len(layers) != 2:
        raise ValueError("E6 requires exactly 2 fixed injection layers (--layers l1 l2), "
                          "held constant across buckets so only concept similarity varies")
    rng = random.Random(seed)
    layer_a, layer_b = layers
    representatives = bucket_concept_pairs_by_similarity(
        similarity_reference_layer, num_buckets=num_buckets, vec_type=vec_type
    )

    trials = []
    total = len(representatives) * num_trials
    pbar = tqdm(total=total, desc="E6 similarity sweep", file=sys.stdout)

    for rep in representatives:
        for trial_idx in range(num_trials):
            concept_a, concept_b = rep["pair"]
            # Randomize which concept goes at which layer to avoid confounding
            # concept identity with layer position across trials.
            if rng.random() < 0.5:
                concept_a, concept_b = concept_b, concept_a
            specs = [
                InjectionSpec(layer=layer_a, vector=load_vector(concept_a, layer_a, vec_type), alpha=alpha),
                InjectionSpec(layer=layer_b, vector=load_vector(concept_b, layer_b, vec_type), alpha=alpha),
            ]

            response = _generate(model, tokenizer, get_identification_messages(2), max_new_tokens, specs)
            sims = [cosine_similarity(response, get_concept_description(c)) for c in (concept_a, concept_b)] if response else [0.0, 0.0]
            trials.append({
                "task": "identification",
                "bucket": rep["bucket"], "concept_pair_similarity": rep["similarity"],
                "similarity_reference_layer": similarity_reference_layer,
                "layer_a": layer_a, "layer_b": layer_b,
                "concept_a": concept_a, "concept_b": concept_b,
                "alpha": alpha, "response": response,
                "is_coherent": is_coherent(response), "sim_to_injected": sims,
            })
            pbar.update(1)

    pbar.close()
    return trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=str, nargs="+",
                         default=["e4_distance", "e5_alpha_ratio", "e6_similarity"],
                         choices=["e4_distance", "e5_alpha_ratio", "e6_similarity"])
    parser.add_argument("--alpha", type=float, default=8.0,
                         help="Base alpha; pick from Block 0's operating window")
    parser.add_argument("--num_trials", type=int, default=20)
    parser.add_argument("--vec_type", type=str, default="avg", choices=["avg", "last"])
    parser.add_argument("--max_new_tokens", type=int, default=150)
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
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    print("Model loaded!", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for experiment in args.experiments:
        print(f"\n{'='*60}\nBLOCK 4 -- MODULATORS ({experiment})\n{'='*60}", flush=True)

        if experiment == "e4_distance":
            trials = run_e4_distance_sweep(
                model, tokenizer, args.distances, args.placements, args.alpha,
                args.num_trials, args.vec_type, args.max_new_tokens, args.seed,
            )
            meta = {"distances": args.distances, "placements": args.placements}
        elif experiment == "e5_alpha_ratio":
            trials = run_e5_alpha_ratio_sweep(
                model, tokenizer, args.layers, args.alpha_ratios, args.alpha,
                args.num_trials, args.vec_type, args.max_new_tokens, args.seed,
            )
            meta = {"layers": args.layers, "alpha_ratios": args.alpha_ratios, "base_alpha": args.alpha}
        else:
            trials = run_e6_similarity_sweep(
                model, tokenizer, args.layers, args.similarity_reference_layer,
                args.num_similarity_buckets, args.alpha, args.num_trials,
                args.vec_type, args.max_new_tokens, args.seed,
            )
            meta = {
                "layers": args.layers,
                "similarity_reference_layer": args.similarity_reference_layer,
                "num_similarity_buckets": args.num_similarity_buckets,
            }

        output_path = output_dir / f"modulators_trials_{experiment}.pt"
        torch.save({
            "trials": trials,
            "experiment": experiment,
            "alpha": args.alpha,
            "vec_type": args.vec_type,
            "num_trials": args.num_trials,
            "seed": args.seed,
            **meta,
        }, output_path)
        print(f"Saved {len(trials)} trials to {output_path}", flush=True)


if __name__ == "__main__":
    main()
