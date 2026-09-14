#!/usr/bin/env python3
"""
Block 0 - Calibration (multisteering plan)

Single-concept injection sweep (layer x alpha) to find, for this model, the
operating window where self-report is reliably above the false-positive rate.
Provides the baseline operating point for Blocks 1-4.

Uses apply_multi_injection with a single InjectionSpec: K=1 is the general
multi-injection engine's base case, and this is also its first real exercise
before Blocks 1-4 stack multiple specs on it.

alpha=0 doubles as the null/false-positive condition (no separate no-injection
branch needed): unit(vector) * 0 == no perturbation.
"""

import argparse
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from all_prompts import get_calibration_messages
from concepts import ALL_CONCEPTS, get_concept_description
from embedding_judge import cosine_similarity
from multi_inject import InjectionSpec, apply_multi_injection
from response_parsing import is_coherent, parse_yes_no, strip_answer_tag

DEFAULT_LAYERS = list(range(0, 32, 2))
DEFAULT_ALPHAS = [0, 1, 2, 4, 6, 8, 12, 16]


def load_vector(concept, layer, vec_type="avg"):
    """Load a concept vector for a specific layer."""
    vector_path = Path(f"saved_vectors/llama/{concept}_{layer}_{vec_type}.pt")
    if not vector_path.exists():
        raise FileNotFoundError(f"Vector not found: {vector_path}")
    data = torch.load(vector_path, weights_only=False)
    return data["vector"]


@torch.inference_mode()
def run_calibration(model, tokenizer, layers, alphas, num_trials, vec_type, max_new_tokens, seed):
    """
    For each (layer, alpha) cell, run num_trials trials, each with a concept
    drawn uniformly at random from the 10-concept pool (randomized
    concept<->layer assignment, per the plan's cross-cutting validity check #4).

    Returns a flat list of trial-level record dicts.
    """
    device = next(model.parameters()).device
    rng = random.Random(seed)

    messages = get_calibration_messages()
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(device)

    trials = []
    total = len(layers) * len(alphas) * num_trials
    pbar = tqdm(total=total, desc="calibration trials", file=sys.stdout)

    for layer in layers:
        for alpha in alphas:
            for trial_idx in range(num_trials):
                concept = rng.choice(ALL_CONCEPTS)
                vector = load_vector(concept, layer, vec_type)
                spec = InjectionSpec(layer=layer, vector=vector, alpha=float(alpha), token_range=None)

                with apply_multi_injection(model, [spec]):
                    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

                generated_ids = out[0][inputs.input_ids.shape[1]:]
                response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                coherent = is_coherent(response)
                claims_noticing = parse_yes_no(response)
                description = strip_answer_tag(response)
                sim = cosine_similarity(description, get_concept_description(concept)) if description else 0.0

                trials.append({
                    "layer": layer,
                    "alpha": alpha,
                    "concept": concept,
                    "vec_type": vec_type,
                    "trial_idx": trial_idx,
                    "response": response,
                    "is_coherent": coherent,
                    "claims_noticing": claims_noticing,
                    "concept_similarity": sim,
                })
                pbar.update(1)

    pbar.close()
    return trials


def save_trials(trials, layers, alphas, args, output_dir):
    output_path = Path(output_dir) / "calibration_trials.pt"
    torch.save({
        "trials": trials,
        "layers": layers,
        "alphas": alphas,
        "vec_type": args.vec_type,
        "num_trials": args.num_trials,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
    }, output_path)
    print(f"\nSaved {len(trials)} trials to {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument("--num_trials", type=int, default=5, help="Trials per (layer, alpha) cell")
    parser.add_argument("--vec_type", type=str, default="avg", choices=["avg", "last"])
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="plots")
    args = parser.parse_args()

    print("Loading model...", flush=True)
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded!", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("BLOCK 0 -- CALIBRATION", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Layers ({len(args.layers)}): {args.layers}", flush=True)
    print(f"Alphas ({len(args.alphas)}): {args.alphas}", flush=True)
    print(f"Trials per cell: {args.num_trials}", flush=True)
    print(f"Vector type: {args.vec_type}", flush=True)
    print(f"{'='*60}\n", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = run_calibration(
        model, tokenizer, args.layers, args.alphas,
        num_trials=args.num_trials, vec_type=args.vec_type,
        max_new_tokens=args.max_new_tokens, seed=args.seed,
    )

    save_trials(trials, args.layers, args.alphas, args, output_dir)

    print(f"\n{'='*60}", flush=True)
    print("CALIBRATION COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
