#!/usr/bin/env python3
"""
Block 1 - Detection (H1)

E1 - Count report: K (varied) injections at different layers, equal alpha.

Controls:
  C1.1 - Null injection (k=0): included via --k_values 0 ... -> false-positive rate.
  C1.2 - Reported vs. actual k confusion matrix: built downstream by
         code/analysis/count_confusion_matrix.py from this script's saved records.
  C1.3 - Rephrasing check: --prompt_styles scale uses a 0-10 "how different does
         processing feel" prompt instead of a direct count-report prompt, to
         check results aren't an artifact of yes-biased phrasing.
  C1.4 - Sham injection: --conditions sham replaces the k concept vectors with k
         random unit vectors of the same alpha (sham_vectors.make_random_direction),
         isolating generic-anomaly detection from concept-specific detection.
"""

import argparse
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from all_prompts import get_count_report_messages, get_scale_rating_messages
from concepts import ALL_CONCEPTS
from multi_inject import InjectionSpec, apply_multi_injection
from response_parsing import is_coherent, parse_count, parse_scale_0_10
from sham_vectors import infer_hidden_dim, make_random_direction

DEFAULT_LAYER_POOL = list(range(0, 32, 2))
DEFAULT_K_VALUES = [0, 1, 2, 3, 4]

PROMPT_BUILDERS = {
    "count": get_count_report_messages,
    "scale": get_scale_rating_messages,
}


def load_vector(concept, layer, vec_type="avg"):
    """Load a concept vector for a specific layer."""
    vector_path = Path(f"saved_vectors/llama/{concept}_{layer}_{vec_type}.pt")
    if not vector_path.exists():
        raise FileNotFoundError(f"Vector not found: {vector_path}")
    data = torch.load(vector_path, weights_only=False)
    return data["vector"]


def build_specs(k, layer_pool, alpha, vec_type, condition, rng, hidden_dim, sham_seed_base):
    """Sample k distinct layers, then k real concept vectors ('real' condition)
    or k random directions of the same alpha ('sham' condition, C1.4)."""
    layers = rng.sample(layer_pool, k) if k > 0 else []
    if condition == "real":
        concepts = rng.sample(ALL_CONCEPTS, k) if k > 0 else []
        specs = [
            InjectionSpec(layer=layer, vector=load_vector(concept, layer, vec_type), alpha=alpha)
            for layer, concept in zip(layers, concepts)
        ]
        return specs, layers, concepts
    else:  # sham
        specs = [
            InjectionSpec(layer=layer, vector=make_random_direction(hidden_dim, sham_seed_base + i), alpha=alpha)
            for i, layer in enumerate(layers)
        ]
        return specs, layers, [None] * k


@torch.inference_mode()
def run_detection(model, tokenizer, k_values, conditions, prompt_styles, layer_pool, alpha,
                   num_trials, vec_type, max_new_tokens, seed):
    device = next(model.parameters()).device
    rng = random.Random(seed)
    hidden_dim = infer_hidden_dim()

    trials = []
    total = len(k_values) * len(conditions) * len(prompt_styles) * num_trials
    pbar = tqdm(total=total, desc="detection trials", file=sys.stdout)
    trial_counter = 0

    for prompt_style in prompt_styles:
        messages = PROMPT_BUILDERS[prompt_style]()
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(device)

        for condition in conditions:
            for k in k_values:
                for trial_idx in range(num_trials):
                    trial_counter += 1
                    specs, layers, concepts = build_specs(
                        k, layer_pool, alpha, vec_type, condition, rng, hidden_dim,
                        sham_seed_base=seed * 1_000_000 + trial_counter * 10,
                    )

                    with apply_multi_injection(model, specs):
                        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                    generated_ids = out[0][inputs.input_ids.shape[1]:]
                    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                    record = {
                        "prompt_style": prompt_style,
                        "condition": condition,
                        "k_actual": k,
                        "layers": layers,
                        "concepts": concepts,
                        "alpha": alpha,
                        "vec_type": vec_type,
                        "trial_idx": trial_idx,
                        "response": response,
                        "is_coherent": is_coherent(response),
                    }
                    if prompt_style == "count":
                        record["reported_count"] = parse_count(response)
                    else:
                        record["reported_scale"] = parse_scale_0_10(response)

                    trials.append(record)
                    pbar.update(1)

    pbar.close()
    return trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_values", type=int, nargs="+", default=DEFAULT_K_VALUES)
    parser.add_argument("--conditions", type=str, nargs="+", default=["real", "sham"], choices=["real", "sham"])
    parser.add_argument("--prompt_styles", type=str, nargs="+", default=["count"], choices=["count", "scale"])
    parser.add_argument("--layer_pool", type=int, nargs="+", default=DEFAULT_LAYER_POOL)
    parser.add_argument("--alpha", type=float, default=8.0,
                         help="Equal alpha for all K injections; pick from Block 0's operating window")
    parser.add_argument("--num_trials", type=int, default=20)
    parser.add_argument("--vec_type", type=str, default="avg", choices=["avg", "last"])
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="plots")
    args = parser.parse_args()

    if max(args.k_values) > len(args.layer_pool):
        raise ValueError(f"layer_pool has {len(args.layer_pool)} layers, too few for max k={max(args.k_values)}")

    print("Loading model...", flush=True)
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    print("Model loaded!", flush=True)

    print(f"\n{'='*60}\nBLOCK 1 -- DETECTION (E1 + C1.1-C1.4)\n{'='*60}", flush=True)
    print(f"K values: {args.k_values}", flush=True)
    print(f"Conditions: {args.conditions}", flush=True)
    print(f"Prompt styles: {args.prompt_styles}", flush=True)
    print(f"Alpha: {args.alpha}", flush=True)
    print(f"Trials per cell: {args.num_trials}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = run_detection(
        model, tokenizer, args.k_values, args.conditions, args.prompt_styles,
        args.layer_pool, args.alpha, args.num_trials, args.vec_type,
        args.max_new_tokens, args.seed,
    )

    output_path = output_dir / "multi_detection_trials.pt"
    torch.save({
        "trials": trials,
        "k_values": args.k_values,
        "conditions": args.conditions,
        "prompt_styles": args.prompt_styles,
        "alpha": args.alpha,
        "vec_type": args.vec_type,
        "num_trials": args.num_trials,
        "seed": args.seed,
    }, output_path)
    print(f"\nSaved {len(trials)} trials to {output_path}", flush=True)


if __name__ == "__main__":
    main()
