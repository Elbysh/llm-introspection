#!/usr/bin/env python3
"""
Experience 9 - counting distinct injections (plan_of_research.md section 14).

k (varied) injections at different layers. z plays the role of the plan's
standardized dose without the section-4 natural-scale calibration: alpha IS z
here, applied as a raw amplitude on a unit-normalized direction (see
multisteering_implementation_plan.md's scope decision).

Two dosing regimes (section 14.2), selected via --dose_regime:
  individual - every injection gets the same alpha, swept over --alphas.
  budget     - a fixed total budget, swept over --z_totals, split equally
               across the k injections: alpha_i = z_total / sqrt(k) for k>0.

Each dose value in the active sweep is run as its own cell (k x condition x
prompt_style x dose), tagged per trial -- one saved file covers the whole
sweep, so downstream analysis can break accuracy down by dose.

k=0 is the plan's actual sham (no injection at all), always included via
--k_values 0. --conditions random additionally replaces the k concept
vectors with k random unit vectors of the same per-injection alpha
(sham_vectors.make_random_direction) -- this is the plan's "active control"
(section 14.4 step 6: directions aleatoires as controle actif), not its sham.

Reported vs. actual k confusion matrix and false-positive rate are built
downstream by code/analysis/count_confusion_matrix.py from this script's
saved records. --prompt_styles scale uses a 0-10 "how different does
processing feel" prompt instead of a direct count-report prompt, to check
results aren't an artifact of yes-biased phrasing.

For count-style trials, also records a bias-adjusted logit contrast:
logit(digit matching the true k) - logit(digit "1"), the direct digit-level
analog of Exp 11's A/B logit contrast -- the model shows a strong bias
toward reporting "1" regardless of k (even at k=0), so this isolates
whether the true count gets more support than that default, beyond the raw
reported digit. Trivially 0 when k_actual == 1. No separate sham run is
needed for this baseline: k=0 is already in the --k_values sweep.
"""

import argparse
import math
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

DEFAULT_LAYER_POOL = list(range(0, 32))
DEFAULT_K_VALUES = [0, 1, 2, 3, 4]
DEFAULT_ALPHAS = [1, 2, 3, 4, 5, 6, 7]
DEFAULT_Z_TOTALS = [1, 2, 3, 4, 5, 6, 7]

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
    """Sample k distinct layers, then k real concept vectors ('real'
    condition) or k random directions at the same alpha ('random' condition,
    the plan's active control -- not its sham, which is k=0)."""
    layers = rng.sample(layer_pool, k) if k > 0 else []
    if condition == "real":
        concepts = rng.sample(ALL_CONCEPTS, k) if k > 0 else []
        specs = [
            InjectionSpec(layer=layer, vector=load_vector(concept, layer, vec_type), alpha=alpha)
            for layer, concept in zip(layers, concepts)
        ]
        return specs, layers, concepts
    else:  # random
        specs = [
            InjectionSpec(layer=layer, vector=make_random_direction(hidden_dim, sham_seed_base + i), alpha=alpha)
            for i, layer in enumerate(layers)
        ]
        return specs, layers, [None] * k


@torch.inference_mode()
def run_detection(model, tokenizer, k_values, conditions, prompt_styles, layer_pool,
                   alphas, dose_regime, z_totals, num_trials, vec_type, max_new_tokens, seed):
    device = next(model.parameters()).device
    rng = random.Random(seed)
    hidden_dim = infer_hidden_dim()
    # Digit token ids for the bias-adjusted logit contrast (count style only).
    digit_token_ids = {d: tokenizer.encode(str(d), add_special_tokens=False)[0] for d in range(5)}

    # In the individual regime alphas IS the swept dose; in the budget regime
    # z_totals is, and each dose_value is split across k injections below.
    dose_values = alphas if dose_regime == "individual" else z_totals

    trials = []
    total = len(k_values) * len(conditions) * len(prompt_styles) * len(dose_values) * num_trials
    pbar = tqdm(total=total, desc="detection trials", file=sys.stdout)
    trial_counter = 0

    for prompt_style in prompt_styles:
        messages = PROMPT_BUILDERS[prompt_style]()
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(device)

        for condition in conditions:
            for k in k_values:
                for dose_value in dose_values:
                    # Equal-weight budget split (section 14.2): z_i = z_total / sqrt(k)
                    # for k>0; k=0 has no injections so the formula doesn't apply.
                    if dose_regime == "budget":
                        alpha_per_injection = (dose_value / math.sqrt(k)) if k > 0 else 0.0
                    else:
                        alpha_per_injection = dose_value

                    for trial_idx in range(num_trials):
                        trial_counter += 1
                        specs, layers, concepts = build_specs(
                            k, layer_pool, alpha_per_injection, vec_type, condition, rng, hidden_dim,
                            sham_seed_base=seed * 1_000_000 + trial_counter * 10,
                        )

                        with apply_multi_injection(model, specs):
                            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                                  output_scores=True, return_dict_in_generate=True)
                        generated_ids = out.sequences[0][inputs.input_ids.shape[1]:]
                        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                        record = {
                            "prompt_style": prompt_style,
                            "condition": condition,
                            "k_actual": k,
                            "layers": layers,
                            "concepts": concepts,
                            "alpha": alpha_per_injection,
                            "dose_regime": dose_regime,
                            "z_total": dose_value if dose_regime == "budget" else None,
                            "vec_type": vec_type,
                            "trial_idx": trial_idx,
                            "response": response,
                            # min_length=1: both prompt styles demand a single
                            # short token (a digit 0-4, or a 0-10 rating).
                            "is_coherent": is_coherent(response, min_length=1),
                        }
                        if prompt_style == "count":
                            record["reported_count"] = parse_count(response)
                            first_token_logits = out.scores[0][0]
                            logit_true_k = first_token_logits[digit_token_ids[k]].item()
                            logit_default = first_token_logits[digit_token_ids[1]].item()
                            record["logit_true_k"] = logit_true_k
                            record["logit_default"] = logit_default
                            record["logit_contrast_adjusted"] = logit_true_k - logit_default
                        else:
                            record["reported_scale"] = parse_scale_0_10(response)

                        trials.append(record)
                        pbar.update(1)

    pbar.close()
    return trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_values", type=int, nargs="+", default=DEFAULT_K_VALUES)
    parser.add_argument("--conditions", type=str, nargs="+", default=["real", "random"], choices=["real", "random"])
    parser.add_argument("--prompt_styles", type=str, nargs="+", default=["count"], choices=["count", "scale"])
    parser.add_argument("--layer_pool", type=int, nargs="+", default=DEFAULT_LAYER_POOL)
    parser.add_argument("--dose_regime", type=str, default="individual", choices=["individual", "budget"],
                         help="individual: every injection gets an alpha from --alphas. budget: total "
                              "budget from --z_totals split equally, alpha_i = z_total / sqrt(k) for k>0 "
                              "(section 14.2)")
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS,
                         help="Sweep of equal alpha (=z) values for all K injections, --dose_regime individual")
    parser.add_argument("--z_totals", type=float, nargs="+", default=DEFAULT_Z_TOTALS,
                         help="Sweep of total dose budgets, --dose_regime budget; ignored otherwise")
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

    print(f"\n{'='*60}\nEXPERIENCE 9 -- COUNTING\n{'='*60}", flush=True)
    print(f"K values: {args.k_values}", flush=True)
    print(f"Conditions: {args.conditions}", flush=True)
    print(f"Prompt styles: {args.prompt_styles}", flush=True)
    print(f"Dose regime: {args.dose_regime}", flush=True)
    print(f"Alphas: {args.alphas}", flush=True)
    print(f"Z_totals: {args.z_totals}", flush=True)
    print(f"Trials per cell: {args.num_trials}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = run_detection(
        model, tokenizer, args.k_values, args.conditions, args.prompt_styles,
        args.layer_pool, args.alphas, args.dose_regime, args.z_totals,
        args.num_trials, args.vec_type, args.max_new_tokens, args.seed,
    )

    output_path = output_dir / f"multi_detection_trials_{args.dose_regime}.pt"
    torch.save({
        "trials": trials,
        "k_values": args.k_values,
        "conditions": args.conditions,
        "prompt_styles": args.prompt_styles,
        "dose_regime": args.dose_regime,
        "alphas": args.alphas,
        "z_totals": args.z_totals,
        "vec_type": args.vec_type,
        "num_trials": args.num_trials,
        "seed": args.seed,
    }, output_path)
    print(f"\nSaved {len(trials)} trials to {output_path}", flush=True)


if __name__ == "__main__":
    main()
