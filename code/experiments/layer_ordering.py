#!/usr/bin/env python3
"""
Block 3 - Relative Layer-Depth Ordering (H3)

E3 - Two-alternative forced choice: same setup as E2 (2 injections, different
layers, equal alpha). Both injected concept names are revealed directly in the
prompt -- the model is asked purely about relative processing depth (which
entered "first", i.e. at the shallower layer), decoupling this from H2's
identification difficulty.

Controls:
  C3.1 - Presentation order permutation: --presentation_orders runs each trial
         with concept_1/concept_2 naming swapped in the prompt text,
         independent of which concept actually sits at the shallower layer, to
         detect a primacy/recency bias in the response (scored downstream).
  C3.2 - Chance baseline: code/analysis/ordering_accuracy.py runs
         scipy.stats.binomtest against 50%.
"""

import argparse
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from all_prompts import get_ordering_messages
from concepts import ALL_CONCEPTS
from multi_inject import InjectionSpec, apply_multi_injection
from response_parsing import is_coherent, parse_concept_choice

DEFAULT_LAYER_POOL = list(range(0, 32, 2))


def load_vector(concept, layer, vec_type="avg"):
    """Load a concept vector for a specific layer."""
    vector_path = Path(f"saved_vectors/llama/{concept}_{layer}_{vec_type}.pt")
    if not vector_path.exists():
        raise FileNotFoundError(f"Vector not found: {vector_path}")
    data = torch.load(vector_path, weights_only=False)
    return data["vector"]


@torch.inference_mode()
def run_ordering(model, tokenizer, layer_pool, alpha, num_trials, vec_type,
                  max_new_tokens, seed, presentation_orders):
    device = next(model.parameters()).device
    rng = random.Random(seed)

    trials = []
    total = num_trials * len(presentation_orders)
    pbar = tqdm(total=total, desc="ordering trials", file=sys.stdout)

    for trial_idx in range(num_trials):
        concept_shallow, concept_deep = rng.sample(ALL_CONCEPTS, 2)
        layer_shallow, layer_deep = sorted(rng.sample(layer_pool, 2))

        vector_shallow = load_vector(concept_shallow, layer_shallow, vec_type)
        vector_deep = load_vector(concept_deep, layer_deep, vec_type)
        specs = [
            InjectionSpec(layer=layer_shallow, vector=vector_shallow, alpha=alpha),
            InjectionSpec(layer=layer_deep, vector=vector_deep, alpha=alpha),
        ]

        for presentation_order in presentation_orders:
            if presentation_order == "shallow_first":
                named_concept_1, named_concept_2 = concept_shallow, concept_deep
            else:
                named_concept_1, named_concept_2 = concept_deep, concept_shallow

            messages = get_ordering_messages(named_concept_1, named_concept_2)
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(device)

            with apply_multi_injection(model, specs):
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generated_ids = out[0][inputs.input_ids.shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            reported = parse_concept_choice(response, named_concept_1, named_concept_2)
            trials.append({
                "trial_idx": trial_idx,
                "presentation_order": presentation_order,
                "concept_shallow": concept_shallow,
                "concept_deep": concept_deep,
                "layer_shallow": layer_shallow,
                "layer_deep": layer_deep,
                "layer_distance": layer_deep - layer_shallow,
                "named_concept_1": named_concept_1,
                "named_concept_2": named_concept_2,
                "alpha": alpha,
                "vec_type": vec_type,
                "response": response,
                "is_coherent": is_coherent(response),
                "reported_concept": reported,
                "correct": (reported == concept_shallow) if reported is not None else None,
            })
            pbar.update(1)

    pbar.close()
    return trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer_pool", type=int, nargs="+", default=DEFAULT_LAYER_POOL)
    parser.add_argument("--alpha", type=float, default=8.0,
                         help="Equal alpha for both injections; pick from Block 0's operating window")
    parser.add_argument("--num_trials", type=int, default=40)
    parser.add_argument("--vec_type", type=str, default="avg", choices=["avg", "last"])
    parser.add_argument("--max_new_tokens", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--presentation_orders", type=str, nargs="+",
                         default=["shallow_first", "deep_first"],
                         choices=["shallow_first", "deep_first"],
                         help="C3.1: pass both (default) to run the presentation-order control, "
                              "or just one to skip it")
    parser.add_argument("--output_dir", type=str, default="plots")
    args = parser.parse_args()

    print("Loading model...", flush=True)
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    print("Model loaded!", flush=True)

    print(f"\n{'='*60}\nBLOCK 3 -- LAYER-DEPTH ORDERING (E3 + C3.1-C3.2)\n{'='*60}", flush=True)
    print(f"Alpha: {args.alpha}", flush=True)
    print(f"Trials: {args.num_trials}", flush=True)
    print(f"Presentation orders: {args.presentation_orders}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = run_ordering(
        model, tokenizer, args.layer_pool, args.alpha, args.num_trials,
        args.vec_type, args.max_new_tokens, args.seed, args.presentation_orders,
    )

    output_path = output_dir / "layer_ordering_trials.pt"
    torch.save({
        "trials": trials,
        "alpha": args.alpha,
        "vec_type": args.vec_type,
        "num_trials": args.num_trials,
        "presentation_orders": args.presentation_orders,
        "seed": args.seed,
    }, output_path)
    print(f"\nSaved {len(trials)} trials to {output_path}", flush=True)


if __name__ == "__main__":
    main()
