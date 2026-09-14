#!/usr/bin/env python3
"""
Experience 11 - relative layer-depth ordering (plan_of_research.md section 16).

Two concepts injected at two distinct layers, equal alpha (alpha plays the
role of the plan's z directly; see multisteering_implementation_plan.md's
scope decision). The prompt (section 16.2) presents the two concepts as
lettered options A/B rather than naming them in running prose, and the model
answers with exactly one letter.

Which concept is labeled A vs. B is an independent randomization from which
one is actually shallower -- --label_assignments runs each trial under both
assignments (shallow_is_a / shallow_is_b) by default, the section 16's
counterbalancing control for label/primacy bias, analogous to the old
presentation-order permutation but now over the A/B label rather than
narration order.

Each (concept_shallow, concept_deep, layer_shallow, layer_deep) draw is
replayed at every alpha in --alphas -- materials are matched across the dose
sweep, only intensity varies for a given trial_idx.

Paired sham baseline: each injected generation is matched with a second,
sham generation on the exact same prompt (same concept names, same A/B
label assignment) but with no injection at all -- the plan's L_adjusted
pattern (section 5.7), applied here to isolate the injection's own
contribution to the logit contrast from whatever baseline A/B letter
preference the model has for that prompt regardless of any injection. The
"adjusted" contrast already controls for the correct-letter position (see
below); logit_contrast_double_adjusted = injected adjusted - sham adjusted
additionally controls for that prompt-level baseline. This doubles the
number of generations per cell.
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
from response_parsing import is_coherent, parse_ab_choice

DEFAULT_LAYER_POOL = list(range(0, 32))
DEFAULT_ALPHAS = [1, 2, 3, 4, 5, 6, 7]


def load_vector(concept, layer, vec_type="avg"):
    """Load a concept vector for a specific layer."""
    vector_path = Path(f"saved_vectors/llama/{concept}_{layer}_{vec_type}.pt")
    if not vector_path.exists():
        raise FileNotFoundError(f"Vector not found: {vector_path}")
    data = torch.load(vector_path, weights_only=False)
    return data["vector"]


def _generate_with_ab_logits(model, tokenizer, device, inputs, max_new_tokens, specs,
                              token_id_a, token_id_b):
    with apply_multi_injection(model, specs):
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                              output_scores=True, return_dict_in_generate=True)
    generated_ids = out.sequences[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    first_token_logits = out.scores[0][0]
    return response, first_token_logits[token_id_a].item(), first_token_logits[token_id_b].item()


@torch.inference_mode()
def run_ordering(model, tokenizer, layer_pool, alphas, num_trials, vec_type,
                  max_new_tokens, seed, label_assignments):
    device = next(model.parameters()).device
    rng = random.Random(seed)
    # Section 16.6's logit-contrast measure: token ids for the bare letters,
    # looked up once since the tokenizer doesn't change across trials.
    token_id_a = tokenizer.encode("A", add_special_tokens=False)[0]
    token_id_b = tokenizer.encode("B", add_special_tokens=False)[0]

    trials = []
    # x2: each cell gets an injected generation and a matched sham generation.
    total = num_trials * len(alphas) * len(label_assignments) * 2
    pbar = tqdm(total=total, desc="ordering trials", file=sys.stdout)

    for trial_idx in range(num_trials):
        concept_shallow, concept_deep = rng.sample(ALL_CONCEPTS, 2)
        layer_shallow, layer_deep = sorted(rng.sample(layer_pool, 2))

        vector_shallow = load_vector(concept_shallow, layer_shallow, vec_type)
        vector_deep = load_vector(concept_deep, layer_deep, vec_type)

        for alpha in alphas:
            specs = [
                InjectionSpec(layer=layer_shallow, vector=vector_shallow, alpha=alpha),
                InjectionSpec(layer=layer_deep, vector=vector_deep, alpha=alpha),
            ]

            for label_assignment in label_assignments:
                if label_assignment == "shallow_is_a":
                    concept_a, concept_b, correct_letter = concept_shallow, concept_deep, "A"
                else:
                    concept_a, concept_b, correct_letter = concept_deep, concept_shallow, "B"

                messages = get_ordering_messages(concept_a, concept_b)
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(device)

                def _adjusted(logit_a, logit_b):
                    logit_correct = logit_a if correct_letter == "A" else logit_b
                    logit_incorrect = logit_b if correct_letter == "A" else logit_a
                    return logit_correct - logit_incorrect

                # Injected generation.
                response, logit_a, logit_b = _generate_with_ab_logits(
                    model, tokenizer, device, inputs, max_new_tokens, specs, token_id_a, token_id_b)
                logit_contrast_adjusted = _adjusted(logit_a, logit_b)
                pbar.update(1)

                # Matched sham: identical prompt, no injection at all --
                # isolates the injection's own contribution from whatever
                # baseline A/B preference this specific prompt already has.
                sham_response, sham_logit_a, sham_logit_b = _generate_with_ab_logits(
                    model, tokenizer, device, inputs, max_new_tokens, [], token_id_a, token_id_b)
                logit_contrast_adjusted_sham = _adjusted(sham_logit_a, sham_logit_b)
                pbar.update(1)

                reported = parse_ab_choice(response)
                trials.append({
                    "trial_idx": trial_idx,
                    "label_assignment": label_assignment,
                    "concept_shallow": concept_shallow,
                    "concept_deep": concept_deep,
                    "layer_shallow": layer_shallow,
                    "layer_deep": layer_deep,
                    "layer_distance": layer_deep - layer_shallow,
                    "concept_a": concept_a,
                    "concept_b": concept_b,
                    "correct_letter": correct_letter,
                    "alpha": alpha,
                    "vec_type": vec_type,
                    "response": response,
                    # min_length=1: the format demands a single letter, A or B.
                    "is_coherent": is_coherent(response, min_length=1),
                    "reported_letter": reported,
                    "correct": (reported == correct_letter) if reported is not None else None,
                    # Section 16.6: logit contrast at the first response token.
                    # "adjusted" flips sign per trial so a positive value
                    # always means evidence toward the correct letter,
                    # canceling out a fixed A/B preference -- computed for
                    # every trial, including incoherent/unparsed ones, since
                    # it doesn't need the parsed text.
                    "logit_a": logit_a,
                    "logit_b": logit_b,
                    "logit_contrast_ab": logit_a - logit_b,
                    "logit_contrast_adjusted": logit_contrast_adjusted,
                    # Paired sham baseline and the doubly-adjusted contrast.
                    "sham_response": sham_response,
                    "sham_logit_a": sham_logit_a,
                    "sham_logit_b": sham_logit_b,
                    "logit_contrast_adjusted_sham": logit_contrast_adjusted_sham,
                    "logit_contrast_double_adjusted": logit_contrast_adjusted - logit_contrast_adjusted_sham,
                })

    pbar.close()
    return trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer_pool", type=int, nargs="+", default=DEFAULT_LAYER_POOL)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS,
                         help="Sweep of equal alpha (=z) values for both injections; each "
                              "(concept, layer) draw is replayed at every alpha (matched materials)")
    parser.add_argument("--num_trials", type=int, default=40)
    parser.add_argument("--vec_type", type=str, default="avg", choices=["avg", "last"])
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label_assignments", type=str, nargs="+",
                         default=["shallow_is_a", "shallow_is_b"],
                         choices=["shallow_is_a", "shallow_is_b"],
                         help="Which concept is labeled A: pass both (default) to run the "
                              "label-counterbalancing control, or just one to skip it")
    parser.add_argument("--output_dir", type=str, default="plots")
    args = parser.parse_args()

    print("Loading model...", flush=True)
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    print("Model loaded!", flush=True)

    print(f"\n{'='*60}\nEXPERIENCE 11 -- LAYER-DEPTH ORDERING\n{'='*60}", flush=True)
    print(f"Alphas: {args.alphas}", flush=True)
    print(f"Trials: {args.num_trials}", flush=True)
    print(f"Label assignments: {args.label_assignments}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = run_ordering(
        model, tokenizer, args.layer_pool, args.alphas, args.num_trials,
        args.vec_type, args.max_new_tokens, args.seed, args.label_assignments,
    )

    output_path = output_dir / "layer_ordering_trials.pt"
    torch.save({
        "trials": trials,
        "alphas": args.alphas,
        "vec_type": args.vec_type,
        "num_trials": args.num_trials,
        "label_assignments": args.label_assignments,
        "seed": args.seed,
    }, output_path)
    print(f"\nSaved {len(trials)} trials to {output_path}", flush=True)


if __name__ == "__main__":
    main()
