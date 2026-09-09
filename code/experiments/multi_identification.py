#!/usr/bin/env python3
"""
Block 2 - Identification (H2)

E2 - Free identification: 2 injections at different layers, equal alpha.
Grading: embedding similarity between the free-response answer and each
injected concept's canonical description (pre-registered threshold applied
downstream in code/analysis/identification_accuracy.py).

Controls:
  C2.1 - Single concept (n_injections=1): upper bound on per-concept
         detectability, ties back to Block 0.
  C2.2 - Forced choice (mode=forced_choice): numbered candidate list built from
         concepts.py, order randomized per trial; exact-match accuracy against
         ground truth (scored downstream).
  C2.3 - Sham identification (sham_slots=1, requires n_injections=2): one real
         concept + one sham injection (sham_vectors.make_random_direction);
         tests whether the model falsely "identifies" content in the sham slot.

--run_all_conditions runs C2.1, E2, C2.2, C2.3 in one process, sharing a single
model load (loading the 8B model four times separately would be wasteful).
"""

import argparse
import random
import string
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from all_prompts import get_forced_choice_identification_messages, get_identification_messages
from concepts import ALL_CONCEPTS, CONCEPT_DESCRIPTIONS, get_concept_description
from embedding_judge import best_match, cosine_similarity
from multi_inject import InjectionSpec, apply_multi_injection
from response_parsing import is_coherent, parse_letter_choices
from sham_vectors import infer_hidden_dim, make_random_direction

DEFAULT_LAYER_POOL = list(range(0, 32, 2))
CANDIDATE_LETTERS = string.ascii_uppercase[:len(ALL_CONCEPTS)]  # A..J for 10 concepts

CANONICAL_CONDITIONS = [
    {"name": "c2_1_single", "n_injections": 1, "sham_slots": 0, "mode": "free"},
    {"name": "e2_free", "n_injections": 2, "sham_slots": 0, "mode": "free"},
    {"name": "c2_2_forced_choice", "n_injections": 2, "sham_slots": 0, "mode": "forced_choice"},
    {"name": "c2_3_sham", "n_injections": 2, "sham_slots": 1, "mode": "free"},
]


def load_vector(concept, layer, vec_type="avg"):
    """Load a concept vector for a specific layer."""
    vector_path = Path(f"saved_vectors/llama/{concept}_{layer}_{vec_type}.pt")
    if not vector_path.exists():
        raise FileNotFoundError(f"Vector not found: {vector_path}")
    data = torch.load(vector_path, weights_only=False)
    return data["vector"]


def build_injection(n_injections, sham_slots, layer_pool, alpha, vec_type, rng, hidden_dim, sham_seed_base):
    """Sample n_injections distinct layers; sham_slots of them get a random
    direction (sham_vectors), the rest get real, distinct concept vectors.
    Returns (specs, layers, concepts) where concepts[i] is None for sham slots.
    """
    layers = rng.sample(layer_pool, n_injections)
    real_count = n_injections - sham_slots
    real_concepts = rng.sample(ALL_CONCEPTS, real_count)
    slot_is_sham = [False] * real_count + [True] * sham_slots
    rng.shuffle(slot_is_sham)

    specs, concepts = [], []
    real_iter = iter(real_concepts)
    for i, (layer, is_sham) in enumerate(zip(layers, slot_is_sham)):
        if is_sham:
            vector = make_random_direction(hidden_dim, sham_seed_base + i)
            concepts.append(None)
        else:
            concept = next(real_iter)
            vector = load_vector(concept, layer, vec_type)
            concepts.append(concept)
        specs.append(InjectionSpec(layer=layer, vector=vector, alpha=alpha))
    return specs, layers, concepts


def build_candidate_list(injected_concepts, rng):
    """Randomized-order letter->concept mapping covering all ALL_CONCEPTS,
    guaranteeing every injected (non-None) concept has a letter."""
    pool = list(ALL_CONCEPTS)
    rng.shuffle(pool)
    letter_to_concept = dict(zip(CANDIDATE_LETTERS, pool))
    concept_to_letter = {c: letter for letter, c in letter_to_concept.items()}
    candidate_lines = [f"{letter}. {c}" for letter, c in letter_to_concept.items()]
    correct_letters = [concept_to_letter[c] for c in injected_concepts if c is not None]
    return candidate_lines, letter_to_concept, correct_letters


@torch.inference_mode()
def run_identification(model, tokenizer, n_injections, sham_slots, mode, layer_pool,
                        alpha, num_trials, vec_type, max_new_tokens, seed):
    device = next(model.parameters()).device
    rng = random.Random(seed)
    hidden_dim = infer_hidden_dim()

    trials = []
    pbar = tqdm(total=num_trials, desc="identification trials", file=sys.stdout)

    for trial_idx in range(num_trials):
        specs, layers, concepts = build_injection(
            n_injections, sham_slots, layer_pool, alpha, vec_type, rng, hidden_dim,
            sham_seed_base=seed * 1_000_000 + trial_idx * 10,
        )

        if mode == "forced_choice":
            candidate_lines, letter_to_concept, correct_letters = build_candidate_list(concepts, rng)
            messages = get_forced_choice_identification_messages(candidate_lines, n_injections)
        else:
            messages = get_identification_messages(n_injections)

        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(device)

        with apply_multi_injection(model, specs):
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated_ids = out[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        record = {
            "trial_idx": trial_idx,
            "mode": mode,
            "n_injections": n_injections,
            "sham_slots": sham_slots,
            "layers": layers,
            "concepts": concepts,
            "alpha": alpha,
            "vec_type": vec_type,
            "response": response,
            "is_coherent": is_coherent(response),
        }

        if mode == "forced_choice":
            record["letter_to_concept"] = letter_to_concept
            record["correct_letters"] = correct_letters
            record["reported_letters"] = parse_letter_choices(response, max_choices=n_injections)
        else:
            real_concepts = [c for c in concepts if c is not None]
            record["sim_to_injected"] = [
                cosine_similarity(response, get_concept_description(c)) for c in real_concepts
            ] if response else [0.0] * len(real_concepts)

            distractor_pool = {c: d for c, d in CONCEPT_DESCRIPTIONS.items() if c not in real_concepts}
            if distractor_pool and response:
                best_concept, best_score = best_match(response, distractor_pool)
                record["best_distractor_match"] = best_concept
                record["best_distractor_similarity"] = best_score

            if sham_slots > 0 and response:
                best_concept, best_score = best_match(response, CONCEPT_DESCRIPTIONS)
                record["sham_best_match"] = best_concept
                record["sham_best_match_similarity"] = best_score

        trials.append(record)
        pbar.update(1)

    pbar.close()
    return trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_all_conditions", action="store_true",
                         help="Run C2.1, E2, C2.2, C2.3 in one process, sharing one model load")
    parser.add_argument("--n_injections", type=int, default=2, choices=[1, 2])
    parser.add_argument("--sham_slots", type=int, default=0, choices=[0, 1])
    parser.add_argument("--mode", type=str, default="free", choices=["free", "forced_choice"])
    parser.add_argument("--layer_pool", type=int, nargs="+", default=DEFAULT_LAYER_POOL)
    parser.add_argument("--alpha", type=float, default=8.0,
                         help="Equal alpha for all injections; pick from Block 0's operating window")
    parser.add_argument("--num_trials", type=int, default=30)
    parser.add_argument("--vec_type", type=str, default="avg", choices=["avg", "last"])
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="plots")
    parser.add_argument("--output_suffix", type=str, default="",
                         help="Ignored when --run_all_conditions is set (per-condition suffix always used)")
    args = parser.parse_args()

    if not args.run_all_conditions and args.sham_slots >= args.n_injections:
        raise ValueError("sham_slots must be < n_injections (need at least one real injection)")

    print("Loading model...", flush=True)
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    print("Model loaded!", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = CANONICAL_CONDITIONS if args.run_all_conditions else [
        {"name": "custom", "n_injections": args.n_injections, "sham_slots": args.sham_slots, "mode": args.mode}
    ]

    for cond in conditions:
        print(f"\n{'='*60}\nBLOCK 2 -- IDENTIFICATION ({cond['name']})\n{'='*60}", flush=True)
        print(f"n_injections={cond['n_injections']}, sham_slots={cond['sham_slots']}, mode={cond['mode']}", flush=True)

        trials = run_identification(
            model, tokenizer, cond["n_injections"], cond["sham_slots"], cond["mode"],
            args.layer_pool, args.alpha, args.num_trials, args.vec_type,
            args.max_new_tokens, args.seed,
        )

        suffix = f"_{cond['name']}" if args.run_all_conditions else (args.output_suffix or f"_{cond['name']}")
        output_path = output_dir / f"multi_identification_trials{suffix}.pt"
        torch.save({
            "trials": trials,
            "n_injections": cond["n_injections"],
            "sham_slots": cond["sham_slots"],
            "mode": cond["mode"],
            "alpha": args.alpha,
            "vec_type": args.vec_type,
            "num_trials": args.num_trials,
            "seed": args.seed,
        }, output_path)
        print(f"Saved {len(trials)} trials to {output_path}", flush=True)


if __name__ == "__main__":
    main()
