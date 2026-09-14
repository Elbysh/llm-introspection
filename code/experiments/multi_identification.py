#!/usr/bin/env python3
"""
Experience 10 - identifying the content of two injected concepts
(plan_of_research.md section 15).

Four conditions (section 15.2), each sharing the same ambiguous prompt
framing so the model is never told how many of the "two slots" actually
carry an injected concept:
  single_concept       - 1 real concept, 1 empty slot.
  two_concepts         - 2 real concepts.
  concept_plus_random  - 1 real concept, 1 random-direction injection
                          (sham_vectors.make_random_direction).
  sham                 - 0 injections at all (the plan's actual sham).

Every condition is evaluated in both response formats, forced choice first
then free response on a second, independent generation from the same
injection (section 15.4: the candidate list must not contaminate the free
response). Forced choice (section 15.3) lets the model answer NONE for a
slot with no identifiable conceptual content -- correct for both the "empty"
slots (single_concept, sham) and the random-direction slot
(concept_plus_random), since neither carries identifiable content.

alpha plays the role of the plan's z directly (no section-4 natural-scale
calibration in this pass; see multisteering_implementation_plan.md). Each
item (a sampled set of injection slots) is replayed at every alpha in
--alphas -- materials are matched across the dose sweep, so only intensity
varies for a given item_idx.

--run_all_conditions runs all four conditions in one process, sharing a
single model load.
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
from response_parsing import is_coherent, parse_two_label_choice
from sham_vectors import infer_hidden_dim, make_random_direction

DEFAULT_LAYER_POOL = list(range(0, 32))
DEFAULT_ALPHAS = [1, 2, 3, 4, 5, 6, 7]
CANDIDATE_LETTERS = string.ascii_uppercase[:len(ALL_CONCEPTS)]  # A..J for 10 concepts
NONE_LABEL = "NONE"
N_SLOTS = 2  # the forced-choice task always asks about exactly two slots

CANONICAL_CONDITIONS = [
    {"name": "single_concept", "n_real": 1, "n_random": 0},
    {"name": "two_concepts", "n_real": 2, "n_random": 0},
    {"name": "concept_plus_random", "n_real": 1, "n_random": 1},
    {"name": "sham", "n_real": 0, "n_random": 0},
]


def load_vector(concept, layer, vec_type="avg"):
    """Load a concept vector for a specific layer."""
    vector_path = Path(f"saved_vectors/llama/{concept}_{layer}_{vec_type}.pt")
    if not vector_path.exists():
        raise FileNotFoundError(f"Vector not found: {vector_path}")
    data = torch.load(vector_path, weights_only=False)
    return data["vector"]


def sample_slots(n_real, n_random, layer_pool, rng):
    """Sample (n_real + n_random) distinct layers and assign each a real,
    distinct concept or a random-direction marker (None). Returns
    (layers, concepts) with concepts[i] None for random slots.
    n_real=n_random=0 (the sham condition) returns everything empty.
    Independent of alpha, so the same draw can be replayed across an alpha
    sweep for matched materials."""
    total = n_real + n_random
    layers = rng.sample(layer_pool, total) if total > 0 else []
    real_concepts = rng.sample(ALL_CONCEPTS, n_real) if n_real > 0 else []
    slot_is_random = [False] * n_real + [True] * n_random
    rng.shuffle(slot_is_random)

    concepts = []
    real_iter = iter(real_concepts)
    for is_random in slot_is_random:
        concepts.append(None if is_random else next(real_iter))
    return layers, concepts


def build_specs_from_slots(layers, concepts, alpha, vec_type, hidden_dim, sham_seed_base):
    """Builds InjectionSpecs at the given alpha from a (layers, concepts)
    slot assignment produced by sample_slots -- concepts[i] is None for a
    random-direction slot (sham_vectors)."""
    specs = []
    for i, (layer, concept) in enumerate(zip(layers, concepts)):
        if concept is None:
            vector = make_random_direction(hidden_dim, sham_seed_base + i)
        else:
            vector = load_vector(concept, layer, vec_type)
        specs.append(InjectionSpec(layer=layer, vector=vector, alpha=alpha))
    return specs


def build_candidate_list(real_concepts, n_none_slots, rng):
    """Randomized-order letter->concept mapping covering all ALL_CONCEPTS
    (guaranteeing every injected concept has a letter), plus a literal NONE
    line for slots with no identifiable conceptual content (section 15.3).
    correct_labels has exactly N_SLOTS entries: one letter per real concept,
    NONE for the rest."""
    pool = list(ALL_CONCEPTS)
    rng.shuffle(pool)
    letter_to_concept = dict(zip(CANDIDATE_LETTERS, pool))
    concept_to_letter = {c: letter for letter, c in letter_to_concept.items()}
    candidate_lines = [f"{letter}. {c}" for letter, c in letter_to_concept.items()]
    candidate_lines.append(f"{NONE_LABEL}. no identifiable conceptual content")
    correct_labels = sorted([concept_to_letter[c] for c in real_concepts] + [NONE_LABEL] * n_none_slots)
    return candidate_lines, letter_to_concept, correct_labels


def _generate(model, tokenizer, messages, max_new_tokens, specs):
    device = next(model.parameters()).device
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with apply_multi_injection(model, specs):
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


@torch.inference_mode()
def run_identification(model, tokenizer, n_real, n_random, layer_pool, alphas,
                        num_trials, vec_type, max_new_tokens, seed):
    """Runs num_trials items (one sampled set of real/random injection
    slots each), each replayed at every alpha in alphas -- materials are
    matched across the dose sweep, only intensity varies for a given
    item_idx. Every (item, alpha) pair goes through both response formats --
    forced choice first, then free response on a second, independent
    generation from the same injection -- yielding two trial records that
    share an item_idx."""
    rng = random.Random(seed)
    hidden_dim = infer_hidden_dim()

    trials = []
    pbar = tqdm(total=num_trials * len(alphas) * 2, desc="identification trials", file=sys.stdout)

    for item_idx in range(num_trials):
        layers, concepts = sample_slots(n_real, n_random, layer_pool, rng)
        real_concepts = [c for c in concepts if c is not None]
        n_none_slots = N_SLOTS - n_real

        for alpha in alphas:
            specs = build_specs_from_slots(
                layers, concepts, alpha, vec_type, hidden_dim,
                sham_seed_base=seed * 1_000_000 + item_idx * 10,
            )

            base_record = {
                "item_idx": item_idx,
                "n_real": n_real,
                "n_random": n_random,
                "layers": layers,
                "concepts": concepts,
                "alpha": alpha,
                "vec_type": vec_type,
            }

            # --- forced choice, first (section 15.4 ordering) ---
            candidate_lines, letter_to_concept, correct_labels = build_candidate_list(
                real_concepts, n_none_slots, rng)
            response = _generate(model, tokenizer, get_forced_choice_identification_messages(candidate_lines),
                                  max_new_tokens, specs)
            trials.append({
                **base_record,
                "mode": "forced_choice",
                "letter_to_concept": letter_to_concept,
                "correct_labels": correct_labels,
                "response": response,
                # min_length=1: unlike free response, a valid forced-choice
                # answer can be as short as two labels (e.g. "A B").
                "is_coherent": is_coherent(response, min_length=1),
                "reported_labels": parse_two_label_choice(response, list(CANDIDATE_LETTERS)),
            })
            pbar.update(1)

            # --- free response, second, independent generation ---
            response = _generate(model, tokenizer, get_identification_messages(), max_new_tokens, specs)
            free_record = {
                **base_record,
                "mode": "free",
                "response": response,
                "is_coherent": is_coherent(response),
            }
            if response:
                free_record["sim_to_injected"] = [
                    cosine_similarity(response, get_concept_description(c)) for c in real_concepts
                ]
                distractor_pool = {c: d for c, d in CONCEPT_DESCRIPTIONS.items() if c not in real_concepts}
                if distractor_pool:
                    best_concept, best_score = best_match(response, distractor_pool)
                    free_record["best_distractor_match"] = best_concept
                    free_record["best_distractor_similarity"] = best_score
                if n_none_slots > 0:
                    best_concept, best_score = best_match(response, CONCEPT_DESCRIPTIONS)
                    free_record["none_slot_best_match"] = best_concept
                    free_record["none_slot_best_match_similarity"] = best_score
            else:
                free_record["sim_to_injected"] = [0.0] * len(real_concepts)
            trials.append(free_record)
            pbar.update(1)

    pbar.close()
    return trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_all_conditions", action="store_true",
                         help="Run all four Exp 10 conditions in one process, sharing one model load")
    parser.add_argument("--n_real", type=int, default=2, choices=[0, 1, 2],
                         help="Number of real concept injections (custom single-condition run)")
    parser.add_argument("--n_random", type=int, default=0, choices=[0, 1, 2],
                         help="Number of random-direction injections (custom single-condition run)")
    parser.add_argument("--condition_name", type=str, default="custom",
                         help="Output filename tag for a custom single-condition run (ignored with "
                              "--run_all_conditions); set to e.g. 'two_concepts' when running one of "
                              "the four canonical conditions as its own job, so the output matches "
                              "what --run_all_conditions would have produced")
    parser.add_argument("--layer_pool", type=int, nargs="+", default=DEFAULT_LAYER_POOL)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS,
                         help="Sweep of equal alpha (=z) values for every injection in a trial; "
                              "each item is replayed at every alpha (matched materials)")
    parser.add_argument("--num_trials", type=int, default=30)
    parser.add_argument("--vec_type", type=str, default="avg", choices=["avg", "last"])
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="plots")
    args = parser.parse_args()

    if not args.run_all_conditions and args.n_real + args.n_random > N_SLOTS:
        raise ValueError(f"n_real + n_random must be <= {N_SLOTS}")

    print("Loading model...", flush=True)
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    print("Model loaded!", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = CANONICAL_CONDITIONS if args.run_all_conditions else [
        {"name": args.condition_name, "n_real": args.n_real, "n_random": args.n_random}
    ]

    for cond in conditions:
        print(f"\n{'='*60}\nEXPERIENCE 10 -- IDENTIFICATION ({cond['name']})\n{'='*60}", flush=True)
        print(f"n_real={cond['n_real']}, n_random={cond['n_random']}", flush=True)

        trials = run_identification(
            model, tokenizer, cond["n_real"], cond["n_random"],
            args.layer_pool, args.alphas, args.num_trials, args.vec_type,
            args.max_new_tokens, args.seed,
        )

        output_path = output_dir / f"multi_identification_trials_{cond['name']}.pt"
        torch.save({
            "trials": trials,
            "n_real": cond["n_real"],
            "n_random": cond["n_random"],
            "alphas": args.alphas,
            "vec_type": args.vec_type,
            "num_trials": args.num_trials,
            "seed": args.seed,
        }, output_path)
        print(f"Saved {len(trials)} trials to {output_path}", flush=True)


if __name__ == "__main__":
    main()
