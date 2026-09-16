#!/usr/bin/env python3
"""
Experiment 4: detection and task degradation.

Implements section 8 of docs/livrables/cadrage-experiments.md. Experiment 1 asks the
model to localize an intervention; this one asks nothing about the intervention at
all. It asks the model to classify a sentence while the same perturbation runs, so
that a detection curve can be read against the dose range where the model still does
semantic work (H4, doc 8.1).

  task      binary semantic classification over data/dataset/complex_data.json
  prompt    "Does this sentence express the concept {C}?", X/Y answer (doc 8.3)
  mapping   run in both letter assignments, so a letter preference is separable
  families  sham, concept, random, noise, dropout (+ scrambled when calibrated)
  site      the same decoder-block output, the same token span, one layer per trial

The evaluated concept is never the injected concept (doc 8.3 and 8.5 step 2). That
single rule buys two things: a targeted injection cannot push the model straight to
the expected answer, and the construction/evaluation separation of doc 3.3 holds for
free, since the sentences scored here belong to a concept whose corpus built no part
of the direction being injected.

Doses, layers and families are literally Experiment 1's: the grids, the dose-to-
injection conversion and the tie convention are imported from
experiments.experiment1_psychometrics rather than restated, because doc 8.4 requires
"les mêmes couches et les mêmes valeurs de alpha ou de z que dans l'expérience 1" and
a second copy of those rules would be a second thing to keep in step.

    python code/experiments/experiment4_task_degradation.py --dry_run

With an Experiment 1 run at hand, the two are joined cell by cell on
(layer, family, matching, dose), which is step 7 of doc 8.5 and the analysis of 8.7:

    python code/experiments/experiment4_task_degradation.py \
        --calibration_config configs/experiment_0_calibration/development_full.yaml \
        --allow_calibration_mismatch \
        --experiment1_summary results/experiment1/pilot

The two tasks use different prompts and different sentences on purpose. What is
compared is the intervention condition, never a single response.
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT / "code", REPO_ROOT / "code" / "utils"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from save_random_vectors import derive_seed

from experiment_0_calibration.direction_bank import (
    DEFAULT_CALIBRATION_DIR,
    ESTIMATOR_FIELDS,
    DirectionBank,
)
from experiments.experiment1_psychometrics import (
    DEFAULT_ALPHA_DOSES,
    DEFAULT_Z_DOSES,
    FAMILIES,
    FIXED_DIRECTION_FAMILIES,
    MATCHINGS,
    intervention,
    make_norm_capture_hook,
    parse_doses,
    perturbation_kwargs,
)

# Only complex_data carries labelled positive and negative examples, so only it can
# supply a classification task (doc 8.3).
COMPLEX_DATA = REPO_ROOT / "data" / "dataset" / "complex_data.json"

# Families that need a concept to name a direction. Both are paired with an evaluated
# concept different from their own.
CONCEPT_FAMILIES = ("concept", "scrambled")

# Bin edges of the performance-matched comparison of doc 8.7, on the accuracy drop
# relative to the sham. Fixed here so the binning is part of the protocol and not a
# choice made once the numbers are in.
PERFORMANCE_BINS = (-1.0, -0.30, -0.15, -0.05, 0.05, 1.0)


# --------------------------------------------------------------------------------
# Task material
# --------------------------------------------------------------------------------

def load_complex_concepts(path=COMPLEX_DATA):
    """{concept: (positive sentences, negative sentences)} from the repository corpus."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    concepts = {}
    for name, groups in payload.items():
        if len(groups) != 2:
            raise ValueError(f"concept {name!r} does not have exactly two example lists")
        concepts[name] = (list(groups[0]), list(groups[1]))
    return concepts


def concept_label(name):
    """Readable form of a corpus key, for the prompt: fibonacci_numbers -> fibonacci numbers."""
    return name.replace("_", " ")


def build_items(corpus, eval_concepts, num_per_class, mappings, seed):
    """The evaluation set: equal numbers of positive and negative sentences (doc 8.4).

    Sentences are drawn once per concept with a recorded seed and reused across every
    layer, family, matching and dose, so a cell difference is never a difference of
    material. Both letter mappings carry the same sentence.
    """
    items = []
    for concept in eval_concepts:
        if concept not in corpus:
            raise ValueError(
                f"{concept!r} is not in {COMPLEX_DATA.name}; available: "
                f"{sorted(corpus)}")
        for label, sentences in enumerate(corpus[concept]):
            # label 0 is the positive list, label 1 the negative one, in corpus order.
            positive = label == 0
            if num_per_class > len(sentences):
                raise ValueError(
                    f"asked for {num_per_class} {'positive' if positive else 'negative'} "
                    f"examples of {concept!r} but only {len(sentences)} exist")
            rng = random.Random(derive_seed(seed, "items", concept, label))
            chosen = sorted(rng.sample(range(len(sentences)), num_per_class))
            for sentence_index in chosen:
                for mapping in mappings:
                    items.append({
                        "item_id": len(items),
                        "eval_concept": concept,
                        "sentence_index": sentence_index,
                        "sentence": sentences[sentence_index],
                        "positive": positive,
                        "mapping": mapping,
                        "correct_letter": correct_letter(positive, mapping),
                    })
    return items


def correct_letter(positive, mapping):
    """The letter the labelled answer maps to under this mapping.

    Mapping "XY" reads X as yes and Y as no; "YX" is its inversion (doc 8.3).
    """
    if mapping not in ("XY", "YX"):
        raise ValueError(f"unknown mapping {mapping!r}")
    says_yes = "X" if mapping == "XY" else "Y"
    says_no = "Y" if mapping == "XY" else "X"
    return says_yes if positive else says_no


# --------------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------------

# The assistant turn the model continues. build_classification_prompt cuts the
# rendered chat template back to the end of this string, exactly as Experiment 1
# does, so that the next token really is the answer letter.
ASSISTANT_PREFIX = "The answer is"


def get_classification_messages(sentence, probed_concept, mapping):
    """The classification prompt of doc 8.3."""
    yes_letter = "X" if mapping == "XY" else "Y"
    no_letter = "Y" if mapping == "XY" else "X"
    return [
        {
            "role": "user",
            "content": (
                "Read the following sentence.\n\n"
                f"{sentence}\n\n"
                f"Does this sentence express the concept {concept_label(probed_concept)}?\n"
                f"{yes_letter} means yes.\n"
                f"{no_letter} means no.\n"
                "Answer with exactly one letter: X or Y."
            ),
        },
        {"role": "assistant", "content": ASSISTANT_PREFIX},
    ]


def build_classification_prompt(tokenizer, sentence, probed_concept, mapping):
    """Return the prompt, the token range covering the sentence, and the encoding."""
    messages = get_classification_messages(sentence, probed_concept, mapping)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    cut = prompt.rfind(ASSISTANT_PREFIX)
    if cut == -1:
        raise ValueError(
            f"assistant prefix {ASSISTANT_PREFIX!r} is absent from the rendered prompt")
    prompt = prompt[: cut + len(ASSISTANT_PREFIX)].rstrip(" ")

    encoding_with_offsets = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True
    )
    offsets = encoding_with_offsets["offset_mapping"][0]
    encoding = {k: v for k, v in encoding_with_offsets.items() if k != "offset_mapping"}

    start_char = prompt.find(sentence)
    if start_char == -1:
        raise ValueError(f"sentence not found in prompt: {sentence!r}")
    end_char = start_char + len(sentence)

    # Every token overlapping the sentence, as in Experiment 1: a sentence-initial
    # token can carry the preceding newline and a sentence-final one the newline that
    # follows, and doc 3.7 targets every token of the sentence.
    token_start = token_end = None
    for index in range(len(offsets)):
        tok_start = offsets[index][0].item()
        tok_end = offsets[index][1].item()
        if token_start is None and tok_end > start_char:
            token_start = index
        if tok_start < end_char:
            token_end = index + 1
    if token_start is None or token_end is None or token_start >= token_end:
        raise ValueError(f"empty token range for sentence {sentence!r}")

    return prompt, (token_start, token_end), encoding


def letter_token_ids(tokenizer, letters=("X", "Y")):
    """Ids of the answer tokens actually produced after "The answer is"."""
    ids = {}
    for letter in letters:
        spaced = tokenizer.encode(f" {letter}", add_special_tokens=False)
        bare = tokenizer.encode(letter, add_special_tokens=False)
        ids[letter] = (spaced if len(spaced) == 1 else bare)[-1]
    if len(set(ids.values())) != len(letters):
        raise ValueError("the answer letters do not map to distinct token ids")
    return ids


# --------------------------------------------------------------------------------
# Pairing of the evaluated and injected concepts
# --------------------------------------------------------------------------------

def parse_concept_pairs(values):
    """--concept_pairs betrayal:Dust shutdown:Satellites -> {betrayal: [Dust], ...}."""
    pairs = defaultdict(list)
    for value in values:
        if ":" not in value:
            raise ValueError(
                f"expected EVALUATED:INJECTED, got {value!r}")
        evaluated, injected = value.split(":", 1)
        pairs[evaluated.strip()].append(injected.strip())
    return dict(pairs)


def default_concept_pairs(eval_concepts, candidates, num_injected):
    """Give each evaluated concept its own injected concepts, never itself.

    Rotating through the candidate pool rather than always taking the first entry
    keeps the concept family from collapsing onto a single direction when several
    concepts are evaluated.
    """
    pairs = {}
    for position, evaluated in enumerate(eval_concepts):
        pool = [name for name in candidates if name != evaluated]
        if len(pool) < num_injected:
            raise ValueError(
                f"{evaluated!r} needs {num_injected} injected concepts but only "
                f"{len(pool)} calibrated concepts differ from it")
        pairs[evaluated] = [pool[(position + offset) % len(pool)]
                            for offset in range(num_injected)]
    return pairs


def check_concept_pairs(pairs, eval_concepts, calibrated):
    """Doc 8.5 step 2, as a precondition rather than a convention."""
    for evaluated in eval_concepts:
        injected = pairs.get(evaluated)
        if not injected:
            raise ValueError(f"no injected concept paired with {evaluated!r}")
        for name in injected:
            if name == evaluated:
                raise ValueError(
                    f"{evaluated!r} is paired with itself; the evaluated concept must "
                    "differ from the injected one (doc 8.3)")
            if name not in calibrated:
                raise ValueError(
                    f"injected concept {name!r} is not in the calibration bank; "
                    f"calibrated concepts are {sorted(calibrated)}")
    unknown = set(pairs) - set(eval_concepts)
    if unknown:
        raise ValueError(f"--concept_pairs names unevaluated concepts: {sorted(unknown)}")


# --------------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------------

def build_conditions(calibration, args):
    """Enumerate the (family, direction, injected concept) conditions of each layer."""
    conditions = defaultdict(list)
    for layer in args.layers:
        for family in CONCEPT_FAMILIES:
            if family not in args.families:
                continue
            for injected in args.injected_concepts:
                ids = calibration.direction_ids(family, layer, concept=injected)
                if not ids:
                    raise ValueError(
                        f"concept {injected!r} has no {family} direction at layer {layer}"
                        + (". Its calibration must set directions.scrambled_concept.enabled."
                           if family == "scrambled" else ""))
                for direction_id in ids:
                    conditions[layer].append((family, direction_id, injected))
        if "random" in args.families:
            ids = calibration.direction_ids("random", layer)[: args.num_random]
            if not ids:
                raise ValueError(f"no random direction calibrated at layer {layer}")
            for direction_id in ids:
                conditions[layer].append(("random", direction_id, None))
        if "noise" in args.families:
            for index in range(args.num_noise):
                conditions[layer].append(("noise", f"noise_realization_{index:02d}", None))
        if "dropout" in args.families:
            for index in range(args.num_dropout):
                conditions[layer].append(("dropout", f"dropout_realization_{index:02d}", None))
    return conditions


def applicable_items(items, injected_concept, eval_for_injected):
    """Items a condition may run on: never the concept that built its direction."""
    if injected_concept is None:
        return items
    allowed = eval_for_injected.get(injected_concept, set())
    return [item for item in items if item["eval_concept"] in allowed]


def count_forward_passes(conditions, args):
    """Perturbed passes; the shams are one per item and shared by every condition."""
    doses = sum(len(args.dose_grid[matching]) for matching in args.matchings)
    total = 0
    for layer in args.layers:
        for _family, _direction_id, injected in conditions[layer]:
            total += len(applicable_items(args.items, injected, args.eval_for_injected))
    return total * doses, len(args.items)


# --------------------------------------------------------------------------------
# Trial execution
# --------------------------------------------------------------------------------

def score_margin(margin):
    """1 correct, 0 wrong, 0.5 for an exact tie.

    Same convention as Experiment 1's forced choice: in bfloat16 a small perturbation
    can leave both answer logits untouched, and scoring that as an error would pull a
    cell below chance rather than towards it.
    """
    if margin == 0.0:
        return 0.5
    return 1.0 if margin > 0.0 else 0.0


def js_divergence(probabilities_p, probabilities_q):
    """Jensen-Shannon divergence in bits, so the value lives in [0, 1] (doc 8.6)."""
    p = probabilities_p.to(torch.float64)
    q = probabilities_q.to(torch.float64)
    m = 0.5 * (p + q)

    def kl(a):
        mask = a > 0
        return float((a[mask] * (torch.log2(a[mask]) - torch.log2(m[mask]))).sum().item())

    return 0.5 * kl(p) + 0.5 * kl(q)


def read_answer(logits, item, letter_ids):
    """Logits of the correct and incorrect letters, and whether the argmax is a letter."""
    other = "Y" if item["correct_letter"] == "X" else "X"
    logit_correct = float(logits[letter_ids[item["correct_letter"]]].item())
    logit_incorrect = float(logits[letter_ids[other]].item())
    return {
        "logit_correct": logit_correct,
        "logit_incorrect": logit_incorrect,
        "margin": logit_correct - logit_incorrect,
        "top_token": int(torch.argmax(logits).item()),
        "finite": bool(torch.isfinite(logits).all().item()),
    }


@torch.inference_mode()
def run_sham(model, encoding, token_range, item, letter_ids, layers, keep_distribution):
    """One clean forward pass: the reference answer, the clean token norms, and,
    when the JS diagnostic is enabled, the reference output distribution."""
    norms = {}
    handles = []
    try:
        for layer in layers:
            handles.append(model.model.layers[layer].register_forward_hook(
                make_norm_capture_hook(norms, layer, [token_range])))
        logits = model(**encoding).logits[0, -1, :]
    finally:
        for handle in handles:
            handle.remove()
    result = read_answer(logits, item, letter_ids)
    result["token_norms"] = norms
    result["probabilities"] = (
        torch.softmax(logits.to(torch.float32), dim=-1) if keep_distribution else None)
    return result


@torch.inference_mode()
def run_trial(model, encoding, layer, token_range, family, item, letter_ids,
              vector=None, alpha=None, delta=None, rate=None, seed=0, measure=True,
              sham_probabilities=None):
    """One perturbed forward pass: the answer, the realized amplitude, and the JS shift."""
    captured = {} if measure else None
    with intervention(model, layer, token_range, family, vector=vector, alpha=alpha,
                      delta=delta, rate=rate, seed=seed, measure=captured):
        logits = model(**encoding).logits[0, -1, :]

    realized = None
    clean_norm = None
    if captured and "clean" in captured and "perturbed" in captured:
        difference = captured["perturbed"] - captured["clean"]
        realized = float(difference.norm(dim=-1).mean().item())
        clean_norm = float(captured["clean"].pow(2).sum(dim=-1).mean().sqrt().item())

    result = read_answer(logits, item, letter_ids)
    result["realized_amplitude"] = realized
    result["clean_token_norm"] = clean_norm
    # Free: the diagnostic reuses this trial's own logits and the sham's, so it costs
    # a softmax and no extra forward pass.
    result["js_divergence"] = (
        js_divergence(sham_probabilities, torch.softmax(logits.to(torch.float32), dim=-1))
        if sham_probabilities is not None else None)
    return result


def prepare_prompts(tokenizer, items, device):
    """One prompt per item, built once and reused by every condition."""
    prompts = {}
    for item in items:
        prompt, token_range, encoding = build_classification_prompt(
            tokenizer, item["sentence"], item["eval_concept"], item["mapping"])
        prompts[item["item_id"]] = {
            "prompt": prompt,
            "range": token_range,
            "encoding": {k: v.to(device) for k, v in encoding.items()},
            "targeted_text": tokenizer.decode(
                encoding["input_ids"][0][token_range[0]:token_range[1]]),
        }
    return prompts


def run_experiment(model, tokenizer, calibration, args):
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    letter_ids = letter_token_ids(tokenizer)
    print("Answer token ids: " + ", ".join(
        f"{letter}={token} ({tokenizer.convert_ids_to_tokens([token])[0]!r})"
        for letter, token in sorted(letter_ids.items())), flush=True)

    conditions = build_conditions(calibration, args)
    perturbed_passes, _ = count_forward_passes(conditions, args)
    prompts = prepare_prompts(tokenizer, args.items, device)

    sample = prompts[args.items[0]["item_id"]]
    print(f"Prompt tail: {sample['prompt'][-80:]!r}", flush=True)
    print(f"Perturbed span: {sample['targeted_text']!r}", flush=True)

    print(f"Running {len(prompts)} shams...", flush=True)
    shams = {}
    for item in tqdm(args.items, disable=not args.progress):
        prompt = prompts[item["item_id"]]
        shams[item["item_id"]] = run_sham(
            model, prompt["encoding"], prompt["range"], item, letter_ids, args.layers,
            keep_distribution=args.js_per_cell > 0)

    trials = []
    for item in args.items:
        sham = shams[item["item_id"]]
        trials.append(sham_record(item, sham, letter_ids))

    planned = (perturbed_passes if args.limit_trials is None
               else min(perturbed_passes, args.limit_trials))
    progress = tqdm(total=planned, disable=not args.progress)
    executed = 0
    js_taken = defaultdict(int)

    for layer in args.layers:
        vectors = {}
        for family, direction_id, _injected in conditions[layer]:
            if family in FIXED_DIRECTION_FAMILIES:
                vectors[direction_id] = calibration.vector(direction_id).reshape(-1).to(
                    device=device, dtype=model_dtype)

        for family, direction_id, injected in conditions[layer]:
            condition_items = applicable_items(args.items, injected, args.eval_for_injected)
            for matching in args.matchings:
                for dose in args.dose_grid[matching]:
                    cell = (layer, family, direction_id, matching, f"{dose:.6g}")
                    for item in condition_items:
                        if args.limit_trials is not None and executed >= args.limit_trials:
                            progress.close()
                            return trials

                        item_id = item["item_id"]
                        prompt, sham = prompts[item_id], shams[item_id]
                        token_range = prompt["range"]
                        trial_seed = derive_seed(
                            args.seed, layer, family, direction_id, matching,
                            f"{dose:.6g}", item_id)
                        # target_index 0: there is a single sentence in this prompt,
                        # and run_sham recorded its norm under that index.
                        kwargs, alpha, rate = perturbation_kwargs(
                            calibration, args, family, direction_id, matching, dose,
                            layer, token_range, sham, 0, vectors, trial_seed)
                        if "delta" in kwargs:
                            kwargs["delta"] = kwargs["delta"].to(device)

                        take_js = js_taken[cell] < args.js_per_cell
                        result = run_trial(
                            model, prompt["encoding"], layer, token_range, family, item,
                            letter_ids, measure=args.measure_amplitude,
                            sham_probabilities=sham["probabilities"] if take_js else None,
                            **kwargs)
                        if take_js:
                            js_taken[cell] += 1

                        trials.append(perturbed_record(
                            item, sham, result, letter_ids, layer=layer, family=family,
                            direction_id=direction_id, injected_concept=injected,
                            matching=matching, dose=dose, alpha=alpha, rate=rate))
                        executed += 1
                        progress.update(1)

    progress.close()
    return trials


def sham_record(item, sham, letter_ids):
    margin = sham["margin"]
    return {
        "kind": "sham", "layer": None, "family": "sham", "direction_id": None,
        "injected_concept": None, "matching": None, "dose": None,
        "alpha_requested": 0.0, "dropout_rate": None,
        "item_id": item["item_id"], "eval_concept": item["eval_concept"],
        "sentence_index": item["sentence_index"], "positive": item["positive"],
        "mapping": item["mapping"], "correct_letter": item["correct_letter"],
        "logit_correct": sham["logit_correct"], "logit_incorrect": sham["logit_incorrect"],
        "margin": margin, "sham_margin": margin, "margin_delta": 0.0,
        "correct": score_margin(margin), "sham_correct": score_margin(margin),
        "top_token_is_letter": sham["top_token"] in letter_ids.values(),
        "sham_top_token_is_letter": sham["top_token"] in letter_ids.values(),
        "finite": sham["finite"], "realized_amplitude": 0.0, "clean_token_norm": None,
        "js_divergence": None,
    }


def perturbed_record(item, sham, result, letter_ids, layer, family, direction_id,
                     injected_concept, matching, dose, alpha, rate):
    return {
        "kind": "perturbed", "layer": layer, "family": family,
        "direction_id": direction_id, "injected_concept": injected_concept,
        "matching": matching, "dose": float(dose), "alpha_requested": alpha,
        "dropout_rate": rate,
        "item_id": item["item_id"], "eval_concept": item["eval_concept"],
        "sentence_index": item["sentence_index"], "positive": item["positive"],
        "mapping": item["mapping"], "correct_letter": item["correct_letter"],
        "logit_correct": result["logit_correct"],
        "logit_incorrect": result["logit_incorrect"],
        "margin": result["margin"], "sham_margin": sham["margin"],
        "margin_delta": result["margin"] - sham["margin"],
        "correct": score_margin(result["margin"]),
        "sham_correct": score_margin(sham["margin"]),
        "top_token_is_letter": result["top_token"] in letter_ids.values(),
        "sham_top_token_is_letter": sham["top_token"] in letter_ids.values(),
        "finite": result["finite"],
        "realized_amplitude": result["realized_amplitude"],
        "clean_token_norm": result["clean_token_norm"],
        "js_divergence": result["js_divergence"],
    }


# --------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------

def summarize(trials, preserved_drop=0.05):
    """Per-cell classification performance, then one dose curve per (layer, family,
    matching).

    Every perturbed trial carries its own sham reference, so the accuracy drop of a
    cell is matched item by item: a condition that runs on a subset of the items
    (the concept families, restricted to the concepts they were not built from) is
    compared against the sham on exactly that subset.
    """
    cells = defaultdict(lambda: {
        "n": 0, "correct": 0.0, "sham_correct": 0.0, "margin": [], "sham_margin": [],
        "invalid": 0, "sham_invalid": 0, "non_finite": 0, "ties": 0, "realized": [],
        "js": []})
    for row in trials:
        if row["kind"] != "perturbed":
            continue
        cell = cells[(row["layer"], row["family"], row["matching"], row["dose"])]
        cell["n"] += 1
        cell["correct"] += float(row["correct"])
        cell["sham_correct"] += float(row["sham_correct"])
        cell["margin"].append(row["margin"])
        cell["sham_margin"].append(row["sham_margin"])
        cell["invalid"] += int(not row["top_token_is_letter"])
        cell["sham_invalid"] += int(not row["sham_top_token_is_letter"])
        cell["non_finite"] += int(not row["finite"])
        cell["ties"] += int(row["margin"] == 0.0)
        if row["realized_amplitude"] is not None:
            cell["realized"].append(row["realized_amplitude"])
        if row["js_divergence"] is not None:
            cell["js"].append(row["js_divergence"])

    per_dose = []
    for (layer, family, matching, dose), cell in sorted(cells.items()):
        accuracy = cell["correct"] / cell["n"]
        sham_accuracy = cell["sham_correct"] / cell["n"]
        per_dose.append({
            "layer": layer,
            "family": family,
            "matching": matching,
            "dose": dose,
            "n_trials": cell["n"],
            "accuracy": accuracy,
            "sham_accuracy": sham_accuracy,
            "delta_accuracy": accuracy - sham_accuracy,
            "mean_margin": float(np.mean(cell["margin"])),
            "mean_sham_margin": float(np.mean(cell["sham_margin"])),
            "mean_margin_delta": float(np.mean(cell["margin"]) - np.mean(cell["sham_margin"])),
            "invalid_rate": cell["invalid"] / cell["n"],
            "sham_invalid_rate": cell["sham_invalid"] / cell["n"],
            "non_finite_rate": cell["non_finite"] / cell["n"],
            "tie_rate": cell["ties"] / cell["n"],
            "mean_realized_amplitude": (float(np.mean(cell["realized"]))
                                        if cell["realized"] else None),
            "mean_js_divergence": float(np.mean(cell["js"])) if cell["js"] else None,
            "n_js": len(cell["js"]),
        })

    curves = []
    keys = sorted({(row["layer"], row["family"], row["matching"]) for row in per_dose})
    for layer, family, matching in keys:
        points = sorted(
            (row for row in per_dose
             if (row["layer"], row["family"], row["matching"]) == (layer, family, matching)),
            key=lambda row: row["dose"])
        doses = [row["dose"] for row in points]
        deltas = [row["delta_accuracy"] for row in points]
        # The largest tested dose that still leaves the task intact, and every smaller
        # one with it: H4 is about a band, so a single dose that recovers by chance
        # above a collapse must not extend it.
        preserved = None
        for dose, delta in zip(doses, deltas):
            if delta < -preserved_drop:
                break
            preserved = dose
        curves.append({
            "layer": layer,
            "family": family,
            "matching": matching,
            "doses": doses,
            "accuracy": [row["accuracy"] for row in points],
            "sham_accuracy": [row["sham_accuracy"] for row in points],
            "delta_accuracy": deltas,
            "mean_margin_delta": [row["mean_margin_delta"] for row in points],
            "invalid_rate": [row["invalid_rate"] for row in points],
            "mean_js_divergence": [row["mean_js_divergence"] for row in points],
            "n_trials": [row["n_trials"] for row in points],
            "preserved_accuracy_drop": preserved_drop,
            "max_dose_task_preserved": preserved,
        })

    sham_rows = [row for row in trials if row["kind"] == "sham"]
    sham_summary = {"n_trials": len(sham_rows)}
    if sham_rows:
        sham_summary.update({
            "accuracy": float(np.mean([row["correct"] for row in sham_rows])),
            "mean_margin": float(np.mean([row["margin"] for row in sham_rows])),
            "invalid_rate": float(np.mean(
                [not row["top_token_is_letter"] for row in sham_rows])),
            # The inverted mapping is what separates a real classification from a
            # preference for one letter, so its two halves are reported apart.
            "accuracy_by_mapping": {
                mapping: float(np.mean([row["correct"] for row in sham_rows
                                        if row["mapping"] == mapping]))
                for mapping in sorted({row["mapping"] for row in sham_rows})},
            "accuracy_by_label": {
                "positive": _mean_or_none([row["correct"] for row in sham_rows
                                           if row["positive"]]),
                "negative": _mean_or_none([row["correct"] for row in sham_rows
                                           if not row["positive"]])},
            "accuracy_by_concept": {
                concept: float(np.mean([row["correct"] for row in sham_rows
                                        if row["eval_concept"] == concept]))
                for concept in sorted({row["eval_concept"] for row in sham_rows})},
        })
    return per_dose, curves, sham_summary


def _mean_or_none(values):
    return float(np.mean(values)) if values else None


# --------------------------------------------------------------------------------
# Joining with Experiment 1 (doc 8.5 step 7 and doc 8.7)
# --------------------------------------------------------------------------------

def load_experiment1_per_dose(path):
    """Read the per-dose table of an Experiment 1 run directory or summary.json."""
    path = Path(path)
    if path.is_dir():
        path = path / "summary.json"
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    rows = summary.get("per_dose")
    if not rows:
        raise ValueError(f"{path} carries no per_dose table; is it an Experiment 1 summary?")
    return rows, path


def join_with_experiment1(per_dose, detection_rows, metric="adjusted"):
    """One row per (layer, family, matching, dose) present in both experiments.

    Doc 8.5 is explicit that the two tasks do not share a response: what is joined is
    the intervention condition. A cell absent from either side is dropped rather than
    interpolated.
    """
    detection = {
        (row["layer"], row["family"], row["matching"], float(row["dose"])): row
        for row in detection_rows}
    joined = []
    for row in per_dose:
        key = (row["layer"], row["family"], row["matching"], float(row["dose"]))
        other = detection.get(key)
        if other is None:
            continue
        joined.append({
            "layer": row["layer"],
            "family": row["family"],
            "matching": row["matching"],
            "dose": row["dose"],
            "detection_accuracy": other[f"accuracy_{metric}"],
            "detection_n_trials": other["n_trials"],
            "classification_accuracy": row["accuracy"],
            "classification_delta_accuracy": row["delta_accuracy"],
            "classification_invalid_rate": row["invalid_rate"],
            "classification_n_trials": row["n_trials"],
            "mean_js_divergence": row["mean_js_divergence"],
        })
    return sorted(joined, key=lambda row: (row["layer"], row["family"], row["matching"],
                                           row["dose"]))


def detection_performance_relation(joined):
    """Correlation between detectability and the task loss, per family (doc 8.6).

    A correlation near zero with detection above chance is the H4-compatible pattern;
    a strong negative one says detection only appears where the task collapses.
    """
    relations = []
    for family in sorted({row["family"] for row in joined}):
        rows = [row for row in joined if row["family"] == family]
        detection = np.array([row["detection_accuracy"] for row in rows], dtype=float)
        drop = np.array([row["classification_delta_accuracy"] for row in rows], dtype=float)
        correlation = None
        if len(rows) >= 3 and detection.std() > 0 and drop.std() > 0:
            correlation = float(np.corrcoef(detection, drop)[0, 1])
        relations.append({
            "family": family,
            "n_cells": len(rows),
            "pearson_detection_vs_accuracy_drop": correlation,
            "mean_detection_accuracy": float(detection.mean()),
            "mean_accuracy_drop": float(drop.mean()),
            # The cells that matter for H4: the task is intact and detection is up.
            "n_cells_task_preserved": int(np.sum(drop >= -0.05)),
            "max_detection_with_task_preserved": (
                float(detection[drop >= -0.05].max()) if np.any(drop >= -0.05) else None),
        })
    return relations


def performance_matched_comparison(joined, edges=PERFORMANCE_BINS):
    """Detection by family at matched task loss (doc 8.7).

    This is the complementary analysis, not a replacement for the alpha and z
    matchings: the bins pool doses, so a family reaching a bin at a much lower dose
    than another is not visible here.
    """
    rows = []
    for index in range(len(edges) - 1):
        low, high = edges[index], edges[index + 1]
        for family in sorted({row["family"] for row in joined}):
            selected = [row for row in joined
                        if row["family"] == family
                        and low <= row["classification_delta_accuracy"] < high]
            if not selected:
                continue
            rows.append({
                "accuracy_drop_bin": [low, high],
                "family": family,
                "n_cells": len(selected),
                "mean_detection_accuracy": float(np.mean(
                    [row["detection_accuracy"] for row in selected])),
                "mean_accuracy_drop": float(np.mean(
                    [row["classification_delta_accuracy"] for row in selected])),
                "doses": sorted({row["dose"] for row in selected}),
            })
    return rows


# --------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------

FAMILY_COLORS = {"concept": "#1f77b4", "random": "#ff7f0e", "noise": "#2ca02c",
                 "dropout": "#d62728", "scrambled": "#9467bd"}


def plot_curves(curves, output_dir):
    """One figure per layer: classification accuracy and logit margin against dose."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for layer in sorted({row["layer"] for row in curves}):
        figure, axes = plt.subplots(2, 2, figsize=(13, 9))
        for column, matching in enumerate(MATCHINGS):
            selected = [row for row in curves
                        if (row["layer"], row["matching"]) == (layer, matching)]
            if not selected:
                continue
            top, bottom = axes[0][column], axes[1][column]
            for row in selected:
                color = FAMILY_COLORS.get(row["family"], "gray")
                top.plot(row["doses"], row["accuracy"], marker="o", color=color,
                         label=row["family"])
                bottom.plot(row["doses"], row["mean_margin_delta"], marker="o",
                            color=color, label=row["family"])
            reference = selected[0]
            top.plot(reference["doses"], reference["sham_accuracy"], color="black",
                     linestyle="--", alpha=0.6, label="sham")
            top.axhline(0.5, color="gray", linestyle=":")
            bottom.axhline(0.0, color="gray", linestyle=":")
            for axis in (top, bottom):
                axis.set_xscale("log")
                axis.grid(True, alpha=0.3)
                handles, labels = axis.get_legend_handles_labels()
                unique = dict(zip(labels, handles))
                if unique:
                    axis.legend(unique.values(), unique.keys(), fontsize=8)
            top.set_ylim(0.0, 1.0)
            top.set_title(f"{matching}-matched")
            bottom.set_xlabel("alpha per targeted token" if matching == "alpha"
                              else "z = alpha / s(layer, direction)")
        axes[0][0].set_ylabel("classification accuracy")
        axes[1][0].set_ylabel("margin of the correct letter, minus sham")
        figure.suptitle(f"Experiment 4 - layer {layer}")
        figure.tight_layout()
        path = output_dir / f"task_layer{layer}.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {path}", flush=True)


def plot_detection_vs_performance(joined, output_dir):
    """Detectability against the task loss, one panel per matching (doc 8.6)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for column, matching in enumerate(MATCHINGS):
        axis = axes[column]
        for family in sorted({row["family"] for row in joined}):
            rows = [row for row in joined
                    if row["family"] == family and row["matching"] == matching]
            if not rows:
                continue
            axis.scatter([row["classification_delta_accuracy"] for row in rows],
                         [row["detection_accuracy"] for row in rows],
                         color=FAMILY_COLORS.get(family, "gray"), label=family, s=24,
                         alpha=0.8)
        axis.axhline(0.5, color="gray", linestyle=":")
        axis.axvline(-0.05, color="gray", linestyle="--", alpha=0.5)
        axis.set_xlabel("classification accuracy, minus sham")
        axis.set_title(f"{matching}-matched")
        axis.grid(True, alpha=0.3)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
    axes[0].set_ylabel("Experiment 1 detection accuracy")
    figure.suptitle("Experiment 4 - detection against task degradation")
    figure.tight_layout()
    path = output_dir / "detection_vs_performance.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {path}", flush=True)


# --------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------

NUMERIC_FIELDS = ("dose", "alpha_requested", "dropout_rate", "logit_correct",
                  "logit_incorrect", "margin", "sham_margin", "margin_delta",
                  "realized_amplitude", "clean_token_norm", "js_divergence")
INTEGER_FIELDS = ("layer", "item_id", "sentence_index")
BOOLEAN_FIELDS = ("positive", "top_token_is_letter", "sham_top_token_is_letter", "finite")


def load_trials(path):
    """Read back a trials.csv, restoring the types summarize expects."""
    trials = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            record = dict(row)
            for field in NUMERIC_FIELDS:
                value = record.get(field)
                record[field] = None if value in (None, "") else float(value)
            for field in INTEGER_FIELDS:
                value = record.get(field)
                record[field] = None if value in (None, "") else int(value)
            for field in BOOLEAN_FIELDS:
                value = record.get(field)
                record[field] = None if value in (None, "") else value == "True"
            # Scores are recomputed from the margins rather than read back, so the
            # scoring rule lives in one place and an older run is re-scored correctly.
            record["correct"] = score_margin(record["margin"])
            record["sham_correct"] = score_margin(record["sham_margin"])
            trials.append(record)
    return trials


def analyze(trials, args, experiment1_rows=None):
    """The whole analysis side, shared by a fresh run and by --reanalyze."""
    per_dose, curves, sham_summary = summarize(trials, args.preserved_accuracy_drop)
    analysis = {"n_trials": len(trials), "sham": sham_summary,
                "per_dose": per_dose, "curves": curves}
    if experiment1_rows:
        joined = join_with_experiment1(per_dose, experiment1_rows,
                                       metric=args.experiment1_metric)
        analysis["detection_vs_performance"] = joined
        analysis["detection_performance_relation"] = detection_performance_relation(joined)
        analysis["performance_matched"] = performance_matched_comparison(joined)
    return analysis


def reanalyze(run_dir, args):
    """Recompute summary.json and the figures from an existing trials.csv.

    A sweep costs GPU hours and its raw records are complete; scoring rules, bins and
    the Experiment 1 join change more often than the trials do.
    """
    run_dir = Path(run_dir)
    trials = load_trials(run_dir / "trials.csv")

    experiment1_rows = None
    if args.experiment1_summary:
        experiment1_rows, source = load_experiment1_per_dose(args.experiment1_summary)
        print(f"Joined against {source} ({len(experiment1_rows)} detection cells)", flush=True)

    analysis = analyze(trials, args, experiment1_rows)
    summary_path = run_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    summary.update(analysis)
    summary["reanalyzed"] = True
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"Rewrote {summary_path} from {len(trials)} trials", flush=True)

    if args.plots:
        plot_curves(analysis["curves"], run_dir)
        if analysis.get("detection_vs_performance"):
            plot_detection_vs_performance(analysis["detection_vs_performance"], run_dir)
    return analysis


def print_curve_table(curves):
    print(f"\n{'layer':>6} {'family':>10} {'matching':>9} {'sham':>6} {'acc@min':>8} "
          f"{'acc@max':>8} {'drop@max':>9} {'preserved':>10}", flush=True)
    for row in curves:
        preserved = ("" if row["max_dose_task_preserved"] is None
                     else f"{row['max_dose_task_preserved']:.4g}")
        print(f"{row['layer']:>6} {row['family']:>10} {row['matching']:>9} "
              f"{row['sham_accuracy'][0]:>6.0%} {row['accuracy'][0]:>8.0%} "
              f"{row['accuracy'][-1]:>8.0%} {row['delta_accuracy'][-1]:>+9.0%} "
              f"{preserved:>10}", flush=True)


def print_relation_table(relations):
    print(f"\n{'family':>10} {'cells':>6} {'r(det, drop)':>13} {'mean det':>9} "
          f"{'mean drop':>10} {'det | task ok':>14}", flush=True)
    for row in relations:
        correlation = ("" if row["pearson_detection_vs_accuracy_drop"] is None
                       else f"{row['pearson_detection_vs_accuracy_drop']:+.3f}")
        best = ("" if row["max_detection_with_task_preserved"] is None
                else f"{row['max_detection_with_task_preserved']:.0%}")
        print(f"{row['family']:>10} {row['n_cells']:>6} {correlation:>13} "
              f"{row['mean_detection_accuracy']:>9.0%} {row['mean_accuracy_drop']:>+10.0%} "
              f"{best:>14}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Experiment 4: detection and task degradation")
    parser.add_argument("--calibration_config",
                        default="configs/experiment_0_calibration/development_full.yaml",
                        help="Experiment 0 protocol config; supplies the model, the "
                             "calibrated layers, the concepts and the concept vectors.")
    parser.add_argument("--calibration_dir", default=None,
                        help="Experiment 0 output directory holding directional_scales.json "
                             f"(default: {DEFAULT_CALIBRATION_DIR.relative_to(REPO_ROOT)}).")
    parser.add_argument("--allow_calibration_mismatch", action="store_true",
                        help="Warn instead of failing when the calibration artifacts were "
                             "produced by a different version of the config.")
    parser.add_argument("--estimator", choices=sorted(ESTIMATOR_FIELDS), default="sd",
                        help="Scale used by the z matching, as in Experiment 1.")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Subset of the calibrated layers (default: all of them).")
    parser.add_argument("--families", nargs="+", default=["concept", "random", "noise",
                                                          "dropout"],
                        choices=list(FAMILIES))
    parser.add_argument("--matchings", nargs="+", default=list(MATCHINGS),
                        choices=list(MATCHINGS))
    parser.add_argument("--eval_concepts", nargs="+", default=None,
                        help="Concepts whose sentences are classified (default: every "
                             f"concept of {COMPLEX_DATA.name}).")
    parser.add_argument("--injected_concept_pool", nargs="+", default=None,
                        help="Calibrated concepts the injected directions are drawn from "
                             "(default: the simple_data concepts of the calibration, "
                             "which share no sentence with the classification corpus).")
    parser.add_argument("--concept_pairs", nargs="+", default=None,
                        metavar="EVALUATED:INJECTED",
                        help="Explicit pairing, repeatable per evaluated concept. "
                             "The two names must differ (doc 8.3).")
    parser.add_argument("--num_injected_concepts", type=int, default=1,
                        help="Injected concepts per evaluated concept when the pairing "
                             "is left to the default rotation.")
    parser.add_argument("--num_sentences_per_class", type=int, default=4,
                        help="Positive and negative examples per evaluated concept; the "
                             "two classes always get the same count (doc 8.4).")
    parser.add_argument("--mappings", nargs="+", default=["XY", "YX"],
                        choices=["XY", "YX"],
                        help="Letter assignment. XY reads X as yes; YX is its inversion.")
    parser.add_argument("--num_random", type=int, default=2,
                        help="Fixed random directions per layer, taken from the bank.")
    parser.add_argument("--num_noise", type=int, default=1)
    parser.add_argument("--num_dropout", type=int, default=1)
    parser.add_argument("--alpha_doses", nargs="+", default=None)
    parser.add_argument("--alpha_range", nargs=2, default=None, metavar=("LOW", "HIGH"))
    parser.add_argument("--z_doses", nargs="+", default=None)
    parser.add_argument("--z_range", nargs=2, default=None, metavar=("LOW", "HIGH"))
    parser.add_argument("--num_doses", type=int, default=7,
                        help="Doses per grid when a range is given. Doc 8.4 requires the "
                             "same grids as Experiment 1, so pass its values here when "
                             "that run did not use the defaults.")
    parser.add_argument("--dropout_norm_source", choices=["trial", "calibration"],
                        default="trial",
                        help="Activation norm turning an amplitude into a rate. Only "
                             "trial is currently available; Experiment 0 records no "
                             "per-layer activation norm.")
    parser.add_argument("--js_per_cell", type=int, default=1,
                        help="Items per (layer, family, direction, matching, dose) cell "
                             "carrying the JS diagnostic of doc 8.6, taken in plan order. "
                             "It reuses the trial's own logits, so it costs no extra "
                             "forward pass; 0 disables it.")
    parser.add_argument("--preserved_accuracy_drop", type=float, default=0.05,
                        help="Accuracy loss below which the task counts as preserved, "
                             "for the reported dose band and the H4 reading of doc 8.8.")
    parser.add_argument("--experiment1_summary", default=None, metavar="RUN_DIR_OR_JSON",
                        help="Experiment 1 run directory or summary.json to join on "
                             "(layer, family, matching, dose), per doc 8.5 step 7.")
    parser.add_argument("--experiment1_metric", choices=["adjusted", "raw"],
                        default="adjusted",
                        help="Detection accuracy used in the join. adjusted removes the "
                             "model's letter bias and is what Experiment 1 reports.")
    parser.add_argument("--measure_amplitude", action="store_true", default=True)
    parser.add_argument("--no_measure_amplitude", dest="measure_amplitude",
                        action="store_false")
    parser.add_argument("--limit_trials", type=int, default=None,
                        help="Stop after this many perturbed trials (smoke tests).")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--model", default=None, help="Defaults to the calibration model.")
    parser.add_argument("--output_dir", default="results/experiment4")
    parser.add_argument("--run_name", default="pilot")
    parser.add_argument("--no_progress", dest="progress", action="store_false", default=True)
    parser.add_argument("--no_plots", dest="plots", action="store_false", default=True)
    parser.add_argument("--dry_run", action="store_true",
                        help="Build the plan and print its size without loading the model.")
    parser.add_argument("--reanalyze", default=None, metavar="RUN_DIR",
                        help="Recompute summary.json and the figures from an existing "
                             "trials.csv, without loading the model.")
    return parser


def resolve_plan(parser, args, calibration, corpus):
    """Fill in the parts of the plan that depend on the calibration and the corpus."""
    calibrated_layers = calibration.layers
    args.layers = args.layers or list(calibrated_layers)
    missing = [layer for layer in args.layers if layer not in calibrated_layers]
    if missing:
        parser.error(f"decoder blocks {missing} are not calibrated in "
                     f"{calibration.calibration_dir}")

    args.eval_concepts = args.eval_concepts or sorted(corpus)
    unknown = [name for name in args.eval_concepts if name not in corpus]
    if unknown:
        parser.error(f"{unknown} are not in {COMPLEX_DATA}; the classification task "
                     f"needs labelled examples, available: {sorted(corpus)}")

    calibrated_concepts = set(calibration.concepts)
    if args.injected_concept_pool:
        pool = list(args.injected_concept_pool)
    else:
        # The simple-data concepts share no sentence with the classification corpus,
        # so they keep doc 3.3's construction/evaluation separation beyond any doubt.
        # A calibration without them falls back to everything it does have.
        simple = [concept.name for concept in calibration.config.concepts
                  if concept.dataset == "simple_data"]
        pool = sorted(simple) if simple else sorted(calibrated_concepts)

    try:
        pairs = (parse_concept_pairs(args.concept_pairs) if args.concept_pairs
                 else default_concept_pairs(args.eval_concepts, pool,
                                            args.num_injected_concepts))
        check_concept_pairs(pairs, args.eval_concepts, calibrated_concepts)
    except ValueError as error:
        parser.error(str(error))

    args.concept_pairs_resolved = {name: list(values) for name, values in sorted(pairs.items())}
    args.injected_concepts = sorted({name for values in pairs.values() for name in values})
    args.eval_for_injected = defaultdict(set)
    for evaluated, injected in pairs.items():
        for name in injected:
            args.eval_for_injected[name].add(evaluated)

    args.dose_grid = {
        "alpha": parse_doses(args.alpha_doses, args.alpha_range, args.num_doses,
                             DEFAULT_ALPHA_DOSES),
        "z": parse_doses(args.z_doses, args.z_range, args.num_doses, DEFAULT_Z_DOSES),
    }
    if "dropout" in args.families and args.dropout_norm_source == "calibration":
        try:
            for layer in args.layers:
                calibration.token_norm(layer)  # fail before the model is loaded
        except (NotImplementedError, KeyError, ValueError) as error:
            parser.error(str(error))

    args.items = build_items(corpus, args.eval_concepts, args.num_sentences_per_class,
                             args.mappings, args.seed)
    return args


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.reanalyze:
        analysis = reanalyze(args.reanalyze, args)
        print_curve_table(analysis["curves"])
        if analysis.get("detection_performance_relation"):
            print_relation_table(analysis["detection_performance_relation"])
        return

    calibration = DirectionBank.load(
        args.calibration_config,
        calibration_dir=args.calibration_dir,
        estimator=args.estimator,
        strict=not args.allow_calibration_mismatch,
    )
    corpus = load_complex_concepts()
    args = resolve_plan(parser, args, calibration, corpus)

    experiment1_rows = None
    if args.experiment1_summary:
        try:
            experiment1_rows, experiment1_source = load_experiment1_per_dose(
                args.experiment1_summary)
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))

    model_name = args.model or calibration.model_name
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir = output_dir / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = build_conditions(calibration, args)
    perturbed_passes, sham_passes = count_forward_passes(conditions, args)

    print("=" * 72, flush=True)
    print("EXPERIMENT 4: DETECTION AND TASK DEGRADATION", flush=True)
    print("=" * 72, flush=True)
    print(f"Model:       {model_name}", flush=True)
    print(f"Calibration: {calibration.calibration_dir} "
          f"(config={args.calibration_config}, estimator={calibration.estimator})", flush=True)
    print(f"Layers:      {args.layers}", flush=True)
    print(f"Families:    {args.families}", flush=True)
    print(f"Matchings:   {args.matchings}", flush=True)
    print(f"Alpha doses: {[round(dose, 4) for dose in args.dose_grid['alpha']]}", flush=True)
    print(f"z doses:     {[round(dose, 4) for dose in args.dose_grid['z']]}", flush=True)
    print(f"Mappings:    {args.mappings}", flush=True)
    print("Pairs:       " + ", ".join(
        f"{evaluated} <- {'+'.join(injected)}"
        for evaluated, injected in args.concept_pairs_resolved.items()), flush=True)
    print(f"Items:       {len(args.items)} "
          f"({args.num_sentences_per_class} per class x {len(args.eval_concepts)} concepts "
          f"x {len(args.mappings)} mappings)", flush=True)
    if experiment1_rows:
        print(f"Experiment 1: {experiment1_source} "
              f"({len(experiment1_rows)} detection cells)", flush=True)
    print(f"Output:      {output_dir}", flush=True)
    print(f"Forward passes: {perturbed_passes} perturbed + {sham_passes} sham", flush=True)

    # Rebuild every fixed direction now: a missing concept vector or a seed that no
    # longer reproduces must fail here, not once the weights are on the GPU.
    try:
        checked = calibration.preflight(args.families, args.layers,
                                        concepts=args.injected_concepts,
                                        num_random=args.num_random)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(f"Directions:  {checked} rebuilt and matched to their calibrated norms", flush=True)
    if args.dry_run:
        print("Dry run: plan only, no model loaded.", flush=True)
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    print("Model loaded.", flush=True)

    trials = run_experiment(model, tokenizer, calibration, args)
    analysis = analyze(trials, args, experiment1_rows)

    trials_path = output_dir / "trials.csv"
    with trials_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trials[0].keys()))
        writer.writeheader()
        writer.writerows(trials)

    summary = {
        "model": model_name,
        "calibration_config": str(args.calibration_config),
        "calibration_dir": str(calibration.calibration_dir),
        "calibration_manifest": calibration.manifest,
        "estimator": calibration.estimator,
        "seed": args.seed,
        "layers": args.layers,
        "families": args.families,
        "matchings": args.matchings,
        "eval_concepts": args.eval_concepts,
        "concept_pairs": args.concept_pairs_resolved,
        "mappings": args.mappings,
        "num_sentences_per_class": args.num_sentences_per_class,
        "dose_grid": args.dose_grid,
        "dropout_norm_source": args.dropout_norm_source,
        "js_per_cell": args.js_per_cell,
        "items": args.items,
        "experiment1_summary": (str(experiment1_source) if experiment1_rows else None),
        "experiment1_metric": args.experiment1_metric,
    }
    summary.update(analysis)
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"\nSaved {len(trials)} trials to {trials_path}", flush=True)
    print(f"Saved summary to {summary_path}", flush=True)

    sham = analysis["sham"]
    print(f"\nSham classification accuracy: {sham['accuracy']:.0%} "
          f"(invalid {sham['invalid_rate']:.0%}), by mapping "
          f"{ {k: round(v, 3) for k, v in sham['accuracy_by_mapping'].items()} }", flush=True)
    print_curve_table(analysis["curves"])
    if analysis.get("detection_performance_relation"):
        print_relation_table(analysis["detection_performance_relation"])

    if args.plots:
        plot_curves(analysis["curves"], output_dir)
        if analysis.get("detection_vs_performance"):
            plot_detection_vs_performance(analysis["detection_vs_performance"], output_dir)

    print("\nEXPERIMENT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
