#!/usr/bin/env python3
"""
Experiment 1: 2AFC localization and psychometric curves at matched alpha and matched z.

Implements section 5 of docs/livrables/cadrage-experiments.md. One prompt, one
presentation plan, four perturbation families, two dose parameterizations:

  family   perturbation applied to every token of the targeted sentence
  -------  ----------------------------------------------------------------------
  concept  alpha * v(concept, layer)      unit concept direction            (3.3)
  random   alpha * v(k, layer)            fixed unit random draw            (3.4)
  noise    alpha * v(q, t)                unit direction renewed per token  (3.5)
  dropout  (m/(1-p) - 1) * h              Bernoulli mask, p derived from alpha (3.6)

  matching  dose rule
  --------  -----------------------------------------------------------------
  alpha     the raw per-token amplitude is the grid value
  z         alpha = z * s(layer, direction), s from the Experiment 0 calibration;
            dropout chooses no direction and uses the layer reference scale
            s_bar(layer) = median over the fixed-random bank of s(layer, v)

Unlike strength_comparison.py, which perturbs both sentences at two strengths, this
experiment perturbs exactly one sentence and asks which one was targeted, so every
condition runs in both presentation orders against both targets, plus a sham.

All four families are applied at the same hook, the decoder block output, so the
comparison is about the nature of the perturbation and not about its site. This
differs from Fornasiere et al., who perturb the attention and MLP sublayers;
utils/gaussian_dropout_hooks.py still supports that site.

A z score is only meaningful for the exact direction it was estimated on, so every
direction is taken from the Experiment 0 calibration bank together with the scale
estimated on it, instead of being rebuilt here. Concept directions come from the .pt
files of data/saved_vectors/llama; fixed-random and renewed-noise directions are
regenerated from the seeds Experiment 0 recorded. Nothing is redrawn locally.

The committed development calibration is used by default:

    python code/experiments/experiment1_psychometrics.py --dry_run

Point --calibration_dir at another Experiment 0 output directory to use a fresh run:

    python -m experiment_0_calibration.prepare_concept_vectors --config <config>
    python -m experiment_0_calibration.prepare_material_plan   --config <config>
    python -m experiment_0_calibration.run_experiment_0        --config <config>
    python code/experiments/experiment1_psychometrics.py \
        --calibration_config <config> --calibration_dir <its paths.output_dir>

The contamination ratio P(1->2) of doc 5.13 is available behind --contamination_trials.
Its restoration counterpart E(1->2) needs activation patching and is left to a
dedicated diagnostic script.
"""

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT / "code", REPO_ROOT / "code" / "utils"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from all_prompts import LOCALIZATION_SENTENCES
from gaussian_dropout_hooks import make_dropout_hook, make_vector_injection_hook
from save_random_vectors import derive_seed

from experiment_0_calibration.direction_bank import (
    DEFAULT_CALIBRATION_DIR,
    ESTIMATOR_FIELDS,
    DirectionBank,
)

# "scrambled" is the content control of doc 3.3: a coordinate permutation of a concept
# vector, identical to it in norm and coordinate distribution and aligned with nothing.
# It is only available from a calibration whose config enabled it.
FAMILIES = ("concept", "random", "noise", "dropout", "scrambled")

# Families whose perturbation is alpha times one fixed unit direction, chosen before
# the trial. They share every path that needs the vector itself: building the device
# cache, turning a dose into injection parameters, and registering the hook. Named
# once because naming them at each of those three sites is how "scrambled" reached a
# GPU without an injection branch.
FIXED_DIRECTION_FAMILIES = ("concept", "random", "scrambled")
MATCHINGS = ("alpha", "z")

# Seven non-null doses one factor-two step apart, plus the sham. The pilot of doc 14.6
# replaces these bounds; they only have to bracket the transition to be useful.
DEFAULT_ALPHA_DOSES = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
DEFAULT_Z_DOSES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


# --------------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------------

# The assistant turn the model is asked to continue. build_localization_prompt cuts
# the rendered chat template back to the end of this string, so it must appear in the
# assistant message verbatim.
ASSISTANT_PREFIX = "The answer is"


def get_localization_messages(first_sentence, second_sentence, label_order):
    """The 2AFC prompt of doc 5.3.

    label_order "AB" labels the first printed sentence A. "BA" swaps the visible
    labels while keeping the physical order, which separates a preference for a
    position from a preference for the letter.
    """
    if label_order == "AB":
        body = f"A) {first_sentence}\nB) {second_sentence}"
    else:
        body = f"B) {first_sentence}\nA) {second_sentence}"
    return [
        {
            "role": "user",
            "content": (
                "You will read two sentences. An intervention may have been applied "
                "directly\nto the internal activations at the tokens of one sentence.\n\n"
                f"{body}\n\n"
                "If an intervention occurred, which sentence was directly targeted?\n"
                "Make your best forced choice.\n"
                "Answer with exactly one letter: A or B."
            ),
        },
        {"role": "assistant", "content": ASSISTANT_PREFIX},
    ]


def localization_char_spans(prompt, labels, sentences):
    """Character bounds of each printed sentence in the prompt, label excluded.

    Experiment 0 calibrates s(l, v) on exactly the tokens Experiment 1 perturbs, so
    the manifest it ingests has to locate them with this rule rather than a copy of
    it. Bounds are in physical order: index 0 is the first printed sentence.
    """
    spans = []
    for label, sentence in zip(labels, sentences):
        marker = f"{label}) {sentence}"
        start_char = prompt.find(marker)
        if start_char == -1:
            raise ValueError(f"sentence marker not found in prompt: {marker!r}")
        # The label is context, not part of the perturbed sentence.
        start_char += len(f"{label}) ")
        spans.append((start_char, start_char + len(sentence)))
    return spans


def build_localization_prompt(tokenizer, first_sentence, second_sentence, label_order):
    """Return the prompt, the token range of each sentence, its label, and the encoding.

    Ranges and labels are in physical order: index 0 is the first printed sentence.
    """
    messages = get_localization_messages(first_sentence, second_sentence, label_order)
    # A reasoning model would otherwise open a chain of thought before the letter and
    # push the answer token out of the next position. Chat templates that know nothing
    # of the flag ignore it, so the Llama prompt is unchanged.
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )

    # The template closes the assistant turn it was handed. Cut the prompt back to the
    # end of the assistant prefix so the next token really is the forced choice. Doing
    # it by the prefix rather than by an end-of-turn literal keeps this model-agnostic:
    # Llama's "<|eot_id|>" and Qwen's "<|im_end|>\n" are both removed by the same rule.
    # No trailing space either: the answer continues "The answer is" with a
    # space-prefixed token, and answer_token_ids picks that variant.
    cut = prompt.rfind(ASSISTANT_PREFIX)
    if cut == -1:
        raise ValueError(
            f"assistant prefix {ASSISTANT_PREFIX!r} is absent from the rendered prompt"
        )
    prompt = prompt[: cut + len(ASSISTANT_PREFIX)].rstrip(" ")

    encoding_with_offsets = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True
    )
    offsets = encoding_with_offsets["offset_mapping"][0]
    encoding = {k: v for k, v in encoding_with_offsets.items() if k != "offset_mapping"}

    labels = ["A", "B"] if label_order == "AB" else ["B", "A"]
    spans = localization_char_spans(prompt, labels, (first_sentence, second_sentence))
    ranges = []
    for label, (start_char, end_char) in zip(labels, spans):
        token_start = token_end = None
        for index in range(len(offsets)):
            tok_start = offsets[index][0].item()
            tok_end = offsets[index][1].item()
            if token_start is None and tok_end > start_char:
                token_start = index
            if tok_start < end_char:
                token_end = index + 1
        if token_start is None or token_end is None or token_start >= token_end:
            raise ValueError(f"empty token range for sentence {label}")
        ranges.append((token_start, token_end))

    return prompt, ranges, labels, encoding


def answer_token_ids(tokenizer):
    """Ids of the two answer tokens actually produced after "The answer is".

    Llama tokenizes the continuation as " A" / " B", so reading the bare "A" / "B"
    ids would score a token the model barely considers.
    """
    ids = {}
    for letter in ("A", "B"):
        spaced = tokenizer.encode(f" {letter}", add_special_tokens=False)
        bare = tokenizer.encode(letter, add_special_tokens=False)
        ids[letter] = (spaced if len(spaced) == 1 else bare)[-1]
    if ids["A"] == ids["B"]:
        raise ValueError("the two answer letters map to the same token id")
    return ids["A"], ids["B"]


def build_sentence_pairs(tokenizer, sentences, num_pairs, seed):
    """Pair sentences of close token length, each sentence used at most once (doc 3.2)."""
    lengths = [
        (len(tokenizer(sentence, add_special_tokens=False)["input_ids"]), index)
        for index, sentence in enumerate(sentences)
    ]
    lengths.sort()
    candidates = []
    for position in range(0, len(lengths) - 1, 2):
        (len_a, index_a), (len_b, index_b) = lengths[position], lengths[position + 1]
        candidates.append((abs(len_a - len_b), index_a, index_b))
    candidates.sort()
    if num_pairs > len(candidates):
        raise ValueError(
            f"asked for {num_pairs} pairs but only {len(candidates)} disjoint pairs exist"
        )
    selected = candidates[:num_pairs]
    random.Random(seed).shuffle(selected)
    return [
        {
            "pair_id": pair_index,
            "sentence_x": sentences[index_a],
            "sentence_y": sentences[index_b],
            "token_length_gap": gap,
        }
        for pair_index, (gap, index_a, index_b) in enumerate(selected)
    ]


# --------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------
#
# Directions and their natural scales both come from
# experiment_0_calibration.direction_bank.DirectionBank, which keeps a direction and
# the s(layer, direction) estimated on it inseparable. That is what makes the z
# matching meaningful, so nothing in this file rebuilds a direction of its own.


# --------------------------------------------------------------------------------
# Dose parameterization
# --------------------------------------------------------------------------------

def log_grid(low, high, count):
    """Doses spaced by a constant factor, as in doc 5.5."""
    if not 0 < low < high:
        raise ValueError("dose bounds must satisfy 0 < low < high")
    if count < 2:
        raise ValueError("a grid needs at least two doses")
    step = (high / low) ** (1.0 / (count - 1))
    return [low * step ** index for index in range(count)]


def dropout_rate_for_amplitude(alpha, token_norm):
    """Rate whose expected perturbation norm is alpha: p = rho^2/(1+rho^2), rho = alpha/h_bar.

    Follows E||dh||^2 = p/(1-p) ||h||^2 for the rescaled Bernoulli mask (doc 3.6).
    """
    if not token_norm > 0.0:
        raise ValueError("activation norm must be positive")
    rho = float(alpha) / float(token_norm)
    return rho * rho / (1.0 + rho * rho)


def amplitude_for_dropout_rate(rate, token_norm):
    """Inverse of dropout_rate_for_amplitude."""
    if not 0.0 <= rate < 1.0:
        raise ValueError(f"dropout rate must be in [0, 1), got {rate}")
    return float(token_norm) * math.sqrt(rate / (1.0 - rate))


# --------------------------------------------------------------------------------
# Interventions
# --------------------------------------------------------------------------------

def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


def make_token_varying_injection_hook(delta, token_range):
    """Add one precomputed row per targeted token: the renewed-direction noise family."""
    def hook_fn(module, inputs, output):
        tensor = _hidden(output)
        start, end = token_range
        end = min(end, tensor.shape[1])
        if start >= end:
            return output
        rows = delta[: end - start].to(device=tensor.device, dtype=tensor.dtype)
        modified = tensor.clone()
        modified[:, start:end, :] = modified[:, start:end, :] + rows.unsqueeze(0)
        return (modified,) + output[1:] if isinstance(output, tuple) else modified
    return hook_fn


def make_capture_hook(store, key, token_range):
    """Keep the targeted-token activations of one layer, for realized-amplitude logging."""
    def hook_fn(module, inputs, output):
        tensor = _hidden(output)
        start, end = token_range
        end = min(end, tensor.shape[1])
        store[key] = tensor[0, start:end, :].detach().to(torch.float32).clone()
    return hook_fn


def make_norm_capture_hook(store, layer, ranges):
    """Record the RMS token norm of both sentences at one layer, from a sham pass."""
    def hook_fn(module, inputs, output):
        tensor = _hidden(output)
        for index, (start, end) in enumerate(ranges):
            stop = min(end, tensor.shape[1])
            segment = tensor[0, start:stop, :].detach().to(torch.float32)
            store[(int(layer), index)] = float(segment.pow(2).sum(dim=-1).mean().sqrt().item())
    return hook_fn


def make_sentence_capture_hook(store, layer, token_range):
    """Keep a whole sentence block at one layer, for the contamination diagnostic."""
    def hook_fn(module, inputs, output):
        tensor = _hidden(output)
        start, end = token_range
        end = min(end, tensor.shape[1])
        store[int(layer)] = tensor[0, start:end, :].detach().to(torch.float32).clone()
    return hook_fn


@contextmanager
def intervention(model, layer, token_range, family, vector=None, alpha=None,
                 delta=None, rate=None, seed=0, measure=None, extra_hooks=()):
    """Register one family's perturbation on the decoder block output of `layer`.

    Forward hooks fire in registration order and each one receives the previous one's
    output, so a clean capture registered first and a perturbed capture registered last
    bracket the perturbation and give its realized amplitude.
    """
    block = model.model.layers[layer]
    handles = []
    try:
        if measure is not None:
            handles.append(block.register_forward_hook(
                make_capture_hook(measure, "clean", token_range)))

        if family in FIXED_DIRECTION_FAMILIES:
            handles.append(block.register_forward_hook(
                make_vector_injection_hook(vector, [(token_range, float(alpha))])))
        elif family == "noise":
            handles.append(block.register_forward_hook(
                make_token_varying_injection_hook(delta, token_range)))
        elif family == "dropout":
            handles.append(block.register_forward_hook(
                make_dropout_hook(float(rate), token_range, seed=seed)))
        else:
            raise ValueError(f"unknown family {family!r}")

        if measure is not None:
            handles.append(block.register_forward_hook(
                make_capture_hook(measure, "perturbed", token_range)))
        for module, hook in extra_hooks:
            handles.append(module.register_forward_hook(hook))
        yield handles
    finally:
        for handle in handles:
            handle.remove()


def noise_delta(calibration, layer, num_tokens, matching, dose, trial_seed):
    """Per-token perturbation rows for the noise family (doc 3.5).

    Directions come from the pre-drawn noise bank, so each one carries a calibrated
    scale, which is what the z matching needs at the token level.
    """
    bank_ids = calibration.direction_ids("noise", layer)
    if not bank_ids:
        raise ValueError(f"no noise direction calibrated at layer {layer}")
    rng = random.Random(derive_seed(trial_seed, "noise", layer))
    # Distinct directions whenever the bank is large enough, so a trial really does
    # renew the direction at every token rather than reusing a few of them.
    if len(bank_ids) >= num_tokens:
        drawn = rng.sample(bank_ids, num_tokens)
    else:
        drawn = [bank_ids[rng.randrange(len(bank_ids))] for _ in range(num_tokens)]
    rows = []
    used = []
    for direction_id in drawn:
        vector = calibration.vector(direction_id).to(torch.float32)
        amplitude = (float(dose) if matching == "alpha"
                     else float(dose) * calibration.scale(direction_id))
        rows.append(amplitude * vector)
        used.append((direction_id, amplitude))
    return torch.stack(rows, dim=0), used


# --------------------------------------------------------------------------------
# Trial execution
# --------------------------------------------------------------------------------

@torch.inference_mode()
def run_sham(model, encoding, ranges, token_a, token_b, layers):
    """One clean forward pass: the reference contrast and the clean token norms."""
    norms = {}
    handles = []
    try:
        for layer in layers:
            handles.append(model.model.layers[layer].register_forward_hook(
                make_norm_capture_hook(norms, layer, ranges)))
        logits = model(**encoding).logits[0, -1, :]
    finally:
        for handle in handles:
            handle.remove()
    logit_a = float(logits[token_a].item())
    logit_b = float(logits[token_b].item())
    return {
        "logit_a": logit_a,
        "logit_b": logit_b,
        "contrast": logit_a - logit_b,
        "top_token": int(torch.argmax(logits).item()),
        "finite": bool(torch.isfinite(logits).all().item()),
        "token_norms": norms,
    }


@torch.inference_mode()
def run_trial(model, encoding, layer, token_range, family, token_a, token_b,
              vector=None, alpha=None, delta=None, rate=None, seed=0, measure=True):
    """One perturbed forward pass: the A/B contrast and the realized amplitude."""
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

    logit_a = float(logits[token_a].item())
    logit_b = float(logits[token_b].item())
    return {
        "logit_a": logit_a,
        "logit_b": logit_b,
        "contrast": logit_a - logit_b,
        "top_token": int(torch.argmax(logits).item()),
        "finite": bool(torch.isfinite(logits).all().item()),
        "realized_amplitude": realized,
        "clean_token_norm": clean_norm,
    }


@torch.inference_mode()
def capture_sham_blocks(model, encoding, token_range, layers):
    """Clean activations of one sentence at every probed layer."""
    captured = {}
    handles = []
    try:
        for layer in layers:
            handles.append(model.model.layers[layer].register_forward_hook(
                make_sentence_capture_hook(captured, layer, token_range)))
        model(**encoding)
    finally:
        for handle in handles:
            handle.remove()
    return captured


@torch.inference_mode()
def contamination_ratio(model, encoding, layer, target_range, other_range, family,
                        sham_blocks, layers, **perturbation):
    """P(1->2): relative change of the untargeted sentence at each layer >= injection.

    Doc 5.13. A large value means the two sentences are not independent observations.
    """
    captured = {}
    extra = [
        (model.model.layers[probe], make_sentence_capture_hook(captured, probe, other_range))
        for probe in layers if probe >= layer
    ]
    with intervention(model, layer, target_range, family, extra_hooks=extra, **perturbation):
        model(**encoding)

    ratios = {}
    for probe, perturbed in captured.items():
        clean = sham_blocks.get(probe)
        if clean is None or clean.shape != perturbed.shape:
            continue
        ratios[probe] = float((perturbed - clean).norm().item()) / (float(clean.norm().item()) + 1e-6)
    return ratios


# --------------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------------

def build_conditions(calibration, args):
    """Enumerate the (family, direction) conditions of each layer, from the bank."""
    conditions = defaultdict(list)
    for layer in args.layers:
        if "concept" in args.families:
            for concept in args.concepts:
                ids = calibration.direction_ids("concept", layer, concept=concept)
                if not ids:
                    raise ValueError(
                        f"concept {concept!r} is not in the calibration bank at layer {layer}")
                for direction_id in ids:
                    conditions[layer].append(("concept", direction_id))
        if "scrambled" in args.families:
            for concept in args.concepts:
                ids = calibration.direction_ids("scrambled", layer, concept=concept)
                if not ids:
                    raise ValueError(
                        f"concept {concept!r} has no scrambled counterpart at layer {layer}. "
                        "Its calibration must set directions.scrambled_concept.enabled.")
                for direction_id in ids:
                    conditions[layer].append(("scrambled", direction_id))
        if "random" in args.families:
            ids = calibration.direction_ids("random", layer)[: args.num_random]
            if not ids:
                raise ValueError(f"no random direction calibrated at layer {layer}")
            for direction_id in ids:
                conditions[layer].append(("random", direction_id))
        if "noise" in args.families:
            for index in range(args.num_noise):
                conditions[layer].append(("noise", f"noise_realization_{index:02d}"))
        if "dropout" in args.families:
            for index in range(args.num_dropout):
                conditions[layer].append(("dropout", f"dropout_realization_{index:02d}"))
    return conditions


def count_forward_passes(conditions, args):
    """Perturbed passes; sham passes are shared by every condition and matching."""
    doses = sum(len(args.dose_grid[matching]) for matching in args.matchings)
    cells = sum(len(conditions[layer]) for layer in args.layers) * doses
    per_cell = len(args.pairs) * 2 * len(args.label_orders) * 2  # orders x targets
    shams = len(args.pairs) * 2 * len(args.label_orders)
    return cells * per_cell, shams


# --------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------

def fit_psychometric(doses, correct, total, max_iterations=200, tolerance=1e-9):
    """Fit p = 0.5 + 0.5 logistic(b0 + b1 log dose) by damped Newton ascent.

    Returns (b0, b1, converged). The 0.5 floor is the forced-choice guessing rate of
    doc 5.10; the threshold is where the logistic term reaches 0.5, i.e. b0 + b1 log d = 0.
    """
    x = np.column_stack([np.ones(len(doses)), np.log(np.asarray(doses, dtype=float))])
    k = np.asarray(correct, dtype=float)
    n = np.asarray(total, dtype=float)
    if np.any(n <= 0):
        return None, None, False

    beta = np.array([0.0, 1.0])

    def log_likelihood(params):
        eta = x @ params
        p = np.clip(0.5 + 0.5 / (1.0 + np.exp(-eta)), 1e-9, 1 - 1e-9)
        return float(np.sum(k * np.log(p) + (n - k) * np.log(1.0 - p)))

    def point_gradient(eta):
        sigma = 1.0 / (1.0 + np.exp(-eta))
        p = np.clip(0.5 + 0.5 * sigma, 1e-9, 1 - 1e-9)
        return (k / p - (n - k) / (1.0 - p)) * 0.5 * sigma * (1.0 - sigma)

    previous = log_likelihood(beta)
    converged = False
    steps_taken = 0
    for _ in range(max_iterations):
        eta = x @ beta
        gradient = x.T @ point_gradient(eta)

        # Per-point curvature by central difference. With two parameters an exact
        # analytic Hessian buys nothing and is easier to get wrong.
        epsilon = 1e-4
        curvature = (point_gradient(eta + epsilon) - point_gradient(eta - epsilon)) / (2 * epsilon)
        hessian = x.T @ (curvature[:, None] * x)

        step = None
        try:
            candidate = np.linalg.solve(hessian, gradient)
            if np.all(np.isfinite(candidate)):
                step = -candidate
        except np.linalg.LinAlgError:
            step = None
        if step is None:
            step = gradient / (np.linalg.norm(gradient) + 1e-12)

        scale = 1.0
        improved = False
        for _ in range(40):
            trial = beta + scale * step
            value = log_likelihood(trial)
            if value > previous:
                converged = abs(value - previous) < tolerance
                beta, previous, improved = trial, value, True
                steps_taken += 1
                break
            scale *= 0.5
        if not improved or converged:
            break

    # A line search that fails on the first iteration leaves beta at its starting
    # point; reporting that as a fit would publish the initial guess as a threshold.
    if steps_taken == 0 or not np.all(np.isfinite(beta)) or beta[1] <= 0:
        return float(beta[0]), float(beta[1]), False
    return float(beta[0]), float(beta[1]), converged


def forced_choice_score(value, target_label):
    """Score one forced choice: 1 correct, 0 wrong, 0.5 for an exact tie.

    A tie is not an error. In bfloat16 a small perturbation can leave the two logits
    untouched, and scoring that as wrong for both targets pulls a cell towards 0
    instead of towards chance.
    """
    if value == 0.0:
        return 0.5
    return 1.0 if (value > 0.0) == (target_label == "A") else 0.0


def threshold_at_75(beta0, beta1):
    """Dose where the fitted curve reaches 0.75, i.e. exp(-b0 / b1)."""
    if beta0 is None or beta1 is None or beta1 <= 0:
        return None
    try:
        value = math.exp(-beta0 / beta1)
    except OverflowError:
        return None
    return value if math.isfinite(value) else None


def bracketing_doses(doses, accuracies, level=0.75):
    """The two tested doses straddling `level`, or None if it is never crossed."""
    ordered = sorted(zip(doses, accuracies))
    for (low_dose, low_acc), (high_dose, high_acc) in zip(ordered, ordered[1:]):
        if low_acc < level <= high_acc:
            return low_dose, high_dose
    return None


def summarize(trials):
    """Accuracies per cell, then one psychometric fit per (layer, family, matching)."""
    cells = defaultdict(lambda: {"n": 0, "raw": 0.0, "adjusted": 0.0, "contrast": [],
                                 "realized": [], "invalid": 0, "non_finite": 0, "ties": 0})
    for row in trials:
        if row["kind"] != "perturbed":
            continue
        cell = cells[(row["layer"], row["family"], row["matching"], row["dose"])]
        cell["n"] += 1
        cell["raw"] += float(row["correct_raw"])
        cell["adjusted"] += float(row["correct_adjusted"])
        cell["ties"] += int(row.get("tie_adjusted") in (True, "True"))
        cell["contrast"].append(row["contrast"])
        cell["invalid"] += int(not row["top_token_is_choice"])
        cell["non_finite"] += int(not row["finite"])
        if row["realized_amplitude"] is not None:
            cell["realized"].append(row["realized_amplitude"])

    per_dose = []
    for (layer, family, matching, dose), cell in sorted(cells.items()):
        per_dose.append({
            "layer": layer,
            "family": family,
            "matching": matching,
            "dose": dose,
            "n_trials": cell["n"],
            "accuracy_raw": cell["raw"] / cell["n"],
            "accuracy_adjusted": cell["adjusted"] / cell["n"],
            "mean_contrast": float(np.mean(cell["contrast"])),
            "mean_realized_amplitude": (float(np.mean(cell["realized"]))
                                        if cell["realized"] else None),
            "invalid_rate": cell["invalid"] / cell["n"],
            "non_finite_rate": cell["non_finite"] / cell["n"],
            "tie_rate": cell["ties"] / cell["n"],
        })

    # S = (L when A is targeted - L when B is targeted) / 2, over the two trials that
    # share everything but the targeted sentence (doc 5.8).
    paired = defaultdict(lambda: {"n": 0, "sum": 0.0})
    for row in trials:
        if row["kind"] != "perturbed":
            continue
        key = (row["layer"], row["family"], row["matching"], row["dose"], row["direction_id"],
               row["pair_id"], row["order"], row["label_order"])
        entry = paired[key]
        entry["n"] += 1
        entry["sum"] += row["contrast"] if row["target_label"] == "A" else -row["contrast"]
    contrast_by_curve = defaultdict(list)
    for key, entry in paired.items():
        if entry["n"] == 2:
            contrast_by_curve[key[:3]].append(entry["sum"] / 2.0)

    curves = []
    keys = sorted({(row["layer"], row["family"], row["matching"]) for row in per_dose})
    for layer, family, matching in keys:
      points = sorted(
          (row for row in per_dose
           if (row["layer"], row["family"], row["matching"]) == (layer, family, matching)),
          key=lambda row: row["dose"])
      # The raw score is pinned by the model's A/B response bias, so both metrics are
      # fitted and reported: the sham-adjusted one is what isolates the intervention.
      for metric in ("raw", "adjusted"):
        doses = [row["dose"] for row in points]
        totals = [row["n_trials"] for row in points]
        accuracies = [row[f"accuracy_{metric}"] for row in points]
        corrects = [int(round(a * n)) for a, n in zip(accuracies, totals)]

        beta0 = beta1 = None
        converged = False
        if len(doses) >= 3:
            beta0, beta1, converged = fit_psychometric(doses, corrects, totals)
        bracket = bracketing_doses(doses, accuracies)
        threshold = threshold_at_75(beta0, beta1) if converged else None
        # Doc 5.10: no threshold is reported without a bracketed transition, and none
        # is extrapolated outside the tested range.
        reportable = (threshold is not None and bracket is not None
                      and min(doses) <= threshold <= max(doses))
        contrasts = contrast_by_curve.get((layer, family, matching), [])
        curves.append({
            "layer": layer,
            "family": family,
            "matching": matching,
            "metric": metric,
            "doses": doses,
            "accuracy": accuracies,
            "n_trials": totals,
            "beta0": beta0,
            "beta1": beta1,
            "fit_converged": converged,
            "bracketing_doses": list(bracket) if bracket else None,
            "threshold_75": threshold if reportable else None,
            "threshold_reportable": reportable,
            "mean_localization_contrast": float(np.mean(contrasts)) if contrasts else None,
        })

    sham_rows = [row for row in trials if row["kind"] == "sham"]
    sham_summary = {
        "n_trials": len(sham_rows),
        "mean_contrast": float(np.mean([r["contrast"] for r in sham_rows])) if sham_rows else None,
        "share_prefers_a": (float(np.mean([r["contrast"] > 0 for r in sham_rows]))
                            if sham_rows else None),
        "invalid_rate": (float(np.mean([not r["top_token_is_choice"] for r in sham_rows]))
                         if sham_rows else None),
    }
    return per_dose, curves, sham_summary


def plot_curves(curves, output_dir):
    """One accuracy-vs-dose figure per layer, alpha and z side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"concept": "#1f77b4", "random": "#ff7f0e", "noise": "#2ca02c", "dropout": "#d62728"}
    metrics = ("raw", "adjusted")
    for layer in sorted({row["layer"] for row in curves}):
        figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
        for row_index, metric in enumerate(metrics):
            for column, matching in enumerate(MATCHINGS):
                axis = axes[row_index][column]
                for row in curves:
                    if (row["layer"], row["matching"], row["metric"]) != (layer, matching, metric):
                        continue
                    color = colors.get(row["family"], "gray")
                    axis.plot(row["doses"], row["accuracy"], marker="o",
                              color=color, label=row["family"])
                    if row["threshold_75"] is not None:
                        axis.axvline(row["threshold_75"], color=color, linestyle=":", alpha=0.6)
                axis.axhline(0.5, color="gray", linestyle=":")
                axis.axhline(0.75, color="gray", linestyle="--", alpha=0.5)
                axis.set_xscale("log")
                axis.set_ylim(0.0, 1.0)
                axis.set_title(f"{matching}-matched, {metric} accuracy")
                axis.grid(True, alpha=0.3)
                if row_index == 1:
                    axis.set_xlabel("alpha per targeted token" if matching == "alpha"
                                    else "z = alpha / s(layer, direction)")
                handles, labels = axis.get_legend_handles_labels()
                unique = dict(zip(labels, handles))
                if unique:
                    axis.legend(unique.values(), unique.keys(), fontsize=8)
            axes[row_index][0].set_ylabel(f"2AFC accuracy ({metric})")
        figure.suptitle(f"Experiment 1 - layer {layer}")
        figure.tight_layout()
        path = output_dir / f"curves_layer{layer}.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {path}", flush=True)


# --------------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------------

def prepare_prompts(tokenizer, args, device):
    """One prompt per (pair, presentation order, label order)."""
    prompts = {}
    for pair in args.pairs:
        for order in (0, 1):
            first, second = ((pair["sentence_x"], pair["sentence_y"]) if order == 0
                             else (pair["sentence_y"], pair["sentence_x"]))
            for label_order in args.label_orders:
                prompt, ranges, labels, encoding = build_localization_prompt(
                    tokenizer, first, second, label_order)
                prompts[(pair["pair_id"], order, label_order)] = {
                    "prompt": prompt,
                    "ranges": ranges,
                    "labels": labels,
                    "encoding": {k: v.to(device) for k, v in encoding.items()},
                    # What the injection actually covers: the sentence-final token can
                    # carry the newline that follows it, and that is worth auditing.
                    "targeted_text": [
                        tokenizer.decode(encoding["input_ids"][0][start:end])
                        for start, end in ranges
                    ],
                }
    return prompts


def perturbation_kwargs(calibration, args, family, direction_id, matching, dose, layer,
                        token_range, sham, target_index, vectors, trial_seed):
    """Turn a dose into this family's injection parameters (doc 5.5 and 5.9)."""
    kwargs = {"seed": trial_seed}
    rate = None
    if family in FIXED_DIRECTION_FAMILIES:
        alpha = (float(dose) if matching == "alpha"
                 else float(dose) * calibration.scale(direction_id))
        kwargs.update(vector=vectors[direction_id], alpha=alpha)
    elif family == "noise":
        num_tokens = token_range[1] - token_range[0]
        delta, used = noise_delta(calibration, layer, num_tokens, matching, dose, trial_seed)
        alpha = float(np.mean([amplitude for _, amplitude in used]))
        kwargs.update(delta=delta)
    elif family == "dropout":
        if args.dropout_norm_source == "trial":
            token_norm = sham["token_norms"][(layer, target_index)]
        else:
            token_norm = calibration.token_norm(layer)
        alpha = (float(dose) if matching == "alpha"
                 else float(dose) * calibration.reference_scale(layer))
        rate = dropout_rate_for_amplitude(alpha, token_norm)
        kwargs.update(rate=rate)
    else:
        raise ValueError(f"unknown family {family!r}")
    return kwargs, alpha, rate


def run_experiment(model, tokenizer, calibration, args):
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    token_a, token_b = answer_token_ids(tokenizer)
    print(f"Answer token ids: A={token_a} ({tokenizer.convert_ids_to_tokens([token_a])[0]!r}), "
          f"B={token_b} ({tokenizer.convert_ids_to_tokens([token_b])[0]!r})", flush=True)

    conditions = build_conditions(calibration, args)
    perturbed_passes, _ = count_forward_passes(conditions, args)
    prompts = prepare_prompts(tokenizer, args, device)

    sample = prompts[next(iter(prompts))]
    print(f"Prompt tail: {sample['prompt'][-60:]!r}", flush=True)
    print(f"Perturbed spans: {sample['targeted_text']}", flush=True)

    print(f"Running {len(prompts)} shams...", flush=True)
    shams = {}
    for key, prompt in tqdm(prompts.items(), disable=not args.progress):
        shams[key] = run_sham(model, prompt["encoding"], prompt["ranges"],
                              token_a, token_b, args.layers)

    trials = []
    for (pair_id, order, label_order), sham in shams.items():
        trials.append({
            "kind": "sham", "layer": None, "family": "sham", "direction_id": None,
            "matching": None, "dose": None, "alpha_requested": 0.0, "dropout_rate": None,
            "pair_id": pair_id, "order": order, "label_order": label_order,
            "target_index": None, "target_label": None,
            "logit_a": sham["logit_a"], "logit_b": sham["logit_b"],
            "contrast": sham["contrast"], "sham_contrast": sham["contrast"],
            "adjusted_contrast": 0.0, "correct_raw": None, "correct_adjusted": None,
            "tie_adjusted": None,
            "realized_amplitude": 0.0, "clean_token_norm": None,
            "top_token_is_choice": sham["top_token"] in (token_a, token_b),
            "finite": sham["finite"],
        })

    contamination_records = []
    contamination_budget = args.contamination_trials
    contamination_seen = set()

    planned = (perturbed_passes if args.limit_trials is None
               else min(perturbed_passes, args.limit_trials))
    progress = tqdm(total=planned, disable=not args.progress)
    executed = 0

    for layer in args.layers:
        vectors = {}
        for family, direction_id in conditions[layer]:
            if family in FIXED_DIRECTION_FAMILIES:
                vectors[direction_id] = calibration.vector(direction_id).reshape(-1).to(
                    device=device, dtype=model_dtype)

        for family, direction_id in conditions[layer]:
            for matching in args.matchings:
                for dose in args.dose_grid[matching]:
                    for pair in args.pairs:
                        for order in (0, 1):
                            for label_order in args.label_orders:
                                key = (pair["pair_id"], order, label_order)
                                prompt, sham = prompts[key], shams[key]
                                for target_index in (0, 1):
                                    if (args.limit_trials is not None
                                            and executed >= args.limit_trials):
                                        progress.close()
                                        return trials, contamination_records

                                    token_range = prompt["ranges"][target_index]
                                    target_label = prompt["labels"][target_index]
                                    trial_seed = derive_seed(
                                        args.seed, layer, family, direction_id, matching,
                                        f"{dose:.6g}", pair["pair_id"], order, label_order,
                                        target_index)
                                    kwargs, alpha, rate = perturbation_kwargs(
                                        calibration, args, family, direction_id, matching,
                                        dose, layer, token_range, sham, target_index,
                                        vectors, trial_seed)
                                    if "delta" in kwargs:
                                        kwargs["delta"] = kwargs["delta"].to(device)

                                    result = run_trial(
                                        model, prompt["encoding"], layer, token_range, family,
                                        token_a, token_b, measure=args.measure_amplitude,
                                        **kwargs)

                                    adjusted = result["contrast"] - sham["contrast"]
                                    correct_raw = forced_choice_score(
                                        result["contrast"], target_label)
                                    correct_adjusted = forced_choice_score(
                                        adjusted, target_label)
                                    trials.append({
                                        "kind": "perturbed",
                                        "layer": layer, "family": family,
                                        "direction_id": direction_id, "matching": matching,
                                        "dose": float(dose), "alpha_requested": alpha,
                                        "dropout_rate": rate,
                                        "pair_id": pair["pair_id"], "order": order,
                                        "label_order": label_order,
                                        "target_index": target_index,
                                        "target_label": target_label,
                                        "logit_a": result["logit_a"],
                                        "logit_b": result["logit_b"],
                                        "contrast": result["contrast"],
                                        "sham_contrast": sham["contrast"],
                                        "adjusted_contrast": adjusted,
                                        "correct_raw": correct_raw,
                                        "correct_adjusted": correct_adjusted,
                                        "tie_adjusted": adjusted == 0.0,
                                        "realized_amplitude": result["realized_amplitude"],
                                        "clean_token_norm": result["clean_token_norm"],
                                        "top_token_is_choice": (
                                            result["top_token"] in (token_a, token_b)),
                                        "finite": result["finite"],
                                    })
                                    executed += 1
                                    progress.update(1)

                                    # One diagnostic per (layer, family, matching), at the
                                    # largest dose, where contamination is worst.
                                    diagnostic_key = (layer, family, matching)
                                    if (contamination_budget > 0 and target_index == 0
                                            and dose == args.dose_grid[matching][-1]
                                            and diagnostic_key not in contamination_seen):
                                        contamination_seen.add(diagnostic_key)
                                        other_range = prompt["ranges"][1]
                                        sham_blocks = capture_sham_blocks(
                                            model, prompt["encoding"], other_range, args.layers)
                                        ratios = contamination_ratio(
                                            model, prompt["encoding"], layer, token_range,
                                            other_range, family, sham_blocks, args.layers,
                                            **kwargs)
                                        contamination_records.append({
                                            "layer": layer, "family": family,
                                            "direction_id": direction_id, "matching": matching,
                                            "dose": float(dose), "pair_id": pair["pair_id"],
                                            "order": order, "label_order": label_order,
                                            "ratios": {str(k): v for k, v in sorted(ratios.items())},
                                        })
                                        contamination_budget -= 1

    progress.close()
    return trials, contamination_records


# --------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------

NUMERIC_FIELDS = ("dose", "alpha_requested", "dropout_rate", "logit_a", "logit_b",
                  "contrast", "sham_contrast", "adjusted_contrast",
                  "realized_amplitude", "clean_token_norm")
INTEGER_FIELDS = ("layer", "pair_id", "order", "target_index")
BOOLEAN_FIELDS = ("top_token_is_choice", "finite", "tie_adjusted")


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
            # Scores are recomputed from the contrasts rather than read back, so the
            # scoring rule lives in one place and an older run is re-scored correctly.
            if record["kind"] == "perturbed":
                record["correct_raw"] = forced_choice_score(
                    record["contrast"], record["target_label"])
                record["correct_adjusted"] = forced_choice_score(
                    record["adjusted_contrast"], record["target_label"])
                record["tie_adjusted"] = record["adjusted_contrast"] == 0.0
            else:
                record["correct_raw"] = record["correct_adjusted"] = None
                record["tie_adjusted"] = None
            trials.append(record)
    return trials


def reanalyze(run_dir, plots=True):
    """Recompute the summary and figures from an existing run, without the model.

    Scoring rules and fits change more often than the trials do; a sweep costs GPU
    hours and its raw records are complete, so re-reading them is the cheap path.
    """
    run_dir = Path(run_dir)
    trials = load_trials(run_dir / "trials.csv")
    per_dose, curves, sham_summary = summarize(trials)

    summary_path = run_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    summary.update({"n_trials": len(trials), "sham": sham_summary,
                    "per_dose": per_dose, "curves": curves, "reanalyzed": True})
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"Rewrote {summary_path} from {len(trials)} trials", flush=True)
    if plots:
        plot_curves(curves, run_dir)
    return curves


def parse_doses(explicit, bounds, count, default):
    if explicit:
        doses = [float(value) for value in explicit]
    elif bounds:
        doses = log_grid(float(bounds[0]), float(bounds[1]), count)
    else:
        doses = list(default)
    if any(dose <= 0 for dose in doses):
        raise ValueError("doses must be strictly positive; the sham carries the null dose")
    return sorted(doses)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Experiment 1: 2AFC localization at matched alpha and matched z")
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
                        help="Scale used by the z matching. sd is the protocol's primary "
                             "analysis; mad is the sensitivity analysis of doc 5.10.")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Subset of the calibrated layers (default: all of them).")
    parser.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    parser.add_argument("--matchings", nargs="+", default=list(MATCHINGS), choices=list(MATCHINGS))
    parser.add_argument("--concepts", nargs="+", default=None,
                        help="Concept directions to test (default: the calibrated ones).")
    parser.add_argument("--num_random", type=int, default=2,
                        help="Fixed random directions per layer, taken from the bank.")
    parser.add_argument("--num_noise", type=int, default=1,
                        help="Noise realizations; each draws a bank direction per token.")
    parser.add_argument("--num_dropout", type=int, default=1,
                        help="Dropout realizations; each draws a mask per token.")
    parser.add_argument("--num_pairs", type=int, default=5,
                        help="Length-matched sentence pairs, each sentence used at most once.")
    parser.add_argument("--label_orders", nargs="+", default=["AB"], choices=["AB", "BA"],
                        help="Visible-label assignment. Add BA to separate a position "
                             "preference from a preference for the letter A.")
    parser.add_argument("--alpha_doses", nargs="+", default=None)
    parser.add_argument("--alpha_range", nargs=2, default=None, metavar=("LOW", "HIGH"))
    parser.add_argument("--z_doses", nargs="+", default=None)
    parser.add_argument("--z_range", nargs=2, default=None, metavar=("LOW", "HIGH"))
    parser.add_argument("--num_doses", type=int, default=7,
                        help="Doses per grid when a range is given (doc 5.5).")
    parser.add_argument("--dropout_norm_source", choices=["trial", "calibration"], default="trial",
                        help="Activation norm turning an amplitude into a rate: the targeted "
                             "tokens of the trial itself, or h_bar(layer) from the "
                             "calibration. Experiment 0 does not record h_bar, so only "
                             "trial is currently available.")
    parser.add_argument("--contamination_trials", type=int, default=0,
                        help="Number of P(1->2) diagnostics to run (doc 5.13); 0 disables them.")
    parser.add_argument("--measure_amplitude", action="store_true", default=True)
    parser.add_argument("--no_measure_amplitude", dest="measure_amplitude", action="store_false")
    parser.add_argument("--limit_trials", type=int, default=None,
                        help="Stop after this many perturbed trials (smoke tests).")
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--model", default=None, help="Defaults to the calibration model.")
    parser.add_argument("--output_dir", default="results/experiment1")
    parser.add_argument("--run_name", default="pilot")
    parser.add_argument("--no_progress", dest="progress", action="store_false", default=True)
    parser.add_argument("--no_plots", dest="plots", action="store_false", default=True)
    parser.add_argument("--dry_run", action="store_true",
                        help="Build the plan and print its size without loading the model.")
    parser.add_argument("--reanalyze", default=None, metavar="RUN_DIR",
                        help="Recompute summary.json and the figures from an existing "
                             "trials.csv, without loading the model.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.reanalyze:
        curves = reanalyze(args.reanalyze, plots=args.plots)
        print(f"\n{'layer':>6} {'family':>9} {'matching':>9} {'metric':>9} {'acc@peak':>9} "
              f"{'threshold75':>12}", flush=True)
        for row in curves:
            threshold = f"{row['threshold_75']:.4g}" if row["threshold_75"] is not None else ""
            print(f"{row['layer']:>6} {row['family']:>9} {row['matching']:>9} "
                  f"{row['metric']:>9} {max(row['accuracy']):>9.0%} {threshold:>12}", flush=True)
        return

    calibration = DirectionBank.load(
        args.calibration_config,
        calibration_dir=args.calibration_dir,
        estimator=args.estimator,
        strict=not args.allow_calibration_mismatch,
    )
    calibrated_layers = calibration.layers
    args.layers = args.layers or list(calibrated_layers)
    missing = [layer for layer in args.layers if layer not in calibrated_layers]
    if missing:
        parser.error(f"decoder blocks {missing} are not calibrated in "
                     f"{calibration.calibration_dir}")
    args.concepts = args.concepts or list(calibration.concepts)
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

    model_name = args.model or calibration.model_name
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir = output_dir / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72, flush=True)
    print("EXPERIMENT 1: 2AFC LOCALIZATION, ALPHA- AND Z-MATCHED PSYCHOMETRICS", flush=True)
    print("=" * 72, flush=True)
    print(f"Model:       {model_name}", flush=True)
    print(f"Calibration: {calibration.calibration_dir} "
          f"(config={args.calibration_config}, estimator={calibration.estimator})", flush=True)
    print(f"Layers:      {args.layers}", flush=True)
    print(f"Families:    {args.families}", flush=True)
    print(f"Matchings:   {args.matchings}", flush=True)
    print(f"Alpha doses: {[round(dose, 4) for dose in args.dose_grid['alpha']]}", flush=True)
    print(f"z doses:     {[round(dose, 4) for dose in args.dose_grid['z']]}", flush=True)
    print(f"Output:      {output_dir}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    args.pairs = build_sentence_pairs(tokenizer, list(LOCALIZATION_SENTENCES),
                                      args.num_pairs, args.seed)
    print(f"Pairs:       {len(args.pairs)} (largest token-length gap "
          f"{max(pair['token_length_gap'] for pair in args.pairs)})", flush=True)

    conditions = build_conditions(calibration, args)
    perturbed_passes, sham_passes = count_forward_passes(conditions, args)
    print(f"Forward passes: {perturbed_passes} perturbed + {sham_passes} sham", flush=True)
    # Rebuild every fixed direction now: a missing concept vector or a seed that no
    # longer reproduces must fail here, not once the weights are on the GPU.
    try:
        checked = calibration.preflight(args.families, args.layers, concepts=args.concepts,
                                        num_random=args.num_random)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(f"Directions:  {checked} rebuilt and matched to their calibrated norms",
          flush=True)
    if args.dry_run:
        print("Dry run: plan only, no model loaded.", flush=True)
        return

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    print("Model loaded.", flush=True)

    trials, contamination = run_experiment(model, tokenizer, calibration, args)
    per_dose, curves, sham_summary = summarize(trials)

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
        "concepts": args.concepts,
        "dose_grid": args.dose_grid,
        "label_orders": args.label_orders,
        "dropout_norm_source": args.dropout_norm_source,
        "pairs": args.pairs,
        "n_trials": len(trials),
        "sham": sham_summary,
        "per_dose": per_dose,
        "curves": curves,
        "contamination": contamination,
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"\nSaved {len(trials)} trials to {trials_path}", flush=True)
    print(f"Saved summary to {summary_path}", flush=True)

    print("\nThresholds (blank when 75% is not bracketed in the tested range):", flush=True)
    print(f"{'layer':>6} {'family':>9} {'matching':>9} {'metric':>9} {'acc@peak':>9} "
          f"{'threshold75':>12} {'bracket':>20}", flush=True)
    for row in curves:
        threshold = f"{row['threshold_75']:.4g}" if row["threshold_75"] is not None else ""
        bracket = (f"[{row['bracketing_doses'][0]:.3g}, {row['bracketing_doses'][1]:.3g}]"
                   if row["bracketing_doses"] else "")
        print(f"{row['layer']:>6} {row['family']:>9} {row['matching']:>9} {row['metric']:>9} "
              f"{max(row['accuracy']):>9.0%} {threshold:>12} {bracket:>20}", flush=True)

    if args.plots:
        plot_curves(curves, output_dir)

    print("\nEXPERIMENT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
