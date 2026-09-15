"""Is s(l,v) estimated on a token distribution that Experiment 1 never perturbs?

Experiment 0 renders each sentence ALONE (`template: "{sentence}"`,
`add_special_tokens=False`), so the sentence's first token is position 0 of its own
sequence -- the attention-sink slot, where Llama parks a massive activation, and which
`position_policy: all_sentence_tokens` includes. Experiment 1 embeds the same sentences
deep inside the chat-templated 2AFC prompt, where no perturbed token is ever at
position 0.

This measures the projections in both contexts, on the same sentences and the same
directions, and compares the resulting scales. Run it with
`jobs/probe_calibration_context.sbatch`.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "code/utils"))
from all_prompts import LOCALIZATION_SENTENCES                        # noqa: E402
from experiments.experiment1_psychometrics import build_localization_prompt  # noqa: E402

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BLOCKS = [1, 5, 9, 13]
CONCEPTS = ["Dust", "Satellites", "recursion", "shutdown"]
# Mirrors configs/experiment_0_calibration/development_full.yaml: directions.fixed_random
RANDOM_BASE_SEED = 2026091001
RANDOM_LAYER_STRIDE = 100000
NUM_RANDOM = 3
MAD_GAUSSIAN_CONSISTENCY = 1.4826
OUT_DIR = REPO_ROOT / "results/experiment1/context_probe"


def unit(vector):
    array = vector.float().numpy().ravel()
    return array / np.linalg.norm(array)


def build_directions(hidden_size):
    """The concept vectors Experiment 0 calibrated on, plus its fixed-random bank."""
    directions = {}
    for block in BLOCKS:
        entry = {}
        for name in CONCEPTS:
            # HF hidden_states[0] is the embedding output, so decoder block l is l + 1.
            path = REPO_ROOT / f"data/saved_vectors/llama/{name}_{block + 1}_avg.pt"
            entry[f"concept:{name}"] = unit(torch.load(path, map_location="cpu")["vector"])
        for sample in range(NUM_RANDOM):
            rng = np.random.RandomState(
                RANDOM_BASE_SEED + block * RANDOM_LAYER_STRIDE + sample)
            vector = rng.randn(hidden_size)
            entry[f"random:{sample}"] = vector / np.linalg.norm(vector)
        directions[block] = entry
    return directions


def capture(model, input_ids, spans):
    """Activations at `spans` (list of (start, end)) for every block in BLOCKS."""
    store = {}
    handles = []

    def make_hook(index):
        def hook(_module, _inputs, output):
            store[index] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook

    for index in BLOCKS:
        handles.append(model.model.layers[index].register_forward_hook(make_hook(index)))
    try:
        with torch.no_grad():
            model(input_ids=input_ids.to(next(model.parameters()).device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return {index: np.concatenate(
        [store[index][0, start:end, :].float().cpu().numpy() for start, end in spans], 0)
        for index in BLOCKS}


def describe(projections, positions):
    """SD, corrected MAD and the tail diagnostics of one direction in one context."""
    sd = float(projections.std(ddof=1))
    median = float(np.median(projections))
    mad = float(MAD_GAUSSIAN_CONSISTENCY
                * np.median(np.abs(projections - median)))
    mean = float(projections.mean())
    extreme = float(np.percentile(projections, 5 if mean < median else 95))
    at_zero = positions == 0
    return {
        "sd": sd, "mad": mad, "ratio": sd / mad, "mean": mean, "median": median,
        # Tail mass under a "bulk near the median plus a fraction at the extreme" model.
        "tail_mass": abs(mean - median) / (abs(extreme - median) + 1e-12),
        "share_pos0": float(at_zero.mean()),
        "max_abs_at_pos0": (float(np.abs(projections[at_zero]).max())
                            if at_zero.any() else None),
        "max_abs_elsewhere": float(np.abs(projections[~at_zero]).max()),
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    directions = build_directions(model.config.hidden_size)

    # Context A: the isolated sentence, exactly as Experiment 0 renders it.
    isolated = {index: [] for index in BLOCKS}
    isolated_positions = {index: [] for index in BLOCKS}
    for sentence in LOCALIZATION_SENTENCES:
        ids = tokenizer(sentence, return_tensors="pt",
                        add_special_tokens=False)["input_ids"]
        captured = capture(model, ids, [(0, ids.shape[1])])
        for index in BLOCKS:
            isolated[index].append(captured[index])
            isolated_positions[index].append(np.arange(ids.shape[1]))

    # Context B: the real 2AFC prompt, over the span Experiment 1 actually perturbs.
    prompt = {index: [] for index in BLOCKS}
    prompt_positions = {index: [] for index in BLOCKS}
    pairs = [(LOCALIZATION_SENTENCES[k], LOCALIZATION_SENTENCES[(k + 1) % 100])
             for k in range(0, len(LOCALIZATION_SENTENCES), 2)]
    for first, second in pairs:
        _, ranges, _, encoding = build_localization_prompt(tokenizer, first, second, "AB")
        captured = capture(model, encoding["input_ids"], ranges)
        for index in BLOCKS:
            prompt[index].append(captured[index])
            prompt_positions[index].append(
                np.concatenate([np.arange(start, end) for start, end in ranges]))

    results = {}
    header = (f"{'blk':>3} {'direction':>18} {'context':>9} {'SD':>10} {'MAD':>9} "
              f"{'SD/MAD':>9} {'|mean|':>9} {'tail':>8} {'@pos0':>7}")
    print("\n" + header, flush=True)
    for block in BLOCKS:
        contexts = {
            "isolated": (np.concatenate(isolated[block], 0),
                         np.concatenate(isolated_positions[block])),
            "prompt": (np.concatenate(prompt[block], 0),
                       np.concatenate(prompt_positions[block])),
        }
        for name, vector in directions[block].items():
            row = {}
            for context_name, (activations, positions) in contexts.items():
                stats = describe(activations @ vector, positions)
                row[context_name] = stats
                print(f"{block:>3} {name:>18} {context_name:>9} {stats['sd']:10.3f} "
                      f"{stats['mad']:9.4f} {stats['ratio']:9.1f} "
                      f"{abs(stats['mean']):9.3f} {stats['tail_mass']:8.3f} "
                      f"{stats['share_pos0']:7.3f}", flush=True)
            results[f"{block}|{name}"] = row
        print(flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUT_DIR / "context_probe.json"
    destination.write_text(json.dumps(results, indent=1))
    print("wrote", destination, flush=True)


if __name__ == "__main__":
    main()
