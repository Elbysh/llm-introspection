"""Why do concept directions pick up ~60x more of the sink token than random ones?

Every direction is unit-norm, so the ratio cannot come from vector magnitude; it must
come from orientation. This measures (a) the activation norm at position 0 against every
other position, and (b) the cosine between each calibrated direction and the mean
position-0 activation. A random unit vector in R^d has expected |cosine| = 1/sqrt(d),
which is 0.0156 for d = 4096.

Companion to `probe_calibration_context.py`. Run with `jobs/probe_sink_alignment.sbatch`.
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
sys.path.insert(0, str(REPO_ROOT / "code/analysis"))
from all_prompts import LOCALIZATION_SENTENCES                  # noqa: E402
from probe_calibration_context import (BLOCKS, MODEL, OUT_DIR,  # noqa: E402
                                       build_directions, capture)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    directions = build_directions(model.config.hidden_size)
    isotropic = 1.0 / np.sqrt(model.config.hidden_size)

    # The calibration context: each sentence alone, so its first token is at position 0.
    sink = {block: [] for block in BLOCKS}
    other = {block: [] for block in BLOCKS}
    for sentence in LOCALIZATION_SENTENCES:
        ids = tokenizer(sentence, return_tensors="pt",
                        add_special_tokens=False)["input_ids"]
        captured = capture(model, ids, [(0, ids.shape[1])])
        for block in BLOCKS:
            sink[block].append(captured[block][0])
            other[block].append(captured[block][1:])

    results = {}
    print(f"\n{'blk':>3} {'||h_pos0||':>11} {'||h_ailleurs||':>15} {'rapport':>9}", flush=True)
    for block in BLOCKS:
        at_zero = np.stack(sink[block])
        elsewhere = np.concatenate(other[block], 0)
        norm_zero = float(np.linalg.norm(at_zero, axis=1).mean())
        norm_other = float(np.linalg.norm(elsewhere, axis=1).mean())
        results[f"{block}|norms"] = {"pos0": norm_zero, "elsewhere": norm_other,
                                     "ratio": norm_zero / norm_other}
        print(f"{block:>3} {norm_zero:11.1f} {norm_other:15.2f} "
              f"{norm_zero / norm_other:9.1f}", flush=True)

    print(f"\n{'blk':>3} {'direction':>18} {'|cos| with h_pos0':>18} "
          f"{'isotropic expectation':>22}", flush=True)
    for block in BLOCKS:
        axis = np.stack(sink[block]).mean(0)
        axis = axis / np.linalg.norm(axis)
        for name, vector in directions[block].items():
            cosine = abs(float(axis @ vector))
            results[f"{block}|{name}"] = {"abs_cosine": cosine,
                                          "isotropic": float(isotropic)}
            print(f"{block:>3} {name:>18} {cosine:18.4f} {isotropic:22.4f}", flush=True)
        print(flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUT_DIR / "sink_alignment.json"
    destination.write_text(json.dumps(results, indent=1))
    print("wrote", destination, flush=True)


if __name__ == "__main__":
    main()
