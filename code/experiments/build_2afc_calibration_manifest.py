"""Emit the 2AFC presentation manifest Experiment 0 calibrates s(l, v) on.

Experiment 0's development configuration renders each sentence alone, so the
sentence's first token sits at position 0 -- the attention-sink slot, where Llama
parks an activation two to three orders of magnitude above every other token.
Experiment 1 never perturbs a token at position 0: its targets sit inside the 2AFC
prompt. Estimating s on the isolated context therefore estimates it on a
distribution the behavioural experiment does not contain (see
docs/livrables/calibration-contexte-probleme-et-correctif.md).

`prepare_material.py` already accepts the fix: its `external_manifest` presentation
mode reads rows shaped as
`{context_id, rendered_text, targets: [{sentence_id, char_start, char_end}]}`, with
several targets per context. Only the manifest itself was missing, and its character
bounds are the ones `build_localization_prompt` computes and then discards.

The plan, following section 8.3 of the note: the 100 corpus sentences paired into 50
length-matched pairs, each pair rendered in both physical orders and both label
orders. Every sentence then appears four times, twice in each slot, so the sample is
balanced over slot and over label and no scale has to be estimated per slot.

    python code/experiments/build_2afc_calibration_manifest.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --output data/experiment_0_calibration/contexts_2afc_llama.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "code/utils"))

from all_prompts import LOCALIZATION_SENTENCES                        # noqa: E402
from experiments.experiment1_psychometrics import (                   # noqa: E402
    build_localization_prompt,
    build_sentence_pairs,
    localization_char_spans,
)

# Experiment 0 names its corpus rows this way; prepare_material.py rejects any other
# sentence_id, which is what keeps the manifest and the protocol corpus in step.
SENTENCE_ID = "localization_{:03d}"


def build_rows(tokenizer, sentences, num_pairs, seed):
    """One manifest row per rendered prompt, two targets each."""
    index_of = {sentence: index for index, sentence in enumerate(sentences)}
    pairs = build_sentence_pairs(tokenizer, sentences, num_pairs, seed)
    rows = []
    for pair in pairs:
        x, y = pair["sentence_x"], pair["sentence_y"]
        for order_index, (first, second) in enumerate(((x, y), (y, x))):
            for label_order in ("AB", "BA"):
                prompt, _, labels, _ = build_localization_prompt(
                    tokenizer, first, second, label_order
                )
                spans = localization_char_spans(prompt, labels, (first, second))
                targets = []
                for sentence, (char_start, char_end) in zip((first, second), spans):
                    # prepare_material.py re-checks this, but a mismatch is far easier
                    # to read here, where the pair and the order are still in hand.
                    if prompt[char_start:char_end] != sentence:
                        raise ValueError(
                            "char bounds do not select the sentence back: "
                            f"pair {pair['pair_id']}, order {order_index}, {label_order}"
                        )
                    targets.append(
                        {
                            "sentence_id": SENTENCE_ID.format(index_of[sentence]),
                            "char_start": char_start,
                            "char_end": char_end,
                        }
                    )
                rows.append(
                    {
                        "context_id": "loc2afc__pair_{:02d}__order_{}__{}".format(
                            pair["pair_id"], order_index, label_order
                        ),
                        "rendered_text": prompt,
                        "targets": targets,
                    }
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Tokenizer whose chat template renders the prompt. It must "
                             "be the model the calibration and Experiment 1 both use.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--num_pairs", type=int, default=50,
                        help="50 disjoint pairs use each of the 100 sentences exactly "
                             "once. The scale is a property of (block, direction), not "
                             "of the behavioural plan's five pairs, so the whole corpus "
                             "is presented rather than only those.")
    parser.add_argument("--seed", type=int, default=20260908,
                        help="Only shuffles the pair order; the pairing itself is "
                             "deterministic in token length.")
    parser.add_argument("--output", default="data/experiment_0_calibration/contexts_2afc_llama.jsonl")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, use_fast=True)
    sentences = list(LOCALIZATION_SENTENCES)
    if len(sentences) != 100 or len(set(sentences)) != 100:
        raise ValueError("the protocol corpus must hold exactly 100 distinct sentences")

    rows = build_rows(tokenizer, sentences, args.num_pairs, args.seed)

    covered = {target["sentence_id"] for row in rows for target in row["targets"]}
    expected = {SENTENCE_ID.format(index) for index in range(len(sentences))}
    if covered != expected:
        raise ValueError(
            "manifest misses {} sentences".format(len(expected - covered))
        )
    context_ids = [row["context_id"] for row in rows]
    if len(context_ids) != len(set(context_ids)):
        raise ValueError("context IDs must be unique")

    # Slot and label balance, the property that lets one scale per (block, direction)
    # stand in for a per-slot scale.
    slots = {}
    for row in rows:
        for target_index, target in enumerate(row["targets"]):
            slots.setdefault(target["sentence_id"], []).append(target_index)
    if any(sorted(v) != [0, 0, 1, 1] for v in slots.values()):
        raise ValueError("each sentence must appear twice in each slot")

    positions = sum(
        len(tokenizer(
            row["rendered_text"][target["char_start"]:target["char_end"]],
            add_special_tokens=False)["input_ids"])
        for row in rows for target in row["targets"]
    )

    output = REPO_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output.relative_to(REPO_ROOT)),
        "model": args.model,
        "n_contexts": len(rows),
        "n_sentences": len(covered),
        "approx_n_positions": positions,
        "prompt_token_length": len(tokenizer(rows[0]["rendered_text"],
                                             add_special_tokens=False)["input_ids"]),
    }, indent=2))


if __name__ == "__main__":
    main()
