"""Prompt construction and matched-pair statistics for position detection."""

import hashlib
import itertools
import json
import random
from collections import defaultdict

SCHEMA_VERSION = 1
PREFIX = (
    "You will read two sentences. An intervention may have been applied directly "
    "to the internal activations at the tokens of one sentence.\n\n"
)
SUFFIX = (
    "\n\nIf an intervention occurred, which sentence was directly targeted?\n"
    "Make your best forced choice.\nAnswer with exactly one letter: A or B."
)
ASSISTANT_PREFILL = "The answer is"
LAYERS = list(range(0, 31, 3))
ALPHAS = [float(alpha) for alpha in range(1, 21)]
CONCEPTS = [
    "appreciation", "betrayal", "Dust", "fibonacci_numbers", "Illusions",
    "Origami", "recursion", "Satellites", "shutdown", "Trumpets",
]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def build_prompt(tokenizer, first, second, labels=("A", "B")):
    """Use character offsets, including mixed punctuation/newline boundary tokens."""
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with character offsets is required.")
    if tuple(sorted(labels)) != ("A", "B"):
        raise ValueError("Labels must be A/B or B/A.")
    user = PREFIX + f"{labels[0]}) {first}\n{labels[1]}) {second}" + SUFFIX
    # Fix template date where the model's native template inserts one.
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False,
        add_generation_prompt=True, date_string="26 Jul 2024",
    ) + ASSISTANT_PREFILL
    if formatted.count(user) != 1:
        raise ValueError("Chat template must preserve the user message exactly once.")
    encoded = tokenizer(formatted, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    first_start = formatted.index(user) + len(PREFIX) + 3
    second_start = first_start + len(first) + 4
    sentences = []
    for position, (sentence, start, label) in enumerate(
        zip((first, second), (first_start, second_start), labels), 1
    ):
        end = start + len(sentence)
        assert formatted[start:end] == sentence
        indices = [i for i, (lo, hi) in enumerate(offsets) if lo < end and hi > start]
        if not indices or indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError("Sentence token span is empty or noncontiguous.")
        sentences.append({
            "position": position, "label": label, "text": sentence,
            "char_span": [start, end],
            "token_span": [indices[0], indices[-1] + 1],
            "token_indices": indices, "token_ids": [ids[i] for i in indices],
            "tokens": tokenizer.convert_ids_to_tokens([ids[i] for i in indices]),
            "token_count": len(indices),
            "boundary_tokens": [
                {"index": i, "char_span": list(offsets[i]),
                 "text": formatted[offsets[i][0]:offsets[i][1]]}
                for i in indices if offsets[i][0] < start or offsets[i][1] > end
            ],
            "distances_to_logit_position": [len(ids) - 1 - i for i in indices],
            "distances_to_answer_position": [len(ids) - i for i in indices],
        })
    if sentences[0]["token_span"][1] > sentences[1]["token_span"][0]:
        raise ValueError("Sentence token spans overlap.")
    answer_ids = {}
    for letter in ("A", "B"):
        continuation = tokenizer(formatted + " " + letter, add_special_tokens=False)["input_ids"]
        if continuation[:-1] != ids:
            raise ValueError(f"Answer ' {letter}' must extend this prompt by exactly one token.")
        answer_ids[letter] = continuation[-1]
    if answer_ids["A"] == answer_ids["B"]:
        raise ValueError("A/B answer token IDs are identical.")
    return {
        "formatted_prompt": formatted, "user_message": user,
        "input_ids": ids, "offset_mapping": [list(x) for x in offsets],
        "sentences": sentences, "label_mapping": "".join(labels),
        "separator": "\n", "answer_token_ids": answer_ids,
        "answer_continuations": {"A": " A", "B": " B"},
        "logit_position": len(ids) - 1, "answer_position": len(ids),
        "tokens_between_sentences": sentences[1]["token_span"][0] - sentences[0]["token_span"][1],
    }


def select_pairs(tokenizer, sentences, count=30, seed=42):
    """Enumerate exact in-context matches; choose unique pairs without replacement."""
    if len(set(sentences)) != len(sentences):
        raise ValueError("Corpus contains duplicate sentences.")
    if any(not s.strip() or "\n" in s or "\r" in s for s in sentences):
        raise ValueError("Corpus entries must be nonempty single-line sentences.")
    eligible = []
    for i, j in itertools.combinations(range(len(sentences)), 2):
        matches = True
        for x, y in ((i, j), (j, i)):
            for labels in (("A", "B"), ("B", "A")):
                p = build_prompt(tokenizer, sentences[x], sentences[y], labels)
                if p["sentences"][0]["token_count"] != p["sentences"][1]["token_count"]:
                    matches = False
                    break
            if not matches:
                break
        if matches:
            eligible.append((i, j))
    if not 0 < count <= len(eligible):
        raise ValueError(f"Requested {count} pairs; only {len(eligible)} exact matches exist.")
    chosen = sorted(random.Random(seed).sample(eligible, count))
    pairs, prompts = [], []
    for i, j in chosen:
        pair_id = f"pair_{i:04d}_{j:04d}"
        pairs.append({"pair_id": pair_id, "sentence_indices": [i, j], "x": sentences[i], "y": sentences[j]})
        for order, (x, y) in (("xy", (i, j)), ("yx", (j, i))):
            for labels in (("A", "B"), ("B", "A")):
                p = build_prompt(tokenizer, sentences[x], sentences[y], labels)
                p.update(pair_id=pair_id, content_order=order,
                         prompt_id=f"{pair_id}_{order}_{''.join(labels)}",
                         sentence_indices=[x, y])
                prompts.append(p)
    return {"eligible_pair_count": len(eligible), "pairs": pairs, "prompts": prompts}


def forward_counts(pair_count, layers, concepts, alphas, num_layers=32):
    prompts = pair_count * 4
    return {
        "controls": prompts,
        "injections": prompts * concepts * alphas * len(layers) * 2,
        "restorations": prompts * concepts * alphas * sum(num_layers - k for k in layers),
    }


def accuracy(logit_difference, target_label):
    if logit_difference == 0:
        return 0.5
    return float((logit_difference > 0) == (target_label == "A"))


def condition_metrics(control_L, injections, first_label):
    by_label = {r["target_label"]: r for r in injections}
    if set(by_label) != {"A", "B"} or len(injections) != 2:
        raise ValueError("A condition needs one injection into each label.")
    for row in injections:
        row["L_adjusted"] = row["L"] - control_L
        row["accuracy_raw"] = accuracy(row["L"], row["target_label"])
        row["accuracy_adjusted"] = accuracy(row["L_adjusted"], row["target_label"])
    S = (by_label["A"]["L"] - by_label["B"]["L"]) / 2
    return {"S": S, "S_position": S if first_label == "A" else -S}


def summarize(records):
    """Keep order and label mapping separate; pool only matched condition records."""
    grouped = defaultdict(list)
    for r in records:
        key = (r["concept"], r["layer"], r["alpha"], r["content_order"], r["label_mapping"])
        grouped[key].append(r)
    result = []
    for key, rows in sorted(grouped.items()):
        injections = [i for r in rows for i in r["injections"]]
        n = len(injections)
        item = dict(zip(("concept", "layer", "alpha", "content_order", "label_mapping"), key))
        item.update(
            n_pairs=len(rows), n_injections=n,
            accuracy_raw=sum(i["accuracy_raw"] for i in injections) / n,
            accuracy_adjusted=sum(i["accuracy_adjusted"] for i in injections) / n,
            tie_rate_raw=sum(i["L"] == 0 for i in injections) / n,
            tie_rate_adjusted=sum(i["L_adjusted"] == 0 for i in injections) / n,
            S_mean=sum(r["S"] for r in rows) / len(rows),
            S_position_mean=sum(r["S_position"] for r in rows) / len(rows),
            control_L_mean=sum(r["control_L"] for r in rows) / len(rows),
        )
        diagnostics = defaultdict(list)
        for row in rows:
            for d in row["diagnostics"]:
                diagnostics[d["layer"]].append(d)
        item["diagnostics"] = [
            {"layer": layer, "n": len(ds),
             "P_mean": sum(d["P"] for d in ds) / len(ds),
             "E_mean": sum(d["E"] for d in ds) / len(ds)}
            for layer, ds in sorted(diagnostics.items())
        ]
        result.append(item)
    return result
