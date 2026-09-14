"""Section 8.5, step 1: select balanced labelled sentences and build both mappings."""

import hashlib


def select_examples(dataset, selections):
    rows = []
    for spec in selections:
        concept = spec["probed_concept"]
        positive, negative = dataset[concept]
        for label, values, indices in ((True, positive, spec["positive_indices"]),
                                       (False, negative, spec["negative_indices"])):
            for index in indices:
                sentence = values[index]
                if not isinstance(sentence, str) or not sentence.strip():
                    raise ValueError("Empty semantic example")
                rows.append({
                    "example_id": f"{concept}__{'positive' if label else 'negative'}__{index:02d}",
                    "probed_concept": concept, "sentence": sentence, "label": label,
                    "source_index": index,
                    "sentence_sha256": hashlib.sha256(sentence.encode()).hexdigest(),
                })
    # Conflicting duplicate labels would invalidate the semantic task.
    seen = {}
    for row in rows:
        key = row["probed_concept"], row["sentence"]
        if key in seen:
            raise ValueError("Duplicate or contradictory example within a probed concept")
        seen[key] = row["label"]
    return rows


def build_classification_prompt(tokenizer, example, yes_label):
    """Exact section 8.3 wording, no affirmative assistant prefill.

    Character offsets select only sentence tokens. Both answer candidates must
    extend the rendered prompt by exactly one token; never truncate a tokenization.
    """
    if yes_label not in ("X", "Y"):
        raise ValueError("yes_label must be X or Y")
    meanings = {yes_label: "yes", "Y" if yes_label == "X" else "X": "no"}
    prefix = "Read the following sentence.\n\n"
    user = (prefix + example["sentence"] + "\n\n"
            f"Does this sentence express the concept {example['probed_concept']}?\n"
            f"X means {meanings['X']}.\nY means {meanings['Y']}.\n"
            "Answer with exactly one letter: X or Y.")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False,
        add_generation_prompt=True, date_string="26 Jul 2024",
    )
    if rendered.count(user) != 1:
        raise ValueError("Chat template must preserve the exact user message")
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    start = rendered.index(user) + len(prefix)
    end = start + len(example["sentence"])
    indices = [i for i, (lo, hi) in enumerate(offsets) if hi > lo and lo < end and hi > start]
    if not indices or indices != list(range(indices[0], indices[-1] + 1)) or indices[-1] >= len(ids)-1:
        raise ValueError("Invalid sentence token span")
    answers = {}
    for label in ("X", "Y"):
        continuation = tokenizer(rendered + label, add_special_tokens=False)["input_ids"]
        if continuation[:-1] != ids:
            raise ValueError(f"{label} must be a single continuation token in this prompt")
        answers[label] = continuation[-1]
    if answers["X"] == answers["Y"]:
        raise ValueError("Answer tokens must be distinct")
    correct = yes_label if example["label"] else ("Y" if yes_label == "X" else "X")
    return {**example, "prompt_id": f"{example['example_id']}__yes_{yes_label}",
            "yes_label": yes_label, "correct_label": correct,
            "rendered_text": rendered, "input_ids": ids, "answer_token_ids": answers,
            "target_token_indices": indices, "target_char_span": [start, end],
            "target_token_ids": [ids[i] for i in indices],
            "target_token_offsets": [list(offsets[i]) for i in indices],
            "distances_to_answer": [len(ids)-i for i in indices]}
