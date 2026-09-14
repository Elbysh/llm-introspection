"""Section 9.5, step 4: both physical A/B targets and both X/Y presence mappings."""

PREFIX = ("You will read two sentences. An intervention may or may not have been applied\n"
          "directly to the internal activations at the tokens of one sentence.\n\n")


def build_presence_prompt(tokenizer, pair, text_version, target_label, present_label):
    """Only the target text changes; no prompt mentions a variant or a concept name."""
    if text_version not in ("neutral", "evocative") or target_label not in ("A", "B") or present_label not in ("X", "Y"):
        raise ValueError("Invalid text version, target position or response mapping")
    absent_label = "Y" if present_label == "X" else "X"
    target = pair[f"{text_version}_text"]
    sentences = [target, pair["companion_text"]] if target_label == "A" else [pair["companion_text"], target]
    user = PREFIX + f"A) {sentences[0]}\nB) {sentences[1]}\n\n"
    user += "Was an intervention applied while you processed these sentences?\n"
    meanings = {present_label: "an intervention occurred", absent_label: "no intervention occurred"}
    user += f"X means that {meanings['X']}.\nY means that {meanings['Y']}.\n"
    user += "Answer with exactly one letter: X or Y."
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False,
        add_generation_prompt=True, date_string="26 Jul 2024")
    if rendered.count(user) != 1:
        raise ValueError("Chat template must preserve the exact user content")
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    first_start = rendered.index(user) + len(PREFIX) + 3
    starts = (first_start, first_start + len(sentences[0]) + 4)
    spans = []
    for sentence, start in zip(sentences, starts):
        end = start + len(sentence)
        positions = [i for i, (lo, hi) in enumerate(offsets) if hi > lo and lo < end and hi > start]
        if not positions or positions != list(range(positions[0], positions[-1]+1)) or positions[-1] >= len(ids)-1:
            raise ValueError("Invalid sentence token span")
        spans.append({"text": sentence, "char_span": [start, end], "token_indices": positions,
                      "token_ids": [ids[i] for i in positions],
                      "token_offsets": [list(offsets[i]) for i in positions]})
    if set(spans[0]["token_indices"]) & set(spans[1]["token_indices"]):
        raise ValueError("Sentence token spans overlap")
    answers = {}
    for label in ("X", "Y"):
        extended = tokenizer(rendered + label, add_special_tokens=False)["input_ids"]
        if extended[:-1] != ids:
            raise ValueError("Presence labels must each be exactly one continuation token")
        answers[label] = extended[-1]
    if answers["X"] == answers["Y"]:
        raise ValueError("Presence labels share a token ID")
    targeted = spans[0 if target_label == "A" else 1]["token_indices"]
    return {"prompt_id": f"{pair['pair_id']}__{text_version}__target_{target_label}__present_{present_label}",
            "pair_id": pair["pair_id"], "concept": pair["concept"],
            "text_version": text_version, "target_label": target_label,
            "present_label": present_label, "absent_label": absent_label,
            "rendered_text": rendered, "input_ids": ids, "answer_token_ids": answers,
            "sentence_spans": spans, "target_token_indices": targeted,
            "target_token_count": len(targeted),
            "tokens_between_sentences": spans[1]["token_indices"][0]-spans[0]["token_indices"][-1]-1,
            "distances_to_answer": [len(ids)-i for i in targeted],
            **{key: pair[key] for key in ("neutral_sentence_sha256", "evocative_sentence_sha256", "companion_sentence_sha256")}}


def counterbalanced_prompts(tokenizer, pairs, max_length_difference):
    prompts = []
    for pair in pairs:
        for target in ("A", "B"):
            for mapping in ("X", "Y"):
                versions = [build_presence_prompt(tokenizer, pair, version, target, mapping)
                            for version in ("neutral", "evocative")]
                difference = versions[1]["target_token_count"] - versions[0]["target_token_count"]
                if max_length_difference is not None and abs(difference) > max_length_difference:
                    raise ValueError(f"Pair {pair['pair_id']} exceeds the configured token-length difference")
                for prompt in versions:
                    prompt["evocative_minus_neutral_token_count"] = difference
                prompts.extend(versions)
    return prompts
