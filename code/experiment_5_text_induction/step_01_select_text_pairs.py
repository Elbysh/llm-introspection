"""Section 9.5, step 1: fixed neutral/evocative text pairs with a shared companion."""

import hashlib


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_text_pairs(specifications, corpus):
    """Review is a human declaration, not an automatic semantic classifier.

    The neutral target and companion must come from LOCALIZATION_SENTENCES.
    Save the authored variant verbatim; never generate or adjust it after seeing
    the model's reports. Duplicate pair definitions cannot inflate sample size.
    """
    if not specifications:
        raise ValueError("Supply reviewed text pairs before preparation")
    ids, definitions, rows = set(), set(), []
    for spec in specifications:
        if not spec["pair_id"] or spec["pair_id"] in ids:
            raise ValueError("Pair IDs must be nonempty and unique")
        ids.add(spec["pair_id"])
        if spec.get("reviewed") is not True:
            raise ValueError("Review target neutrality, variant evocation and companion neutrality first")
        for key in ("neutral_sentence_index", "companion_sentence_index"):
            index = spec[key]
            if type(index) is not int or index not in range(len(corpus)):
                raise ValueError("Sentence indices must refer to the protocol corpus")
        neutral = corpus[spec["neutral_sentence_index"]]
        companion = corpus[spec["companion_sentence_index"]]
        evocative = spec["evocative_text"]
        if not isinstance(evocative, str) or not evocative.strip() or any(c in evocative for c in "\r\n"):
            raise ValueError("Evocative text must be one nonempty sentence line")
        if len({neutral, companion, evocative}) != 3:
            raise ValueError("Target, variant and companion must be distinct texts")
        definition = (spec["concept"], neutral, evocative, companion)
        if definition in definitions:
            raise ValueError("Duplicate text-pair definition")
        definitions.add(definition)
        rows.append({**spec, "neutral_text": neutral, "companion_text": companion,
                     "neutral_sentence_sha256": text_hash(neutral),
                     "evocative_sentence_sha256": text_hash(evocative),
                     "companion_sentence_sha256": text_hash(companion)})
    return rows
