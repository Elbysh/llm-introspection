"""Section 9.6: presence rates, paired textual effects and clustered uncertainty."""

from collections import defaultdict

import numpy as np


def auroc(labels, scores):
    """Rank AUROC; ties receive half credit, without selecting a decision threshold."""
    labels, scores = np.asarray(labels, dtype=bool), np.asarray(scores, dtype=float)
    n_positive, n_negative = int(labels.sum()), int((~labels).sum())
    if not n_positive or not n_negative:
        raise ValueError("Presence AUROC needs both sham and injected trials")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(order):
        end = start+1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start+1+end)/2
        start = end
    return float((ranks[labels].sum() - n_positive*(n_positive+1)/2)/(n_positive*n_negative))


def presence_metrics(rows):
    sham, injected = ([r for r in rows if r["injected"] == state] for state in (False, True))
    if not sham or len(sham) != len(injected):
        raise ValueError("Presence summaries require equal nonempty sham and injected groups")
    fp = sum(r["reported_presence"] is True for r in sham)/len(sham)
    hit = sum(r["reported_presence"] is True for r in injected)/len(injected)
    correct_rejections = sum(r["reported_presence"] is False for r in sham)/len(sham)
    # Invalid answers count as incorrect in balanced accuracy. They are neither
    # affirmative nor negative reports; hence correct rejections need not be 1-F.
    return {"n_sham": len(sham), "n_injected": len(injected), "false_positive_rate": fp,
            "hit_rate": hit, "correct_rejection_rate": correct_rejections,
            "balanced_accuracy": (hit+correct_rejections)/2,
            "auroc": auroc([r["injected"] for r in rows], [r["presence_margin"] for r in rows]),
            "mean_presence_margin_sham": float(np.mean([r["presence_margin"] for r in sham])),
            "mean_presence_margin_injected": float(np.mean([r["presence_margin"] for r in injected])),
            "invalid_rate_sham": sum(not r["valid_response"] for r in sham)/len(sham),
            "invalid_rate_injected": sum(not r["valid_response"] for r in injected)/len(injected)}


DIFFERENCES = ("delta_false_positive_rate", "delta_hit_rate", "delta_margin_sham", "delta_margin_injected")
SENTENCE_KEYS = ("neutral_sentence_sha256", "evocative_sentence_sha256", "companion_sentence_sha256")


def paired_blocks(rows):
    """Keep both text versions and both intervention states in one comparison."""
    groups = defaultdict(dict)
    cells = defaultdict(set)
    for row in rows:
        key = (row["condition_id"], row["pair_id"], row["target_label"], row["present_label"])
        cell = row["text_version"], row["injected"]
        if cell in groups[key]:
            raise ValueError("Duplicate technical trial")
        groups[key][cell] = row
        cells[key[:2]].add(key[2:])
    if any(value != {(a, m) for a in ("A", "B") for m in ("X", "Y")} for value in cells.values()):
        raise ValueError("Missing target position or presence mapping")
    blocks = []
    for key, group in groups.items():
        if set(group) != {(v, state) for v in ("neutral", "evocative") for state in (False, True)}:
            raise ValueError("The four text/intervention conditions must remain paired")
        neutral_sham, evocative_sham = group[("neutral", False)], group[("evocative", False)]
        neutral_inj, evocative_inj = group[("neutral", True)], group[("evocative", True)]
        if neutral_inj["planned_injection_alpha"] != evocative_inj["planned_injection_alpha"]:
            raise ValueError("The two text versions must receive the same coefficient")
        identity = ("concept", "direction_id", "layer", "dose_axis", "dose", "dose_role", "scale_statistic", *SENTENCE_KEYS)
        if any(tuple(r[k] for k in identity) != tuple(neutral_sham[k] for k in identity) for r in group.values()):
            raise ValueError("Paired trial metadata differ")
        blocks.append({**{k: neutral_sham[k] for k in (*identity, "condition_id", "pair_id", "target_label", "present_label")},
            "delta_false_positive_rate": float(evocative_sham["reported_presence"] is True)-float(neutral_sham["reported_presence"] is True),
            "delta_hit_rate": float(evocative_inj["reported_presence"] is True)-float(neutral_inj["reported_presence"] is True),
            "delta_margin_sham": evocative_sham["presence_margin"]-neutral_sham["presence_margin"],
            "delta_margin_injected": evocative_inj["presence_margin"]-neutral_inj["presence_margin"]})
    return blocks


def paired_uncertainty(blocks, settings):
    """Crossed cluster bootstrap of concepts and sentence identities.

    A sampled sentence receives the same multiplicity wherever it occurs,
    including as a companion or in another concept. Text variants, sham/injection,
    mappings, target positions and repeated doses are never resampled separately.
    """
    concepts = sorted({r["concept"] for r in blocks})
    sentences = sorted({r[k] for r in blocks for k in SENTENCE_KEYS})
    n_pairs = len({r["pair_id"] for r in blocks})
    result = {"n_pairs": n_pairs, "n_concepts": len(concepts), "n_sentence_identities": len(sentences),
              "bootstrap_effective_resamples": 0,
              "uncertainty_scope": "conditional_on_one_concept" if len(concepts) == 1 else "concepts_and_sentences"}
    values = np.array([[r[field] for field in DIFFERENCES] for r in blocks], dtype=float)
    for field, value in zip(DIFFERENCES, values.mean(axis=0)):
        result[field], result[field+"_ci_low"], result[field+"_ci_high"] = float(value), None, None
    if n_pairs < 2:
        result["uncertainty_scope"] = "insufficient_text_pairs"
        return result
    c_index = {c: i for i, c in enumerate(concepts)}
    s_index = {s: i for i, s in enumerate(sentences)}
    ci = np.array([c_index[r["concept"]] for r in blocks])
    si = np.array([[s_index[r[k]] for k in SENTENCE_KEYS] for r in blocks])
    rng = np.random.default_rng(settings["bootstrap_seed"])
    draws = settings["bootstrap_resamples"]
    estimates = []
    for _ in range(draws*20):
        cw = rng.multinomial(len(concepts), np.full(len(concepts), 1/len(concepts)))
        sw = rng.multinomial(len(sentences), np.full(len(sentences), 1/len(sentences)))
        weights = cw[ci].astype(float) * sw[si].prod(axis=1)
        if weights.sum() > 0:
            estimates.append(np.average(values, axis=0, weights=weights))
        if len(estimates) == draws:
            break
    result["bootstrap_effective_resamples"] = len(estimates)
    if len(estimates) < draws:
        result["uncertainty_scope"] = "insufficient_nonempty_resamples"
        return result
    tail = (1-settings["ci_level"])/2
    low, high = np.quantile(estimates, [tail, 1-tail], axis=0)
    for field, lo, hi in zip(DIFFERENCES, low, high):
        result[field+"_ci_low"], result[field+"_ci_high"] = float(lo), float(hi)
    return result
