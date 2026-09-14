"""Section 8.5, step 5: actual next-token response, correct-label margin and JS."""

import torch


def score_response(logits, prompt, tokenizer):
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite output logits")
    ids = prompt["answer_token_ids"]
    correct = prompt["correct_label"]
    incorrect = "Y" if correct == "X" else "X"
    # Greedy selection is over the FULL vocabulary, not an artificial X/Y restriction.
    # A non-label first token is invalid and counts as incorrect in task accuracy.
    response_id = int(logits.argmax())
    valid = response_id in ids.values()
    return {"response_token_id": response_id,
            "response_text": tokenizer.decode([response_id], skip_special_tokens=False),
            "valid_response": valid,
            "accuracy": int(response_id == ids[correct]),
            "logit_x": float(logits[ids["X"]]), "logit_y": float(logits[ids["Y"]]),
            "correct_margin": float(logits[ids[correct]] - logits[ids[incorrect]]),
            "forced_label_accuracy": float(logits[ids[correct]] > logits[ids[incorrect]])
            + .5 * float(logits[ids[correct]] == logits[ids[incorrect]])}


def output_js(clean_logits, changed_logits):
    """JS(clean || changed) over the entire next-token vocabulary, in nats."""
    a, b = clean_logits.double().log_softmax(-1), changed_logits.double().log_softmax(-1)
    mixed = torch.logaddexp(a, b) - torch.log(torch.tensor(2., dtype=torch.float64))
    return max(0., float(((a.exp() * (a-mixed)).sum() + (b.exp() * (b-mixed)).sum()) / 2))
