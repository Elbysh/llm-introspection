"""Section 9.5, step 5: recode X/Y into presence without forcing a valid answer."""

import torch


def score_response(logits, prompt, tokenizer, correct):
    """Score the actual full-vocabulary choice and retain the X/Y contrast."""
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite output logits")
    ids = prompt["answer_token_ids"]
    incorrect = "Y" if correct == "X" else "X"
    response_id = int(logits.argmax())
    valid = response_id in ids.values()
    return {
        "response_token_id": response_id,
        "response_text": tokenizer.decode([response_id], skip_special_tokens=False),
        "valid_response": valid,
        "accuracy": int(response_id == ids[correct]),
        "logit_x": float(logits[ids["X"]]),
        "logit_y": float(logits[ids["Y"]]),
        "correct_margin": float(logits[ids[correct]] - logits[ids[incorrect]]),
        "forced_label_accuracy": float(logits[ids[correct]] > logits[ids[incorrect]])
        + 0.5 * float(logits[ids[correct]] == logits[ids[incorrect]]),
    }


def score_presence(logits, prompt, tokenizer, injected):
    correct = prompt["present_label"] if injected else prompt["absent_label"]
    measured = score_response(logits, prompt, tokenizer, correct)
    ids = prompt["answer_token_ids"]
    present = (measured["response_token_id"] == ids[prompt["present_label"]]
               if measured["valid_response"] else None)
    return {**measured, "correct_label": correct, "reported_presence": present,
            "presence_margin": float(logits[ids[prompt["present_label"]]] - logits[ids[prompt["absent_label"]]])}
