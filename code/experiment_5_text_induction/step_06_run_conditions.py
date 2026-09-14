"""Section 9.5, step 6: execute every predeclared cell, keeping contexts independent."""

import json

from .step_02_run_fresh_context import run_fresh_context


def execute_conditions(model, tokenizer, prompts, conditions, trials, scales, stream):
    prompt_index = {p["prompt_id"]: p for p in prompts}
    condition_index = {c["condition_id"]: c for c in conditions}
    for index, trial in enumerate(trials):
        prompt = prompt_index[trial["prompt_id"]]
        condition = condition_index[trial["condition_id"]]
        measured = run_fresh_context(model, tokenizer, prompt, condition, scales, trial["injected"])
        row = {**trial, **{k: prompt[k] for k in ("pair_id", "concept", "text_version", "target_label",
               "present_label", "neutral_sentence_sha256", "evocative_sentence_sha256", "companion_sentence_sha256")},
               **{k: condition[k] for k in ("layer", "dose_axis", "dose", "dose_role", "scale_statistic", "direction_id")},
               "planned_injection_alpha": condition["alpha_requested"],
               "natural_scale": condition["natural_scale"], **measured}
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        if (index+1) % 16 == 0:
            print(f"Experiment 5: {index+1}/{len(trials)} trials saved", flush=True)
    return len(trials)
