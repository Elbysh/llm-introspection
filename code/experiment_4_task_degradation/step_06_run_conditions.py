"""Section 8.5, step 6: paired execution with identical texts, targets and mappings."""

import json
from collections import defaultdict

from .step_03_run_sham import run_sham
from .step_04_apply_intervention import run_intervention
from .step_05_score_responses import output_js, score_response


def execute_conditions(model, tokenizer, prompts, conditions, trials, scales, stream):
    """Share a sham only between identical prompt/layer forwards.

    Process one prompt/layer group at a time so full-vocabulary reference logits
    are released promptly. Every emitted line is flushed; interrupted results
    remain inspectable but are never mistaken for a completed experiment.
    """
    prompt_index = {p["prompt_id"]: p for p in prompts}
    condition_index = {c["condition_id"]: c for c in conditions}
    groups = defaultdict(list)
    for trial in trials:
        groups[(trial["prompt_id"], condition_index[trial["condition_id"]]["layer"])].append(trial)
    n_written = 0

    def emit(row):
        nonlocal n_written
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        n_written += 1

    for (prompt_id, layer), group in groups.items():
        prompt = prompt_index[prompt_id]
        base = {key: prompt[key] for key in ("prompt_id", "example_id", "probed_concept",
                "label", "sentence_sha256", "yes_label", "correct_label")}
        base.update(layer=layer, target_token_indices=prompt["target_token_indices"])
        clean_logits, reference = run_sham(model, tokenizer, prompt, layer)
        sham_id = f"sham__{prompt_id}__block_{layer:02d}"
        emit({**base, **reference, "trial_id": sham_id, "family": "sham", "status": "ok"})
        for trial in group:
            condition = condition_index[trial["condition_id"]]
            row = {**base, **{k: condition[k] for k in (
                "condition_id", "family", "intervention_id", "dose_axis", "dose", "scale_statistic")},
                "trial_id": trial["trial_id"], "sham_id": sham_id,
                "sham_accuracy": reference["accuracy"],
                "sham_correct_margin": reference["correct_margin"]}
            # Numerical failures abort rather than silently removing doses or
            # returning a scientifically arbitrary score. The unfinished run is
            # recorded on disk by the orchestrator and must not enter analysis.
            logits, amplitudes = run_intervention(model, prompt, condition, trial["direction_ids"], scales)
            measured = score_response(logits, prompt, tokenizer)
            row.update(measured)
            row.update(amplitudes)
            row.update(status="ok", delta_accuracy=measured["accuracy"]-reference["accuracy"],
                       delta_correct_margin=measured["correct_margin"]-reference["correct_margin"],
                       js=output_js(clean_logits, logits) if trial["measure_js"] else None)
            emit(row)
        print(f"{prompt_id}, block {layer}: {n_written} rows saved", flush=True)
    return n_written
