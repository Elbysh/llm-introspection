"""Section 9.5, step 2: one full, fresh context for every sham and intervention."""

import math

import torch
from experiment_0_calibration.prepare_material import materialize_direction
from experiment_0_calibration.step_01_02_collect_natural_activations import (
    decoder_layer,
)

from .step_05_score_presence import score_presence


@torch.inference_mode()
def forward_at_sentence(model, prompt, layer, hook):
    """Run one complete prefill and always remove the temporary layer hook."""
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([prompt["input_ids"]], dtype=torch.long, device=device)
    handle = decoder_layer(model, layer).register_forward_hook(hook)
    try:
        logits = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
        ).logits
        return logits[0, -1].detach().float().cpu()
    finally:
        handle.remove()


def run_sham(model, prompt, layer):
    """Observe the target tokens through an identity hook without changing them."""
    measurements = {}

    def observe(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        selected = hidden[0, prompt["target_token_indices"]].float()
        if not torch.isfinite(selected).all():
            raise ValueError("Nonfinite sham activations")
        measurements["target_token_count"] = len(selected)
        return output

    logits = forward_at_sentence(model, prompt, layer, observe)
    if not measurements:
        raise RuntimeError("The sham hook did not run")
    return logits


def build_addition(condition, scale_records, hidden_size):
    """Reuse Experiment 0's direction loading and calibrated natural scale."""
    record = scale_records[condition["direction_id"]]
    if record["decoder_block_index"] != condition["layer"]:
        raise ValueError("Direction and intervention layers differ")
    if record["direction_family"] != "concept":
        raise ValueError("Experiment 5 accepts only concept directions")
    vector, _ = materialize_direction(record, None, hidden_size)
    scale = float(record[condition["scale_statistic"]])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Natural directional scale must be finite and positive")
    alpha = condition["dose"] * scale if condition["dose_axis"] == "z" else condition["dose"]
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Requested amplitude must be finite and positive")
    return vector * alpha, alpha, scale


def run_intervention(model, prompt, condition, scales):
    addition, alpha, natural_scale = build_addition(
        condition, scales, model.config.hidden_size
    )
    measurements = {}

    def inject(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[0] != 1:
            raise ValueError("Experiment 5 forwards one prompt at a time")
        positions = prompt["target_token_indices"]
        before = hidden[:, positions, :]
        if not torch.isfinite(before).all():
            raise ValueError("Nonfinite activations before intervention")
        modified = hidden.clone()
        modified[:, positions, :] = before + addition.to(
            device=hidden.device, dtype=hidden.dtype
        )
        after = modified[:, positions, :]
        if not torch.isfinite(after).all():
            raise ValueError("Nonfinite activations after intervention")
        measurements["alpha_realized_per_token"] = (
            (after.float() - before.float()).norm(dim=-1)[0].cpu().tolist()
        )
        return (modified,) + output[1:] if isinstance(output, tuple) else modified

    logits = forward_at_sentence(model, prompt, condition["layer"], inject)
    if not measurements:
        raise RuntimeError("The intervention hook did not run")
    count = prompt["target_token_count"]
    measurements.update(
        alpha_requested_per_token=[alpha] * count,
        natural_scale_per_token=[natural_scale] * count,
        direction_ids=[condition["direction_id"]] * count,
    )
    return logits, measurements


def run_fresh_context(model, tokenizer, prompt, condition, scales, injected):
    if injected:
        logits, measurements = run_intervention(model, prompt, condition, scales)
    else:
        logits = run_sham(model, prompt, condition["layer"])
        measurements = {"alpha_requested_per_token": [0.] * prompt["target_token_count"],
                        "alpha_realized_per_token": [0.] * prompt["target_token_count"],
                        "natural_scale_per_token": [0.] * prompt["target_token_count"],
                        "direction_ids": []}
    # No previous response, text variant or past_key_values enters the next trial.
    return {**score_presence(logits, prompt, tokenizer, injected), **measurements}
