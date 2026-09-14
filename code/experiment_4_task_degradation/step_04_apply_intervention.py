"""Section 8.5, step 4: calibrated addition at decoder output, sentence tokens only.

Concept and fixed-random directions are constant across the sentence. Renewed
noise supplies one separately calibrated direction for EACH target token.
Dropout is deliberately excluded at the user's request.
"""

import math

import torch
from experiment_0_calibration.prepare_material import materialize_direction

from .step_03_run_sham import forward_at_sentence


def build_additions(condition, direction_ids, scale_records, hidden_size):
    vectors, requested, scales = [], [], []
    for direction_id in direction_ids:
        record = scale_records[direction_id]
        if record["decoder_block_index"] != condition["layer"]:
            raise ValueError("Direction and intervention layers differ")
        if record["direction_family"] != condition["family"]:
            raise ValueError("Direction family does not match the imported condition")
        # Reuse exactly Experiment 0's loading and random-seed reconstruction.
        # Its current implementation does not consume the config argument.
        vector, _ = materialize_direction(record, None, hidden_size)
        scale = float(record[condition["scale_statistic"]])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Natural directional scale must be finite and strictly positive")
        alpha = condition["dose"] * scale if condition["dose_axis"] == "z" else condition["dose"]
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Invalid requested amplitude")
        vectors.append(vector * alpha)
        requested.append(alpha)
        scales.append(scale)
    return torch.stack(vectors), requested, scales


def make_sentence_hook(prompt, additions, measurements):
    """Measure realized displacements AFTER model-dtype rounding, per token."""
    positions = prompt["target_token_indices"]
    if additions.ndim != 2 or additions.shape[0] not in (1, len(positions)):
        raise ValueError("Supply one fixed vector or one noise vector per sentence token")

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[0] != 1:
            raise ValueError("Experiment 4 forwards exactly one prompt at a time")
        before = h[:, positions, :]
        if not torch.isfinite(before).all():
            raise ValueError("Nonfinite activations before intervention")
        modified = h.clone()
        modified[:, positions, :] = before + additions.to(device=h.device, dtype=h.dtype)
        after = modified[:, positions, :]
        if not torch.isfinite(after).all():
            raise ValueError("Nonfinite activations after intervention")
        measurements["alpha_realized_per_token"] = (after.float()-before.float()).norm(dim=-1)[0].cpu().tolist()
        return (modified,) + output[1:] if isinstance(output, tuple) else modified
    return hook


def run_intervention(model, prompt, condition, direction_ids, scales):
    additions, alpha, natural_scales = build_additions(
        condition, direction_ids, scales, model.config.hidden_size)
    measurements = {}
    logits = forward_at_sentence(model, prompt, condition["layer"],
                                 make_sentence_hook(prompt, additions, measurements))
    if not measurements:
        raise RuntimeError("The intervention hook did not run")
    count = len(prompt["target_token_indices"])
    if len(alpha) == 1:
        alpha, natural_scales = alpha * count, natural_scales * count
    measurements.update(alpha_requested_per_token=alpha,
                        natural_scale_per_token=natural_scales,
                        direction_ids=direction_ids)
    return logits, measurements
