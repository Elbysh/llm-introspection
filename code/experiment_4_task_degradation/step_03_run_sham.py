"""Section 8.5, step 3: unmodified reference with an active observation hook."""

import torch
from experiment_0_calibration.step_01_02_collect_natural_activations import (
    decoder_layer,
)

from .step_05_score_responses import score_response


@torch.inference_mode()
def forward_at_sentence(model, prompt, layer, hook):
    """One full prefill already gives first-response logits; never replay a token."""
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([prompt["input_ids"]], dtype=torch.long, device=device)
    handle = decoder_layer(model, layer).register_forward_hook(hook)
    try:
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        return logits[0, -1].detach().float().cpu()
    finally:
        handle.remove()


def run_sham(model, tokenizer, prompt, layer):
    """An active identity hook checks the exact admissible activations."""
    measurements = {}

    def observe(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        selected = h[0, prompt["target_token_indices"]].float()
        if not torch.isfinite(selected).all():
            raise ValueError("Nonfinite sham activations")
        measurements["target_token_count"] = len(selected)
        return output

    logits = forward_at_sentence(model, prompt, layer, observe)
    if not measurements:
        raise RuntimeError("The sham hook did not run")
    return logits, {**score_response(logits, prompt, tokenizer), **measurements}
