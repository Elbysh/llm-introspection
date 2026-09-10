"""Experiment 0 steps 1-2: natural forwards and admissible activations."""

from collections import defaultdict
from typing import Any, Dict, List, Sequence

import torch


def decoder_layer(model: Any, decoder_block_index: int) -> Any:
    """Resolve the output-hook site named in the Experiment 0 protocol."""
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or decoder_block_index not in range(len(layers)):
        raise ValueError(
            "model does not expose decoder block {}".format(decoder_block_index)
        )
    return layers[decoder_block_index]


def select_admissible_activations(
    hidden_states: torch.Tensor,
    batch_contexts: Sequence[Dict[str, Any]],
    observations_by_context: Dict[str, List[Dict[str, Any]]],
) -> torch.Tensor:
    """Step 2: retain only the token positions declared by the material plan."""
    selected = []
    for batch_index, context in enumerate(batch_contexts):
        token_positions = [
            observation["token_index"]
            for observation in observations_by_context[context["context_id"]]
        ]
        selected.append(hidden_states[batch_index, token_positions, :].float().cpu())
    return torch.cat(selected, dim=0)


@torch.inference_mode()
def collect_natural_activations(
    model: Any,
    tokenizer: Any,
    contexts: Sequence[Dict[str, Any]],
    observations: Sequence[Dict[str, Any]],
    decoder_block_indices: Sequence[int],
    batch_size: int,
) -> Dict[int, torch.Tensor]:
    """Step 1: run without intervention; step 2: extract planned positions.

    The two steps share a module because activation extraction must happen while
    the natural forward hooks are alive. Their logic remains separated in the
    `select_admissible_activations` function above.
    """
    observations_by_context = defaultdict(list)
    for observation in observations:
        observations_by_context[observation["context_id"]].append(observation)
    for rows in observations_by_context.values():
        rows.sort(key=lambda row: row["observation_index"])

    captured: Dict[int, torch.Tensor] = {}
    chunks = {index: [] for index in decoder_block_indices}
    handles = []

    def capture(decoder_block_index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden_states = output[0] if isinstance(output, tuple) else output
            captured[decoder_block_index] = hidden_states.detach()

        return hook

    for index in decoder_block_indices:
        handles.append(decoder_layer(model, index).register_forward_hook(capture(index)))
    try:
        input_device = next(model.parameters()).device
        tokenizer.padding_side = "right"
        for start in range(0, len(contexts), batch_size):
            batch_contexts = contexts[start : start + batch_size]
            encoded = tokenizer(
                [context["rendered_text"] for context in batch_contexts],
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
            )
            for batch_index, context in enumerate(batch_contexts):
                token_count = len(context["input_ids"])
                actual_ids = encoded["input_ids"][batch_index, :token_count].tolist()
                if actual_ids != context["input_ids"]:
                    raise RuntimeError(
                        "tokenization drift for {}".format(context["context_id"])
                    )
            captured.clear()
            model(
                **{key: value.to(input_device) for key, value in encoded.items()},
                use_cache=False
            )
            for index in decoder_block_indices:
                chunks[index].append(
                    select_admissible_activations(
                        captured[index], batch_contexts, observations_by_context
                    )
                )
            print(
                "Steps 1-2 — natural contexts: {}/{}".format(
                    min(start + len(batch_contexts), len(contexts)), len(contexts)
                ),
                flush=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    activations = {
        index: torch.cat(chunks[index], dim=0) for index in decoder_block_indices
    }
    for index, matrix in activations.items():
        if matrix.shape[0] != len(observations):
            raise RuntimeError(
                "observation count mismatch at decoder block {}".format(index)
            )
    return activations

