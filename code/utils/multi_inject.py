"""
Multi-injection engine: K simultaneous concept-vector injections at arbitrary
(possibly repeated) layers, generalizing inject_concept_vector.py's single-hook
pattern to the plan's core mechanic: h(l) <- h(l) + alpha_k * v_k for each injection k.
"""

import torch
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class InjectionSpec:
    layer: int
    vector: torch.Tensor
    alpha: float
    # None = inject at every token (prompt + every generated token), matching
    # inject_concept_vector.py's default "inject at all tokens" behavior, which
    # keeps the injection sustained through generation via KV-cached incremental
    # forward calls. A (start, end) tuple restricts injection to that absolute
    # token-position range (prompt-relative), for position-specific use cases.
    token_range: tuple | None = None


def _prepare_vector(vector, device, dtype):
    """Unit-normalize then cast, matching inject_concept_vector.py's ordering exactly."""
    if not isinstance(vector, torch.Tensor):
        vector = torch.tensor(vector, dtype=dtype)
    vector = vector / torch.norm(vector, p=2)
    vector = vector.to(dtype=dtype, device=device)
    if vector.dim() == 1:
        vector = vector.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
    elif vector.dim() == 2:
        vector = vector.unsqueeze(0)  # [1, 1, hidden_dim]
    return vector


def _make_layer_hook(layer_specs):
    """
    layer_specs: list of (unit_vector [1,1,H], alpha, token_range) sharing one layer.

    Tracks absolute sequence position across successive forward calls (nonlocal
    `pos`), so a `token_range` given in prompt-relative token indices stays correct
    across the prompt-processing call and the seq_len==1 incremental decode calls
    that follow under KV caching — mirroring inject_concept_vector.py's
    prompt_processed/injection_start_token tracking, generalized to several specs.
    """
    state = {"pos": 0}

    def hook_fn(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        batch_size, seq_len, hidden_dim = hidden_states.shape
        start_pos = state["pos"]
        end_pos = start_pos + seq_len
        state["pos"] = end_pos

        delta = torch.zeros_like(hidden_states)
        for vector, alpha, token_range in layer_specs:
            steer = vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
            if token_range is None:
                delta = delta + alpha * steer.expand(batch_size, seq_len, -1)
            else:
                r_start, r_end = token_range
                lo, hi = max(r_start, start_pos), min(r_end, end_pos)
                if lo < hi:
                    local_lo, local_hi = lo - start_pos, hi - start_pos
                    delta[:, local_lo:local_hi, :] += alpha * steer.expand(
                        batch_size, local_hi - local_lo, -1
                    )

        modified_hidden_states = hidden_states + delta
        return (modified_hidden_states,) + output[1:] if isinstance(output, tuple) else modified_hidden_states

    return hook_fn


@contextmanager
def apply_multi_injection(model, specs: list[InjectionSpec], layers=None):
    """
    Registers one forward hook per distinct layer in `specs`, summing every spec's
    alpha * unit(vector) contribution assigned to that layer (handles two injections
    landing on the same layer). Yields with hooks active; guarantees hook removal
    even on exception.

    `layers`: the model's decoder-layer list/sequence to index InjectionSpec.layer
    into. Defaults to `model.model.layers` (Llama's attribute path) when not given,
    for backward compatibility -- pass e.g. `model.model.language_model.layers` for
    a model shaped differently (see code/utils/model_registry.py's
    ModelSpec.get_layers for the per-model resolution).

    K=1, single-layer, token_range=None reduces to identical tensor math as
    inject_concept_vector.py's "inject at all tokens" branch. The hook already
    handles a decoder layer whose forward() returns a bare tensor instead of a
    (hidden_states, ...) tuple (see the isinstance check in _make_layer_hook), so
    no other change is needed for a differently-shaped decoder layer class.
    """
    if not specs:
        yield
        return

    if layers is None:
        layers = model.model.layers

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    by_layer: dict[int, list] = {}
    for spec in specs:
        vector = _prepare_vector(spec.vector, device, dtype)
        by_layer.setdefault(spec.layer, []).append((vector, spec.alpha, spec.token_range))

    handles = []
    try:
        for layer, layer_specs in by_layer.items():
            handle = layers[layer].register_forward_hook(_make_layer_hook(layer_specs))
            handles.append(handle)
        yield
    finally:
        for handle in handles:
            handle.remove()
