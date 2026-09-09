"""
Synthetic hook test for multi_inject.py: a tiny fake decoder stack (a few
nn.Modules returning (hidden_states,) tuples, small hidden dim, no HF download)
exercising apply_multi_injection with multiple specs. Needs only CPU torch, no
8B model or cluster access.

Run with: python code/utils/test_multi_inject.py
(or: uv run pytest code/utils/test_multi_inject.py, if pytest is available)
"""

import torch
import torch.nn as nn

from multi_inject import InjectionSpec, apply_multi_injection


HIDDEN_DIM = 8
NUM_LAYERS = 4


class FakeDecoderLayer(nn.Module):
    """Identity layer returning (hidden_states,) like a LlamaDecoderLayer."""

    def forward(self, hidden_states):
        return (hidden_states,)


class FakeModelInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([FakeDecoderLayer() for _ in range(NUM_LAYERS)])


class FakeModel(nn.Module):
    """Mimics the model.model.layers[l] access pattern used by apply_multi_injection."""

    def __init__(self):
        super().__init__()
        self.model = FakeModelInner()
        self._dummy_param = nn.Parameter(torch.zeros(1))

    def run(self, hidden_states):
        h = hidden_states
        for layer in self.model.layers:
            h = layer(h)[0]
        return h


def _unit(v):
    return v / torch.norm(v, p=2)


def test_single_spec_matches_manual_addition():
    torch.manual_seed(0)
    model = FakeModel()
    x = torch.randn(1, 3, HIDDEN_DIM)
    vector = torch.randn(HIDDEN_DIM)
    spec = InjectionSpec(layer=1, vector=vector, alpha=3.0)

    with apply_multi_injection(model, [spec]):
        out = model.run(x)

    expected = x.clone()
    expected = expected  # layer 0 identity
    expected = expected + 3.0 * _unit(vector)  # layer 1 injection, all tokens
    assert torch.allclose(out, expected, atol=1e-6)


def test_multiple_layers_additive_and_independent():
    torch.manual_seed(1)
    model = FakeModel()
    x = torch.randn(1, 3, HIDDEN_DIM)
    v0 = torch.randn(HIDDEN_DIM)
    v2 = torch.randn(HIDDEN_DIM)
    specs = [
        InjectionSpec(layer=0, vector=v0, alpha=2.0),
        InjectionSpec(layer=2, vector=v2, alpha=-1.5),
    ]

    with apply_multi_injection(model, specs):
        out = model.run(x)

    expected = x + 2.0 * _unit(v0)
    expected = expected + (-1.5) * _unit(v2)
    assert torch.allclose(out, expected, atol=1e-6)


def test_same_layer_sums_contributions():
    torch.manual_seed(2)
    model = FakeModel()
    x = torch.randn(1, 3, HIDDEN_DIM)
    v_a = torch.randn(HIDDEN_DIM)
    v_b = torch.randn(HIDDEN_DIM)
    specs = [
        InjectionSpec(layer=1, vector=v_a, alpha=2.0),
        InjectionSpec(layer=1, vector=v_b, alpha=5.0),
    ]

    with apply_multi_injection(model, specs):
        out = model.run(x)

    expected = x + 2.0 * _unit(v_a) + 5.0 * _unit(v_b)
    assert torch.allclose(out, expected, atol=1e-6)


def test_token_range_restricts_to_given_positions():
    torch.manual_seed(3)
    model = FakeModel()
    x = torch.randn(1, 5, HIDDEN_DIM)
    vector = torch.randn(HIDDEN_DIM)
    spec = InjectionSpec(layer=0, vector=vector, alpha=4.0, token_range=(1, 3))

    with apply_multi_injection(model, [spec]):
        out = model.run(x)

    expected = x.clone()
    expected[:, 1:3, :] += 4.0 * _unit(vector)
    assert torch.allclose(out, expected, atol=1e-6)


def test_token_range_tracks_absolute_position_across_incremental_calls():
    """Simulates KV-cached generation: one prompt-length forward call followed by
    seq_len==1 decode calls. token_range is given in absolute prompt-relative
    token indices and must stay correct across calls."""
    torch.manual_seed(4)
    model = FakeModel()
    vector = torch.randn(HIDDEN_DIM)
    # Range covers position 4 only (the first generated token, right after a
    # 4-token prompt: positions 0-3 are prompt, position 4 is first decode step).
    spec = InjectionSpec(layer=0, vector=vector, alpha=6.0, token_range=(4, 5))

    prompt = torch.randn(1, 4, HIDDEN_DIM)
    decode_step = torch.randn(1, 1, HIDDEN_DIM)

    with apply_multi_injection(model, [spec]):
        prompt_out = model.run(prompt)
        decode_out = model.run(decode_step)

    assert torch.allclose(prompt_out, prompt, atol=1e-6)  # untouched: range starts after prompt
    assert torch.allclose(decode_out, decode_step + 6.0 * _unit(vector), atol=1e-6)


def test_hooks_removed_after_context_manager_exits():
    model = FakeModel()
    x = torch.randn(1, 3, HIDDEN_DIM)
    spec = InjectionSpec(layer=0, vector=torch.randn(HIDDEN_DIM), alpha=1.0)

    with apply_multi_injection(model, [spec]):
        pass

    for layer in model.model.layers:
        assert len(layer._forward_hooks) == 0


def test_hooks_removed_on_exception():
    model = FakeModel()
    x = torch.randn(1, 3, HIDDEN_DIM)
    spec = InjectionSpec(layer=0, vector=torch.randn(HIDDEN_DIM), alpha=1.0)

    try:
        with apply_multi_injection(model, [spec]):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    for layer in model.model.layers:
        assert len(layer._forward_hooks) == 0


def test_empty_specs_is_a_noop():
    model = FakeModel()
    x = torch.randn(1, 3, HIDDEN_DIM)
    with apply_multi_injection(model, []):
        out = model.run(x)
    assert torch.allclose(out, x)


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASSED: {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
