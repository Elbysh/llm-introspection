from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from experiment_0_calibration.prepare_material import unit


def test_direction_normalization_preserves_original_norm():
    normalized, original_norm = unit(torch.tensor([3.0, 4.0]))

    assert original_norm == pytest.approx(5.0)
    assert normalized.tolist() == pytest.approx([0.6, 0.8])


def test_zero_direction_is_rejected():
    with pytest.raises(ValueError, match="non-zero"):
        unit(torch.zeros(4))


class _SpaceTokenizer:
    """Tokens are whitespace-prefixed words, so a span boundary can fall inside one."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        offsets = []
        index = 0
        while index < len(text):
            end = index + 1
            while end < len(text) and not text[end].isspace():
                end += 1
            offsets.append((index, end))
            index = end
        encoded = {"input_ids": list(range(len(offsets)))}
        if return_offsets_mapping:
            encoded["offset_mapping"] = offsets
        return encoded

    @staticmethod
    def convert_ids_to_tokens(token_id):
        return "t{}".format(token_id)


def _config(position_policy, sentence):
    from experiment_0_calibration.protocol_config import Experiment0Config

    return Experiment0Config(
        protocol_version="test", protocol_status="development",
        model_name="m", model_revision="r", tokenizer_revision="r",
        num_decoder_blocks=None, layers=[0], activation_site="decoder_block_output",
        concepts=[], vector_type="avg", hidden_state_offset=1,
        fixed_random_count_per_layer=1, fixed_random_base_seed=1,
        fixed_random_layer_stride=1, noise_repetitions_per_position=1,
        renewed_noise_base_seed=1, scrambled_concept_enabled=False,
        scrambled_concept_base_seed=0, presentation_mode="external_manifest",
        context_id="ctx", context_template="", context_manifest=None,
        position_policy=position_policy, point_weighting="equal_token",
        bootstrap_unit="sentence", bootstrap_resamples_sd=2, bootstrap_resamples_mad=2,
        bootstrap_seed=1, ci_level=0.95, batch_size=1, direction_chunk_size=1,
        vector_dir=Path("."), plan_dir=Path("."), output_dir=Path("."),
    )


def _rows(monkeypatch, policy, sentence):
    """Resolve one manifest context whose span starts inside a token."""
    from experiment_0_calibration import prepare_material

    rendered = "A) " + sentence + "\n"
    start = rendered.index(sentence)
    monkeypatch.setattr(
        prepare_material, "read_jsonl",
        lambda _path: [{
            "context_id": "c0",
            "rendered_text": rendered,
            "targets": [{"sentence_id": "localization_000",
                         "char_start": start, "char_end": start + len(sentence)}],
        }],
    )
    return prepare_material.build_context_and_observation_rows(
        _SpaceTokenizer(), [sentence], _config(policy, sentence)
    )


def test_the_strict_policy_refuses_a_token_that_straddles_the_span(monkeypatch):
    # "A)" and the sentence's first word share a token here, as they do in the real
    # 2AFC prompt, where the leading space of "A) " belongs to the first word's token.
    with pytest.raises(ValueError, match="ambiguous token boundary"):
        _rows(monkeypatch, "all_sentence_tokens", "the cat sleeps")


def test_the_overlapping_policy_keeps_the_tokens_experiment_1_perturbs(monkeypatch):
    contexts, observations = _rows(
        monkeypatch, "all_overlapping_sentence_tokens", "the cat sleeps"
    )

    # Experiment 1 perturbs every token overlapping the span, boundary ones included:
    # " the" carries the label's trailing space and " sleeps\n" the newline after the
    # sentence, so the admissible set is those two plus " cat" -- not the single token
    # strictly inside. Dropping the boundary tokens would estimate s on a subset of
    # what the experiment perturbs.
    assert contexts[0]["admissible_token_positions"] == [1, 2, 3]
    assert len(observations) == 3
    assert [row["target_index"] for row in observations] == [0, 0, 0]
