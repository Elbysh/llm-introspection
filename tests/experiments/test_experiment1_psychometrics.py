import math

import pytest

torch = pytest.importorskip("torch")

from experiments.experiment1_psychometrics import (  # noqa: E402
    amplitude_for_dropout_rate,
    answer_token_ids,
    bracketing_doses,
    build_localization_prompt,
    build_sentence_pairs,
    count_forward_passes,
    dropout_rate_for_amplitude,
    forced_choice_score,
    fit_psychometric,
    log_grid,
    threshold_at_75,
)


class Offset:
    """Minimal stand-in for one entry of a tokenizer offset mapping."""

    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class CharacterTokenizer:
    """One token per character, so a token range must recover the sentence exactly."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return (
            "<|bos|>user\n"
            + messages[0]["content"]
            + "<|eot_id|>assistant\n"
            + messages[1]["content"]
            + "<|eot_id|>"
        )

    def __call__(self, text, return_tensors=None, add_special_tokens=False,
                 return_offsets_mapping=False):
        encoding = {
            "input_ids": list(range(len(text))),
            "attention_mask": [1] * len(text),
        }
        if return_offsets_mapping:
            encoding["offset_mapping"] = [
                [[Offset(index), Offset(index + 1)] for index in range(len(text))]
            ]
        return encoding


class SpacedTokenizer:
    """Tokenizer whose " A" is a single token, like Llama's."""

    def __init__(self, single_space_tokens=True):
        self.single_space_tokens = single_space_tokens

    def encode(self, text, add_special_tokens=False):
        if text.startswith(" ") and self.single_space_tokens:
            return [1000 + ord(text[1:])]
        return [ord(character) for character in text]


def test_answer_tokens_prefer_the_space_prefixed_variant():
    spaced = answer_token_ids(SpacedTokenizer())
    bare = answer_token_ids(SpacedTokenizer(single_space_tokens=False))

    assert spaced == (1000 + ord("A"), 1000 + ord("B"))
    # A tokenizer that splits " A" into two tokens falls back to the bare letters.
    assert bare == (ord("A"), ord("B"))


def test_log_grid_uses_a_constant_factor():
    assert log_grid(0.5, 32.0, 7) == pytest.approx([0.5, 1, 2, 4, 8, 16, 32])


def test_log_grid_rejects_a_non_positive_lower_bound():
    with pytest.raises(ValueError):
        log_grid(0.0, 32.0, 7)


@pytest.mark.parametrize("alpha,token_norm", [(1.0, 10.0), (8.0, 10.0), (50.0, 10.0)])
def test_dropout_rate_realizes_the_requested_amplitude(alpha, token_norm):
    rate = dropout_rate_for_amplitude(alpha, token_norm)

    assert 0.0 <= rate < 1.0
    # E||dh||^2 = p/(1-p) ||h||^2, so the rate must invert back to alpha.
    assert token_norm * math.sqrt(rate / (1.0 - rate)) == pytest.approx(alpha)
    assert amplitude_for_dropout_rate(rate, token_norm) == pytest.approx(alpha)


def test_a_tie_scores_as_chance_for_either_target():
    # bfloat16 can leave the logits untouched at a negligible dose; scoring that as
    # wrong for both targets would drag a cell towards 0 instead of towards chance.
    assert forced_choice_score(0.0, "A") == 0.5
    assert forced_choice_score(0.0, "B") == 0.5
    assert forced_choice_score(1.5, "A") == 1.0
    assert forced_choice_score(1.5, "B") == 0.0
    assert forced_choice_score(-1.5, "B") == 1.0


def test_psychometric_fit_recovers_a_known_threshold():
    beta0, beta1 = -2.0, 1.5
    doses = [0.5, 1, 2, 4, 8, 16, 32]
    totals = [200] * len(doses)
    probabilities = [
        0.5 + 0.5 / (1 + math.exp(-(beta0 + beta1 * math.log(dose)))) for dose in doses
    ]
    corrects = [int(round(p * n)) for p, n in zip(probabilities, totals)]

    fitted0, fitted1, converged = fit_psychometric(doses, corrects, totals)

    assert converged
    assert threshold_at_75(fitted0, fitted1) == pytest.approx(
        math.exp(-beta0 / beta1), rel=0.05
    )


def test_flat_data_yields_no_reportable_threshold():
    doses = [0.5, 1, 2, 4, 8, 16, 32]
    _, _, converged = fit_psychometric(doses, [100] * 7, [200] * 7)

    assert not converged


def test_bracketing_only_reports_a_crossed_transition():
    assert bracketing_doses([1, 2, 4, 8], [0.5, 0.6, 0.8, 0.9]) == (2, 4)
    assert bracketing_doses([1, 2, 4], [0.5, 0.55, 0.6]) is None


@pytest.mark.parametrize("label_order,expected", [("AB", ["A", "B"]), ("BA", ["B", "A"])])
def test_token_ranges_cover_the_sentences_under_both_label_orders(label_order, expected):
    tokenizer = CharacterTokenizer()
    first, second = "the cat sleeps", "a dog barks now"

    prompt, ranges, labels, encoding = build_localization_prompt(
        tokenizer, first, second, label_order
    )

    assert labels == expected
    assert prompt[ranges[0][0]:ranges[0][1]] == first
    assert prompt[ranges[1][0]:ranges[1][1]] == second
    assert prompt.endswith("The answer is")
    assert "offset_mapping" not in encoding


def test_pairs_are_length_matched_disjoint_and_deterministic():
    tokenizer = CharacterTokenizer()
    sentences = ["{}{:02d}".format("word " * (3 + index % 5), index) for index in range(20)]

    pairs = build_sentence_pairs(tokenizer, sentences, 5, seed=1)

    used = [pair["sentence_x"] for pair in pairs] + [pair["sentence_y"] for pair in pairs]
    assert len(used) == len(set(used))
    assert all(pair["token_length_gap"] == 0 for pair in pairs)
    assert build_sentence_pairs(tokenizer, sentences, 5, seed=1) == pairs


def test_pairs_refuse_more_than_the_corpus_allows():
    tokenizer = CharacterTokenizer()
    sentences = ["sentence {}".format(index) for index in range(20)]

    with pytest.raises(ValueError, match="disjoint pairs"):
        build_sentence_pairs(tokenizer, sentences, 50, seed=1)


def test_plan_counts_both_orders_both_targets_and_shared_shams():
    class Args:
        layers = [3, 16]
        matchings = ["alpha", "z"]
        dose_grid = {"alpha": [1, 2, 4], "z": [1, 2, 4]}
        pairs = [{"pair_id": index} for index in range(5)]
        label_orders = ["AB"]

    conditions = {3: [("concept", "c1"), ("random", "r1")], 16: [("noise", "n0")]}

    perturbed, shams = count_forward_passes(conditions, Args())

    assert perturbed == 3 * 6 * (5 * 2 * 1 * 2)
    assert shams == 5 * 2 * 1
