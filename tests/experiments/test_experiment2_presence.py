import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from experiments.experiment2_presence import (
    EXPERIMENT1_ALPHA_DOSES,
    EXPERIMENT1_Z_DOSES,
    Experiment0Calibration,
    auroc,
    binary_score,
    bootstrap_intervals,
    build_conditions,
    build_plan,
    build_presence_prompt,
    choice_token_ids,
    detection_metrics,
    loglinear_rate,
    pair_mapping_trials,
    recoded_score,
    response_labels,
)


def test_default_ranges_equal_the_recorded_experiment1_sweep():
    assert EXPERIMENT1_ALPHA_DOSES == (
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
        64.0,
        128.0,
    )
    assert EXPERIMENT1_Z_DOSES == (
        0.01,
        0.02,
        0.04,
        0.08,
        0.16,
        0.32,
        0.64,
        1.28,
        2.56,
        5.12,
        10.24,
        20.48,
    )


class Offset:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class CharacterTokenizer:
    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):
        return (
            "<|bos|>user\n"
            + messages[0]["content"]
            + "<|eot_id|>assistant\n"
            + messages[1]["content"]
            + "<|eot_id|>"
        )

    def __call__(
        self,
        text,
        return_tensors=None,
        add_special_tokens=False,
        return_offsets_mapping=False,
    ):
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
    def encode(self, text, add_special_tokens=False):
        return [1000 + ord(text[1])] if text.startswith(" ") else [ord(text)]


def test_prompt_ranges_and_mapping_are_explicit():
    prompt, ranges, labels, encoding = build_presence_prompt(
        CharacterTokenizer(), "first sentence", "second sentence", "BA", "YX"
    )

    assert labels == ["B", "A"]
    assert prompt[ranges[0][0] : ranges[0][1]] == "first sentence"
    assert prompt[ranges[1][0] : ranges[1][1]] == "second sentence"
    assert "X means that no intervention occurred" in prompt
    assert prompt.endswith("Answer:")
    assert "offset_mapping" not in encoding


def test_response_tokens_recode_and_semantics():
    assert choice_token_ids(SpacedTokenizer()) == (1000 + ord("X"), 1000 + ord("Y"))
    assert recoded_score(5, 2, "XY") == 3
    assert recoded_score(2, 5, "YX") == 3
    assert response_labels(4, 1, "XY") == ("X", "intervention")
    assert response_labels(4, 1, "YX") == ("X", "no_intervention")
    assert binary_score(0) == 0.5


def paired_record(signal=(2.0, 2.0), sham=(-2.0, -2.0), pair="p1", direction="d1"):
    return {
        "signal_scores": list(signal),
        "sham_scores": list(sham),
        "sham_trial_ids": [f"{pair}-XY", f"{pair}-YX"],
        "pair_id": pair,
        "direction_id": direction,
    }


def test_detection_metrics_separate_sensitivity_from_affirmative_bias():
    separated = detection_metrics([paired_record(), paired_record(pair="p2")])
    biased = detection_metrics(
        [
            paired_record(signal=(2, 2), sham=(2, 2)),
            paired_record(signal=(3, 3), sham=(3, 3), pair="p2"),
        ]
    )

    assert separated["d_prime"] > 0
    assert separated["balanced_accuracy"] == 1
    assert separated["auroc"] == 1
    assert biased["d_prime"] == pytest.approx(0)
    assert biased["criterion"] < 0
    assert biased["balanced_accuracy"] == 0.5


def test_statistics_handle_boundaries_ties_and_crossed_bootstrap():
    assert loglinear_rate(0, 10) == pytest.approx(0.5 / 11)
    assert auroc([1, 1], [1, 1]) == 0.5
    records = [
        paired_record(pair=pair, direction=direction)
        for pair in ("p1", "p2")
        for direction in ("d1", "d2")
    ]
    first = bootstrap_intervals(records, 30, 123)
    assert first == bootstrap_intervals(records, 30, 123)
    assert all(
        math.isfinite(bound) for interval in first.values() for bound in interval
    )


def trial(mapping, score, sham_score, trial_id):
    return {
        "kind": "perturbed",
        "layer": 3,
        "family": "concept",
        "direction_id": "concept:3",
        "matching": "z",
        "dose": 1.0,
        "pair_id": 0,
        "order": 0,
        "label_order": "AB",
        "target_index": 0,
        "mapping": mapping,
        "score": score,
        "sham_score": sham_score,
        "sham_trial_id": trial_id,
        "finite": True,
        "sham_finite": True,
        "top_token_is_choice": True,
        "sham_top_token_is_choice": True,
        "realized_amplitude": 1.0,
        "clean_token_norm": 2.0,
    }


def test_mapping_rows_pair_and_plan_keeps_mappings_adjacent():
    paired = pair_mapping_trials([trial("XY", 2, -1, "s1"), trial("YX", 4, -3, "s2")])
    assert len(paired) == 1
    assert paired[0]["signal_scores"] == [2.0, 4.0]

    args = SimpleNamespace(
        layers=[3],
        matchings=["z"],
        dose_grid={"z": [1.0, 2.0]},
        pairs=[{"pair_id": 0}, {"pair_id": 1}],
        label_orders=["AB", "BA"],
        seed=9,
    )
    plan = build_plan({3: [("concept", "c1")]}, args)
    for index in range(0, len(plan), 2):
        assert [row["mapping"] for row in plan[index : index + 2]] == ["XY", "YX"]


def test_all_layer_calibration_adapter_reads_main_schema(tmp_path, monkeypatch):
    vector_path = tmp_path / "concept.pt"
    torch.save({"vector": torch.tensor([3.0, 4.0])}, vector_path)
    config = SimpleNamespace(
        model_name="model",
        layers=[0, 1],
        concepts=[SimpleNamespace(name="Dust")],
    )
    rows = []
    for layer in config.layers:
        rows.extend(
            [
                {
                    "direction_id": f"concept-{layer}",
                    "direction_family": "concept",
                    "decoder_block_index": layer,
                    "concept": "Dust",
                    "source_path": str(vector_path),
                    "sd": 2.0,
                    "mad_corrected": 1.5,
                },
                {
                    "direction_id": f"random-{layer}",
                    "direction_family": "fixed_random",
                    "decoder_block_index": layer,
                    "concept": None,
                    "seed": 100 + layer,
                    "sd": 1.0,
                    "mad_corrected": 0.9,
                },
                {
                    "direction_id": f"noise-{layer}",
                    "direction_family": "renewed_noise",
                    "decoder_block_index": layer,
                    "concept": None,
                    "seed": 200 + layer,
                    "sd": 1.1,
                    "mad_corrected": 1.0,
                },
            ]
        )
    monkeypatch.setattr("experiments.experiment2_presence.REPO_ROOT", tmp_path)
    calibration = Experiment0Calibration(config, rows, {}, tmp_path, "sd")

    assert calibration.config.layers == [0, 1]
    assert calibration.direction_ids("concept", 1, "Dust") == ["concept-1"]
    assert calibration.direction_ids("random", 0) == ["random-0"]
    assert calibration.scale("concept-0") == 2.0
    assert calibration.vector("random-0").norm().item() == pytest.approx(1.0)


def test_calibration_uses_and_validates_committed_fixed_random_vector(tmp_path):
    concept_path = tmp_path / "concept.pt"
    torch.save({"vector": torch.tensor([3.0, 4.0])}, concept_path)
    direction_id = "fixed_random__block_00__0000"
    seed = 101
    generator = torch.Generator(device="cpu").manual_seed(seed)
    vector = torch.randn(2, generator=generator)
    vector = vector / torch.linalg.vector_norm(vector)
    random_path = tmp_path / "random_s0_0_avg.pt"
    torch.save(
        {
            "vector": vector,
            "model_name": "model",
            "seed_scheme": "linear",
            "draw_seed": seed,
            "sample": 0,
            "hidden_dim": 2,
            "decoder_block_index": 0,
            "experiment_0_direction_id": direction_id,
        },
        random_path,
    )
    config = SimpleNamespace(
        model_name="model",
        layers=[0],
        concepts=[SimpleNamespace(name="Dust")],
        vector_dir=tmp_path,
        vector_type="avg",
    )
    rows = [
        {
            "direction_id": "concept-0",
            "direction_family": "concept",
            "decoder_block_index": 0,
            "concept": "Dust",
            "source_path": str(concept_path),
            "sd": 2.0,
            "mad_corrected": 1.5,
        },
        {
            "direction_id": direction_id,
            "direction_family": "fixed_random",
            "decoder_block_index": 0,
            "concept": None,
            "seed": seed,
            "sd": 1.0,
            "mad_corrected": 0.9,
        },
    ]
    calibration = Experiment0Calibration(config, rows, {}, tmp_path, "sd")

    assert torch.equal(calibration.vector(direction_id), vector)
    assert calibration.provenance()["fixed_random_vector_files"] == 1


def test_build_conditions_covers_every_selected_layer():
    class Calibration:
        def direction_ids(self, kind, layer, concept=None):
            if kind == "concept":
                return [f"{concept}-{layer}"]
            return [f"{kind}-{layer}-{index}" for index in range(10)]

    args = SimpleNamespace(
        layers=list(range(32)),
        families=["concept", "random", "noise", "dropout"],
        concepts=["Dust"],
        num_random=3,
        num_noise=2,
        num_dropout=2,
    )
    conditions = build_conditions(Calibration(), args)

    assert sorted(conditions) == list(range(32))
    assert all(len(conditions[layer]) == 8 for layer in range(32))


def test_detection_cli_has_no_capability_task():
    from experiments import experiment2_presence as experiment
    destinations = {action.dest for action in experiment.build_parser()._actions}
    assert not any('capability' in name for name in destinations)
    assert not hasattr(experiment, 'run_capability')


def test_answer_token_suffix_does_not_count_as_single_token():
    class MultiToken:
        def encode(self, text, add_special_tokens=False):
            return [17, 18]
    with pytest.raises(ValueError):
        choice_token_ids(MultiToken())
