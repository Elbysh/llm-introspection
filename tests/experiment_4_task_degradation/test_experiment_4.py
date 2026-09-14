"""Protocol invariants using transparent tokenization and a tiny real Llama."""

import copy
import json
from unittest.mock import patch

import pytest
import torch
from experiment_0_calibration.prepare_material import file_sha256, unit
from experiment_4_task_degradation.analyze_experiment_4 import (
    analyze,
    summarize_classification,
)
from experiment_4_task_degradation.protocol_config import (
    Experiment4Config,
    validate_conditions,
)
from experiment_4_task_degradation.step_01_select_examples import (
    build_classification_prompt,
    select_examples,
)
from experiment_4_task_degradation.step_02_check_concepts import (
    check_concept_separation,
)
from experiment_4_task_degradation.step_03_run_sham import forward_at_sentence, run_sham
from experiment_4_task_degradation.step_04_apply_intervention import (
    build_additions,
    make_sentence_hook,
)
from experiment_4_task_degradation.step_05_score_responses import (
    output_js,
    score_response,
)


class CharacterTokenizer:
    def __init__(self):
        self.init_kwargs = {}

    def apply_chat_template(self, messages, **kwargs):
        return "User:\n" + messages[0]["content"] + "\nAssistant:\n"

    def __call__(self, text, return_offsets_mapping=False, **kwargs):
        result = {"input_ids": [ord(c) for c in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i+1) for i in range(len(text))]
        return result

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)


@pytest.fixture
def tokenizer():
    return CharacterTokenizer()


@pytest.fixture
def model():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(5)
    return LlamaForCausalLM(LlamaConfig(vocab_size=256, hidden_size=16,
        intermediate_size=24, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, attention_dropout=0.)).eval()


@pytest.fixture
def labelled():
    return select_examples({"betrayal": [["A secret."], ["A promise."]]},
                           [{"probed_concept": "betrayal", "positive_indices": [0], "negative_indices": [0]}])


def test_balanced_selection_mapping_and_token_spans(tokenizer, labelled):
    assert [r["label"] for r in labelled] == [True, False]
    for example in labelled:
        prompts = [build_classification_prompt(tokenizer, example, mapping) for mapping in ("X", "Y")]
        assert prompts[0]["correct_label"] != prompts[1]["correct_label"]
        for prompt in prompts:
            chars = "".join(prompt["rendered_text"][i] for i in prompt["target_token_indices"])
            assert chars == example["sentence"]
            assert prompt["rendered_text"].endswith("Assistant:\n")
            assert prompt["answer_token_ids"] == {"X": ord("X"), "Y": ord("Y")}


def test_direct_concept_and_training_leakage_rejected(labelled):
    row = {"direction_family": "concept", "concept": "betrayal", "concept_dataset": "complex_data"}
    with pytest.raises(ValueError, match="must differ"):
        check_concept_separation(labelled[0], row, {})
    row["concept"] = "another"
    with pytest.raises(ValueError, match="construction"):
        check_concept_separation(labelled[0], row, {"another": [["A secret."], ["Else."]]})
    row.update(concept="Dust", concept_dataset="simple_data")
    check_concept_separation(labelled[0], row, {})


@pytest.mark.parametrize("tuple_output", [False, True])
def test_hook_localization_realized_norm_and_tokenwise_noise(tuple_output):
    h = torch.randn(1, 7, 16)
    additions = torch.zeros(2, 16)
    additions[0, 0], additions[1, 1] = 2, 3
    measured = {}
    hook = make_sentence_hook({"target_token_indices": [2, 3]}, additions, measured)
    result = hook(None, None, (h, "cache") if tuple_output else h)
    changed = result[0] if tuple_output else result
    assert torch.equal(changed[:, :2], h[:, :2]) and torch.equal(changed[:, 4:], h[:, 4:])
    assert measured["alpha_realized_per_token"] == pytest.approx([2, 3])
    assert not torch.equal(changed[0, 2]-h[0, 2], changed[0, 3]-h[0, 3])
    if tuple_output:
        assert result[1] == "cache"


def test_realized_norm_accounts_for_dtype_rounding():
    h = torch.full((1, 3, 4), 1024., dtype=torch.bfloat16)
    measurements = {}
    hook = make_sentence_hook({"target_token_indices": [1]}, torch.full((1, 4), .01), measurements)
    hook(None, None, h)
    assert measurements["alpha_realized_per_token"] == [0.]


def test_response_not_forced_into_xy_and_js(tokenizer, labelled):
    prompt = build_classification_prompt(tokenizer, labelled[0], "X")
    logits = torch.zeros(256)
    logits[ord("X")], logits[ord("Y")], logits[ord("Z")] = 3, 1, 5
    scored = score_response(logits, prompt, tokenizer)
    assert scored["response_text"] == "Z" and not scored["valid_response"]
    assert scored["accuracy"] == 0 and scored["forced_label_accuracy"] == 1
    assert scored["correct_margin"] == 2
    assert output_js(logits, logits) == pytest.approx(0., abs=1e-14)
    assert 0 < output_js(logits, -logits) < .693148


def test_single_forward_no_cache_and_hook_cleanup(model, tokenizer, labelled):
    prompt = build_classification_prompt(tokenizer, labelled[0], "X")
    clean, _ = run_sham(model, tokenizer, prompt, 0)
    with torch.no_grad():
        ids = torch.tensor([prompt["input_ids"]])
        expected = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1]
    assert torch.equal(clean, expected)
    with patch.object(model, "forward", side_effect=RuntimeError("test failure")), pytest.raises(RuntimeError):
        forward_at_sentence(model, prompt, 0, lambda m, a, o: o)
    assert not model.model.layers[0]._forward_hooks


def test_alpha_z_and_mad_use_actual_direction_scales():
    scales = {f"n{i}": {"decoder_block_index": 0, "direction_family": "renewed_noise",
                        "seed": i, "sd": i+1., "mad_corrected": (i+1.)/2}
              for i in range(2)}
    condition = {"family": "renewed_noise", "layer": 0, "dose_axis": "z", "dose": 3., "scale_statistic": "sd"}
    additions, requested, _ = build_additions(condition, ["n0", "n1"], scales, 16)
    assert requested == [3, 6]
    assert additions.norm(dim=1).tolist() == pytest.approx([3, 6])
    condition["scale_statistic"] = "mad_corrected"
    assert build_additions(condition, ["n0", "n1"], scales, 16)[1] == [1.5, 3]
    condition["dose_axis"] = "alpha"
    assert build_additions(condition, ["n0", "n1"], scales, 16)[1] == [3, 3]


@pytest.fixture
def prepared_inputs(tmp_path, tokenizer):
    from experiment_4_task_degradation.protocol_config import write_json
    dataset_path = tmp_path / "dataset.json"
    write_json(dataset_path, {"betrayal": [["A secret."], ["A promise."]]})
    vector_path = tmp_path / "Dust.pt"
    torch.save({"vector": torch.arange(16.)}, vector_path)
    scales_path = tmp_path / "scales.json"
    write_json(scales_path, [{"direction_id": "dust0", "direction_family": "concept", "concept": "Dust",
        "concept_dataset": "simple_data", "source_path": str(vector_path),
        "source_sha256": file_sha256(vector_path), "decoder_block_index": 0, "sd": 2., "mad_corrected": 1.}])
    calibration_path = tmp_path / "calibration.json"
    write_json(calibration_path, {"model_name": "toy", "requested_model_revision": "main", "protocol_status": "development"})
    source_model = {"name": "toy", "revision": "main", "tokenizer_revision": "main"}
    condition = {"condition_id": "concept0alpha1", "family": "concept", "intervention_id": "dust0",
                 "layer": 0, "dose_axis": "alpha", "dose": 1., "scale_statistic": "sd",
                 "direction_id": "dust0", "probed_concepts": ["betrayal"]}
    raw = {"protocol": {"status": "development", "version": "test"}, "model": source_model,
           "examples": [{"probed_concept": "betrayal", "positive_indices": [0], "negative_indices": [0]}],
           "diagnostics": {"js_example_ids": ["betrayal__positive__00"]},
           "execution": {"dtype": "float32"}, "paths": {"complex_data": str(dataset_path),
               "directional_scales": str(scales_path), "calibration_manifest": str(calibration_path),
               "plan_dir": str(tmp_path / "plan"),
               "output_dir": str(tmp_path / "run")}}
    raw["conditions"] = [condition]
    config_path = tmp_path / "config.json"
    write_json(config_path, raw)
    config = Experiment4Config(raw, config_path)
    config.validate()
    return config, condition


def test_prepare_run_analysis_and_no_overwrite(prepared_inputs, model, tokenizer):
    from experiment_0_calibration.prepare_material import read_jsonl
    from experiment_4_task_degradation.prepare_material_plan import prepare
    from experiment_4_task_degradation.run_experiment_4 import run
    config, _condition = prepared_inputs
    manifest = prepare(config, tokenizer)
    assert manifest["n_prompts"] == manifest["n_interventions"] == manifest["n_sham"] == 4
    assert manifest["n_js"] == 2
    assert run(config, model, tokenizer) == 8
    rows = read_jsonl(config.output_dir / "trials.jsonl")
    assert sum(r["family"] == "sham" for r in rows) == 4
    classification = summarize_classification(rows)
    assert classification[0]["n_js"] == 2
    output = config.output_dir.parent / "analysis"
    assert analyze(config.output_dir, output) == classification
    assert (output / "classification_summary.csv").exists()
    assert (output / "classification__block_00__alpha__sd.png").exists()
    assert not list(output.glob("*detection*"))
    with pytest.raises(FileExistsError):
        prepare(config, tokenizer)
    with pytest.raises(FileExistsError):
        run(config, model, tokenizer)


def test_config_rejects_unbalanced_and_dropout(prepared_inputs):
    config, condition = prepared_inputs
    raw = copy.deepcopy(config.raw)
    raw["examples"][0]["negative_indices"] = []
    with pytest.raises(ValueError, match="equally many"):
        Experiment4Config(raw, config.source_path).validate()
    condition["family"] = "dropout"
    with pytest.raises(ValueError, match="Unknown"):
        validate_conditions(config)


def test_changed_plan_rejected_before_run(prepared_inputs, tokenizer):
    from experiment_4_task_degradation.prepare_material_plan import prepare
    from experiment_4_task_degradation.run_experiment_4 import validate_plan
    config, _ = prepared_inputs
    prepare(config, tokenizer)
    with (config.plan_dir / "trials.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="artifact changed"):
        validate_plan(config)


def test_all_additive_families_noise_independence_and_sham_reuse(prepared_inputs, model, tokenizer):
    from experiment_0_calibration.prepare_material import read_jsonl
    from experiment_4_task_degradation.prepare_material_plan import prepare
    from experiment_4_task_degradation.protocol_config import read_json
    from experiment_4_task_degradation.run_experiment_4 import run
    config, concept = prepared_inputs
    records = read_json(config.path("directional_scales"))
    seed = 987
    fixed, _ = unit(torch.randn(16, generator=torch.Generator().manual_seed(seed)))
    source = config.output_dir.parent / "random.pt"
    torch.save({"vector": fixed, "draw_seed": seed}, source)
    records.append({"direction_id": "random0", "direction_family": "fixed_random",
                    "decoder_block_index": 0, "seed": seed,
                    "source_path": str(source), "source_sha256": file_sha256(source),
                    "sd": 2., "mad_corrected": 1.})
    records.extend({"direction_id": f"noise{i:03d}", "direction_family": "renewed_noise",
                    "decoder_block_index": 0, "seed": i, "sd": .5+i/100,
                    "mad_corrected": .3+i/100} for i in range(80))
    config.path("directional_scales").write_text(json.dumps(records), encoding="utf-8")
    config.raw["conditions"].append({**concept, "family": "fixed_random", "condition_id": "random_z",
        "intervention_id": "random0", "direction_id": "random0", "dose_axis": "z", "dose": 2.})
    config.raw["conditions"].append({**concept, "family": "renewed_noise", "condition_id": "noise_z",
        "intervention_id": "noise_rule", "noise_rule": "independent_unit_gaussian_per_trial_token",
        "dose_axis": "z", "dose": 2.})
    manifest = prepare(config, tokenizer)
    assert manifest["n_interventions"] == 12 and manifest["n_sham"] == 4
    trials = read_jsonl(config.plan_dir / "trials.jsonl")
    draws = [d for t in trials if t["condition_id"] == "noise_z" for d in t["direction_ids"]]
    assert len(draws) == len(set(draws))
    assert run(config, model, tokenizer) == 16
    rows = read_jsonl(config.output_dir / "trials.jsonl")
    noisy = [r for r in rows if r["family"] == "renewed_noise"]
    assert all(len(r["alpha_requested_per_token"]) == len(r["target_token_indices"]) for r in noisy)
    assert all(len(set(r["natural_scale_per_token"])) > 1 for r in noisy)


def test_insufficient_noise_calibration_is_not_silently_reused(prepared_inputs, tokenizer):
    from experiment_4_task_degradation.prepare_material_plan import prepare
    config, condition = prepared_inputs
    condition.update(family="renewed_noise", noise_rule="independent_unit_gaussian_per_trial_token")
    with pytest.raises(ValueError, match="Insufficient calibrated"):
        prepare(config, tokenizer)
    assert not config.plan_dir.exists()


def test_multi_token_label_continuation_is_rejected(tokenizer, labelled):
    class BadContinuationTokenizer(CharacterTokenizer):
        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            if text.endswith("Assistant:\nX"):
                result["input_ids"].append(42)
            return result
    with pytest.raises(ValueError, match="single continuation"):
        build_classification_prompt(BadContinuationTokenizer(), labelled[0], "X")
