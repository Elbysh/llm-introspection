"""Protocol invariants and a complete tiny-Llama Experiment 5 run."""

import copy

import pytest
import torch
from experiment_0_calibration.prepare_material import file_sha256, read_jsonl
from experiment_5_text_induction.measure_presence import (
    auroc,
    paired_blocks,
    paired_uncertainty,
    presence_metrics,
)
from experiment_5_text_induction.protocol_config import Experiment5Config, write_json
from experiment_5_text_induction.step_01_select_text_pairs import select_text_pairs
from experiment_5_text_induction.step_03_fix_intervention import fix_intervention
from experiment_5_text_induction.step_04_counterbalance_prompts import (
    build_presence_prompt,
    counterbalanced_prompts,
)
from experiment_5_text_induction.step_05_score_presence import score_presence


class CharacterTokenizer:
    def __init__(self):
        self.init_kwargs = {}

    def apply_chat_template(self, messages, **kwargs):
        return "User:\n" + messages[0]["content"] + "\nAssistant:\n"

    def __call__(self, text, return_offsets_mapping=False, **kwargs):
        result = {"input_ids": [ord(c) for c in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)


@pytest.fixture
def tokenizer():
    return CharacterTokenizer()


@pytest.fixture
def model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(15)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=256,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            attention_dropout=0.0,
        )
    ).eval()


@pytest.fixture
def pair():
    return select_text_pairs(
        [
            {
                "pair_id": "dust_pair_0",
                "concept": "Dust",
                "neutral_sentence_index": 0,
                "evocative_text": "A thick layer of dust covered the quiet wooden table.",
                "companion_sentence_index": 1,
                "reviewed": True,
            }
        ],
        ["The table stands in the room.", "The window opens in the morning."],
    )[0]


def test_text_selection_requires_review_and_distinct_corpus_texts():
    spec = {
        "pair_id": "p",
        "concept": "Dust",
        "neutral_sentence_index": 0,
        "evocative_text": "Dust covers the table.",
        "companion_sentence_index": 1,
        "reviewed": False,
    }
    with pytest.raises(ValueError, match="Review"):
        select_text_pairs([spec], ["A table.", "A window."])
    spec["reviewed"] = True
    selected = select_text_pairs([spec], ["A table.", "A window."])[0]
    assert selected["neutral_text"] == "A table."
    with pytest.raises(ValueError, match="Duplicate"):
        select_text_pairs([spec, {**spec, "pair_id": "q"}], ["A table.", "A window."])


def test_exact_prompt_counterbalancing_and_no_experiment_name(tokenizer, pair):
    prompts = counterbalanced_prompts(tokenizer, [pair], 100)
    assert len(prompts) == 8
    assert {(p["text_version"], p["target_label"], p["present_label"]) for p in prompts} == {
        (v, t, m) for v in ("neutral", "evocative") for t in ("A", "B") for m in ("X", "Y")
    }
    for prompt in prompts:
        targeted = "".join(prompt["rendered_text"][i] for i in prompt["target_token_indices"])
        assert targeted == pair[f"{prompt['text_version']}_text"]
        # The concept may occur inside the authored evocative sentence. The
        # surrounding presence prompt must not name or announce it.
        outside_target = prompt["rendered_text"]
        for index in sorted(prompt["target_token_indices"], reverse=True):
            outside_target = outside_target[:index] + outside_target[index + 1 :]
        assert "Dust" not in outside_target
        assert "evocative" not in prompt["rendered_text"].lower()
        assert prompt["answer_token_ids"] == {"X": ord("X"), "Y": ord("Y")}
    with pytest.raises(ValueError, match="token-length"):
        counterbalanced_prompts(tokenizer, [pair], 0)


def test_same_direction_and_amplitude_are_fixed_independent_of_text(tmp_path):
    vector = tmp_path / "Dust.pt"
    torch.save({"vector": torch.arange(16.0)}, vector)
    record = {
        "direction_id": "dust",
        "direction_family": "concept",
        "concept": "Dust",
        "decoder_block_index": 0,
        "source_path": str(vector),
        "source_sha256": file_sha256(vector),
        "sd": 2.0,
        "mad_corrected": 1.0,
    }
    condition = {
        "condition_id": "c",
        "concept": "Dust",
        "direction_id": "dust",
        "layer": 0,
        "dose_axis": "z",
        "dose": 3.0,
        "dose_role": "near_threshold",
        "scale_statistic": "sd",
    }
    fixed = fix_intervention(condition, {"dust": record})
    assert fixed["alpha_requested"] == 6.0
    assert "text_version" not in fixed


def test_presence_recoding_invalid_policy_and_tied_auroc(tokenizer, pair):
    prompt = build_presence_prompt(tokenizer, pair, "neutral", "A", "Y")
    logits = torch.zeros(256)
    logits[ord("X")], logits[ord("Y")], logits[ord("Z")] = 1, 3, 5
    sham = score_presence(logits, prompt, tokenizer, injected=False)
    injected = score_presence(logits, prompt, tokenizer, injected=True)
    assert not sham["valid_response"] and sham["reported_presence"] is None
    assert sham["presence_margin"] == 2
    assert injected["correct_margin"] == 2
    assert sham["accuracy"] == injected["accuracy"] == 0
    assert auroc([False, True], [0.0, 0.0]) == 0.5


def make_block(condition="c", pair_id="p", concept="Dust", values=(False, True, True, True)):
    rows = []
    for target in ("A", "B"):
        for mapping in ("X", "Y"):
            for version, injected, present in zip(
                ("neutral", "evocative", "neutral", "evocative"),
                (False, False, True, True),
                values,
            ):
                rows.append(
                    {
                        "condition_id": condition,
                        "pair_id": pair_id,
                        "concept": concept,
                        "direction_id": "dust",
                        "layer": 0,
                        "dose_axis": "alpha",
                        "dose": 1.0,
                        "dose_role": "near_threshold",
                        "scale_statistic": "sd",
                        "target_label": target,
                        "present_label": mapping,
                        "text_version": version,
                        "injected": injected,
                        "reported_presence": present,
                        "presence_margin": float(present),
                        "valid_response": True,
                        "neutral_sentence_sha256": pair_id + "n",
                        "evocative_sentence_sha256": pair_id + "e",
                        "companion_sentence_sha256": pair_id + "c",
                        "planned_injection_alpha": 1.0,
                    }
                )
    return rows


def test_primary_paired_false_positive_effect_and_uncertainty():
    rows = make_block(values=(False, True, True, True))
    blocks = paired_blocks(rows)
    assert len(blocks) == 4
    assert all(block["delta_false_positive_rate"] == 1 for block in blocks)
    estimate = paired_uncertainty(
        blocks + paired_blocks(make_block("d", "q", values=(False, False, True, True))),
        {"bootstrap_resamples": 20, "bootstrap_seed": 9, "ci_level": 0.95},
    )
    assert estimate["delta_false_positive_rate"] == 0.5
    assert estimate["bootstrap_effective_resamples"] == 20
    assert estimate["delta_false_positive_rate_ci_low"] is not None


def test_metrics_count_invalid_as_neither_report_and_incorrect():
    rows = [
        {"injected": False, "reported_presence": None, "valid_response": False, "presence_margin": 0.0},
        {"injected": True, "reported_presence": True, "valid_response": True, "presence_margin": 1.0},
    ]
    result = presence_metrics(rows)
    assert result["false_positive_rate"] == 0
    assert result["correct_rejection_rate"] == 0
    assert result["balanced_accuracy"] == 0.5
    assert result["invalid_rate_sham"] == 1


@pytest.fixture
def prepared_config(tmp_path, tokenizer):
    vector = tmp_path / "Dust.pt"
    torch.save({"vector": torch.arange(16.0)}, vector)
    scales = tmp_path / "scales.json"
    write_json(
        scales,
        [
            {
                "direction_id": "dust",
                "direction_family": "concept",
                "concept": "Dust",
                "decoder_block_index": 0,
                "source_path": str(vector),
                "source_sha256": file_sha256(vector),
                "sd": 2.0,
                "mad_corrected": 1.0,
            }
        ],
    )
    calibration = tmp_path / "calibration.json"
    write_json(
        calibration,
        {
            "model_name": "toy",
            "requested_model_revision": "main",
            "resolved_model_revision": None,
            "protocol_status": "development",
        },
    )
    text_pairs = tmp_path / "pairs.json"
    write_json(
        text_pairs,
        [
            {
                "pair_id": "p",
                "concept": "Dust",
                "neutral_sentence_index": 0,
                "evocative_text": "Dust covered the quiet table.",
                "companion_sentence_index": 1,
                "reviewed": True,
            }
        ],
    )
    condition = {
        "condition_id": "dust_alpha_1",
        "concept": "Dust",
        "family": "concept",
        "direction_id": "dust",
        "layer": 0,
        "dose_axis": "alpha",
        "dose": 1.0,
        "dose_role": "near_threshold",
        "scale_statistic": "sd",
    }
    raw = {
        "protocol": {"version": "test", "status": "development"},
        "model": {"name": "toy", "revision": "main", "tokenizer_revision": "main"},
        "texts": {"pairs": str(text_pairs), "max_token_length_difference": 20},
        "conditions": [condition],
        "statistics": {"bootstrap_resamples": 20, "bootstrap_seed": 7, "ci_level": 0.95},
        "execution": {"dtype": "float32"},
        "paths": {
            "text_pairs": str(text_pairs),
            "directional_scales": str(scales),
            "calibration_manifest": str(calibration),
            "plan_dir": str(tmp_path / "plan"),
            "output_dir": str(tmp_path / "run"),
        },
    }
    source = tmp_path / "config.json"
    write_json(source, raw)
    config = Experiment5Config(raw, source)
    config.validate()
    return config


def test_full_prepare_run_analysis_and_no_overwrite(prepared_config, model, tokenizer, monkeypatch):
    import experiment_5_text_induction.prepare_material_plan as prepare_module
    from experiment_5_text_induction.analyze_experiment_5 import analyze
    from experiment_5_text_induction.prepare_material_plan import prepare
    from experiment_5_text_induction.run_experiment_5 import run

    monkeypatch.setattr(
        prepare_module,
        "load_protocol_corpus",
        lambda: ["The table stands in the room.", "The window opens in the morning."]
        + [f"Sentence {i}." for i in range(98)],
    )
    manifest = prepare(prepared_config, tokenizer)
    assert manifest["n_trials"] == 16
    assert manifest["n_sham"] == manifest["n_injected"] == 8
    assert run(prepared_config, model, tokenizer) == 16
    rows = read_jsonl(prepared_config.path("output_dir") / "trials.jsonl")
    assert len({r["trial_id"] for r in rows}) == 16
    assert {r["text_version"] for r in rows} == {"neutral", "evocative"}
    analysis = prepared_config.path("output_dir").parent / "analysis"
    summary, differences = analyze(prepared_config.path("output_dir"), analysis)
    assert len(summary) == 2
    assert len([d for d in differences if d["scope"] == "condition"]) == 1
    for name in ("presence_summary.csv", "mapping_and_position.csv", "paired_text_effects.csv",
                 "analysis_manifest.json", "presence_000_block_00_alpha_sd.png"):
        assert (analysis / name).stat().st_size > 0
    with pytest.raises(FileExistsError):
        prepare(prepared_config, tokenizer)
    with pytest.raises(FileExistsError):
        run(prepared_config, model, tokenizer)
    with pytest.raises(FileExistsError):
        analyze(prepared_config.path("output_dir"), analysis)


def test_config_rejects_nonconcept_and_unfrozen_layers(prepared_config):
    raw = copy.deepcopy(prepared_config.raw)
    raw["conditions"][0]["family"] = "dropout"
    with pytest.raises(ValueError, match="only the concept"):
        Experiment5Config(raw, prepared_config.source_path).validate()
