"""Small causal-model tests; no model weights or network required."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "utils"))
from position_detection_utils import (
    PREFIX, SUFFIX, accuracy, build_prompt, condition_metrics, forward_counts, select_pairs,
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_module("position_detection", ROOT / "code" / "experiments" / "position_detection.py")
analysis = load_module("position_analysis", ROOT / "code" / "analysis" / "compute_position_detection_accuracy.py")


class CharacterTokenizer:
    """Transparent offsets make expected spans independently inspectable."""
    is_fast = True

    def apply_chat_template(self, messages, **kwargs):
        return "<user>" + messages[0]["content"] + "<assistant>"

    def __call__(self, text, **kwargs):
        # Match the actual answer convention: a leading space plus A/B is one token.
        ids, offsets = [], []
        i = 0
        while i < len(text):
            if text[i:i+2] in (" A", " B"):
                ids.append(200 if text[i+1] == "A" else 201)
                offsets.append((i, i+2))
                i += 2
            else:
                ids.append(ord(text[i]) % 190)
                offsets.append((i, i+1))
                i += 1
        return {"input_ids": ids, "offset_mapping": offsets}

    def convert_ids_to_tokens(self, ids):
        return [str(i) for i in ids]


@pytest.fixture
def tiny_model():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=256, hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=1024, attention_dropout=0,
    )
    config._attn_implementation = "eager"
    return LlamaForCausalLM(config).eval()


@pytest.fixture
def prompt():
    p = build_prompt(CharacterTokenizer(), "Dogs run.", "Cats sit.")
    p.update(prompt_id="p", pair_id="pair", content_order="xy")
    return p


def test_prompt_exact_text_spans_and_answer(prompt):
    assert prompt["user_message"] == PREFIX + "A) Dogs run.\nB) Cats sit." + SUFFIX
    assert prompt["formatted_prompt"].endswith("<assistant>The answer is")
    for s in prompt["sentences"]:
        lo, hi = s["char_span"]
        assert prompt["formatted_prompt"][lo:hi] == s["text"]
        assert len(s["token_indices"]) == len(s["text"])
        assert s["distances_to_answer_position"][0] == len(prompt["input_ids"]) - s["token_indices"][0]
    assert prompt["answer_token_ids"] == {"A": 200, "B": 201}
    assert prompt["tokens_between_sentences"] == 4  # newline, B, ), space


def test_pair_selection_reproducible_unique_and_counterbalanced():
    sentences = ["Dogs run.", "Cats sit.", "Bird fly.", "Ox walks.", "Longer sentence."]
    a = select_pairs(CharacterTokenizer(), sentences, 3, 42)
    assert a == select_pairs(CharacterTokenizer(), sentences, 3, 42)
    assert a["eligible_pair_count"] == 6
    assert len({p["pair_id"] for p in a["pairs"]}) == 3
    assert len(a["prompts"]) == 12
    assert {(p["content_order"], p["label_mapping"]) for p in a["prompts"]} == {
        ("xy", "AB"), ("xy", "BA"), ("yx", "AB"), ("yx", "BA")}
    with pytest.raises(ValueError, match="only 6"):
        select_pairs(CharacterTokenizer(), sentences, 7)


def test_budget():
    counts = forward_counts(30, list(range(0,31,3)), 10, 4)
    assert counts == {"controls": 120, "injections": 105600, "restorations": 897600}
    assert sum(counts.values()) == 1003320


def test_metrics_use_matched_baseline_and_labels():
    rows = [{"L": 7., "target_label": "B"}, {"L": 10., "target_label": "A"}]
    metrics = condition_metrics(8., rows, "B")
    assert [r["accuracy_raw"] for r in rows] == [0., 1.]
    assert [r["accuracy_adjusted"] for r in rows] == [1., 1.]
    assert metrics == {"S": 1.5, "S_position": -1.5}
    assert accuracy(0, "A") == accuracy(0, "B") == .5


def test_forward_scores_full_prompt_once_and_targets_exact_tokens(tiny_model, prompt):
    captures = []
    handle = tiny_model.model.layers[0].register_forward_hook(
        lambda m, i, o: captures.append((o[0] if isinstance(o, tuple) else o).detach().clone()))
    clean, _, _ = runner.forward(tiny_model, prompt)
    with torch.inference_mode():
        logits = tiny_model(torch.tensor([prompt["input_ids"]]), use_cache=False).logits[0, -1]
    assert len(captures) == 2  # runner performs one pass, followed by our reference pass
    assert clean["L"] == pytest.approx(logits[200].item() - logits[201].item(), abs=1e-7)
    handle.remove()
    before = captures[0]
    # Observe after runner's injection hook, using a downstream pre-hook.
    observed = []
    handle = tiny_model.model.layers[1].register_forward_pre_hook(
        lambda m, inputs: observed.append(inputs[0].detach().clone()))
    vector = torch.ones(16) / 4
    _, _, info = runner.forward(tiny_model, prompt, injection=(0, 1, vector, 2.))
    handle.remove()
    difference = observed[0] - before
    lo, hi = prompt["sentences"][0]["token_span"]
    assert torch.count_nonzero(difference[:, :lo]) == 0
    assert torch.count_nonzero(difference[:, hi:]) == 0
    torch.testing.assert_close(difference[:, lo:hi], torch.full_like(difference[:, lo:hi], .5))
    assert info["realized_per_token_l2"] == pytest.approx([2.] * (hi-lo))
    assert all(not layer._forward_hooks for layer in tiny_model.model.layers)


def test_zero_injection_and_restoration_identity(tiny_model, prompt):
    control, clean, _ = runner.forward(tiny_model, prompt, capture_layers=range(3))
    vector = torch.ones(16) / 4
    row, captures = runner.run_condition(tiny_model, prompt, control, clean, vector, "Dust", 0, 0., 1e-8)
    assert row["S"] == 0
    assert all(r["accuracy_adjusted"] == .5 for r in row["injections"])
    assert all(d["P"] == d["E"] == 0 for d in row["diagnostics"])
    for layer in clean:
        torch.testing.assert_close(captures[layer], clean[layer], rtol=0, atol=0)


def test_causal_contamination_and_independent_restorations(tiny_model, prompt):
    control, clean, _ = runner.forward(tiny_model, prompt, capture_layers=range(3))
    vector = torch.arange(16).float(); vector /= vector.norm()
    row, _ = runner.run_condition(tiny_model, prompt, control, clean, vector, "Dust", 0, 2., 1e-8)
    assert row["diagnostics"][0]["P"] == row["diagnostics"][0]["E"] == 0
    assert row["diagnostics"][1]["P"] > 0
    # Restoring another position after the final attention cannot affect the answer.
    assert row["diagnostics"][-1]["E"] == 0
    for d in row["diagnostics"]:
        restored, _, _ = runner.forward(tiny_model, prompt, injection=(0,1,vector,2.),
                                         restoration=(d["layer"], clean[d["layer"]]))
        assert d["E"] == row["injections"][0]["L"] - restored["L"]
    # Restored block output must exactly equal its control slice.
    _, captured, _ = runner.forward(tiny_model, prompt, injection=(0,1,vector,2.),
                                     restoration=(1,clean[1]), capture_layers=[1])
    torch.testing.assert_close(captured[1], clean[1], rtol=0, atol=0)


def test_hooks_removed_after_failure(tiny_model, prompt):
    with pytest.raises(ValueError, match="shape"):
        runner.forward(tiny_model, prompt, restoration=(0, torch.zeros(1, 2)))
    assert all(not layer._forward_hooks for layer in tiny_model.model.layers)


def test_saved_analysis_and_partial_status(tiny_model, prompt, tmp_path):
    prompt.update(prompt_id="p", pair_id="pair", content_order="xy")
    control, clean, _ = runner.forward(tiny_model, prompt, capture_layers=range(3))
    row, captured = runner.run_condition(tiny_model, prompt, control, clean, torch.ones(16)/4,
                                         "Dust", 0, 0., 1e-8)
    row.update(activation_file="activations/p.pt")
    runner.atomic_tensor(tmp_path / row["activation_file"], captured)
    runner.atomic_json(tmp_path / "controls/p.json", control)
    runner.atomic_json(tmp_path / "conditions/p/dust.json", row)
    manifest = {"schema_version": 1, "prompts": [prompt],
                "config": {"concepts": ["Dust"], "layers": [0], "alphas": [0.]}}
    runner.atomic_json(tmp_path / "manifest.json", manifest)
    result = analysis.analyze(tmp_path)
    assert result["complete"]
    assert result["summary"][0]["accuracy_adjusted"] == .5
    manifest["config"]["alphas"].append(2.)
    runner.atomic_json(tmp_path / "manifest.json", manifest)
    assert not analysis.analyze(tmp_path)["complete"]


def test_cli_run_resume_and_configuration_guard(monkeypatch, tmp_path):
    torch.set_num_threads(1)
    config = LlamaConfig(vocab_size=256, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=32, num_attention_heads=2, num_key_value_heads=2)
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    tokenizer = CharacterTokenizer()
    tokenizer.backend_tokenizer = type("Backend", (), {"to_str": lambda self: "test tokenizer"})()
    tokenizer.chat_template = "test template"
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: model)
    vector_dir = tmp_path / "vectors"
    vector_dir.mkdir()
    torch.save({"vector": torch.ones(8), "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "layer": 31, "concept_name": "Dust", "vec_type": "avg"}, vector_dir / "Dust_31_avg.pt")
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps(["Dogs run.", "Cats sit."]))
    out = tmp_path / "results"
    argv = ["position_detection.py", "--num-pairs", "1", "--concepts", "Dust", "--layers", "31",
            "--alphas", "0", "--corpus", str(corpus), "--vector-dir", str(vector_dir),
            "--output-dir", str(out), "--device", "cpu", "--dtype", "float32"]
    monkeypatch.setattr(sys, "argv", argv)
    calls = []
    handle = model.register_forward_hook(lambda *args: calls.append(1))
    with pytest.warns(UserWarning):
        runner.main()
    assert len(calls) == 16  # 4 prompts * (control + two injections + one restoration)
    assert analysis.analyze(out)["complete"]
    records = list((out / "conditions").glob("*/*.json"))
    assert len(records) == 4
    original = {p: p.read_bytes() for p in records}
    monkeypatch.setattr(sys, "argv", argv + ["--resume"])
    with pytest.warns(UserWarning):
        runner.main()
    assert len(calls) == 16  # completed conditions and controls are reused
    assert original == {p: p.read_bytes() for p in records}
    # Simulate interruption after matrix write but before condition JSON commit.
    records[0].unlink()
    with pytest.warns(UserWarning):
        runner.main()
    assert len(calls) == 19  # only the incomplete condition is rerun
    assert analysis.analyze(out)["complete"]
    monkeypatch.setattr(sys, "argv", argv + ["--resume", "--seed", "43"])
    with pytest.raises(ValueError, match="Resume configuration"):
        runner.main()
    handle.remove()
