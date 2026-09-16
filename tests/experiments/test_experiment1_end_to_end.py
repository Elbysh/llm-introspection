"""One complete Experiment 1 sweep, on a small model and a synthetic Experiment 0.

The unit tests cover the prompt, the dose grids and the fits in isolation. This one
exercises what only a real run touches: hooks firing at the right token range, the
four families reaching the logits, the contamination diagnostic, and the trials
surviving a CSV round trip. It uses a randomly initialized 32-block Llama and a
calibration built the way Experiment 0 builds one, so it needs no GPU and no weights.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from experiment_0_calibration.direction_bank import DirectionBank  # noqa: E402
from experiment_0_calibration.prepare_material import materialize_direction  # noqa: E402
from experiments.experiment1_psychometrics import (  # noqa: E402
    build_conditions,
    count_forward_passes,
    load_trials,
    run_experiment,
    summarize,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_CONFIG = REPO_ROOT / "configs" / "experiment_0_calibration" / "development_full.yaml"

HIDDEN_SIZE = 64
NUM_BLOCKS = 32
VOCAB_SIZE = 256
CONCEPTS = ("Dust", "Satellites")
NUM_RANDOM = 3
NUM_NOISE_DIRECTIONS = 16


class ByteTokenizer:
    """One token per character, so a token range must recover the sentence exactly."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return (
            "<|bot|>" + messages[0]["content"] + "<|eot_id|>" + messages[1]["content"]
        )

    def __call__(self, text, return_tensors=None, add_special_tokens=False,
                 return_offsets_mapping=False):
        ids = [ord(character) % VOCAB_SIZE for character in text]
        encoding = {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        }
        if return_offsets_mapping:
            spans = [[index, index + 1] for index in range(len(text))]
            encoding["offset_mapping"] = torch.tensor([spans], dtype=torch.long)
        return encoding

    def encode(self, text, add_special_tokens=False):
        return [ord(character) % VOCAB_SIZE for character in text]

    def decode(self, ids):
        return "".join(chr(int(value)) for value in ids)

    def convert_ids_to_tokens(self, ids):
        return [chr(int(value)) for value in ids]


def _tiny_model():
    from transformers.models.llama import LlamaConfig, LlamaForCausalLM

    # Thirty-two very small blocks on a many-core node spend nearly all of their
    # time handing work between threads; capping them keeps this test in seconds.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=2 * HIDDEN_SIZE,
        num_hidden_layers=NUM_BLOCKS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        use_cache=False,
    )
    model = LlamaForCausalLM(config).to(torch.float32)
    model.eval()
    return model


def _synthetic_calibration(tmp_path):
    """A calibration with the shape and the invariants Experiment 0 produces."""
    with REAL_CONFIG.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    vector_dir = tmp_path / "vectors"
    vector_dir.mkdir()
    raw["directions"]["concepts"] = [
        {"name": name, "dataset": "simple_data", "split": "unassigned"}
        for name in CONCEPTS
    ]
    raw["directions"]["fixed_random"]["count_per_layer"] = NUM_RANDOM
    raw["paths"]["concept_vectors"] = str(vector_dir)
    raw["paths"]["plan_dir"] = str(tmp_path / "plan")
    raw["paths"]["output_dir"] = str(tmp_path / "calibration")
    config_path = tmp_path / "calibration.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle)

    generator = torch.Generator().manual_seed(11)
    records = []
    for layer in range(NUM_BLOCKS):
        hidden_state_index = layer + 1
        for concept in CONCEPTS:
            source = vector_dir / "{}_{}_avg.pt".format(concept, hidden_state_index)
            torch.save(
                {
                    "vector": torch.randn(HIDDEN_SIZE, generator=generator),
                    "model_name": "tiny",
                    "concept_name": concept,
                    "layer": hidden_state_index,
                    "vec_type": "avg",
                },
                source,
            )
            records.append(
                {
                    "direction_id": "concept__block_{:02d}__{}".format(layer, concept),
                    "direction_family": "concept",
                    "decoder_block_index": layer,
                    "concept": concept,
                    "source_path": str(source),
                    "seed": None,
                }
            )
        for index in range(NUM_RANDOM):
            records.append(
                {
                    "direction_id": "fixed_random__block_{:02d}__{:04d}".format(layer, index),
                    "direction_family": "fixed_random",
                    "decoder_block_index": layer,
                    "concept": None,
                    "source_path": None,
                    "seed": 7_000_000 + layer * 1000 + index,
                }
            )
        for index in range(NUM_NOISE_DIRECTIONS):
            records.append(
                {
                    "direction_id": "renewed_noise__block_{:02d}__token_{:04d}".format(
                        layer, index),
                    "direction_family": "renewed_noise",
                    "decoder_block_index": layer,
                    "concept": None,
                    "source_path": None,
                    "seed": 9_000_000 + layer * 1000 + index,
                }
            )

    # The recorded norm is what a rebuilt direction is checked against, so it has to
    # come from the same materialization the bank will run.
    config_dir = tmp_path / "calibration"
    config_dir.mkdir()
    scales = torch.rand(len(records), generator=generator) * 4.0 + 0.25
    for record, scale in zip(records, scales.tolist()):
        _, original_norm = materialize_direction(record, None, HIDDEN_SIZE)
        record["original_direction_norm"] = original_norm
        record["sd"] = scale
        record["mad_corrected"] = 0.7 * scale
        record["valid_for_sd_normalization"] = True
    with (config_dir / "directional_scales.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle)
    return config_path, config_dir


@pytest.fixture(scope="module")
def sweep(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("experiment1")
    config_path, calibration_dir = _synthetic_calibration(tmp_path)
    bank = DirectionBank.load(config_path, calibration_dir=calibration_dir)
    args = SimpleNamespace(
        layers=[3],
        families=["concept", "random", "noise", "dropout"],
        matchings=["alpha", "z"],
        concepts=["Dust"],
        num_random=1,
        num_noise=1,
        num_dropout=1,
        label_orders=["AB"],
        dose_grid={"alpha": [1.0, 8.0], "z": [0.5, 4.0]},
        pairs=[{"pair_id": 0, "sentence_x": "the cat sleeps here",
                "sentence_y": "a dog barks loudly", "token_length_gap": 1}],
        dropout_norm_source="trial",
        contamination_trials=2,
        measure_amplitude=True,
        limit_trials=None,
        seed=7,
        progress=False,
    )
    trials, contamination = run_experiment(_tiny_model(), ByteTokenizer(), bank, args)
    return args, bank, trials, contamination, tmp_path


def test_the_sweep_runs_the_plan_it_announced(sweep):
    args, bank, trials, _, _ = sweep
    conditions = build_conditions(bank, args)
    perturbed_passes, sham_passes = count_forward_passes(conditions, args)

    perturbed = [row for row in trials if row["kind"] == "perturbed"]
    shams = [row for row in trials if row["kind"] == "sham"]

    assert len(perturbed) == perturbed_passes
    assert len(shams) == sham_passes
    assert {row["family"] for row in perturbed} == set(args.families)
    assert all(row["finite"] for row in trials)


def test_every_family_actually_moves_the_activations(sweep):
    _, _, trials, _, _ = sweep
    largest = {}
    for row in trials:
        if row["kind"] != "perturbed" or row["matching"] != "alpha":
            continue
        key = row["family"]
        if key not in largest or row["dose"] > largest[key]["dose"]:
            largest[key] = row

    for family, row in largest.items():
        # A hook that silently misses its token range would leave this at zero.
        assert row["realized_amplitude"] > 0.0, family
        assert row["contrast"] != row["sham_contrast"], family


def test_the_z_dose_uses_the_scale_of_the_direction_it_names(sweep):
    _, bank, trials, _, _ = sweep
    for row in trials:
        if row["kind"] != "perturbed" or row["matching"] != "z":
            continue
        if row["family"] == "concept" or row["family"] == "random":
            expected = row["dose"] * bank.scale(row["direction_id"])
            assert row["alpha_requested"] == pytest.approx(expected)
        elif row["family"] == "dropout":
            expected = row["dose"] * bank.reference_scale(row["layer"])
            assert row["alpha_requested"] == pytest.approx(expected)


def test_dropout_rates_stay_inside_the_admissible_range(sweep):
    _, _, trials, _, _ = sweep
    rates = [row["dropout_rate"] for row in trials if row["family"] == "dropout"]

    assert rates
    assert all(0.0 <= rate < 1.0 for rate in rates)


def test_the_contamination_diagnostic_reports_every_probed_block(sweep):
    args, _, _, contamination, _ = sweep

    assert len(contamination) == args.contamination_trials
    for record in contamination:
        probes = [int(probe) for probe in record["ratios"]]
        # P(1->2) is only defined at or above the injection site.
        assert probes and min(probes) >= record["layer"]
        assert all(value >= 0.0 for value in record["ratios"].values())


def test_the_summary_scores_both_targets_of_each_cell(sweep):
    _, _, trials, _, _ = sweep
    per_dose, curves, sham = summarize(trials)

    assert per_dose and curves
    for row in per_dose:
        assert row["n_trials"] > 0
        assert 0.0 <= row["accuracy_raw"] <= 1.0
        assert 0.0 <= row["accuracy_adjusted"] <= 1.0
    assert sham["n_trials"] > 0
    assert {row["metric"] for row in curves} == {"raw", "adjusted"}


def test_trials_survive_the_csv_round_trip(sweep):
    import csv

    _, _, trials, _, tmp_path = sweep
    path = tmp_path / "trials.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trials[0].keys()))
        writer.writeheader()
        writer.writerows(trials)

    reloaded = load_trials(path)
    before = summarize(trials)[0]
    after = summarize(reloaded)[0]

    assert len(reloaded) == len(trials)
    assert [row["accuracy_adjusted"] for row in before] == pytest.approx(
        [row["accuracy_adjusted"] for row in after]
    )
