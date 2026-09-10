"""Experiment 0, material and steps 1-2: auditable calibration plans."""

import hashlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

from .protocol_config import Experiment0Config, REPO_ROOT


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol_corpus() -> List[str]:
    """Load and validate the 100 distinct Experiment 0 sentences."""
    source = REPO_ROOT / "code" / "utils" / "all_prompts.py"
    spec = importlib.util.spec_from_file_location("experiment_0_prompts", str(source))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load LOCALIZATION_SENTENCES")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sentences = list(module.LOCALIZATION_SENTENCES)
    if len(sentences) != 100 or len(set(sentences)) != 100:
        raise ValueError("Experiment 0 requires exactly 100 distinct sentences")
    return sentences


def build_context_and_observation_rows(
    tokenizer: Any,
    sentences: Sequence[str],
    config: Experiment0Config,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Resolve configured contexts and exact admissible token positions.

    Development mode renders one template per sentence. A frozen run must use
    an external JSONL manifest with rows shaped as
    `{context_id, rendered_text, targets: [{sentence_id, char_start, char_end}]}`.
    This supports paired prompts and repeated sentence presentations without
    coupling Experiment 0 to behavioural-task code.
    """
    corpus_by_id = {
        "localization_{:03d}".format(index): sentence
        for index, sentence in enumerate(sentences)
    }
    if config.presentation_mode == "template_per_sentence":
        prefix, suffix = config.context_template.split("{sentence}")
        specifications = []
        for sentence_id, sentence in corpus_by_id.items():
            rendered = prefix + sentence + suffix
            specifications.append(
                {
                    "context_id": "{}__{}".format(config.context_id, sentence_id),
                    "rendered_text": rendered,
                    "targets": [
                        {
                            "sentence_id": sentence_id,
                            "char_start": len(prefix),
                            "char_end": len(prefix) + len(sentence),
                        }
                    ],
                }
            )
    else:
        specifications = read_jsonl(config.context_manifest)

    context_ids = [row["context_id"] for row in specifications]
    if len(context_ids) != len(set(context_ids)):
        raise ValueError("presentation context IDs must be unique")
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", str(value)) for value in context_ids):
        raise ValueError("context IDs may contain only letters, numbers, dot, dash and underscore")
    contexts = []
    observations = []
    observation_index = 0
    observed_sentence_ids = set()
    for specification in specifications:
        context_id = str(specification["context_id"])
        rendered = str(specification["rendered_text"])
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        input_ids = list(encoded["input_ids"])
        offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
        context_positions = []
        resolved_targets = []
        for target_index, target in enumerate(specification["targets"]):
            sentence_id = str(target["sentence_id"])
            if sentence_id not in corpus_by_id:
                raise ValueError("unknown corpus sentence ID: {}".format(sentence_id))
            sentence = corpus_by_id[sentence_id]
            char_start = int(target["char_start"])
            char_end = int(target["char_end"])
            if rendered[char_start:char_end] != sentence:
                raise ValueError("context span does not equal {}".format(sentence_id))
            observed_sentence_ids.add(sentence_id)
            admissible_positions = []
            sentence_token_index = 0
            for token_index, ((start, end), token_id) in enumerate(zip(offsets, input_ids)):
                if end <= start:
                    continue
                overlaps = start < char_end and end > char_start
                contained = start >= char_start and end <= char_end
                if overlaps and not contained:
                    raise ValueError(
                        "ambiguous token boundary for {} in {}".format(sentence_id, context_id)
                    )
                if not contained:
                    continue
                observation_id = "{}__target_{:02d}__token_{:04d}".format(
                    context_id, target_index, sentence_token_index
                )
                observations.append(
                    {
                        "observation_index": observation_index,
                        "observation_id": observation_id,
                        "context_id": context_id,
                        "sentence_id": sentence_id,
                        "sentence_sha256": hashlib.sha256(sentence.encode("utf-8")).hexdigest(),
                        "target_index": target_index,
                        "token_index": token_index,
                        "sentence_token_index": sentence_token_index,
                        "token_id": int(token_id),
                        "token_text": tokenizer.convert_ids_to_tokens(int(token_id)),
                        "char_start": int(start),
                        "char_end": int(end),
                    }
                )
                admissible_positions.append(token_index)
                context_positions.append(token_index)
                sentence_token_index += 1
                observation_index += 1
            if not admissible_positions:
                raise ValueError("no admissible token for {}".format(sentence_id))
            resolved_targets.append(
                {
                    "sentence_id": sentence_id,
                    "sentence_char_start": char_start,
                    "sentence_char_end": char_end,
                    "admissible_token_positions": admissible_positions,
                }
            )
        contexts.append(
            {
                "context_id": context_id,
                "presentation_context": config.context_id,
                "rendered_text": rendered,
                "input_ids": input_ids,
                "targets": resolved_targets,
                "admissible_token_positions": sorted(set(context_positions)),
            }
        )
    if observed_sentence_ids != set(corpus_by_id):
        missing = sorted(set(corpus_by_id).difference(observed_sentence_ids))
        raise ValueError("presentation plan does not cover all 100 sentences: {}".format(missing))
    return contexts, observations


def build_direction_rows(
    observations: Sequence[Dict[str, Any]], config: Experiment0Config
) -> List[Dict[str, Any]]:
    """Build every concept, fixed-random and renewed-noise direction record.

    Noise is indexed by target observation and repetition. It is therefore a
    renewed draw for each planned token, unlike a second fixed random bank.
    Seeds are sufficient to regenerate exact vectors with the recorded PyTorch
    version and Gaussian CPU generator.
    """
    rows = []
    for layer in config.layers:
        hidden_state_index = layer + config.hidden_state_offset
        for concept in config.concepts:
            source = config.vector_dir / "{}_{}_{}.pt".format(
                concept.name, hidden_state_index, config.vector_type
            )
            if not source.exists():
                raise FileNotFoundError("missing concept vector: {}".format(source))
            payload = torch.load(source, map_location="cpu", weights_only=False)
            provenance_complete = all(
                payload.get(field) is not None
                for field in (
                    "model_name",
                    "model_revision",
                    "concept_name",
                    "dataset",
                    "layer",
                    "vec_type",
                )
            )
            if config.protocol_status == "frozen" and not provenance_complete:
                raise ValueError(
                    "frozen calibration requires complete concept provenance: {}".format(source)
                )
            rows.append(
                {
                    "direction_id": "concept__block_{:02d}__{}".format(layer, concept.name),
                    "direction_family": "concept",
                    "decoder_block_index": layer,
                    "hidden_state_index": hidden_state_index,
                    "concept": concept.name,
                    "concept_dataset": concept.dataset,
                    "concept_split": concept.split,
                    "concept_vector_type": config.vector_type,
                    "source_path": str(source.relative_to(REPO_ROOT)),
                    "source_sha256": file_sha256(source),
                    "source_model_name": payload.get("model_name"),
                    "source_model_revision": payload.get("model_revision"),
                    "source_dataset": payload.get("dataset"),
                    "source_provenance_complete": provenance_complete,
                    "seed": None,
                    "target_observation_id": None,
                    "noise_repetition": None,
                }
            )
        for index in range(config.fixed_random_count_per_layer):
            # Match the persisted random_s{index}_{block}_avg.pt bank exactly.
            # Global uniqueness against renewed noise is checked below before
            # the material plan can be persisted.
            seed = (
                config.fixed_random_base_seed
                + layer * config.fixed_random_layer_stride
                + index
            )
            source = config.vector_dir / "random_s{}_{}_{}.pt".format(
                index, layer, config.vector_type
            )
            if not source.is_file():
                raise FileNotFoundError(
                    "missing persisted fixed-random direction: {}".format(source)
                )
            rows.append(
                {
                    "direction_id": "fixed_random__block_{:02d}__{:04d}".format(layer, index),
                    "direction_family": "fixed_random",
                    "decoder_block_index": layer,
                    "hidden_state_index": None,
                    "concept": None,
                    "concept_dataset": None,
                    "concept_split": None,
                    "concept_vector_type": None,
                    "source_path": str(source.relative_to(REPO_ROOT)),
                    "source_sha256": file_sha256(source),
                    "seed": seed,
                    "target_observation_id": None,
                    "noise_repetition": None,
                }
            )
        for observation in observations:
            for repetition in range(config.noise_repetitions_per_position):
                index = (
                    int(observation["observation_index"])
                    * config.noise_repetitions_per_position
                    + repetition
                )
                noise_count_per_layer = (
                    len(observations) * config.noise_repetitions_per_position
                )
                seed = (
                    config.renewed_noise_base_seed
                    + layer * noise_count_per_layer
                    + index
                )
                rows.append(
                    {
                        "direction_id": "renewed_noise__block_{:02d}__{}__rep_{:02d}".format(
                            layer, observation["observation_id"], repetition
                        ),
                        "direction_family": "renewed_noise",
                        "decoder_block_index": layer,
                        "hidden_state_index": None,
                        "concept": None,
                        "concept_dataset": None,
                        "concept_split": None,
                        "concept_vector_type": None,
                        "source_path": None,
                        "seed": seed,
                        "target_observation_id": observation["observation_id"],
                        "noise_repetition": repetition,
                    }
                )
    stochastic_rows = [
        row for row in rows if row["direction_family"] != "concept"
    ]
    stochastic_seeds = [int(row["seed"]) for row in stochastic_rows]
    if len(stochastic_seeds) != len(set(stochastic_seeds)):
        # A shared seed would make two nominally independent control
        # directions exactly identical after deterministic regeneration.
        raise ValueError(
            "fixed-random and renewed-noise seeds must be globally unique"
        )
    return rows


def unit(vector: torch.Tensor) -> Tuple[torch.Tensor, float]:
    vector = vector.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    norm = float(torch.linalg.vector_norm(vector).item())
    if not norm > 0.0 or not bool(torch.isfinite(vector).all()):
        raise ValueError("direction must be finite and non-zero")
    return vector / norm, norm


def materialize_direction(
    row: Dict[str, Any], config: Experiment0Config, hidden_size: int
) -> Tuple[torch.Tensor, float]:
    """Load or deterministically regenerate one direction from its record."""
    if row["direction_family"] == "concept":
        payload = torch.load(REPO_ROOT / row["source_path"], map_location="cpu", weights_only=False)
        vector = payload["vector"]
        normalized, original_norm = unit(vector)
    elif row["direction_family"] == "fixed_random":
        payload = torch.load(
            REPO_ROOT / row["source_path"], map_location="cpu", weights_only=False
        )
        if int(payload.get("draw_seed", -1)) != int(row["seed"]):
            raise ValueError(
                "persisted random direction has an unexpected draw_seed: {}".format(
                    row["direction_id"]
                )
            )
        stored_vector = payload["vector"].detach().to(
            device="cpu", dtype=torch.float32
        ).reshape(-1)
        # Validate finiteness/non-zero norm without normalizing the persisted
        # vector a second time, which could alter its last floating-point bits.
        unit(stored_vector)
        generator = torch.Generator(device="cpu").manual_seed(int(row["seed"]))
        regenerated, original_norm = unit(
            torch.randn(hidden_size, generator=generator)
        )
        if not torch.equal(stored_vector, regenerated):
            raise ValueError(
                "persisted random direction does not match its seed: {}".format(
                    row["direction_id"]
                )
            )
        normalized = stored_vector
    else:
        generator = torch.Generator(device="cpu").manual_seed(int(row["seed"]))
        vector = torch.randn(hidden_size, generator=generator)
        normalized, original_norm = unit(vector)
    if normalized.numel() != hidden_size:
        raise ValueError("direction {} has the wrong hidden size".format(row["direction_id"]))
    return normalized, original_norm
