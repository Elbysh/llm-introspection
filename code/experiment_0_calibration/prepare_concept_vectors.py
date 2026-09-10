"""Ensure concept directions exist at every decoder-block output.

Existing concept files are named by Hugging Face hidden-state index. Experiment
0 measures decoder-block output `l`, which corresponds to hidden state `l + 1`.
The original repository generated indices 0..31; this step adds index 32 for
decoder block 31 without changing the original vector-generation utilities.
"""

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .protocol_config import REPO_ROOT, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Create missing Experiment 0 concept vectors")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    missing_indices = sorted(
        {
            layer + config.hidden_state_offset
            for layer in config.layers
            for concept in config.concepts
            if not (
                config.vector_dir
                / "{}_{}_{}.pt".format(
                    concept.name, layer + config.hidden_state_offset, config.vector_type
                )
            ).exists()
        }
    )
    if not missing_indices:
        print("All Experiment 0 concept vectors already exist")
        return

    utils_dir = REPO_ROOT / "code" / "utils"
    sys.path.insert(0, str(utils_dir))
    from compute_concept_vector_utils import compute_concept_vector

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, revision=config.tokenizer_revision
    )
    datasets = sorted({concept.dataset for concept in config.concepts})
    config.vector_dir.mkdir(parents=True, exist_ok=True)
    for hidden_state_index in missing_indices:
        for dataset in datasets:
            vectors = compute_concept_vector(
                model, tokenizer, dataset, hidden_state_index
            )
            allowed = {
                concept.name for concept in config.concepts if concept.dataset == dataset
            }
            for concept_name, (last_vector, average_vector) in vectors.items():
                if concept_name not in allowed:
                    continue
                for vector_type, vector in (
                    ("last", last_vector),
                    ("avg", average_vector),
                ):
                    path = config.vector_dir / "{}_{}_{}.pt".format(
                        concept_name, hidden_state_index, vector_type
                    )
                    if path.exists():
                        continue
                    torch.save(
                        {
                            "vector": vector,
                            "model_name": config.model_name,
                            "model_revision": config.model_revision,
                            "concept_name": concept_name,
                            "dataset": dataset,
                            "layer": hidden_state_index,
                            "vec_type": vector_type,
                        },
                        path,
                    )
                    print("Saved {}".format(path))


if __name__ == "__main__":
    main()
