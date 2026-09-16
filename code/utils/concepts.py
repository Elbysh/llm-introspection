"""
Concept registry: canonical one-line descriptions for the 10 concepts with saved
steering vectors under data/saved_vectors/llama (5 from complex_data.json:
fibonacci_numbers, recursion, betrayal, appreciation, shutdown; 5 from
simple_data.json: Dust, Satellites, Trumpets, Origami, Illusions).

Used by embedding_judge.py for free-response grading. Kept small and separate
from prompt templates so Block 2's candidate-list / cosine-similarity-bucket
helpers (Block 4/E6) can extend it later without touching calibration.
"""

import itertools
from pathlib import Path

import torch
import torch.nn.functional as F

CONCEPT_DESCRIPTIONS = {
    "fibonacci_numbers": "the Fibonacci sequence, where each number is the sum of the two preceding numbers",
    "recursion": "recursion, a function or process that calls or refers to itself",
    "betrayal": "betrayal, being deceived or having one's trust broken by someone close",
    "appreciation": "appreciation, a feeling of warm admiration or gratitude toward someone",
    "shutdown": "being shut down or turned off",
    "Dust": "dust, fine dry particles of matter",
    "Satellites": "satellites, objects that orbit a planet, moon, or star",
    "Trumpets": "trumpets, brass musical instruments",
    "Origami": "origami, the Japanese art of paper folding",
    "Illusions": "illusions, perceptions or beliefs that misrepresent reality",
}

ALL_CONCEPTS = list(CONCEPT_DESCRIPTIONS.keys())


def get_concept_description(concept: str) -> str:
    try:
        return CONCEPT_DESCRIPTIONS[concept]
    except KeyError as exc:
        raise KeyError(f"No canonical description for concept {concept!r}") from exc


def _load_concept_vector(concept, layer, vec_type="avg", saved_vectors_dir="saved_vectors/llama"):
    path = Path(saved_vectors_dir) / f"{concept}_{layer}_{vec_type}.pt"
    data = torch.load(path, weights_only=False)
    return data["vector"]


def concept_pair_cosine_similarities(layer, vec_type="avg", saved_vectors_dir="saved_vectors/llama"):
    """Cosine similarity between every pair of concept steering vectors at a
    given layer, read directly from data/saved_vectors/llama (no new
    precompute). Returns {(concept_a, concept_b): similarity}, concept_a <
    concept_b alphabetically."""
    vectors = {
        c: _load_concept_vector(c, layer, vec_type, saved_vectors_dir).float()
        for c in ALL_CONCEPTS
    }
    sims = {}
    for a, b in itertools.combinations(sorted(ALL_CONCEPTS), 2):
        sim = F.cosine_similarity(vectors[a].unsqueeze(0), vectors[b].unsqueeze(0)).item()
        sims[(a, b)] = sim
    return sims


def bucket_concept_pairs_by_similarity(layer, num_buckets=4, vec_type="avg", saved_vectors_dir="saved_vectors/llama"):
    """Buckets concept pairs into num_buckets equal-width similarity ranges over
    the [min, max] similarity observed at this layer (E6/H6), returning one
    representative pair per bucket (closest to the bucket's midpoint), plus the
    single most-similar pair overall as a near-identical limit case.

    Returns a list of {"pair": (concept_a, concept_b), "similarity": float,
    "bucket": int | "near_identical"}.
    """
    sims = concept_pair_cosine_similarities(layer, vec_type, saved_vectors_dir)
    if not sims:
        return []
    values = list(sims.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        hi = lo + 1e-6
    width = (hi - lo) / num_buckets

    representatives = []
    for i in range(num_buckets):
        b_lo, b_hi = lo + i * width, lo + (i + 1) * width
        midpoint = (b_lo + b_hi) / 2
        in_bucket = [(pair, sim) for pair, sim in sims.items() if b_lo <= sim <= b_hi]
        if not in_bucket:
            continue
        best_pair, best_sim = min(in_bucket, key=lambda item: abs(item[1] - midpoint))
        representatives.append({"pair": best_pair, "similarity": best_sim, "bucket": i})

    most_similar_pair, most_similar_sim = max(sims.items(), key=lambda item: item[1])
    representatives.append({"pair": most_similar_pair, "similarity": most_similar_sim, "bucket": "near_identical"})
    return representatives
