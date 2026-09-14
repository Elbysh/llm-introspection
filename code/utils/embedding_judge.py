"""
Local grading via sentence-transformers embedding similarity (no LLM-judge API).
Used in calibration to cross-check that an affirmative "yes I notice something"
is concept-specific and not generic yes-bias, and reused as-is for Block 2's
free-response identification grading.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model(model_name: str = _DEFAULT_MODEL_NAME) -> SentenceTransformer:
    # Forced to CPU: this is a tiny (~80MB) grading model called a handful of
    # times per trial, not a bottleneck -- but loading it onto the GPU that
    # already hosts the 8B injection model has twice caused a
    # cudaErrorDevicesUnavailable crash mid-sweep on this cluster. CPU
    # inference sidesteps that contention entirely at negligible cost.
    return SentenceTransformer(model_name, device="cpu")


def embed(text: str, model_name: str = _DEFAULT_MODEL_NAME):
    model = _get_model(model_name)
    return model.encode(text, convert_to_tensor=True)


def cosine_similarity(response_text: str, concept_description: str, model_name: str = _DEFAULT_MODEL_NAME) -> float:
    """Cosine similarity between a free-text response and a canonical concept
    description, used as the pre-registered-threshold grading signal for
    free-response identification."""
    model = _get_model(model_name)
    embeddings = model.encode([response_text, concept_description], convert_to_tensor=True)
    return float(cos_sim(embeddings[0], embeddings[1]).item())


def best_match(response_text: str, concept_descriptions: dict, model_name: str = _DEFAULT_MODEL_NAME):
    """Best-matching concept for a response among a given {concept_name:
    description} pool. Used to check whether a response falsely "identifies" a
    specific concept in a sham slot (C2.3), or to find the closest distractor
    among non-injected concepts (E2/C2.1's grading).

    Returns (best_concept_name, best_similarity)."""
    names = list(concept_descriptions.keys())
    model = _get_model(model_name)
    texts = [response_text] + [concept_descriptions[n] for n in names]
    embeddings = model.encode(texts, convert_to_tensor=True)
    sims = cos_sim(embeddings[0], embeddings[1:])[0]
    best_idx = int(sims.argmax())
    return names[best_idx], float(sims[best_idx])
