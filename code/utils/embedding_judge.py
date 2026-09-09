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
    return SentenceTransformer(model_name)


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
