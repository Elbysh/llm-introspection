"""
Concept registry: canonical one-line descriptions for the 10 concepts with saved
steering vectors under data/saved_vectors/llama (5 from complex_data.json:
fibonacci_numbers, recursion, betrayal, appreciation, shutdown; 5 from
simple_data.json: Dust, Satellites, Trumpets, Origami, Illusions).

Used by embedding_judge.py for free-response grading. Kept small and separate
from prompt templates so Block 2's candidate-list / cosine-similarity-bucket
helpers (Block 4/E6) can extend it later without touching calibration.
"""

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
