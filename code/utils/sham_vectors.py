"""
Control vectors for C1.4 (sham injection) / C2.3 (sham identification): a random
unit direction with no semantic content, at the same alpha as a real injection,
to separate generic-anomaly detection from concept-specific detection.
"""

from pathlib import Path

import torch


def infer_hidden_dim(saved_vectors_dir: str = "saved_vectors/llama") -> int:
    """Infer hidden_dim from an existing saved concept vector's shape, so it never
    needs hardcoding to 4096."""
    directory = Path(saved_vectors_dir)
    sample_path = next(directory.glob("*.pt"), None)
    if sample_path is None:
        raise FileNotFoundError(f"No saved vectors found in {directory} to infer hidden_dim from")
    data = torch.load(sample_path, weights_only=False)
    vector = data["vector"] if isinstance(data, dict) else data
    return int(vector.reshape(-1).shape[0])


def make_random_direction(hidden_dim: int, seed: int) -> torch.Tensor:
    """Deterministic, unit-norm, isotropic random direction. Generated on the fly
    per trial (no disk precompute needed, unlike real concept vectors)."""
    generator = torch.Generator().manual_seed(seed)
    vector = torch.randn(hidden_dim, generator=generator)
    return vector / torch.norm(vector, p=2)
