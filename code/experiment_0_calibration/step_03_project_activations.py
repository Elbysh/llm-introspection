"""Experiment 0 step 3: project natural activations onto planned directions."""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .prepare_material import materialize_direction
from .protocol_config import Experiment0Config


def project_activations(
    activations: torch.Tensor,
    direction_rows: List[Dict[str, Any]],
    config: Experiment0Config,
) -> Tuple[np.ndarray, List[float]]:
    """Return `<h, v>` with shape [admissible positions, directions]."""
    hidden_size = int(activations.shape[1])
    projection_chunks = []
    original_norms = []
    for start in range(0, len(direction_rows), config.direction_chunk_size):
        rows = direction_rows[start : start + config.direction_chunk_size]
        materialized = [
            materialize_direction(row, config, hidden_size) for row in rows
        ]
        direction_matrix = torch.stack([item[0] for item in materialized])
        projection_chunks.append(
            (activations @ direction_matrix.T).numpy().astype(np.float32, copy=False)
        )
        original_norms.extend(item[1] for item in materialized)
    return np.concatenate(projection_chunks, axis=1), original_norms

