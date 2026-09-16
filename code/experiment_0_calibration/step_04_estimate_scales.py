"""Experiment 0 step 4: point estimates of each natural directional scale."""

from typing import Dict

import numpy as np


MAD_GAUSSIAN_CONSISTENCY = 1.4826


def corrected_mad(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """MAD on the SD scale under a Gaussian reference distribution."""
    center = np.median(values, axis=axis, keepdims=True)
    return MAD_GAUSSIAN_CONSISTENCY * np.median(
        np.abs(values - center), axis=axis
    )


def estimate_point_scales(projections: np.ndarray) -> Dict[str, np.ndarray]:
    """Calculate the protocol's mean, SD, median, MAD and quantiles."""
    values = projections.astype(np.float64, copy=False)
    quantiles = np.quantile(values, [0.01, 0.05, 0.95, 0.99], axis=0)
    return {
        "mean": values.mean(axis=0),
        "sd": values.std(axis=0, ddof=1),
        "median": np.median(values, axis=0),
        "mad_corrected": corrected_mad(values, axis=0),
        "p01": quantiles[0],
        "p05": quantiles[1],
        "p95": quantiles[2],
        "p99": quantiles[3],
    }

