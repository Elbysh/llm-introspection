"""Experiment 0, steps 4 and 6: estimators and phrase-cluster bootstrap."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


MAD_GAUSSIAN_CONSISTENCY = 1.4826


def corrected_mad(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Gaussian-consistent median absolute deviation."""
    center = np.median(values, axis=axis, keepdims=True)
    return MAD_GAUSSIAN_CONSISTENCY * np.median(
        np.abs(values - center), axis=axis
    )


@dataclass(frozen=True)
class BootstrapPlan:
    """Shared sentence resamples, reused for every layer and direction."""

    sentence_ids: List[str]
    row_indices: List[np.ndarray]
    sentence_draws_sd: np.ndarray
    row_draws_mad: List[np.ndarray]


def build_bootstrap_plan(
    observation_sentence_ids: List[str],
    resamples_sd: int,
    resamples_mad: int,
    seed: int,
) -> BootstrapPlan:
    sentence_ids = list(dict.fromkeys(observation_sentence_ids))
    rows_by_sentence = [
        np.flatnonzero(np.asarray(observation_sentence_ids) == sentence_id)
        for sentence_id in sentence_ids
    ]
    # RandomState keeps this planning step compatible with the older NumPy
    # available on local development machines as well as the Ruche environment.
    rng = np.random.RandomState(seed)
    draws_sd = rng.randint(0, len(sentence_ids), size=(resamples_sd, len(sentence_ids)))
    draws_mad = rng.randint(0, len(sentence_ids), size=(resamples_mad, len(sentence_ids)))
    row_draws_mad = [
        np.concatenate([rows_by_sentence[index] for index in draw]) for draw in draws_mad
    ]
    return BootstrapPlan(sentence_ids, rows_by_sentence, draws_sd, row_draws_mad)


def summarize_projection_matrix(
    projections: np.ndarray,
    plan: BootstrapPlan,
    ci_level: float,
) -> Dict[str, np.ndarray]:
    """Summarize [observations, directions] projections.

    Point estimates weight every admissible token equally. Bootstrap samples
    whole sentences, preserving all tokens and their dependence within a
    sentence. The same sentence draws are used for every direction.
    """
    if projections.ndim != 2 or projections.shape[0] != sum(
        len(rows) for rows in plan.row_indices
    ):
        raise ValueError("projection rows do not match the bootstrap observations")

    values = projections.astype(np.float64, copy=False)
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1)
    median = np.median(values, axis=0)
    mad = corrected_mad(values, axis=0)
    quantiles = np.quantile(values, [0.01, 0.05, 0.95, 0.99], axis=0)

    # SD bootstrap uses sufficient statistics per sentence. This avoids
    # repeatedly materializing all token rows for thousands of resamples.
    counts = np.asarray([len(rows) for rows in plan.row_indices], dtype=np.float64)
    sums = np.stack([values[rows].sum(axis=0) for rows in plan.row_indices])
    sum_squares = np.stack([(values[rows] ** 2).sum(axis=0) for rows in plan.row_indices])
    multiplicities = np.stack(
        [
            np.bincount(draw, minlength=len(plan.sentence_ids))
            for draw in plan.sentence_draws_sd
        ]
    ).astype(np.float64)
    bootstrap_n = multiplicities @ counts
    bootstrap_sum = multiplicities @ sums
    bootstrap_sum_squares = multiplicities @ sum_squares
    bootstrap_variance = (
        bootstrap_sum_squares - bootstrap_sum ** 2 / bootstrap_n[:, None]
    ) / (bootstrap_n[:, None] - 1.0)
    bootstrap_sd = np.sqrt(np.maximum(bootstrap_variance, 0.0))

    bootstrap_mad = np.stack(
        [corrected_mad(values[rows], axis=0) for rows in plan.row_draws_mad]
    )
    tail = (1.0 - ci_level) / 2.0
    low, high = tail, 1.0 - tail
    sd_ci = np.quantile(bootstrap_sd, [low, high], axis=0)
    mad_ci = np.quantile(bootstrap_mad, [low, high], axis=0)

    return {
        "mean": mean,
        "sd": sd,
        "median": median,
        "mad_corrected": mad,
        "p01": quantiles[0],
        "p05": quantiles[1],
        "p95": quantiles[2],
        "p99": quantiles[3],
        "sd_ci_low": sd_ci[0],
        "sd_ci_high": sd_ci[1],
        "mad_ci_low": mad_ci[0],
        "mad_ci_high": mad_ci[1],
        "sd_bootstrap_cv": bootstrap_sd.std(axis=0, ddof=1)
        / np.maximum(bootstrap_sd.mean(axis=0), 1e-12),
        "mad_bootstrap_cv": bootstrap_mad.std(axis=0, ddof=1)
        / np.maximum(bootstrap_mad.mean(axis=0), 1e-12),
    }
