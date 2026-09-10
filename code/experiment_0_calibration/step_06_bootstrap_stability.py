"""Experiment 0 step 6: phrase-cluster bootstrap of SD and corrected MAD."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .step_04_estimate_scales import corrected_mad


@dataclass(frozen=True)
class BootstrapPlan:
    """Shared sentence resamples, reused for all layers and directions."""

    sentence_ids: List[str]
    row_indices: List[np.ndarray]
    sentence_draws_sd: np.ndarray
    row_draws_mad: List[np.ndarray]


def build_phrase_bootstrap_plan(
    observation_sentence_ids: List[str],
    resamples_sd: int,
    resamples_mad: int,
    seed: int,
) -> BootstrapPlan:
    sentence_ids = list(dict.fromkeys(observation_sentence_ids))
    sentence_id_array = np.asarray(observation_sentence_ids)
    rows_by_sentence = [
        np.flatnonzero(sentence_id_array == sentence_id)
        for sentence_id in sentence_ids
    ]
    rng = np.random.RandomState(seed)
    draws_sd = rng.randint(
        0, len(sentence_ids), size=(resamples_sd, len(sentence_ids))
    )
    draws_mad = rng.randint(
        0, len(sentence_ids), size=(resamples_mad, len(sentence_ids))
    )
    row_draws_mad = [
        np.concatenate([rows_by_sentence[index] for index in draw])
        for draw in draws_mad
    ]
    return BootstrapPlan(sentence_ids, rows_by_sentence, draws_sd, row_draws_mad)


def estimate_bootstrap_stability(
    projections: np.ndarray,
    plan: BootstrapPlan,
    ci_level: float,
) -> Dict[str, np.ndarray]:
    """Return phrase-bootstrap intervals and coefficients of variation."""
    if projections.ndim != 2 or projections.shape[0] != sum(
        len(rows) for rows in plan.row_indices
    ):
        raise ValueError("projection rows do not match bootstrap observations")
    values = projections.astype(np.float64, copy=False)

    # Sentence-level sufficient statistics make the 1,000 SD resamples fast.
    counts = np.asarray([len(rows) for rows in plan.row_indices], dtype=np.float64)
    sums = np.stack([values[rows].sum(axis=0) for rows in plan.row_indices])
    sum_squares = np.stack(
        [(values[rows] ** 2).sum(axis=0) for rows in plan.row_indices]
    )
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

    # MAD has no equivalent sufficient-statistic shortcut because it depends on
    # order statistics. The configuration therefore records its own resample count.
    bootstrap_mad = np.stack(
        [corrected_mad(values[rows], axis=0) for rows in plan.row_draws_mad]
    )
    tail = (1.0 - ci_level) / 2.0
    sd_ci = np.quantile(bootstrap_sd, [tail, 1.0 - tail], axis=0)
    mad_ci = np.quantile(bootstrap_mad, [tail, 1.0 - tail], axis=0)
    return {
        "sd_ci_low": sd_ci[0],
        "sd_ci_high": sd_ci[1],
        "mad_ci_low": mad_ci[0],
        "mad_ci_high": mad_ci[1],
        "sd_bootstrap_cv": bootstrap_sd.std(axis=0, ddof=1)
        / np.maximum(bootstrap_sd.mean(axis=0), 1e-12),
        "mad_bootstrap_cv": bootstrap_mad.std(axis=0, ddof=1)
        / np.maximum(bootstrap_mad.mean(axis=0), 1e-12),
    }

