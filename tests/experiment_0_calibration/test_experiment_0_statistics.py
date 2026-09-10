import numpy as np
import pytest

from experiment_0_calibration.step_04_estimate_scales import (
    corrected_mad,
    estimate_point_scales,
)
from experiment_0_calibration.step_06_bootstrap_stability import (
    build_phrase_bootstrap_plan,
    estimate_bootstrap_stability,
)


def test_corrected_mad_is_robust_to_one_extreme_value():
    clean = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    contaminated = np.array([-2.0, -1.0, 0.0, 1.0, 2000.0])

    assert corrected_mad(contaminated) == corrected_mad(clean)


def test_phrase_bootstrap_keeps_token_clusters_and_returns_intervals():
    sentence_ids = ["sentence_a", "sentence_a", "sentence_b", "sentence_b"]
    projections = np.array(
        [[0.0, 10.0], [1.0, 11.0], [3.0, 13.0], [4.0, 14.0]]
    )
    plan = build_phrase_bootstrap_plan(
        sentence_ids, resamples_sd=50, resamples_mad=20, seed=7
    )
    point_estimates = estimate_point_scales(projections)
    bootstrap = estimate_bootstrap_stability(projections, plan, ci_level=0.95)

    assert [rows.tolist() for rows in plan.row_indices] == [[0, 1], [2, 3]]
    assert point_estimates["sd"].shape == (2,)
    assert point_estimates["mean"].tolist() == [2.0, 12.0]
    assert point_estimates["median"].tolist() == [2.0, 12.0]
    assert point_estimates["sd"].tolist() == pytest.approx(
        [np.sqrt(10.0 / 3.0), np.sqrt(10.0 / 3.0)]
    )
    assert np.all(bootstrap["sd_ci_low"] <= point_estimates["sd"])
    assert np.all(bootstrap["sd_ci_high"] >= point_estimates["sd"])
