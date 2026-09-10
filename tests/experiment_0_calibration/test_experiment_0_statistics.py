import numpy as np

from experiment_0_calibration.statistics import (
    build_bootstrap_plan,
    corrected_mad,
    summarize_projection_matrix,
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
    plan = build_bootstrap_plan(sentence_ids, resamples_sd=50, resamples_mad=20, seed=7)
    result = summarize_projection_matrix(projections, plan, ci_level=0.95)

    assert [rows.tolist() for rows in plan.row_indices] == [[0, 1], [2, 3]]
    assert result["sd"].shape == (2,)
    assert np.all(result["sd_ci_low"] <= result["sd"])
    assert np.all(result["sd_ci_high"] >= result["sd"])
