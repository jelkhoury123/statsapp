"""Standardisation helpers for the SLR Standardisation lesson."""

from __future__ import annotations

import numpy as np
import pytest

from src.fitting_line import (
    STD_PREDICTOR_KEYS,
    _students_10k_frame,
    load_fitting_dataset,
    ols_fit,
    pearson_r,
    r_squared,
    r_standardisation_commands,
    sample_sd,
    standardisation_student_frame,
    standardised_slope,
    subsample_observations,
    total_sum_of_squares,
    zscore,
)


def test_standardised_slope_matches_zscore_regression_and_pearson_r():
    rng = np.random.default_rng(21)
    x = rng.normal(10.0, 2.5, size=80)
    y = 3.0 + 1.7 * x + rng.normal(0.0, 1.2, size=80)
    ols = ols_fit(x, y)
    sx, sy = sample_sd(x), sample_sd(y)
    beta_star = standardised_slope(float(ols.beta), sx, sy)
    ols_z = ols_fit(zscore(x), zscore(y))
    r = pearson_r(x, y)
    assert float(ols_z.alpha) == pytest.approx(0.0, abs=1e-10)
    assert float(ols_z.beta) == pytest.approx(beta_star, rel=1e-10, abs=1e-10)
    assert float(ols_z.beta) == pytest.approx(r, rel=1e-10, abs=1e-10)


def test_zscore_has_mean_zero_and_unit_sd():
    values = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    z = zscore(values)
    assert float(np.mean(z)) == pytest.approx(0.0, abs=1e-12)
    assert sample_sd(z) == pytest.approx(1.0, abs=1e-12)


def test_r_standardisation_commands_are_concise_for_one_predictor():
    code = r_standardisation_commands(predictor="hours_studied", response="exam_score")
    assert "Method 1" in code
    assert "Method 2" in code
    assert "hours_studied" in code
    assert "exam_score" in code
    assert "scale(" in code
    assert "attendance" not in code
    assert "sleep_hours" not in code


def test_students_pool_has_all_standardisation_predictors():
    _students_10k_frame.cache_clear()
    frame = _students_10k_frame()
    for key in STD_PREDICTOR_KEYS:
        assert key in frame.columns
    r_h = abs(pearson_r(frame["hours_studied"], frame["exam_score"]))
    r_a = abs(pearson_r(frame["attendance"], frame["exam_score"]))
    r_s = abs(pearson_r(frame["sleep_hours"], frame["exam_score"]))
    assert r_a < r_h
    assert r_s < r_h


def test_standardisation_frame_same_sample_and_r2_equals_beta_star_sq():
    pool = load_fitting_dataset("students")
    view = subsample_observations(pool, 90, np.random.default_rng(3))
    sample = standardisation_student_frame(view)
    assert sample is not None
    assert len(sample) == 90
    assert sample["hours_studied"].to_numpy() == pytest.approx(view.x)
    assert sample["exam_score"].to_numpy() == pytest.approx(view.y)

    scores = sample["exam_score"].to_numpy()
    sst = total_sum_of_squares(scores)
    for key in ("hours_studied", "sleep_hours"):
        x = sample[key].to_numpy()
        ols = ols_fit(x, scores)
        beta_star = float(ols_fit(zscore(x), zscore(scores)).beta)
        r2 = r_squared(float(ols.sse), float(sst))
        assert r2 == pytest.approx(beta_star**2, rel=1e-10, abs=1e-10)


def test_hours_vs_sleep_often_shows_raw_vs_standardised_reversal():
    """Study hours typically win on β*; sleep can win on raw β̂ (different SD)."""
    pool = load_fitting_dataset("students")
    view = subsample_observations(pool, 200, np.random.default_rng(11))
    sample = standardisation_student_frame(view)
    assert sample is not None
    scores = sample["exam_score"].to_numpy()
    hours = sample["hours_studied"].to_numpy()
    sleep = sample["sleep_hours"].to_numpy()
    ols_h = ols_fit(hours, scores)
    ols_s = ols_fit(sleep, scores)
    star_h = abs(float(ols_fit(zscore(hours), zscore(scores)).beta))
    star_s = abs(float(ols_fit(zscore(sleep), zscore(scores)).beta))
    assert star_h > star_s
