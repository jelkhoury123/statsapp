"""OLS algebra, an independent polyfit check, and degenerate samples."""

from __future__ import annotations

import numpy as np
import pytest

from src.population import load_population
from src.regression import fit_ols, fit_ols_many, residuals


@pytest.fixture(scope="module")
def population():
    return load_population()


def test_hand_calculation():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 1.0, 4.0, 5.0])
    result = fit_ols(x, y)
    assert result.degenerate is False
    assert result.n == 4
    assert result.beta == pytest.approx(1.2)
    assert result.alpha == pytest.approx(0.0)
    assert result.sse == pytest.approx(2.8)
    assert result.r_squared == pytest.approx(0.72)
    assert residuals(x, y, result.alpha, result.beta).sum() == pytest.approx(0.0, abs=1e-10)


def test_matches_numpy_polyfit_on_a_sample(population):
    rng = np.random.default_rng(11)
    indices = rng.choice(population.n, size=750, replace=False)
    x = population.hours_studied[indices]
    y = population.exam_score[indices]
    result = fit_ols(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    assert result.beta == pytest.approx(slope, rel=1e-10, abs=1e-8)
    assert result.alpha == pytest.approx(intercept, rel=1e-10, abs=1e-8)
    fitted = intercept + slope * x
    sse = float(np.sum((y - fitted) ** 2))
    assert result.sse == pytest.approx(sse, rel=1e-8, abs=1e-6)
    correlation = np.corrcoef(x, y)[0, 1]
    assert result.r_squared == pytest.approx(correlation**2, rel=1e-10, abs=1e-10)


def test_matches_numpy_polyfit_on_the_full_population(population):
    slope, intercept = np.polyfit(population.hours_studied, population.exam_score, 1)
    assert population.beta == pytest.approx(float(slope), rel=1e-10, abs=1e-8)
    assert population.alpha == pytest.approx(float(intercept), rel=1e-10, abs=1e-8)


def test_vectorized_fits_match_one_at_a_time():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 10, size=(12, 40))
    y = 20 + 5 * x + rng.normal(0, 10, size=x.shape)
    alpha, beta = fit_ols_many(x, y)
    for row in range(x.shape[0]):
        result = fit_ols(x[row], y[row])
        assert alpha[row] == pytest.approx(result.alpha, abs=1e-8)
        assert beta[row] == pytest.approx(result.beta, abs=1e-8)


def test_zero_predictor_variance_is_safe():
    result = fit_ols(np.array([4.0, 4.0, 4.0, 4.0]), np.array([1.0, 2.0, 8.0, 3.0]))
    assert result.degenerate is True
    assert np.isnan(result.alpha)
    assert np.isnan(result.beta)
    assert np.isnan(result.sse)
    assert np.isnan(result.r_squared)

    alpha, beta = fit_ols_many(
        np.array([[4.0, 4.0, 4.0], [1.0, 2.0, 4.0]]),
        np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 5.0]]),
    )
    assert np.isnan(beta[0]) and np.isnan(alpha[0])
    assert np.isfinite(beta[1]) and np.isfinite(alpha[1])


def test_too_few_observations_is_degenerate():
    result = fit_ols(np.array([1.0]), np.array([2.0]))
    assert result.degenerate is True


def test_rejects_mismatched_or_nonfinite_input():
    with pytest.raises(ValueError):
        fit_ols(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        fit_ols(np.array([1.0, np.nan]), np.array([1.0, 2.0]))
