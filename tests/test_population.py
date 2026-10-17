"""Population file, metadata, and the fixed-population regression."""

from __future__ import annotations

import numpy as np
import pytest

from src.population import (
    DEFAULT_CSV_PATH,
    THEORETICAL_ALPHA,
    THEORETICAL_BETA,
    THEORETICAL_SIGMA,
    coefficients_match_metadata,
    file_sha256,
    load_metadata,
    load_population,
)


@pytest.fixture(scope="module")
def population():
    return load_population()


@pytest.fixture(scope="module")
def metadata():
    return load_metadata()


def test_population_has_expected_shape_and_columns(population):
    assert population.n == 100_000
    assert population.student_id.shape == (100_000,)
    assert population.hours_studied.shape == (100_000,)
    assert population.exam_score.shape == (100_000,)
    assert population.individual_error.shape == (100_000,)
    assert np.isfinite(population.hours_studied).all()
    assert np.isfinite(population.exam_score).all()
    assert len(np.unique(population.student_id)) == population.n


def test_population_ols_matches_metadata(population, metadata):
    assert coefficients_match_metadata(population, metadata, atol=1e-4)
    assert population.alpha == pytest.approx(metadata.alpha, abs=1e-4)
    assert population.beta == pytest.approx(metadata.beta, abs=1e-4)
    assert population.r_squared == pytest.approx(metadata.r_squared, abs=1e-4)
    assert population.n == metadata.population_size
    assert file_sha256(DEFAULT_CSV_PATH) == metadata.sha256
    assert population.sha256 == metadata.sha256


def test_sampling_benchmark_is_not_the_generating_constants(population, metadata):
    """The finite population line is the benchmark, not intercept 20 and slope 5."""
    assert metadata.theoretical_alpha == THEORETICAL_ALPHA
    assert metadata.theoretical_beta == THEORETICAL_BETA
    assert metadata.theoretical_sigma == THEORETICAL_SIGMA
    assert abs(population.alpha - THEORETICAL_ALPHA) > 0.01
    assert abs(population.beta - THEORETICAL_BETA) > 0.005
    assert population.alpha == pytest.approx(20.018, abs=0.01)
    assert population.beta == pytest.approx(4.994, abs=0.01)


def test_exam_scores_are_the_line_plus_noise_and_stay_on_the_exam_scale(population, metadata):
    """Scores are not clipped. The normal error is narrow enough to stay in 0–100."""
    reconstructed = (
        metadata.theoretical_alpha
        + metadata.theoretical_beta * population.hours_studied
        + population.individual_error
    )
    assert np.max(np.abs(population.exam_score - reconstructed)) < 1e-6
    assert float(population.exam_score.min()) > 0
    assert float(population.exam_score.max()) < 100
    outside = (population.exam_score < 0) | (population.exam_score > 100)
    assert float(outside.mean()) == 0.0
    assert population.individual_error.std(ddof=1) == pytest.approx(metadata.theoretical_sigma, abs=0.05)
