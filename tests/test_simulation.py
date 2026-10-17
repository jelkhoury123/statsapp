"""Repeated sampling: bias, precision, noise, and an unchanged data file."""

from __future__ import annotations

import numpy as np
import pytest

from src.charts import coefficient_histogram, overlay_density_figure, sample_scatter_figure
from src.population import (
    DATA_DIR,
    DEFAULT_CSV_PATH,
    file_sha256,
    load_population,
)
from src.simulation import (
    compare_noise_levels,
    compare_sample_sizes,
    generate_synthetic_population,
    simulate_coefficients,
)


@pytest.fixture(scope="module")
def population():
    return load_population()


def test_repeated_samples_are_approximately_unbiased(population):
    result = simulate_coefficients(
        population.hours_studied,
        population.exam_score,
        sample_size=400,
        repetitions=500,
        rng=np.random.default_rng(7),
        alpha_population=population.alpha,
        beta_population=population.beta,
    )
    assert result.beta.population == pytest.approx(population.beta)
    assert result.alpha.population == pytest.approx(population.alpha)
    assert result.n_degenerate == 0

    beta_se = result.beta.std / np.sqrt(result.repetitions)
    alpha_se = result.alpha.std / np.sqrt(result.repetitions)
    assert abs(result.beta.bias) < 4 * beta_se
    assert abs(result.alpha.bias) < 4 * alpha_se


def test_slope_spread_falls_as_sample_size_grows(population):
    comparison = compare_sample_sizes(
        population.hours_studied,
        population.exam_score,
        [25, 100, 400],
        repetitions=200,
        rng=np.random.default_rng(8),
        alpha_population=population.alpha,
        beta_population=population.beta,
    )
    spreads = [row["SD slope"] for row in comparison.rows()]
    assert spreads[0] > spreads[1] > spreads[2]


def test_intercept_spread_falls_as_sample_size_grows(population):
    comparison = compare_sample_sizes(
        population.hours_studied,
        population.exam_score,
        [25, 100, 400],
        repetitions=200,
        rng=np.random.default_rng(9),
        alpha_population=population.alpha,
        beta_population=population.beta,
    )
    spreads = [row["SD intercept"] for row in comparison.rows()]
    assert spreads[0] > spreads[1] > spreads[2]


def test_same_seed_reproduces_a_simulation(population):
    kwargs = dict(
        sample_size=40,
        repetitions=60,
        alpha_population=population.alpha,
        beta_population=population.beta,
    )
    first = simulate_coefficients(
        population.hours_studied,
        population.exam_score,
        rng=np.random.default_rng(21),
        **kwargs,
    )
    second = simulate_coefficients(
        population.hours_studied,
        population.exam_score,
        rng=np.random.default_rng(21),
        **kwargs,
    )
    assert np.array_equal(first.beta.estimates, second.beta.estimates)
    assert np.array_equal(first.alpha.estimates, second.alpha.estimates)


def test_simulation_survives_a_degenerate_population():
    x = np.ones(25)
    y = np.linspace(0, 1, 25)
    calls = []
    result = simulate_coefficients(
        x,
        y,
        sample_size=5,
        repetitions=8,
        rng=np.random.default_rng(0),
        alpha_population=np.nan,
        beta_population=np.nan,
        progress=lambda done, total: calls.append((done, total)),
    )
    assert result.n_degenerate == 8
    assert np.isnan(result.beta.estimates).all()
    assert calls[-1] == (8, 8)


def test_sample_size_cannot_exceed_population():
    with pytest.raises(ValueError):
        simulate_coefficients(
            np.arange(10, dtype=float),
            np.arange(10, dtype=float),
            sample_size=11,
            repetitions=2,
            rng=np.random.default_rng(0),
            alpha_population=0.0,
            beta_population=1.0,
        )


def test_synthetic_noise_uses_the_generating_line_and_spreads_with_sigma():
    quiet = np.random.default_rng(4)
    hours, scores = generate_synthetic_population(300, sigma=0.0, rng=quiet)
    # sigma 0: the outcome is exactly 20 + 5x, so the fit recovers the line.
    from src.regression import fit_ols

    exact = fit_ols(hours, scores)
    assert exact.alpha == pytest.approx(20.0, abs=1e-8)
    assert exact.beta == pytest.approx(5.0, abs=1e-8)

    rebuilt = np.random.default_rng(4)
    expected_hours = rebuilt.uniform(0.0, 10.0, size=300)
    expected_error = rebuilt.normal(0.0, 0.0, size=300)
    assert np.allclose(hours, expected_hours)
    assert np.allclose(scores, 20 + 5 * expected_hours + expected_error)

    comparison = compare_noise_levels(
        [2.0, 20.0],
        sample_size=40,
        repetitions=150,
        population_size=4_000,
        seed=15,
    )
    assert comparison.levels[1].beta.std > comparison.levels[0].beta.std * 3
    again = compare_noise_levels(
        [2.0, 20.0],
        sample_size=40,
        repetitions=150,
        population_size=4_000,
        seed=15,
    )
    assert np.array_equal(
        comparison.levels[0].beta_estimates,
        again.levels[0].beta_estimates,
    )


def test_experiments_leave_the_original_population_file_unchanged(population):
    digest = file_sha256(DEFAULT_CSV_PATH)
    csv_names = sorted(path.name for path in DATA_DIR.glob("*.csv"))
    simulate_coefficients(
        population.hours_studied,
        population.exam_score,
        sample_size=30,
        repetitions=20,
        rng=np.random.default_rng(1),
        alpha_population=population.alpha,
        beta_population=population.beta,
    )
    generate_synthetic_population(500, sigma=15, rng=np.random.default_rng(2))
    compare_noise_levels(
        [5.0, 10.0],
        sample_size=20,
        repetitions=10,
        population_size=300,
        seed=3,
    )
    assert file_sha256(DEFAULT_CSV_PATH) == digest
    assert sorted(path.name for path in DATA_DIR.glob("*.csv")) == csv_names
    assert csv_names == ["students_population.csv"]


def test_charts_accept_simulation_output(population):
    figure = sample_scatter_figure(
        x=population.hours_studied[:40],
        y=population.exam_score[:40],
        sample_alpha=21.0,
        sample_beta=4.8,
        population_alpha=population.alpha,
        population_beta=population.beta,
        x_range=population.x_range,
        y_range=population.y_range,
        show_population=True,
        show_sample=True,
        show_residuals=True,
        student_ids=population.student_id[:40],
    )
    assert len(figure.data) >= 3

    estimates = np.array([4.8, 5.0, 5.2, 4.9, 5.1])
    histogram = coefficient_histogram(
        estimates,
        population=population.beta,
        show_population=True,
        title="Estimated slope",
        x_title="Slope",
    )
    assert len(histogram.data) >= 1

    overlay = overlay_density_figure(
        [("n = 10", estimates, "#E69F00"), ("n = 100", estimates + 0.01, "#0072B2")],
        x_title="Estimated slope",
        title="Slope",
        benchmark=population.beta,
        benchmark_name="Population slope",
    )
    assert len(overlay.data) >= 2
