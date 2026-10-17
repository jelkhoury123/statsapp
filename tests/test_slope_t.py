"""Classical t-test for a slope when the population slope is zero."""

import numpy as np
from scipy.stats import t as student_t

from src.null_slope import (
    NULL_ALPHA,
    NULL_SEED,
    NULL_SIGMA,
    NULL_SIZE,
    SIGNIFICANCE_LEVEL,
    draw_null_sample,
    null_population,
    slope_t_test,
)
from src.population import load_population
from src.sampling_distributions import one_sample_standard_errors


def test_null_population_has_slope_zero_and_its_own_seed():
    population = null_population()
    generator = np.random.default_rng(NULL_SEED)
    hours = generator.uniform(0.0, 10.0, NULL_SIZE)
    scores = NULL_ALPHA + generator.normal(0.0, NULL_SIGMA, NULL_SIZE)
    assert population.hours.shape == (NULL_SIZE,)
    assert np.array_equal(population.hours, hours)
    assert np.array_equal(population.scores, scores)
    assert population.beta == 0.0
    assert population.hours.min() >= 0.0
    assert population.hours.max() <= 10.0
    original = load_population()
    assert not np.array_equal(population.hours, original.hours_studied)
    assert abs(original.beta - 5) < 0.1


def test_slope_t_uses_residual_standard_error_and_student_t():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    y = np.array([44, 46, 43, 47, 45, 48, 42, 46], dtype=float)
    error = one_sample_standard_errors(x, y)
    result = slope_t_test(x, y)
    assert result.degrees_of_freedom == len(x) - 2
    assert result.beta == error.beta
    assert result.standard_error == error.se_beta
    assert result.t_statistic == error.beta / error.se_beta
    assert result.p_value == min(1.0, 2.0 * student_t.sf(abs(result.t_statistic), result.degrees_of_freedom))
    assert result.critical_value == student_t.ppf(0.975, result.degrees_of_freedom)
    assert result.reject is (result.p_value < SIGNIFICANCE_LEVEL)


def test_exactly_zero_slope_fails_to_reject():
    x = np.array([-2, -1, 0, 1, 2], dtype=float)
    y = np.array([1, -1, 0, -1, 1], dtype=float)
    result = slope_t_test(x, y)
    assert result.beta == 0.0
    assert result.t_statistic == 0.0
    assert result.p_value == 1.0
    assert result.reject is False


def test_large_slope_rejects():
    x = np.linspace(0.0, 10.0, 40)
    y = 45.0 + 2.0 * x
    result = slope_t_test(x, y + np.array([0.2, -0.2] * 20))
    assert result.reject is True
    assert result.p_value < SIGNIFICANCE_LEVEL


def test_undefined_when_x_has_no_spread():
    result = slope_t_test(np.ones(6), np.arange(6, dtype=float))
    assert result.t_statistic is None
    assert result.p_value is None
    assert result.reject is None


def test_true_slope_changes_scores_only():
    baseline = null_population(0.0)
    shifted = null_population(1.5)
    assert np.array_equal(baseline.hours, shifted.hours)
    assert np.allclose(shifted.scores, baseline.scores + 1.5 * baseline.hours)
    assert shifted.alpha == NULL_ALPHA
    assert shifted.beta == 1.5
    assert baseline.beta == 0.0


def test_positive_population_slope_can_produce_a_negative_estimate():
    population = null_population(0.1)
    generator = np.random.default_rng(0)
    estimates = []
    for _ in range(40):
        hours, scores = draw_null_sample(population, 12, generator)
        result = slope_t_test(hours, scores)
        estimates.append(result.beta)
        assert result.t_statistic == result.beta / result.standard_error
    assert any(estimate < 0 for estimate in estimates)


def test_reference_charts_stay_centred_at_zero():
    from src.charts import null_slope_scatter, slope_sampling_figure, t_test_figure
    from src.null_slope import t_density_curve

    sampling = slope_sampling_figure(beta=-0.4, standard_error=0.2)
    assert sampling.layout.title.text == "Distribution of \u03b2\u0302 if \u03b2 = 0"
    assert sampling.layout.title.subtitle.text == "Sampling distribution of the estimated slope."
    centres = [round(float(shape.x0), 6) for shape in sampling.layout.shapes]
    assert 0.0 in centres
    assert -0.4 in centres
    curve = sampling.data[0]
    peak = float(curve.x[int(np.argmax(curve.y))])
    assert abs(peak) < 0.05
    upper_labels = [annotation.text for annotation in sampling.layout.annotations]
    assert "Observed \u03b2\u0302 = -0.400" in upper_labels
    assert "5% rejection thresholds" in upper_labels
    assert sampling.layout.height == 236

    critical = 1.984
    grid, density = t_density_curve(98, -1.2, critical)
    tested = t_test_figure(grid=grid, density=density, statistic=-1.2, critical=critical, degrees=98)
    assert tested.layout.title.text.startswith("Distribution of t if \u03b2 = 0")
    assert "df = n − 2 = 98" in tested.layout.title.text
    assert "Significance level: 5%" in tested.layout.title.subtitle.text
    labels = [annotation.text for annotation in tested.layout.annotations]
    assert labels.count("2.5%") == 2
    assert "95%" in labels
    assert "Critical t values" in labels
    assert "Observed t = -1.200" in labels
    assert "Significance level: 5%" not in labels
    edges = [round(float(shape.x0), 3) for shape in tested.layout.shapes]
    assert -1.984 in edges and 1.984 in edges and -1.2 in edges
    assert tested.layout.height == sampling.layout.height
    assert tested.layout.margin.to_plotly_json() == sampling.layout.margin.to_plotly_json()

    scatter = null_slope_scatter(
        sample_x=np.array([0.0, 10.0]),
        sample_y=np.array([45.0, 60.0]),
        sample_alpha=45.0,
        sample_beta=1.5,
        population_alpha=45.0,
        population_beta=1.5,
    )
    orange = scatter.data[-1]
    assert np.allclose(orange.y, [45.0, 60.0])


def test_draw_is_a_sample_from_the_null_population():
    population = null_population()
    hours, scores = draw_null_sample(population, 25, np.random.default_rng(7))
    assert hours.shape == (25,)
    assert len(np.unique(hours)) == 25
    assert set(np.round(hours, 10)).issubset(set(np.round(population.hours, 10)))
    assert set(np.round(scores, 10)).issubset(set(np.round(population.scores, 10)))
