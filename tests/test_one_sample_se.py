"""One sample's residuals estimate the standard errors. They do not use the reference."""

from __future__ import annotations

import numpy as np
import pytest

from src.charts import comparison_histogram, distribution_scatter
from src.regression import fit_ols
from src.sampling import draw_sample
from src.sampling_distributions import (
    OneSampleState,
    histogram_density,
    new_experiment,
    normal_density,
    one_sample_for_size,
    one_sample_standard_errors,
    residual_segment_indices,
)
from src.population import load_population


@pytest.fixture(scope="module")
def population():
    return load_population()


def test_residual_sd_uses_n_minus_two_and_matches_the_ols_formulas():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 1.2, 2.4, 3.1, 4.2])
    fit = fit_ols(x, y)
    result = one_sample_standard_errors(x, y)
    residual_sd = np.sqrt(fit.sse / (len(x) - 2))
    x_mean = float(x.mean())
    sxx = float(np.sum((x - x_mean) ** 2))
    assert result.n == 5
    assert result.residual_sd == pytest.approx(residual_sd)
    assert result.sxx == pytest.approx(sxx)
    assert result.se_beta == pytest.approx(residual_sd / np.sqrt(sxx))
    assert result.se_alpha == pytest.approx(residual_sd * np.sqrt(1 / len(x) + x_mean**2 / sxx))
    assert result.alpha == pytest.approx(fit.alpha)
    assert result.beta == pytest.approx(fit.beta)


def test_standard_errors_are_undefined_without_residual_degrees_of_freedom():
    short = one_sample_standard_errors(np.array([0.0, 1.0]), np.array([1.0, 3.0]))
    assert short.se_alpha is None
    assert short.se_beta is None
    assert short.residual_sd is None

    flat = one_sample_standard_errors(np.array([2.0, 2.0, 2.0, 2.0]), np.array([1.0, 2.0, 3.0, 4.0]))
    assert flat.se_beta is None
    assert flat.residual_sd is None


def test_another_sample_leaves_the_reference_distribution_untouched(population):
    experiment = new_experiment(30, population.hours_studied, population.exam_score, population.alpha, population.beta)
    held = OneSampleState(
        sample_size=30,
        x=population.hours_studied[:30].copy(),
        y=population.exam_score[:30].copy(),
    )
    before_alpha = experiment.alphas.copy()
    before_edges = experiment.alpha_edges.copy()
    fresh = draw_sample(population, 30, np.random.default_rng(7))
    held = OneSampleState(30, fresh.hours_studied, fresh.exam_score, show_residuals=True, show_band=False)
    error = one_sample_standard_errors(held.x, held.y)
    assert error.se_alpha is not None
    assert error.n == 30
    assert np.array_equal(experiment.alphas, before_alpha)
    assert np.array_equal(experiment.alpha_edges, before_edges)
    assert experiment.n_repetitions == 0


def test_one_sample_is_kept_until_the_sample_size_changes(population):
    experiment = new_experiment(40, population.hours_studied, population.exam_score, population.alpha, population.beta)
    experiment = type(experiment)(
        sample_size=40,
        alphas=experiment.alphas,
        betas=experiment.betas,
        sample_x=population.hours_studied[:40],
        sample_y=population.exam_score[:40],
        show_normals=False,
        alpha_edges=experiment.alpha_edges,
        beta_edges=experiment.beta_edges,
    )
    seeded = one_sample_for_size(None, 40, experiment)
    assert np.array_equal(seeded.x, experiment.sample_x)
    kept = one_sample_for_size(seeded, 40, experiment)
    assert kept.x is seeded.x
    waiting = OneSampleState(40, None, None, show_residuals=True)
    filled = one_sample_for_size(waiting, 40, experiment)
    assert np.array_equal(filled.x, experiment.sample_x)
    assert filled.show_residuals is True
    cleared = one_sample_for_size(seeded, 80, experiment)
    assert cleared.x is None
    assert cleared.sample_size == 80


def test_density_histogram_and_normal_curve_share_the_probability_scale():
    edges = np.linspace(0.0, 4.0, 21)
    values = np.array([1.0, 1.4, 1.8, 2.2, 2.6, 3.0])
    density = histogram_density(values, edges)
    width = float(edges[1] - edges[0])
    assert density.sum() * width == pytest.approx(1.0)
    spread = 0.4
    curve = normal_density(np.array([2.0]), 2.0, spread)
    assert curve[0] == pytest.approx(1.0 / (spread * np.sqrt(2.0 * np.pi)))

    figure = comparison_histogram(
        values=values,
        edges=edges,
        population=2.1,
        title="Sampling distribution of β̂",
        x_title="Estimated slope β̂",
        decimals=3,
        reference_mean=2.0,
        reference_sd=spread,
        reference_normal_x=np.array([2.0]),
        reference_normal_y=curve,
        estimate=2.2,
        standard_error=0.35,
        sample_normal_x=np.array([2.2]),
        sample_normal_y=normal_density(np.array([2.2]), 2.2, 0.35),
    )
    assert figure.layout.yaxis.title.text == "Density"
    assert figure.data[0].y.sum() * width == pytest.approx(1.0)
    names = [trace.name for trace in figure.data]
    assert names == ["Samples", "Normal approximation", "Estimated SE"]
    dashed = [shape for shape in figure.layout.shapes if shape.type == "line" and shape.line.dash == "dash"]
    colours = {shape.line.color for shape in dashed}
    assert "#6B7280" in colours
    assert "#1D4ED8" in colours
    orange = [shape for shape in figure.layout.shapes if shape.type == "line" and shape.line.color == "#EA580C"]
    assert len(orange) == 1


def test_residual_segments_are_a_subset_and_the_band_is_optional():
    x = np.linspace(0.0, 10.0, 200)
    assert residual_segment_indices(x).size == 80
    assert residual_segment_indices(x[:10]).size == 10
    figure = distribution_scatter(
        cloud_image="data:image/png;base64,aaaa",
        x_range=(0.0, 10.0),
        y_range=(0.0, 100.0),
        population_alpha=20.0,
        population_beta=5.0,
        sample_x=x,
        sample_y=20 + 5 * x,
        sample_alpha=20.0,
        sample_beta=5.0,
        show_residuals=True,
        residual_sd=4.0,
        show_residual_band=True,
        residual_indices=residual_segment_indices(x),
    )
    names = [trace.name for trace in figure.data]
    assert "Residuals" in names
    assert "±1 residual SD" in names
    assert any(annotation.text == "±1 residual SD" for annotation in figure.layout.annotations)
