"""Sampling distributions are built from repeated OLS fits, not from a normal draw."""

from __future__ import annotations

import numpy as np
import pytest

from src.charts import sampling_histogram
from src.population import DEFAULT_CSV_PATH, file_sha256, load_population
from src.regression import fit_ols
from src.sampling_distributions import (
    animated_draws,
    batch_sizes,
    draw_regressions,
    estimate_mean,
    estimate_sd,
    expand_edges,
    experiment_for_sample_size,
    histogram_counts,
    BIN_COUNT,
    initial_edges,
    new_experiment,
    normal_frequency,
)


@pytest.fixture(scope="module")
def population():
    return load_population()


def test_each_repetition_is_one_ols_fit_and_the_population_file_stays(population):
    digest = file_sha256(DEFAULT_CSV_PATH)
    before = population.hours_studied.copy()
    alphas, betas, sample_x, sample_y = draw_regressions(
        population.hours_studied,
        population.exam_score,
        sample_size=25,
        repetitions=5,
        rng=np.random.default_rng(4),
    )
    assert alphas.shape == (5,)
    assert betas.shape == (5,)
    fitted = fit_ols(sample_x, sample_y)
    assert fitted.alpha == pytest.approx(alphas[-1])
    assert fitted.beta == pytest.approx(betas[-1])

    sequential_alpha = []
    rng = np.random.default_rng(4)
    for _ in range(5):
        alpha, beta, hours, scores = draw_regressions(
            population.hours_studied,
            population.exam_score,
            sample_size=25,
            repetitions=1,
            rng=rng,
        )
        assert fit_ols(hours, scores).alpha == pytest.approx(alpha[0])
        assert fit_ols(hours, scores).beta == pytest.approx(beta[0])
        sequential_alpha.append(alpha[0])
    assert alphas == pytest.approx(sequential_alpha)
    assert np.array_equal(population.hours_studied, before)
    assert file_sha256(DEFAULT_CSV_PATH) == digest
    again = load_population()
    assert again.sha256 == population.sha256
    assert again.alpha == pytest.approx(population.alpha)


def test_draws_accumulate_until_reset_or_a_new_sample_size(population):
    experiment = new_experiment(
        40,
        population.hours_studied,
        population.exam_score,
        population.alpha,
        population.beta,
    )
    frames = animated_draws(
        experiment,
        population.hours_studied,
        population.exam_score,
        1,
        np.random.default_rng(11),
    )
    assert len(frames) == 1
    assert frames[-1].n_repetitions == 1
    more = animated_draws(
        frames[-1],
        population.hours_studied,
        population.exam_score,
        10,
        np.random.default_rng(12),
    )
    assert [frame.n_repetitions for frame in more] == list(range(2, 12))
    assert more[-1].alphas.size == 11
    assert more[-1].betas.size == 11
    assert np.array_equal(more[-1].alphas[:1], frames[-1].alphas)

    cleared = new_experiment(
        more[-1].sample_size,
        population.hours_studied,
        population.exam_score,
        population.alpha,
        population.beta,
    )
    assert cleared.n_repetitions == 0
    assert cleared.sample_x is None
    changed = experiment_for_sample_size(
        more[-1],
        100,
        population.hours_studied,
        population.exam_score,
        population.alpha,
        population.beta,
    )
    assert changed.n_repetitions == 0
    assert changed.sample_size == 100
    assert experiment_for_sample_size(
        changed,
        100,
        population.hours_studied,
        population.exam_score,
        population.alpha,
        population.beta,
    ) is changed


def test_means_approach_the_population_line_and_spread_shrinks_with_n(population):
    large = animated_draws(
        new_experiment(200, population.hours_studied, population.exam_score, population.alpha, population.beta),
        population.hours_studied,
        population.exam_score,
        400,
        np.random.default_rng(21),
    )[-1]
    alpha_se = estimate_sd(large.alphas) / np.sqrt(large.n_repetitions)
    beta_se = estimate_sd(large.betas) / np.sqrt(large.n_repetitions)
    assert abs(estimate_mean(large.alphas) - population.alpha) < 4 * alpha_se
    assert abs(estimate_mean(large.betas) - population.beta) < 4 * beta_se

    run = animated_draws(
        new_experiment(80, population.hours_studied, population.exam_score, population.alpha, population.beta),
        population.hours_studied,
        population.exam_score,
        600,
        np.random.default_rng(22),
    )[-1]
    early = estimate_sd(run.betas[:40])
    middle = estimate_sd(run.betas[:300])
    full = estimate_sd(run.betas)
    assert abs(middle - full) < abs(early - full)

    small_n = animated_draws(
        new_experiment(30, population.hours_studied, population.exam_score, population.alpha, population.beta),
        population.hours_studied,
        population.exam_score,
        200,
        np.random.default_rng(23),
    )[-1]
    big_n = animated_draws(
        new_experiment(300, population.hours_studied, population.exam_score, population.alpha, population.beta),
        population.hours_studied,
        population.exam_score,
        200,
        np.random.default_rng(24),
    )[-1]
    assert estimate_sd(small_n.betas) > estimate_sd(big_n.betas)
    assert estimate_sd(small_n.alphas) > estimate_sd(big_n.alphas)


def test_histogram_bins_hold_still_and_the_normal_curve_uses_frequency(population):
    assert BIN_COUNT == 50
    edges = initial_edges(population.beta, 0.2)
    assert edges.size == BIN_COUNT + 1
    experiment = new_experiment(
        100,
        population.hours_studied,
        population.exam_score,
        population.alpha,
        population.beta,
    )
    assert experiment.alpha_edges.size == BIN_COUNT + 1
    assert experiment.beta_edges.size == BIN_COUNT + 1
    assert np.array_equal(expand_edges(edges, np.array([population.beta - 0.1, population.beta + 0.1])), edges)
    wider = expand_edges(edges, np.array([population.beta, edges[-1] + 0.5]))
    assert wider[1] - wider[0] == pytest.approx(edges[1] - edges[0])
    assert wider[-1] > edges[-1]
    values = np.array([edges[0] + 0.01, edges[-1] - 0.01, edges[2]])
    counts = histogram_counts(values, edges)
    assert int(counts.sum()) == 3

    assert estimate_mean(np.array([3.0])) == pytest.approx(3.0)
    assert estimate_sd(np.array([3.0])) is None
    assert estimate_mean(np.array([])) is None

    mean = 5.0
    spread = 0.2
    width = float(edges[1] - edges[0])
    height = normal_frequency(np.array([mean]), mean, spread, 800, width)[0]
    density = 1.0 / (spread * np.sqrt(2.0 * np.pi))
    assert height == pytest.approx(800 * width * density)

    figure = sampling_histogram(
        values=np.array([mean, mean + 0.05]),
        edges=edges,
        population=population.beta,
        title="Sampling distribution of β̂",
        x_title="Estimated slope β̂",
        normal_x=np.array([mean]),
        normal_y=np.array([height]),
    )
    assert figure.data[1].name == "Normal approximation"
    assert list(figure.data[1].y) == pytest.approx([height])


def test_histogram_marks_one_standard_error_around_the_empirical_mean():
    edges = np.linspace(0.0, 4.0, 9)
    alone = sampling_histogram(
        values=np.array([1.5]),
        edges=edges,
        population=2.0,
        title="Sampling distribution of α̂",
        x_title="Estimated intercept α̂",
        mean=1.5,
        standard_error=None,
    )
    assert not any(shape.type == "rect" for shape in alone.layout.shapes)
    assert not any(annotation.text == "±1 SE" for annotation in alone.layout.annotations)

    values = np.array([1.0, 2.0, 3.0])
    center = float(values.mean())
    spread = float(np.std(values, ddof=1))
    figure = sampling_histogram(
        values=values,
        edges=edges,
        population=2.2,
        title="Sampling distribution of α̂",
        x_title="Estimated intercept α̂",
        mean=center,
        standard_error=spread,
        decimals=2,
    )
    band = next(shape for shape in figure.layout.shapes if shape.type == "rect")
    assert band.layer == "below"
    assert float(band.x0) == pytest.approx(center - spread)
    assert float(band.x1) == pytest.approx(center + spread)
    dashed = sorted(
        float(shape.x0)
        for shape in figure.layout.shapes
        if shape.type == "line" and shape.line.dash == "dash"
    )
    assert dashed == pytest.approx([center - spread, center + spread])
    orange = [
        shape
        for shape in figure.layout.shapes
        if shape.type == "line" and shape.line.color == "#EA580C"
    ]
    assert len(orange) == 1
    assert float(orange[0].x0) == pytest.approx(2.2)
    assert any(annotation.text == "±1 SE" for annotation in figure.layout.annotations)
    labels = {annotation.text: annotation.yshift for annotation in figure.layout.annotations}
    assert labels["1.00"] == labels["3.00"]
    assert labels["2.00"] > labels["1.00"]
    assert figure.layout.xaxis.tickfont.color == "#6B7280"

    tight = sampling_histogram(
        values=values,
        edges=edges,
        population=center,
        title="Sampling distribution of β̂",
        x_title="Estimated slope β̂",
        mean=center,
        standard_error=0.02,
        decimals=3,
    )
    tight_labels = {
        annotation.text: annotation.yshift
        for annotation in tight.layout.annotations
        if annotation.text != "±1 SE"
    }
    assert set(tight_labels) == {"1.980", "2.000", "2.020"}
    assert len(set(tight_labels.values())) == 3


def test_animation_batches_cover_every_repetition():
    assert batch_sizes(1) == [1]
    assert batch_sizes(10) == [1] * 10
    assert sum(batch_sizes(100)) == 100
    assert sum(batch_sizes(1000)) == 1000
    assert len(batch_sizes(1000)) == 16
