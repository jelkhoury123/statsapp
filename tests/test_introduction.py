"""The opening experiment: one fixed population, then a sample, then a line."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from src.charts import introduction_figure, population_cloud_image
from src.introduction import (
    IntroductionView,
    coefficient_gap,
    displayed_coefficient_gap,
    estimate_sample,
    format_sampling_fraction,
    hide_population,
    population_cloud_caption,
    reveal_population,
    sampling_fraction,
    select_sample,
)
from src.population import DEFAULT_CSV_PATH, file_sha256, load_population
from src.regression import fit_ols


@pytest.fixture(scope="module")
def population():
    return load_population()


def _trace_names(figure) -> list[str]:
    return [trace.name for trace in figure.data]


def _finite_coordinates(figure) -> int:
    total = 0
    for trace in figure.data:
        if trace.x is None:
            continue
        total += sum(value is not None and np.isfinite(value) for value in trace.x)
    return total


def test_population_file_is_unchanged_by_the_introduction(population):
    digest = file_sha256(DEFAULT_CSV_PATH)
    alpha, beta = population.alpha, population.beta
    rng = np.random.default_rng(11)
    view = select_sample(IntroductionView(), population, 100, rng)
    view = reveal_population(estimate_sample(view))
    select_sample(view, population, 100, rng)

    assert file_sha256(DEFAULT_CSV_PATH) == digest
    assert population.sha256 == digest
    assert population.n == 100_000
    assert population.alpha == alpha
    assert population.beta == beta


def test_sample_ids_belong_to_the_population_and_are_unique(population):
    view = select_sample(IntroductionView(), population, 250, np.random.default_rng(3))
    sample = view.sample
    assert sample is not None
    assert sample.n == 250
    assert len(np.unique(sample.student_id)) == 250
    assert len(np.unique(sample.indices)) == 250
    assert set(sample.student_id.tolist()).issubset(set(population.student_id.tolist()))
    assert np.all(population.student_id[sample.indices] == sample.student_id)


def test_sampling_fraction_matches_n_over_n(population):
    assert sampling_fraction(100, population.n) == pytest.approx(100 / 100_000)
    assert format_sampling_fraction(100, population.n) == "0.10%"
    assert format_sampling_fraction(5000, population.n) == "5.00%"
    with pytest.raises(ValueError):
        sampling_fraction(population.n + 1, population.n)


def test_sample_regression_uses_only_the_sampled_rows(population):
    view = select_sample(IntroductionView(), population, 80, np.random.default_rng(8))
    fitted = estimate_sample(view)
    sample = fitted.sample
    direct = fit_ols(sample.hours_studied, sample.exam_score)
    from_indices = fit_ols(
        population.hours_studied[sample.indices],
        population.exam_score[sample.indices],
    )

    assert fitted.fit.n == 80
    assert fitted.fit.n != population.n
    assert fitted.fit.alpha == pytest.approx(direct.alpha)
    assert fitted.fit.beta == pytest.approx(direct.beta)
    assert fitted.fit.alpha == pytest.approx(from_indices.alpha)
    assert fitted.fit.beta == pytest.approx(from_indices.beta)
    assert fitted.fit.alpha != pytest.approx(population.alpha)
    assert view.fit is None


def test_population_regression_uses_all_students(population):
    full = fit_ols(population.hours_studied, population.exam_score)
    assert full.n == 100_000
    assert full.alpha == pytest.approx(population.alpha)
    assert full.beta == pytest.approx(population.beta)

    view = reveal_population(
        estimate_sample(select_sample(IntroductionView(), population, 40, np.random.default_rng(2)))
    )
    assert view.population_revealed
    assert view.fit.n == 40
    gap_alpha, gap_beta = coefficient_gap(view.fit, population.alpha, population.beta)
    assert gap_alpha == pytest.approx(view.fit.alpha - population.alpha)
    assert gap_beta == pytest.approx(view.fit.beta - population.beta)
    shown_alpha, shown_beta = displayed_coefficient_gap(view.fit, population.alpha, population.beta)
    assert shown_alpha == pytest.approx(round(view.fit.alpha, 3) - round(population.alpha, 3))
    assert shown_beta == pytest.approx(round(view.fit.beta, 3) - round(population.beta, 3))


def test_population_comparison_can_be_hidden_and_shown_again(population):
    view = reveal_population(
        estimate_sample(select_sample(IntroductionView(), population, 40, np.random.default_rng(2)))
    )
    hidden = hide_population(view)
    assert hidden.population_revealed is False
    assert hidden.fit.n == 40
    assert hidden.fit.alpha == pytest.approx(view.fit.alpha)
    assert hidden.fit.beta == pytest.approx(view.fit.beta)
    assert hidden.sample.n == view.sample.n

    shown = reveal_population(hidden)
    assert shown.population_revealed is True
    assert shown.fit.alpha == pytest.approx(view.fit.alpha)
    assert shown.fit.beta == pytest.approx(view.fit.beta)


def test_another_sample_changes_the_highlight_and_clears_the_fit(population):
    rng = np.random.default_rng(21)
    first = estimate_sample(select_sample(IntroductionView(), population, 100, rng))
    revealed = reveal_population(first)
    second = select_sample(revealed, population, 100, rng)

    assert set(first.sample.student_id.tolist()) != set(second.sample.student_id.tolist())
    assert second.fit is None
    assert second.population_revealed is True
    assert population.alpha == pytest.approx(fit_ols(population.hours_studied, population.exam_score).alpha)
    assert population.beta == pytest.approx(fit_ols(population.hours_studied, population.exam_score).beta)


def test_cloud_is_cached_and_the_chart_does_not_plot_every_student(population):
    image = population_cloud_image(
        population.hours_studied,
        population.exam_score,
        population.x_range,
        population.y_range,
        cache_key=population.sha256,
    )
    again = population_cloud_image(
        population.hours_studied,
        population.exam_score,
        population.x_range,
        population.y_range,
        cache_key=population.sha256,
    )
    assert again is image
    assert image.startswith("data:image/png;base64,")
    assert len(image) < 1_500_000

    raw = base64.b64decode(image.split(",", 1)[1])
    pixels = np.asarray(Image.open(io.BytesIO(raw)))
    assert int(np.any(pixels < 250, axis=-1).sum()) > 10_000

    view = select_sample(IntroductionView(), population, 60, np.random.default_rng(6))
    hidden = introduction_figure(
        cloud_image=image,
        x_range=population.x_range,
        y_range=population.y_range,
        sample_x=view.sample.hours_studied,
        sample_y=view.sample.exam_score,
        sample_ids=view.sample.student_id,
        sample_alpha=None,
        sample_beta=None,
        population_alpha=population.alpha,
        population_beta=population.beta,
        show_sample_line=False,
        show_population_line=False,
    )
    assert _finite_coordinates(hidden) == 60
    assert "Sample regression" not in _trace_names(hidden)
    assert "Population regression" not in _trace_names(hidden)
    assert hidden.layout.xaxis.range == pytest.approx(population.x_range)
    assert hidden.layout.yaxis.range == pytest.approx(population.y_range)

    fitted = estimate_sample(view)
    shown = introduction_figure(
        cloud_image=image,
        x_range=population.x_range,
        y_range=population.y_range,
        sample_x=fitted.sample.hours_studied,
        sample_y=fitted.sample.exam_score,
        sample_alpha=fitted.fit.alpha,
        sample_beta=fitted.fit.beta,
        population_alpha=population.alpha,
        population_beta=population.beta,
        show_sample_line=True,
        show_population_line=True,
    )
    names = _trace_names(shown)
    assert "Sampled students" in names
    assert "Sample regression" in names
    assert "Population regression" in names
    # Two line endpoints plus the sample. Not the other 99,940 students.
    assert _finite_coordinates(shown) == 64
    sample_line = shown.data[names.index("Sample regression")]
    population_line = shown.data[names.index("Population regression")]
    assert list(sample_line.y) == pytest.approx(fitted.fit.alpha + fitted.fit.beta * sample_line.x)
    assert list(population_line.y) == pytest.approx(
        population.alpha + population.beta * population_line.x
    )
    assert list(sample_line.y) != pytest.approx(list(population_line.y))


def test_cloud_caption_says_every_student_is_shown():
    caption = population_cloud_caption(100_000)
    assert "every student" in caption
    assert "N = 100,000" in caption
    assert "display subset" in caption
