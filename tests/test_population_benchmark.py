"""The population benchmark is a separate Monte Carlo, not one regression on everyone."""

from __future__ import annotations

import copy
import inspect

import numpy as np
import pytest

from src.population import load_population
from src.regression import fit_ols
from src.sampling import draw_sample
from src.sampling_distributions import (
    BENCHMARK_REPETITIONS,
    _BENCHMARK_CACHE,
    add_draws,
    draw_regressions,
    monte_carlo_standard_errors,
    new_experiment,
    one_sample_standard_errors,
    population_benchmark,
    relative_se_error,
)


@pytest.fixture(scope="module")
def population():
    return load_population()


def test_shared_sampler_draws_without_replacement():
    assert "replace=False" in inspect.getsource(draw_regressions)
    assert "draw_regressions" in monte_carlo_standard_errors.__code__.co_names


def test_monte_carlo_is_reproducible_and_ignores_other_generators(population):
    interactive = np.random.default_rng(19)
    before = copy.deepcopy(interactive.bit_generator.state)
    interactive.random()
    moved = copy.deepcopy(interactive.bit_generator.state)
    first = monte_carlo_standard_errors(
        population.hours_studied, population.exam_score, 25, 40, 7
    )
    second = monte_carlo_standard_errors(
        population.hours_studied, population.exam_score, 25, 40, 7
    )
    assert first == second
    assert first[1] is not None
    interactive.random()
    assert interactive.bit_generator.state != moved
    restored = np.random.default_rng(19)
    restored.bit_generator.state = before
    restored.random()
    assert restored.bit_generator.state == moved


def test_benchmark_is_cached_and_is_not_the_full_population_regression(population, monkeypatch):
    _BENCHMARK_CACHE.clear()
    calls = {"n": 0}
    real = monte_carlo_standard_errors

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(
        "src.sampling_distributions.monte_carlo_standard_errors",
        wrapped,
    )
    first = population_benchmark(
        population.hours_studied, population.exam_score, 60, population.sha256
    )
    second = population_benchmark(
        population.hours_studied, population.exam_score, 60, population.sha256
    )
    assert first is second
    assert calls["n"] == 1
    assert first.repetitions == BENCHMARK_REPETITIONS
    assert first.sample_size == 60

    fit = fit_ols(population.hours_studied, population.exam_score)
    whole_sd = np.sqrt(fit.sse / (population.n - 2))
    centered = population.hours_studied - population.hours_studied.mean()
    whole_se = whole_sd / np.sqrt(float(np.dot(centered, centered)))
    assert first.se_beta > whole_se * 10


def test_a_larger_sample_has_a_smaller_benchmark_and_a_new_sample_does_not_move_it(population):
    small = monte_carlo_standard_errors(
        population.hours_studied, population.exam_score, 30, 600, 3
    )
    large = monte_carlo_standard_errors(
        population.hours_studied, population.exam_score, 300, 600, 3
    )
    assert small[0] > large[0]
    assert small[1] > large[1]

    _BENCHMARK_CACHE.clear()
    benchmark = population_benchmark(
        population.hours_studied, population.exam_score, 45, population.sha256
    )
    before = (benchmark.se_alpha, benchmark.se_beta)
    sample = one_sample_standard_errors(population.hours_studied[:45], population.exam_score[:45])
    again = population_benchmark(
        population.hours_studied, population.exam_score, 45, population.sha256
    )
    assert again is benchmark
    assert (again.se_alpha, again.se_beta) == before
    assert sample.se_beta != pytest.approx(benchmark.se_beta)


def test_draws_do_not_change_the_benchmark_or_the_single_sample_formula(population):
    x = population.hours_studied[:40]
    y = population.exam_score[:40]
    before = one_sample_standard_errors(x, y)
    benchmark = population_benchmark(
        population.hours_studied, population.exam_score, 80, population.sha256
    )
    after = one_sample_standard_errors(x, y)
    assert (after.se_alpha, after.se_beta, after.sxx) == (before.se_alpha, before.se_beta, before.sxx)

    interactive = np.random.default_rng(4)
    state = copy.deepcopy(interactive.bit_generator.state)
    draw_sample(population, 80, interactive)
    assert interactive.bit_generator.state != state
    assert (
        population_benchmark(
            population.hours_studied, population.exam_score, 80, population.sha256
        )
        is benchmark
    )

    experiment = new_experiment(
        80,
        population.hours_studied,
        population.exam_score,
        population.alpha,
        population.beta,
    )
    add_draws(experiment, population.hours_studied, population.exam_score, 5, interactive)
    again = population_benchmark(
        population.hours_studied, population.exam_score, 80, population.sha256
    )
    assert again is benchmark
    other = population_benchmark(
        population.hours_studied, population.exam_score, 120, population.sha256
    )
    assert other is not benchmark
    assert other.sample_size == 120
    assert other.se_beta < benchmark.se_beta


def test_relative_error_is_the_signed_gap_in_the_standard_error():
    assert relative_se_error(0.186, 0.163) == pytest.approx(100 * (0.186 - 0.163) / 0.163)
    assert relative_se_error(0.10, 0.20) == pytest.approx(-50.0)
    assert relative_se_error(None, 0.2) is None
    assert relative_se_error(0.2, 0.0) is None
