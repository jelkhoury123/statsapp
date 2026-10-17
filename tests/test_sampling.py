"""Sampling without replacement from the fixed population."""

from __future__ import annotations

import numpy as np
import pytest

from src.population import DEFAULT_CSV_PATH, file_sha256, load_population
from src.sampling import draw_sample


@pytest.fixture(scope="module")
def population():
    return load_population()


def test_sample_ids_are_unique_and_belong_to_the_population(population):
    sample = draw_sample(population, 500, np.random.default_rng(5))
    assert sample.n == 500
    assert len(np.unique(sample.student_id)) == 500
    assert len(np.unique(sample.indices)) == 500
    assert set(sample.student_id.tolist()).issubset(set(population.student_id.tolist()))


def test_sample_cannot_exceed_the_population(population):
    with pytest.raises(ValueError):
        draw_sample(population, population.n + 1, np.random.default_rng(1))


def test_same_seed_draws_the_same_students(population):
    first = draw_sample(population, 80, np.random.default_rng(19))
    second = draw_sample(population, 80, np.random.default_rng(19))
    assert np.array_equal(first.student_id, second.student_id)
    assert np.array_equal(first.hours_studied, second.hours_studied)


def test_drawing_does_not_change_the_population_or_the_file(population):
    digest = file_sha256(DEFAULT_CSV_PATH)
    hours_before = population.hours_studied.copy()
    draw_sample(population, 1000, np.random.default_rng(2))
    assert np.array_equal(hours_before, population.hours_studied)
    assert file_sha256(DEFAULT_CSV_PATH) == digest
