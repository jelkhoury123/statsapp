"""Simple random samples from the fixed population, without replacement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.population import Population


@dataclass
class Sample:
    """One sample of students. Order within the sample does not matter."""

    student_id: np.ndarray
    hours_studied: np.ndarray
    sleep_hours: np.ndarray
    attendance: np.ndarray
    exam_score: np.ndarray
    indices: np.ndarray

    @property
    def n(self) -> int:
        return int(self.student_id.size)


def validate_sample_size(population_size: int, sample_size: int) -> None:
    """Reject sample sizes that cannot be drawn without replacement."""
    if sample_size < 2:
        raise ValueError("Sample size must be at least 2 to fit a line")
    if sample_size > population_size:
        raise ValueError(
            f"Sample size {sample_size} exceeds the population size {population_size}"
        )


def draw_sample(population: Population, sample_size: int, rng: np.random.Generator) -> Sample:
    """Draw one simple random sample without replacement.

    Each call advances ``rng``. Restoring the same generator state, or
    constructing a new generator with the same seed and the same preceding
    draws, reproduces the sample. Students are not put back: IDs inside
    one sample are unique. A later sample is drawn independently and may
    include students who appeared before.
    """
    validate_sample_size(population.n, sample_size)
    indices = rng.choice(population.n, size=sample_size, replace=False)
    return Sample(
        student_id=population.student_id[indices].copy(),
        hours_studied=population.hours_studied[indices].copy(),
        sleep_hours=population.sleep_hours[indices].copy(),
        attendance=population.attendance[indices].copy(),
        exam_score=population.exam_score[indices].copy(),
        indices=np.asarray(indices, dtype=np.int64).copy(),
    )
