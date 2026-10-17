"""A fixed population whose true slope is zero, and the classical t-test for that slope."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.stats import t as student_t

from src.sampling_distributions import one_sample_standard_errors

NULL_SIZE = 100_000
NULL_ALPHA = 45.0
NULL_BETA = 0.0
NULL_SIGMA = 5.0
NULL_SEED = 20261016
SIGNIFICANCE_LEVEL = 0.05


@dataclass(frozen=True)
class NullPopulation:
    hours: np.ndarray
    scores: np.ndarray
    alpha: float
    beta: float


@dataclass(frozen=True)
class SlopeTTest:
    """Two-sided t-test of H0: β = 0 from one sample."""

    n: int
    beta: float | None
    standard_error: float | None
    t_statistic: float | None
    degrees_of_freedom: int
    p_value: float | None
    critical_value: float | None
    reject: bool | None


@lru_cache(maxsize=1)
def _null_design() -> tuple[np.ndarray, np.ndarray]:
    """Fixed hours and noise. The same students are reused for every true slope."""
    generator = np.random.default_rng(NULL_SEED)
    hours = generator.uniform(0.0, 10.0, NULL_SIZE)
    noise = generator.normal(0.0, NULL_SIGMA, NULL_SIZE)
    return hours, noise


def null_population(beta: float = NULL_BETA) -> NullPopulation:
    """Scores are 45 + β × hours + noise. β = 0 leaves scores equal to 45 plus noise."""
    hours, noise = _null_design()
    slope = float(beta)
    return NullPopulation(
        hours=hours.copy(),
        scores=NULL_ALPHA + slope * hours + noise,
        alpha=NULL_ALPHA,
        beta=slope,
    )


def draw_null_sample(
    population: NullPopulation,
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    indices = rng.choice(population.hours.size, size=int(sample_size), replace=False)
    return population.hours[indices].copy(), population.scores[indices].copy()


def slope_t_test(x: np.ndarray, y: np.ndarray) -> SlopeTTest:
    """t = β̂ / SE(β̂), with SE from this sample's residuals and df = n − 2."""
    error = one_sample_standard_errors(x, y)
    degrees = int(error.n) - 2
    if (
        error.beta is None
        or error.se_beta is None
        or not np.isfinite(error.beta)
        or not np.isfinite(error.se_beta)
        or error.se_beta <= 0
        or degrees <= 0
    ):
        return SlopeTTest(int(error.n), error.beta, error.se_beta, None, degrees, None, None, None)
    statistic = float(error.beta) / float(error.se_beta)
    p_value = float(min(1.0, 2.0 * student_t.sf(abs(statistic), degrees)))
    critical = float(student_t.ppf(1.0 - SIGNIFICANCE_LEVEL / 2.0, degrees))
    return SlopeTTest(
        n=int(error.n),
        beta=float(error.beta),
        standard_error=float(error.se_beta),
        t_statistic=statistic,
        degrees_of_freedom=degrees,
        p_value=p_value,
        critical_value=critical,
        reject=p_value < SIGNIFICANCE_LEVEL,
    )


def t_density_curve(degrees: int, statistic: float | None, critical: float | None) -> tuple[np.ndarray, np.ndarray]:
    reach = 4.0
    if statistic is not None and np.isfinite(statistic):
        reach = max(reach, abs(statistic) * 1.35)
    if critical is not None and np.isfinite(critical):
        reach = max(reach, abs(critical) * 1.25)
    grid = np.linspace(-reach, reach, 401)
    return grid, student_t.pdf(grid, degrees)
