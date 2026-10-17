"""Repeated sampling of ordinary-least-squares coefficients.

Each repetition draws its own simple random sample without replacement
from the same fixed arrays, fits one line, and stores the coefficients.
The Python loop is over samples. The regression itself is vectorized.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from src.regression import fit_ols, fit_ols_many

ProgressCallback = Callable[[int, int], None]

THEORETICAL_ALPHA = 20.0
THEORETICAL_BETA = 5.0


@dataclass(frozen=True)
class CoefficientSummary:
    """Sampling distribution of one coefficient, judged against the population."""

    population: float
    estimates: np.ndarray

    @property
    def finite(self) -> np.ndarray:
        values = np.asarray(self.estimates, dtype=np.float64)
        return values[np.isfinite(values)]

    @property
    def mean(self) -> float:
        values = self.finite
        if values.size == 0:
            return float("nan")
        return float(values.mean())

    @property
    def std(self) -> float:
        """Sample standard deviation of the estimates (ddof=1).

        This is the simulation's approximation to the standard error.
        """
        values = self.finite
        if values.size < 2:
            return float("nan")
        return float(np.std(values, ddof=1))

    @property
    def bias(self) -> float:
        """Mean estimate minus the population coefficient."""
        return self.mean - self.population


@dataclass(frozen=True)
class SimulationResult:
    sample_size: int
    repetitions: int
    alpha: CoefficientSummary
    beta: CoefficientSummary
    n_degenerate: int


@dataclass
class SampleSizeComparison:
    sample_sizes: tuple[int, ...]
    repetitions: int
    alpha_population: float
    beta_population: float
    alpha_estimates: dict[int, np.ndarray]
    beta_estimates: dict[int, np.ndarray]

    def rows(self) -> list[dict[str, float]]:
        table = []
        for size in self.sample_sizes:
            alpha = CoefficientSummary(self.alpha_population, self.alpha_estimates[size])
            beta = CoefficientSummary(self.beta_population, self.beta_estimates[size])
            table.append(
                {
                    "Sample size": size,
                    "Mean slope": beta.mean,
                    "SD slope": beta.std,
                    "Mean intercept": alpha.mean,
                    "SD intercept": alpha.std,
                }
            )
        return table


@dataclass
class NoiseLevel:
    sigma: float
    alpha_population: float
    beta_population: float
    alpha_estimates: np.ndarray
    beta_estimates: np.ndarray

    @property
    def alpha(self) -> CoefficientSummary:
        return CoefficientSummary(self.alpha_population, self.alpha_estimates)

    @property
    def beta(self) -> CoefficientSummary:
        return CoefficientSummary(self.beta_population, self.beta_estimates)


@dataclass
class NoiseComparison:
    """Sampling distributions from synthetic populations with different noise.

    These populations are built in memory. Nothing is written to the
    original student file.
    """

    sample_size: int
    repetitions: int
    population_size: int
    seed: int
    theoretical_alpha: float = THEORETICAL_ALPHA
    theoretical_beta: float = THEORETICAL_BETA
    levels: list[NoiseLevel] = field(default_factory=list)


def _draw_index_batch(
    population_size: int,
    sample_size: int,
    batch: int,
    rng: np.random.Generator,
) -> np.ndarray:
    indices = np.empty((batch, sample_size), dtype=np.int64)
    for row in range(batch):
        indices[row] = rng.choice(population_size, size=sample_size, replace=False)
    return indices


def simulate_coefficients(
    x: np.ndarray,
    y: np.ndarray,
    *,
    sample_size: int,
    repetitions: int,
    rng: np.random.Generator,
    alpha_population: float,
    beta_population: float,
    progress: ProgressCallback | None = None,
    batch_size: int = 400,
) -> SimulationResult:
    """Estimate the sampling distribution of the intercept and slope.

    Benchmarks are the caller-supplied population coefficients. Pass the
    ordinary-least-squares fit of the full population, not the theoretical
    generating values, when the data are the fixed student file.

    ``rng`` is advanced once per sample. A generator built with the same
    seed, and no earlier draws, reproduces the estimates.
    """
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if x_values.ndim != 1 or y_values.shape != x_values.shape:
        raise ValueError("x and y must be one-dimensional and the same length")
    if sample_size < 2:
        raise ValueError("Sample size must be at least 2")
    if sample_size > x_values.size:
        raise ValueError(
            f"Sample size {sample_size} exceeds the population size {x_values.size}"
        )
    if repetitions < 1:
        raise ValueError("Repetitions must be at least 1")
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1")

    alphas = np.empty(repetitions, dtype=np.float64)
    betas = np.empty(repetitions, dtype=np.float64)
    population_size = int(x_values.size)

    for start in range(0, repetitions, batch_size):
        stop = min(start + batch_size, repetitions)
        indices = _draw_index_batch(population_size, sample_size, stop - start, rng)
        alpha_batch, beta_batch = fit_ols_many(x_values[indices], y_values[indices])
        alphas[start:stop] = alpha_batch
        betas[start:stop] = beta_batch
        if progress is not None:
            progress(stop, repetitions)

    n_degenerate = int(np.isnan(betas).sum())
    return SimulationResult(
        sample_size=sample_size,
        repetitions=repetitions,
        alpha=CoefficientSummary(float(alpha_population), alphas),
        beta=CoefficientSummary(float(beta_population), betas),
        n_degenerate=n_degenerate,
    )


def compare_sample_sizes(
    x: np.ndarray,
    y: np.ndarray,
    sample_sizes: Sequence[int],
    *,
    repetitions: int,
    rng: np.random.Generator,
    alpha_population: float,
    beta_population: float,
    progress: ProgressCallback | None = None,
) -> SampleSizeComparison:
    """Run the same number of regressions at each requested sample size."""
    sizes = tuple(int(size) for size in sample_sizes)
    if not sizes:
        raise ValueError("At least one sample size is required")

    grand_total = repetitions * len(sizes)
    completed = 0
    alpha_estimates: dict[int, np.ndarray] = {}
    beta_estimates: dict[int, np.ndarray] = {}

    for size in sizes:
        def batch_progress(done: int, total: int, offset: int = completed) -> None:
            del total
            if progress is not None:
                progress(offset + done, grand_total)

        result = simulate_coefficients(
            x,
            y,
            sample_size=size,
            repetitions=repetitions,
            rng=rng,
            alpha_population=alpha_population,
            beta_population=beta_population,
            progress=batch_progress,
        )
        completed += repetitions
        alpha_estimates[size] = result.alpha.estimates
        beta_estimates[size] = result.beta.estimates

    return SampleSizeComparison(
        sample_sizes=sizes,
        repetitions=repetitions,
        alpha_population=float(alpha_population),
        beta_population=float(beta_population),
        alpha_estimates=alpha_estimates,
        beta_estimates=beta_estimates,
    )


def generate_synthetic_population(
    population_size: int,
    sigma: float,
    rng: np.random.Generator,
    *,
    alpha: float = THEORETICAL_ALPHA,
    beta: float = THEORETICAL_BETA,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a new population in memory. This does not touch the student file.

    Hours are drawn from Uniform(0, 10), then errors from Normal(0, sigma²),
    and the outcome is ``alpha + beta * hours + error``. The random draws
    happen in that order.
    """
    if population_size < 2:
        raise ValueError("Synthetic population must contain at least 2 rows")
    if sigma < 0:
        raise ValueError("Error standard deviation cannot be negative")
    hours = rng.uniform(0.0, 10.0, size=population_size)
    error = rng.normal(0.0, sigma, size=population_size)
    scores = alpha + beta * hours + error
    return hours, scores


def compare_noise_levels(
    sigmas: Sequence[float],
    *,
    sample_size: int,
    repetitions: int,
    population_size: int,
    seed: int,
    progress: ProgressCallback | None = None,
    alpha: float = THEORETICAL_ALPHA,
    beta: float = THEORETICAL_BETA,
) -> NoiseComparison:
    """Compare sampling spread at several error standard deviations.

    Each sigma gets its own synthetic population and its own sample stream,
    both derived from ``seed``. The same seed and settings reproduce the
    same estimates. The original student file is not read or written.

    The benchmark for each level is the least-squares line of that
    synthetic population, not a line refit after the samples are pooled.
    """
    levels: list[NoiseLevel] = []
    grand_total = repetitions * len(tuple(sigmas))
    completed = 0

    for sigma in sigmas:
        sigma_value = float(sigma)
        # Integer key so 5, 10 and 20 do not collide through float noise.
        sigma_key = int(round(sigma_value * 1000))
        population_rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), sigma_key, 1, population_size])
        )
        sample_rng = np.random.default_rng(
            np.random.SeedSequence(
                [int(seed), sigma_key, 2, population_size, sample_size, repetitions]
            )
        )
        hours, scores = generate_synthetic_population(
            population_size,
            sigma_value,
            population_rng,
            alpha=alpha,
            beta=beta,
        )
        population_fit = fit_ols(hours, scores)

        def batch_progress(done: int, total: int, offset: int = completed) -> None:
            del total
            if progress is not None:
                progress(offset + done, grand_total)

        result = simulate_coefficients(
            hours,
            scores,
            sample_size=sample_size,
            repetitions=repetitions,
            rng=sample_rng,
            alpha_population=population_fit.alpha,
            beta_population=population_fit.beta,
            progress=batch_progress,
        )
        completed += repetitions
        levels.append(
            NoiseLevel(
                sigma=sigma_value,
                alpha_population=population_fit.alpha,
                beta_population=population_fit.beta,
                alpha_estimates=result.alpha.estimates,
                beta_estimates=result.beta.estimates,
            )
        )

    return NoiseComparison(
        sample_size=sample_size,
        repetitions=repetitions,
        population_size=population_size,
        seed=int(seed),
        theoretical_alpha=alpha,
        theoretical_beta=beta,
        levels=levels,
    )
