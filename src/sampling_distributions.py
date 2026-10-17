"""Sampling distributions of the intercept and slope.

Each stored coefficient is an ordinary-least-squares fit of one simple
random sample drawn without replacement from the fixed population.
Repetitions are independent, so a student may appear in more than one sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.regression import fit_ols, fit_ols_many

BIN_COUNT = 50
_RANGE_IN_STANDARD_ERRORS = 4.5


@dataclass(frozen=True)
class SamplingExperiment:
    """Estimates accumulated at one sample size, plus stable histogram bins."""

    sample_size: int
    alphas: np.ndarray
    betas: np.ndarray
    sample_x: np.ndarray | None
    sample_y: np.ndarray | None
    show_normals: bool
    alpha_edges: np.ndarray
    beta_edges: np.ndarray

    @property
    def n_repetitions(self) -> int:
        return int(self.alphas.size)


def expected_standard_errors(
    hours: np.ndarray,
    scores: np.ndarray,
    sample_size: int,
) -> tuple[float, float]:
    """Rough standard errors for a sample of this size from these arrays.

    They set the histogram window. They are not residual-based standard
    errors from a single sample.
    """
    fit = fit_ols(hours, scores)
    residual_df = max(int(hours.size) - 2, 1)
    sigma = float(np.sqrt(fit.sse / residual_df))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    centered = np.asarray(hours, dtype=np.float64) - float(np.mean(hours))
    sxx_population = float(np.dot(centered, centered))
    sxx = (sample_size - 1) / max(int(hours.size) - 1, 1) * sxx_population
    sxx = max(sxx, 1e-8)
    se_beta = sigma / np.sqrt(sxx)
    se_alpha = sigma * np.sqrt(1.0 / sample_size + float(np.mean(hours)) ** 2 / sxx)
    return float(se_alpha), float(se_beta)


def initial_edges(center: float, standard_error: float) -> np.ndarray:
    """Fixed bin edges, wide enough for typical sampling variation."""
    scale = max(float(standard_error), 1e-3)
    half = _RANGE_IN_STANDARD_ERRORS * scale
    width = (2.0 * half) / BIN_COUNT
    left = float(center) - half
    return left + width * np.arange(BIN_COUNT + 1)


def expand_edges(edges: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Grow the axis by whole bins when an estimate falls outside.

    The bin width stays the same, so bars that were already drawn do not
    move. Nothing is dropped.
    """
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return edges
    width = float(edges[1] - edges[0])
    left = float(edges[0])
    right = float(edges[-1])
    low = float(finite.min())
    high = float(finite.max())
    if low < left:
        left -= int(np.ceil((left - low) / width)) * width
    if high > right:
        right += int(np.ceil((high - right) / width)) * width
    if left == float(edges[0]) and right == float(edges[-1]):
        return edges
    count = int(round((right - left) / width))
    return left + width * np.arange(count + 1)


def new_experiment(
    sample_size: int,
    hours: np.ndarray,
    scores: np.ndarray,
    population_alpha: float,
    population_beta: float,
) -> SamplingExperiment:
    se_alpha, se_beta = expected_standard_errors(hours, scores, int(sample_size))
    return SamplingExperiment(
        sample_size=int(sample_size),
        alphas=np.empty(0, dtype=np.float64),
        betas=np.empty(0, dtype=np.float64),
        sample_x=None,
        sample_y=None,
        show_normals=False,
        alpha_edges=initial_edges(population_alpha, se_alpha),
        beta_edges=initial_edges(population_beta, se_beta),
    )


def experiment_for_sample_size(
    experiment: SamplingExperiment,
    sample_size: int,
    hours: np.ndarray,
    scores: np.ndarray,
    population_alpha: float,
    population_beta: float,
) -> SamplingExperiment:
    """A new sample size starts a new histogram. Estimates are not mixed."""
    if int(sample_size) == experiment.sample_size:
        return experiment
    return new_experiment(sample_size, hours, scores, population_alpha, population_beta)


def draw_regressions(
    hours: np.ndarray,
    scores: np.ndarray,
    *,
    sample_size: int,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit one OLS line per repetition. Returns coefficients and the last sample."""
    x_values = np.asarray(hours, dtype=np.float64)
    y_values = np.asarray(scores, dtype=np.float64)
    alphas = np.empty(repetitions, dtype=np.float64)
    betas = np.empty(repetitions, dtype=np.float64)
    last_x = np.empty(sample_size, dtype=np.float64)
    last_y = np.empty(sample_size, dtype=np.float64)
    population_size = int(x_values.size)
    batch_size = 400
    for start in range(0, repetitions, batch_size):
        stop = min(start + batch_size, repetitions)
        count = stop - start
        indices = np.empty((count, sample_size), dtype=np.int64)
        for row in range(count):
            indices[row] = rng.choice(population_size, size=sample_size, replace=False)
        alpha_batch, beta_batch = fit_ols_many(x_values[indices], y_values[indices])
        alphas[start:stop] = alpha_batch
        betas[start:stop] = beta_batch
        last_x = x_values[indices[-1]]
        last_y = y_values[indices[-1]]
    return alphas, betas, last_x, last_y


def add_draws(
    experiment: SamplingExperiment,
    hours: np.ndarray,
    scores: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> SamplingExperiment:
    """Append repetitions. Previous estimates stay."""
    alphas, betas, sample_x, sample_y = draw_regressions(
        hours,
        scores,
        sample_size=experiment.sample_size,
        repetitions=int(repetitions),
        rng=rng,
    )
    return SamplingExperiment(
        sample_size=experiment.sample_size,
        alphas=np.concatenate([experiment.alphas, alphas]),
        betas=np.concatenate([experiment.betas, betas]),
        sample_x=sample_x,
        sample_y=sample_y,
        show_normals=experiment.show_normals,
        alpha_edges=expand_edges(experiment.alpha_edges, alphas),
        beta_edges=expand_edges(experiment.beta_edges, betas),
    )


def batch_sizes(repetitions: int) -> list[int]:
    """How to reveal a draw so the histograms are seen filling in."""
    count = int(repetitions)
    if count <= 10:
        return [1] * count
    if count <= 100:
        return [5] * (count // 5) + ([count % 5] if count % 5 else [])
    frames = 16
    base, extra = divmod(count, frames)
    return [base + (1 if index < extra else 0) for index in range(frames)]


def animated_draws(
    experiment: SamplingExperiment,
    hours: np.ndarray,
    scores: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> list[SamplingExperiment]:
    """Snapshots after each animation step, ending at the full draw."""
    current = experiment
    frames: list[SamplingExperiment] = []
    for batch in batch_sizes(repetitions):
        current = add_draws(current, hours, scores, batch, rng)
        frames.append(current)
    return frames


def finite_estimates(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def estimate_mean(values: np.ndarray) -> float | None:
    finite = finite_estimates(values)
    if finite.size == 0:
        return None
    return float(finite.mean())


def estimate_sd(values: np.ndarray) -> float | None:
    """Sample standard deviation. Undefined for fewer than two estimates."""
    finite = finite_estimates(values)
    if finite.size < 2:
        return None
    spread = float(np.std(finite, ddof=1))
    if not np.isfinite(spread):
        return None
    return spread


BENCHMARK_REPETITIONS = 100_000
BENCHMARK_SEED = 20261016
_SAMPLING_DESIGN = "simple_random_without_replacement"
_BENCHMARK_CACHE: dict[tuple, "PopulationBenchmark"] = {}


@dataclass(frozen=True)
class PopulationBenchmark:
    """Sampling SD of the OLS coefficients, from a large independent simulation."""

    sample_size: int
    repetitions: int
    se_alpha: float | None
    se_beta: float | None


def monte_carlo_standard_errors(
    hours: np.ndarray,
    scores: np.ndarray,
    sample_size: int,
    repetitions: int,
    seed: int,
) -> tuple[float | None, float | None]:
    """SD of OLS estimates over independent samples drawn without replacement.

    The generator is created here, so the call does not advance any other stream.
    """
    rng = np.random.default_rng(int(seed))
    alphas, betas, _, _ = draw_regressions(
        hours,
        scores,
        sample_size=int(sample_size),
        repetitions=int(repetitions),
        rng=rng,
    )
    return estimate_sd(alphas), estimate_sd(betas)


def _benchmark_key(population_key: str, sample_size: int) -> tuple:
    return (
        population_key,
        int(sample_size),
        _SAMPLING_DESIGN,
        BENCHMARK_REPETITIONS,
        BENCHMARK_SEED,
    )


def benchmark_is_cached(population_key: str, sample_size: int) -> bool:
    return _benchmark_key(population_key, sample_size) in _BENCHMARK_CACHE


def population_benchmark(
    hours: np.ndarray,
    scores: np.ndarray,
    sample_size: int,
    population_key: str,
) -> PopulationBenchmark:
    """Cached Monte Carlo benchmark for this population and sample size.

    This is the sampling SD of the estimator, not the OLS standard error
    of one regression fitted to all 100,000 students.
    """
    key = _benchmark_key(population_key, sample_size)
    cached = _BENCHMARK_CACHE.get(key)
    if cached is not None:
        return cached
    se_alpha, se_beta = monte_carlo_standard_errors(
        hours,
        scores,
        sample_size,
        BENCHMARK_REPETITIONS,
        BENCHMARK_SEED,
    )
    cached = PopulationBenchmark(
        sample_size=int(sample_size),
        repetitions=BENCHMARK_REPETITIONS,
        se_alpha=se_alpha,
        se_beta=se_beta,
    )
    _BENCHMARK_CACHE[key] = cached
    return cached


def relative_se_error(estimated: float | None, benchmark: float | None) -> float | None:
    """Signed percent by which an SE estimate misses the population benchmark.

    Positive means the estimate is larger than the benchmark. This is not
    the error in the coefficient itself.
    """
    if estimated is None or benchmark is None:
        return None
    if not np.isfinite(estimated) or not np.isfinite(benchmark) or benchmark == 0:
        return None
    return 100.0 * (float(estimated) - float(benchmark)) / float(benchmark)


def can_show_normal(values: np.ndarray) -> bool:
    spread = estimate_sd(values)
    return spread is not None and spread > 0


def histogram_counts(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    finite = finite_estimates(values)
    counts, _ = np.histogram(finite, bins=edges)
    return counts


def normal_frequency(
    x: np.ndarray,
    mean: float,
    standard_deviation: float,
    count: int,
    bin_width: float,
) -> np.ndarray:
    """Normal density scaled to a frequency histogram: N times bin width."""
    scale = float(count) * float(bin_width)
    coeff = scale / (float(standard_deviation) * np.sqrt(2.0 * np.pi))
    z = (np.asarray(x, dtype=np.float64) - float(mean)) / float(standard_deviation)
    return coeff * np.exp(-0.5 * z * z)


@dataclass(frozen=True)
class OneSampleState:
    """The single sample used to estimate standard errors. Not part of the reference."""

    sample_size: int
    x: np.ndarray | None
    y: np.ndarray | None
    show_residuals: bool = True
    show_band: bool = False


@dataclass(frozen=True)
class OneSampleError:
    """Uncertainty estimated from one sample's residuals, not from repeated draws."""

    n: int
    alpha: float
    beta: float
    residual_sd: float | None
    se_alpha: float | None
    se_beta: float | None
    sxx: float | None


def one_sample_standard_errors(x: np.ndarray, y: np.ndarray) -> OneSampleError:
    """Homoskedastic OLS standard errors from this sample alone.

    Residual variance uses n − 2. The population coefficients are not used.
    A sample with no residual degrees of freedom, or no spread in x, has no SE.
    """
    fit = fit_ols(x, y)
    n = int(fit.n)
    empty = OneSampleError(n, fit.alpha, fit.beta, None, None, None, None)
    if fit.degenerate or n < 3:
        return empty
    x_values = np.asarray(x, dtype=np.float64)
    x_mean = float(x_values.mean())
    centered = x_values - x_mean
    sxx = float(np.dot(centered, centered))
    if not np.isfinite(sxx) or sxx <= 1e-12:
        return empty
    residual_sd = float(np.sqrt(fit.sse / (n - 2)))
    if not np.isfinite(residual_sd):
        return empty
    se_beta = residual_sd / np.sqrt(sxx)
    se_alpha = residual_sd * np.sqrt(1.0 / n + (x_mean * x_mean) / sxx)
    return OneSampleError(
        n=n,
        alpha=float(fit.alpha),
        beta=float(fit.beta),
        residual_sd=residual_sd,
        se_alpha=float(se_alpha),
        se_beta=float(se_beta),
        sxx=sxx,
    )


def one_sample_for_size(
    state: OneSampleState | None,
    sample_size: int,
    experiment: SamplingExperiment,
) -> OneSampleState:
    """Keep the one-sample view when it already matches n.

    A different sample size drops it. The first visit copies Step 1's latest
    sample when there is one, and does not draw a new sample by itself.
    """
    if state is not None and state.sample_size == int(sample_size) and state.x is not None:
        return state
    show_residuals = True
    show_band = False
    if state is not None and state.sample_size == int(sample_size):
        show_residuals = state.show_residuals
        show_band = state.show_band
    if (
        experiment.sample_x is not None
        and experiment.sample_y is not None
        and experiment.sample_size == int(sample_size)
    ):
        return OneSampleState(
            sample_size=int(sample_size),
            x=np.array(experiment.sample_x, copy=True),
            y=np.array(experiment.sample_y, copy=True),
            show_residuals=show_residuals,
            show_band=show_band,
        )
    if state is not None and state.sample_size == int(sample_size):
        return state
    return OneSampleState(sample_size=int(sample_size), x=None, y=None)


def histogram_density(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Histogram as a probability density. The bars integrate to one."""
    finite = finite_estimates(values)
    counts, _ = np.histogram(finite, bins=edges)
    density = counts.astype(np.float64)
    if finite.size == 0:
        return density
    width = float(edges[1] - edges[0])
    return density / (finite.size * width)


def normal_density(x: np.ndarray, mean: float, standard_deviation: float) -> np.ndarray:
    """Standard normal density on the probability scale, not a frequency scale."""
    coeff = 1.0 / (float(standard_deviation) * np.sqrt(2.0 * np.pi))
    z = (np.asarray(x, dtype=np.float64) - float(mean)) / float(standard_deviation)
    return coeff * np.exp(-0.5 * z * z)


def residual_segment_indices(x: np.ndarray, limit: int = 80) -> np.ndarray:
    """A spread of positions for drawing residual segments.

    Calculations still use every observation. This only chooses which
    segments to draw when the sample is large.
    """
    values = np.asarray(x)
    count = int(values.size)
    if count <= limit:
        return np.arange(count)
    order = np.argsort(values, kind="mergesort")
    pick = np.linspace(0, count - 1, limit).astype(np.int64)
    return order[pick]
