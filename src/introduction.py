"""The opening experiment: population, then one sample, then a fitted line.

The population file is only read. Drawing a sample does not fit a line.
The fit, when requested, uses the sampled rows and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.population import Population
from src.regression import RegressionResult, fit_ols
from src.sampling import Sample, draw_sample


@dataclass(frozen=True)
class IntroductionView:
    """What the opening chart is allowed to show."""

    sample: Sample | None = None
    fit: RegressionResult | None = None
    population_revealed: bool = False


def sampling_fraction(sample_size: int, population_size: int) -> float:
    """Share of the population that is in the sample, n / N."""
    if population_size <= 0:
        raise ValueError("Population size must be positive")
    if sample_size < 0 or sample_size > population_size:
        raise ValueError("Sample size must fall between 0 and the population size")
    return sample_size / population_size


def format_sampling_fraction(sample_size: int, population_size: int) -> str:
    """The sampling fraction as a percent, for the metric card."""
    fraction = sampling_fraction(sample_size, population_size)
    return f"{fraction:.2%}"


def population_cloud_caption(population_size: int) -> str:
    """Label for the grey cloud. It is every student, not a display subset."""
    return (
        f"Grey cloud: every student in the population (N = {population_size:,}). "
        "It is drawn once and cached. Blue points are the sampled students. "
        "Sampling and regression use the complete population, not a display subset."
    )


def coefficient_gap(
    fit: RegressionResult,
    population_alpha: float,
    population_beta: float,
) -> tuple[float, float]:
    """Sample estimate minus the population coefficient."""
    return (fit.alpha - population_alpha, fit.beta - population_beta)


def displayed_coefficient_gap(
    fit: RegressionResult,
    population_alpha: float,
    population_beta: float,
    digits: int = 3,
) -> tuple[float, float]:
    """Difference of the rounded coefficients shown to students.

    Rounding each coefficient first keeps the arithmetic on screen consistent:
    the displayed difference equals the displayed estimate minus the displayed
    population value.
    """
    return (
        round(fit.alpha, digits) - round(population_alpha, digits),
        round(fit.beta, digits) - round(population_beta, digits),
    )


def select_sample(
    view: IntroductionView,
    population: Population,
    sample_size: int,
    rng: np.random.Generator,
) -> IntroductionView:
    """Draw a new sample and clear the sample regression.

    The population line stays visible if it has already been revealed.
    The population itself is not modified.
    """
    sample = draw_sample(population, sample_size, rng)
    return IntroductionView(
        sample=sample,
        fit=None,
        population_revealed=view.population_revealed,
    )


def estimate_sample(view: IntroductionView) -> IntroductionView:
    """Fit the line from the current sample only."""
    if view.sample is None:
        raise ValueError("Draw a sample before estimating a regression")
    fit = fit_ols(view.sample.hours_studied, view.sample.exam_score)
    if fit.n != view.sample.n:
        raise RuntimeError("The sample regression was not fit on the sample alone")
    return IntroductionView(
        sample=view.sample,
        fit=fit,
        population_revealed=view.population_revealed,
    )


def reveal_population(view: IntroductionView) -> IntroductionView:
    """Show the population comparison. The sample fit is left as it is."""
    return IntroductionView(
        sample=view.sample,
        fit=view.fit,
        population_revealed=True,
    )


def hide_population(view: IntroductionView) -> IntroductionView:
    """Hide the population comparison. The sample fit is left as it is."""
    return IntroductionView(
        sample=view.sample,
        fit=view.fit,
        population_revealed=False,
    )
