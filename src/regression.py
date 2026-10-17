"""Ordinary least squares for a straight line with an intercept."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Predictor sums of squares at or below this are treated as zero variance.
_VARIANCE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class RegressionResult:
    """Fitted intercept and slope, plus summary fit statistics.

    ``alpha`` and ``beta`` are NaN when the regression is degenerate.
    """

    n: int
    alpha: float
    beta: float
    sse: float
    r_squared: float
    degenerate: bool


def _as_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        array = np.reshape(array, -1)
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    return array


def _degenerate(n: int) -> RegressionResult:
    return RegressionResult(
        n=n,
        alpha=np.nan,
        beta=np.nan,
        sse=np.nan,
        r_squared=np.nan,
        degenerate=True,
    )


def fit_ols(x: np.ndarray, y: np.ndarray) -> RegressionResult:
    """Fit ``Y = alpha + beta X`` by ordinary least squares.

    The slope is the covariance of x and y divided by the variance of x,
    using sums of squares about the sample means. The intercept makes the
    line pass through the sample means.

    If every predictor value is identical, the slope is undefined. The
    function then returns a degenerate result instead of raising, so a
    simulation can continue.
    """
    x_values = _as_vector(x, "x")
    y_values = _as_vector(y, "y")
    if x_values.shape != y_values.shape:
        raise ValueError("x and y must have the same length")
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise ValueError("x and y must contain only finite values")

    n = int(x_values.size)
    if n < 2:
        return _degenerate(n)

    x_mean = float(x_values.mean())
    y_mean = float(y_values.mean())
    x_centered = x_values - x_mean
    y_centered = y_values - y_mean
    sxx = float(np.dot(x_centered, x_centered))
    if not np.isfinite(sxx) or sxx <= _VARIANCE_TOLERANCE:
        return _degenerate(n)

    sxy = float(np.dot(x_centered, y_centered))
    beta = sxy / sxx
    alpha = y_mean - beta * x_mean
    sst = float(np.dot(y_centered, y_centered))
    # Residual sum of squares for a model with an intercept.
    sse = sst - beta * sxy
    if sse < 0.0:
        sse = 0.0
    if sst <= _VARIANCE_TOLERANCE:
        r_squared = np.nan
    else:
        r_squared = 1.0 - sse / sst
    return RegressionResult(
        n=n,
        alpha=float(alpha),
        beta=float(beta),
        sse=float(sse),
        r_squared=float(r_squared),
        degenerate=False,
    )


def fit_ols_many(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit many simple regressions at once.

    ``x`` and ``y`` have shape ``(repetitions, sample_size)``. Each row is
    one sample. The fit is vectorized across rows: there is no Python loop
    over observations.

    Returns ``alpha`` and ``beta`` arrays of length ``repetitions``.
    Degenerate rows are NaN.
    """
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if x_values.ndim != 2 or y_values.shape != x_values.shape:
        raise ValueError("x and y must be two-dimensional arrays of the same shape")

    repetitions, sample_size = x_values.shape
    alpha = np.full(repetitions, np.nan, dtype=np.float64)
    beta = np.full(repetitions, np.nan, dtype=np.float64)
    if sample_size < 2 or repetitions == 0:
        return alpha, beta

    x_mean = x_values.mean(axis=1)
    y_mean = y_values.mean(axis=1)
    x_centered = x_values - x_mean[:, None]
    y_centered = y_values - y_mean[:, None]
    sxx = np.einsum("ij,ij->i", x_centered, x_centered)
    sxy = np.einsum("ij,ij->i", x_centered, y_centered)
    valid = np.isfinite(sxx) & np.isfinite(sxy) & (sxx > _VARIANCE_TOLERANCE)
    np.divide(sxy, sxx, out=beta, where=valid)
    alpha_values = y_mean - beta * x_mean
    alpha[valid] = alpha_values[valid]
    return alpha, beta


def residuals(x: np.ndarray, y: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Observed outcome minus the value on the fitted line."""
    x_values = _as_vector(x, "x")
    y_values = _as_vector(y, "y")
    if x_values.shape != y_values.shape:
        raise ValueError("x and y must have the same length")
    if not np.isfinite(alpha) or not np.isfinite(beta):
        return np.full(x_values.shape, np.nan, dtype=np.float64)
    return y_values - (alpha + beta * x_values)
