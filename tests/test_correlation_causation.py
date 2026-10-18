"""Mathematical checks for the Correlation & Causation lab."""

import numpy as np

from src.correlation_causation import (
    correlated_pair,
    fit_multiple,
    ice_cream_scene,
)
from src.fitting_line import pearson_r, sample_sd, standardised_slope
from src.regression import fit_ols


def test_correlated_pair_matches_target_r():
    for target in (-1.0, -0.5, 0.0, 0.5, 1.0):
        x, y = correlated_pair(200, target, seed=42)
        r = pearson_r(x, y)
        assert abs(r - target) < 0.12


def test_r_squared_equals_r_squared_in_simple_ols():
    x, y = correlated_pair(150, 0.8, seed=7)
    fit = fit_ols(x, y)
    r = pearson_r(x, y)
    assert not fit.degenerate
    assert abs(fit.r_squared - r**2) < 1e-9


def test_standardised_slope_equals_pearson_r():
    x, y = correlated_pair(180, -0.65, seed=11)
    fit = fit_ols(x, y)
    beta_star = standardised_slope(fit.beta, sample_sd(x), sample_sd(y))
    r = pearson_r(x, y)
    assert abs(beta_star - r) < 1e-9


def test_ice_cream_scene_has_positive_spurious_correlation():
    temp, ice, drown = ice_cream_scene(120, seed=99, temp_shift=0.0)
    r = pearson_r(ice, drown)
    assert r > 0.45
    # Temperature is associated with both outcomes.
    assert pearson_r(temp, ice) > 0.7
    assert pearson_r(temp, drown) > 0.7


def test_controlling_for_temperature_shrinks_ice_cream_coef():
    temp, ice, drown = ice_cream_scene(200, seed=20261018, temp_shift=0.0)
    model_a = fit_ols(ice, drown)
    model_b = fit_multiple(drown, {"ice_cream": ice, "temperature": temp})
    assert not model_a.degenerate
    beta_a = abs(model_a.beta)
    beta_b = abs(model_b.coefs[0])
    # Apparent ice-cream effect shrinks once temperature is held fixed.
    assert beta_b < beta_a * 0.5
    # Temperature retains a clear positive association with drownings.
    assert model_b.coefs[1] > 0.15


def test_temperature_shift_moves_both_means():
    cool = ice_cream_scene(80, seed=5, temp_shift=-6.0)
    warm = ice_cream_scene(80, seed=5, temp_shift=6.0)
    assert float(np.mean(warm[0])) > float(np.mean(cool[0]))
    assert float(np.mean(warm[1])) > float(np.mean(cool[1]))
    assert float(np.mean(warm[2])) > float(np.mean(cool[2]))


def test_main_lab_tabs_order_includes_correlation():
    from pathlib import Path

    text = Path("app.py").read_text(encoding="utf-8")
    start = text.index("MAIN_LAB_TABS")
    block = text[start : start + 280]
    assert '("fitting", "Simple Linear Regression")' in block
    assert '("correlation", "Correlation & Causation")' in block
    assert block.index("fitting") < block.index("correlation") < block.index("sampling")
    assert "render_correlation_causation_tab" in text
