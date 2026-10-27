"""Correlation & Causation lessons — association, prediction, confounders."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.charts import BLUE, FONT, GRID, INK, ORANGE
from src.fitting_line import pearson_r, sample_sd, standardised_slope
from src.regression import fit_ols
from src.stat_tips import tipify

N_CORR = 120
N_SCENE = 100
DEFAULT_SEED = 20261018
PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

LESSON_KEYS = ("understanding", "causation", "control")
LESSON_LABELS = {
    "understanding": "Understanding correlation",
    "causation": "Correlation ≠ causation",
    "control": "Controlling for a third variable",
}


@dataclass(frozen=True)
class MultiFit:
    """OLS with intercept: y ~ 1 + x1 + …"""

    intercept: float
    coefs: tuple[float, ...]
    names: tuple[str, ...]
    r_squared: float
    sse: float
    n: int


def correlated_pair(
    n: int,
    r: float,
    *,
    seed: int,
    x_mean: float = 5.0,
    y_mean: float = 50.0,
    x_sd: float = 2.0,
    y_sd: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (x, y) with approximately the target Pearson correlation r."""
    rng = np.random.default_rng(int(seed))
    n = int(max(n, 3))
    target = float(np.clip(r, -1.0, 1.0))
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    # Exact construction on standard normals, then affine transform.
    y_z = target * z1 + float(np.sqrt(max(0.0, 1.0 - target**2))) * z2
    x = x_mean + x_sd * z1
    y = y_mean + y_sd * y_z
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def ice_cream_scene(
    n: int,
    *,
    seed: int,
    temp_shift: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Illustrative confounder scene: temperature drives ice cream and drownings.

    Causal structure (no direct ice-cream → drowning path):
        temperature → ice cream sales
        temperature → drownings
    """
    rng = np.random.default_rng(int(seed))
    n = int(max(n, 5))
    # Temperature in °C, shifted by the interactive control.
    temperature = rng.normal(24.0 + float(temp_shift), 4.5, size=n)
    temperature = np.clip(temperature, 12.0, 38.0)
    ice_cream = 40.0 + 3.2 * temperature + rng.normal(0.0, 6.0, size=n)
    ice_cream = np.clip(ice_cream, 20.0, 200.0)
    drownings = 1.5 + 0.35 * temperature + rng.normal(0.0, 1.1, size=n)
    drownings = np.clip(drownings, 0.2, None)
    return (
        np.asarray(temperature, dtype=np.float64),
        np.asarray(ice_cream, dtype=np.float64),
        np.asarray(drownings, dtype=np.float64),
    )


def fit_multiple(
    y: np.ndarray,
    predictors: dict[str, np.ndarray],
) -> MultiFit:
    """OLS with intercept via least squares; predictors in insertion order."""
    yv = np.asarray(y, dtype=np.float64)
    names = tuple(predictors.keys())
    cols = [np.ones(yv.size, dtype=np.float64)]
    for name in names:
        cols.append(np.asarray(predictors[name], dtype=np.float64))
    x = np.column_stack(cols)
    coef, _, _, _ = np.linalg.lstsq(x, yv, rcond=None)
    fitted = x @ coef
    resid = yv - fitted
    sse = float(np.dot(resid, resid))
    sst = float(np.sum((yv - float(np.mean(yv))) ** 2))
    r2 = float("nan") if sst <= 0 else float(1.0 - sse / sst)
    return MultiFit(
        intercept=float(coef[0]),
        coefs=tuple(float(c) for c in coef[1:]),
        names=names,
        r_squared=r2,
        sse=sse,
        n=int(yv.size),
    )


def correlation_scatter(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float,
    beta: float,
    x_title: str,
    y_title: str,
    title: str,
    height: int = 420,
    color: str = BLUE,
    color_by: np.ndarray | None = None,
    color_title: str | None = None,
    show_line: bool = True,
) -> go.Figure:
    fig = go.Figure()
    marker: dict
    if color_by is not None:
        marker = dict(
            size=9,
            color=color_by,
            colorscale="RdYlBu_r",
            showscale=True,
            colorbar=dict(title=color_title or "", thickness=14, len=0.7),
            line=dict(width=0.4, color="white"),
        )
    else:
        marker = dict(size=9, color=color, opacity=0.75, line=dict(width=0.4, color="white"))
    fig.add_scatter(
        x=x,
        y=y,
        mode="markers",
        name="Illustrative data",
        marker=marker,
        hovertemplate=f"{x_title}: %{{x:.2f}}<br>{y_title}: %{{y:.2f}}<extra></extra>",
    )
    if show_line:
        x_line = np.array([float(np.min(x)), float(np.max(x))])
        fig.add_scatter(
            x=x_line,
            y=float(alpha) + float(beta) * x_line,
            mode="lines",
            name="OLS line",
            line=dict(color=ORANGE if color_by is not None else BLUE, width=3.2),
            hoverinfo="skip",
        )
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK, family=FONT)),
        height=height,
        margin=dict(l=48, r=18, t=42, b=48),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family=FONT, color=INK, size=12),
        showlegend=False,
        xaxis=dict(title=x_title, gridcolor=GRID, zeroline=False),
        yaxis=dict(title=y_title, gridcolor=GRID, zeroline=False),
    )
    return fig


def _residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Residuals of y after removing its linear association with x."""
    fit = fit_ols(x, y)
    return np.asarray(y, dtype=np.float64) - (
        float(fit.alpha) + float(fit.beta) * np.asarray(x, dtype=np.float64)
    )


def _fmt(v: float, digits: int = 3) -> str:
    if not np.isfinite(v):
        return "—"
    return f"{float(v):.{digits}f}"


def _fmt_pct(v: float) -> str:
    if not np.isfinite(v):
        return "—"
    return f"{100.0 * float(v):.1f}%"


def _corr_interpretation(r_hat: float) -> str:
    """Concise, dynamic read of the observed (not target) sample correlation."""
    if not np.isfinite(r_hat):
        return "—"
    a = abs(float(r_hat))
    if a < 0.2:
        return (
            "Little or no linear relationship — a non-linear relationship "
            "may still exist."
        )
    direction = "positive" if r_hat > 0 else "negative"
    strength = "Strong" if a >= 0.7 else "Moderate" if a >= 0.4 else "Weak"
    return f"{strength} {direction} linear relationship."


def render_correlation_causation_tab() -> None:
    """Top-level Correlation & Causation lab."""
    st.markdown(
        tipify('<p class="fit-r2-heading">Correlation &amp; Causation</p>'),
        unsafe_allow_html=True,
    )
    st.caption("Illustrative simulations — not real observational data.")

    if "corr_lesson" not in st.session_state:
        st.session_state.corr_lesson = "understanding"
    lesson = st.segmented_control(
        "Lesson",
        options=list(LESSON_KEYS),
        format_func=lambda k: LESSON_LABELS[k],
        key="corr_lesson",
        label_visibility="collapsed",
        width="content",
    )
    if lesson not in LESSON_KEYS:
        lesson = "understanding"

    if lesson == "understanding":
        _render_understanding()
    elif lesson == "causation":
        _render_causation()
    else:
        _render_control()


def _render_understanding() -> None:
    st.markdown(
        tipify(
            "<p style='margin:0.2rem 0 0.15rem 0;color:#475569;font-weight:400;"
            "font-size:0.9rem;line-height:1.35rem;'>"
            "Correlation measures the direction and strength of a linear "
            "relationship between two variables.</p>"
            "<p class='mr-note' style='margin:0 0 0.35rem 0;'>"
            "Move r from −1 to +1. Watch how the line changes direction and "
            "how tightly the points cluster around it.</p>"
        ),
        unsafe_allow_html=True,
    )
    r_target = st.slider(
        "Target Pearson correlation r",
        min_value=-1.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
        key="corr_r_slider",
    )
    # Resample when r changes (seed folds in rounded r for stability).
    seed = DEFAULT_SEED + int(round(float(r_target) * 1000))
    x, y = correlated_pair(N_CORR, float(r_target), seed=seed)
    fit = fit_ols(x, y)
    r_hat = pearson_r(x, y)
    r2 = float(fit.r_squared) if not fit.degenerate else float("nan")
    beta_star = standardised_slope(float(fit.beta), sample_sd(x), sample_sd(y))

    left, right = st.columns([1.1, 0.9], gap="medium")
    with left:
        st.plotly_chart(
            correlation_scatter(
                x,
                y,
                alpha=float(fit.alpha),
                beta=float(fit.beta),
                x_title="X (illustrative)",
                y_title="Y (illustrative)",
                title="Correlation and regression line",
                height=440,
            ),
            width="stretch",
            config=PLOTLY_CONFIG,
            key="corr_r_scatter",
        )
        st.markdown(
            tipify(
                '<p class="fit-r2-muted-note">'
                f"Target r = <b>{r_target:+.2f}</b> · Observed r = "
                f"<b>{_fmt(r_hat)}</b></p>"
                '<p class="fit-r2-muted-note">Random samples may differ '
                "slightly from the target correlation.</p>"
            ),
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            tipify(
                '<p style="color:#334155;font-weight:650;font-size:0.86rem;'
                f'margin:0 0 0.35rem 0;">{_corr_interpretation(r_hat)}</p>'
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            tipify(
                '<table class="fit-r2-summary-table">'
                "<thead><tr><th>Statistic</th><th>Value</th></tr></thead>"
                "<tbody>"
                f"<tr><td>Pearson correlation r</td><td>{_fmt(r_hat)}</td></tr>"
                f"<tr><td>Standardised slope β*</td><td>{_fmt(beta_star)}</td></tr>"
                f"<tr><td>R²</td><td>{_fmt_pct(r2)}</td></tr>"
                "</tbody></table>"
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="fit-std-col-lab" style="margin-top:0.4rem;">'
            "Connection to regression</p>",
            unsafe_allow_html=True,
        )
        _l, mid, _r = st.columns([1, 2, 1])
        with mid:
            st.latex(r"\beta^{*} = r \qquad R^2 = r^2")
        with st.expander("Connections to regression", expanded=False):
            st.latex(r"\hat{\beta} = r \times \dfrac{\mathrm{SD}(Y)}{\mathrm{SD}(X)}")
            st.markdown(
                tipify(
                    "<p style='color:#334155;font-weight:400;font-size:0.84rem;"
                    "line-height:1.3rem;'>"
                    "Standardising X and Y rescales each by its own SD, so "
                    "SD(X) and SD(Y) cancel out of the slope formula above — "
                    "leaving β* = r. This is why the standardised slope "
                    "and Pearson's r are the same number.</p>"
                ),
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="corr-key-insight">'
        '<p class="title">KEY INSIGHT — Understanding correlation</p>'
        '<p class="lede">The sign of r tells us the direction. The '
        "magnitude |r| tells us the strength.</p>"
        '<div class="row">'
        "<span><b>r &gt; 0</b> → Positive linear relationship</span>"
        "<span><b>r &lt; 0</b> → Negative linear relationship</span>"
        "<span><b>r ≈ 0</b> → Little or no linear relationship</span>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def _toggle_corr_reveal() -> None:
    st.session_state.corr_hidden_revealed = True


def _render_causation() -> None:
    st.markdown(
        tipify(
            "<p style='margin:0.2rem 0 0.35rem 0;color:#475569;font-weight:400;"
            "font-size:0.9rem;line-height:1.35rem;'>"
            "Ice cream sales and drownings rise together. Does selling ice "
            "cream cause drownings?</p>"
        ),
        unsafe_allow_html=True,
    )
    revealed = bool(st.session_state.get("corr_hidden_revealed", False))

    # Fixed draw — identical before and after reveal, so clicking Reveal
    # never regenerates the observations.
    temp, ice, drown = ice_cream_scene(N_SCENE, seed=DEFAULT_SEED + 17, temp_shift=0.0)
    fit = fit_ols(ice, drown)
    r_hat = pearson_r(ice, drown)
    r2 = float(fit.r_squared) if not fit.degenerate else float("nan")

    if not revealed:
        st.plotly_chart(
            correlation_scatter(
                ice,
                drown,
                alpha=float(fit.alpha),
                beta=float(fit.beta),
                x_title="Ice cream sales (illustrative)",
                y_title="Drownings (illustrative)",
                title="Ice cream sales vs drownings",
                height=440,
            ),
            width="stretch",
            config=PLOTLY_CONFIG,
            key="corr_ice_drown",
        )
        st.markdown(
            tipify(
                '<p class="fit-std-slope">'
                f"r = <b>{_fmt(r_hat)}</b> &nbsp;·&nbsp; R² = <b>{_fmt(r2)}</b>"
                "</p>"
            ),
            unsafe_allow_html=True,
        )
        st.button(
            "Reveal the hidden variable",
            key="corr_reveal_btn",
            type="primary",
            on_click=_toggle_corr_reveal,
        )
        return

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.plotly_chart(
            correlation_scatter(
                ice,
                drown,
                alpha=float(fit.alpha),
                beta=float(fit.beta),
                x_title="Ice cream sales (illustrative)",
                y_title="Drownings (illustrative)",
                title="Ice cream sales vs drownings",
                height=380,
            ),
            width="stretch",
            config=PLOTLY_CONFIG,
            key="corr_ice_drown",
        )
        st.markdown(
            tipify(
                '<p class="fit-std-slope">'
                f"r = <b>{_fmt(r_hat)}</b> &nbsp;·&nbsp; R² = <b>{_fmt(r2)}</b>"
                "</p>"
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.plotly_chart(
            correlation_scatter(
                ice,
                drown,
                alpha=float(fit.alpha),
                beta=float(fit.beta),
                x_title="Ice cream sales",
                y_title="Drownings",
                title="Same points, coloured by temperature",
                height=380,
                color_by=temp,
                color_title="Temp °C",
                show_line=False,
            ),
            width="stretch",
            config=PLOTLY_CONFIG,
            key="corr_ice_drown_temp",
        )

    st.markdown(
        tipify(
            "<p style='margin:0.5rem 0 0.3rem 0;color:#334155;font-weight:400;"
            "font-size:0.88rem;line-height:1.3rem;'>Hotter weather increases "
            "both ice cream sales and swimming activity — and more swimming "
            "exposure can lead to more drownings.</p>"
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="fit-r2-lesson" style="padding:0.5rem 0.7rem;">'
        '<div class="result" style="margin:0;">'
        '<p class="big">Correlation does not imply causation.</p>'
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="corr-causal-diagram">'
        '<div class="node wide">Temperature</div>'
        '<div class="arrow">↙</div><div class="arrow">↘</div>'
        '<div class="node">Ice cream sales</div>'
        '<div class="node">Swimming activity</div>'
        "<div></div><div class=\"arrow\">↓</div>"
        '<div></div><div class="node">Drownings</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def _render_control() -> None:
    st.markdown(
        tipify(
            "<p style='margin:0.2rem 0 0.35rem 0;color:#475569;font-weight:400;"
            "font-size:0.9rem;line-height:1.35rem;'>"
            "Same illustrative world: temperature causes both ice cream sales and "
            "drownings; ice cream has no direct causal effect on drownings. "
            "Compare the simple regression with the association that remains "
            "after removing temperature's influence from both variables.</p>"
        ),
        unsafe_allow_html=True,
    )
    temp, ice, drown = ice_cream_scene(N_SCENE, seed=DEFAULT_SEED + 41, temp_shift=0.0)
    model_a = fit_ols(ice, drown)
    model_b = fit_multiple(drown, {"ice_cream": ice, "temperature": temp})
    r2_a = float(model_a.r_squared) if not model_a.degenerate else float("nan")
    beta_ice_a = float(model_a.beta)
    # ice cream is first named predictor in model B
    beta_ice_b = float(model_b.coefs[0])
    beta_temp_b = float(model_b.coefs[1])

    # Partial-regression (added-variable) plot: residualise both ice cream
    # and drownings against temperature, then fit OLS on the residuals.
    # By the Frisch–Waugh–Lovell theorem its slope equals beta_ice_b exactly.
    ice_resid = _residualize(ice, temp)
    drown_resid = _residualize(drown, temp)
    partial_fit = fit_ols(ice_resid, drown_resid)

    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown(
            '<p class="fit-std-col-lab">Simple regression</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            correlation_scatter(
                ice,
                drown,
                alpha=float(model_a.alpha),
                beta=float(model_a.beta),
                x_title="Ice cream sales",
                y_title="Drownings",
                title="Drownings ~ Ice cream",
                height=340,
            ),
            width="stretch",
            config=PLOTLY_CONFIG,
            key="corr_model_a",
        )
        st.markdown(
            tipify(
                '<p class="fit-std-slope">'
                "Drownings ~ Ice cream sales<br>"
                f"β̂<sub>ice</sub> = <b>{_fmt(beta_ice_a)}</b><br>"
                f"R² = <b>{_fmt(r2_a)}</b>"
                "</p>"
            ),
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            '<p class="fit-std-col-lab">Partial regression — temperature removed</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            correlation_scatter(
                ice_resid,
                drown_resid,
                alpha=float(partial_fit.alpha),
                beta=float(partial_fit.beta),
                x_title="Ice cream sales (residualised against temperature)",
                y_title="Drownings (residualised against temperature)",
                title="Residuals after removing temperature",
                height=340,
            ),
            width="stretch",
            config=PLOTLY_CONFIG,
            key="corr_model_b",
        )
        st.markdown(
            tipify(
                '<p class="fit-std-slope">'
                "Drownings ~ Ice cream + Temperature<br>"
                f"β̂<sub>ice</sub> = <b>{_fmt(beta_ice_b)}</b> "
                f"<span class='fit-std-same'>(= slope of the partial-regression line)</span><br>"
                f"β̂<sub>temp</sub> = <b>{_fmt(beta_temp_b)}</b><br>"
                f"R² = <b>{_fmt(model_b.r_squared)}</b>"
                "</p>"
            ),
            unsafe_allow_html=True,
        )

    shrink = abs(beta_ice_a) - abs(beta_ice_b)
    st.markdown(
        tipify(
            '<div class="fit-r2-lesson" style="padding:0.55rem 0.7rem;">'
            "<p style='font-weight:400;color:#334155;'>The apparent positive "
            "association largely disappears after controlling for "
            "temperature: the ice-cream coefficient moves from "
            f"<b>{_fmt(beta_ice_a)}</b> to <b>{_fmt(beta_ice_b)}</b>"
            + (
                f" (shrinks by about {_fmt(shrink)})."
                if shrink > 0.02
                else "."
            )
            + " This is not exactly zero, and controlling for a measured "
            "confounder does not by itself prove causation.</p>"
            '<p class="fit-std-note" style="margin-bottom:0;">The '
            "multiple-regression slope on ice cream is the association "
            "<em>holding temperature constant</em> — that is what "
            "\"controlling for\" a predictor means, and it is exactly what "
            "the partial-regression plot above shows.</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
