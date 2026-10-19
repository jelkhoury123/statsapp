"""The F-test lesson: is a nested model's improvement statistically significant?

Phase 1: only the "Why an F-test?" subtab is implemented. The remaining
three (Building the F-statistic, The F-distribution, Interpreting the
Result) are minimal placeholders for a later phase.

Reuses the Multiple Regression lesson's sample, fits and Model Fit
scenarios rather than duplicating them.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import f as fisher_f
from scipy.stats import t as student_t

from src.charts import BLUE, FONT, GREY, INK, ORANGE
from src.correlation_causation import MultiFit, fit_multiple
from src.multiple_regression import mr_sample
from src.population import Population
from src.sampling import Sample

_ALPHA_OPTIONS: tuple[float, ...] = (0.01, 0.05, 0.10)

PLOTLY_CONFIG = {"displaylogo": False, "responsive": True}


def _format_coef(value: float, decimals: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{decimals}f}"


FT_SUBTABS: tuple[tuple[str, str], ...] = (
    ("why", "Why an F-test?"),
    ("build", "Building the F-statistic"),
    ("dist", "The F-distribution"),
    ("dof", "Degrees of Freedom"),
    ("interpret", "Interpreting the Result"),
)
FT_SUBTAB_LABELS = dict(FT_SUBTABS)

# Three nested-model comparisons, all fit to the same observations and the
# same illustrative scores (see _f_test_scores): a single added predictor
# (where F = t²), a joint test of two added predictors, and an overall test
# of every slope at once.
_FT_SCENARIOS: tuple[tuple[str, str], ...] = (
    ("one", "A. One additional predictor (F = t²)"),
    ("joint", "B. Several additional predictors (Joint F-test)"),
    ("overall", "C. All predictors (Overall F-test)"),
)
FT_SCENARIO_LABELS = dict(_FT_SCENARIOS)

_FT_SCENARIO_PREDICTORS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "one": (("hours",), ("hours", "attendance")),
    "joint": (("hours",), ("hours", "attendance", "sleep")),
    "overall": ((), ("hours", "attendance", "sleep")),
}

_FT_SCENARIO_H0: dict[str, str] = {
    "one": "β<sub>attendance</sub> = 0",
    "joint": "β<sub>attendance</sub> = β<sub>sleep</sub> = 0",
    "overall": (
        "β<sub>hours</sub> = β<sub>attendance</sub> = β<sub>sleep</sub> = 0"
    ),
}
_FT_SCENARIO_H1: dict[str, str] = {
    "one": "β<sub>attendance</sub> ≠ 0",
    "joint": "at least one of β<sub>attendance</sub>, β<sub>sleep</sub> ≠ 0",
    "overall": (
        "at least one of β<sub>hours</sub>, β<sub>attendance</sub>, "
        "β<sub>sleep</sub> ≠ 0"
    ),
}

_PREDICTOR_LABELS: dict[str, str] = {
    "hours": "Hours", "attendance": "Attendance", "sleep": "Sleep",
}
_PREDICTOR_SYMBOLS: dict[str, str] = {
    "hours": "X₁", "attendance": "X₂", "sleep": "X₃",
}


def _predictor_word(q: int) -> str:
    return "predictor" if q == 1 else "predictors"


def _tested_names(scenario: str) -> tuple[str, ...]:
    restricted_names, full_names = _FT_SCENARIO_PREDICTORS[scenario]
    return tuple(n for n in full_names if n not in restricted_names)


def _added_predictors_phrase(names: tuple[str, ...]) -> str:
    labels = [_PREDICTOR_LABELS[n].lower() for n in names]
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def _model_label(names: tuple[str, ...]) -> str:
    if not names:
        return "Intercept only"
    labels = [_PREDICTOR_LABELS[n] for n in names]
    if len(labels) == 1:
        return f"{labels[0]} only"
    return " + ".join(labels)


def _model_equation(fit: MultiFit) -> str:
    parts = [_format_coef(fit.intercept)]
    for name, coef in zip(fit.names, fit.coefs):
        parts.append(f"{_format_coef(coef)}·{_PREDICTOR_SYMBOLS[name]}")
    return "ŷ = " + " + ".join(parts)


def _f_test_scores(sample: Sample, seed: int = 777) -> np.ndarray:
    """Illustrative scores where hours, attendance and sleep each have a
    genuine effect, so every comparison scenario below has real signal to
    detect (same role as Multiple Regression's "strong" scenario, extended
    with a sleep effect for the joint/overall F-tests)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 4.0, size=sample.n)
    return (
        10.0 + 3.0 * sample.hours_studied + 0.55 * sample.attendance
        + 1.0 * sample.sleep_hours + noise
    )


def _predictor_arrays(sample: Sample) -> dict[str, np.ndarray]:
    return {
        "hours": sample.hours_studied,
        "attendance": sample.attendance,
        "sleep": sample.sleep_hours,
    }


def _fit_nested_models(sample: Sample, scenario: str) -> tuple[MultiFit, MultiFit]:
    """Restricted vs full model for this scenario, same observations."""
    y = _f_test_scores(sample)
    arrays = _predictor_arrays(sample)
    restricted_names, full_names = _FT_SCENARIO_PREDICTORS[scenario]
    restricted_fit = fit_multiple(y, {n: arrays[n] for n in restricted_names})
    full_fit = fit_multiple(y, {n: arrays[n] for n in full_names})
    return restricted_fit, full_fit


def _coefficient_t_stats(
    sample: Sample, full_names: tuple[str, ...],
) -> dict[str, tuple[float, int, float]]:
    """t-statistic, df and two-sided p-value for each coefficient in the full model."""
    y = _f_test_scores(sample)
    arrays = _predictor_arrays(sample)
    n = sample.n
    x = np.column_stack([np.ones(n)] + [arrays[name] for name in full_names])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    df_residual = int(n - x.shape[1])
    sigma2 = float(resid @ resid) / df_residual
    xtx_inv = np.linalg.inv(x.T @ x)
    result: dict[str, tuple[float, int, float]] = {}
    for i, name in enumerate(full_names, start=1):
        se = float(np.sqrt(sigma2 * xtx_inv[i, i]))
        t_stat = float(beta[i] / se)
        p_value = 2.0 * float(student_t.sf(abs(t_stat), df_residual))
        result[name] = (t_stat, df_residual, p_value)
    return result


def _render_why(sample: Sample, scenario: str) -> None:
    restricted_names, full_names = _FT_SCENARIO_PREDICTORS[scenario]
    added_phrase = _added_predictors_phrase(_tested_names(scenario))

    # Not wrapped in tipify(): its "SSE"→"SSR" legacy alias would silently
    # relabel every literal "SSE" in this text to "SSR" when rendered.
    st.markdown(
        f'<p class="mr-sub">Does adding {added_phrase} genuinely improve '
        "our model, or does it only look better by chance?</p>"
        '<p class="mr-note">Adding predictors can only reduce — or at '
        "worst preserve — SSE on the same observations. So some "
        "improvement is <b>guaranteed</b>, even when they have no real "
        "relationship with exam scores. An <b>F-test</b> tells us whether "
        "this improvement is more than that.</p>",
        unsafe_allow_html=True,
    )
    if scenario == "overall":
        st.markdown(
            '<p class="mr-note">Here Model 1 has no predictors at all — it '
            "just predicts every student's score as the <b>sample mean</b>. "
            "The overall F-test asks whether the full model beats that "
            "no-information baseline.</p>",
            unsafe_allow_html=True,
        )

    restricted_fit, full_fit = _fit_nested_models(sample, scenario)

    sse_reduction = restricted_fit.sse - full_fit.sse
    sse_reduction_pct = (
        sse_reduction / restricted_fit.sse * 100.0 if restricted_fit.sse > 0 else 0.0
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            '<div class="mr-model-card-lg">'
            f'<p class="mr-model-name">Model 1 — {_model_label(restricted_names)}</p>'
            f'<p class="mr-model-eq">{_model_equation(restricted_fit)}</p>'
            '<div class="mr-metric-row">'
            '<div class="mr-metric-tile"><p class="mr-metric-label">SSE</p>'
            f'<p class="mr-metric-value">{restricted_fit.sse:,.0f}</p></div>'
            '<div class="mr-metric-tile"><p class="mr-metric-label">R²</p>'
            f'<p class="mr-metric-value">{restricted_fit.r_squared:.3f}</p></div>'
            "</div></div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            '<div class="mr-model-card-lg">'
            f'<p class="mr-model-name">Model 2 — {_model_label(full_names)}</p>'
            f'<p class="mr-model-eq">{_model_equation(full_fit)}</p>'
            '<div class="mr-metric-row">'
            '<div class="mr-metric-tile accent"><p class="mr-metric-label">SSE</p>'
            f'<p class="mr-metric-value">{full_fit.sse:,.0f}</p></div>'
            '<div class="mr-metric-tile accent"><p class="mr-metric-label">R²</p>'
            f'<p class="mr-metric-value">{full_fit.r_squared:.3f}</p></div>'
            "</div></div>",
            unsafe_allow_html=True,
        )

    max_sse = max(restricted_fit.sse, full_fit.sse, 1e-9)
    w1 = restricted_fit.sse / max_sse * 100.0
    w2 = full_fit.sse / max_sse * 100.0
    st.markdown(
        '<div class="mr-sse-compare">'
        f'<p class="mr-sse-compare-label">SSE reduction: {sse_reduction:,.0f} '
        f"(−{sse_reduction_pct:.1f}%)</p>"
        '<div class="mr-sse-row"><span class="mr-sse-tag">Model 1</span>'
        f'<div class="mr-sse-track"><div class="mr-sse-fill m1" style="width:{w1:.1f}%"></div></div>'
        f'<span class="mr-sse-val">{restricted_fit.sse:,.0f}</span></div>'
        '<div class="mr-sse-row"><span class="mr-sse-tag">Model 2</span>'
        f'<div class="mr-sse-track"><div class="mr-sse-fill m2" style="width:{w2:.1f}%"></div></div>'
        f'<span class="mr-sse-val">{full_fit.sse:,.0f}</span></div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="mr-insight-banner">Better fit is guaranteed. '
        "Statistical significance isn't.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="mr-sub">How large is this improvement compared with the '
        "noise still left in the model? Let's build the F-statistic.</p>",
        unsafe_allow_html=True,
    )


def _render_build(sample: Sample, scenario: str) -> None:
    st.markdown(
        '<p class="mr-sub">How large is our improvement compared with the '
        "noise still left in the model?</p>",
        unsafe_allow_html=True,
    )

    restricted_fit, full_fit = _fit_nested_models(sample, scenario)
    restricted_names, full_names = _FT_SCENARIO_PREDICTORS[scenario]

    n = int(sample.n)
    q = len(full_names) - len(restricted_names)  # predictors added in this scenario
    p_full = len(full_names)  # predictors in the full model
    df_residual = n - p_full - 1

    sse_restricted = restricted_fit.sse
    sse_full = full_fit.sse
    sse_reduction = sse_restricted - sse_full
    numerator = sse_reduction / q
    denominator = sse_full / df_residual
    f_stat = numerator / denominator

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            '<div class="mr-panel-lg">'
            '<p class="mr-panel-title">Numerator — improvement</p>'
            '<p class="mr-panel-sub">Improvement per added predictor, '
            f"<b>q = {q}</b>.</p>"
            f'<p class="mr-panel-calc">SSE₁ − SSE₂ = {sse_restricted:,.0f} − '
            f"{sse_full:,.0f} = {sse_reduction:,.0f}</p>"
            f'<p class="mr-panel-calc">÷ q = {sse_reduction:,.0f} / {q}</p>'
            f'<p class="mr-panel-result">{numerator:,.1f}</p>'
            "</div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            '<div class="mr-panel-lg">'
            '<p class="mr-panel-title">Denominator — remaining noise</p>'
            "<p class=\"mr-panel-sub\">Noise per residual degree of freedom, "
            f"<b>df = n − p − 1 = {n} − {p_full} − 1 = {df_residual}</b>.</p>"
            f'<p class="mr-panel-calc">SSE₂ = {sse_full:,.0f}</p>'
            f'<p class="mr-panel-calc">÷ df = {sse_full:,.0f} / {df_residual}</p>'
            f'<p class="mr-panel-result">{denominator:,.1f}</p>'
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<p class="mr-panel-arrow">↓ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓</p>', unsafe_allow_html=True)
    st.markdown(
        '<div style="display:flex;align-items:center;justify-content:center;'
        'gap:0.7rem;margin:0.2rem 0 0.8rem 0;font-size:1.3rem;color:#12263a;">'
        '<span style="font-weight:700;">F</span><span>=</span>'
        '<div style="display:inline-flex;flex-direction:column;align-items:center;'
        'line-height:1.6;font-size:1.1rem;">'
        f'<span style="padding:0 0.4rem;">{numerator:,.1f}</span>'
        '<span style="border-top:2px solid #374151;width:100%;"></span>'
        f'<span style="padding:0 0.4rem;">{denominator:,.1f}</span>'
        "</div>"
        '<span>=</span>'
        f'<span style="font-weight:800;color:#EA580C;font-size:1.6rem;">{f_stat:.3f}</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="mr-note">A larger F means greater improvement relative to '
        "the remaining unexplained variation.</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<p class="mr-coef-card">df1 = {q}, df2 = {df_residual} (as above). '
        "Under H₀, both the numerator and denominator estimate the same "
        "noise variance, so F is typically around 1 — much larger values "
        "provide evidence against H₀.</p>",
        unsafe_allow_html=True,
    )


def _compute_f_statistic(sample: Sample, scenario: str) -> tuple[float, int, int]:
    """F-statistic and (df1, df2) for the restricted vs full model in this scenario."""
    restricted_fit, full_fit = _fit_nested_models(sample, scenario)
    restricted_names, full_names = _FT_SCENARIO_PREDICTORS[scenario]
    n = int(sample.n)
    q = len(full_names) - len(restricted_names)
    p_full = len(full_names)
    df_residual = n - p_full - 1
    numerator = (restricted_fit.sse - full_fit.sse) / q
    denominator = full_fit.sse / df_residual
    return numerator / denominator, q, df_residual


def _format_p_value(p: float) -> str:
    if p < 0.0001:
        return f"{p:.3e}"
    return f"{p:.4f}"


_DIST_X_MAX = 10.0


def _f_distribution_figure(
    *, df1: int, df2: int, f_stat: float, critical: float, height: int = 420,
    p_value: float | None = None,
) -> go.Figure:
    """F-distribution under H0, rejection region, observed F and p-value area.

    F(1, df2)-style densities diverge as F → 0, which would otherwise stretch
    the y-axis so far that the body and right tail become invisible. The
    y-axis is clipped to the visible "body" of the distribution (excluding
    that near-zero spike) — the curve itself is still the exact, mathematically
    correct density; only the displayed range is capped.
    """
    grid = np.linspace(0.001, _DIST_X_MAX, 500)
    density = fisher_f.pdf(grid, df1, df2)
    body_mask = grid >= max(0.1, _DIST_X_MAX * 0.01)
    body_peak = float(density[body_mask].max()) if body_mask.any() else float(np.max(density))
    y_cap = max(body_peak * 1.25, 1e-6)
    peak = y_cap
    fig = go.Figure()

    # Rejection region beyond the critical value (alpha-based).
    if critical <= _DIST_X_MAX:
        mask = grid >= critical
        fig.add_trace(
            go.Scatter(
                x=grid[mask], y=density[mask], mode="lines", fill="tozeroy",
                line=dict(width=0), fillcolor="rgba(220,38,38,0.28)",
                hoverinfo="skip", showlegend=False,
            )
        )

    # p-value area beyond the observed F (diagonal hatch, distinct from the
    # solid rejection-region fill above).
    if 0 < f_stat <= _DIST_X_MAX:
        mask = grid >= f_stat
        fig.add_trace(
            go.Scatter(
                x=grid[mask], y=density[mask], mode="lines", fill="tozeroy",
                line=dict(width=0), fillcolor="rgba(29,78,216,0.0)",
                fillpattern=dict(shape="/", fgcolor="rgba(29,78,216,0.45)", size=6, solidity=0.25),
                hoverinfo="skip", showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=grid, y=density, mode="lines", name="F distribution",
            line=dict(color=INK, width=2.4),
            hovertemplate="F %{x:.2f}<br>Density %{y:.3f}<extra></extra>",
        )
    )

    if critical <= _DIST_X_MAX:
        fig.add_vline(x=critical, line_color=GREY, line_width=1.4, line_dash="dash")
        fig.add_annotation(
            x=critical, y=peak * 0.9, text=f"Critical F = {critical:.3f}",
            showarrow=False, xanchor="left", xshift=6,
            font=dict(size=11, color="#52606D", family=FONT),
        )

    if 0 < f_stat <= _DIST_X_MAX:
        pval_x = min(f_stat + (_DIST_X_MAX - f_stat) * 0.3, _DIST_X_MAX - 0.3)
        fig.add_annotation(
            x=pval_x, y=0, yshift=14, text="p-value",
            showarrow=False, xanchor="center",
            font=dict(size=10, color="#1D4ED8", family=FONT),
        )

    if f_stat <= _DIST_X_MAX:
        fig.add_vline(x=f_stat, line_color=BLUE, line_width=2.4)
        fig.add_annotation(
            x=f_stat, y=peak * 1.04, text=f"Observed F = {f_stat:.3f}",
            showarrow=False, xanchor="left" if f_stat < _DIST_X_MAX * 0.8 else "right",
            xshift=6 if f_stat < _DIST_X_MAX * 0.8 else -6,
            bgcolor="rgba(255,255,255,0.88)", borderpad=2,
            font=dict(size=12, color=BLUE, family=FONT),
        )
    else:
        # Off-scale: an arrow at the right edge pointing toward the real value.
        off_scale_text = f"Observed F = {f_stat:.3f} (off scale)"
        if p_value is not None:
            p_text = "< 0.0001" if p_value < 0.0001 else f"{p_value:.4f}"
            off_scale_text += f" · p = {p_text}"
        fig.add_annotation(
            x=_DIST_X_MAX, y=peak * 0.55, ax=_DIST_X_MAX * 0.82, ay=peak * 0.55,
            xref="x", yref="y", axref="x", ayref="y",
            text=off_scale_text,
            showarrow=True, arrowhead=2, arrowcolor=BLUE,
            font=dict(size=12, color=BLUE, family=FONT),
            xanchor="right",
        )

    fig.update_xaxes(title_text="F statistic", range=[0, _DIST_X_MAX])
    fig.update_yaxes(title_text="Density", range=[0, y_cap])
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=50, r=20, t=30, b=40),
        font=dict(family=FONT, color=INK, size=12),
        showlegend=False,
    )
    return fig


def _dist_p_text(p: float) -> str:
    """Like _format_p_value, but never scientific notation — always '< 0.0001'."""
    if p < 0.0001:
        return "< 0.0001"
    return f"{p:.4f}"


_DF_DEMO_DF1_OPTIONS: tuple[int, ...] = (1, 2, 5, 10)
_DF_DEMO_DF2_OPTIONS: tuple[int, ...] = (10, 30, 147, 500)
_DF_DEMO_REFERENCE = (1, 147)


def _f_distribution_compare_figure(
    df1: int, df2: int, alpha: float = 0.05, height: int = 260,
) -> go.Figure:
    """Selected F(df1, df2) vs the reference F(1, 147), for the df demo.

    Draws each distribution's own critical F at `alpha` as a matching
    dashed vertical line, so the shape change and the critical-value
    change are visible together.
    """
    ref_df1, ref_df2 = _DF_DEMO_REFERENCE
    x_max = 6.0
    grid = np.linspace(0.001, x_max, 400)
    ref_density = fisher_f.pdf(grid, ref_df1, ref_df2)
    sel_density = fisher_f.pdf(grid, df1, df2)
    body_mask = grid >= 0.1
    y_cap = max(
        float(ref_density[body_mask].max()) if body_mask.any() else float(ref_density.max()),
        float(sel_density[body_mask].max()) if body_mask.any() else float(sel_density.max()),
    ) * 1.2

    ref_critical = float(fisher_f.ppf(1.0 - alpha, ref_df1, ref_df2))
    sel_critical = float(fisher_f.ppf(1.0 - alpha, df1, df2))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=grid, y=ref_density, mode="lines",
            name=f"Reference F({ref_df1}, {ref_df2})",
            line=dict(color=GREY, width=2, dash="dash"),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grid, y=sel_density, mode="lines",
            name=f"F({df1}, {df2})",
            line=dict(color=ORANGE, width=2.6),
            hoverinfo="skip",
        )
    )

    for critical, color, label in (
        (ref_critical, GREY, f"Critical F({ref_df1},{ref_df2}) = {ref_critical:.2f}"),
        (sel_critical, ORANGE, f"Critical F({df1},{df2}) = {sel_critical:.2f}"),
    ):
        if critical <= x_max:
            fig.add_vline(x=critical, line_color=color, line_width=1.6, line_dash="dash")
            fig.add_annotation(
                x=critical, y=y_cap * (0.95 if color == GREY else 0.82),
                text=label, showarrow=False, xanchor="left", xshift=6,
                font=dict(size=10, color=color, family=FONT),
            )

    fig.update_xaxes(title_text="F statistic", range=[0, x_max])
    fig.update_yaxes(title_text="Density", range=[0, max(y_cap, 1e-6)])
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=50, r=20, t=10, b=40),
        font=dict(family=FONT, color=INK, size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _render_dist(sample: Sample, scenario: str) -> None:
    q = len(_tested_names(scenario))
    contributes = "contributes" if q == 1 else "contribute"
    st.markdown(
        f'<p class="mr-note">H₀: the added {_predictor_word(q)} '
        f"{contributes} nothing beyond the predictors already in the "
        f"model — here, H₀: {_FT_SCENARIO_H0[scenario]}.</p>"
        '<p class="mr-note">If H₀ were true, different random samples would '
        "produce different F-statistics simply from sampling noise. The "
        "F-distribution describes how those statistics would vary under "
        "H₀, assuming the regression model's assumptions hold.</p>"
        '<p class="mr-note">Unlike the t-distribution, the F-distribution is '
        "asymmetric and cannot take negative values. Its shape depends on "
        "two degrees of freedom.</p>",
        unsafe_allow_html=True,
    )

    f_stat, df1, df2 = _compute_f_statistic(sample, scenario)
    p_value = float(fisher_f.sf(f_stat, df1, df2))

    alpha = st.radio(
        "Significance level (α)", options=_ALPHA_OPTIONS, index=1,
        format_func=lambda a: f"α = {a:.2f}", key="ft_alpha", horizontal=True,
    )
    critical = float(fisher_f.ppf(1.0 - alpha, df1, df2))

    st.markdown(
        f'<p class="mr-delta-badge">Observed F = {f_stat:.3f} '
        f"&nbsp;|&nbsp; p = {_dist_p_text(p_value)}</p>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        _f_distribution_figure(
            df1=df1, df2=df2, f_stat=f_stat, critical=critical, p_value=p_value,
        ),
        width="stretch", config=PLOTLY_CONFIG, key="ft_dist_chart",
    )
    st.markdown(
        '<p class="mr-note"><span style="color:#DC2626;font-weight:700;">Red</span> '
        "= rejection region (α). "
        '<span style="color:#1D4ED8;font-weight:700;">Blue hatching</span> '
        "= p-value region (observed F). The two can overlap.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="mr-stat">df1 = q = <b>{df1}</b> &nbsp;·&nbsp; '
        f"df2 = n − p − 1 = <b>{df2}</b> &nbsp;·&nbsp; "
        f"Observed F = <b>{f_stat:.3f}</b></p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="mr-stat">df1 = {df1}, df2 = {df2} — see '
        '<b>"Degrees of Freedom"</b> for what these mean and how they '
        "shape the distribution.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="mr-note">The p-value is the probability, assuming H₀ is '
        "true, of obtaining an F-statistic at least as large as the one "
        f"observed: <b>p = {_dist_p_text(p_value)}</b>.</p>",
        unsafe_allow_html=True,
    )

    reject = p_value < alpha
    added_word = _predictor_word(q)
    provide_verb = "provides" if q == 1 else "provide"
    improve_verb = "improves" if q == 1 else "improve"
    if reject:
        conclusion = (
            f"Reject H₀. The added {added_word} {provide_verb} statistically "
            "significant explanatory power."
        )
    else:
        conclusion = (
            f"Do not reject H₀. We lack sufficient evidence that the "
            f"added {added_word} {improve_verb} the model."
        )
    st.markdown(f'<p class="mr-delta-badge">{conclusion}</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="mr-compare">Statistical significance does not establish '
        "causality or practical importance.</p>",
        unsafe_allow_html=True,
    )


def _render_dof(sample: Sample, scenario: str) -> None:
    f_stat, df1, df2 = _compute_f_statistic(sample, scenario)

    card_col1, card_col2 = st.columns(2, gap="small")
    with card_col1:
        st.markdown(
            f'<p class="mr-coef-card"><b>df1 = {df1}</b><br>'
            "One additional predictor tested (attendance).</p>",
            unsafe_allow_html=True,
        )
    with card_col2:
        st.markdown(
            f'<p class="mr-coef-card"><b>df2 = {df2}</b><br>'
            f"{int(sample.n)} observations minus 3 estimated parameters "
            "(intercept + two slopes).</p>",
            unsafe_allow_html=True,
        )

    demo_col1, demo_col2 = st.columns(2, gap="small")
    with demo_col1:
        demo_df1 = st.selectbox(
            "df1 (illustrative)", options=_DF_DEMO_DF1_OPTIONS, index=1, key="ft_demo_df1",
        )
    with demo_col2:
        demo_df2 = st.selectbox(
            "df2 (illustrative)", options=_DF_DEMO_DF2_OPTIONS, index=1, key="ft_demo_df2",
        )

    st.plotly_chart(
        _f_distribution_compare_figure(demo_df1, demo_df2, height=380),
        width="stretch", config=PLOTLY_CONFIG, key="ft_demo_dist_chart",
    )

    ref_df1, ref_df2 = _DF_DEMO_REFERENCE
    sel_critical = float(fisher_f.ppf(0.95, demo_df1, demo_df2))
    ref_critical = float(fisher_f.ppf(0.95, ref_df1, ref_df2))
    if sel_critical > ref_critical:
        comparison = "higher than"
    elif sel_critical < ref_critical:
        comparison = "lower than"
    else:
        comparison = "equal to"
    st.markdown(
        f'<p class="mr-stat">With df1 = {demo_df1}, df2 = {demo_df2}, the '
        f"critical F at α = 0.05 is <b>{sel_critical:.2f}</b> — "
        f"{comparison} the reference F(1, 147)'s <b>{ref_critical:.2f}</b>.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="mr-compare">Illustrative values only — they do not '
        "change the regression's own F-statistic, p-value or decision in "
        '"The F-distribution" tab.</p>',
        unsafe_allow_html=True,
    )


def _significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return ""


def _render_interpret(sample: Sample, scenario: str) -> None:
    restricted_names, full_names = _FT_SCENARIO_PREDICTORS[scenario]
    tested_names = _tested_names(scenario)
    q = len(tested_names)

    st.markdown(
        f'<p class="mr-note">H₀: the added {_predictor_word(q)} '
        f"{'has' if q == 1 else 'have'} no additional explanatory power "
        f"after accounting for the existing predictors — H₀: "
        f"{_FT_SCENARIO_H0[scenario]}.<br>"
        f"H₁: {_FT_SCENARIO_H1[scenario]}.</p>",
        unsafe_allow_html=True,
    )

    restricted_fit, full_fit = _fit_nested_models(sample, scenario)
    n = int(sample.n)
    df1_res = n - len(restricted_names) - 1
    df2_res = n - len(full_names) - 1
    rss1, rss2 = restricted_fit.sse, full_fit.sse
    sum_sq = rss1 - rss2
    f_stat, df1, df2 = _compute_f_statistic(sample, scenario)
    p_value = float(fisher_f.sf(f_stat, df1, df2))

    restricted_formula = "1" if not restricted_names else " + ".join(restricted_names)
    full_formula = " + ".join(full_names)

    st.code(
        f"model1 <- lm(score ~ {restricted_formula}, data = students)\n"
        f"model2 <- lm(score ~ {full_formula}, data = students)\n\n"
        "anova(model1, model2)",
        language="r",
    )

    p_str = _format_p_value(p_value)
    stars = _significance_stars(p_value)
    st.code(
        "Analysis of Variance Table\n\n"
        f"Model 1: score ~ {restricted_formula}\n"
        f"Model 2: score ~ {full_formula}\n"
        "  Res.Df      RSS Df Sum of Sq       F  Pr(>F)\n"
        f"1 {df1_res:6d} {rss1:8.2f}\n"
        f"2 {df2_res:6d} {rss2:8.2f}  {q} {sum_sq:9.2f} {f_stat:7.3f} {p_str} {stars}\n"
        "---\n"
        "Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1",
        language=None,
    )
    st.markdown(
        '<p class="mr-stat">RSS = SSE — R labels the residual sum of squares '
        '"RSS"; it is the same quantity used throughout this lesson.</p>',
        unsafe_allow_html=True,
    )

    with st.expander("What do these columns mean?"):
        st.markdown(
            '<p class="mr-stat"><b>Res.Df</b> — residual degrees of freedom '
            "left in that model.<br>"
            "<b>RSS</b> — residual sum of squares (SSE) for that model.<br>"
            "<b>Df</b> — number of predictors added between the two models (q).<br>"
            "<b>Sum of Sq</b> — reduction in RSS from adding them.<br>"
            "<b>F</b> — the F-statistic for the comparison.<br>"
            "<b>Pr(&gt;F)</b> — the p-value.</p>",
            unsafe_allow_html=True,
        )

    reject = p_value < 0.05
    decision = "Reject H₀" if reject else "Do not reject H₀"
    added_word = _predictor_word(q)
    conclusion = (
        f"the added {added_word} {'provides' if q == 1 else 'provide'} "
        "statistically significant explanatory power." if reject else
        f"we lack sufficient evidence that the added {added_word} "
        f"{'improves' if q == 1 else 'improve'} the model."
    )
    st.markdown(
        f'<p class="mr-stat">F = <b>{f_stat:.3f}</b> &nbsp;·&nbsp; '
        f"p = <b>{p_str}</b> &nbsp;·&nbsp; at α = 0.05: <b>{decision}</b></p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="mr-delta-badge">{decision}. {conclusion[0].upper()}{conclusion[1:]}</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<p class="mr-sub">Connecting to the t-test</p>', unsafe_allow_html=True)
    t_stats = _coefficient_t_stats(sample, full_names)

    if q == 1:
        name = tested_names[0]
        t_stat, t_df, p_t = t_stats[name]
        st.markdown(
            "<p class=\"mr-note\">When testing a single coefficient, "
            f"F = t². The {_PREDICTOR_LABELS[name].lower()} t-statistic "
            "from the full regression:</p>"
            f'<p class="mr-stat">t = <b>{t_stat:.3f}</b> (df = {t_df}) &nbsp;·&nbsp; '
            f"t² = <b>{t_stat ** 2:.3f}</b> &nbsp;≈&nbsp; F = <b>{f_stat:.3f}</b></p>"
            f'<p class="mr-stat">Two-sided t-test p-value = <b>{_format_p_value(p_t)}</b> '
            f"&nbsp;≈&nbsp; F-test p-value = <b>{p_str}</b></p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="mr-sub">An F-test can also test several predictors '
            "simultaneously. That's its key advantage over testing one "
            "coefficient at a time.</p>",
            unsafe_allow_html=True,
        )
    else:
        rows = "".join(
            f'<p class="mr-stat">{_PREDICTOR_LABELS[name]}: t = '
            f"<b>{t_stats[name][0]:.3f}</b> &nbsp;·&nbsp; p = "
            f"<b>{_format_p_value(t_stats[name][2])}</b> "
            f"({'significant' if t_stats[name][2] < 0.05 else 'not significant'} "
            "at α = 0.05)</p>"
            for name in tested_names
        )
        any_individually_significant = any(t_stats[name][2] < 0.05 for name in tested_names)
        st.markdown(
            "<p class=\"mr-note\">A joint F-test checks whether these "
            f"{added_word} matter together; individual t-tests check each "
            "one alone, holding the others fixed. The two can reach "
            "different conclusions.</p>" + rows,
            unsafe_allow_html=True,
        )
        if reject != any_individually_significant:
            st.markdown(
                '<p class="mr-compare">Here the joint F-test and the '
                "individual t-tests disagree — exactly the situation "
                "where testing one coefficient at a time would mislead "
                "us.</p>",
                unsafe_allow_html=True,
            )
        st.markdown(
            '<p class="mr-sub">The key insight: an F-test can assess '
            "whether several predictors contribute explanatory power "
            "collectively, which no single t-test can do.</p>",
            unsafe_allow_html=True,
        )


def _render_placeholder(title: str) -> None:
    st.markdown(
        f'<p class="mr-note"><b>{title}</b> — coming in a later phase.</p>',
        unsafe_allow_html=True,
    )


def render_f_test_tab(population: Population) -> None:
    """The F-test lesson: hours studied + attendance → exam score, nested models."""
    sample = mr_sample(population)
    st.markdown(
        '<p class="mr-head">The F-test: Comparing Nested Models</p>',
        unsafe_allow_html=True,
    )
    if "ft_sub" not in st.session_state:
        st.session_state.ft_sub = "why"
    sub = st.segmented_control(
        "F-test section",
        options=[key for key, _ in FT_SUBTABS],
        format_func=lambda key: FT_SUBTAB_LABELS[key],
        key="ft_sub",
        label_visibility="collapsed",
        width="content",
    )
    if sub not in FT_SUBTAB_LABELS:
        sub = "why"

    if sub == "dof":
        # Degrees of Freedom compares illustrative F-distributions by df
        # alone; the comparison scenario has no effect on its content.
        scenario = "one"
    else:
        if "ft_scenario" not in st.session_state:
            st.session_state.ft_scenario = "joint"
        st.markdown(
            """<style>
            div[class*="st-key-ft_scenario_card_"] button {
                height: 100%; min-height: 3.6rem; width: 100%;
                white-space: normal; text-align: center; line-height: 1.35;
                border: 1px solid #E2E8F0; border-radius: 8px;
                background: #F8FAFC; color: #374151;
            }
            div[class*="st-key-ft_scenario_card_"] button p { margin: 0; }
            div[class*="st-key-ft_scenario_card_"] button:disabled {
                border: 2px solid #1D4ED8; background: #EFF6FF; color: #12263a;
                opacity: 1;
            }
            </style>""",
            unsafe_allow_html=True,
        )
        card_cols = st.columns(3, gap="small")
        for col, (key, title) in zip(card_cols, _FT_SCENARIOS):
            with col:
                with st.container(key=f"ft_scenario_card_{key}"):
                    selected = st.session_state.ft_scenario == key
                    if st.button(
                        title, key=f"ft_scenario_btn_{key}",
                        width="stretch", disabled=selected,
                    ):
                        st.session_state.ft_scenario = key
                        st.rerun()
        scenario = st.session_state.ft_scenario
        restricted_names, full_names = _FT_SCENARIO_PREDICTORS[scenario]
        st.markdown(
            f'<p class="mr-stat">Comparing <b>{_model_label(restricted_names)}</b> '
            f"vs <b>{_model_label(full_names)}</b> — H₀: "
            f"{_FT_SCENARIO_H0[scenario]}</p>",
            unsafe_allow_html=True,
        )

    if sub == "why":
        _render_why(sample, scenario)
    elif sub == "build":
        _render_build(sample, scenario)
    elif sub == "dist":
        _render_dist(sample, scenario)
    elif sub == "dof":
        _render_dof(sample, scenario)
    elif sub == "interpret":
        _render_interpret(sample, scenario)
