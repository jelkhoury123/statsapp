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

from src.charts import BLUE, FONT, GREY, INK
from src.correlation_causation import fit_multiple
from src.multiple_regression import (
    _MODEL_FIT_SCENARIOS,
    _strong_attendance_exam_scores,
    evaluate_model,
    mr_sample,
)
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
    ("interpret", "Interpreting the Result"),
)
FT_SUBTAB_LABELS = dict(FT_SUBTABS)


def _fit_nested_models(sample: Sample, scenario: str):
    """Model 1 (hours only) vs Model 2 (hours + attendance), same observations."""
    y = _strong_attendance_exam_scores(sample) if scenario == "strong" else sample.exam_score
    simple_raw = fit_multiple(y, {"hours": sample.hours_studied})
    simple_fit = evaluate_model(
        sample.hours_studied, np.zeros_like(sample.attendance), y,
        simple_raw.intercept, simple_raw.coefs[0], 0.0, p=1,
    )
    multi_raw = fit_multiple(y, {"hours": sample.hours_studied, "attendance": sample.attendance})
    multi_fit = evaluate_model(
        sample.hours_studied, sample.attendance, y,
        multi_raw.intercept, *multi_raw.coefs, p=2,
    )
    return simple_raw, simple_fit, multi_raw, multi_fit


def _render_why(sample: Sample, scenario: str) -> None:
    # Not wrapped in tipify(): its "SSE"→"SSR" legacy alias would silently
    # relabel every literal "SSE" in this text to "SSR" when rendered.
    st.markdown(
        '<p class="mr-sub">Does adding attendance genuinely improve our '
        "model, or does it only look better by chance?</p>"
        '<p class="mr-note">Adding a predictor can only reduce — or at '
        "worst preserve — SSE on the same observations. So some "
        "improvement is <b>guaranteed</b>, even when the predictor has no "
        "real relationship with exam scores. An <b>F-test</b> tells us "
        "whether this improvement is more than that.</p>",
        unsafe_allow_html=True,
    )

    simple_raw, simple_fit, multi_raw, multi_fit = _fit_nested_models(sample, scenario)

    sse_reduction = simple_fit.sse - multi_fit.sse
    sse_reduction_pct = (sse_reduction / simple_fit.sse * 100.0) if simple_fit.sse > 0 else 0.0

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            '<div class="mr-model-card-lg">'
            '<p class="mr-model-name">Model 1 — Hours only</p>'
            f'<p class="mr-model-eq">ŷ = {_format_coef(simple_raw.intercept)} + '
            f"{_format_coef(simple_raw.coefs[0])}·X₁</p>"
            '<div class="mr-metric-row">'
            '<div class="mr-metric-tile"><p class="mr-metric-label">SSE</p>'
            f'<p class="mr-metric-value">{simple_fit.sse:,.0f}</p></div>'
            '<div class="mr-metric-tile"><p class="mr-metric-label">R²</p>'
            f'<p class="mr-metric-value">{simple_fit.r2:.3f}</p></div>'
            "</div></div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            '<div class="mr-model-card-lg">'
            '<p class="mr-model-name">Model 2 — Hours + attendance</p>'
            f'<p class="mr-model-eq">ŷ = {_format_coef(multi_raw.intercept)} + '
            f"{_format_coef(multi_raw.coefs[0])}·X₁ + "
            f"{_format_coef(multi_raw.coefs[1])}·X₂</p>"
            '<div class="mr-metric-row">'
            '<div class="mr-metric-tile accent"><p class="mr-metric-label">SSE</p>'
            f'<p class="mr-metric-value">{multi_fit.sse:,.0f}</p></div>'
            '<div class="mr-metric-tile accent"><p class="mr-metric-label">R²</p>'
            f'<p class="mr-metric-value">{multi_fit.r2:.3f}</p></div>'
            "</div></div>",
            unsafe_allow_html=True,
        )

    max_sse = max(simple_fit.sse, multi_fit.sse, 1e-9)
    w1 = simple_fit.sse / max_sse * 100.0
    w2 = multi_fit.sse / max_sse * 100.0
    st.markdown(
        '<div class="mr-sse-compare">'
        f'<p class="mr-sse-compare-label">SSE reduction: {sse_reduction:,.0f} '
        f"(−{sse_reduction_pct:.1f}%)</p>"
        '<div class="mr-sse-row"><span class="mr-sse-tag">Model 1</span>'
        f'<div class="mr-sse-track"><div class="mr-sse-fill m1" style="width:{w1:.1f}%"></div></div>'
        f'<span class="mr-sse-val">{simple_fit.sse:,.0f}</span></div>'
        '<div class="mr-sse-row"><span class="mr-sse-tag">Model 2</span>'
        f'<div class="mr-sse-track"><div class="mr-sse-fill m2" style="width:{w2:.1f}%"></div></div>'
        f'<span class="mr-sse-val">{multi_fit.sse:,.0f}</span></div>'
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

    _, simple_fit, _, multi_fit = _fit_nested_models(sample, scenario)

    n = int(sample.n)
    q = 1  # predictors added: attendance
    p_full = 2  # predictors in the full model: hours, attendance
    df_residual = n - p_full - 1

    sse_restricted = simple_fit.sse
    sse_full = multi_fit.sse
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


def _compute_f_statistic(sample: Sample, scenario: str) -> tuple[float, int, int]:
    """F-statistic and (df1, df2) for Model 1 vs Model 2 — same formula as Phase 2."""
    _, simple_fit, _, multi_fit = _fit_nested_models(sample, scenario)
    n = int(sample.n)
    q = 1  # predictors added: attendance
    p_full = 2  # predictors in the full model: hours, attendance
    df_residual = n - p_full - 1
    numerator = (simple_fit.sse - multi_fit.sse) / q
    denominator = multi_fit.sse / df_residual
    return numerator / denominator, q, df_residual


def _format_p_value(p: float) -> str:
    if p < 0.0001:
        return f"{p:.3e}"
    return f"{p:.4f}"


_DIST_X_MAX = 10.0


def _f_distribution_figure(
    *, df1: int, df2: int, f_stat: float, critical: float, height: int = 420
) -> go.Figure:
    """F-distribution under H0, rejection region, observed F and p-value area."""
    grid = np.linspace(0.001, _DIST_X_MAX, 500)
    density = fisher_f.pdf(grid, df1, df2)
    peak = float(np.max(density)) if density.size else 1.0
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
        fig.add_annotation(
            x=_DIST_X_MAX, y=peak * 0.55, ax=_DIST_X_MAX * 0.82, ay=peak * 0.55,
            xref="x", yref="y", axref="x", ayref="y",
            text=f"Observed F = {f_stat:.3f} (off scale)",
            showarrow=True, arrowhead=2, arrowcolor=BLUE,
            font=dict(size=12, color=BLUE, family=FONT),
            xanchor="right",
        )

    fig.update_xaxes(title_text="F statistic", range=[0, _DIST_X_MAX])
    fig.update_yaxes(title_text="Density", rangemode="tozero")
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=50, r=20, t=30, b=40),
        font=dict(family=FONT, color=INK, size=12),
        showlegend=False,
    )
    return fig


def _render_dist(sample: Sample, scenario: str) -> None:
    st.markdown(
        '<p class="mr-note">H₀: the additional predictor contributes nothing '
        "beyond the predictors already in the model — here, "
        "H₀: β<sub>attendance</sub> = 0.</p>"
        '<p class="mr-note">If H₀ were true, different random samples would '
        "produce different F-statistics simply from sampling noise. The "
        "F-distribution describes how those statistics would vary under "
        "H₀, assuming the regression model's assumptions hold.</p>",
        unsafe_allow_html=True,
    )

    f_stat, df1, df2 = _compute_f_statistic(sample, scenario)
    p_value = float(fisher_f.sf(f_stat, df1, df2))

    alpha = st.radio(
        "Significance level (α)", options=_ALPHA_OPTIONS, index=1,
        format_func=lambda a: f"α = {a:.2f}", key="ft_alpha", horizontal=True,
    )
    critical = float(fisher_f.ppf(1.0 - alpha, df1, df2))

    st.plotly_chart(
        _f_distribution_figure(df1=df1, df2=df2, f_stat=f_stat, critical=critical),
        width="stretch", config=PLOTLY_CONFIG, key="ft_dist_chart",
    )

    st.markdown(
        f'<p class="mr-stat">df1 = q = <b>{df1}</b> &nbsp;·&nbsp; '
        f"df2 = n − p − 1 = <b>{df2}</b> &nbsp;·&nbsp; "
        f"Observed F = <b>{f_stat:.3f}</b></p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="mr-note">The p-value is the probability, assuming H₀ is '
        "true, of obtaining an F-statistic at least as large as the one "
        f"observed: <b>p = {_format_p_value(p_value)}</b>.</p>",
        unsafe_allow_html=True,
    )

    reject = p_value < alpha
    if reject:
        conclusion = (
            "Reject H₀. The additional predictor provides statistically "
            "significant explanatory power."
        )
    else:
        conclusion = (
            "Do not reject H₀. We lack sufficient evidence that the "
            "additional predictor improves the model."
        )
    st.markdown(f'<p class="mr-delta-badge">{conclusion}</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="mr-compare">Statistical significance does not establish '
        "causality or practical importance.</p>",
        unsafe_allow_html=True,
    )


def _attendance_t_statistic(sample: Sample, scenario: str) -> tuple[float, int]:
    """t-statistic for beta_attendance in the full model (hours + attendance), and its df."""
    y = _strong_attendance_exam_scores(sample) if scenario == "strong" else sample.exam_score
    n = sample.n
    x = np.column_stack([np.ones(n), sample.hours_studied, sample.attendance])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    df_residual = int(n - x.shape[1])  # n − p − 1
    sigma2 = float(resid @ resid) / df_residual
    xtx_inv = np.linalg.inv(x.T @ x)
    se_attendance = float(np.sqrt(sigma2 * xtx_inv[2, 2]))
    return float(beta[2] / se_attendance), df_residual


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
    st.markdown(
        '<p class="mr-note">H₀: the added predictor has no additional '
        "explanatory power after accounting for the existing predictor — "
        "H₀: β<sub>attendance</sub> = 0.<br>"
        "H₁: the added predictor contributes explanatory power — "
        "H₁: β<sub>attendance</sub> ≠ 0.</p>",
        unsafe_allow_html=True,
    )

    _, simple_fit, _, multi_fit = _fit_nested_models(sample, scenario)
    n = int(sample.n)
    q = 1
    df1_res = n - 1 - 1
    df2_res = n - 2 - 1
    rss1, rss2 = simple_fit.sse, multi_fit.sse
    sum_sq = rss1 - rss2
    f_stat, df1, df2 = _compute_f_statistic(sample, scenario)
    p_value = float(fisher_f.sf(f_stat, df1, df2))

    st.code(
        "model1 <- lm(score ~ hours, data = students)\n"
        "model2 <- lm(score ~ hours + attendance, data = students)\n\n"
        "anova(model1, model2)",
        language="r",
    )

    p_str = _format_p_value(p_value)
    stars = _significance_stars(p_value)
    st.code(
        "Analysis of Variance Table\n\n"
        "Model 1: score ~ hours\n"
        "Model 2: score ~ hours + attendance\n"
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
    conclusion = (
        "the additional predictor provides statistically significant "
        "explanatory power." if reject else
        "we lack sufficient evidence that the additional predictor "
        "improves the model."
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
    t_stat, t_df = _attendance_t_statistic(sample, scenario)
    p_t = 2.0 * float(student_t.sf(abs(t_stat), t_df))
    st.markdown(
        "<p class=\"mr-note\">When testing a single coefficient, "
        "F = t². The attendance t-statistic from the full regression:</p>"
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

    scenario = st.segmented_control(
        "Scenario",
        options=[key for key, _ in _MODEL_FIT_SCENARIOS],
        format_func=lambda k: dict(_MODEL_FIT_SCENARIOS)[k],
        key="ft_scenario",
        label_visibility="collapsed",
        default="strong",
    )
    if scenario not in dict(_MODEL_FIT_SCENARIOS):
        scenario = "strong"

    if sub == "why":
        _render_why(sample, scenario)
    elif sub == "build":
        _render_build(sample, scenario)
    elif sub == "dist":
        _render_dist(sample, scenario)
    elif sub == "interpret":
        _render_interpret(sample, scenario)
