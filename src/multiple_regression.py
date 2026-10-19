"""Multiple regression lesson: two predictors, one regression plane.

Reuses the fixed student population (hours studied, attendance, exam score)
and the generic OLS-with-intercept fit from correlation_causation.fit_multiple.
No t-tests or F-tests here — that belongs to the single-predictor lessons.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.charts import BLUE, FONT, GREY, INK, ORANGE
from src.correlation_causation import fit_multiple
from src.population import Population
from src.sampling import Sample, draw_sample
from src.stat_tips import tipify

MR_SAMPLE_SIZE = 150
MR_SEED = 20261019
PLOTLY_CONFIG = {"displaylogo": False, "responsive": True}
_BETA_HAT = "β̂"
_ALPHA_HAT = "α̂"


def mr_sample(population: Population) -> Sample:
    """One fixed sample, drawn once per population, shared by the whole lesson."""
    key = f"mr_sample_{population.sha256}"
    if key not in st.session_state:
        rng = np.random.default_rng(MR_SEED)
        st.session_state[key] = draw_sample(population, MR_SAMPLE_SIZE, rng)
    return st.session_state[key]


@dataclass(frozen=True)
class MRFit:
    """Fit statistics for a given pair of coefficients (manual or OLS)."""

    alpha: float
    beta1: float
    beta2: float
    fitted: np.ndarray
    residuals: np.ndarray
    sst: float
    sse: float
    ssr: float
    r2: float
    adj_r2: float
    n: int


def evaluate_model(x1: np.ndarray, x2: np.ndarray, y: np.ndarray, alpha: float, beta1: float, beta2: float, p: int = 2) -> MRFit:
    """Fit statistics for ANY coefficients — manual sliders or the OLS estimate.

    `p` is the number of predictors used for adjusted R² (1 for the
    single-predictor model, 2 for the two-predictor model).
    """
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    fitted = alpha + beta1 * x1 + beta2 * x2
    resid = y - fitted
    sse = float(np.dot(resid, resid))
    sst = float(np.sum((y - y.mean()) ** 2))
    ssr = sst - sse
    r2 = float("nan") if sst <= 0 else 1.0 - sse / sst
    n = int(y.size)
    k = p
    adj_r2 = float("nan") if n <= k + 1 or np.isnan(r2) else 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)
    return MRFit(
        alpha=alpha, beta1=beta1, beta2=beta2, fitted=fitted, residuals=resid,
        sst=sst, sse=sse, ssr=ssr, r2=r2, adj_r2=adj_r2, n=n,
    )


def fit_ols_mr(sample: Sample) -> MRFit:
    fit = fit_multiple(sample.exam_score, {"hours": sample.hours_studied, "attendance": sample.attendance})
    return evaluate_model(sample.hours_studied, sample.attendance, sample.exam_score, fit.intercept, *fit.coefs)


def regression_plane_figure(
    sample: Sample,
    alpha: float,
    beta1: float,
    beta2: float,
    *,
    highlight_index: int | None = None,
    show_residuals: bool = True,
    prediction_point: tuple[float, float, float] | None = None,
    height: int = 560,
) -> go.Figure:
    x1, x2, y = sample.hours_studied, sample.attendance, sample.exam_score
    grid1 = np.linspace(float(x1.min()), float(x1.max()), 18)
    grid2 = np.linspace(float(x2.min()), float(x2.max()), 18)
    g1, g2 = np.meshgrid(grid1, grid2)
    plane_z = alpha + beta1 * g1 + beta2 * g2
    fitted = alpha + beta1 * x1 + beta2 * x2

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=grid1, y=grid2, z=plane_z,
            colorscale=[[0, "rgba(29,78,216,0.28)"], [1, "rgba(29,78,216,0.28)"]],
            showscale=False, opacity=0.55, hoverinfo="skip", name="Fitted plane",
        )
    )
    if show_residuals:
        xs: list = []
        ys: list = []
        zs: list = []
        for i in range(x1.size):
            xs += [float(x1[i]), float(x1[i]), None]
            ys += [float(x2[i]), float(x2[i]), None]
            zs += [float(y[i]), float(fitted[i]), None]
        fig.add_trace(
            go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines",
                line=dict(color=GREY, width=2), opacity=0.45,
                hoverinfo="skip", name="Residuals",
            )
        )
    marker_colors = [
        ORANGE if highlight_index is not None and i == highlight_index else BLUE
        for i in range(x1.size)
    ]
    marker_sizes = [
        7 if highlight_index is not None and i == highlight_index else 4
        for i in range(x1.size)
    ]
    fig.add_trace(
        go.Scatter3d(
            x=x1, y=x2, z=y, mode="markers",
            marker=dict(size=marker_sizes, color=marker_colors, line=dict(width=0)),
            name="Students",
            hovertemplate="Hours %{x:.1f}<br>Attendance %{y:.0f}%<br>Score %{z:.1f}<extra></extra>",
        )
    )
    if prediction_point is not None:
        px1, px2, pz = prediction_point
        fig.add_trace(
            go.Scatter3d(
                x=[px1], y=[px2], z=[pz], mode="markers",
                marker=dict(size=8, color=ORANGE, symbol="diamond", line=dict(width=1, color="white")),
                name="Prediction",
            )
        )
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            xaxis=dict(title="Hours studied / week", backgroundcolor="white", gridcolor="#E6EAF0"),
            yaxis=dict(title="Attendance (%)", backgroundcolor="white", gridcolor="#E6EAF0"),
            zaxis=dict(title="Exam score", backgroundcolor="white", gridcolor="#E6EAF0"),
            camera=dict(eye=dict(x=1.65, y=-1.75, z=1.05)),
            aspectmode="cube",
        ),
        showlegend=False,
        font=dict(family=FONT, color=INK, size=11),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _slice_figure(
    sample: Sample,
    alpha: float,
    beta1: float,
    beta2: float,
    *,
    vary: str,
    hold_value: float,
    height: int = 260,
) -> go.Figure:
    """2D cross-section: predicted score as ONE predictor varies, the other held fixed."""
    if vary == "hours":
        grid = np.linspace(float(sample.hours_studied.min()), float(sample.hours_studied.max()), 60)
        line_y = alpha + beta1 * grid + beta2 * hold_value
        x_title = "Hours studied / week"
        caption = f"Attendance held at {hold_value:.0f}%"
    else:
        grid = np.linspace(float(sample.attendance.min()), float(sample.attendance.max()), 60)
        line_y = alpha + beta1 * hold_value + beta2 * grid
        x_title = "Attendance (%)"
        caption = f"Hours studied held at {hold_value:.1f}/week"
    fig = go.Figure(
        go.Scatter(x=grid, y=line_y, mode="lines", line=dict(color=ORANGE, width=2.6), hoverinfo="skip")
    )
    fig.update_layout(
        height=height,
        margin=dict(l=40, r=10, t=24, b=36),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=11),
        title=dict(text=caption, font=dict(size=11, color="#52606D")),
        xaxis=dict(title=x_title, showgrid=True, gridcolor="#E6EAF0"),
        yaxis=dict(title="Predicted exam score", showgrid=True, gridcolor="#E6EAF0"),
    )
    return fig


def _format_coef(value: float, decimals: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{decimals}f}"


def _fit_ols_callback(sample: Sample) -> None:
    fit = fit_ols_mr(sample)
    st.session_state.mr_alpha = round(fit.alpha, 2)
    st.session_state.mr_beta1 = round(fit.beta1, 2)
    st.session_state.mr_beta2 = round(fit.beta2, 2)
    st.session_state.mr_is_ols = True


def _slider_changed() -> None:
    st.session_state.mr_is_ols = False


def _render_why_and_equation(sample: Sample) -> None:
    st.markdown(
        tipify(
            '<p class="mr-note">A single predictor rarely tells the whole story. '
            "Exam score depends on more than hours studied alone — attendance "
            "matters too. <b>Multiple regression</b> lets a model use several "
            "predictors at once.</p>"
            '<p class="mr-note">With one predictor, OLS fits a straight '
            "<b>line</b>. With two predictors, it fits a flat <b>plane</b> "
            "through the cloud of points — the best plane in the sense of "
            "minimizing the sum of squared vertical distances.</p>"
        ),
        unsafe_allow_html=True,
    )
    st.latex(r"\hat{y} = \hat{\alpha} + \hat{\beta}_1 X_1 + \hat{\beta}_2 X_2")
    st.markdown(
        '<p class="mr-note">'
        f"<b>{_ALPHA_HAT}</b> (intercept): the predicted score when hours studied "
        "and attendance are both zero.<br>"
        f"<b>{_BETA_HAT}₁</b>: the predicted change in score per extra hour "
        "studied per week, holding attendance fixed.<br>"
        f"<b>{_BETA_HAT}₂</b>: the predicted change in score per extra "
        "percentage point of attendance, holding hours studied fixed."
        "</p>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        regression_plane_figure(sample, *_ols_defaults(sample), show_residuals=False, height=420),
        width="stretch",
        config=PLOTLY_CONFIG,
        key="mr_why_plane",
    )


def _ols_defaults(sample: Sample) -> tuple[float, float, float]:
    fit = fit_ols_mr(sample)
    return fit.alpha, fit.beta1, fit.beta2


def _ensure_slider_defaults(sample: Sample) -> None:
    if "mr_alpha" not in st.session_state:
        alpha, beta1, beta2 = _ols_defaults(sample)
        st.session_state.mr_alpha = round(alpha, 2)
        st.session_state.mr_beta1 = round(beta1, 2)
        st.session_state.mr_beta2 = round(beta2, 2)
        st.session_state.mr_is_ols = True


def _render_fit_plane(sample: Sample) -> None:
    _ensure_slider_defaults(sample)
    st.markdown(
        '<p class="mr-note">Adjust the coefficients by hand and watch the plane '
        "move, or let OLS find the plane that minimizes the sum of squared "
        "residuals (the vertical distances from each point to the plane).</p>",
        unsafe_allow_html=True,
    )
    toolbar = st.container(horizontal=True, vertical_alignment="bottom", gap="small", wrap=False)
    with toolbar:
        st.slider(f"Intercept α", -40.0, 60.0, key="mr_alpha", step=0.5, on_change=_slider_changed, width=180)
        st.slider(f"β₁ (hours)", -5.0, 15.0, key="mr_beta1", step=0.1, on_change=_slider_changed, width=180)
        st.slider(f"β₂ (attendance)", -1.0, 2.0, key="mr_beta2", step=0.05, on_change=_slider_changed, width=180)
        st.button("Fit OLS", key="mr_fit_ols_btn", type="primary", on_click=_fit_ols_callback, args=(sample,), width=110)

    alpha = float(st.session_state.mr_alpha)
    beta1 = float(st.session_state.mr_beta1)
    beta2 = float(st.session_state.mr_beta2)
    is_ols = bool(st.session_state.get("mr_is_ols", False))
    fit = evaluate_model(sample.hours_studied, sample.attendance, sample.exam_score, alpha, beta1, beta2)

    hat = _BETA_HAT if is_ols else ""
    label = "OLS estimate" if is_ols else "Manual coefficients (not fitted)"
    st.markdown(
        f'<p class="mr-coef">{label}: '
        f"ŷ = {_format_coef(alpha, 2) if is_ols else _format_coef(alpha, 2)} "
        f"+ {_format_coef(beta1, 2)}·X₁ + {_format_coef(beta2, 2)}·X₂"
        f"</p>",
        unsafe_allow_html=True,
    )

    obs_options = list(range(sample.n))
    chart_col, side_col = st.columns([68, 32], gap="medium")
    with side_col:
        selected = st.selectbox(
            "Select a student",
            options=obs_options,
            format_func=lambda i: f"Student {i + 1}",
            key="mr_selected_obs",
        )
        y_obs = float(sample.exam_score[selected])
        y_hat = float(fit.fitted[selected])
        residual = y_obs - y_hat
        st.markdown(
            f'<p class="mr-stat">Observed Y: <b>{y_obs:.1f}</b></p>'
            f'<p class="mr-stat">Predicted ŷ: <b>{y_hat:.1f}</b></p>'
            f'<p class="mr-stat">Residual e = Y − ŷ: <b>{residual:+.1f}</b></p>',
            unsafe_allow_html=True,
        )
        st.markdown('<p class="mr-sub">Predict a new student</p>', unsafe_allow_html=True)
        pred_hours = st.number_input(
            "Hours studied / week", 0.0, 10.0, value=5.0, step=0.5, key="mr_pred_hours"
        )
        pred_att = st.number_input(
            "Attendance (%)", 40.0, 100.0, value=80.0, step=1.0, key="mr_pred_att"
        )
        pred_y = alpha + beta1 * pred_hours + beta2 * pred_att
        st.markdown(f'<p class="mr-stat">Predicted score: <b>{pred_y:.1f}</b></p>', unsafe_allow_html=True)

    with chart_col:
        st.plotly_chart(
            regression_plane_figure(
                sample, alpha, beta1, beta2,
                highlight_index=selected,
                prediction_point=(pred_hours, pred_att, pred_y),
                height=520,
            ),
            width="stretch",
            config=PLOTLY_CONFIG,
            key="mr_fit_plane_chart",
        )


def _render_coefficients(sample: Sample) -> None:
    _ensure_slider_defaults(sample)
    alpha = float(st.session_state.mr_alpha)
    beta1 = float(st.session_state.mr_beta1)
    beta2 = float(st.session_state.mr_beta2)
    st.markdown(
        tipify(
            '<p class="mr-note">Each coefficient is the slope along ONE axis of '
            "the plane while the other predictor is held fixed. The two "
            "coefficients have different units, so their sizes cannot be "
            "compared directly.</p>"
        ),
        unsafe_allow_html=True,
    )
    hold_hours = st.slider("Hold hours studied at", 0.0, 10.0, 5.0, step=0.5, key="mr_hold_hours")
    hold_att = st.slider("Hold attendance at", 40.0, 100.0, 80.0, step=1.0, key="mr_hold_att")
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown(
            f'<p class="mr-sub">{_BETA_HAT}₁ = {_format_coef(beta1)} points per study hour</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _slice_figure(sample, alpha, beta1, beta2, vary="hours", hold_value=hold_att),
            width="stretch", config=PLOTLY_CONFIG, key="mr_slice_hours",
        )
    with right:
        st.markdown(
            f'<p class="mr-sub">{_BETA_HAT}₂ = {_format_coef(beta2)} points per attendance point</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _slice_figure(sample, alpha, beta1, beta2, vary="attendance", hold_value=hold_hours),
            width="stretch", config=PLOTLY_CONFIG, key="mr_slice_att",
        )


def _render_model_fit(sample: Sample) -> None:
    _ensure_slider_defaults(sample)
    ols_fit = fit_ols_mr(sample)
    simple_alpha, simple_beta1 = _single_predictor_ols(sample)
    simple_fit = evaluate_model(
        sample.hours_studied,
        np.zeros_like(sample.attendance),
        sample.exam_score,
        simple_alpha,
        simple_beta1,
        0.0,
        p=1,
    )
    st.markdown(
        f'<div class="mr-calc">'
        f"<p>SST = Σ(y − ȳ)² = <b>{ols_fit.sst:,.1f}</b></p>"
        f"<p>SSR = Σ(ŷ − ȳ)² = <b>{ols_fit.ssr:,.1f}</b></p>"
        f"<p>SSE = Σ(y − ŷ)² = <b>{ols_fit.sse:,.1f}</b></p>"
        f'<p class="mr-formula">R² = 1 − SSE/SST = <b>{ols_fit.r2:.3f}</b></p>'
        f'<p class="mr-formula">Adjusted R² = <b>{ols_fit.adj_r2:.3f}</b></p>'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        tipify(
            '<p class="mr-note">Adding attendance as a second predictor cannot '
            "lower in-sample R² — OLS could always set its coefficient to "
            "zero and recover the one-predictor fit, so the best two-predictor "
            "plane fits at least as well.</p>"
            '<p class="mr-note">Adjusted R² can still fall: it penalizes each '
            "extra predictor, so a predictor that barely helps can make "
            "adjusted R² go down even while R² goes up.</p>"
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="mr-compare">Simple regression (hours only): R² = '
        f"{simple_fit.r2:.3f}. With attendance added: R² = {ols_fit.r2:.3f}, "
        f"adjusted R² = {ols_fit.adj_r2:.3f}.</p>",
        unsafe_allow_html=True,
    )


def _single_predictor_ols(sample: Sample) -> tuple[float, float]:
    fit = fit_multiple(sample.exam_score, {"hours": sample.hours_studied})
    return fit.intercept, fit.coefs[0]


MR_SUBTABS: tuple[tuple[str, str], ...] = (
    ("why", "Why & The Equation"),
    ("fit", "Fit the Plane"),
    ("coef", "Interpreting Coefficients"),
    ("model_fit", "Model Fit"),
)
MR_SUBTAB_LABELS = dict(MR_SUBTABS)


def render_multiple_regression_tab(population: Population) -> None:
    """Multiple regression lesson: hours studied + attendance → exam score."""
    sample = mr_sample(population)
    st.markdown(
        '<p class="mr-head">Multiple Regression: Two Predictors, One Plane</p>',
        unsafe_allow_html=True,
    )
    if "mr_sub" not in st.session_state:
        st.session_state.mr_sub = "why"
    sub = st.segmented_control(
        "Multiple regression section",
        options=[key for key, _ in MR_SUBTABS],
        format_func=lambda key: MR_SUBTAB_LABELS[key],
        key="mr_sub",
        label_visibility="collapsed",
        width="content",
    )
    if sub not in MR_SUBTAB_LABELS:
        sub = "why"
    if sub == "why":
        _render_why_and_equation(sample)
    elif sub == "fit":
        _render_fit_plane(sample)
    elif sub == "coef":
        _render_coefficients(sample)
    else:
        _render_model_fit(sample)
