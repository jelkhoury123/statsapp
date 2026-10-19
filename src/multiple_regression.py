"""Multiple regression lesson: two predictors, one regression plane.

Reuses the fixed student population (hours studied, attendance, exam score)
and the generic OLS-with-intercept fit from correlation_causation.fit_multiple.
No t-tests or F-tests here — that belongs to the single-predictor lessons.
"""

from __future__ import annotations

import time
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
    plane_color: str = "rgba(29,78,216,0.28)",
    plane_opacity: float = 0.55,
    camera_eye: dict | None = None,
    scene_domain: dict | None = None,
    uirevision: str | None = None,
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
            colorscale=[[0, plane_color], [1, plane_color]],
            showscale=False, opacity=plane_opacity, hoverinfo="skip", name="Fitted plane",
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
    # These trace groups are ALWAYS added, just with empty data when inactive,
    # so the trace count never changes between modes — Plotly resets the 3D
    # camera (even with uirevision set and no explicit camera) whenever the
    # number of traces changes between renders, confirmed by direct testing.
    have_highlight = highlight_index is not None and 0 <= highlight_index < x1.size
    if have_highlight:
        hx1, hx2 = float(x1[highlight_index]), float(x2[highlight_index])
        hy_obs, hy_hat = float(y[highlight_index]), float(fitted[highlight_index])
    else:
        hx1 = hx2 = hy_obs = hy_hat = None
    fig.add_trace(
        go.Scatter3d(
            x=[hx1] if have_highlight else [], y=[hx2] if have_highlight else [],
            z=[hy_hat] if have_highlight else [], mode="markers",
            marker=dict(size=6, color="white", line=dict(width=2, color=ORANGE)),
            name="Predicted Ŷ",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[hx1, hx1] if have_highlight else [], y=[hx2, hx2] if have_highlight else [],
            z=[hy_obs, hy_hat] if have_highlight else [], mode="lines",
            line=dict(color=ORANGE, width=6), opacity=0.9,
            hoverinfo="skip", name="Selected residual",
        )
    )

    have_prediction = prediction_point is not None
    if have_prediction:
        px1, px2, pz = prediction_point
        z_floor = float(min(float(y.min()), float(plane_z.min())))
        x_edge, y_edge = float(grid1.min()), float(grid2.min())
    else:
        px1 = px2 = pz = z_floor = x_edge = y_edge = None
    proj_style = dict(color=ORANGE, width=1.8, dash="dot")

    def _pts(*vals):
        return list(vals) if have_prediction else []

    # (X1, X2) marked on the predictors' base plane.
    fig.add_trace(
        go.Scatter3d(
            x=_pts(px1), y=_pts(px2), z=_pts(z_floor), mode="markers",
            marker=dict(size=4, color=ORANGE, symbol="x", line=dict(width=0)),
            opacity=0.7, hoverinfo="skip", name="Prediction (X₁, X₂)",
        )
    )
    # Dashed lines from the base X, parallel to each predictor axis,
    # reaching the opposite axis — reads off the X1 and X2 values.
    fig.add_trace(
        go.Scatter3d(
            x=_pts(px1, px1), y=_pts(px2, y_edge), z=_pts(z_floor, z_floor), mode="lines",
            line=proj_style, opacity=0.5, hoverinfo="skip", name="Projection",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=_pts(px1, x_edge), y=_pts(px2, px2), z=_pts(z_floor, z_floor), mode="lines",
            line=proj_style, opacity=0.5, hoverinfo="skip", name="Projection",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=_pts(px1), y=_pts(y_edge), z=_pts(z_floor), mode="markers+text",
            marker=dict(size=3, color=ORANGE, line=dict(width=0)), opacity=0.9,
            text=[f"{px1:.1f} h"] if have_prediction else [], textposition="bottom center",
            textfont=dict(size=10, color=ORANGE),
            hoverinfo="skip", name="Hours value",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=_pts(x_edge), y=_pts(px2), z=_pts(z_floor), mode="markers+text",
            marker=dict(size=3, color=ORANGE, line=dict(width=0)), opacity=0.9,
            text=[f"{px2:.0f}%"] if have_prediction else [], textposition="bottom center",
            textfont=dict(size=10, color=ORANGE),
            hoverinfo="skip", name="Attendance value",
        )
    )
    # Dashed vertical projection up to the regression plane (the diamond).
    fig.add_trace(
        go.Scatter3d(
            x=_pts(px1, px1), y=_pts(px2, px2), z=_pts(z_floor, pz), mode="lines",
            line=proj_style, opacity=0.6, hoverinfo="skip", name="Projection",
        )
    )
    # Dashed horizontal projection toward the score (z) axis.
    fig.add_trace(
        go.Scatter3d(
            x=_pts(px1, x_edge), y=_pts(px2, y_edge), z=_pts(pz, pz), mode="lines",
            line=proj_style, opacity=0.6, hoverinfo="skip", name="Projection",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=_pts(px1), y=_pts(px2), z=_pts(pz), mode="markers",
            marker=dict(size=8, color=ORANGE, symbol="diamond", line=dict(width=1, color="white")),
            name="Prediction",
        )
    )
    scene_kwargs = dict(
        xaxis=dict(title="Hours studied / week", backgroundcolor="white", gridcolor="#E6EAF0"),
        yaxis=dict(title="Attendance (%)", backgroundcolor="white", gridcolor="#E6EAF0"),
        zaxis=dict(title="Exam score", backgroundcolor="white", gridcolor="#E6EAF0"),
        aspectmode="cube",
    )
    if camera_eye is not None:
        # IMPORTANT: an explicit camera here overrides Plotly's uirevision-
        # preserved camera on every rebuild, even when the value is unchanged —
        # confirmed by direct testing. Callers that need rotation to persist
        # across reruns must only pass this on the chart's very first render
        # and omit it (None) afterward.
        scene_kwargs["camera"] = dict(eye=camera_eye)
    if scene_domain is not None:
        scene_kwargs["domain"] = scene_domain
    if uirevision is not None:
        # Keeps the user's camera orientation/zoom across reruns that redraw
        # the same figure (e.g. only the prediction marker moved).
        scene_kwargs["uirevision"] = uirevision
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=10, b=0),
        scene=scene_kwargs,
        showlegend=False,
        font=dict(family=FONT, color=INK, size=11),
        paper_bgcolor="rgba(0,0,0,0)",
        uirevision=uirevision,
    )
    return fig


def _plane_value_range(
    alpha: float, beta1: float, beta2: float,
    x1_bounds: tuple[float, float], x2_bounds: tuple[float, float],
    pad: float = 0.08,
) -> tuple[float, float]:
    """Min/max of the fitted plane over the full domain of both predictors.

    The plane is linear, so its extremes over a rectangular domain always
    land on the four corners.
    """
    vals = [alpha + beta1 * x1 + beta2 * x2 for x1 in x1_bounds for x2 in x2_bounds]
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 1e-6)
    return lo - pad * span, hi + pad * span


def _slice_figure(
    sample: Sample,
    alpha: float,
    beta1: float,
    beta2: float,
    *,
    vary: str,
    hold_value: float,
    reference_hold: float,
    y_range: tuple[float, float],
    height: int = 280,
) -> go.Figure:
    """2D cross-section: predicted score as ONE predictor varies, the other held fixed.

    Draws both the current line (orange) and a fixed dashed-grey reference
    line (at the page's initial hold value), on a stable y-axis range, so a
    vertical shift from changing the held predictor is visible rather than
    hidden by autoscaling.
    """
    if vary == "hours":
        grid = np.linspace(float(sample.hours_studied.min()), float(sample.hours_studied.max()), 60)
        line_y = alpha + beta1 * grid + beta2 * hold_value
        ref_y = alpha + beta1 * grid + beta2 * reference_hold
        x_title = "Hours studied / week"
        caption = f"Attendance held at {hold_value:.0f}%"
    else:
        grid = np.linspace(float(sample.attendance.min()), float(sample.attendance.max()), 60)
        line_y = alpha + beta1 * hold_value + beta2 * grid
        ref_y = alpha + beta1 * reference_hold + beta2 * grid
        x_title = "Attendance (%)"
        caption = f"Hours studied held at {hold_value:.1f}/week"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=grid, y=ref_y, mode="lines",
            line=dict(color=GREY, width=2, dash="dash"),
            name="Reference", hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grid, y=line_y, mode="lines",
            line=dict(color=ORANGE, width=2.6),
            name="Current", hoverinfo="skip",
        )
    )
    fig.update_layout(
        height=height,
        margin=dict(l=40, r=10, t=24, b=36),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=11),
        title=dict(text=caption, font=dict(size=11, color="#52606D")),
        showlegend=False,
        xaxis=dict(title=x_title, showgrid=True, gridcolor="#E6EAF0"),
        yaxis=dict(title="Predicted exam score", showgrid=True, gridcolor="#E6EAF0", range=list(y_range)),
    )
    return fig


def _format_coef(value: float, decimals: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{decimals}f}"


_PLANE_ANIM_FRAMES = 18
_PLANE_ANIM_FRAME_DELAY = 0.12
# Deliberately poor starting plane (not OLS) so Fit OLS has a real fit to show.
_PLANE_INITIAL = (0.0, 2.0, 0.3)


def _plane_animation_steps(
    start: tuple[float, float, float], end: tuple[float, float, float], count: int = _PLANE_ANIM_FRAMES
) -> list[tuple[float, float, float]]:
    """Even steps from a candidate plane to the OLS plane."""
    fractions = np.linspace(0.0, 1.0, int(count))
    return [
        tuple(float(s) + float(f) * (float(e) - float(s)) for s, e in zip(start, end))
        for f in fractions
    ]


def _plane_slider_changed() -> None:
    st.session_state.mr_plane_is_ols = False


def _render_why_and_equation(sample: Sample) -> None:
    # 1. Introduction
    st.markdown(
        tipify(
            '<p class="mr-note"><b>Can we improve our model by '
            "adding another predictor?</b></p>"
            '<p class="mr-note">So far, we\'ve used hours studied to predict '
            "exam scores. But what if we also consider class attendance?</p>"
            '<ul class="mr-list">'
            "<li>Could attendance explain some of the variation in exam "
            "scores that study hours alone cannot explain?</li>"
            "<li>Could adding attendance improve our predictions?</li>"
            "</ul>"
            '<p class="mr-note">Multiple regression allows us to investigate '
            "these questions by including several predictors in the same "
            "model.</p>"
        ),
        unsafe_allow_html=True,
    )

    # 2. From line to plane
    st.markdown('<p class="mr-sub">From line to plane</p>', unsafe_allow_html=True)
    eq_old, eq_new = st.columns(2, gap="medium")
    with eq_old:
        st.markdown('<p class="mr-eq-label">Simple regression</p>', unsafe_allow_html=True)
        st.latex(r"\hat{y} = \hat{\alpha} + \hat{\beta}_1 X_1")
    with eq_new:
        st.markdown('<p class="mr-eq-label">Multiple regression</p>', unsafe_allow_html=True)
        st.latex(rf"\hat{{y}} = \hat{{\alpha}} + \hat{{\beta}}_1 X_1 + \color{{{ORANGE}}}{{\hat{{\beta}}_2 X_2}}")
    st.markdown(
        '<p class="mr-note">With one predictor, OLS fits a <b>line</b>. '
        "With two predictors, OLS fits a <b>plane</b>.</p>",
        unsafe_allow_html=True,
    )

    # 3. Regression plane + 4. Coefficients, side by side
    st.markdown('<p class="mr-sub">The regression plane</p>', unsafe_allow_html=True)
    chart_col, coef_col = st.columns([0.75, 0.25], gap="medium", vertical_alignment="top")
    with chart_col:
        with st.container(key="mr_why_chart_frame"):
            st.plotly_chart(
                regression_plane_figure(
                    sample, *_ols_defaults(sample), show_residuals=False, height=435,
                    plane_color="rgba(234,88,12,0.5)", plane_opacity=0.65,
                    camera_eye=dict(x=1.3, y=-1.6, z=1.25),
                    scene_domain=dict(x=[0, 1], y=[0, 1]),
                ),
                width="stretch",
                config=PLOTLY_CONFIG,
                key="mr_why_plane",
            )
    with coef_col:
        st.markdown(
            f'<p class="mr-coef-card"><b>{_ALPHA_HAT}</b> — Intercept<br>'
            "Predicted score when both hours studied and attendance "
            "are zero.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p class="mr-coef-card"><b>{_BETA_HAT}₁</b> — Hours studied<br>'
            "Change in predicted score per extra hour studied, "
            "<b>holding attendance constant</b>.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p class="mr-coef-card"><b>{_BETA_HAT}₂</b> — Attendance<br>'
            "Change in predicted score per extra point of attendance, "
            "<b>holding hours studied constant</b>.</p>",
            unsafe_allow_html=True,
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


def _ensure_plane_defaults() -> None:
    if "mr_plane_alpha" not in st.session_state:
        st.session_state.mr_plane_alpha = _PLANE_INITIAL[0]
        st.session_state.mr_plane_beta1 = _PLANE_INITIAL[1]
        st.session_state.mr_plane_beta2 = _PLANE_INITIAL[2]
        st.session_state.mr_plane_is_ols = False
    if "mr_plane_mode" not in st.session_state:
        st.session_state.mr_plane_mode = "default"


def _mr_obs_selected() -> None:
    """Examine mode: picking a student clears any active prediction."""
    if st.session_state.get("mr_selected_obs") is not None:
        st.session_state.mr_plane_mode = "examine"
    else:
        st.session_state.mr_plane_mode = "default"


def _mr_prediction_changed() -> None:
    """Predict mode: editing the prediction inputs clears the selected student."""
    st.session_state.mr_plane_mode = "predict"
    st.session_state.mr_selected_obs = None


@st.fragment
def _render_fit_plane(sample: Sample) -> None:
    _ensure_plane_defaults()
    st.markdown(
        '<p class="mr-note">Adjust the coefficients by hand, or click Fit OLS '
        "to watch the plane move to the best fit.</p>",
        unsafe_allow_html=True,
    )

    animating = bool(st.session_state.get("mr_plane_animating"))
    anim_steps = st.session_state.get("mr_plane_anim_steps") or []
    anim_frame = int(st.session_state.get("mr_plane_anim_frame", 0))
    if animating and anim_steps and 0 <= anim_frame < len(anim_steps):
        step_alpha, step_beta1, step_beta2 = anim_steps[anim_frame]
        st.session_state.mr_plane_alpha = round(float(step_alpha), 3)
        st.session_state.mr_plane_beta1 = round(float(step_beta1), 3)
        st.session_state.mr_plane_beta2 = round(float(step_beta2), 3)

    toolbar = st.container(horizontal=True, vertical_alignment="bottom", gap="small", wrap=False)
    with toolbar:
        st.slider(
            "Intercept α", -40.0, 60.0, key="mr_plane_alpha", step=0.5,
            on_change=_plane_slider_changed, width=180, disabled=animating,
        )
        st.slider(
            "β₁ (hours)", -5.0, 15.0, key="mr_plane_beta1", step=0.1,
            on_change=_plane_slider_changed, width=180, disabled=animating,
        )
        st.slider(
            "β₂ (attendance)", -1.0, 2.0, key="mr_plane_beta2", step=0.05,
            on_change=_plane_slider_changed, width=180, disabled=animating,
        )
        fit_clicked = st.button(
            "Fit OLS", key="mr_fit_ols_btn", type="primary", width=110, disabled=animating,
        )

    if fit_clicked and not animating:
        start = (
            float(st.session_state.mr_plane_alpha),
            float(st.session_state.mr_plane_beta1),
            float(st.session_state.mr_plane_beta2),
        )
        fit_before = evaluate_model(sample.hours_studied, sample.attendance, sample.exam_score, *start)
        st.session_state.mr_plane_sse_before = float(fit_before.sse)
        st.session_state.mr_plane_r2_before = float(fit_before.r2)
        end = _ols_defaults(sample)
        st.session_state.mr_plane_anim_steps = _plane_animation_steps(start, end)
        st.session_state.mr_plane_anim_frame = 0
        st.session_state.mr_plane_animating = True
        st.rerun(scope="fragment")

    alpha = float(st.session_state.mr_plane_alpha)
    beta1 = float(st.session_state.mr_plane_beta1)
    beta2 = float(st.session_state.mr_plane_beta2)
    is_ols = bool(st.session_state.get("mr_plane_is_ols", False))
    fit = evaluate_model(sample.hours_studied, sample.attendance, sample.exam_score, alpha, beta1, beta2)

    label = "OLS fitted model" if is_ols else "Current model"
    st.markdown(
        f'<p class="mr-coef">{label}: '
        f"ŷ = {_format_coef(alpha, 2)} + {_format_coef(beta1, 2)}·X₁ + {_format_coef(beta2, 2)}·X₂"
        f"</p>",
        unsafe_allow_html=True,
    )
    # SST / SSE / R², updated every animation frame since `fit` is
    # recomputed from the current coefficients each run.
    st.markdown(
        f'<p class="mr-stat">SST = <b>{fit.sst:,.1f}</b>'
        f"&nbsp;&nbsp;·&nbsp;&nbsp;SSE = <b>{fit.sse:,.1f}</b>"
        f'&nbsp;&nbsp;·&nbsp;&nbsp;<span class="mr-formula">R² = <b>{fit.r2:.3f}</b></span></p>',
        unsafe_allow_html=True,
    )
    sse_before = st.session_state.get("mr_plane_sse_before")
    r2_before = st.session_state.get("mr_plane_r2_before")
    if is_ols and not animating and sse_before is not None and float(sse_before) > 0:
        sse_before = float(sse_before)
        r2_before = float(r2_before) if r2_before is not None else float("nan")
        sse_pct = (sse_before - fit.sse) / sse_before * 100.0
        r2_delta = fit.r2 - r2_before
        st.markdown(
            f'<p class="mr-compare">Initial SSE {sse_before:,.1f} → Final SSE '
            f"{fit.sse:,.1f} <b>(−{sse_pct:.1f}%)</b><br>"
            f"Initial R² {r2_before:.3f} → Final R² {fit.r2:.3f} "
            f"<b>(+{r2_delta:.3f})</b></p>",
            unsafe_allow_html=True,
        )

    chart_col, side_col = st.columns([0.7, 0.3], gap="medium", vertical_alignment="top")
    with side_col:
        with st.container(key="mr_examine_panel"):
            st.markdown('<p class="mr-sub">Examine an observation</p>', unsafe_allow_html=True)
            obs_options = [None] + list(range(sample.n))
            selected = st.selectbox(
                "Select a student",
                options=obs_options,
                format_func=lambda i: "None" if i is None else f"Student {i + 1}",
                key="mr_selected_obs",
                on_change=_mr_obs_selected,
            )
            if selected is None:
                st.markdown(
                    '<p class="mr-stat">Select a student to see their observed score, '
                    "predicted score and residual.</p>",
                    unsafe_allow_html=True,
                )
            else:
                y_obs = float(sample.exam_score[selected])
                y_hat = float(fit.fitted[selected])
                residual = y_obs - y_hat
                st.markdown(
                    f'<p class="mr-stat">Observed Y: <b>{y_obs:.1f}</b></p>'
                    f'<p class="mr-stat">Predicted ŷ: <b>{y_hat:.1f}</b></p>'
                    f'<p class="mr-stat">Residual e = Y − ŷ: <b>{residual:+.1f}</b></p>',
                    unsafe_allow_html=True,
                )

        with st.container(key="mr_predict_panel"):
            st.markdown('<p class="mr-sub">Make a prediction</p>', unsafe_allow_html=True)
            pred_hours = st.number_input(
                "Hours studied / week", 0.0, 10.0, value=5.0, step=0.5, key="mr_pred_hours",
                on_change=_mr_prediction_changed,
            )
            pred_att = st.number_input(
                "Attendance (%)", 40.0, 100.0, value=80.0, step=1.0, key="mr_pred_att",
                on_change=_mr_prediction_changed,
            )
            pred_y = alpha + beta1 * pred_hours + beta2 * pred_att
            st.markdown(f'<p class="mr-stat">Predicted score: <b>{pred_y:.1f}</b></p>', unsafe_allow_html=True)

    mode = st.session_state.get("mr_plane_mode", "default")
    chart_highlight = selected if mode == "examine" else None
    chart_prediction = (pred_hours, pred_att, pred_y) if mode == "predict" else None

    # Only the chart's very first render gets an explicit camera: passing one
    # on every rebuild would override the user's own rotation/zoom (Plotly's
    # uirevision only protects attributes the figure doesn't keep re-asserting).
    chart_camera_eye = None
    if not st.session_state.get("mr_plane_camera_set"):
        chart_camera_eye = dict(x=1.05, y=-1.3, z=1.0)
        st.session_state.mr_plane_camera_set = True

    with chart_col:
        st.plotly_chart(
            regression_plane_figure(
                sample, alpha, beta1, beta2,
                highlight_index=chart_highlight,
                prediction_point=chart_prediction,
                height=580,
                plane_color="rgba(234,88,12,0.5)", plane_opacity=0.65,
                camera_eye=chart_camera_eye,
                scene_domain=dict(x=[0, 1], y=[0, 1]),
                uirevision="mr_fit_plane",
            ),
            width="stretch",
            config=PLOTLY_CONFIG,
            key="mr_fit_plane_chart",
            theme=None,
        )
        st.markdown(
            '<p class="mr-legend">'
            '<span><i class="mr-legend-dot" style="background:#1D4ED8"></i>Observed students</span>'
            '<span><i class="mr-legend-swatch" style="background:rgba(234,88,12,0.5)"></i>Current regression model</span>'
            '<span><i class="mr-legend-line"></i>Residuals</span>'
            '<span><i class="mr-legend-diamond"></i>Predicted new student</span>'
            "</p>",
            unsafe_allow_html=True,
        )

    if animating:
        if anim_frame < len(anim_steps) - 1:
            time.sleep(_PLANE_ANIM_FRAME_DELAY)
            st.session_state.mr_plane_anim_frame = anim_frame + 1
            st.rerun(scope="fragment")
        else:
            # Last frame already applied the exact OLS values (above, before
            # the sliders were instantiated) — just close out the animation.
            st.session_state.mr_plane_is_ols = True
            st.session_state.mr_plane_animating = False
            st.session_state.pop("mr_plane_anim_steps", None)
            st.session_state.pop("mr_plane_anim_frame", None)
            st.rerun(scope="fragment")


_HOLD_HOURS_REFERENCE = 5.0
_HOLD_ATT_REFERENCE = 80.0


def _render_coefficients(sample: Sample) -> None:
    _ensure_slider_defaults(sample)
    alpha = float(st.session_state.mr_alpha)
    beta1 = float(st.session_state.mr_beta1)
    beta2 = float(st.session_state.mr_beta2)
    st.markdown(
        tipify(
            '<p class="mr-note"><b>How can we measure the relationship between '
            "one predictor and exam scores independently of the other "
            "predictor?</b></p>"
            '<p class="mr-note">Each coefficient is the slope along ONE axis of '
            "the plane while the other predictor is held fixed. The two "
            "coefficients have different units, so their sizes cannot be "
            "compared directly.</p>"
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="mr-mini-legend">'
        '<span><i class="mr-mini-legend-line dash"></i>Reference</span>'
        '<span><i class="mr-mini-legend-line solid"></i>Current prediction</span>'
        "</p>",
        unsafe_allow_html=True,
    )

    hours_y_range = _plane_value_range(alpha, beta1, beta2, (0.0, 10.0), (40.0, 100.0))
    att_y_range = hours_y_range  # same plane, same domain — keep both charts on one scale

    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown('<p class="mr-sub">Study Hours</p>', unsafe_allow_html=True)
        hold_att = st.slider(
            "Hold attendance constant at", 40.0, 100.0, _HOLD_ATT_REFERENCE, step=1.0, key="mr_hold_att"
        )
        delta_hours = beta2 * (hold_att - _HOLD_ATT_REFERENCE)
        st.markdown(
            f'<p class="mr-stat">{_BETA_HAT}₁ = {_format_coef(beta1)} points per study hour</p>'
            f'<p class="mr-delta-badge">ΔŶ = {delta_hours:+.1f}</p>'
            f'<p class="mr-stat">ΔŶ = {_format_coef(beta2)} × ({hold_att:.0f} − '
            f"{_HOLD_ATT_REFERENCE:.0f}) = {delta_hours:+.1f}</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _slice_figure(
                sample, alpha, beta1, beta2, vary="hours", hold_value=hold_att,
                reference_hold=_HOLD_ATT_REFERENCE, y_range=hours_y_range,
            ),
            width="stretch", config=PLOTLY_CONFIG, key="mr_slice_hours",
        )
    with right:
        st.markdown('<p class="mr-sub">Attendance</p>', unsafe_allow_html=True)
        hold_hours = st.slider(
            "Hold study hours constant at", 0.0, 10.0, _HOLD_HOURS_REFERENCE, step=0.5, key="mr_hold_hours"
        )
        delta_att = beta1 * (hold_hours - _HOLD_HOURS_REFERENCE)
        st.markdown(
            f'<p class="mr-stat">{_BETA_HAT}₂ = {_format_coef(beta2)} points per attendance point</p>'
            f'<p class="mr-delta-badge">ΔŶ = {delta_att:+.1f}</p>'
            f'<p class="mr-stat">ΔŶ = {_format_coef(beta1)} × ({hold_hours:.1f} − '
            f"{_HOLD_HOURS_REFERENCE:.1f}) = {delta_att:+.1f}</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _slice_figure(
                sample, alpha, beta1, beta2, vary="attendance", hold_value=hold_hours,
                reference_hold=_HOLD_HOURS_REFERENCE, y_range=att_y_range,
            ),
            width="stretch", config=PLOTLY_CONFIG, key="mr_slice_att",
        )
    st.markdown(
        '<p class="mr-sub">Key insight</p>'
        '<p class="mr-note">Each coefficient measures how predicted Y changes '
        "when its predictor increases by one unit, holding the other "
        "predictors constant.</p>"
        '<p class="mr-note">Changing the held-constant predictor shifts the '
        "line vertically without changing its slope.</p>"
        '<p class="mr-compare">Statistical adjustment does not establish '
        "causation.</p>",
        unsafe_allow_html=True,
    )


_MODEL_FIT_SCENARIOS: tuple[tuple[str, str], ...] = (
    ("weak", "Current data — attendance adds little"),
    ("strong", "Illustrative — attendance adds a lot"),
)


def _strong_attendance_exam_scores(sample: Sample, seed: int = 777) -> np.ndarray:
    """Synthetic scores, same N and same predictor values, where attendance
    is the dominant driver — for contrast with the real (weak-effect) data."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 4.0, size=sample.n)
    return 10.0 + 3.0 * sample.hours_studied + 0.55 * sample.attendance + noise


def _r2_bar_row(label: str, model1_val: float, model2_val: float) -> str:
    v1 = max(0.0, min(1.0, model1_val))
    v2 = max(0.0, min(1.0, model2_val))
    return (
        f'<p class="mr-bar-label">{label}</p>'
        f'<div class="mr-bar-row"><span class="mr-bar-tag">M1</span>'
        f'<div class="mr-bar-track"><div class="mr-bar-fill m1" style="width:{v1 * 100:.1f}%"></div></div>'
        f'<span class="mr-bar-value">{model1_val:.3f}</span></div>'
        f'<div class="mr-bar-row"><span class="mr-bar-tag">M2</span>'
        f'<div class="mr-bar-track"><div class="mr-bar-fill m2" style="width:{v2 * 100:.1f}%"></div></div>'
        f'<span class="mr-bar-value">{model2_val:.3f}</span></div>'
    )


def _render_model_fit(sample: Sample) -> None:
    _ensure_slider_defaults(sample)
    st.markdown(
        '<p class="mr-note">Two nested models, fit to the <b>same '
        "150 students</b>: one uses hours studied alone, the other adds "
        "attendance.</p>",
        unsafe_allow_html=True,
    )

    scenario = st.segmented_control(
        "Scenario",
        options=[key for key, _ in _MODEL_FIT_SCENARIOS],
        format_func=lambda k: dict(_MODEL_FIT_SCENARIOS)[k],
        key="mr_model_fit_scenario",
        label_visibility="collapsed",
    )
    if scenario not in dict(_MODEL_FIT_SCENARIOS):
        scenario = "weak"

    if scenario == "strong":
        y = _strong_attendance_exam_scores(sample)
    else:
        y = sample.exam_score

    simple_fit_raw = fit_multiple(y, {"hours": sample.hours_studied})
    simple_fit = evaluate_model(
        sample.hours_studied, np.zeros_like(sample.attendance), y,
        simple_fit_raw.intercept, simple_fit_raw.coefs[0], 0.0, p=1,
    )
    multi_fit_raw = fit_multiple(y, {"hours": sample.hours_studied, "attendance": sample.attendance})
    ols_fit = evaluate_model(
        sample.hours_studied, sample.attendance, y,
        multi_fit_raw.intercept, *multi_fit_raw.coefs, p=2,
    )

    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown(
            f'<p class="mr-coef-card"><b>Model 1 — Hours only</b><br>'
            f"ŷ = {_format_coef(simple_fit.alpha, 2)} + {_format_coef(simple_fit.beta1, 2)}·X₁<br>"
            f"R² = <b>{simple_fit.r2:.3f}</b> · Adjusted R² = <b>{simple_fit.adj_r2:.3f}</b>"
            f"</p>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f'<p class="mr-coef-card"><b>Model 2 — Hours + attendance</b><br>'
            f"ŷ = {_format_coef(ols_fit.alpha, 2)} + {_format_coef(ols_fit.beta1, 2)}·X₁ "
            f"+ {_format_coef(ols_fit.beta2, 2)}·X₂<br>"
            f"R² = <b>{ols_fit.r2:.3f}</b> · Adjusted R² = <b>{ols_fit.adj_r2:.3f}</b>"
            f"</p>",
            unsafe_allow_html=True,
        )

    delta_r2 = ols_fit.r2 - simple_fit.r2
    delta_adj_r2 = ols_fit.adj_r2 - simple_fit.adj_r2

    bar_left, bar_right = st.columns(2, gap="medium")
    with bar_left:
        st.markdown(_r2_bar_row("R² (0–1 scale)", simple_fit.r2, ols_fit.r2), unsafe_allow_html=True)
    with bar_right:
        st.markdown(
            _r2_bar_row("Adjusted R² (0–1 scale)", simple_fit.adj_r2, ols_fit.adj_r2),
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<p class="mr-stat">ΔR² = {delta_r2:+.4f} &nbsp;·&nbsp; '
        f"ΔAdjusted R² = {delta_adj_r2:+.4f}</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        tipify(
            '<ul class="mr-list">'
            "<li>Adding a predictor can never lower in-sample R² — OLS could "
            "always set its coefficient to zero and recover Model 1, so "
            "Model 2 fits at least as well.</li>"
            "<li>Adjusted R² penalizes each extra predictor, so it can "
            "<b>decrease</b> even while R² goes up, if the predictor barely "
            "helps.</li>"
            "<li>A higher adjusted R² does not guarantee better "
            "<b>out-of-sample</b> predictions — it is still an in-sample "
            "statistic, measured on the data used to fit the model.</li>"
            "</ul>"
        ),
        unsafe_allow_html=True,
    )

    adj_verb = "also improves" if delta_adj_r2 > 0 else "falls" if delta_adj_r2 < 0 else "stays flat"
    st.markdown(
        f'<p class="mr-compare">On this {"illustrative" if scenario == "strong" else "real"} data, '
        f"adding attendance {'raises' if delta_r2 > 0 else 'does not raise'} R² by "
        f"<b>{delta_r2:.4f}</b>, and adjusted R² {adj_verb} "
        f"(<b>{delta_adj_r2:+.4f}</b>).</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="mr-compare">Does attendance provide a '
        "<b>statistically significant</b> improvement, or could the R² gain "
        "be due to chance? The F-test lesson answers this precisely.</p>",
        unsafe_allow_html=True,
    )



MR_SUBTABS: tuple[tuple[str, str], ...] = (
    ("why", "Why & The Equation"),
    ("fit", "Fit the Plane"),
    ("coef", "Holding Other Predictors Constant"),
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
