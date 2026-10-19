"""The Statistical Journey — single-page roadmap homepage for Weeks 1–4."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from scipy.stats import f as fisher_f
from scipy.stats import t as student_t

from src.charts import BLUE, FONT, GREY, INK, ORANGE, MUTED, rgba
from src.regression import fit_ols
from src.stat_tips import tipify

JOURNEY_SEED = 20261018
SAMPLE_N = 28
MINI_HEIGHT = 195
MINI_CONFIG = {
    "displaylogo": False,
    "displayModeBar": False,
    "staticPlot": True,
    "responsive": True,
}

# The seven implemented lessons (clickable cards).
STAGE_KEYS = ("why", "build", "evaluate", "sampling_se", "test", "question", "use")
# "improve" is reserved content for the Part II (multiple regression) journey.

# Two coming-soon cards (Part II), presentational only — no lesson behind them yet.
PLACEHOLDER_KEYS = ("multiple_regression", "f_tests")
PLACEHOLDER_META = {
    "multiple_regression": {
        "title": "MULTIPLE REGRESSION",
        "question": "What if we have two predictors?",
    },
    "f_tests": {
        "title": "F-TESTS",
        "question": "Does adding predictors improve the model?",
    },
}

# Reading order of the homepage's 3×3 grid, left to right, top to bottom.
GRID_ORDER = (
    "why", "build", "evaluate",
    "sampling_se", "test", "question",
    "use", "multiple_regression", "f_tests",
)
# Color phase per real stage (background tint + title color — see app.py CSS).
_PHASE_BY_STAGE = {
    "why": "build", "build": "build", "evaluate": "build", "use": "build",
    "sampling_se": "eval", "test": "eval", "question": "eval",
}

STAGE_META = {
    "why": {
        "title": "WHY",
        "question": "Why do we need a statistical model?",
        "lesson": ("fitting", {"fit_lab_panel": "why"}),
        "link": "Explore lesson →",
        "back": (
            "Reality is complex, and many factors influence "
            "what happens. A statistical model helps us identify "
            "patterns in observations, understand relationships "
            "and make predictions."
        ),
        "concepts": ("Variation", "Relationships", "Statistical model"),
    },
    "build": {
        "title": "BUILD",
        "question": "How do we construct a model?",
        "lesson": ("fitting", {"fit_lab_panel": "fit"}),
        "link": "Explore lesson →",
        "back": (
            "We collect observations and look for a relationship "
            "between two variables. Linear regression finds the "
            "straight line that best fits the data by making the "
            "squared prediction errors as small as possible."
        ),
        "concepts": ("Sample", "Regression", "Least squares"),
    },
    "use": {
        "title": "USE",
        "question": "How can we make predictions?",
        "lesson": ("fitting", {"fit_lab_panel": "use"}),
        "link": "Explore lesson →",
        "back": (
            "Once fitted, a model helps us describe relationships "
            "and predict outcomes for new observations. "
            "Predictions remain uncertain."
        ),
        "concepts": ("Fitted value", "Prediction", "Residual"),
    },
    "evaluate": {
        "title": "EVALUATE",
        "question": "How good is our model?",
        "lesson": ("fitting", {"fit_lab_panel": "r2"}),
        "link": "Explore lesson →",
        "back": (
            "Some models fit observations better than others. "
            "We measure how much variation the model accounts for "
            "and how large its errors are."
        ),
        "concepts": ("SSE", "R²", "Residual SE"),
    },
    "sampling_se": {
        "title": "SAMPLING",
        "question": "Why do estimates vary between samples?",
        "lesson": ("sampling", {"sampling_inference_sub": "one_sample"}),
        "link": "Explore lesson →",
        "back": (
            "Draw a different sample and the estimated slope changes. The "
            "spread of these estimates across many samples — the standard "
            "error — tells us how much to trust a result from just one."
        ),
        "concepts": ("Sampling variability", "Standard error", "Repeated samples"),
    },
    "test": {
        "title": "TEST",
        "question": "Is the slope significantly different from zero?",
        "lesson": ("ttest", {}),
        "link": "Explore lesson →",
        "back": (
            "Every sample is different. An apparent relationship "
            "may arise from sampling noise. "
            "Statistical tests help us assess the evidence."
        ),
        "concepts": ("Standard error", "t-test", "p-value"),
    },
    "improve": {
        "title": "IMPROVE",
        "question": "Could another model be better?",
        "lesson": ("correlation", {"corr_lesson": "control"}),
        "link": "Explore lesson →",
        "back": (
            "Adding predictors may improve a model, but "
            "greater complexity does not guarantee better results. "
            "We compare models to assess meaningful improvements."
        ),
        "concepts": ("Multiple regression", "Adjusted R²", "F-test"),
    },
    "question": {
        "title": "QUESTION",
        "question": "Does correlation imply causation?",
        "lesson": ("correlation", {"corr_lesson": "causation"}),
        "link": "Explore lesson →",
        "back": (
            "Variables that move together do not necessarily cause "
            "one another. A strong statistical relationship does not "
            "establish cause and effect — other variables may "
            "explain it."
        ),
        "concepts": ("Correlation", "Causation", "Confounding"),
    },
}


@dataclass(frozen=True)
class JourneySample:
    """Compact synthetic sample shared by all miniature illustrations."""

    hours: np.ndarray
    scores: np.ndarray
    motivation: np.ndarray
    alpha: float
    beta: float
    scores_b: np.ndarray
    alpha_b: float
    beta_b: float


@lru_cache(maxsize=1)
def build_journey_sample(seed: int = JOURNEY_SEED) -> JourneySample:
    rng = np.random.default_rng(int(seed))
    hours = np.sort(rng.uniform(0.8, 9.2, size=SAMPLE_N))
    motivation = np.clip(2.5 + 0.4 * hours + rng.normal(0, 1.2, size=SAMPLE_N), 0.5, 10)
    scores = 20.0 + 4.5 * hours + 1.4 * motivation + rng.normal(0, 4.5, size=SAMPLE_N)
    scores = np.clip(scores, 12.0, 98.0)
    fit_a = fit_ols(hours, scores)
    scores_b = 16.0 + 3.2 * hours + 2.4 * motivation
    fit_b = fit_ols(hours, scores_b)
    return JourneySample(
        hours=np.asarray(hours, dtype=np.float64),
        scores=np.asarray(scores, dtype=np.float64),
        motivation=np.asarray(motivation, dtype=np.float64),
        alpha=float(fit_a.alpha),
        beta=float(fit_a.beta),
        scores_b=np.asarray(scores_b, dtype=np.float64),
        alpha_b=float(fit_b.alpha),
        beta_b=float(fit_b.beta),
    )


def _base_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        height=MINI_HEIGHT,
        margin=dict(l=4, r=4, t=4, b=4),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=9),
        showlegend=False,
        xaxis=dict(visible=False, showgrid=False, zeroline=False, fixedrange=True),
        yaxis=dict(visible=False, showgrid=False, zeroline=False, fixedrange=True),
    )
    return fig


def _scatter_cloud(sample: JourneySample, *, opacity: float = 0.85) -> go.Scatter:
    return go.Scatter(
        x=sample.hours,
        y=sample.scores,
        mode="markers",
        marker=dict(size=6, color=BLUE, opacity=opacity, line=dict(width=0)),
        hoverinfo="skip",
    )


def _ols_line(sample: JourneySample, *, color: str = BLUE, width: float = 2.2) -> go.Scatter:
    x = np.array([float(sample.hours.min()), float(sample.hours.max())])
    return go.Scatter(
        x=x,
        y=sample.alpha + sample.beta * x,
        mode="lines",
        line=dict(color=color, width=width),
        hoverinfo="skip",
    )


def mini_why(sample: JourneySample) -> go.Figure:
    """Raw observations, unmodeled — reality before simplification."""
    fig = go.Figure([_scatter_cloud(sample, opacity=0.85)])
    return _base_layout(fig)


def mini_build(sample: JourneySample) -> go.Figure:
    """Scatterplot with fitted regression line."""
    fig = go.Figure([_scatter_cloud(sample), _ols_line(sample, color=ORANGE)])
    return _base_layout(fig)


def mini_use(sample: JourneySample) -> go.Figure:
    """Regression line with a predicted point."""
    x0 = float(np.percentile(sample.hours, 70))
    y_hat = sample.alpha + sample.beta * x0
    i = int(np.argmin(np.abs(sample.hours - x0)))
    fig = go.Figure(
        [
            _scatter_cloud(sample, opacity=0.55),
            _ols_line(sample, color=BLUE),
            go.Scatter(
                x=[x0, x0],
                y=[float(sample.scores.min()) - 2, y_hat],
                mode="lines",
                line=dict(color=MUTED, width=1, dash="dot"),
                hoverinfo="skip",
            ),
            go.Scatter(
                x=[x0, sample.hours[i]],
                y=[y_hat, sample.scores[i]],
                mode="lines",
                line=dict(color=rgba(ORANGE, 0.9), width=1.6),
                hoverinfo="skip",
            ),
            go.Scatter(
                x=[x0],
                y=[y_hat],
                mode="markers",
                marker=dict(size=9, color=ORANGE, symbol="diamond"),
                hoverinfo="skip",
            ),
        ]
    )
    return _base_layout(fig)


def mini_evaluate(sample: JourneySample) -> go.Figure:
    """Regression line with residual segments."""
    y_hat = sample.alpha + sample.beta * sample.hours
    traces: list = [_scatter_cloud(sample, opacity=0.7), _ols_line(sample, color=BLUE)]
    step = max(1, SAMPLE_N // 10)
    for i in range(0, SAMPLE_N, step):
        traces.append(
            go.Scatter(
                x=[sample.hours[i], sample.hours[i]],
                y=[y_hat[i], sample.scores[i]],
                mode="lines",
                line=dict(color=rgba(ORANGE, 0.85), width=1.4),
                hoverinfo="skip",
            )
        )
    return _base_layout(go.Figure(traces))


def mini_sampling_se(sample: JourneySample) -> go.Figure:
    """Resampled fits fanning around the line — different samples, different slopes."""
    rng = np.random.default_rng(JOURNEY_SEED + 7)
    x = np.array([float(sample.hours.min()), float(sample.hours.max())])
    n = sample.hours.size
    traces: list = [_scatter_cloud(sample, opacity=0.5)]
    for _ in range(10):
        idx = rng.choice(n, size=n, replace=True)
        fit = fit_ols(sample.hours[idx], sample.scores[idx])
        if fit.degenerate:
            continue
        traces.append(
            go.Scatter(
                x=x,
                y=fit.alpha + fit.beta * x,
                mode="lines",
                line=dict(color=rgba(ORANGE, 0.35), width=1.2),
                hoverinfo="skip",
            )
        )
    traces.append(_ols_line(sample, color=BLUE, width=2.2))
    return _base_layout(go.Figure(traces))


def mini_test(sample: JourneySample) -> go.Figure:
    """Sampling distribution with a marked test statistic."""
    del sample
    rng = np.random.default_rng(JOURNEY_SEED + 3)
    df = 26
    draws = rng.standard_t(df, size=400)
    t_obs = 2.35
    xs = np.linspace(-4.2, 4.2, 160)
    dens = student_t.pdf(xs, df)
    fig = go.Figure(
        [
            go.Histogram(
                x=draws,
                histnorm="probability density",
                nbinsx=28,
                marker=dict(color=rgba(BLUE, 0.35), line=dict(width=0)),
                hoverinfo="skip",
            ),
            go.Scatter(
                x=xs,
                y=dens,
                mode="lines",
                line=dict(color=GREY, width=1.8),
                hoverinfo="skip",
            ),
            go.Scatter(
                x=[t_obs, t_obs],
                y=[0, float(student_t.pdf(t_obs, df))],
                mode="lines",
                line=dict(color=ORANGE, width=2.4),
                hoverinfo="skip",
            ),
        ]
    )
    fig.update_layout(
        height=MINI_HEIGHT,
        margin=dict(l=4, r=4, t=4, b=4),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        bargap=0.05,
        xaxis=dict(visible=False, fixedrange=True, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, fixedrange=True, showgrid=False, zeroline=False),
    )
    return fig


def mini_improve(sample: JourneySample) -> go.Figure:
    """Two competing model fits on the same scatter."""
    x = np.array([float(sample.hours.min()), float(sample.hours.max())])
    fig = go.Figure(
        [
            _scatter_cloud(sample, opacity=0.6),
            go.Scatter(
                x=x,
                y=sample.alpha + sample.beta * x,
                mode="lines",
                line=dict(color=BLUE, width=2.2),
                hoverinfo="skip",
            ),
            go.Scatter(
                x=x,
                y=sample.alpha_b + sample.beta_b * x,
                mode="lines",
                line=dict(color=ORANGE, width=2.2, dash="dash"),
                hoverinfo="skip",
            ),
        ]
    )
    return _base_layout(fig)


def mini_question(sample: JourneySample) -> go.Figure:
    """Two correlated variables coloured by a third (confounder)."""
    fig = go.Figure(
        go.Scatter(
            x=sample.hours,
            y=sample.scores,
            mode="markers",
            marker=dict(
                size=7,
                color=sample.motivation,
                colorscale="Teal",
                showscale=False,
                opacity=0.9,
                line=dict(width=0),
                cmin=float(np.min(sample.motivation)),
                cmax=float(np.max(sample.motivation)),
            ),
            hoverinfo="skip",
        )
    )
    # Soft OLS hint — association without claiming cause.
    fig.add_trace(_ols_line(sample, color=rgba(GREY, 0.55), width=1.6))
    return _base_layout(fig)


def mini_multiple_regression(sample: JourneySample) -> go.Figure:
    """Points sized and coloured by a second predictor — more than one driver of Y."""
    spread = float(sample.motivation.max() - sample.motivation.min()) or 1.0
    sizes = 5 + 9 * (sample.motivation - sample.motivation.min()) / spread
    fig = go.Figure(
        go.Scatter(
            x=sample.hours,
            y=sample.scores,
            mode="markers",
            marker=dict(
                size=sizes,
                color=sample.motivation,
                colorscale="Purples",
                showscale=False,
                opacity=0.85,
                line=dict(width=0.6, color="white"),
            ),
            hoverinfo="skip",
        )
    )
    fig.add_trace(_ols_line(sample, color=INK, width=1.8))
    return _base_layout(fig)


def mini_f_tests(sample: JourneySample) -> go.Figure:
    """F-distribution with a shaded rejection region — comparing nested models."""
    del sample
    dfn, dfd = 2, 30
    xs = np.linspace(0.001, 6, 160)
    dens = fisher_f.pdf(xs, dfn, dfd)
    crit = float(fisher_f.ppf(0.95, dfn, dfd))
    tail_x = xs[xs >= crit]
    tail_y = dens[xs >= crit]
    fig = go.Figure(
        [
            go.Scatter(
                x=np.concatenate([[crit], tail_x]),
                y=np.concatenate([[0.0], tail_y]),
                mode="lines",
                fill="tozeroy",
                line=dict(width=0),
                fillcolor=rgba(ORANGE, 0.35),
                hoverinfo="skip",
            ),
            go.Scatter(x=xs, y=dens, mode="lines", line=dict(color=GREY, width=1.8), hoverinfo="skip"),
            go.Scatter(
                x=[crit, crit],
                y=[0, float(fisher_f.pdf(crit, dfn, dfd))],
                mode="lines",
                line=dict(color=ORANGE, width=2.2),
                hoverinfo="skip",
            ),
        ]
    )
    return _base_layout(fig)


PLACEHOLDER_BUILDERS = {
    "multiple_regression": mini_multiple_regression,
    "f_tests": mini_f_tests,
}


MINI_BUILDERS = {
    "why": mini_why,
    "build": mini_build,
    "use": mini_use,
    "evaluate": mini_evaluate,
    "test": mini_test,
    "improve": mini_improve,
    "question": mini_question,
    "sampling_se": mini_sampling_se,
}


def stage_figure(stage: str, sample: JourneySample | None = None) -> go.Figure:
    sample = sample or build_journey_sample()
    return MINI_BUILDERS[stage](sample)


def open_lesson(stage: str) -> None:
    """Navigate to a lesson. Intended for button ``on_click`` (before widgets rerun)."""
    st.session_state.journey_flipped = None
    tab, updates = STAGE_META[stage]["lesson"]
    st.session_state.main_lab_tab = tab
    # Keep the lesson nav row's own widget state in sync with the target tab.
    st.session_state.main_lab_lesson_tab = tab
    for key, value in updates.items():
        st.session_state[key] = value


def flip_to_back(stage: str) -> None:
    """Show this stage's back; closes any other flipped card."""
    st.session_state.journey_flipped = stage


def flip_to_front() -> None:
    """Return the flipped card to its front face."""
    st.session_state.journey_flipped = None


def _render_card_front(stage: str, sample: JourneySample, *, phase: str) -> None:
    meta = STAGE_META[stage]
    # Hit-target first so the overlay stacks above the chart.
    st.button(
        f"Flip {meta['title']} card",
        key=f"journey_flip_{stage}",
        type="tertiary",
        on_click=flip_to_back,
        args=(stage,),
        help=f"Show why {meta['title'].title()} matters",
    )
    st.markdown(
        f'<div class="journey-is-front" aria-hidden="true"></div>'
        f'<div class="journey-stage journey-phase-{phase}">'
        f'<p class="journey-stage-title">{meta["title"]}</p>'
        f'<p class="journey-stage-q">{meta["question"]}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        stage_figure(stage, sample),
        width="stretch",
        config=MINI_CONFIG,
        key=f"journey_mini_{stage}",
    )


def _concepts_line(concepts: tuple[str, ...]) -> str:
    terms = " · ".join(concepts)
    return (
        f'<p class="journey-key-concepts">'
        f"<span>Key concepts</span> {terms}"
        f"</p>"
    )


def _render_card_back(stage: str, *, phase: str) -> None:
    meta = STAGE_META[stage]
    # Unflip hit-target behind the visible Explore button.
    st.button(
        f"Flip {meta['title']} card back",
        key=f"journey_unflip_{stage}",
        type="tertiary",
        on_click=flip_to_front,
        help="Return to the chart",
    )
    st.markdown(
        tipify(
            f'<div class="journey-is-back" aria-hidden="true"></div>'
            f'<div class="journey-stage journey-phase-{phase} journey-back">'
            f'<p class="journey-stage-title">{meta["title"]}</p>'
            f'<p class="journey-back-body">{meta["back"]}</p>'
            f'{_concepts_line(meta["concepts"])}'
            f"</div>"
        ),
        unsafe_allow_html=True,
    )
    st.button(
        "Explore lesson →",
        key=f"journey_explore_{stage}",
        type="primary",
        on_click=open_lesson,
        args=(stage,),
    )


def _render_stage_card(stage: str, sample: JourneySample, *, phase: str) -> None:
    flipped = st.session_state.get("journey_flipped") == stage
    if flipped:
        _render_card_back(stage, phase=phase)
    else:
        _render_card_front(stage, sample, phase=phase)


def _render_placeholder_card(stage: str, sample: JourneySample) -> None:
    meta = PLACEHOLDER_META[stage]
    st.markdown(
        '<div class="journey-stage journey-phase-soon">'
        f'<p class="journey-stage-title">{meta["title"]}</p>'
        f'<p class="journey-stage-q">{meta["question"]}</p>'
        '<span class="journey-soon-badge">Coming soon</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        PLACEHOLDER_BUILDERS[stage](sample),
        width="stretch",
        config=MINI_CONFIG,
        key=f"journey_mini_{stage}",
    )


def _render_grid(sample: JourneySample) -> None:
    """Nine equally sized cards, three per row, in GRID_ORDER."""
    for start in range(0, len(GRID_ORDER), 3):
        cols = st.columns(3, gap="medium")
        for col, stage in zip(cols, GRID_ORDER[start : start + 3]):
            with col:
                with st.container(key=f"journey_card_{stage}"):
                    if stage in PLACEHOLDER_KEYS:
                        _render_placeholder_card(stage, sample)
                    else:
                        _render_stage_card(stage, sample, phase=_PHASE_BY_STAGE[stage])


def _mount_card_click_bridge() -> None:
    """Delegate clicks anywhere on a journey card to its flip/unflip button.

    Streamlit tertiary buttons do not reliably fill an absolute overlay; this
    keeps the front visual unchanged while making the whole card clickable.
    """
    components.html(
        """
        <script>
        (() => {
          const doc = window.parent.document;
          if (doc.documentElement.dataset.journeyFlipBridge === "1") return;
          doc.documentElement.dataset.journeyFlipBridge = "1";
          doc.addEventListener("click", (event) => {
            const card = event.target.closest('[class*="st-key-journey_card_"]');
            if (!card) return;
            if (event.target.closest('[class*="st-key-journey_explore_"]')) return;
            const btn = card.querySelector(
              '[class*="st-key-journey_flip_"] button, [class*="st-key-journey_unflip_"] button'
            );
            if (!btn || btn.contains(event.target) || event.target === btn) return;
            event.preventDefault();
            event.stopPropagation();
            btn.click();
          }, true);
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def render_statistical_journey() -> None:
    """Nine-card grid overview of the full statistical journey."""
    if "journey_flipped" not in st.session_state:
        st.session_state.journey_flipped = None
    sample = build_journey_sample()
    _mount_card_click_bridge()

    st.markdown(
        tipify(
            '<div class="journey-hero">'
            "<h1>The Statistical Journey</h1>"
            "<p class=\"journey-lead\">We build models to simplify reality, understand "
            "relationships and predict outcomes.</p>"
            '<p class="journey-hint">Click any card to explore that step.</p>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    _render_grid(sample)
