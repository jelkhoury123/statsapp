"""A short visual bridge from fitting and sampling to testing a slope."""
from functools import lru_cache
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.charts import BLUE, FONT, GREY, INK, ORANGE
from src.regression import fit_ols_many

# Per-step reading time. Step 3 ("Even then, sample slopes vary around
# zero") carries a two-sentence caption, denser than the others, so it
# gets extra time; the rest share the app-wide 5.5s pace. The last entry
# is unused (step 4 is the final frame and never auto-advances).
HOLD_SECONDS = (5.5, 5.5, 5.5, 8.0, 5.5)
TITLES = (
    "We fitted the best line for our sample",
    "A different sample gives a different slope",
    "What if the true slope were zero?",
    "Even then, sample slopes vary around zero",
    "Could our nonzero slope be sampling variation?",
)

@lru_cache(maxsize=1)
def intro_data():
    """Fixed-X Gaussian experiment with a true slope of exactly zero.

    All lines and histogram values come from the same repeated-sampling
    experiment. No estimated population slope is substituted for the null.
    """
    rng = np.random.default_rng(20261028)
    x = np.linspace(0.5, 9.5, 40)
    scores = 45.0 + rng.normal(0, 6, size=(600, x.size))
    alphas, betas = fit_ols_many(np.broadcast_to(x, scores.shape), scores)
    se = 6.0 / np.sqrt(np.sum((x - x.mean()) ** 2))
    # A visible but plausible nonzero estimate, not an extreme outlier.
    index = int(np.flatnonzero((betas > 1.3 * se) & (betas < 1.7 * se))[0])
    return x, scores, alphas, betas, index, float(se)


def intro_figure(step: int) -> go.Figure:
    """Steps 0-2 build up one scatter (our sample, other samples, the null
    line); steps 3-4 switch to the sampling distribution of β̂ under that
    null, built from the same 600 simulated samples."""
    x, scores, alphas, betas, index, se = intro_data()
    observed = float(betas[index])
    fig = go.Figure()
    if step <= 2:
        fig.add_scatter(x=x, y=scores[index], mode="markers", name="Our sample",
                        marker=dict(color=BLUE, size=7, opacity=0.8))
        if step == 1:
            for j in range(12):
                fig.add_scatter(x=x, y=alphas[j] + betas[j] * x, mode="lines",
                                line=dict(color="rgba(37,99,235,0.22)", width=1.5),
                                name="Other samples", showlegend=j == 0, hoverinfo="skip")
        if step == 2:
            fig.add_scatter(x=[0, 10], y=[45, 45], mode="lines", name="If true β = 0",
                            line=dict(color=GREY, width=3, dash="dash"))
        fig.add_scatter(x=x, y=alphas[index] + observed * x, mode="lines",
                        name=f"Our fitted slope β<sup>ˆ</sup> = {observed:.2f}",
                        line=dict(color=ORANGE, width=3))
        fig.update_xaxes(title="Hours studied", range=[0, 10])
        fig.update_yaxes(title="Exam score", range=[20, 70])
    else:
        reach = 4 * se
        grid = np.linspace(-reach, reach, 300)
        density = np.exp(-0.5 * (grid / se) ** 2) / (se * np.sqrt(2 * np.pi))
        fig.add_histogram(x=betas, histnorm="probability density",
                          xbins=dict(start=-reach, end=reach, size=8 * se / 32),
                          marker_color=BLUE, opacity=0.35, name="600 sample slopes")
        fig.add_scatter(x=grid, y=density, mode="lines", name="Bell-shaped variation",
                        line=dict(color=BLUE, width=2.5))
        fig.add_vline(x=0, line=dict(color=GREY, width=2, dash="dash"))
        fig.add_annotation(x=0, y=1, yref="paper", text="True β = 0", showarrow=False,
                           yshift=18, font=dict(color=INK, size=13))
        if step == 4:
            fig.add_vline(x=observed, line=dict(color=ORANGE, width=3))
            fig.add_annotation(x=observed, y=0.72, yref="paper",
                               text=f"Our estimate<br>β<sup>ˆ</sup> = {observed:.2f}", showarrow=True,
                               ax=65, ay=-35, arrowcolor=ORANGE,
                               font=dict(color=ORANGE, size=14, family=FONT), bgcolor="white")
        fig.update_xaxes(title="Estimated slope β<sup>ˆ</sup> across samples", range=[-reach, reach])
        fig.update_yaxes(title="Density", range=[0, 1.25 * density.max()], showticklabels=False)
    fig.update_layout(
        height=380, margin=dict(l=50, r=24, t=35, b=65),
        font=dict(family=FONT, color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white",
        legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=12)),
        bargap=0.06, transition=dict(duration=450),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EEF2F7", zeroline=False, fixedrange=True)
    fig.update_yaxes(showgrid=True, gridcolor="#EEF2F7", zeroline=False, fixedrange=True)
    return fig


def _toggle_playback():
    st.session_state.tt_intro_playing = not st.session_state.tt_intro_playing
    st.session_state.tt_intro_tick = time.monotonic()


def _replay():
    st.session_state.tt_intro_step = 0
    st.session_state.tt_intro_playing = True
    st.session_state.tt_intro_tick = time.monotonic()


# Clock-based advancing (time.monotonic() ticks checked on a 0.5s fragment
# poll) rather than the sleep()+st.rerun() loop used by "Why a Model?".
# That pattern mixed several widgets (st.latex, st.columns) inside one
# manually-rerun container and could leave the previous step's content
# visible for a frame when the whole page reran very quickly; a fragment
# that only re-executes itself, checking a wall-clock tick each time, keeps
# the state transition and the redraw tied to the same elapsed time without
# relying on how fast a full-page rerun happens to land.
@st.fragment(run_every=0.5)
def render_ttest_intro():
    """Timed stages rerender independently, keeping navigation responsive."""
    state = st.session_state
    state.setdefault("tt_intro_step", 0)
    state.setdefault("tt_intro_playing", True)
    state.setdefault("tt_intro_tick", time.monotonic())
    step = int(state.tt_intro_step)
    if state.tt_intro_playing and step < 4 and time.monotonic() - state.tt_intro_tick >= HOLD_SECONDS[step]:
        step += 1
        state.tt_intro_step = step
        state.tt_intro_tick = time.monotonic()
    if step == 4:
        state.tt_intro_playing = False  # final frame: stop, wait for Replay

    st.markdown(f'<p class="fit-why-step-title">{TITLES[step]}</p>', unsafe_allow_html=True)
    st.plotly_chart(intro_figure(step), width="stretch", key="tt_intro_chart",
                    config={"displayModeBar": False, "displaylogo": False})
    with st.container(height=50, border=False):
        if step == 4:
            st.markdown('<p style="text-align:center;color:#C2410C;font-weight:700;">A nonzero estimate alone doesn\'t prove a nonzero true slope.</p>', unsafe_allow_html=True)
    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
        if step < 4:
            st.button("Pause ⏸" if state.tt_intro_playing else "Resume ▶", key="tt_intro_pause", on_click=_toggle_playback)
        st.button("Replay animation ↺" if step == 4 else "Replay ↺", key="tt_intro_replay", on_click=_replay)
        st.caption(f"{step + 1} / 5")
