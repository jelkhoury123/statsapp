"""Check that the visual story and its data tell the same null-slope story."""
import numpy as np
from streamlit.testing.v1 import AppTest

from src.regression import fit_ols
from src.ttest_intro import HOLD_SECONDS, intro_data, intro_figure


def test_null_experiment_and_observed_line_agree():
    x, scores, alphas, betas, index, se = intro_data()
    fitted = fit_ols(x, scores[index])
    assert np.isclose(fitted.beta, betas[index])
    assert np.isclose(fitted.alpha, alphas[index])
    assert abs(betas.mean()) < 0.15 * se
    assert np.isclose(betas.std(ddof=1), se, rtol=0.1)
    assert 1.3 * se < betas[index] < 1.7 * se
    fig = intro_figure(4)
    assert np.array_equal(fig.data[0].x, betas)
    assert np.isclose(fig.layout.shapes[-1].x0, fitted.beta)


def test_all_stages_render_and_playback_controls_work():
    app = AppTest.from_string('from src.ttest_intro import render_ttest_intro\nrender_ttest_intro()')
    for step in range(5):
        app.session_state['tt_intro_step'] = step
        app.session_state['tt_intro_playing'] = False
        app.run()
        assert not app.exception
        assert len(app.get('plotly_chart')) == 1
    app.button(key='tt_intro_replay').click().run()
    assert app.session_state['tt_intro_step'] == 0
    assert app.session_state['tt_intro_playing']
    app.button(key='tt_intro_pause').click().run()
    assert not app.session_state['tt_intro_playing']
    app.session_state['tt_intro_tick'] = -HOLD_SECONDS[0]
    app.run()
    assert app.session_state['tt_intro_step'] == 0
    app.button(key='tt_intro_pause').click().run()
    app.session_state['tt_intro_tick'] = -HOLD_SECONDS[0]
    app.run()
    assert app.session_state['tt_intro_step'] == 1
