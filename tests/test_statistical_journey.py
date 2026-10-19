"""Checks for The Statistical Journey map homepage."""

from pathlib import Path

import numpy as np

from src.regression import fit_ols
from src.statistical_journey import (
    GRID_ORDER,
    PLACEHOLDER_BUILDERS,
    PLACEHOLDER_KEYS,
    PLACEHOLDER_META,
    STAGE_KEYS,
    STAGE_META,
    build_journey_sample,
    flip_to_back,
    flip_to_front,
    stage_figure,
)


def test_journey_sample_is_reproducible():
    a = build_journey_sample(20261018)
    b = build_journey_sample(20261018)
    assert np.array_equal(a.hours, b.hours)
    assert np.array_equal(a.scores, b.scores)
    assert a.alpha == b.alpha
    assert a.beta == b.beta


def test_sample_ols_matches_fit():
    sample = build_journey_sample()
    fit = fit_ols(sample.hours, sample.scores)
    assert fit.alpha == sample.alpha
    assert fit.beta == sample.beta


def test_grid_reading_order():
    assert STAGE_KEYS == (
        "why",
        "build",
        "evaluate",
        "sampling_se",
        "test",
        "question",
        "use",
    )
    assert GRID_ORDER == (
        "why", "build", "evaluate",
        "sampling_se", "test", "question",
        "use", "multiple_regression", "f_tests",
    )
    assert len(GRID_ORDER) == 9
    assert set(GRID_ORDER) == set(STAGE_KEYS) | set(PLACEHOLDER_KEYS)


def test_placeholder_cards_have_no_dead_lesson_link():
    for key in PLACEHOLDER_KEYS:
        meta = PLACEHOLDER_META[key]
        assert meta["title"]
        assert meta["question"]
        assert "lesson" not in meta
    sample = build_journey_sample()
    for key in PLACEHOLDER_KEYS:
        fig = PLACEHOLDER_BUILDERS[key](sample)
        assert len(fig.data) >= 1
        assert fig.layout.height == 150


def test_stage_metadata_covers_all_stages():
    for stage in STAGE_KEYS:
        meta = STAGE_META[stage]
        assert meta["title"]
        assert meta["question"]
        assert meta["back"]
        assert 2 <= len(meta["concepts"]) <= 3
        assert all(isinstance(term, str) and term for term in meta["concepts"])
        assert meta["lesson"][0] in {"fitting", "sampling", "correlation", "ttest"}
        assert "takeaway" not in meta


def test_back_copy_and_key_concepts():
    assert "complex" in STAGE_META["why"]["back"]
    assert STAGE_META["why"]["concepts"] == (
        "Variation",
        "Relationships",
        "Statistical model",
    )
    assert "squared prediction errors" in STAGE_META["build"]["back"]
    assert STAGE_META["build"]["concepts"] == ("Sample", "Regression", "Least squares")
    assert STAGE_META["use"]["concepts"] == ("Fitted value", "Prediction", "Residual")
    assert STAGE_META["evaluate"]["concepts"] == ("SSE", "R²", "Residual SE")
    assert STAGE_META["test"]["concepts"] == ("Standard error", "t-test", "p-value")
    # "improve" is reserved content for the Part II (multiple regression) journey.
    assert STAGE_META["improve"]["concepts"] == (
        "Multiple regression",
        "Adjusted R²",
        "F-test",
    )
    assert STAGE_META["question"]["concepts"] == (
        "Correlation",
        "Causation",
        "Confounding",
    )
    text = Path("src/statistical_journey.py").read_text(encoding="utf-8")
    assert "journey-key-concepts" in text
    assert "journey-takeaway" not in text


def test_flip_state_helpers_use_session(monkeypatch):
    state: dict = {}

    class _Session(dict):
        def __getattr__(self, name):
            return self[name]

        def __setattr__(self, name, value):
            self[name] = value

    session = _Session()
    monkeypatch.setattr("src.statistical_journey.st.session_state", session)

    flip_to_back("build")
    assert session["journey_flipped"] == "build"
    flip_to_back("test")
    assert session["journey_flipped"] == "test"
    flip_to_front()
    assert session["journey_flipped"] is None


def test_lesson_targets_existing_tabs():
    assert STAGE_META["why"]["lesson"] == ("fitting", {"fit_lab_panel": "why"})
    assert STAGE_META["build"]["lesson"] == ("fitting", {"fit_lab_panel": "fit"})
    assert STAGE_META["evaluate"]["lesson"] == ("fitting", {"fit_lab_panel": "r2"})
    assert STAGE_META["use"]["lesson"] == ("fitting", {"fit_lab_panel": "use"})
    assert STAGE_META["sampling_se"]["lesson"] == (
        "sampling",
        {"sampling_inference_sub": "one_sample"},
    )
    assert STAGE_META["test"]["lesson"] == ("ttest", {})
    # "improve" is reserved content for the Part II (multiple regression) journey.
    assert STAGE_META["improve"]["lesson"] == ("correlation", {"corr_lesson": "control"})
    assert STAGE_META["question"]["lesson"] == ("correlation", {"corr_lesson": "causation"})


def test_all_mini_figures_build():
    sample = build_journey_sample()
    for stage in STAGE_KEYS:
        fig = stage_figure(stage, sample)
        assert len(fig.data) >= 1
        assert fig.layout.height == 150


def test_homepage_has_flip_controls():
    text = Path("src/statistical_journey.py").read_text(encoding="utf-8")
    assert "flip_to_back" in text
    assert "flip_to_front" in text
    assert "Explore lesson →" in text
    assert "journey_flipped" in text
    assert "_render_card_back" in text
    assert "_render_card_front" in text


def test_homepage_css_supports_flip_and_reduced_motion():
    text = Path("app.py").read_text(encoding="utf-8")
    assert "journeyFlipIn" in text
    assert "prefers-reduced-motion" in text
    assert "st-key-journey_explore_build" in text
    assert "st-key-journey_unflip_build" in text
