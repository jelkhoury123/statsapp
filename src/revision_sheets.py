"""Revision Sheets: one PDF revision sheet per week, each with a quiz.

Utility tab (like Dataset Library) — not part of the Statistical Journey's
lesson grid. Quiz state lives only in st.session_state: no database, no
persistence across sessions.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import streamlit as st

from src.quiz_data import QUIZZES

_SHEETS_DIR = Path(__file__).resolve().parent.parent / "data" / "revision_sheets"


class _Sheet(TypedDict):
    key: str
    title: str
    file: str


_SHEETS: tuple[_Sheet, ...] = (
    {"key": "w1", "title": "Week 1 — Introduction to Linear Regression", "file": "w1_linear_regression.pdf"},
    {"key": "w2", "title": "Week 2 — Interpreting Linear Models", "file": "w2_interpreting_linear_models.pdf"},
    {"key": "w3", "title": "Week 3 — Testing & Evaluating Linear Models", "file": "w3_testing_evaluating.pdf"},
    {"key": "w4", "title": "Week 4 — F-tests & Model Comparison", "file": "w4_f_tests_model_comparison.pdf"},
)
_SHEET_TITLES = {sheet["key"]: sheet["title"] for sheet in _SHEETS}


def _state_key(week: str, suffix: str) -> str:
    return f"rs_quiz_{week}_{suffix}"


def _reset_quiz_state(week: str) -> None:
    for suffix in ("idx", "score", "answers", "revealed"):
        st.session_state.pop(_state_key(week, suffix), None)


def _render_sheet_card(sheet: _Sheet) -> None:
    path = _SHEETS_DIR / sheet["file"]
    st.markdown(f'<p class="rs-card-title">{sheet["title"]}</p>', unsafe_allow_html=True)
    try:
        st.pdf(str(path), height=380)
    except Exception:
        st.markdown(
            '<p class="rs-note">PDF preview unavailable in this environment '
            "— use Download below.</p>",
            unsafe_allow_html=True,
        )
    col1, col2 = st.columns(2, gap="small")
    with col1:
        if path.exists():
            st.download_button(
                "Download PDF",
                data=path.read_bytes(),
                file_name=sheet["file"],
                mime="application/pdf",
                key=f"rs_download_{sheet['key']}",
                width="stretch",
            )
    with col2:
        if st.button(
            "Quiz", key=f"rs_quiz_btn_{sheet['key']}", type="primary", width="stretch",
        ):
            st.session_state.rs_active_quiz = sheet["key"]
            _reset_quiz_state(sheet["key"])
            st.rerun()


def _render_quiz_question(week: str, idx: int, n: int) -> None:
    question = QUIZZES[week][idx]
    revealed_key = _state_key(week, "revealed")
    answers_key = _state_key(week, "answers")
    answers = st.session_state.get(answers_key, [])
    revealed = st.session_state.get(revealed_key, False)

    st.progress(idx / n)
    st.markdown(
        f'<p class="rs-quiz-progress">Question {idx + 1} of {n}</p>'
        f'<p class="rs-quiz-q">{question["q"]}</p>',
        unsafe_allow_html=True,
    )

    choice_key = f"rs_choice_{week}_{idx}"
    selected = st.radio(
        "Choose an answer",
        options=list(range(len(question["choices"]))),
        format_func=lambda i: question["choices"][i],
        key=choice_key,
        label_visibility="collapsed",
        index=None,
        disabled=revealed,
    )

    if not revealed:
        if st.button(
            "Submit Answer", key=f"rs_submit_{week}_{idx}",
            type="primary", disabled=selected is None,
        ):
            st.session_state[answers_key] = answers + [
                {
                    "q": question["q"],
                    "choices": question["choices"],
                    "chosen": selected,
                    "correct": question["correct"],
                    "explanation": question["explanation"],
                }
            ]
            if selected == question["correct"]:
                score_key = _state_key(week, "score")
                st.session_state[score_key] = st.session_state.get(score_key, 0) + 1
            st.session_state[revealed_key] = True
            st.rerun()
        return

    last = st.session_state[answers_key][-1]
    is_correct = last["chosen"] == last["correct"]
    css_class = "correct" if is_correct else "incorrect"
    label = "Correct" if is_correct else "Not quite"
    st.markdown(
        f'<div class="rs-feedback {css_class}">'
        f'<p class="title">{label}</p>'
        f'<p class="body">Correct answer: <b>{question["choices"][question["correct"]]}</b><br>'
        f'{question["explanation"]}</p></div>',
        unsafe_allow_html=True,
    )
    if st.button(
        "Next Question" if idx + 1 < n else "See Results",
        key=f"rs_next_{week}_{idx}", type="primary",
    ):
        st.session_state[_state_key(week, "idx")] = idx + 1
        st.session_state[revealed_key] = False
        st.rerun()


def _render_quiz_results(week: str, title: str, n: int) -> None:
    score = st.session_state.get(_state_key(week, "score"), 0)
    answers = st.session_state.get(_state_key(week, "answers"), [])
    incorrect = [a for a in answers if a["chosen"] != a["correct"]]

    st.markdown(
        f'<p class="rs-quiz-score">Score: {score} / {n}</p>',
        unsafe_allow_html=True,
    )
    if incorrect:
        st.markdown(
            '<p class="rs-quiz-review-head">Review your incorrect answers</p>',
            unsafe_allow_html=True,
        )
        for i, a in enumerate(incorrect, start=1):
            with st.expander(f"{i}. {a['q']}"):
                chosen_text = a["choices"][a["chosen"]] if a["chosen"] is not None else "—"
                st.markdown(
                    f"**Your answer:** {chosen_text}  \n"
                    f"**Correct answer:** {a['choices'][a['correct']]}  \n\n"
                    f"{a['explanation']}"
                )
    else:
        st.markdown(
            '<p class="rs-quiz-review-head">Perfect score — no incorrect answers to review.</p>',
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2, gap="small")
    with col1:
        if st.button("Retry Quiz", key=f"rs_retry_{week}", type="primary", width="stretch"):
            _reset_quiz_state(week)
            st.rerun()
    with col2:
        if st.button("Back to Revision Sheets", key=f"rs_done_back_{week}", width="stretch"):
            st.session_state.rs_active_quiz = None
            st.rerun()


def _render_quiz(week: str) -> None:
    title = _SHEET_TITLES[week]
    questions = QUIZZES[week]
    n = len(questions)

    if st.button("← Back to Revision Sheets", key=f"rs_back_{week}"):
        st.session_state.rs_active_quiz = None
        st.rerun()

    st.markdown(f'<p class="rs-card-title">{title} — Quiz</p>', unsafe_allow_html=True)

    idx = st.session_state.get(_state_key(week, "idx"), 0)
    if idx >= n:
        _render_quiz_results(week, title, n)
    else:
        _render_quiz_question(week, idx, n)


def render_revision_sheets_tab() -> None:
    """Weekly revision-sheet PDFs, each paired with a short self-test quiz."""
    st.markdown("**Revision Sheets**")
    st.caption(
        "Weekly revision sheets with a short quiz for each — answer one "
        "question at a time, see the explanation, and retry as often as "
        "you like. Nothing here is saved between visits."
    )

    active = st.session_state.get("rs_active_quiz")
    if active in QUIZZES:
        _render_quiz(active)
        return

    cols = st.columns(2, gap="large")
    for i, sheet in enumerate(_SHEETS):
        with cols[i % 2]:
            _render_sheet_card(sheet)
