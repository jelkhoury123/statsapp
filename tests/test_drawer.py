"""The data drawer: same students in the table and on the chart."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from src.charts import introduction_figure
from src.data_drawer import (
    EXPORT_COLUMNS,
    R_COMMANDS,
    DrawerSnapshot,
    chart_student_id,
    dataframe_rows,
    frame_to_csv,
    highlight_for,
    highlights_for,
    page_for_student,
    population_csv,
    population_page,
    sample_count_label,
    sample_csv,
    sample_frame,
    sample_token,
    select_observation,
    selection_after_new_sample,
    set_drawer_open,
    state_after_chart_click,
    student_id_at,
    student_id_from_chart_point,
    toggle_student,
)
from src.introduction import IntroductionView, estimate_sample, reveal_population, select_sample
from src.population import DEFAULT_CSV_PATH, file_sha256, load_population
from src.regression import fit_ols


@pytest.fixture(scope="module")
def population():
    return load_population()


def _trace(figure, name):
    names = [trace.name for trace in figure.data]
    return figure.data[names.index(name)]


def test_population_table_matches_the_original_file(population):
    frame, match_count, page = population_page(population, sort_by="student_id", ascending=True)
    assert match_count == 100_000
    assert page == 0
    assert list(frame.columns) == list(EXPORT_COLUMNS)
    assert "individual_error" not in frame.columns
    order = np.argsort(population.student_id, kind="mergesort")
    expected = order[: len(frame)]
    assert frame["student_id"].tolist() == population.student_id[expected].tolist()
    assert frame["hours_studied"].tolist() == pytest.approx(population.hours_studied[expected].tolist())
    assert frame["exam_score"].tolist() == pytest.approx(population.exam_score[expected].tolist())

    found, count, _ = population_page(population, student_query=str(int(population.student_id[40])))
    assert count == 1
    assert int(found.iloc[0]["student_id"]) == int(population.student_id[40])
    assert found.iloc[0]["hours_studied"] == pytest.approx(population.hours_studied[40])


def test_sample_table_is_exactly_the_drawn_students(population):
    view = select_sample(IntroductionView(), population, 80, np.random.default_rng(4))
    frame = sample_frame(view.sample)
    assert len(frame) == 80
    assert len(frame["student_id"].unique()) == 80
    assert frame["student_id"].tolist() == view.sample.student_id.tolist()
    assert frame["hours_studied"].tolist() == pytest.approx(view.sample.hours_studied.tolist())
    assert "individual_error" not in frame.columns
    assert sample_count_label(80, population.n) == "80 students selected from 100,000"


def test_selected_student_id_maps_to_the_population_coordinates(population):
    view = select_sample(IntroductionView(), population, 30, np.random.default_rng(9))
    student_id = int(view.sample.student_id[3])
    spec = highlight_for(population, view.sample, None, student_id)
    position = int(np.flatnonzero(population.student_id == student_id)[0])
    assert spec.student_id == student_id
    assert spec.hours_studied == pytest.approx(population.hours_studied[position])
    assert spec.exam_score == pytest.approx(population.exam_score[position])
    assert spec.in_sample
    assert spec.hours_studied == pytest.approx(view.sample.hours_studied[3])
    assert spec.show_sample_residual is False

    outsider = int(population.student_id[0])
    if outsider in set(view.sample.student_id.tolist()):
        outsider = int(population.student_id[1])
    outside = highlight_for(population, view.sample, None, outsider)
    assert outside.in_sample is False


def test_table_selection_does_not_change_the_sample_or_the_fit(population):
    view = estimate_sample(select_sample(IntroductionView(), population, 40, np.random.default_rng(12)))
    ids = view.sample.student_id.copy()
    alpha, beta = view.fit.alpha, view.fit.beta
    chosen = int(view.sample.student_id[0])
    drawer = select_observation(DrawerSnapshot(True, "Sample"), chosen)
    highlight_for(population, view.sample, view.fit, drawer.selected_id)

    assert drawer.selected_id == chosen
    assert np.array_equal(view.sample.student_id, ids)
    assert view.fit.alpha == alpha
    assert view.fit.beta == beta
    assert student_id_at(sample_frame(view.sample), 0) == chosen


def test_prediction_and_sample_residual_use_the_fitted_line(population):
    view = estimate_sample(select_sample(IntroductionView(), population, 25, np.random.default_rng(15)))
    student_id = int(view.sample.student_id[4])
    spec = highlight_for(population, view.sample, view.fit, student_id)
    predicted = view.fit.alpha + view.fit.beta * spec.hours_studied
    assert spec.predicted == pytest.approx(predicted)
    assert spec.residual == pytest.approx(spec.exam_score - spec.predicted)
    assert spec.show_sample_residual

    outsider = next(
        int(student)
        for student in population.student_id[:50]
        if int(student) not in set(view.sample.student_id.tolist())
    )
    other = highlight_for(population, view.sample, view.fit, outsider)
    assert other.predicted == pytest.approx(view.fit.alpha + view.fit.beta * other.hours_studied)
    assert other.residual is None
    assert other.show_sample_residual is False


def test_selected_residual_is_drawn_at_the_students_hours(population):
    view = estimate_sample(select_sample(IntroductionView(), population, 20, np.random.default_rng(18)))
    spec = highlight_for(population, view.sample, view.fit, int(view.sample.student_id[2]))
    figure = introduction_figure(
        cloud_image="data:image/png;base64,aa",
        x_range=population.x_range,
        y_range=population.y_range,
        sample_x=view.sample.hours_studied,
        sample_y=view.sample.exam_score,
        sample_ids=view.sample.student_id,
        sample_alpha=view.fit.alpha,
        sample_beta=view.fit.beta,
        population_alpha=population.alpha,
        population_beta=population.beta,
        show_sample_line=True,
        show_population_line=False,
        selected_id=spec.student_id,
        selected_x=spec.hours_studied,
        selected_y=spec.exam_score,
        selected_in_sample=True,
        show_residual=True,
    )
    residual = _trace(figure, "Residual")
    assert list(residual.x) == pytest.approx([spec.hours_studied, spec.hours_studied])
    assert list(residual.y) == pytest.approx([spec.exam_score, spec.predicted])
    selected = _trace(figure, "Selected student")
    assert list(selected.x) == pytest.approx([spec.hours_studied])
    assert int(selected.customdata[0][0]) == spec.student_id


def test_several_selected_students_keep_their_coordinates(population):
    view = select_sample(IntroductionView(), population, 30, np.random.default_rng(19))
    chosen = [int(view.sample.student_id[1]), int(view.sample.student_id[4])]
    outsider = next(
        int(student)
        for student in population.student_id[:80]
        if int(student) not in set(chosen)
    )
    chosen.append(outsider)
    specs = highlights_for(population, view.sample, None, tuple(chosen))
    figure = introduction_figure(
        cloud_image="data:image/png;base64,aa",
        x_range=population.x_range,
        y_range=population.y_range,
        sample_x=view.sample.hours_studied,
        sample_y=view.sample.exam_score,
        sample_ids=view.sample.student_id,
        sample_alpha=None,
        sample_beta=None,
        population_alpha=population.alpha,
        population_beta=population.beta,
        show_sample_line=False,
        show_population_line=False,
        selected_ids=np.array([spec.student_id for spec in specs]),
        selected_xs=np.array([spec.hours_studied for spec in specs]),
        selected_ys=np.array([spec.exam_score for spec in specs]),
        selected_membership=np.array([spec.in_sample for spec in specs]),
        show_residual=False,
    )
    selected = _trace(figure, "Selected student")
    assert list(selected.x) == pytest.approx([spec.hours_studied for spec in specs])
    assert list(selected.y) == pytest.approx([spec.exam_score for spec in specs])
    assert [int(row[0]) for row in selected.customdata] == chosen
    assert "Residual" not in [trace.name for trace in figure.data]


def test_row_clicks_toggle_and_chart_clicks_add(population):
    chosen = toggle_student((), 10)
    chosen = toggle_student(chosen, 20)
    assert chosen == (10, 20)
    assert toggle_student(chosen, 10) == (20,)
    order = np.argsort(population.student_id, kind="mergesort")
    target = int(population.student_id[order[250]])
    assert page_for_student(population, target, "student_id", True, page_size=100) == 2
    assert page_for_student(population, -1, "student_id", True) is None


def test_csv_exports_contain_the_sampled_and_population_students(population):
    digest = file_sha256(DEFAULT_CSV_PATH)
    view = select_sample(IntroductionView(), population, 15, np.random.default_rng(21))
    sample_text = sample_csv(view.sample).decode("utf-8")
    sample_table = pd.read_csv(io.StringIO(sample_text))
    assert list(sample_table.columns) == list(EXPORT_COLUMNS)
    assert len(sample_table) == 15
    assert sample_table["student_id"].tolist() == view.sample.student_id.tolist()

    population_table = pd.read_csv(io.BytesIO(population_csv(population)))
    assert list(population_table.columns) == list(EXPORT_COLUMNS)
    assert len(population_table) == 100_000
    assert population_table["student_id"].tolist() == population.student_id.tolist()
    assert "lm" in R_COMMANDS and "exam_score ~ hours_studied" in R_COMMANDS
    assert file_sha256(DEFAULT_CSV_PATH) == digest
    assert frame_to_csv(sample_frame(view.sample)) == sample_csv(view.sample)


def test_opening_the_drawer_preserves_the_experiment(population):
    view = reveal_population(
        estimate_sample(select_sample(IntroductionView(), population, 12, np.random.default_rng(27)))
    )
    ids = view.sample.student_id.copy()
    alpha, beta = view.fit.alpha, view.fit.beta
    drawer = DrawerSnapshot(False, "Population", (int(ids[0]),))
    opened = set_drawer_open(drawer, True)
    closed = set_drawer_open(opened, False)
    clicked = state_after_chart_click(closed, int(ids[1]))

    assert opened.open and not closed.open
    assert closed.selected_id == int(ids[0])
    assert clicked.open and clicked.dataset == "Population"
    assert clicked.selected_ids == (int(ids[0]), int(ids[1]))
    assert clicked.selected_id == int(ids[1])
    assert np.array_equal(view.sample.student_id, ids)
    assert view.fit.alpha == alpha and view.fit.beta == beta
    assert view.population_revealed
    assert fit_ols(population.hours_studied, population.exam_score).alpha == pytest.approx(population.alpha)


def test_main_lab_tabs_cover_every_section():
    from app import (
        MAIN_LAB_TAB_LABELS,
        MAIN_LAB_TABS,
        SAMPLING_INFERENCE_SUB_LABELS,
        SAMPLING_INFERENCE_SUBTABS,
    )

    keys = [key for key, _ in MAIN_LAB_TABS]
    assert keys == [
        "journey",
        "fitting",
        "correlation",
        "sampling",
        "ttest",
        "library",
    ]
    assert "intro" not in keys
    assert "sample" not in keys
    assert "comparison" not in keys
    assert "noise" not in keys
    assert MAIN_LAB_TAB_LABELS["sampling"] == "Sampling & Standard Error"
    assert MAIN_LAB_TAB_LABELS["ttest"] == "The t-test"
    assert all(label.strip() for _, label in MAIN_LAB_TABS)
    assert [key for key, _ in SAMPLING_INFERENCE_SUBTABS] == [
        "one_sample",
        "repeated",
        "estimate_se",
    ]
    assert SAMPLING_INFERENCE_SUB_LABELS["one_sample"] == "One Sample"
    assert SAMPLING_INFERENCE_SUB_LABELS["repeated"] == "Repeated Sampling"
    assert SAMPLING_INFERENCE_SUB_LABELS["estimate_se"] == "Estimate SE from One Sample"
    assert "Population & one sample" not in MAIN_LAB_TAB_LABELS.values()
    assert "Sample size comparison" not in MAIN_LAB_TAB_LABELS.values()
    assert "Noise experiment" not in MAIN_LAB_TAB_LABELS.values()
    assert "Sampling distributions" not in MAIN_LAB_TAB_LABELS.values()


def test_dataset_status_line_format():
    from app import _dataset_status_html, _dataset_status_name
    from src.fitting_line import DATASET_CATALOGS

    html = _dataset_status_html(
        "Students",
        200,
        10_000,
        "Hours studied",
        "Exam score",
    )
    assert "Students · n = 200 / 10,000 · Hours studied → Exam score" in html
    assert _dataset_status_name(DATASET_CATALOGS["students"]) == "Students"
    assert _dataset_status_name(DATASET_CATALOGS["penguins"]) == "Penguins"


def test_another_sample_keeps_selected_population_students(population):
    rng = np.random.default_rng(33)
    first = select_sample(IntroductionView(), population, 50, rng)
    chosen = int(first.sample.student_id[0])
    second = select_sample(first, population, 50, rng)
    assert sample_token(first.sample) != sample_token(second.sample)
    assert set(sample_frame(second.sample)["student_id"]) == set(second.sample.student_id.tolist())
    assert set(sample_frame(first.sample)["student_id"]) != set(sample_frame(second.sample)["student_id"])
    kept = selection_after_new_sample((chosen,), second.sample)
    assert kept == (chosen,)
    assert selection_after_new_sample(None, second.sample) == ()


def test_chart_and_table_events_use_student_id_not_row_position():
    point = {"customdata": [1872], "point_index": 4, "x": 6.8, "y": 58.1}
    assert student_id_from_chart_point(point) == 1872
    assert chart_student_id({"selection": {"points": [point]}}) == 1872
    assert chart_student_id({"selection": {"points": [{"x": 1, "y": 2}]}}) is None
    frame = pd.DataFrame({"student_id": [10, 20, 30], "hours_studied": [1, 2, 3], "exam_score": [4, 5, 6]})
    assert dataframe_rows({"selection": {"rows": [2]}}) == [2]
    assert student_id_at(frame, 2) == 30
    assert student_id_at(frame, 9) is None
