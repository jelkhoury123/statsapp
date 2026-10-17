"""Targeted tests for Dataset Manager: CSV/Excel, R catalogue, validation, OLS, session."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from src.dataset_import.analyse import (
    build_preview,
    missing_value_summary,
    numeric_column_names,
    pair_complete_count,
)
from src.dataset_import.io_files import parse_uploaded_table, read_excel_sheets
from src.dataset_import.r_catalogue import (
    is_r_registry_key,
    load_r_frame,
    r_registry_key,
    search_r_datasets,
)
from src.dataset_import.registry import (
    activate_dataset,
    download_csv_bytes,
    ensure_registry_state,
    get_active_fitting_dataset,
    is_synthetic_dataset,
    library_options,
    register_upload,
    resolve_fitting_dataset,
    unified_dataset_options,
)
from src.fitting_line import ols_fit, r_squared, sum_of_squared_errors, total_sum_of_squares
from src.regression import fit_ols


def _session() -> dict:
    state: dict = {}
    ensure_registry_state(state)
    return state


def test_parse_csv_upload_and_numeric_columns():
    csv = b"x,y,label\n1,2,a\n3,4,b\n5,6,c\n"
    parsed = parse_uploaded_table(filename="toy.csv", data=csv)
    assert parsed.name == "toy"
    assert list(parsed.frame.columns) == ["x", "y", "label"]
    assert numeric_column_names(parsed.frame) == ["x", "y"]
    assert pair_complete_count(parsed.frame, "x", "y") == 3


def test_parse_excel_with_worksheet_selection():
    frame_a = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    frame_b = pd.DataFrame({"speed": [10, 20], "dist": [5, 15], "note": ["x", "y"]})
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame_a.to_excel(writer, sheet_name="SheetA", index=False)
        frame_b.to_excel(writer, sheet_name="Cars", index=False)
    data = buffer.getvalue()

    sheets = read_excel_sheets(data)
    assert sheets == ["SheetA", "Cars"]

    parsed = parse_uploaded_table(filename="book.xlsx", data=data, sheet_name="Cars")
    assert parsed.active_sheet == "Cars"
    assert numeric_column_names(parsed.frame) == ["speed", "dist"]
    assert pair_complete_count(parsed.frame, "speed", "dist") == 2


def test_r_dataset_selection_bundled_locally():
    options = search_r_datasets("mtcars")
    assert any(key == "r_mtcars" for key, _ in options)
    assert is_r_registry_key("r_mtcars")
    frame = load_r_frame("mtcars")
    assert "mpg" in frame.columns and "wt" in frame.columns
    assert "rownames" not in frame.columns
    assert len(frame) == 32

    source = resolve_fitting_dataset("r_mtcars", x_name="wt", y_name="mpg")
    assert source.simulated is False
    assert source.n == 32
    assert source.x_name == "wt"
    assert source.y_name == "mpg"
    assert "population" not in (source.caveat or "").lower() or "no known" in (
        source.caveat or ""
    ).lower()


def test_airquality_missing_values_and_valid_pairs():
    frame = load_r_frame("airquality")
    missing = missing_value_summary(frame)
    assert missing.get("Ozone", 0) > 0
    preview = build_preview(
        name="airquality",
        description="NY air quality",
        frame=frame,
    )
    assert preview.n_rows == len(frame)
    assert len(preview.preview) <= 10

    complete = pair_complete_count(frame, "Temp", "Ozone")
    assert complete < len(frame)
    assert complete >= 2

    source = resolve_fitting_dataset("r_airquality", x_name="Temp", y_name="Ozone")
    assert source.n == complete
    assert np.isfinite(source.x).all() and np.isfinite(source.y).all()


def test_variable_validation_rejects_same_xy_and_non_numeric():
    session = _session()
    frame = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "z": ["a", "b", "c"]})
    key = register_upload(session, name="tiny", frame=frame, source="test")
    with pytest.raises(ValueError, match="different"):
        activate_dataset(session, key=key, x_name="x", y_name="x")
    with pytest.raises(ValueError, match="numeric"):
        activate_dataset(session, key=key, x_name="x", y_name="z")
    with pytest.raises(ValueError, match="at least two"):
        register_upload(
            session,
            name="one_num",
            frame=pd.DataFrame({"x": [1, 2], "label": ["a", "b"]}),
            source="test",
        )


def test_regression_on_imported_csv_matches_engine():
    session = _session()
    rng = np.random.default_rng(7)
    x = rng.normal(size=40)
    y = 2.5 + 1.5 * x + rng.normal(scale=0.2, size=40)
    frame = pd.DataFrame({"predictor": x, "outcome": y})
    key = register_upload(session, name="sim", frame=frame, source="unit test")
    activate_dataset(session, key=key, x_name="predictor", y_name="outcome")

    source = get_active_fitting_dataset(session)
    assert source.key == key
    assert source.simulated is False
    assert is_synthetic_dataset(key) is False

    ols = ols_fit(source.x, source.y)
    engine = fit_ols(source.x, source.y)
    assert abs(ols.alpha - engine.alpha) < 1e-12
    assert abs(ols.beta - engine.beta) < 1e-12
    sse = sum_of_squared_errors(source.x, source.y, ols.alpha, ols.beta)
    sst = total_sum_of_squares(source.y)
    r2 = r_squared(sse, sst)
    assert abs(sse - engine.sse) < 1e-9
    assert sst > 0
    assert 0.0 <= r2 <= 1.0
    assert abs(ols.beta - 1.5) < 0.15


def test_session_persistence_of_upload_and_xy():
    session = _session()
    frame = pd.DataFrame({"hours": [1.0, 2.0, 3.0, 4.0], "score": [50.0, 55.0, 61.0, 70.0]})
    key = register_upload(session, name="study", frame=frame, source="upload")
    activate_dataset(session, key=key, x_name="hours", y_name="score")

    assert session["fit_dataset_key"] == key
    assert session["fit_x"] == "hours"
    assert session["fit_y"] == "score"
    assert key in session["dm_uploads"]

    # Simulate a Streamlit rerun that keeps session_state.
    again = get_active_fitting_dataset(session)
    assert again.n == 4
    assert again.x_name == "hours"
    csv_bytes = download_csv_bytes(key, session)
    assert b"hours" in csv_bytes and b"score" in csv_bytes


def test_synthetic_vs_observed_flags():
    assert is_synthetic_dataset("students") is True
    assert is_synthetic_dataset(r_registry_key("cars")) is False
    session = _session()
    key = register_upload(
        session,
        name="obs",
        frame=pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}),
        source="upload",
    )
    assert is_synthetic_dataset(key) is False
    observed = resolve_fitting_dataset("r_cars")
    assert observed.simulated is False
    students = resolve_fitting_dataset("students")
    assert students.simulated is True


def test_r_search_filters():
    all_opts = search_r_datasets("")
    assert len(all_opts) >= 8
    faith = search_r_datasets("faithful")
    assert len(faith) == 1 and faith[0][0] == "r_faithful"
    assert search_r_datasets("zzz_no_match") == []


def test_unified_selector_includes_lab_r_and_uploads():
    session = _session()
    keys = dict(unified_dataset_options(session))
    assert "students" in keys
    assert "r_mtcars" in keys
    assert keys["students"].startswith("Examples ·")
    assert keys["r_mtcars"].startswith("R datasets ·")
    upload_key = register_upload(
        session,
        name="extra",
        frame=pd.DataFrame({"u": [1.0, 2.0], "v": [3.0, 4.0]}),
        source="test",
    )
    keys_after = dict(unified_dataset_options(session))
    assert upload_key in keys_after
    assert keys_after[upload_key].startswith("My datasets ·")


def test_library_import_does_not_activate():
    session = _session()
    session["fit_dataset_key"] = "students"
    session["fit_x"] = "hours_studied"
    session["fit_y"] = "exam_score"
    key = register_upload(
        session,
        name="parked",
        frame=pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}),
        source="library import",
    )
    assert session["fit_dataset_key"] == "students"
    assert key in session["dm_uploads"]
    lab = library_options(session, group="Examples", query="student")
    assert any(k == "students" for k, _ in lab)
    imported = library_options(session, group="My datasets", query="")
    assert any(k == key for k, _ in imported)
    # Legacy group name still resolves.
    assert library_options(session, group="Imported", query="") == imported
    r_opts = library_options(session, group="R datasets", query="airquality")
    assert r_opts and r_opts[0][0] == "r_airquality"
