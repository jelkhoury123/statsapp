"""Session registry unifying built-in, R, and uploaded datasets."""

from __future__ import annotations

import io
import uuid
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from src.dataset_import.analyse import numeric_column_names, pair_complete_count
from src.dataset_import.r_catalogue import (
    is_r_registry_key,
    load_r_frame,
    r_dataset_catalogs,
    r_name_from_key,
    search_r_datasets,
)
from src.fitting_line import (
    DATASET_CATALOGS,
    DATASET_OPTIONS,
    DatasetCatalog,
    FittingDataset,
    VariableSpec,
    dataset_option_label,
    load_fitting_dataset,
)


class DatasetSourceKind(str, Enum):
    BUILTIN = "builtin"
    R = "r"
    UPLOAD = "upload"


SOURCE_GROUP_LABELS: dict[DatasetSourceKind, str] = {
    DatasetSourceKind.BUILTIN: "Examples",
    DatasetSourceKind.R: "R datasets",
    DatasetSourceKind.UPLOAD: "My datasets",
}

# Legacy library-tab labels → current origin labels.
_LIBRARY_GROUP_ALIASES: dict[str, str] = {
    "Regression Lab": "Examples",
    "Built-in": "Examples",
    "Imported": "My datasets",
    "Upload": "My datasets",
}


def ensure_registry_state(session: dict[str, Any]) -> None:
    """Initialise session keys used by Dataset Manager."""
    session.setdefault("dm_uploads", {})  # key -> {name, frame, source, catalog fields}
    session.setdefault("dm_draft_key", None)
    session.setdefault("dm_draft_x", None)
    session.setdefault("dm_draft_y", None)
    session.setdefault("dm_source_tab", "Examples")
    session.setdefault("dm_r_query", "")
    session.setdefault("dm_view", "load")  # "load" | "library"
    session.setdefault("dm_lib_source", "Examples")
    session.setdefault("dm_lib_query", "")
    session.setdefault("dm_lib_key", None)
    # Migrate renamed library category labels from older sessions.
    raw_lib = session.get("dm_lib_source")
    if raw_lib in _LIBRARY_GROUP_ALIASES:
        session["dm_lib_source"] = _LIBRARY_GROUP_ALIASES[raw_lib]


def builtin_options() -> tuple[tuple[str, str], ...]:
    return DATASET_OPTIONS


def list_r_options(query: str = "") -> list[tuple[str, str]]:
    return search_r_datasets(query)


def source_group_label(key: str) -> str:
    return SOURCE_GROUP_LABELS[source_kind(key)]


def unified_dataset_options(session: dict[str, Any]) -> list[tuple[str, str]]:
    """Single selector list: Examples, R datasets, and My datasets."""
    ensure_registry_state(session)
    options: list[tuple[str, str]] = []
    for key, label in builtin_options():
        options.append((key, f"Examples · {label}"))
    for key, label in list_r_options(""):
        options.append((key, f"R datasets · {label}"))
    for key, label in upload_options(session):
        options.append((key, f"My datasets · {label}"))
    return options


def library_options(
    session: dict[str, Any],
    *,
    group: str,
    query: str = "",
) -> list[tuple[str, str]]:
    """Browse options for one library source group."""
    ensure_registry_state(session)
    group = _LIBRARY_GROUP_ALIASES.get(group, group)
    needle = (query or "").strip().lower()
    if group == "Examples":
        items = list(builtin_options())
    elif group == "R datasets":
        items = list_r_options(query)
        return items
    elif group == "My datasets":
        items = upload_options(session)
    else:
        items = []
    if not needle or group == "R datasets":
        return items
    out: list[tuple[str, str]] = []
    for key, label in items:
        try:
            catalog = get_catalog(key, session)
            hay = f"{key} {label} {catalog.short_name} {catalog.source}".lower()
        except KeyError:
            hay = f"{key} {label}".lower()
        if needle in hay:
            out.append((key, label))
    return out


def is_upload_key(key: str) -> bool:
    return str(key).startswith("upload_")


def is_synthetic_dataset(key: str) -> bool:
    """Only the simulated students pool has known population parameters."""
    if key in DATASET_CATALOGS:
        return bool(DATASET_CATALOGS[key].simulated)
    return False


def source_kind(key: str) -> DatasetSourceKind:
    if is_upload_key(key):
        return DatasetSourceKind.UPLOAD
    if is_r_registry_key(key):
        return DatasetSourceKind.R
    if key in DATASET_CATALOGS:
        return DatasetSourceKind.BUILTIN
    raise KeyError(f"Unknown dataset key: {key}")


def _upload_catalog(entry: dict[str, Any]) -> DatasetCatalog:
    variables = tuple(
        VariableSpec(name, name) for name in entry["numeric_columns"]
    )
    return DatasetCatalog(
        key=entry["key"],
        short_name=entry["name"],
        r_data=_safe_r_name(entry["name"]),
        source=entry["source"],
        variables=variables,
        default_x=entry["default_x"],
        default_y=entry["default_y"],
        caveat="Uploaded observed data — no known population α or β.",
        simulated=False,
    )


def get_catalog(key: str, session: dict[str, Any] | None = None) -> DatasetCatalog:
    if key in DATASET_CATALOGS:
        return DATASET_CATALOGS[key]
    if is_r_registry_key(key):
        return r_dataset_catalogs()[key]
    if is_upload_key(key):
        if session is None:
            raise KeyError("Upload datasets require session state")
        entry = session.get("dm_uploads", {}).get(key)
        if entry is None:
            raise KeyError(f"Unknown upload dataset: {key}")
        return _upload_catalog(entry)
    raise KeyError(f"Unknown dataset key: {key}")


def get_frame(key: str, session: dict[str, Any] | None = None) -> pd.DataFrame:
    if key in DATASET_CATALOGS:
        from src.fitting_line import _catalog_frame

        return _catalog_frame(DATASET_CATALOGS[key]).copy()
    if is_r_registry_key(key):
        return load_r_frame(r_name_from_key(key)).copy()
    if is_upload_key(key):
        if session is None:
            raise KeyError("Upload datasets require session state")
        entry = session.get("dm_uploads", {}).get(key)
        if entry is None:
            raise KeyError(f"Unknown upload dataset: {key}")
        return entry["frame"].copy()
    raise KeyError(f"Unknown dataset key: {key}")


def register_upload(
    session: dict[str, Any],
    *,
    name: str,
    frame: pd.DataFrame,
    source: str,
) -> str:
    """Store an uploaded frame in session; return registry key."""
    ensure_registry_state(session)
    numeric = numeric_column_names(frame)
    if len(numeric) < 2:
        raise ValueError(
            "Need at least two numeric columns to run simple linear regression."
        )
    key = f"upload_{uuid.uuid4().hex[:10]}"
    default_x, default_y = numeric[0], numeric[1]
    session["dm_uploads"][key] = {
        "key": key,
        "name": name or "upload",
        "frame": frame.copy(),
        "source": source,
        "numeric_columns": numeric,
        "default_x": default_x,
        "default_y": default_y,
    }
    return key


def resolve_fitting_dataset(
    key: str,
    *,
    x_name: str | None = None,
    y_name: str | None = None,
    session: dict[str, Any] | None = None,
    seed: int = 20261018,
) -> FittingDataset:
    """Load X/Y for any registry key using the shared OLS-ready FittingDataset."""
    if key in DATASET_CATALOGS:
        return load_fitting_dataset(key, seed, x_name=x_name, y_name=y_name)

    catalog = get_catalog(key, session)
    frame = get_frame(key, session)
    predictor = x_name or catalog.default_x
    outcome = y_name or catalog.default_y
    names = catalog.variable_names()
    if predictor not in names or outcome not in names:
        raise KeyError(f"Unknown variable for {key}: {predictor!r}, {outcome!r}")
    if predictor == outcome:
        raise ValueError("Predictor (X) and Outcome (Y) must be different variables")

    work = frame.reset_index(drop=True)
    numeric = work[[predictor, outcome]].apply(pd.to_numeric, errors="coerce")
    mask = numeric.notna().all(axis=1).to_numpy()
    if not bool(np.any(mask)):
        raise ValueError(f"No complete observations for {predictor} / {outcome}")
    row_indices = np.flatnonzero(mask)
    x = np.asarray(numeric.loc[mask, predictor], dtype=np.float64)
    y = np.asarray(numeric.loc[mask, outcome], dtype=np.float64)
    label = (
        dataset_option_label(key)
        if key in DATASET_CATALOGS
        else f"{catalog.short_name} (N = {len(frame):,})"
    )
    return FittingDataset(
        key=key,
        label=label,
        x=x,
        y=y,
        x_name=predictor,
        y_name=outcome,
        x_axis=catalog.variable_label(predictor),
        y_axis=catalog.variable_label(outcome),
        r_data=catalog.r_data,
        r_response=outcome,
        r_predictor=predictor,
        available_n=int(x.size),
        source=catalog.source,
        simulated=bool(catalog.simulated),
        seed=None,
        caveat=catalog.caveat,
        row_indices=row_indices,
    )


def activate_dataset(
    session: dict[str, Any],
    *,
    key: str,
    x_name: str,
    y_name: str,
) -> None:
    """Make a dataset the active fitting source (persists across tabs)."""
    ensure_registry_state(session)
    catalog = get_catalog(key, session)
    names = catalog.variable_names()
    if x_name not in names or y_name not in names:
        raise ValueError("X and Y must be numeric columns from this dataset.")
    if x_name == y_name:
        raise ValueError("Predictor (X) and Outcome (Y) must be different.")
    frame = get_frame(key, session)
    if pair_complete_count(frame, x_name, y_name) < 2:
        raise ValueError("Need at least two complete (X, Y) observations.")

    session["fit_dataset_key"] = key
    session["fit_x"] = x_name
    session["fit_y"] = y_name
    session["fit_source_key"] = key
    session["fit_dataset_ui"] = key
    session["fit_x_ui"] = x_name
    session["fit_y_ui"] = y_name
    session["dm_draft_key"] = key
    session["dm_draft_x"] = x_name
    session["dm_draft_y"] = y_name


def get_active_fitting_dataset(
    session: dict[str, Any],
    *,
    seed: int = 20261018,
) -> FittingDataset:
    ensure_registry_state(session)
    key = session.get("fit_dataset_key", "students")
    x_name = session.get("fit_x")
    y_name = session.get("fit_y")
    return resolve_fitting_dataset(
        key, x_name=x_name, y_name=y_name, session=session, seed=seed
    )


def download_csv_bytes(key: str, session: dict[str, Any] | None = None) -> bytes:
    frame = get_frame(key, session)
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def _safe_r_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name.strip())
    return cleaned or "uploaded"


def upload_options(session: dict[str, Any]) -> list[tuple[str, str]]:
    ensure_registry_state(session)
    options: list[tuple[str, str]] = []
    for key, entry in session.get("dm_uploads", {}).items():
        n = len(entry["frame"])
        options.append((key, f"{entry['name']} (N = {n:,})"))
    return options


def dataset_display_label(key: str, session: dict[str, Any] | None = None) -> str:
    """Short label without group prefix (for banners and library detail)."""
    if key in DATASET_CATALOGS:
        return dataset_option_label(key)
    catalog = get_catalog(key, session)
    frame = get_frame(key, session)
    return f"{catalog.short_name} (N = {len(frame):,})"
