"""Preview, missingness, and numeric-variable helpers for Dataset Manager."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetPreview:
    """Compact summary shown before activating a dataset."""

    name: str
    description: str
    n_rows: int
    n_cols: int
    n_numeric: int
    missing_by_column: dict[str, int]
    preview: pd.DataFrame


def numeric_column_names(frame: pd.DataFrame) -> list[str]:
    """Columns that are mostly numeric after coercion (eligible for X/Y)."""
    names: list[str] = []
    for column in frame.columns:
        name = str(column)
        if name.lower() in {"rownames", "row.names", "index"}:
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        valid = int(series.notna().sum())
        if valid == 0:
            continue
        # Require at least half non-missing values to be numeric-coercible,
        # or the column already numeric dtype with some finite values.
        if pd.api.types.is_numeric_dtype(frame[column]):
            if np.isfinite(series.to_numpy(dtype=np.float64)).any():
                names.append(name)
            continue
        original_non_null = int(frame[column].notna().sum())
        if original_non_null == 0:
            continue
        if valid / original_non_null >= 0.5:
            names.append(name)
    return names


def missing_value_summary(frame: pd.DataFrame) -> dict[str, int]:
    """Per-column count of missing values (including non-numeric coercion NaNs for display)."""
    summary: dict[str, int] = {}
    for column in frame.columns:
        summary[str(column)] = int(frame[column].isna().sum())
    return summary


def pair_complete_count(frame: pd.DataFrame, x_name: str, y_name: str) -> int:
    if x_name not in frame.columns or y_name not in frame.columns:
        return 0
    clean = (
        frame[[x_name, y_name]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )
    return int(len(clean))


def preview_frame(frame: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return frame.head(int(n)).copy()


def build_preview(
    *,
    name: str,
    description: str,
    frame: pd.DataFrame,
) -> DatasetPreview:
    numeric = numeric_column_names(frame)
    return DatasetPreview(
        name=name,
        description=description,
        n_rows=int(len(frame)),
        n_cols=int(frame.shape[1]),
        n_numeric=len(numeric),
        missing_by_column=missing_value_summary(frame),
        preview=preview_frame(frame, 10),
    )
