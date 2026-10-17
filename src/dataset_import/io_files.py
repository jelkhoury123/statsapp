"""Parse uploaded CSV / Excel tables for Dataset Manager."""

from __future__ import annotations

import io
from dataclasses import dataclass

import pandas as pd


_NA_VALUES = ("", "NA", "N/A", "na", "n/a", "NaN", "nan", "NULL", "null", ".")


@dataclass(frozen=True)
class ParsedUpload:
    """Result of reading a user file (one sheet for Excel)."""

    name: str
    frame: pd.DataFrame
    sheet_names: tuple[str, ...]
    active_sheet: str | None
    source: str


def read_excel_sheets(data: bytes) -> list[str]:
    book = pd.ExcelFile(io.BytesIO(data), engine="openpyxl")
    return list(book.sheet_names)


def parse_uploaded_table(
    *,
    filename: str,
    data: bytes,
    sheet_name: str | None = None,
) -> ParsedUpload:
    """Load CSV/TSV or Excel into a DataFrame."""
    lower = filename.lower()
    if lower.endswith((".xlsx", ".xlsm", ".xls")):
        sheets = read_excel_sheets(data)
        if not sheets:
            raise ValueError("The Excel workbook has no worksheets.")
        chosen = sheet_name if sheet_name in sheets else sheets[0]
        frame = pd.read_excel(
            io.BytesIO(data),
            sheet_name=chosen,
            engine="openpyxl",
            na_values=list(_NA_VALUES),
            keep_default_na=True,
        )
        source = f"Uploaded Excel: {filename} · sheet “{chosen}”"
        return ParsedUpload(
            name=_stem(filename),
            frame=_clean_columns(frame),
            sheet_names=tuple(sheets),
            active_sheet=chosen,
            source=source,
        )

    if lower.endswith((".csv", ".txt", ".tsv")):
        sep = "\t" if lower.endswith(".tsv") else ","
        frame = pd.read_csv(
            io.BytesIO(data),
            sep=sep,
            na_values=list(_NA_VALUES),
            keep_default_na=True,
        )
        source = f"Uploaded file: {filename}"
        return ParsedUpload(
            name=_stem(filename),
            frame=_clean_columns(frame),
            sheet_names=(),
            active_sheet=None,
            source=source,
        )

    raise ValueError("Supported uploads: CSV, TSV, or Excel (.xlsx).")


def _stem(filename: str) -> str:
    base = filename.rsplit("/", 1)[-1]
    if "." in base:
        return base.rsplit(".", 1)[0]
    return base or "upload"


def _clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(col).strip() for col in out.columns]
    # Drop fully empty Unnamed columns from Excel exports.
    drop = [
        col
        for col in out.columns
        if col.lower().startswith("unnamed") and out[col].isna().all()
    ]
    if drop:
        out = out.drop(columns=drop)
    return out
