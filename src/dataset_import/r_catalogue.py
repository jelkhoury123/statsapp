"""Bundled R `datasets` package tables (no runtime R dependency)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.fitting_line import DatasetCatalog, VariableSpec

_R_DIR = Path(__file__).resolve().parents[1] / "r_datasets"
_CATALOGUE_PATH = _R_DIR / "catalogue.json"


@lru_cache(maxsize=1)
def _raw_catalogue() -> dict:
    with _CATALOGUE_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def r_dataset_keys() -> tuple[str, ...]:
    return tuple(sorted(_raw_catalogue().keys()))


def r_registry_key(name: str) -> str:
    return f"r_{name}"


def is_r_registry_key(key: str) -> bool:
    return key.startswith("r_") and key[2:] in _raw_catalogue()


def r_name_from_key(key: str) -> str:
    if not is_r_registry_key(key):
        raise KeyError(key)
    return key[2:]


@lru_cache(maxsize=16)
def load_r_frame(name: str) -> pd.DataFrame:
    meta = _raw_catalogue().get(name)
    if meta is None:
        raise KeyError(f"Unknown R dataset: {name}")
    path = _R_DIR / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Bundled R dataset missing: {path.name}")
    frame = pd.read_csv(path)
    if "rownames" in frame.columns:
        frame = frame.drop(columns=["rownames"])
    # Keep original R column names; restrict to declared numeric set for X/Y.
    return frame


def r_dataset_catalog(name: str) -> DatasetCatalog:
    meta = _raw_catalogue()[name]
    numeric = tuple(meta["numeric"])
    frame = load_r_frame(name)
    present = [col for col in numeric if col in frame.columns]
    if len(present) < 2:
        raise ValueError(f"R dataset {name} needs at least two numeric columns")
    default_x = meta["default_x"] if meta["default_x"] in present else present[0]
    default_y = meta["default_y"] if meta["default_y"] in present else present[1]
    if default_x == default_y:
        default_y = next(col for col in present if col != default_x)
    variables = tuple(VariableSpec(col, col) for col in present)
    return DatasetCatalog(
        key=r_registry_key(name),
        short_name=str(meta["short_name"]),
        r_data=name,
        source=f"R datasets::{name}. {meta['description']}",
        variables=variables,
        default_x=default_x,
        default_y=default_y,
        file_stem=None,
        caveat="Observed data from R's datasets package — no known population α or β.",
        simulated=False,
        csv_path=_R_DIR / f"{name}.csv",
    )


@lru_cache(maxsize=1)
def r_dataset_catalogs() -> dict[str, DatasetCatalog]:
    return {r_registry_key(name): r_dataset_catalog(name) for name in r_dataset_keys()}


def r_option_label(name: str) -> str:
    catalog = r_dataset_catalog(name)
    n = len(load_r_frame(name))
    return f"{catalog.short_name} (N = {n:,})"


def search_r_datasets(query: str = "") -> list[tuple[str, str]]:
    """Return (registry_key, label) filtered by name/description."""
    needle = (query or "").strip().lower()
    options: list[tuple[str, str]] = []
    for name in r_dataset_keys():
        meta = _raw_catalogue()[name]
        hay = f"{name} {meta['short_name']} {meta['description']}".lower()
        if needle and needle not in hay:
            continue
        options.append((r_registry_key(name), r_option_label(name)))
    return options
