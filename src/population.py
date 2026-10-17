"""The fixed population of 100,000 students.

Sampling experiments draw from this file. They never rebuild it and never
write it back.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.regression import fit_ols

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CSV_PATH = DATA_DIR / "students_population.csv"
DEFAULT_METADATA_PATH = DATA_DIR / "population_metadata.json"

REQUIRED_COLUMNS = (
    "student_id",
    "hours_studied",
    "sleep_hours",
    "attendance",
    "exam_score",
    "individual_error",
)

# Parameters of the process that created the file. They are not the
# benchmark for sampling experiments; the population OLS fit is.
THEORETICAL_ALPHA = 20.0
THEORETICAL_BETA = 5.0
THEORETICAL_SIGMA = 5.0


@dataclass(frozen=True)
class Population:
    """Students, their variables, and the population least-squares line."""

    student_id: np.ndarray
    hours_studied: np.ndarray
    sleep_hours: np.ndarray
    attendance: np.ndarray
    exam_score: np.ndarray
    individual_error: np.ndarray
    alpha: float
    beta: float
    r_squared: float
    sse: float
    sha256: str
    source_path: str

    @property
    def n(self) -> int:
        return int(self.student_id.size)

    @property
    def x_range(self) -> tuple[float, float]:
        low = float(self.hours_studied.min())
        high = float(self.hours_studied.max())
        pad = 0.04 * (high - low)
        return (low - pad, high + pad)

    @property
    def y_range(self) -> tuple[float, float]:
        low = float(self.exam_score.min())
        high = float(self.exam_score.max())
        pad = 0.04 * (high - low)
        return (low - pad, high + pad)


@dataclass(frozen=True)
class PopulationMetadata:
    """Recorded description of the population file."""

    population_size: int
    alpha: float
    beta: float
    r_squared: float
    sse: float
    theoretical_alpha: float
    theoretical_beta: float
    theoretical_sigma: float
    sha256: str


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hash a file without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_population(path: Path | None = None) -> Population:
    """Load the student file and cache the population regression.

    The cache key includes the file's modification time and size, so a
    changed file is reread. Repeated Streamlit reruns reuse the arrays
    and the population coefficients.
    """
    csv_path = Path(path) if path is not None else DEFAULT_CSV_PATH
    stat = csv_path.stat()
    return _load_population_cached(str(csv_path.resolve()), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=2)
def _load_population_cached(path_str: str, mtime_ns: int, size: int) -> Population:
    del mtime_ns, size  # Included so a replaced file misses the cache.
    path = Path(path_str)
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Population file is missing columns: {missing}")

    student_id = np.array(frame["student_id"], dtype=np.int64, copy=True)
    hours = np.array(frame["hours_studied"], dtype=np.float64, copy=True)
    sleep = np.array(frame["sleep_hours"], dtype=np.float64, copy=True)
    attendance = np.array(frame["attendance"], dtype=np.float64, copy=True)
    scores = np.array(frame["exam_score"], dtype=np.float64, copy=True)
    errors = np.array(frame["individual_error"], dtype=np.float64, copy=True)
    if (
        not np.isfinite(hours).all()
        or not np.isfinite(sleep).all()
        or not np.isfinite(attendance).all()
        or not np.isfinite(scores).all()
        or not np.isfinite(errors).all()
    ):
        raise ValueError("Population file contains non-finite values")

    fit = fit_ols(hours, scores)
    if fit.degenerate:
        raise ValueError("Population regression is degenerate")

    for array in (student_id, hours, sleep, attendance, scores, errors):
        array.setflags(write=False)

    return Population(
        student_id=student_id,
        hours_studied=hours,
        sleep_hours=sleep,
        attendance=attendance,
        exam_score=scores,
        individual_error=errors,
        alpha=fit.alpha,
        beta=fit.beta,
        r_squared=fit.r_squared,
        sse=fit.sse,
        sha256=file_sha256(path),
        source_path=str(path),
    )


def load_metadata(path: Path | None = None) -> PopulationMetadata:
    """Load the recorded population coefficients and file hash."""
    meta_path = Path(path) if path is not None else DEFAULT_METADATA_PATH
    stat = meta_path.stat()
    return _load_metadata_cached(str(meta_path.resolve()), stat.st_mtime_ns)


@lru_cache(maxsize=2)
def _load_metadata_cached(path_str: str, mtime_ns: int) -> PopulationMetadata:
    del mtime_ns
    with Path(path_str).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    ols = raw["population_ols"]
    theoretical = raw["data_generating_process"]
    return PopulationMetadata(
        population_size=int(raw["population_size"]),
        alpha=float(ols["alpha"]),
        beta=float(ols["beta"]),
        r_squared=float(ols["r_squared"]),
        sse=float(ols["sse"]),
        theoretical_alpha=float(theoretical["alpha"]),
        theoretical_beta=float(theoretical["beta"]),
        theoretical_sigma=float(theoretical["sigma"]),
        sha256=str(raw["sha256"]),
    )


def coefficients_match_metadata(
    population: Population,
    metadata: PopulationMetadata,
    *,
    atol: float = 1e-4,
) -> bool:
    """Whether the computed population line agrees with the metadata."""
    return (
        population.n == metadata.population_size
        and abs(population.alpha - metadata.alpha) <= atol
        and abs(population.beta - metadata.beta) <= atol
    )
