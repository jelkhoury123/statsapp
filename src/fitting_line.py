"""Lesson: how OLS chooses the best-fitting line, with real and simulated data."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.regression import fit_ols

DEFAULT_SEED = 20261018
MIN_SAMPLE_N = 5
MAX_INTERACTIVE_N = 1000
DEFAULT_SAMPLE_N = 100
DEFAULT_STUDENT_SAMPLE_N = 100
STUDENTS_10K_N = 10_000
STUDENTS_10K_SEED = 20261018
MIN_SAMPLE_PCT = 2
MAX_SAMPLE_PCT = 20
DEFAULT_SAMPLE_PCT = 2
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = Path(__file__).resolve().parent / "fitting_data"
_STUDENTS_10K_PATH = _PROJECT_ROOT / "data" / "students_10k.csv"


@dataclass(frozen=True)
class FittingDataset:
    """One teaching dataset with display metadata and R formula info."""

    key: str
    label: str
    x: np.ndarray
    y: np.ndarray
    x_name: str
    y_name: str
    x_axis: str
    y_axis: str
    r_data: str
    r_response: str
    r_predictor: str
    available_n: int
    source: str
    simulated: bool = False
    seed: int | None = None
    caveat: str | None = None
    # Positions into the catalog frame (same order as x/y) — used to align
    # additional predictors (e.g. sleep_hours) with the current sample.
    row_indices: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(self.x.size)

    @property
    def hours(self) -> np.ndarray:
        """Backward-compatible alias used by older call sites."""
        return self.x

    @property
    def scores(self) -> np.ndarray:
        return self.y

    @property
    def r_command(self) -> str:
        return f"model <- lm({self.r_response} ~ {self.r_predictor}, data = {self.r_data})"

    def r_sample_commands(self, sample_n: int | None = None) -> str:
        """R snippet: optional sample, then lm() on sample_data."""
        n = int(self.available_n if sample_n is None else sample_n)
        if n >= int(self.available_n):
            return (
                f"sample_data <- {self.r_data}\n"
                f"model <- lm({self.r_response} ~ {self.r_predictor}, data = sample_data)"
            )
        return (
            f"sample_data <- {self.r_data}[sample(nrow({self.r_data}), {n}), ]\n"
            f"model <- lm({self.r_response} ~ {self.r_predictor}, data = sample_data)"
        )

    def slider_bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """((alpha_min, alpha_max, alpha_step), (beta_min, beta_max, beta_step))."""
        ols = fit_ols(self.x, self.y)
        y_span = float(np.ptp(self.y))
        x_span = float(np.ptp(self.x))
        if y_span <= 0:
            y_span = max(abs(float(ols.alpha)), 1.0)
        if x_span <= 0:
            x_span = 1.0
        # Keep slider ranges near the data cloud so candidates stay interpretable.
        alpha_pad = max(0.25 * y_span, abs(float(ols.alpha)) * 0.15, 1.0)
        alpha_min = float(ols.alpha - alpha_pad)
        alpha_max = float(ols.alpha + alpha_pad)
        if alpha_min >= alpha_max:
            alpha_min, alpha_max = float(ols.alpha) - 1.0, float(ols.alpha) + 1.0
        typical_slope = y_span / x_span
        beta_pad = max(abs(float(ols.beta)) * 0.7, typical_slope * 0.35, 0.15)
        beta_min = float(ols.beta - beta_pad)
        beta_max = float(ols.beta + beta_pad)
        if beta_min >= beta_max:
            beta_min, beta_max = float(ols.beta) - 1.0, float(ols.beta) + 1.0
        alpha_step = _nice_step(alpha_max - alpha_min)
        beta_step = _nice_step(beta_max - beta_min)
        return (alpha_min, alpha_max, alpha_step), (beta_min, beta_max, beta_step)

    def slope_grid(self, count: int = 161) -> np.ndarray:
        (_, _, _), (beta_min, beta_max, _) = self.slider_bounds()
        return np.linspace(beta_min, beta_max, int(count))


def _nice_step(span: float) -> float:
    span = abs(float(span))
    if span <= 0:
        return 0.1
    raw = span / 200.0
    power = 10 ** np.floor(np.log10(raw))
    for factor in (1.0, 2.0, 5.0, 10.0):
        step = factor * power
        if step >= raw:
            return float(step)
    return float(10.0 * power)


def default_sample_n(available_n: int, *, key: str | None = None) -> int:
    if key == "students":
        return int(min(DEFAULT_STUDENT_SAMPLE_N, int(available_n)))
    return int(min(DEFAULT_SAMPLE_N, int(available_n)))


def interactive_n_cap(available_n: int) -> int:
    """Cap interactive draws for huge files; allow full use up to the 10k students pool."""
    size = int(available_n)
    if size <= STUDENTS_10K_N:
        return size
    return int(min(MAX_INTERACTIVE_N, size))


def n_from_percentage(available_n: int, percentage: float) -> int:
    """n = round(N × percentage / 100), at least MIN_SAMPLE_N when N allows."""
    size = int(available_n)
    raw = int(round(size * float(percentage) / 100.0))
    if size >= MIN_SAMPLE_N:
        return int(np.clip(raw, MIN_SAMPLE_N, size))
    return max(1, min(raw, size))


def resolve_sample_n(
    available_n: int,
    percentage: float,
    *,
    use_all: bool = False,
) -> tuple[int, int, bool]:
    """Return (requested_n, actual_n, capped).

    ``requested_n`` is the uncapped percentage conversion (or full N).
    ``actual_n`` is what the app will draw (may hit the interactive cap).
    """
    size = int(available_n)
    cap = interactive_n_cap(size)
    if use_all:
        requested = size
        actual = min(size, cap)
        return requested, actual, actual < requested
    requested = n_from_percentage(size, percentage)
    actual = min(requested, cap)
    return requested, actual, actual < requested


@dataclass(frozen=True)
class VariableSpec:
    """One numeric column that may be used as predictor or outcome."""

    name: str
    label: str


@dataclass(frozen=True)
class DatasetCatalog:
    """Metadata for a fitting dataset and its selectable numeric variables."""

    key: str
    short_name: str
    r_data: str
    source: str
    variables: tuple[VariableSpec, ...]
    default_x: str
    default_y: str
    file_stem: str | None = None
    caveat: str | None = None
    simulated: bool = False
    csv_path: Path | None = None

    def variable_label(self, name: str) -> str:
        for variable in self.variables:
            if variable.name == name:
                return variable.label
        raise KeyError(name)

    def variable_names(self) -> tuple[str, ...]:
        return tuple(variable.name for variable in self.variables)


DATASET_CATALOGS: dict[str, DatasetCatalog] = {
    "penguins": DatasetCatalog(
        key="penguins",
        short_name="Penguins",
        file_stem="penguins",
        r_data="penguins",
        source="Palmer Penguins (Horst, Hill & Gorman).",
        default_x="flipper_length_mm",
        default_y="body_mass_g",
        variables=(
            VariableSpec("flipper_length_mm", "Flipper length (mm)"),
            VariableSpec("bill_length_mm", "Bill length (mm)"),
            VariableSpec("bill_depth_mm", "Bill depth (mm)"),
            VariableSpec("body_mass_g", "Body mass (g)"),
        ),
    ),
    "auto": DatasetCatalog(
        key="auto",
        short_name="Auto",
        file_stem="auto",
        r_data="Auto",
        source="ISLR Auto dataset.",
        default_x="horsepower",
        default_y="mpg",
        variables=(
            VariableSpec("horsepower", "Horsepower"),
            VariableSpec("weight", "Weight"),
            VariableSpec("displacement", "Displacement"),
            VariableSpec("acceleration", "Acceleration"),
            VariableSpec("cylinders", "Cylinders"),
            VariableSpec("year", "Model year"),
            VariableSpec("mpg", "Fuel efficiency (mpg)"),
        ),
    ),
    "boston": DatasetCatalog(
        key="boston",
        short_name="Boston",
        file_stem="boston",
        r_data="Boston",
        source="Boston housing dataset (Harrison & Rubinfeld).",
        caveat=(
            "Historical dataset with known ethical and methodological limitations; "
            "used here only to practise simple linear regression."
        ),
        default_x="rm",
        default_y="medv",
        variables=(
            VariableSpec("rm", "Average rooms (RM)"),
            VariableSpec("lstat", "Lower status % (LSTAT)"),
            VariableSpec("ptratio", "Pupil–teacher ratio"),
            VariableSpec("crim", "Crime rate (CRIM)"),
            VariableSpec("nox", "Nitric oxides (NOX)"),
            VariableSpec("dis", "Distance to employment"),
            VariableSpec("age", "Proportion older units"),
            VariableSpec("tax", "Property tax rate"),
            VariableSpec("indus", "Non-retail business acres"),
            VariableSpec("medv", "Median value (MEDV, $1,000s)"),
        ),
    ),
    "diamonds": DatasetCatalog(
        key="diamonds",
        short_name="Diamonds",
        file_stem="diamonds",
        r_data="diamonds",
        source="ggplot2 diamonds dataset.",
        default_x="carat",
        default_y="price",
        variables=(
            VariableSpec("carat", "Carat weight"),
            VariableSpec("depth", "Depth (%)"),
            VariableSpec("table", "Table (%)"),
            VariableSpec("x", "Length x (mm)"),
            VariableSpec("y", "Width y (mm)"),
            VariableSpec("z", "Depth z (mm)"),
            VariableSpec("price", "Price (USD)"),
        ),
    ),
    "students": DatasetCatalog(
        key="students",
        short_name="Students",
        file_stem=None,
        csv_path=_STUDENTS_10K_PATH,
        r_data="students_10k",
        source=(
            "Fixed pool of 10,000 students drawn once from "
            "data/students_population.csv (seed "
            f"{STUDENTS_10K_SEED}); samples are drawn from data/students_10k.csv."
        ),
        simulated=True,
        default_x="hours_studied",
        default_y="exam_score",
        variables=(
            VariableSpec("hours_studied", "Hours studied"),
            VariableSpec("sleep_hours", "Sleep (hours)"),
            VariableSpec("attendance", "Attendance (%)"),
            VariableSpec("exam_score", "Exam score"),
        ),
    ),
}

# Predictors offered in the Standardisation lesson (same Students sample).
STD_PREDICTOR_KEYS: tuple[str, ...] = (
    "hours_studied",
    "attendance",
    "sleep_hours",
)
STD_PREDICTOR_META: dict[str, tuple[str, str]] = {
    # key -> (axis label, raw slope unit phrase after β̂)
    "hours_studied": ("Hours studied", "exam points / hour studied"),
    "attendance": ("Attendance (%)", "exam points / %-attendance"),
    "sleep_hours": ("Sleep (hours)", "exam points / hour of sleep"),
}


@lru_cache(maxsize=8)
def _load_frame(stem: str) -> pd.DataFrame:
    path = _DATA_DIR / f"{stem}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Fitting dataset file missing: {path.name}. "
            "Authentic data must be bundled under src/fitting_data/."
        )
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def _students_10k_frame() -> pd.DataFrame:
    """Load the fixed 10k fitting subset (never regenerates from the 100k CSV)."""
    path = _STUDENTS_10K_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Fitting student subset missing: {path}. "
            "Expected data/students_10k.csv (fixed 10,000-row file)."
        )
    frame = pd.read_csv(path)
    required = ("hours_studied", "sleep_hours", "attendance", "exam_score")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"students_10k.csv missing columns: {missing}")
    if len(frame) != STUDENTS_10K_N:
        raise ValueError(
            f"students_10k.csv must contain exactly {STUDENTS_10K_N:,} rows; "
            f"found {len(frame):,}."
        )
    return frame


def _catalog_frame(catalog: DatasetCatalog) -> pd.DataFrame:
    if catalog.csv_path is not None:
        if catalog.key == "students":
            return _students_10k_frame()
        return pd.read_csv(catalog.csv_path)
    if catalog.file_stem is None:
        raise ValueError(f"Dataset {catalog.key} has no data source")
    return _load_frame(catalog.file_stem)


def pair_complete_n(key: str, x_name: str, y_name: str) -> int:
    """Number of complete cases for a predictor/outcome pair."""
    catalog = DATASET_CATALOGS[key]
    frame = _catalog_frame(catalog)
    clean = frame[[x_name, y_name]].apply(pd.to_numeric, errors="coerce").dropna()
    return int(len(clean))


def dataset_display_n(key: str) -> int:
    """N shown in the dataset selector (complete rows for the default pair)."""
    catalog = DATASET_CATALOGS[key]
    return pair_complete_n(key, catalog.default_x, catalog.default_y)


def dataset_option_label(key: str) -> str:
    catalog = DATASET_CATALOGS[key]
    return f"{catalog.short_name} (N = {dataset_display_n(key):,})"


DATASET_OPTIONS = tuple((key, dataset_option_label(key)) for key in DATASET_CATALOGS)


def load_fitting_dataset(
    key: str,
    seed: int = DEFAULT_SEED,
    *,
    x_name: str | None = None,
    y_name: str | None = None,
) -> FittingDataset:
    """Load one X/Y pair from a catalogued dataset (pairwise complete cases)."""
    del seed
    if key not in DATASET_CATALOGS:
        raise KeyError(f"Unknown fitting dataset: {key}")
    catalog = DATASET_CATALOGS[key]
    predictor = x_name or catalog.default_x
    outcome = y_name or catalog.default_y
    names = catalog.variable_names()
    if predictor not in names or outcome not in names:
        raise KeyError(f"Unknown variable for {key}: {predictor!r}, {outcome!r}")
    if predictor == outcome:
        raise ValueError("Predictor (X) and Outcome (Y) must be different variables")

    frame = _catalog_frame(catalog).reset_index(drop=True)
    numeric = frame[[predictor, outcome]].apply(pd.to_numeric, errors="coerce")
    mask = numeric.notna().all(axis=1).to_numpy()
    if not bool(np.any(mask)):
        raise ValueError(f"No complete observations for {predictor} / {outcome}")
    row_indices = np.flatnonzero(mask)
    x = np.asarray(numeric.loc[mask, predictor], dtype=np.float64)
    y = np.asarray(numeric.loc[mask, outcome], dtype=np.float64)
    return FittingDataset(
        key=key,
        label=dataset_option_label(key),
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
        simulated=catalog.simulated,
        seed=None,
        caveat=catalog.caveat,
        row_indices=row_indices,
    )


def make_student_sample(seed: int = DEFAULT_SEED) -> FittingDataset:
    """Fixed 10,000-student subset used by the fitting tab."""
    return load_fitting_dataset("students", seed)


def make_fitting_sample(seed: int = DEFAULT_SEED) -> FittingDataset:
    """Small fixed subsample from the shared CSV (tests / legacy helpers)."""
    pool = make_student_sample()
    return subsample_observations(pool, DEFAULT_STUDENT_SAMPLE_N, np.random.default_rng(int(seed)))


def predicted_scores(hours: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return float(alpha) + float(beta) * np.asarray(hours, dtype=np.float64)


def residuals(hours: np.ndarray, scores: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return np.asarray(scores, dtype=np.float64) - predicted_scores(hours, alpha, beta)


def sum_of_squared_errors(hours: np.ndarray, scores: np.ndarray, alpha: float, beta: float) -> float:
    errors = residuals(hours, scores, alpha, beta)
    return float(np.dot(errors, errors))


def total_sum_of_squares(scores: np.ndarray) -> float:
    """SST = Σ(yi − mean(y))²."""
    y = np.asarray(scores, dtype=np.float64)
    centered = y - float(np.mean(y))
    return float(np.dot(centered, centered))


def regression_sum_of_squares(
    hours: np.ndarray,
    scores: np.ndarray,
    alpha: float,
    beta: float,
) -> float:
    """SS Model = Σ(ŷᵢ − ȳ)² for predictions ŷᵢ = α + βxᵢ."""
    y = np.asarray(scores, dtype=np.float64)
    x = np.asarray(hours, dtype=np.float64)
    y_hat = float(alpha) + float(beta) * x
    centered = y_hat - float(np.mean(y))
    return float(np.dot(centered, centered))


def r_squared(sse: float, sst: float) -> float:
    """R² = 1 − SSE/SST. May be negative for a poor line; NaN if SST = 0."""
    if not np.isfinite(sst) or float(sst) <= 0.0:
        return float("nan")
    if not np.isfinite(sse):
        return float("nan")
    return float(1.0 - float(sse) / float(sst))


def sample_sd(values: np.ndarray) -> float:
    """Sample SD with Bessel's correction (n − 1), matching R's sd()."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return float("nan")
    return float(np.std(arr, ddof=1))


def zscore(values: np.ndarray) -> np.ndarray:
    """z = (value − mean) / SD; zeros when SD is undefined or zero."""
    arr = np.asarray(values, dtype=np.float64)
    sd = sample_sd(arr)
    if not np.isfinite(sd) or sd == 0.0:
        return np.zeros(arr.shape, dtype=np.float64)
    return (arr - float(np.mean(arr))) / sd


def standardised_slope(beta: float, sd_x: float, sd_y: float) -> float:
    """β* = β̂ × SD(X) / SD(Y)."""
    if not np.isfinite(beta) or not np.isfinite(sd_x) or not np.isfinite(sd_y):
        return float("nan")
    if sd_y == 0.0:
        return float("nan")
    return float(beta) * float(sd_x) / float(sd_y)


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation r between predictor and outcome (simple linear case)."""
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    if xv.size < 2 or yv.size < 2:
        return float("nan")
    if sample_sd(xv) == 0.0 or sample_sd(yv) == 0.0:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def r_standardisation_commands(*, predictor: str, response: str) -> str:
    """Concise R for both standardisation routes (one predictor)."""
    return (
        f"# Method 1 — standardise the fitted coefficient\n"
        f"m <- lm({response} ~ {predictor}, data = sample_data)\n"
        f"beta_star <- coef(m)[[\"{predictor}\"]] *\n"
        f"  sd(sample_data${predictor}) / sd(sample_data${response})\n"
        f"round(beta_star, 3)\n"
        f"\n"
        f"# Method 2 — standardise X and Y, then lm()\n"
        f"zx <- scale(sample_data${predictor})\n"
        f"zy <- scale(sample_data${response})\n"
        f"round(coef(lm(zy ~ zx))[[\"zx\"]], 3)  # same β*\n"
    )


def standardisation_student_frame(view: FittingDataset) -> pd.DataFrame | None:
    """Students sample rows with all Standardisation predictors + exam score.

    Uses the fitting sample's row indices — never regenerates predictors or
    draws a separate sample. Returns None outside the Students catalog.
    """
    if getattr(view, "key", None) != "students":
        return None
    indices = getattr(view, "row_indices", None)
    if indices is None:
        return None
    idx = np.asarray(indices, dtype=np.int64)
    frame = _students_10k_frame().reset_index(drop=True)
    if idx.size == 0 or int(np.max(idx)) >= len(frame):
        return None
    cols = list(STD_PREDICTOR_KEYS) + ["exam_score"]
    missing = [name for name in cols if name not in frame.columns]
    if missing:
        raise ValueError(f"Students frame missing Standardisation columns: {missing}")
    out = frame.loc[idx, cols].copy()
    return out.reset_index(drop=True)


def standardisation_student_sample(
    view: FittingDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Backward-compatible: hours, sleep, exam for the current Students sample."""
    frame = standardisation_student_frame(view)
    if frame is None:
        return None
    return (
        np.asarray(frame["hours_studied"], dtype=np.float64),
        np.asarray(frame["sleep_hours"], dtype=np.float64),
        np.asarray(frame["exam_score"], dtype=np.float64),
    )


def residual_standard_error(sse_ols: float, n: int) -> float:
    """s = sqrt(SSE_OLS / (n − 2)) for a fitted simple linear model."""
    df = int(n) - 2
    if df <= 0 or not np.isfinite(sse_ols) or float(sse_ols) < 0.0:
        return float("nan")
    return float(np.sqrt(float(sse_ols) / float(df)))


def r_lm_commands(*, predictor: str, response: str) -> str:
    """Compact R: fit lm() on sample_data and print summary (current X/Y names)."""
    return (
        f"model <- lm({response} ~ {predictor}, data = sample_data)\n"
        "summary(model)"
    )


def _r_format_float(value: float, digits: int = 4) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _r_format_pval(p_value: float) -> str:
    """Approximate R's format.pval for summary.lm coefficients / F-test."""
    if not np.isfinite(p_value):
        return "NA"
    if p_value < 2.2e-16:
        return "< 2.2e-16"
    if p_value < 0.0001:
        return f"{p_value:.2e}"
    return f"{p_value:.4f}"


def _r_signif_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "  "
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "** "
    if p_value < 0.05:
        return "*  "
    if p_value < 0.1:
        return ".  "
    return "   "


def r_lm_summary_text(
    x: np.ndarray,
    y: np.ndarray,
    *,
    predictor: str,
    response: str,
    data_name: str = "sample_data",
) -> str:
    """Console text matching summary(lm(...)) for the current sample's OLS fit."""
    from scipy import stats

    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    n = int(x_values.size)
    ols = fit_ols(x_values, y_values)
    if n < 3 or ols.degenerate or not np.isfinite(ols.sse):
        return "Error: cannot fit lm() for this sample."

    df = n - 2
    resid = y_values - (ols.alpha + ols.beta * x_values)
    quantiles = np.quantile(resid, [0.0, 0.25, 0.5, 0.75, 1.0])
    sigma = residual_standard_error(ols.sse, n)
    x_mean = float(np.mean(x_values))
    sxx = float(np.dot(x_values - x_mean, x_values - x_mean))
    if sxx <= 0 or not np.isfinite(sigma):
        return "Error: cannot fit lm() for this sample."

    se_beta = float(sigma / np.sqrt(sxx))
    se_alpha = float(sigma * np.sqrt((1.0 / n) + (x_mean * x_mean) / sxx))
    t_alpha = float(ols.alpha / se_alpha)
    t_beta = float(ols.beta / se_beta)
    p_alpha = float(2.0 * stats.t.sf(abs(t_alpha), df))
    p_beta = float(2.0 * stats.t.sf(abs(t_beta), df))

    y_mean = float(np.mean(y_values))
    sst = float(np.dot(y_values - y_mean, y_values - y_mean))
    ssr = max(sst - float(ols.sse), 0.0)
    r2 = float("nan") if sst <= 0 else float(1.0 - float(ols.sse) / sst)
    adj_r2 = (
        float("nan")
        if not np.isfinite(r2)
        else float(1.0 - (1.0 - r2) * (n - 1) / df)
    )
    f_stat = float((ssr / 1.0) / (float(ols.sse) / df)) if ols.sse > 0 else float("inf")
    f_p = float(stats.f.sf(f_stat, 1, df)) if np.isfinite(f_stat) else 0.0

    name_w = max(len("(Intercept)"), len(predictor), 12)
    call = f"lm(formula = {response} ~ {predictor}, data = {data_name})"

    def coef_row(name: str, est: float, se: float, t_val: float, p_val: float) -> str:
        return (
            f"{name:<{name_w}} "
            f"{est:>10.4f} "
            f"{se:>10.4f} "
            f"{t_val:>7.3f} "
            f"{_r_format_pval(p_val):>10} "
            f"{_r_signif_stars(p_val)}"
        )

    lines = [
        "Call:",
        call,
        "",
        "Residuals:",
        (
            f"{'Min':>8} {'1Q':>8} {'Median':>8} {'3Q':>8} {'Max':>8}\n"
            f"{quantiles[0]:8.3f} {quantiles[1]:8.3f} {quantiles[2]:8.3f} "
            f"{quantiles[3]:8.3f} {quantiles[4]:8.3f}"
        ),
        "",
        "Coefficients:",
        (
            f"{'':<{name_w}} {'Estimate':>10} {'Std. Error':>10} "
            f"{'t value':>7} {'Pr(>|t|)':>10}  "
        ),
        coef_row("(Intercept)", ols.alpha, se_alpha, t_alpha, p_alpha),
        coef_row(predictor, ols.beta, se_beta, t_beta, p_beta),
        "---",
        "Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1",
        "",
        (
            f"Residual standard error: {_r_format_float(sigma, 3)} on {df} "
            "degrees of freedom"
        ),
        (
            f"Multiple R-squared:  {_r_format_float(r2, 4)},"
            f"\tAdjusted R-squared:  {_r_format_float(adj_r2, 4)}"
        ),
        (
            f"F-statistic: {_r_format_float(f_stat, 1)} on 1 and {df} DF,  "
            f"p-value: {_r_format_pval(f_p)}"
        ),
    ]
    return "\n".join(lines)


def best_intercept_for_slope(hours: np.ndarray, scores: np.ndarray, beta: float) -> float:
    """Intercept that minimises SSE for a fixed slope."""
    return float(np.mean(scores) - float(beta) * np.mean(hours))


def sse_for_slope_with_best_intercept(
    hours: np.ndarray,
    scores: np.ndarray,
    beta: float,
) -> float:
    alpha = best_intercept_for_slope(hours, scores, beta)
    return sum_of_squared_errors(hours, scores, alpha, beta)


def sse_curve(
    hours: np.ndarray,
    scores: np.ndarray,
    slopes: np.ndarray,
) -> np.ndarray:
    """SSE(b) using the intercept that is optimal for each slope b."""
    x = np.asarray(hours, dtype=np.float64)
    y = np.asarray(scores, dtype=np.float64)
    beta = np.asarray(slopes, dtype=np.float64)
    if beta.size == 0:
        return np.asarray([], dtype=np.float64)
    x_bar = float(np.mean(x))
    y_bar = float(np.mean(y))
    alpha = y_bar - beta * x_bar
    predicted = alpha[:, None] + beta[:, None] * x[None, :]
    errors = y[None, :] - predicted
    return np.einsum("ij,ij->i", errors, errors)


def sse_from_ols_identity(
    alpha: float | np.ndarray,
    beta: float | np.ndarray,
    *,
    alpha_ols: float,
    beta_ols: float,
    sse_min: float,
    n: int,
    x_mean: float,
    sxx: float,
) -> float | np.ndarray:
    """SSE(α, β) via the exact OLS quadratic identity."""
    da = np.asarray(alpha, dtype=np.float64) - float(alpha_ols)
    db = np.asarray(beta, dtype=np.float64) - float(beta_ols)
    values = (
        float(sse_min)
        + float(n) * (da + float(x_mean) * db) ** 2
        + float(sxx) * db**2
    )
    if np.ndim(values) == 0:
        return float(values)
    return values


def sse_surface_scales(hours: np.ndarray, scores: np.ndarray) -> tuple[float, float, object]:
    """Adaptive α/β scales so normalised (u, v) axes are visually comparable."""
    x = np.asarray(hours, dtype=np.float64)
    y = np.asarray(scores, dtype=np.float64)
    ols = fit_ols(x, y)
    y_span = float(np.ptp(y))
    x_span = float(np.ptp(x))
    if y_span <= 0:
        y_span = max(abs(float(ols.alpha)), 1.0)
    if x_span <= 0:
        x_span = 1.0
    typical_slope = y_span / x_span
    alpha_scale = max(0.35 * y_span, abs(float(ols.alpha)) * 0.12, 1.0)
    beta_scale = max(abs(float(ols.beta)) * 0.55, typical_slope * 0.4, 0.05)
    return float(alpha_scale), float(beta_scale), ols


def expand_surface_domain(
    *,
    alpha_ols: float,
    beta_ols: float,
    alpha_scale: float,
    beta_scale: float,
    candidate_alpha: float | None = None,
    candidate_beta: float | None = None,
    previous: tuple[float, float, float, float] | None = None,
    pad_scales: float = 2.0,
) -> tuple[float, float, float, float]:
    """α/β domain around OLS (and candidate). Expands monotonically if previous given."""
    half_a = float(pad_scales) * float(alpha_scale)
    half_b = float(pad_scales) * float(beta_scale)
    a0 = float(alpha_ols) - half_a
    a1 = float(alpha_ols) + half_a
    b0 = float(beta_ols) - half_b
    b1 = float(beta_ols) + half_b
    if candidate_alpha is not None and candidate_beta is not None:
        margin_a = 0.2 * float(alpha_scale)
        margin_b = 0.2 * float(beta_scale)
        a0 = min(a0, float(candidate_alpha) - margin_a)
        a1 = max(a1, float(candidate_alpha) + margin_a)
        b0 = min(b0, float(candidate_beta) - margin_b)
        b1 = max(b1, float(candidate_beta) + margin_b)
    if previous is not None:
        p_a0, p_a1, p_b0, p_b1 = previous
        a0, a1 = min(a0, float(p_a0)), max(a1, float(p_a1))
        b0, b1 = min(b0, float(p_b0)), max(b1, float(p_b1))
    if a0 >= a1:
        a0, a1 = float(alpha_ols) - 1.0, float(alpha_ols) + 1.0
    if b0 >= b1:
        b0, b1 = float(beta_ols) - 1.0, float(beta_ols) + 1.0
    return float(a0), float(a1), float(b0), float(b1)


def sse_surface_grids(
    hours: np.ndarray,
    scores: np.ndarray,
    *,
    alpha_bounds: tuple[float, float] | None = None,
    beta_bounds: tuple[float, float] | None = None,
    candidate_alpha: float | None = None,
    candidate_beta: float | None = None,
    previous_domain: tuple[float, float, float, float] | None = None,
    resolution: int = 45,
) -> dict[str, np.ndarray | float | int | tuple]:
    """Build exact SSE surface with normalised (u, v) plotting coordinates."""
    x = np.asarray(hours, dtype=np.float64)
    y = np.asarray(scores, dtype=np.float64)
    alpha_scale, beta_scale, ols = sse_surface_scales(x, y)
    n = int(x.size)
    x_mean = float(np.mean(x))
    centred = x - x_mean
    sxx = float(np.dot(centred, centred))
    if alpha_bounds is not None and beta_bounds is not None:
        a0, a1 = float(alpha_bounds[0]), float(alpha_bounds[1])
        b0, b1 = float(beta_bounds[0]), float(beta_bounds[1])
        a0 = min(a0, float(ols.alpha))
        a1 = max(a1, float(ols.alpha))
        b0 = min(b0, float(ols.beta))
        b1 = max(b1, float(ols.beta))
        if candidate_alpha is not None and candidate_beta is not None:
            a0 = min(a0, float(candidate_alpha))
            a1 = max(a1, float(candidate_alpha))
            b0 = min(b0, float(candidate_beta))
            b1 = max(b1, float(candidate_beta))
        if previous_domain is not None:
            p_a0, p_a1, p_b0, p_b1 = previous_domain
            a0, a1 = min(a0, float(p_a0)), max(a1, float(p_a1))
            b0, b1 = min(b0, float(p_b0)), max(b1, float(p_b1))
    else:
        a0, a1, b0, b1 = expand_surface_domain(
            alpha_ols=float(ols.alpha),
            beta_ols=float(ols.beta),
            alpha_scale=alpha_scale,
            beta_scale=beta_scale,
            candidate_alpha=candidate_alpha,
            candidate_beta=candidate_beta,
            previous=previous_domain,
        )
    alphas = np.linspace(a0, a1, int(resolution))
    betas = np.linspace(b0, b1, int(resolution))
    alpha_grid, beta_grid = np.meshgrid(alphas, betas, indexing="xy")
    sse_grid = np.asarray(
        sse_from_ols_identity(
            alpha_grid,
            beta_grid,
            alpha_ols=float(ols.alpha),
            beta_ols=float(ols.beta),
            sse_min=float(ols.sse),
            n=n,
            x_mean=x_mean,
            sxx=sxx,
        ),
        dtype=np.float64,
    )
    excess_grid = sse_grid - float(ols.sse)
    u_grid = (alpha_grid - float(ols.alpha)) / float(alpha_scale)
    v_grid = (beta_grid - float(ols.beta)) / float(beta_scale)
    return {
        "alphas": alphas,
        "betas": betas,
        "alpha_grid": alpha_grid,
        "beta_grid": beta_grid,
        "u_grid": u_grid,
        "v_grid": v_grid,
        "sse_grid": sse_grid,
        "excess_grid": excess_grid,
        "alpha_ols": float(ols.alpha),
        "beta_ols": float(ols.beta),
        "sse_min": float(ols.sse),
        "alpha_scale": float(alpha_scale),
        "beta_scale": float(beta_scale),
        "domain": (float(a0), float(a1), float(b0), float(b1)),
        "n": n,
        "x_mean": x_mean,
        "sxx": sxx,
    }


def ols_fit(hours: np.ndarray, scores: np.ndarray):
    return fit_ols(hours, scores)


def animation_steps(
    start_alpha: float,
    start_beta: float,
    end_alpha: float,
    end_beta: float,
    count: int = 18,
) -> list[tuple[float, float]]:
    """Even steps from a candidate line to the OLS line."""
    if count < 2:
        return [(float(end_alpha), float(end_beta))]
    fractions = np.linspace(0.0, 1.0, int(count))
    steps = []
    for fraction in fractions:
        alpha = float(start_alpha) + float(fraction) * (float(end_alpha) - float(start_alpha))
        beta = float(start_beta) + float(fraction) * (float(end_beta) - float(start_beta))
        steps.append((alpha, beta))
    return steps


def animation_metric_trajectories(
    hours: np.ndarray,
    scores: np.ndarray,
    steps: list[tuple[float, float]],
    sst: float,
) -> tuple[np.ndarray, np.ndarray]:
    """SSE and R² at each (α, β) animation step — real metrics, not interpolated."""
    sses = np.empty(len(steps), dtype=np.float64)
    r2s = np.empty(len(steps), dtype=np.float64)
    for index, (alpha, beta) in enumerate(steps):
        sse = sum_of_squared_errors(hours, scores, alpha, beta)
        sses[index] = sse
        r2s[index] = r_squared(sse, sst)
    return sses, r2s


def subsample_observations(
    dataset: FittingDataset,
    n: int,
    rng: np.random.Generator,
) -> FittingDataset:
    """Draw n observations without replacement; keep metadata and labels."""
    size = int(dataset.n)
    lower = min(MIN_SAMPLE_N, size)
    count = int(np.clip(int(n), lower, size))
    if count >= size:
        return dataset
    indices = np.sort(rng.choice(size, size=count, replace=False))
    base_rows = dataset.row_indices
    if base_rows is None:
        row_indices = indices.copy()
    else:
        row_indices = np.asarray(base_rows, dtype=np.int64)[indices].copy()
    return FittingDataset(
        key=dataset.key,
        label=dataset.label,
        x=dataset.x[indices].copy(),
        y=dataset.y[indices].copy(),
        x_name=dataset.x_name,
        y_name=dataset.y_name,
        x_axis=dataset.x_axis,
        y_axis=dataset.y_axis,
        r_data=dataset.r_data,
        r_response=dataset.r_response,
        r_predictor=dataset.r_predictor,
        available_n=int(dataset.available_n),
        source=dataset.source,
        simulated=dataset.simulated,
        seed=dataset.seed,
        caveat=dataset.caveat,
        row_indices=row_indices,
    )


def poor_candidate_line(
    hours: np.ndarray,
    scores: np.ndarray,
    *,
    index: int = 0,
    alpha_bounds: tuple[float, float] | None = None,
    beta_bounds: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Visibly imperfect but on-chart candidate for the same fixed sample."""
    x = np.asarray(hours, dtype=np.float64)
    y = np.asarray(scores, dtype=np.float64)
    ols = fit_ols(x, y)
    if alpha_bounds is None or beta_bounds is None:
        dataset = FittingDataset(
            key="tmp",
            label="tmp",
            x=x,
            y=y,
            x_name="x",
            y_name="y",
            x_axis="x",
            y_axis="y",
            r_data="data",
            r_response="y",
            r_predictor="x",
            available_n=int(x.size),
            source="",
        )
        (a0, a1, _), (b0, b1, _) = dataset.slider_bounds()
        alpha_bounds = (a0, a1)
        beta_bounds = (b0, b1)
    a_min, a_max = alpha_bounds
    b_min, b_max = beta_bounds
    x_bar = float(np.mean(x))
    y_bar = float(np.mean(y))
    y_span = float(np.ptp(y)) if float(np.ptp(y)) > 0 else 1.0
    x_span = float(np.ptp(x)) if float(np.ptp(x)) > 0 else 1.0
    typical = y_span / x_span
    slope_step = max(0.28 * abs(float(ols.beta)), 0.22 * typical, 0.05)

    def clip(alpha: float, beta: float) -> tuple[float, float]:
        return (
            float(np.clip(alpha, a_min, a_max)),
            float(np.clip(beta, b_min, b_max)),
        )

    # Pass near the data cloud with a mild intercept offset so residuals stay readable.
    offsets = (0.12 * y_span, -0.12 * y_span, 0.2 * y_span, -0.2 * y_span)
    beta_shifts = (
        slope_step,
        -slope_step,
        0.55 * slope_step,
        -0.55 * slope_step,
        0.85 * slope_step,
        -0.85 * slope_step,
    )
    proposals: list[tuple[float, float]] = []
    for shift in beta_shifts:
        for offset in offsets:
            beta = float(ols.beta) + float(shift)
            alpha = (y_bar + float(offset)) - beta * x_bar
            proposals.append(clip(alpha, beta))

    ordered = [proposals[(int(index) + offset) % len(proposals)] for offset in range(len(proposals))]
    low = 1.35 * float(ols.sse)
    high = 4.0 * float(ols.sse)
    sst = float(np.dot(y - y_bar, y - y_bar))
    chosen = ordered[0]
    chosen_sse = sum_of_squared_errors(x, y, *chosen)
    for candidate in ordered:
        if abs(candidate[0] - ols.alpha) < 1e-9 and abs(candidate[1] - ols.beta) < 1e-9:
            continue
        sse = sum_of_squared_errors(x, y, *candidate)
        r2 = 1.0 - sse / sst if sst > 0 else float("nan")
        if low <= sse <= high and (not np.isfinite(r2) or r2 > -0.35):
            return candidate
        # Prefer moderate SSE over extreme values.
        if low <= sse < chosen_sse or (chosen_sse < low and sse > chosen_sse):
            if np.isfinite(r2) and r2 <= -0.35:
                continue
            chosen = candidate
            chosen_sse = sse
    return chosen
