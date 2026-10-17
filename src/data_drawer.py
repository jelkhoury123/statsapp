"""Tables, selection, and CSV export for the opening experiment.

The population file is only read. Choosing a row never changes the sample
or the fitted line. Student ID is the only identifier: a row's position on
screen is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.population import Population
from src.regression import RegressionResult
from src.sampling import Sample

EXPORT_COLUMNS = (
    "student_id",
    "hours_studied",
    "sleep_hours",
    "attendance",
    "exam_score",
)
PAGE_SIZE = 100

DRAWER_INTRO = (
    "Each row represents one student. "
    "Each point on the graph represents the same student."
)
SELECTION_NOTE = (
    "Changing the selected row does not change the regression. "
    "The regression uses all observations in the current sample."
)
EMPTY_SAMPLE = "Draw a sample to see its observations."
R_COMMANDS = """students <- read.csv("sample.csv")

model <- lm(
  exam_score ~ hours_studied,
  data = students
)

summary(model)"""

_POSITION_CACHE: dict[str, np.ndarray] = {}
_SORT_CACHE: dict[tuple[str, str, bool], np.ndarray] = {}
_CSV_CACHE: dict[str, bytes] = {}


@dataclass(frozen=True)
class DrawerSnapshot:
    """Drawer controls. These do not own the sample or the fitted line.

    ``selected_ids`` is ordered. The last ID is the one most recently chosen,
    and is the row the table scrolls to after a chart click.
    """

    open: bool = False
    dataset: str = "Population"
    selected_ids: tuple[int, ...] = ()

    @property
    def selected_id(self) -> int | None:
        if not self.selected_ids:
            return None
        return int(self.selected_ids[-1])


@dataclass(frozen=True)
class HighlightSpec:
    """One student, located by ID, and how the chart may mark them."""

    student_id: int
    hours_studied: float
    exam_score: float
    in_sample: bool
    predicted: float | None
    residual: float | None
    show_sample_residual: bool


def set_drawer_open(snapshot: DrawerSnapshot, open_drawer: bool) -> DrawerSnapshot:
    """Open or close the drawer without touching the experiment."""
    return DrawerSnapshot(open_drawer, snapshot.dataset, snapshot.selected_ids)


def toggle_student(selected_ids: tuple[int, ...], student_id: int) -> tuple[int, ...]:
    """Add a student, or remove them if they are already selected."""
    chosen = int(student_id)
    if chosen in selected_ids:
        return tuple(student for student in selected_ids if student != chosen)
    return tuple(selected_ids) + (chosen,)


def focus_student(selected_ids: tuple[int, ...], student_id: int) -> tuple[int, ...]:
    """Select a student and make them the most recently chosen."""
    chosen = int(student_id)
    return tuple(student for student in selected_ids if student != chosen) + (chosen,)


def state_after_chart_click(snapshot: DrawerSnapshot, student_id: int) -> DrawerSnapshot:
    """A clicked point selects that student and keeps everyone already chosen."""
    return DrawerSnapshot(
        True,
        snapshot.dataset,
        focus_student(snapshot.selected_ids, student_id),
    )


def select_observation(snapshot: DrawerSnapshot, student_id: int | None) -> DrawerSnapshot:
    """Remember one student. The sample and the fit are not arguments."""
    chosen = () if student_id is None else (int(student_id),)
    return DrawerSnapshot(snapshot.open, snapshot.dataset, chosen)


def selection_after_new_sample(
    selected_ids: tuple[int, ...] | int | None,
    sample: Sample,
) -> tuple[int, ...]:
    """Keep selected students when a new sample replaces the previous one.

    Membership in the sample is decided later, when the chart is drawn.
    A student who left the sample stays selected on the population cloud.
    """
    del sample
    if selected_ids is None:
        return ()
    if isinstance(selected_ids, (int, np.integer)) and not isinstance(selected_ids, bool):
        return (int(selected_ids),)
    return tuple(int(student) for student in selected_ids)


def position_of(population: Population, student_id: int) -> int | None:
    """Row of this student in the population arrays, or None."""
    index = _position_index(population)
    chosen = int(student_id)
    if chosen < 0 or chosen >= index.size:
        return None
    position = int(index[chosen])
    if position < 0:
        return None
    return position


def _position_index(population: Population) -> np.ndarray:
    cached = _POSITION_CACHE.get(population.sha256)
    if cached is not None:
        return cached
    ids = population.student_id
    index = np.full(int(ids.max()) + 1, -1, dtype=np.int32)
    index[ids] = np.arange(ids.size, dtype=np.int32)
    _POSITION_CACHE[population.sha256] = index
    return index


def highlight_for(
    population: Population,
    sample: Sample | None,
    fit: RegressionResult | None,
    student_id: int | None,
) -> HighlightSpec | None:
    """Locate a student and, when allowed, their sample residual.

    Coordinates always come from the population row for that ID. A student
    who is not in the sample can still be marked on the cloud. Their distance
    from the sample line is not reported as a sample residual. A prediction
    from that line is available so the two ideas stay distinct.
    """
    if student_id is None:
        return None
    position = position_of(population, student_id)
    if position is None:
        return None
    hours = float(population.hours_studied[position])
    score = float(population.exam_score[position])
    in_sample = sample is not None and bool(np.any(sample.student_id == int(student_id)))
    predicted = None
    residual = None
    show_sample_residual = False
    if fit is not None and not fit.degenerate and np.isfinite(fit.alpha) and np.isfinite(fit.beta):
        predicted = float(fit.alpha + fit.beta * hours)
        if in_sample:
            residual = float(score - predicted)
            show_sample_residual = True
    return HighlightSpec(
        student_id=int(student_id),
        hours_studied=hours,
        exam_score=score,
        in_sample=in_sample,
        predicted=predicted,
        residual=residual,
        show_sample_residual=show_sample_residual,
    )


def highlights_for(
    population: Population,
    sample: Sample | None,
    fit: RegressionResult | None,
    selected_ids: tuple[int, ...],
) -> list[HighlightSpec]:
    """One spec per selected student, in selection order."""
    specs: list[HighlightSpec] = []
    for student_id in selected_ids:
        spec = highlight_for(population, sample, fit, int(student_id))
        if spec is not None:
            specs.append(spec)
    return specs


def page_for_student(
    population: Population,
    student_id: int,
    sort_by: str,
    ascending: bool,
    page_size: int = PAGE_SIZE,
) -> int | None:
    """Zero-based page that contains this student under the current sort."""
    position = position_of(population, int(student_id))
    if position is None:
        return None
    order = _sorted_positions(population, sort_by, ascending)
    location = np.flatnonzero(order == position)
    if location.size == 0:
        return None
    return int(location[0]) // page_size


def sample_count_label(sample_size: int, population_size: int) -> str:
    return f"{sample_size:,} students selected from {population_size:,}"


def sample_token(sample: Sample | None) -> tuple[int, int, int, int] | None:
    """Identity of a sample, used to ignore a table selection from the previous one."""
    if sample is None:
        return None
    ids = np.asarray(sample.student_id)
    return (int(ids.size), int(ids[0]), int(ids[-1]), int(ids.sum()))


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(EXPORT_COLUMNS))


def sample_frame(sample: Sample) -> pd.DataFrame:
    """The current sample, one row per drawn student, in draw order."""
    return pd.DataFrame(
        {
            "student_id": np.asarray(sample.student_id),
            "hours_studied": np.asarray(sample.hours_studied, dtype=np.float64),
            "sleep_hours": np.asarray(sample.sleep_hours, dtype=np.float64),
            "attendance": np.asarray(sample.attendance, dtype=np.float64),
            "exam_score": np.asarray(sample.exam_score, dtype=np.float64),
        }
    )


def population_page(
    population: Population,
    *,
    student_query: str = "",
    sort_by: str = "student_id",
    ascending: bool = True,
    page: int = 0,
    page_size: int = PAGE_SIZE,
) -> tuple[pd.DataFrame, int, int]:
    """One page of the population, plus the match count and the page used.

    A student-ID query returns that student only. Other queries return no
    rows. The population arrays are not modified.
    """
    if sort_by not in EXPORT_COLUMNS:
        raise ValueError(f"Cannot sort the population by {sort_by}")
    if page_size < 1:
        raise ValueError("Page size must be at least 1")

    query = student_query.strip()
    if query:
        if not query.isdigit():
            return empty_frame(), 0, 0
        position = position_of(population, int(query))
        if position is None:
            return empty_frame(), 0, 0
        return _frame_at(population, np.array([position], dtype=np.int64)), 1, 0

    order = _sorted_positions(population, sort_by, ascending)
    match_count = int(order.size)
    page_count = max(1, (match_count + page_size - 1) // page_size)
    page_index = min(max(int(page), 0), page_count - 1)
    start = page_index * page_size
    positions = order[start : start + page_size]
    return _frame_at(population, positions), match_count, page_index


def _sorted_positions(population: Population, sort_by: str, ascending: bool) -> np.ndarray:
    key = (population.sha256, sort_by, ascending)
    cached = _SORT_CACHE.get(key)
    if cached is not None:
        return cached
    if sort_by == "student_id":
        values = population.student_id
    elif sort_by == "hours_studied":
        values = population.hours_studied
    elif sort_by == "sleep_hours":
        values = population.sleep_hours
    elif sort_by == "attendance":
        values = population.attendance
    else:
        values = population.exam_score
    order = np.argsort(values, kind="mergesort")
    if not ascending:
        order = order[::-1]
    order = np.asarray(order, dtype=np.int64)
    _SORT_CACHE[key] = order
    return order


def _frame_at(population: Population, positions: np.ndarray) -> pd.DataFrame:
    pos = np.asarray(positions, dtype=np.int64)
    return pd.DataFrame(
        {
            "student_id": population.student_id[pos],
            "hours_studied": population.hours_studied[pos],
            "sleep_hours": population.sleep_hours[pos],
            "attendance": population.attendance[pos],
            "exam_score": population.exam_score[pos],
        }
    )


def frame_to_csv(frame: pd.DataFrame) -> bytes:
    """CSV with the export columns R will read. No index and no error term."""
    ordered = frame.loc[:, list(EXPORT_COLUMNS)]
    return ordered.to_csv(index=False).encode("utf-8")


def population_csv(population: Population) -> bytes:
    """Export of every student, cached by the population file hash."""
    cached = _CSV_CACHE.get(population.sha256)
    if cached is not None:
        return cached
    frame = _frame_at(population, np.arange(population.n, dtype=np.int64))
    encoded = frame_to_csv(frame)
    _CSV_CACHE[population.sha256] = encoded
    return encoded


def sample_csv(sample: Sample) -> bytes:
    return frame_to_csv(sample_frame(sample))


def student_id_at(frame: pd.DataFrame, row: int) -> int | None:
    """Student ID at an original row position, not a sorted screen position."""
    if row < 0 or row >= len(frame):
        return None
    return int(frame.iloc[row]["student_id"])


def dataframe_rows(state: object) -> list[int]:
    """Row positions stored for a Streamlit dataframe selection."""
    if not isinstance(state, dict):
        return []
    selection = state.get("selection")
    if not isinstance(selection, dict):
        return []
    rows = selection.get("rows") or []
    return [int(row) for row in rows if isinstance(row, (int, np.integer)) and not isinstance(row, bool)]


def student_id_from_chart_point(point: object) -> int | None:
    """Student ID attached to a Plotly point, if the point has one."""
    if not isinstance(point, dict):
        return None
    custom = point.get("customdata")
    if isinstance(custom, (list, tuple, np.ndarray)):
        if len(custom) == 0:
            return None
        custom = custom[0]
    if isinstance(custom, bool) or custom is None:
        return None
    try:
        return int(custom)
    except (TypeError, ValueError):
        return None


def chart_student_id(state: object) -> int | None:
    """First clicked point that carries a student ID."""
    if not isinstance(state, dict):
        return None
    selection = state.get("selection")
    if not isinstance(selection, dict):
        return None
    points = selection.get("points") or []
    for point in points:
        student_id = student_id_from_chart_point(point)
        if student_id is not None:
            return student_id
    return None


def annotation_text(spec: HighlightSpec) -> str:
    return (
        f"Student {spec.student_id}<br>"
        f"Hours studied {spec.hours_studied:.2f}<br>"
        f"Exam score {spec.exam_score:.1f}"
    )
