"""Plotly figures for the regression lab.

Colours are consistent across tabs: blue for a sample estimate, grey for
the population line, orange for residuals and for the centre of a
sampling distribution.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
from PIL import Image
from scipy.stats import gaussian_kde

from src.sampling_distributions import histogram_density
from src.stat_tips import tip_hover

BLUE = "#1D4ED8"
BLUE_POINT = "#60A5FA"
GREY = "#6B7280"
ORANGE = "#EA580C"
INK = "#1F2933"
MUTED = "#52606D"
GRID = "#E6EAF0"
FONT = "-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif"

# Okabe–Ito, in the order used for sample-size and noise overlays.
SERIES_COLOURS = ("#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7")

LINE_X = np.array([0.0, 10.0])
POINT_LIMIT = 800
RESIDUAL_POINT_LIMIT = 350
_CLOUD_WIDTH = 1400
_CLOUD_HEIGHT = 820
_CLOUD_VERSION = 3
_CLOUD_DOT_OPACITY = 0.10
_CLOUD_GREY = np.array([168.0, 176.0, 188.0], dtype=np.float32)
_CLOUD_CACHE: dict[tuple, str] = {}


def rgba(hex_colour: str, alpha: float) -> str:
    colour = hex_colour.lstrip("#")
    red, green, blue = (int(colour[index : index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red}, {green}, {blue}, {alpha})"


def _apply_layout(
    fig: go.Figure,
    *,
    height: int,
    title: str | None,
    legend_below: bool = False,
) -> go.Figure:
    # A title and a legend both want the top of the figure. When there is a
    # title, the legend sits under the x-axis so the two never collide.
    if legend_below:
        legend = dict(
            orientation="h",
            yanchor="top",
            y=-0.28,
            x=0,
            xanchor="left",
            bgcolor="rgba(255,255,255,0)",
            font=dict(size=12),
        )
        margin = dict(l=56, r=20, t=68, b=96)
    else:
        legend = dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            xanchor="left",
            bgcolor="rgba(255,255,255,0)",
            font=dict(size=12),
        )
        margin = dict(l=56, r=20, t=48, b=48)
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=dict(text=title or "", x=0, xanchor="left", font=dict(size=16, color=INK)),
        margin=margin,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(family=FONT, color=INK, size=13),
        legend=legend,
        hoverlabel=dict(bgcolor="white", font_size=12, font_family=FONT),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        linecolor="#D9E2EC",
        title_font=dict(size=13),
        tickfont=dict(size=12),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        linecolor="#D9E2EC",
        title_font=dict(size=13),
        tickfont=dict(size=12),
    )
    return fig


def empty_figure(message: str, height: int = 520) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        showarrow=False,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        font=dict(size=15, color=MUTED, family=FONT),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _apply_layout(fig, height=height, title=None)


def display_positions(count: int, limit: int, stable_key: int) -> np.ndarray:
    """Indices of a stable subset, used only for drawing.

    The regression is never refit on this subset. The key is derived from
    the sample, so toggling a chart option does not reshuffle the points.
    """
    if count <= limit:
        return np.arange(count)
    generator = np.random.default_rng(np.random.SeedSequence([stable_key, count, limit]))
    return np.sort(generator.choice(count, size=limit, replace=False))


def stable_key_from_ids(student_ids: np.ndarray) -> int:
    ids = np.asarray(student_ids)
    if ids.size == 0:
        return 0
    return int(ids[0]) + 17 * int(ids[-1]) + 29 * int(ids[ids.size // 2]) + 101 * int(ids.size)


def sample_scatter_figure(
    *,
    x: np.ndarray | None,
    y: np.ndarray | None,
    sample_alpha: float | None,
    sample_beta: float | None,
    population_alpha: float,
    population_beta: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    show_population: bool,
    show_sample: bool,
    show_residuals: bool,
    student_ids: np.ndarray | None = None,
    height: int = 560,
) -> go.Figure:
    """Scatter of one sample, with optional lines and residual segments."""
    fig = go.Figure()
    has_sample = x is not None and y is not None and len(x) > 0
    plot_limit = RESIDUAL_POINT_LIMIT if show_residuals else POINT_LIMIT

    if has_sample:
        x_values = np.asarray(x, dtype=np.float64)
        y_values = np.asarray(y, dtype=np.float64)
        if student_ids is None:
            positions = display_positions(x_values.size, plot_limit, x_values.size)
        else:
            positions = display_positions(
                x_values.size,
                plot_limit,
                stable_key_from_ids(student_ids),
            )
        x_shown = x_values[positions]
        y_shown = y_values[positions]
        point_size = 8 if x_shown.size <= 80 else 6 if x_shown.size <= 300 else 5
        point_opacity = 0.9 if x_shown.size <= 120 else 0.62

        if (
            show_residuals
            and sample_alpha is not None
            and sample_beta is not None
            and np.isfinite(sample_alpha)
            and np.isfinite(sample_beta)
        ):
            fitted = sample_alpha + sample_beta * x_shown
            segment_x = np.column_stack(
                [x_shown, x_shown, np.full(x_shown.shape, np.nan)]
            ).ravel()
            segment_y = np.column_stack(
                [y_shown, fitted, np.full(y_shown.shape, np.nan)]
            ).ravel()
            fig.add_trace(
                go.Scatter(
                    x=segment_x,
                    y=segment_y,
                    mode="lines",
                    name="Residuals",
                    line=dict(color=ORANGE, width=1.4),
                    hoverinfo="skip",
                    legendrank=4,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=x_shown,
                y=y_shown,
                mode="markers",
                name="Sampled students",
                marker=dict(
                    size=point_size,
                    color=BLUE_POINT,
                    opacity=point_opacity,
                    line=dict(width=0),
                ),
                hovertemplate="Hours %{x:.2f}<br>Score %{y:.1f}<extra></extra>",
                legendrank=1,
            )
        )

        if (
            show_sample
            and sample_alpha is not None
            and sample_beta is not None
            and np.isfinite(sample_alpha)
            and np.isfinite(sample_beta)
        ):
            fig.add_trace(
                go.Scatter(
                    x=LINE_X,
                    y=sample_alpha + sample_beta * LINE_X,
                    mode="lines",
                    name="Sample regression",
                    line=dict(color=BLUE, width=2.8),
                    hovertemplate="Sample line<br>Hours %{x:.1f}<br>Score %{y:.1f}<extra></extra>",
                    legendrank=2,
                )
            )

    if show_population and np.isfinite(population_alpha) and np.isfinite(population_beta):
        fig.add_trace(
            go.Scatter(
                x=LINE_X,
                y=population_alpha + population_beta * LINE_X,
                mode="lines",
                name="Population regression",
                line=dict(color=GREY, width=2.4, dash="dash"),
                hovertemplate="Population line<br>Hours %{x:.1f}<br>Score %{y:.1f}<extra></extra>",
                legendrank=3,
            )
        )

    if not has_sample:
        fig.add_annotation(
            text="No sample drawn",
            xref="paper",
            yref="paper",
            x=0.0,
            y=1.08,
            showarrow=False,
            xanchor="left",
            font=dict(size=13, color=MUTED, family=FONT),
        )

    fig.update_xaxes(title_text="Hours studied", range=list(x_range))
    fig.update_yaxes(title_text="Exam score", range=list(y_range))
    return _apply_layout(
        fig,
        height=height,
        title="Hours studied and exam score",
        legend_below=True,
    )


def population_cloud_image(
    hours: np.ndarray,
    scores: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    cache_key: str,
) -> str:
    """A cached picture of every student, aligned to the chart axes.

    The picture is built from the complete population once. Later interactions
    reuse it, so the page does not resend 100,000 coordinates.
    """
    key = (
        cache_key,
        (float(x_range[0]), float(x_range[1])),
        (float(y_range[0]), float(y_range[1])),
        _CLOUD_WIDTH,
        _CLOUD_HEIGHT,
        _CLOUD_VERSION,
    )
    cached = _CLOUD_CACHE.get(key)
    if cached is not None:
        return cached
    image = _render_population_cloud(hours, scores, x_range, y_range)
    _CLOUD_CACHE[key] = image
    return image


def _render_population_cloud(
    hours: np.ndarray,
    scores: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> str:
    """One light-grey dot per student, about 1–1.5 px, with no outline.

    Each dot is drawn at low opacity. Where students overlap, the dots
    stay translucent instead of turning into a solid mass. Sample points
    and regression lines are drawn later, on top of this picture.
    """
    width = _CLOUD_WIDTH
    height = _CLOUD_HEIGHT
    x_values = np.asarray(hours, dtype=np.float64)
    y_values = np.asarray(scores, dtype=np.float64)
    x0, x1 = float(x_range[0]), float(x_range[1])
    y0, y1 = float(y_range[0]), float(y_range[1])
    cols = np.floor((x_values - x0) / (x1 - x0) * (width - 1)).astype(np.int32)
    # Row 0 is the top of the image, which is the high end of exam score.
    rows = np.floor((y1 - y_values) / (y1 - y0) * (height - 1)).astype(np.int32)
    inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    counts = np.zeros((height, width), dtype=np.uint16)
    # A 2×2 block is about 1–1.5 px once the picture is fitted to the chart.
    np.add.at(counts, (rows[inside], cols[inside]), 1)
    col2 = cols + 1
    row2 = rows + 1
    ok_col = inside & (col2 < width)
    np.add.at(counts, (rows[ok_col], col2[ok_col]), 1)
    ok_row = inside & (row2 < height)
    np.add.at(counts, (row2[ok_row], cols[ok_row]), 1)
    ok_both = inside & (col2 < width) & (row2 < height)
    np.add.at(counts, (row2[ok_both], col2[ok_both]), 1)

    alpha = 1.0 - np.power(1.0 - _CLOUD_DOT_OPACITY, counts.astype(np.float32))
    white = np.array([255.0, 255.0, 255.0], dtype=np.float32)
    mixed = white * (1.0 - alpha[..., None]) + _CLOUD_GREY * alpha[..., None]
    canvas = mixed.astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(canvas, mode="RGB").save(buffer, format="PNG", compress_level=3)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _add_selected_students(
    fig: go.Figure,
    *,
    selected_ids: np.ndarray,
    selected_x: np.ndarray,
    selected_y: np.ndarray,
    selected_in_sample: np.ndarray,
    show_residual: bool,
    sample_alpha: float | None,
    sample_beta: float | None,
    y_range: tuple[float, float],
) -> None:
    """Mark every selected student. Coordinates are the population values."""
    ids = np.asarray(selected_ids)
    xs = np.asarray(selected_x, dtype=np.float64)
    ys = np.asarray(selected_y, dtype=np.float64)
    membership = np.asarray(selected_in_sample, dtype=bool)
    if ids.size == 0 or xs.size == 0 or ys.size == 0:
        return
    finite = np.isfinite(xs) & np.isfinite(ys)
    ids, xs, ys, membership = ids[finite], xs[finite], ys[finite], membership[finite]
    if ids.size == 0:
        return

    focus = int(ids.size - 1)
    if (
        show_residual
        and bool(membership[focus])
        and sample_alpha is not None
        and sample_beta is not None
        and np.isfinite(sample_alpha)
        and np.isfinite(sample_beta)
    ):
        predicted = float(sample_alpha + sample_beta * xs[focus])
        fig.add_trace(
            go.Scatter(
                x=[xs[focus], xs[focus]],
                y=[ys[focus], predicted],
                mode="lines",
                name="Residual",
                line=dict(color=ORANGE, width=2.6),
                hovertemplate=(
                    "Residual<br>Hours %{x:.2f}<br>Score %{y:.1f}<extra></extra>"
                ),
                legendrank=5,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            name="Selected student",
            marker=dict(size=18, color=BLUE, line=dict(width=3, color="#FFFFFF")),
            customdata=np.asarray(ids, dtype=int).reshape(-1, 1),
            hovertemplate="Student %{customdata[0]}<br>Hours %{x:.2f}<br>Score %{y:.1f}<extra></extra>",
            legendrank=0,
        )
    )
    focus_y = float(ys[focus])
    midpoint = (float(y_range[0]) + float(y_range[1])) / 2
    label_shift = 48 if focus_y > midpoint else -48
    fig.add_annotation(
        x=float(xs[focus]),
        y=focus_y,
        text=(
            f"Student {int(ids[focus])}<br>"
            f"Hours studied {float(xs[focus]):.2f}<br>"
            f"Exam score {focus_y:.1f}"
        ),
        showarrow=True,
        arrowhead=0,
        arrowwidth=1,
        arrowcolor=BLUE,
        ax=36,
        ay=label_shift,
        bgcolor="white",
        bordercolor=BLUE,
        borderwidth=1,
        borderpad=4,
        font=dict(size=12, color=INK, family=FONT),
        align="left",
    )


def introduction_figure(
    *,
    cloud_image: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    sample_x: np.ndarray | None,
    sample_y: np.ndarray | None,
    sample_alpha: float | None,
    sample_beta: float | None,
    population_alpha: float,
    population_beta: float,
    show_sample_line: bool,
    show_population_line: bool,
    sample_ids: np.ndarray | None = None,
    selected_id: int | None = None,
    selected_x: float | None = None,
    selected_y: float | None = None,
    selected_in_sample: bool = False,
    selected_ids: np.ndarray | None = None,
    selected_xs: np.ndarray | None = None,
    selected_ys: np.ndarray | None = None,
    selected_membership: np.ndarray | None = None,
    show_residual: bool = False,
    height: int = 680,
    show_title: bool = True,
) -> go.Figure:
    """Population cloud, with the current sample and lines drawn on top.

    The cloud is a cached image of every student. Only the sample coordinates
    are sent as points. Lines use the coefficients the caller supplies.
    """
    fig = go.Figure()
    fig.add_layout_image(
        source=cloud_image,
        xref="x",
        yref="y",
        x=x_range[0],
        y=y_range[1],
        sizex=x_range[1] - x_range[0],
        sizey=y_range[1] - y_range[0],
        sizing="stretch",
        layer="below",
        xanchor="left",
        yanchor="top",
        opacity=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="All students",
            marker=dict(size=9, color="#A8B0BC", symbol="circle"),
            hoverinfo="skip",
            legendrank=1,
        )
    )

    has_sample = sample_x is not None and sample_y is not None and len(sample_x) > 0
    if has_sample:
        x_values = np.asarray(sample_x, dtype=np.float64)
        y_values = np.asarray(sample_y, dtype=np.float64)
        count = int(x_values.size)
        if count <= 30:
            marker_size = 11
        elif count <= 100:
            marker_size = 8
        elif count <= 500:
            marker_size = 6
        else:
            marker_size = 4
        custom = None if sample_ids is None else np.asarray(sample_ids)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers",
                name="Sampled students",
                marker=dict(
                    size=marker_size,
                    color=BLUE,
                    opacity=0.92,
                    line=dict(width=0.6, color="white"),
                ),
                customdata=custom,
                hovertemplate=(
                    "Student %{customdata}<br>Hours %{x:.2f}<br>Score %{y:.1f}<extra></extra>"
                    if custom is not None
                    else "Hours %{x:.2f}<br>Score %{y:.1f}<extra></extra>"
                ),
                legendrank=2,
            )
        )

    if (
        show_population_line
        and np.isfinite(population_alpha)
        and np.isfinite(population_beta)
    ):
        fig.add_trace(
            go.Scatter(
                x=LINE_X,
                y=population_alpha + population_beta * LINE_X,
                mode="lines",
                name="Population regression",
                line=dict(color=ORANGE, width=3.2),
                hovertemplate="Population line<br>Hours %{x:.1f}<br>Score %{y:.1f}<extra></extra>",
                legendrank=3,
            )
        )

    if (
        has_sample
        and show_sample_line
        and sample_alpha is not None
        and sample_beta is not None
        and np.isfinite(sample_alpha)
        and np.isfinite(sample_beta)
    ):
        fig.add_trace(
            go.Scatter(
                x=LINE_X,
                y=sample_alpha + sample_beta * LINE_X,
                mode="lines",
                name="Sample regression",
                line=dict(color=BLUE, width=2.8),
                hovertemplate="Sample line<br>Hours %{x:.1f}<br>Score %{y:.1f}<extra></extra>",
                legendrank=4,
            )
        )

    if selected_ids is None and selected_id is not None and selected_x is not None and selected_y is not None:
        selected_ids = np.asarray([selected_id])
        selected_xs = np.asarray([selected_x], dtype=np.float64)
        selected_ys = np.asarray([selected_y], dtype=np.float64)
        selected_membership = np.asarray([selected_in_sample])
    if selected_ids is not None and selected_xs is not None and selected_ys is not None:
        membership = (
            np.zeros(len(selected_ids), dtype=bool)
            if selected_membership is None
            else np.asarray(selected_membership, dtype=bool)
        )
        _add_selected_students(
            fig,
            selected_ids=np.asarray(selected_ids),
            selected_x=np.asarray(selected_xs, dtype=np.float64),
            selected_y=np.asarray(selected_ys, dtype=np.float64),
            selected_in_sample=membership,
            show_residual=show_residual,
            sample_alpha=sample_alpha,
            sample_beta=sample_beta,
            y_range=y_range,
        )

    fig.update_xaxes(
        title_text="Hours studied",
        range=list(x_range),
        fixedrange=True,
    )
    fig.update_yaxes(
        title_text="Exam score",
        range=list(y_range),
        fixedrange=True,
    )
    fig = _apply_layout(
        fig,
        height=height,
        title="Hours studied and exam score" if show_title else None,
        legend_below=True,
    )
    if not show_title:
        # The page supplies the title and the legend, so the plot keeps a fixed height.
        fig.update_layout(showlegend=False, margin=dict(l=46, r=8, t=4, b=34))
    # The cloud image is the plot background. A solid white panel would cover it.
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
    return fig


def _finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def histogram_bins(values: np.ndarray) -> int:
    """Freedman–Diaconis bin count, kept in a range that stays readable."""
    clean = _finite(values)
    count = int(clean.size)
    if count < 2:
        return 1
    span = float(clean.max() - clean.min())
    if span <= 0:
        return 1
    quartile_75, quartile_25 = np.percentile(clean, [75, 25])
    iqr = float(quartile_75 - quartile_25)
    if iqr > 0:
        width = 2.0 * iqr * count ** (-1.0 / 3.0)
    else:
        width = 3.5 * float(np.std(clean, ddof=1)) * count ** (-1.0 / 3.0)
    if width <= 0:
        return 30
    return int(np.clip(np.ceil(span / width), 18, 60))


def _tick_format(span: float) -> str:
    if span < 0.05:
        return ".4f"
    if span < 0.5:
        return ".3f"
    if span < 5:
        return ".2f"
    return ".1f"


def _focused_range(values: np.ndarray, extras: Sequence[float]) -> tuple[float, float]:
    clean = _finite(values)
    points = [float(value) for value in extras if value is not None and np.isfinite(value)]
    if clean.size:
        points.extend([float(clean.min()), float(clean.max())])
    if not points:
        return (-1.0, 1.0)
    low = min(points)
    high = max(points)
    span = high - low
    if span <= 0:
        span = max(abs(low) * 0.08, 0.05)
        low -= span
        high += span
    else:
        pad = 0.08 * span
        low -= pad
        high += pad
    return low, high


def coefficient_histogram(
    estimates: np.ndarray,
    *,
    population: float | None,
    show_population: bool,
    title: str,
    x_title: str,
    height: int = 440,
) -> go.Figure:
    """Histogram of one coefficient, zoomed to the estimates."""
    values = _finite(estimates)
    fig = go.Figure()
    if values.size == 0:
        return empty_figure("No estimates to show.", height=height)

    mean = float(values.mean())
    std = float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
    extras: list[float] = [mean]
    if show_population and population is not None and np.isfinite(population):
        extras.append(float(population))
    low, high = _focused_range(values, extras)
    bins = histogram_bins(values)

    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=bins,
            name="Samples",
            marker=dict(color=rgba(BLUE, 0.78), line=dict(width=0)),
            hovertemplate=f"{x_title} %{{x:.3f}}<br>Samples %{{y}}<extra></extra>",
            legendrank=1,
        )
    )

    if np.isfinite(std) and std > 0:
        fig.add_vrect(
            x0=mean - std,
            x1=mean + std,
            fillcolor=rgba(ORANGE, 0.14),
            line_width=0,
            layer="below",
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name="Mean ± 1 SD",
                line=dict(color=rgba(ORANGE, 0.45), width=8),
                legendrank=4,
            )
        )

    fig.add_vline(x=mean, line_color=ORANGE, line_width=2)
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="Mean of estimates",
            line=dict(color=ORANGE, width=2),
            legendrank=2,
        )
    )

    subtitle = f"Mean {mean:.3f}"
    if np.isfinite(std):
        subtitle += f"  ·  SD {std:.3f}"
    if show_population and population is not None and np.isfinite(population):
        fig.add_vline(x=float(population), line_color=GREY, line_width=2, line_dash="dash")
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name="Population value",
                line=dict(color=GREY, width=2, dash="dash"),
                legendrank=3,
            )
        )
        subtitle += f"  ·  Population {population:.3f}"

    fig.update_xaxes(title_text=x_title, range=[low, high], tickformat=_tick_format(high - low))
    fig.update_yaxes(title_text="Samples")
    fig.update_layout(bargap=0.04)
    full_title = f"{title}<br><sup>{subtitle}</sup>"
    return _apply_layout(fig, height=height, title=full_title, legend_below=True)


def _series_range(series: Sequence[tuple[str, np.ndarray, str]], extras: Sequence[float]) -> tuple[float, float]:
    parts = [_finite(values) for _, values, _ in series]
    parts = [part for part in parts if part.size]
    if not parts and not extras:
        return (-1.0, 1.0)
    values = np.concatenate(parts) if parts else np.array([], dtype=np.float64)
    extra = np.asarray([value for value in extras if np.isfinite(value)], dtype=np.float64)
    if extra.size:
        values = np.concatenate([values, extra]) if values.size else extra
    low = float(values.min())
    high = float(values.max())
    if high == low:
        return low - 1.0, high + 1.0
    pad = 0.05 * (high - low)
    return low - pad, high + pad


def overlay_density_figure(
    series: Sequence[tuple[str, np.ndarray, str]],
    *,
    x_title: str,
    title: str,
    benchmark: float | None = None,
    benchmark_name: str | None = None,
    height: int = 420,
) -> go.Figure:
    """Overlaid density curves on one shared horizontal axis."""
    extras = []
    if benchmark is not None and np.isfinite(benchmark):
        extras.append(float(benchmark))
    low, high = _series_range(series, extras)
    grid = np.linspace(low, high, 450)
    fig = go.Figure()

    for label, values, colour in series:
        clean = _finite(values)
        if clean.size < 2 or float(np.std(clean, ddof=1)) == 0:
            continue
        density = gaussian_kde(clean)
        y_values = np.clip(density(grid), 0.0, None)
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=y_values,
                mode="lines",
                name=label,
                line=dict(color=colour, width=2.4),
                fill="tozeroy",
                fillcolor=rgba(colour, 0.16),
                hovertemplate=f"{label}<br>{x_title} %{{x:.3f}}<br>Density %{{y:.2f}}<extra></extra>",
            )
        )

    if benchmark is not None and np.isfinite(benchmark):
        fig.add_vline(x=float(benchmark), line_color=GREY, line_width=2, line_dash="dash")
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=benchmark_name or "Benchmark",
                line=dict(color=GREY, width=2, dash="dash"),
            )
        )

    fig.update_xaxes(title_text=x_title, range=[low, high], tickformat=_tick_format(high - low))
    fig.update_yaxes(title_text="Density")
    return _apply_layout(fig, height=height, title=title, legend_below=True)


def sd_versus_n_figure(
    sample_sizes: Sequence[int],
    standard_deviations: Sequence[float],
    *,
    title: str,
    y_title: str,
    height: int = 380,
) -> go.Figure:
    """Empirical standard deviation against sample size, on a log horizontal axis."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(sample_sizes),
            y=list(standard_deviations),
            mode="lines+markers",
            name="Empirical SD",
            line=dict(color=BLUE, width=2.4),
            marker=dict(size=9, color=BLUE),
            hovertemplate="n = %{x}<br>SD %{y:.3f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_xaxes(
        title_text="Sample size",
        type="log",
        tickmode="array",
        tickvals=list(sample_sizes),
        ticktext=[str(size) for size in sample_sizes],
    )
    fig.update_yaxes(title_text=y_title, rangemode="tozero")
    return _apply_layout(fig, height=height, title=title)


def distribution_scatter(
    *,
    cloud_image: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    population_alpha: float,
    population_beta: float,
    sample_x: np.ndarray | None,
    sample_y: np.ndarray | None,
    sample_alpha: float | None,
    sample_beta: float | None,
    height: int = 520,
    show_residuals: bool = False,
    residual_sd: float | None = None,
    show_residual_band: bool = False,
    residual_indices: np.ndarray | None = None,
) -> go.Figure:
    """Population cloud, the latest sample, and both regression lines."""
    fig = go.Figure()
    fig.add_layout_image(
        source=cloud_image,
        xref="x",
        yref="y",
        x=x_range[0],
        y=y_range[1],
        sizex=x_range[1] - x_range[0],
        sizey=y_range[1] - y_range[0],
        sizing="stretch",
        layer="below",
        xanchor="left",
        yanchor="top",
        opacity=1,
    )
    has_line = (
        sample_alpha is not None
        and sample_beta is not None
        and np.isfinite(sample_alpha)
        and np.isfinite(sample_beta)
    )
    if (
        has_line
        and residual_sd is not None
        and np.isfinite(residual_sd)
        and residual_sd > 0
    ):
        # Keep the band mounted so show/hide does not remount the chart.
        fitted = sample_alpha + sample_beta * LINE_X
        band_alpha = 0.12 if show_residual_band else 0.0
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([LINE_X, LINE_X[::-1]]),
                y=np.concatenate([fitted + residual_sd, (fitted - residual_sd)[::-1]]),
                fill="toself",
                mode="lines",
                name="±1 residual SD",
                line=dict(width=0),
                fillcolor=f"rgba(29, 78, 216, {band_alpha})",
                opacity=1.0 if show_residual_band else 0.0,
                hoverinfo="skip",
            )
        )
        if show_residual_band:
            fig.add_annotation(
                x=float(LINE_X[-1]),
                y=float(fitted[-1] + residual_sd),
                xanchor="right",
                yanchor="bottom",
                text="±1 residual SD",
                showarrow=False,
                font=dict(size=11, color=BLUE, family=FONT),
            )
    if has_line and sample_x is not None and sample_y is not None and len(sample_x):
        indices = np.arange(len(sample_x)) if residual_indices is None else residual_indices
        x_shown = np.asarray(sample_x, dtype=np.float64)[indices]
        y_shown = np.asarray(sample_y, dtype=np.float64)[indices]
        fitted_y = sample_alpha + sample_beta * x_shown
        segment_x = np.column_stack([x_shown, x_shown, np.full(x_shown.shape, np.nan)]).ravel()
        segment_y = np.column_stack([y_shown, fitted_y, np.full(y_shown.shape, np.nan)]).ravel()
        fig.add_trace(
            go.Scatter(
                x=segment_x,
                y=segment_y,
                mode="lines",
                name="Residuals",
                line=dict(color=ORANGE, width=2),
                opacity=0.9 if show_residuals else 0.0,
                hoverinfo="skip",
            )
        )
    if sample_x is not None and sample_y is not None and len(sample_x):
        count = int(len(sample_x))
        if count <= 30:
            marker_size = 9
        elif count <= 100:
            marker_size = 7
        else:
            marker_size = 5
        fig.add_trace(
            go.Scatter(
                x=sample_x,
                y=sample_y,
                mode="markers",
                name="Current sample",
                marker=dict(size=marker_size, color=BLUE, line=dict(width=0.4, color="white")),
                hovertemplate="Hours %{x:.2f}<br>Score %{y:.1f}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=LINE_X,
            y=population_alpha + population_beta * LINE_X,
            mode="lines",
            name="Population regression",
            line=dict(color=ORANGE, width=3),
            hovertemplate="Population line<br>%{y:.1f}<extra></extra>",
        )
    )
    if has_line:
        fig.add_trace(
            go.Scatter(
                x=LINE_X,
                y=sample_alpha + sample_beta * LINE_X,
                mode="lines",
                name="Sample regression",
                line=dict(color=BLUE, width=2.6),
                hovertemplate="Sample line<br>%{y:.1f}<extra></extra>",
            )
        )
    fig.update_xaxes(title_text="Hours studied", range=list(x_range))
    fig.update_yaxes(title_text="Exam score", range=list(y_range))
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=46, r=8, t=8, b=36),
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="rgba(0,0,0,0)",
        uirevision="distribution-scatter",
    )
    return fig


def _se_label_rows(
    left: float,
    center: float,
    right: float,
    population: float,
    span: float,
) -> list[int]:
    """Keep the three numbers apart, including when the mean sits on the truth.

    The boundary labels stay on the axis. The mean steps up only when it
    would collide with a boundary label or with the population line.
    """
    label = 0.09 * span
    rows = [0, 0, 0]
    mean_crowded = (
        abs(center - left) < label
        or abs(right - center) < label
        or abs(center - population) < label * 0.7
    )
    if mean_crowded:
        rows[1] = 1
    if abs(right - left) < label:
        rows = [0, 1, 2]
    return rows


def _add_se_value_label(
    fig: go.Figure,
    value: float,
    row: int,
    x_range: tuple[float, float],
    decimals: int,
    color: str = BLUE,
) -> None:
    x0, x1 = x_range
    shown = min(max(value, x0), x1)
    fraction = (shown - x0) / (x1 - x0) if x1 > x0 else 0.5
    if fraction < 0.08:
        anchor = "left"
    elif fraction > 0.92:
        anchor = "right"
    else:
        anchor = "center"
    fig.add_annotation(
        x=shown,
        y=0,
        yanchor="bottom",
        yshift=4 + row * 16,
        text=f"{value:.{decimals}f}",
        showarrow=False,
        xanchor=anchor,
        bgcolor="rgba(255,255,255,0.94)",
        borderpad=1,
        font=dict(size=13, color=color, family=FONT),
    )


def _add_standard_error_band(
    fig: go.Figure,
    mean: float | None,
    standard_error: float | None,
    x_range: tuple[float, float],
    population: float,
    decimals: int,
    se_label: str = "±1 SE",
) -> None:
    """Light band from the empirical mean minus one SE to plus one SE.

    The band sits behind the bars. It is omitted when the standard error
    is undefined, which is the case for fewer than two estimates.
    """
    if mean is None or standard_error is None:
        return
    if not (np.isfinite(mean) and np.isfinite(standard_error) and standard_error > 0):
        return
    left = float(mean - standard_error)
    right = float(mean + standard_error)
    center = float(mean)
    fig.add_shape(
        type="rect",
        x0=left,
        x1=right,
        y0=0,
        y1=1,
        yref="paper",
        fillcolor="rgba(29, 78, 216, 0.28)",
        line_width=0,
        layer="below",
    )
    fig.add_vline(x=center, line_color=BLUE, line_width=1.2)
    for edge in (left, right):
        fig.add_vline(
            x=edge,
            line_color=BLUE,
            line_width=1,
            line_dash="dash",
        )
    x0, x1 = x_range
    span = x1 - x0
    if span <= 0:
        return
    rows = _se_label_rows(left, center, right, float(population), span)
    for value, row in zip((left, center, right), rows):
        _add_se_value_label(fig, value, row, x_range, decimals)
    visible_left = min(max(left, x0), x1)
    visible_right = min(max(right, x0), x1)
    if visible_right <= visible_left:
        return
    fig.add_annotation(
        x=(visible_left + visible_right) / 2,
        y=1,
        yref="paper",
        yanchor="top",
        yshift=-8,
        text=se_label,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.94)",
        borderpad=3,
        font=dict(size=14, color=BLUE, family=FONT),
    )


def sampling_histogram(
    *,
    values: np.ndarray,
    edges: np.ndarray,
    population: float,
    title: str,
    x_title: str,
    normal_x: np.ndarray | None = None,
    normal_y: np.ndarray | None = None,
    normal_visible: bool = True,
    highlight: float | None = None,
    mean: float | None = None,
    standard_error: float | None = None,
    decimals: int = 3,
    height: int = 280,
    se_label: str = "±1 SE",
    uirevision: str | None = None,
) -> go.Figure:
    """Frequency histogram with the population coefficient and a ±1 SE band."""
    counts, _ = np.histogram(np.asarray(values, dtype=np.float64)[np.isfinite(values)], bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    width = float(edges[1] - edges[0])
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=centers,
            y=counts,
            width=width * 0.9,
            name="Samples",
            marker=dict(color=GREY, line=dict(width=0)),
            hovertemplate=f"{x_title} %{{x:.3f}}<br>Frequency %{{y}}<extra></extra>",
        )
    )
    if normal_x is not None and normal_y is not None:
        # Keep the trace mounted when hidden so toggling does not remount the chart.
        fig.add_trace(
            go.Scatter(
                x=normal_x,
                y=normal_y,
                mode="lines",
                name="Normal approximation",
                line=dict(color=INK, width=2.2, dash="dash"),
                opacity=1.0 if normal_visible else 0.0,
                hovertemplate=(
                    "Normal approximation<br>%{y:.1f}<extra></extra>"
                    if normal_visible
                    else None
                ),
                hoverinfo="skip" if not normal_visible else None,
            )
        )
    x_range = (float(edges[0]), float(edges[-1]))
    _add_standard_error_band(
        fig, mean, standard_error, x_range, float(population), decimals, se_label=se_label
    )
    if highlight is not None and np.isfinite(highlight):
        fig.add_vline(x=float(highlight), line_color=BLUE, line_width=2)
    fig.add_vline(x=float(population), line_color=ORANGE, line_width=2.4)
    ymax = float(counts.max()) if counts.size else 1.0
    if normal_y is not None and len(normal_y):
        ymax = max(ymax, float(np.nanmax(normal_y)))
    fig.update_xaxes(
        title_text=x_title,
        range=[float(edges[0]), float(edges[-1])],
        tickfont=dict(color=GREY, size=11, family=FONT),
        title_font=dict(color=INK, size=12, family=FONT),
    )
    fig.update_yaxes(title_text="Frequency", rangemode="tozero", range=[0, max(ymax * 1.34, 1)])
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=INK)),
        margin=dict(l=48, r=12, t=46, b=40),
        showlegend=False,
        bargap=0,
        paper_bgcolor="white",
        plot_bgcolor="white",
        uirevision=uirevision if uirevision is not None else "sampling-histogram",
    )
    return fig


def comparison_histogram(
    *,
    values: np.ndarray,
    edges: np.ndarray,
    population: float,
    title: str,
    x_title: str,
    decimals: int,
    reference_mean: float | None = None,
    reference_sd: float | None = None,
    reference_normal_x: np.ndarray | None = None,
    reference_normal_y: np.ndarray | None = None,
    estimate: float | None = None,
    standard_error: float | None = None,
    sample_normal_x: np.ndarray | None = None,
    sample_normal_y: np.ndarray | None = None,
    show_estimate_band: bool = True,
    height: int = 210,
) -> go.Figure:
    """Density histogram with the repeated-sampling reference and one sample's SE."""
    density = histogram_density(values, edges)
    centers = (edges[:-1] + edges[1:]) / 2
    width = float(edges[1] - edges[0])
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=centers,
            y=density,
            width=width * 0.9,
            name="Samples",
            marker=dict(color=GREY, line=dict(width=0)),
            hovertemplate=f"{x_title} %{{x:.3f}}<br>Density %{{y:.2f}}<extra></extra>",
        )
    )
    if reference_normal_x is not None and reference_normal_y is not None:
        fig.add_trace(
            go.Scatter(
                x=reference_normal_x,
                y=reference_normal_y,
                mode="lines",
                name="Normal approximation",
                line=dict(color=INK, width=2.2, dash="dash"),
                hovertemplate="Reference normal<br>%{y:.2f}<extra></extra>",
            )
        )
    if sample_normal_x is not None and sample_normal_y is not None:
        fig.add_trace(
            go.Scatter(
                x=sample_normal_x,
                y=sample_normal_y,
                mode="lines",
                name="Estimated SE",
                line=dict(color=BLUE, width=2.4),
                hovertemplate="One-sample normal<br>%{y:.2f}<extra></extra>",
            )
        )
    x_range = (float(edges[0]), float(edges[-1]))
    if (
        reference_mean is not None
        and reference_sd is not None
        and np.isfinite(reference_mean)
        and np.isfinite(reference_sd)
        and reference_sd > 0
    ):
        for edge in (reference_mean - reference_sd, reference_mean + reference_sd):
            fig.add_vline(x=float(edge), line_color=GREY, line_width=1.2, line_dash="dash")
    if estimate is not None and np.isfinite(estimate):
        fig.add_vline(x=float(estimate), line_color=BLUE, line_width=2)
        if (
            show_estimate_band
            and standard_error is not None
            and np.isfinite(standard_error)
            and standard_error > 0
        ):
            for edge in (float(estimate - standard_error), float(estimate + standard_error)):
                fig.add_vline(x=edge, line_color=BLUE, line_width=1.2, line_dash="dash")
    fig.add_vline(x=float(population), line_color=ORANGE, line_width=2.4)
    ymax = float(density.max()) if density.size else 1.0
    for curve in (reference_normal_y, sample_normal_y):
        if curve is not None and len(curve):
            ymax = max(ymax, float(np.nanmax(curve)))
    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0
    fig.update_xaxes(
        title_text=x_title,
        range=list(x_range),
        tickfont=dict(color=GREY, size=12, family=FONT),
        title_font=dict(color=INK, size=13, family=FONT),
    )
    fig.update_yaxes(
        title_text="Density",
        rangemode="tozero",
        range=[0, ymax * 1.08],
        tickfont=dict(color=GREY, size=12, family=FONT),
        title_font=dict(color=INK, size=13, family=FONT),
    )
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=INK)),
        margin=dict(l=48, r=10, t=30, b=32),
        showlegend=False,
        bargap=0,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def null_slope_scatter(
    *,
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    sample_alpha: float | None,
    sample_beta: float | None,
    population_alpha: float,
    population_beta: float = 0.0,
    height: int = 460,
) -> go.Figure:
    """One sample from the zero-slope population, with the true horizontal line."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sample_x,
            y=sample_y,
            mode="markers",
            name="Current sample",
            marker=dict(size=7, color=BLUE, line=dict(width=0.4, color="white")),
            hovertemplate="Hours %{x:.2f}<br>Score %{y:.1f}<extra></extra>",
        )
    )
    line_x = np.array([0.0, 10.0])
    population_beta = 0.0 if population_beta is None else float(population_beta)
    if (
        sample_alpha is not None
        and sample_beta is not None
        and np.isfinite(sample_alpha)
        and np.isfinite(sample_beta)
    ):
        fig.add_trace(
            go.Scatter(
                x=line_x,
                y=sample_alpha + sample_beta * line_x,
                mode="lines",
                name="Blue = estimated regression",
                line=dict(color=BLUE, width=2.6),
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=population_alpha + population_beta * line_x,
            mode="lines",
            name="Orange = true regression",
            line=dict(color=ORANGE, width=3),
            hoverinfo="skip",
        )
    )
    fig.update_xaxes(title_text="Hours studied", range=[-0.3, 10.3], title_font=dict(size=12))
    fig.update_yaxes(title_text="Exam score", title_font=dict(size=12))
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=dict(text="Sample and regressions", x=0, xanchor="left", font=dict(size=14, color=INK)),
        margin=dict(l=44, r=10, t=30, b=34),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=MUTED, family=FONT),
            bgcolor="rgba(255,255,255,0.85)",
            borderwidth=0,
            itemsizing="constant",
            traceorder="reversed",
        ),
        showlegend=True,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


_NULL_DIST_MARGIN = dict(l=42, r=14, t=48, b=30)
_NULL_DIST_AXIS_TITLE = dict(size=12, color=MUTED, family=FONT)
_NULL_DIST_TICK = dict(color=GREY, size=11, family=FONT)
_NULL_DIST_TITLE = dict(size=13, color=INK, family=FONT)
_NULL_DIST_SUBTITLE = dict(size=11, color=MUTED, family=FONT)


def _observed_label_side(value: float) -> tuple[str, int]:
    if value >= 0:
        return "left", 8
    return "right", -8


def slope_sampling_figure(
    *,
    beta: float | None,
    standard_error: float | None,
    height: int = 236,
) -> go.Figure:
    """Approximate normal sampling distribution of the slope when β = 0."""
    fig = go.Figure()
    usable = standard_error is not None and np.isfinite(standard_error) and standard_error > 0
    if usable:
        scale = float(standard_error)
        boundary = 1.96 * scale
        reach = max(boundary * 1.85, 4.0 * scale)
        if beta is not None and np.isfinite(beta):
            reach = max(reach, abs(float(beta)) * 1.45)
        grid = np.linspace(-reach, reach, 401)
        density = np.exp(-0.5 * (grid / scale) ** 2) / (scale * np.sqrt(2.0 * np.pi))
        peak = float(np.max(density))
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=density,
                mode="lines",
                line=dict(color=INK, width=2.4),
                hovertemplate="Slope %{x:.3f}<br>Density %{y:.3f}<extra></extra>",
            )
        )
        fig.add_vline(x=0.0, line_color=GREY, line_width=1)
        for edge in (-boundary, boundary):
            fig.add_vline(x=edge, line_color=GREY, line_width=1.4, line_dash="dash")
        fig.add_annotation(
            x=boundary,
            y=peak * 0.78,
            text="5% rejection thresholds",
            showarrow=False,
            xanchor="left",
            xshift=6,
            font=dict(size=11, color=MUTED, family=FONT),
        )
        if beta is not None and np.isfinite(beta):
            observed = float(beta)
            fig.add_vline(x=observed, line_color=BLUE, line_width=2.4)
            side, shift = _observed_label_side(observed)
            fig.add_annotation(
                x=observed,
                y=peak * 0.96,
                text=f"Observed \u03b2\u0302 = {observed:.3f}",
                showarrow=False,
                xanchor=side,
                xshift=shift,
                bgcolor="rgba(255,255,255,0.88)",
                borderpad=2,
                font=dict(size=12, color=BLUE, family=FONT),
            )
    fig.update_xaxes(title_text="Estimated slope", title_font=_NULL_DIST_AXIS_TITLE, tickfont=_NULL_DIST_TICK)
    fig.update_yaxes(title_text="Density", title_font=_NULL_DIST_AXIS_TITLE, tickfont=_NULL_DIST_TICK, rangemode="tozero")
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=dict(
            text="Distribution of \u03b2\u0302 if \u03b2 = 0",
            subtitle=dict(text="Sampling distribution of the estimated slope.", font=_NULL_DIST_SUBTITLE),
            x=0,
            xanchor="left",
            font=_NULL_DIST_TITLE,
        ),
        margin=_NULL_DIST_MARGIN,
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def t_test_figure(
    *,
    grid: np.ndarray,
    density: np.ndarray,
    statistic: float | None,
    critical: float | None,
    degrees: int,
    height: int = 236,
) -> go.Figure:
    """Student t under H0, with the 5% rejection tails and the observed statistic."""
    fig = go.Figure()
    peak = float(np.max(density)) if density.size else 1.0
    if critical is not None and np.isfinite(critical):
        edge = abs(float(critical))
        for mask in (grid <= -edge, grid >= edge):
            fig.add_trace(
                go.Scatter(
                    x=grid[mask],
                    y=density[mask],
                    mode="lines",
                    fill="tozeroy",
                    line=dict(width=0),
                    fillcolor="rgba(220, 38, 38, 0.28)",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        fig.add_annotation(
            x=0.0,
            y=peak * 0.42,
            text="95%",
            showarrow=False,
            font=dict(size=14, color=INK, family=FONT),
        )
        left_tail = float(grid[0] + (-edge - grid[0]) * 0.55)
        right_tail = float(edge + (grid[-1] - edge) * 0.45)
        for x_pos in (left_tail, right_tail):
            fig.add_annotation(
                x=x_pos,
                y=1.0,
                yref="paper",
                yanchor="top",
                yshift=-2,
                text="2.5%",
                showarrow=False,
                font=dict(size=12, color="#991B1B", family=FONT),
            )
        fig.add_annotation(
            x=edge,
            y=peak * 0.78,
            text="Critical t values",
            showarrow=False,
            xanchor="left",
            xshift=6,
            font=dict(size=11, color=MUTED, family=FONT),
        )
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=density,
            mode="lines",
            name="t distribution",
            line=dict(color=INK, width=2.4),
            hovertemplate="t %{x:.2f}<br>Density %{y:.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=0.0, line_color=GREY, line_width=1)
    if critical is not None and np.isfinite(critical):
        for edge in (-float(critical), float(critical)):
            fig.add_vline(x=edge, line_color=GREY, line_width=1.4, line_dash="dash")
    if statistic is not None and np.isfinite(statistic):
        observed = float(statistic)
        fig.add_vline(x=observed, line_color=BLUE, line_width=2.4)
        side, shift = _observed_label_side(observed)
        fig.add_annotation(
            x=observed,
            y=peak * 0.96,
            text=f"Observed t = {observed:.3f}",
            showarrow=False,
            xanchor=side,
            xshift=shift,
            bgcolor="rgba(255,255,255,0.88)",
            borderpad=2,
            font=dict(size=12, color=BLUE, family=FONT),
        )
    fig.update_xaxes(title_text="t statistic", title_font=_NULL_DIST_AXIS_TITLE, tickfont=_NULL_DIST_TICK)
    fig.update_yaxes(title_text="Density", title_font=_NULL_DIST_AXIS_TITLE, tickfont=_NULL_DIST_TICK, rangemode="tozero")
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=dict(
            text=f"Distribution of t if \u03b2 = 0 · df = n − 2 = {degrees}",
            subtitle=dict(
                text="Standardized \u03b2\u0302 in SE units. Significance level: 5%.",
                font=_NULL_DIST_SUBTITLE,
            ),
            x=0,
            xanchor="left",
            font=_NULL_DIST_TITLE,
        ),
        margin=_NULL_DIST_MARGIN,
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def _nice_delta(span: float) -> float:
    span = abs(float(span))
    if span <= 0:
        return 1.0
    raw = span * 0.18
    power = 10 ** np.floor(np.log10(raw))
    for factor in (1.0, 2.0, 5.0, 10.0):
        step = factor * power
        if step >= raw * 0.6:
            return float(step)
    return float(10.0 * power)


def _format_quantity(value: float) -> str:
    magnitude = abs(float(value))
    if magnitude >= 100:
        return f"{value:,.1f}"
    if magnitude >= 10:
        return f"{value:.2f}"
    if magnitude >= 1:
        return f"{value:.3g}"
    return f"{value:.3g}"


def slope_triangle_vertices(
    alpha: float,
    beta: float,
    x_start: float,
    delta_x: float,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return A, B, C for the rise/run triangle on the regression line."""
    x_a = float(x_start)
    y_a = float(alpha) + float(beta) * x_a
    x_b = x_a + float(delta_x)
    y_b = y_a
    x_c = x_b
    y_c = float(alpha) + float(beta) * x_c
    return (x_a, y_a), (x_b, y_b), (x_c, y_c)


def _local_point_density(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_centre: float,
    y_centre: float,
    x_radius: float,
    y_radius: float,
) -> int:
    return int(
        np.sum(
            (np.abs(x_values - x_centre) <= x_radius)
            & (np.abs(y_values - y_centre) <= y_radius)
        )
    )


def _choose_slope_triangle(
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    alpha: float,
    beta: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[float, float, float]:
    """Pick (x0, ΔX, ΔY) so the triangle stays visible and relatively clear."""
    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)
    base_dx = _nice_delta(x_span)
    candidates: list[tuple[float, float, float, float]] = []
    # Try a few starts along the visible line; prefer quieter regions.
    for fraction in (0.18, 0.28, 0.38, 0.48, 0.58, 0.68):
        for scale in (1.0, 0.7, 1.3, 0.5):
            dx = base_dx * scale
            if dx <= 0 or dx > 0.45 * x_span:
                continue
            x_start = x_min + fraction * x_span
            if x_start + dx > x_max - 0.02 * x_span:
                x_start = x_max - dx - 0.02 * x_span
            if x_start < x_min + 0.02 * x_span:
                continue
            (xa, ya), (xb, yb), (xc, yc) = slope_triangle_vertices(alpha, beta, x_start, dx)
            ys = (ya, yb, yc)
            if min(ys) < y_min + 0.02 * y_span or max(ys) > y_max - 0.02 * y_span:
                continue
            density = _local_point_density(
                x_values,
                y_values,
                0.5 * (xa + xc),
                0.5 * (ya + yc),
                0.55 * dx,
                0.55 * max(abs(yc - ya), 0.08 * y_span),
            )
            # Prefer lower density, then mid-plot placement.
            score = density + 0.15 * abs(fraction - 0.35) * len(x_values)
            candidates.append((score, x_start, dx, float(beta) * dx))
    if not candidates:
        dx = min(base_dx, 0.35 * x_span)
        x_start = x_min + 0.25 * x_span
        if x_start + dx > x_max:
            x_start = max(x_min, x_max - dx)
        return x_start, dx, float(beta) * dx
    candidates.sort(key=lambda item: item[0])
    _, x_start, dx, dy = candidates[0]
    return x_start, dx, dy


def _add_coefficient_geometry(
    fig: go.Figure,
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    alpha: float,
    beta: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_title: str,
    y_title: str,
) -> list[dict]:
    """Draw intercept marker/inset and a geometric slope triangle."""
    annotations: list[dict] = []
    shapes: list[dict] = []
    alpha = float(alpha)
    beta = float(beta)

    # --- Intercept α̂ ---
    zero_in_x = x_min <= 0.0 <= x_max
    alpha_in_y = y_min <= alpha <= y_max
    if zero_in_x and alpha_in_y:
        guide_x1 = x_min + 0.04 * (x_max - x_min)
        fig.add_trace(
            go.Scatter(
                x=[0.0, guide_x1],
                y=[alpha, alpha],
                mode="lines",
                line=dict(color="#0F766E", width=1.6, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0.0],
                y=[alpha],
                mode="markers",
                name="Intercept α̂",
                marker=dict(
                    size=14,
                    color="#0F766E",
                    symbol="diamond",
                    line=dict(width=1.5, color="white"),
                ),
                hovertemplate=(
                    f"Intercept α̂ = {_format_quantity(alpha)}<br>"
                    f"Predicted {y_title} when predictor = 0"
                    "<extra></extra>"
                ),
            )
        )
        annotations.append(
            dict(
                x=0.0,
                y=alpha,
                text=f"Intercept α̂ = {_format_quantity(alpha)}",
                showarrow=True,
                arrowhead=2,
                ax=28,
                ay=-24,
                font=dict(size=11, color="#0F766E", family=FONT),
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="#0F766E",
                borderwidth=1,
                borderpad=3,
            )
        )
    else:
        # Schematic inset — not in scatterplot coordinates.
        reason = []
        if not zero_in_x:
            reason.append("X = 0 is outside the observed X range")
        if zero_in_x and not alpha_in_y:
            reason.append("α̂ is outside the visible Y range")
        reason_text = "; ".join(reason) if reason else "Intercept lies outside this window"
        annotations.append(
            dict(
                xref="paper",
                yref="paper",
                x=0.01,
                y=0.99,
                xanchor="left",
                yanchor="top",
                text=(
                    "<b>Intercept inset</b> "
                    "<span style='color:#7B8794'>(schematic)</span><br>"
                    f"α̂ = {_format_quantity(alpha)} at X = 0<br>"
                    f"Predicted {y_title} when predictor = 0<br>"
                    f"<span style='color:#9B6B2F'>{reason_text}</span>"
                ),
                showarrow=False,
                align="left",
                font=dict(size=10, color=INK, family=FONT),
                bgcolor="rgba(236, 253, 245, 0.96)",
                bordercolor="#0F766E",
                borderwidth=1,
                borderpad=5,
            )
        )
        # Tiny schematic axes inside the inset box (paper coords).
        shapes.extend(
            [
                dict(
                    type="line",
                    xref="paper",
                    yref="paper",
                    x0=0.02,
                    x1=0.11,
                    y0=0.78,
                    y1=0.78,
                    line=dict(color="#94A3B8", width=1),
                ),
                dict(
                    type="line",
                    xref="paper",
                    yref="paper",
                    x0=0.02,
                    x1=0.02,
                    y0=0.78,
                    y1=0.92,
                    line=dict(color="#94A3B8", width=1),
                ),
                dict(
                    type="line",
                    xref="paper",
                    yref="paper",
                    x0=0.02,
                    x1=0.10,
                    y0=0.84,
                    y1=0.90,
                    line=dict(color=BLUE, width=1.5),
                ),
            ]
        )
        annotations.append(
            dict(
                xref="paper",
                yref="paper",
                x=0.02,
                y=0.84,
                xanchor="left",
                yanchor="middle",
                text="α̂",
                showarrow=False,
                font=dict(size=10, color="#0F766E", family=FONT),
            )
        )

    # --- Slope triangle β̂ ---
    x_start, delta_x, delta_y = _choose_slope_triangle(
        x_values=x_values,
        y_values=y_values,
        alpha=alpha,
        beta=beta,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )
    (xa, ya), (xb, yb), (xc, yc) = slope_triangle_vertices(alpha, beta, x_start, delta_x)
    # Closed right-angle path A → B → C → A (A→C lies on the regression line).
    fig.add_trace(
        go.Scatter(
            x=[xa, xb, xc, xa],
            y=[ya, yb, yc, ya],
            mode="lines",
            name="Slope triangle",
            line=dict(color="#C2410C", width=2.2),
            fill="toself",
            fillcolor="rgba(194, 65, 12, 0.10)",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[xa, xc],
            y=[ya, yc],
            mode="markers",
            name="Slope ends",
            marker=dict(size=8, color="#C2410C", line=dict(width=1, color="white")),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    annotations.append(
        dict(
            x=0.5 * (xa + xb),
            y=ya,
            text=f"ΔX = {_format_quantity(delta_x)}",
            showarrow=False,
            yanchor="top" if delta_y >= 0 else "bottom",
            yshift=-8 if delta_y >= 0 else 8,
            font=dict(size=10, color="#9A3412", family=FONT),
            bgcolor="rgba(255,255,255,0.88)",
            borderpad=2,
        )
    )
    annotations.append(
        dict(
            x=xb,
            y=0.5 * (yb + yc),
            text=f"β̂ = {_format_quantity(beta)}",
            showarrow=False,
            xanchor="left",
            xshift=6,
            font=dict(size=10, color="#9A3412", family=FONT),
            bgcolor="rgba(255,255,255,0.88)",
            borderpad=2,
        )
    )
    if shapes:
        existing = list(fig.layout.shapes) if fig.layout.shapes else []
        fig.update_layout(shapes=existing + shapes)
    return annotations


def _unwrap_customdata_index(custom: object) -> int | None:
    """Pull a sample index out of Plotly/Streamlit customdata shapes."""
    value: object = custom
    # Streamlit sometimes nests column customdata as [[i]] or [i].
    for _ in range(3):
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                return None
            value = value[0]
            continue
        break
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def observation_index_from_point(point: object) -> int | None:
    """Sample index from an Observations marker (customdata), or None."""
    if not isinstance(point, dict):
        return None
    return _unwrap_customdata_index(point.get("customdata"))


def _selection_points(selection: object) -> list:
    if selection is None:
        return []
    points = None
    if hasattr(selection, "get"):
        nested = selection.get("selection")
        if nested is not None and hasattr(nested, "get"):
            points = nested.get("points")
        else:
            points = selection.get("points")
    else:
        nested = getattr(selection, "selection", None)
        if nested is not None:
            points = (
                nested.get("points")
                if hasattr(nested, "get")
                else getattr(nested, "points", None)
            )
        else:
            points = getattr(selection, "points", None)
    if not points:
        return []
    return list(points)


def nearest_observation_index(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x: float,
    y: float,
) -> int | None:
    """Index of the sample point closest to (x, y), or None if empty."""
    xs = np.asarray(x_values, dtype=np.float64)
    ys = np.asarray(y_values, dtype=np.float64)
    if xs.size == 0 or ys.size != xs.size:
        return None
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    x_span = max(float(np.ptp(xs)), 1e-9)
    y_span = max(float(np.ptp(ys)), 1e-9)
    dist2 = ((xs - float(x)) / x_span) ** 2 + ((ys - float(y)) / y_span) ** 2
    return int(np.argmin(dist2))


def observation_index_from_selection(
    selection: object,
    *,
    x_values: np.ndarray | None = None,
    y_values: np.ndarray | None = None,
    observations_curve: int | None = None,
) -> int | None:
    """Resolve a clicked observation from a Plotly / Streamlit selection."""
    points = _selection_points(selection)
    if not points:
        return None
    for point in points:
        index = observation_index_from_point(point)
        if index is not None:
            return index
    # Fall back: point_index on the Observations curve (curve 0 when first).
    if observations_curve is not None:
        for point in points:
            if not isinstance(point, dict):
                continue
            curve = point.get("curve_number", point.get("curveNumber"))
            if curve is None or int(curve) != int(observations_curve):
                continue
            for key in ("point_index", "pointIndex", "point_number", "pointNumber"):
                if point.get(key) is not None:
                    try:
                        return int(point[key])
                    except (TypeError, ValueError):
                        pass
    # Last resort: map click coordinates to the nearest sample point
    # (handles residual-line hits that share the observation's X).
    if x_values is not None and y_values is not None:
        for point in points:
            if not isinstance(point, dict):
                continue
            if point.get("x") is None or point.get("y") is None:
                continue
            try:
                return nearest_observation_index(
                    x_values, y_values, float(point["x"]), float(point["y"])
                )
            except (TypeError, ValueError):
                continue
    return None


def fitting_scatter(
    *,
    hours: np.ndarray,
    scores: np.ndarray,
    alpha: float,
    beta: float,
    show_residuals: bool = True,
    selected_index: int | None = None,
    height: int = 340,
    x_title: str = "Hours studied",
    y_title: str = "Exam score",
    title: str = "Regression line & residuals",
    show_coefficients: bool = False,
    prediction_x: float | None = None,
    prediction_alpha: float | None = None,
    prediction_beta: float | None = None,
    show_line: bool = True,
) -> go.Figure:
    """Candidate regression line fitted to a fixed set of observations."""
    fig = go.Figure()
    x_values = np.asarray(hours, dtype=np.float64)
    y_values = np.asarray(scores, dtype=np.float64)
    n_points = int(x_values.size)
    selected = (
        int(selected_index)
        if selected_index is not None and 0 <= int(selected_index) < n_points
        else None
    )
    # Observations first so curve_number 0 is the selectable sample cloud.
    use_webgl = n_points > 800
    scatter_cls = go.Scattergl if use_webgl else go.Scatter
    if n_points > 2000:
        marker = dict(size=4, color="rgba(37, 99, 235, 0.28)", line=dict(width=0))
    elif n_points > 400:
        marker = dict(size=6, color="rgba(37, 99, 235, 0.45)", line=dict(width=0))
    else:
        marker = dict(size=9, color=BLUE_POINT, line=dict(width=0.5, color="white"))
    # [sample_index, student_number] — index drives selection; number is for hover.
    custom = np.column_stack(
        [
            np.arange(n_points, dtype=int),
            np.arange(1, n_points + 1, dtype=int),
        ]
    )
    fig.add_trace(
        scatter_cls(
            x=x_values,
            y=y_values,
            mode="markers",
            name="Observations",
            customdata=custom,
            marker=marker,
            # Keep full opacity when Plotly marks other points "unselected".
            selected=dict(marker=dict(opacity=1.0)),
            unselected=dict(marker=dict(opacity=1.0)),
            hovertemplate=(
                "Observation %{customdata[1]}<br>"
                f"{x_title} %{{x:.2f}}<br>"
                f"{y_title} %{{y:.2f}}"
                "<extra></extra>"
            ),
        )
    )
    # Residual guides after markers so they don't steal the primary hit target.
    # Skip the selected observation — EXPLAIN draws its own blue/orange stems.
    if show_residuals:
        if n_points <= 1500:
            predicted = alpha + beta * x_values
            residual_x: list[float | None] = []
            residual_y: list[float | None] = []
            for index, (x_value, y_value, y_hat) in enumerate(
                zip(x_values, y_values, predicted)
            ):
                if selected is not None and index == selected:
                    continue
                residual_x.extend([float(x_value), float(x_value), None])
                residual_y.extend([float(y_value), float(y_hat), None])
            if residual_x:
                fig.add_trace(
                    go.Scatter(
                        x=residual_x,
                        y=residual_y,
                        mode="lines",
                        name="Residuals",
                        line=dict(color="rgba(107, 114, 128, 0.85)", width=1.6),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
        # Large-n: skip per-point grey stems; EXPLAIN colours cover the selection.
    x_pad = max(0.05 * float(np.ptp(x_values)), 0.15) if x_values.size else 0.5
    y_pad = max(0.08 * float(np.ptp(y_values)), 0.5) if y_values.size else 1.0
    x_obs_min = float(x_values.min()) if n_points else 0.0
    x_obs_max = float(x_values.max()) if n_points else 1.0
    x0 = x_obs_min - x_pad
    x1 = x_obs_max + x_pad
    y0 = float(y_values.min()) - y_pad
    y1 = float(y_values.max()) + y_pad
    y_bar = float(np.mean(y_values)) if n_points else float("nan")
    explain_x = explain_y = explain_yhat = None
    if selected is not None and np.isfinite(y_bar):
        explain_x = float(x_values[selected])
        explain_y = float(y_values[selected])
        explain_yhat = float(alpha + beta * explain_x)
        y0 = min(y0, y_bar - y_pad, explain_yhat - y_pad, explain_y - y_pad)
        y1 = max(y1, y_bar + y_pad, explain_yhat + y_pad, explain_y + y_pad)
    pred_x = pred_y = None
    extrapolating = False
    if (
        prediction_x is not None
        and prediction_alpha is not None
        and prediction_beta is not None
        and np.isfinite(prediction_x)
    ):
        pred_x = float(prediction_x)
        pred_y = float(prediction_alpha) + float(prediction_beta) * pred_x
        extrapolating = bool(n_points) and (
            pred_x < x_obs_min - 1e-12 or pred_x > x_obs_max + 1e-12
        )
        x0 = min(x0, pred_x - x_pad)
        x1 = max(x1, pred_x + x_pad)
        y0 = min(y0, pred_y - y_pad)
        y1 = max(y1, pred_y + y_pad)
    # Solid line across the full displayed X-axis (includes padding past extremes).
    line_x = np.array([x0, x1])
    if show_line:
        fig.add_trace(
            go.Scatter(
                x=line_x,
                y=alpha + beta * line_x,
                mode="lines",
                name="Regression line",
                line=dict(color=BLUE, width=3.0),
                hoverinfo="skip",
            )
        )
    if show_line and extrapolating and pred_x is not None:
        if pred_x > x_obs_max and x1 > x_obs_max:
            fig.add_trace(
                go.Scatter(
                    x=np.array([x_obs_max, x1]),
                    y=alpha + beta * np.array([x_obs_max, x1]),
                    mode="lines",
                    name="Extrapolation",
                    line=dict(color=BLUE, width=2.4, dash="dash"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        if pred_x < x_obs_min and x0 < x_obs_min:
            fig.add_trace(
                go.Scatter(
                    x=np.array([x0, x_obs_min]),
                    y=alpha + beta * np.array([x0, x_obs_min]),
                    mode="lines",
                    name="Extrapolation",
                    line=dict(color=BLUE, width=2.4, dash="dash"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    explain_annotations: list[dict] = []
    if explain_x is not None and explain_y is not None and explain_yhat is not None:
        # EXPLAIN palette matches the panel: mean grey, model BLUE, residual ORANGE.
        mean_colour = "#64748B"
        # Horizontal dashed sample mean — only while a student is selected
        # (never drawn for PREDICT alone).
        fig.add_trace(
            go.Scatter(
                x=line_x,
                y=np.full(2, y_bar),
                mode="lines",
                name="Sample mean Ȳ",
                line=dict(color=mean_colour, width=1.8, dash="dash"),
                hovertemplate=f"Sample mean Ȳ = {y_bar:.2f}<extra></extra>",
                showlegend=False,
            )
        )
        explain_annotations.append(
            dict(
                x=float(x1),
                y=float(y_bar),
                text=f"Sample mean Ȳ = {y_bar:.2f}",
                showarrow=False,
                xanchor="right",
                yanchor="bottom",
                font=dict(size=11, color=mean_colour, family=FONT),
                bgcolor="rgba(255,255,255,0.88)",
                borderpad=2,
            )
        )
        # Side-by-side verticals around the selected X. When Y and Ŷ fall on
        # opposite sides of Ȳ, the residual Y-range contains the model segment;
        # a shared X would hide blue under orange — keep a clear gap.
        x_span = max(float(x1 - x0), float(np.ptp(x_values)) if n_points else 0.0, 1e-9)
        dx = max(0.05 * x_span, 0.12)
        x_model = float(explain_x - dx)
        x_resid = float(explain_x + dx)
        # Orange first (Ŷ → Y), then blue (Ȳ → Ŷ) so model stays readable.
        fig.add_trace(
            go.Scatter(
                x=[x_resid, x_resid],
                y=[explain_yhat, explain_y],
                mode="lines",
                name="Residual",
                line=dict(color=ORANGE, width=5.0),
                hovertemplate=(
                    f"Residual Y − Ŷ = {explain_y - explain_yhat:+.2f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[x_model, x_model],
                y=[y_bar, explain_yhat],
                mode="lines",
                name="Model component",
                line=dict(color=BLUE, width=5.0),
                hovertemplate=(
                    f"Model component Ŷ − Ȳ = {explain_yhat - y_bar:+.2f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )
        # Join at Ŷ plus endpoint ticks so mean / Ŷ / Y stay identifiable.
        fig.add_trace(
            go.Scatter(
                x=[x_model, x_resid],
                y=[explain_yhat, explain_yhat],
                mode="lines",
                name="Ŷ join",
                line=dict(color="#94A3B8", width=2.5),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[explain_x],
                y=[explain_yhat],
                mode="markers+text",
                name="Predicted Ŷ",
                text=["Ŷ"],
                textposition="middle left",
                textfont=dict(size=12, color=ORANGE, family=FONT),
                marker=dict(
                    size=12,
                    color=ORANGE,
                    symbol="diamond",
                    line=dict(width=1.5, color="white"),
                ),
                hovertemplate=f"Predicted Ŷ = {explain_yhat:.2f}<extra></extra>",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[explain_x],
                y=[explain_y],
                mode="markers+text",
                name="Actual Y",
                text=["Y"],
                textposition="middle right",
                textfont=dict(size=12, color=BLUE, family=FONT),
                marker=dict(
                    size=14,
                    color=BLUE,
                    symbol="circle",
                    line=dict(width=2.0, color="white"),
                ),
                hovertemplate=(
                    f"Actual Y = {explain_y:.2f}<br>"
                    f"Predicted Ŷ = {explain_yhat:.2f}<br>"
                    f"Sample mean Ȳ = {y_bar:.2f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )
    shapes: list[dict] = []
    if extrapolating and n_points:
        if x0 < x_obs_min:
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=x0,
                    x1=x_obs_min,
                    y0=0,
                    y1=1,
                    fillcolor="rgba(234, 88, 12, 0.08)",
                    line=dict(width=0),
                    layer="below",
                )
            )
        if x1 > x_obs_max:
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=x_obs_max,
                    x1=x1,
                    y0=0,
                    y1=1,
                    fillcolor="rgba(234, 88, 12, 0.08)",
                    line=dict(width=0),
                    layer="below",
                )
            )
    if pred_x is not None and pred_y is not None:
        guide_colour = ORANGE
        fig.add_trace(
            go.Scatter(
                x=[pred_x, pred_x, None, x0, pred_x],
                y=[y0, pred_y, None, pred_y, pred_y],
                mode="lines",
                name="Prediction guides",
                line=dict(color=guide_colour, width=1.8, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[pred_x],
                y=[pred_y],
                mode="markers",
                name="Prediction",
                marker=dict(
                    size=11,
                    color=ORANGE,
                    symbol="diamond" if extrapolating else "circle",
                    line=dict(width=1.5, color="white"),
                ),
                hovertemplate=(
                    f"{x_title} %{{x:.2f}}<br>"
                    f"Predicted {y_title} %{{y:.2f}}"
                    + ("<br>Extrapolation" if extrapolating else "")
                    + "<extra></extra>"
                ),
                showlegend=False,
            )
        )
    annotations: list[dict] = list(explain_annotations)
    if show_coefficients and x_values.size:
        annotations.extend(
            _add_coefficient_geometry(
                fig,
                x_values=x_values,
                y_values=y_values,
                alpha=alpha,
                beta=beta,
                x_min=x0,
                x_max=x1,
                y_min=y0,
                y_max=y1,
                x_title=x_title,
                y_title=y_title,
            )
        )
    fig.update_xaxes(title_text=x_title, range=[x0, x1], title_font=dict(size=11))
    fig.update_yaxes(title_text=y_title, range=[y0, y1], title_font=dict(size=11))
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=dict(text=title, x=0, xanchor="left", font=dict(size=13, color=INK)),
        margin=dict(l=44, r=8, t=28, b=42),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.16,
            x=0,
            font=dict(size=10, color=MUTED, family=FONT),
            bgcolor="rgba(255,255,255,0.9)",
            borderwidth=0,
            itemsizing="constant",
        ),
        showlegend=True,
        paper_bgcolor="white",
        plot_bgcolor="white",
        clickmode="event+select",
        annotations=annotations,
        shapes=shapes,
        # Keep Plotly identity across toggle/slider updates (no remount flash).
        uirevision="fit-scatter",
    )
    return fig


def r2_lesson_scatter(
    *,
    hours: np.ndarray,
    scores: np.ndarray,
    height: int = 460,
    x_title: str = "Hours studied",
    y_title: str = "Exam score",
    title: str = "Understanding R²",
    show_mean_line: bool = True,
    regression_alpha: float | None = None,
    regression_beta: float | None = None,
    regression_name: str = "Regression",
    ols_alpha: float | None = None,
    ols_beta: float | None = None,
    residual_about: str | None = None,
    mean_trace_name: str | None = None,
    regression_residual_color: str = "rgba(234, 88, 12, 0.75)",
) -> go.Figure:
    """One large scatter for the R² lesson: mean line, candidate, optional OLS, residuals."""
    fig = go.Figure()
    x_values = np.asarray(hours, dtype=np.float64)
    y_values = np.asarray(scores, dtype=np.float64)
    n_points = int(x_values.size)
    y_bar = float(np.mean(y_values)) if n_points else 0.0

    def _residual_segments(predicted: np.ndarray, name: str, colour: str) -> None:
        if n_points <= 0:
            return
        if n_points <= 1500:
            residual_x: list[float | None] = []
            residual_y: list[float | None] = []
            for x_value, y_value, y_hat in zip(x_values, y_values, predicted):
                residual_x.extend([float(x_value), float(x_value), None])
                residual_y.extend([float(y_value), float(y_hat), None])
            fig.add_trace(
                go.Scatter(
                    x=residual_x,
                    y=residual_y,
                    mode="lines",
                    name=name,
                    line=dict(color=colour, width=1.5),
                    hoverinfo="skip",
                )
            )

    if residual_about == "mean" and n_points:
        _residual_segments(
            np.full(n_points, y_bar, dtype=np.float64),
            "Residuals from mean",
            "rgba(107, 114, 128, 0.85)",
        )
    elif (
        residual_about == "regression"
        and regression_alpha is not None
        and regression_beta is not None
        and n_points
    ):
        predicted = float(regression_alpha) + float(regression_beta) * x_values
        _residual_segments(predicted, "Residuals from regression", regression_residual_color)

    use_webgl = n_points > 800
    scatter_cls = go.Scattergl if use_webgl else go.Scatter
    if n_points > 2000:
        marker = dict(size=4, color="rgba(37, 99, 235, 0.28)", line=dict(width=0))
    elif n_points > 400:
        marker = dict(size=6, color="rgba(37, 99, 235, 0.45)", line=dict(width=0))
    else:
        marker = dict(size=9, color=BLUE_POINT, line=dict(width=0.5, color="white"))
    fig.add_trace(
        scatter_cls(
            x=x_values,
            y=y_values,
            mode="markers",
            name="Observations",
            marker=marker,
            hovertemplate=f"{x_title} %{{x:.2f}}<br>{y_title} %{{y:.2f}}<extra></extra>",
        )
    )

    x_pad = max(0.05 * float(np.ptp(x_values)), 0.15) if n_points else 0.5
    y_pad = max(0.08 * float(np.ptp(y_values)), 0.5) if n_points else 1.0
    if n_points:
        x0 = float(x_values.min()) - x_pad
        x1 = float(x_values.max()) + x_pad
        y0 = float(y_values.min()) - y_pad
        y1 = float(y_values.max()) + y_pad
    else:
        x0, x1, y0, y1 = 0.0, 1.0, 0.0, 1.0
    y0 = min(y0, y_bar - y_pad)
    y1 = max(y1, y_bar + y_pad)
    line_x = np.array([x0, x1])

    if show_mean_line:
        mean_name = mean_trace_name if mean_trace_name is not None else f"Mean ȳ = {y_bar:.2f}"
        fig.add_trace(
            go.Scatter(
                x=line_x,
                y=np.full(2, y_bar),
                mode="lines",
                name=mean_name,
                line=dict(color=GREY, width=2.6, dash="dash"),
                hovertemplate=f"ȳ = {y_bar:.3f}<extra></extra>",
            )
        )

    if regression_alpha is not None and regression_beta is not None:
        fig.add_trace(
            go.Scatter(
                x=line_x,
                y=float(regression_alpha) + float(regression_beta) * line_x,
                mode="lines",
                name=regression_name,
                line=dict(color=BLUE, width=3.0),
                hoverinfo="skip",
            )
        )

    if ols_alpha is not None and ols_beta is not None:
        # Avoid a duplicate trace when the candidate already is OLS.
        same_as_reg = (
            regression_alpha is not None
            and regression_beta is not None
            and abs(float(ols_alpha) - float(regression_alpha)) < 1e-9
            and abs(float(ols_beta) - float(regression_beta)) < 1e-9
        )
        if not same_as_reg:
            fig.add_trace(
                go.Scatter(
                    x=line_x,
                    y=float(ols_alpha) + float(ols_beta) * line_x,
                    mode="lines",
                    name="OLS regression",
                    line=dict(color=ORANGE, width=2.6, dash="dot"),
                    hovertemplate=f"{tip_hover('OLS')}<extra></extra>",
                )
            )

    fig.update_xaxes(title_text=x_title, range=[x0, x1], title_font=dict(size=11))
    fig.update_yaxes(title_text=y_title, range=[y0, y1], title_font=dict(size=11))
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=dict(text=title, x=0, xanchor="left", font=dict(size=13, color=INK)),
        margin=dict(l=44, r=8, t=28, b=48),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            x=0,
            font=dict(size=10, color=MUTED, family=FONT),
            bgcolor="rgba(255,255,255,0.9)",
            borderwidth=0,
            itemsizing="constant",
        ),
        showlegend=True,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def sse_slope_figure(
    *,
    slopes: np.ndarray,
    sse_values: np.ndarray,
    ols_slope: float,
    ols_sse: float,
    current_slope: float,
    current_sse: float,
    parabola_sse: float,
    height: int = 340,
    title: str = "Finding the minimum SSR",
) -> go.Figure:
    """SSR against slope, using the best intercept at each slope."""
    fig = go.Figure()
    curve = np.asarray(sse_values, dtype=np.float64)
    # Never plot a candidate below the profile curve (numerical noise only).
    display_current = max(float(current_sse), float(parabola_sse))
    curve_hi = float(np.nanmax(curve)) if curve.size else float(ols_sse)
    y_top = max(curve_hi * 1.2, float(ols_sse) * 2.0, float(ols_sse) + 1.0)
    above_range = display_current > y_top * 1.001

    fig.add_trace(
        go.Scatter(
            x=slopes,
            y=curve,
            mode="lines",
            name="SSR curve (best α̂)",
            line=dict(color=INK, width=2.2),
            hovertemplate=(
                "Slope %{x:.2f}<br>"
                f"{tip_hover('SSR')}"
                ": %{y:.1f}<br>"
                "Best achievable SSR at this slope (intercept optimised)."
                "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[ols_slope],
            y=[ols_sse],
            mode="markers",
            name="OLS minimum",
            marker=dict(
                size=12,
                color=ORANGE,
                symbol="diamond",
                line=dict(width=1.5, color="white"),
            ),
            hovertemplate=(
                f"{tip_hover('OLS')}<br>"
                "Slope %{x:.3f}<br>"
                f"{tip_hover('SSR')}"
                ": %{y:.1f}<extra></extra>"
            ),
        )
    )
    annotations = []
    if above_range:
        fig.add_trace(
            go.Scatter(
                x=[current_slope],
                y=[y_top],
                mode="markers",
                name="Your line (off scale)",
                marker=dict(
                    size=12,
                    color=BLUE,
                    symbol="triangle-up",
                    line=dict(width=2, color="white"),
                ),
                hovertemplate=(
                    f"{tip_hover('SSR')}: {display_current:,.1f}<br>"
                    "Above chart range<extra></extra>"
                ),
            )
        )
        annotations.append(
            dict(
                x=current_slope,
                y=y_top,
                text=f"Your SSR is above chart range<br>({display_current:,.0f})",
                showarrow=True,
                arrowhead=2,
                ax=40,
                ay=-28,
                font=dict(size=11, color=BLUE, family=FONT),
                bgcolor="rgba(255,255,255,0.92)",
                borderpad=3,
            )
        )
    else:
        if abs(display_current - parabola_sse) > 1e-6:
            fig.add_trace(
                go.Scatter(
                    x=[current_slope, current_slope],
                    y=[parabola_sse, display_current],
                    mode="lines",
                    line=dict(color=BLUE, width=1.4, dash="dot"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[current_slope],
                y=[display_current],
                mode="markers",
                name="Your line",
                marker=dict(
                    size=11,
                    color=BLUE,
                    symbol="circle",
                    line=dict(width=2, color="white"),
                ),
                hovertemplate=(
                    "Current candidate<br>Slope %{x:.3g}<br>"
                    f"{tip_hover('SSR')}"
                    ": %{y:,.1f}<br>"
                    "Your intercept and slope (may sit above the curve)."
                    "<extra></extra>"
                ),
            )
        )
    fig.update_xaxes(title_text="Candidate slope β̂", title_font=dict(size=11))
    fig.update_yaxes(
        title_text="SSR",
        title_font=dict(size=11),
        range=[0.0, y_top],
        rangemode="tozero",
    )
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=dict(
            text=title,
            x=0,
            xanchor="left",
            font=dict(size=13, color=INK),
        ),
        margin=dict(l=44, r=8, t=28, b=42),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.16,
            x=0,
            xanchor="left",
            font=dict(size=10, color=MUTED, family=FONT),
            bgcolor="rgba(255,255,255,0.92)",
            borderwidth=0,
            itemsizing="constant",
            traceorder="normal",
        ),
        showlegend=True,
        paper_bgcolor="white",
        plot_bgcolor="white",
        annotations=annotations,
    )
    return fig


def _surface_tick_labels(
    center: float,
    scale: float,
    u_min: float,
    u_max: float,
    count: int = 5,
) -> tuple[list[float], list[str]]:
    """Tick positions in normalised u/v with labels showing real α/β."""
    lo = float(u_min)
    hi = float(u_max)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = -1.0, 1.0
    ticks = np.linspace(lo, hi, int(count))
    labels = []
    for tick in ticks:
        value = float(center) + float(tick) * float(scale)
        if abs(value) >= 1000:
            labels.append(f"{value:,.0f}")
        elif abs(value) >= 10:
            labels.append(f"{value:.1f}")
        else:
            labels.append(f"{value:.2f}")
    return [float(t) for t in ticks], labels


def _floor_contour_polylines(
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    z_grid: np.ndarray,
    *,
    levels: int = 6,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Build floor isoline segments from grid edge crossings."""
    z = np.asarray(z_grid, dtype=np.float64)
    u = np.asarray(u_grid, dtype=np.float64)
    v = np.asarray(v_grid, dtype=np.float64)
    z_max = float(np.nanmax(z))
    if z_max <= 0:
        return [], [], []
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []

    def _emit_segment(x0: float, y0: float, x1: float, y1: float) -> None:
        xs.extend([x0, x1, None])
        ys.extend([y0, y1, None])
        zs.extend([0.0, 0.0, None])

    for level in np.linspace(0.1 * z_max, 0.9 * z_max, int(levels)):
        # Marching-squares style: connect crossings within each cell.
        for i in range(z.shape[0] - 1):
            for j in range(z.shape[1] - 1):
                corners = (
                    (float(u[i, j]), float(v[i, j]), float(z[i, j])),
                    (float(u[i, j + 1]), float(v[i, j + 1]), float(z[i, j + 1])),
                    (float(u[i + 1, j + 1]), float(v[i + 1, j + 1]), float(z[i + 1, j + 1])),
                    (float(u[i + 1, j]), float(v[i + 1, j]), float(z[i + 1, j])),
                )
                pts: list[tuple[float, float]] = []
                for k in range(4):
                    x0, y0, z0 = corners[k]
                    x1, y1, z1 = corners[(k + 1) % 4]
                    if (z0 - level) * (z1 - level) <= 0 and z0 != z1:
                        t = (level - z0) / (z1 - z0)
                        pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
                if len(pts) >= 2:
                    _emit_segment(pts[0][0], pts[0][1], pts[1][0], pts[1][1])
                    if len(pts) == 4:
                        _emit_segment(pts[2][0], pts[2][1], pts[3][0], pts[3][1])
    return xs, ys, zs


def sse_surface_figure(
    *,
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    excess_grid: np.ndarray,
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
    sse_grid: np.ndarray,
    alpha_ols: float,
    beta_ols: float,
    sse_min: float,
    alpha_scale: float,
    beta_scale: float,
    current_alpha: float,
    current_beta: float,
    current_sse: float,
    path_alphas: np.ndarray | list[float] | None = None,
    path_betas: np.ndarray | list[float] | None = None,
    path_excess: np.ndarray | list[float] | None = None,
    height: int = 440,
    title: str = "OLS: Finding the minimum SSR",
    uirevision: str | None = None,
) -> go.Figure:
    """Polished 3D excess-SSR bowl with floor contour projection."""
    fig = go.Figure()
    u = np.asarray(u_grid, dtype=np.float64)
    v = np.asarray(v_grid, dtype=np.float64)
    z = np.asarray(excess_grid, dtype=np.float64)
    alphas = np.asarray(alpha_grid, dtype=np.float64)
    betas = np.asarray(beta_grid, dtype=np.float64)
    sses = np.asarray(sse_grid, dtype=np.float64)
    z_max = float(np.nanmax(z))
    z_span = max(z_max, 1e-9)
    z_top = z_max * 1.05
    hover = np.stack([alphas, betas, sses, z], axis=-1)
    colorscale = [
        [0.0, "#F8FAFC"],
        [0.15, "#E2E8F0"],
        [0.35, "#94A3B8"],
        [0.55, "#F0B27A"],
        [0.78, "#E07A3D"],
        [1.0, "#9A3412"],
    ]
    # Floor heatmap at z = 0 (same excess field) + explicit contour lines.
    fig.add_trace(
        go.Surface(
            x=u,
            y=v,
            z=np.full_like(z, 0.0),
            surfacecolor=z,
            cmin=0.0,
            cmax=z_max,
            colorscale=colorscale,
            opacity=1.0,
            showscale=False,
            hoverinfo="skip",
            lighting=dict(ambient=1.0, diffuse=0.15, specular=0.0, roughness=1.0),
            name="Floor heatmap",
        )
    )
    cx, cy, cz = _floor_contour_polylines(u, v, z, levels=6)
    if cx:
        fig.add_trace(
            go.Scatter3d(
                x=cx,
                y=cy,
                z=cz,
                mode="lines",
                line=dict(color="rgba(51, 65, 85, 0.75)", width=3),
                hoverinfo="skip",
                showlegend=False,
                name="Floor contours",
            )
        )
    fig.add_trace(
        go.Surface(
            x=u,
            y=v,
            z=z,
            customdata=hover,
            colorscale=colorscale,
            opacity=0.52,
            showscale=False,
            contours=dict(
                x=dict(show=False),
                y=dict(show=False),
                z=dict(
                    show=True,
                    usecolormap=True,
                    highlight=True,
                    highlightcolor="rgba(255,255,255,0.75)",
                    width=3,
                    start=0.0,
                    end=z_max,
                    size=max(z_span / 7.0, 1e-9),
                    project=dict(z=True),
                ),
            ),
            lighting=dict(ambient=0.88, diffuse=0.4, specular=0.03, roughness=0.97),
            hovertemplate=(
                "Intercept %{customdata[0]:.4g}<br>"
                "Slope %{customdata[1]:.4g}<br>"
                f"{tip_hover('SSR')}"
                ": %{customdata[2]:,.1f}<br>"
                "Excess SSR %{customdata[3]:,.1f}"
                "<extra></extra>"
            ),
            name="SSR surface",
        )
    )
    current_excess = max(float(current_sse) - float(sse_min), 0.0)
    u_cand = (float(current_alpha) - float(alpha_ols)) / float(alpha_scale)
    v_cand = (float(current_beta) - float(beta_ols)) / float(beta_scale)
    z_pad = max(0.006 * z_span, 1e-12)

    if path_alphas is not None and path_betas is not None and path_excess is not None:
        pa = np.asarray(path_alphas, dtype=np.float64)
        pb = np.asarray(path_betas, dtype=np.float64)
        pz = np.asarray(path_excess, dtype=np.float64)
        if pa.size >= 2:
            fig.add_trace(
                go.Scatter3d(
                    x=(pa - float(alpha_ols)) / float(alpha_scale),
                    y=(pb - float(beta_ols)) / float(beta_scale),
                    z=pz + z_pad,
                    mode="lines",
                    line=dict(color="rgba(37, 99, 235, 0.85)", width=6),
                    name="Path to OLS",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # Drop lines from markers to the floor (scientific style).
    fig.add_trace(
        go.Scatter3d(
            x=[0.0, 0.0, None, u_cand, u_cand],
            y=[0.0, 0.0, None, v_cand, v_cand],
            z=[0.0, z_pad, None, 0.0, current_excess + z_pad],
            mode="lines",
            line=dict(color="rgba(100, 116, 139, 0.55)", width=3, dash="dot"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    # Floor markers at projected positions.
    fig.add_trace(
        go.Scatter3d(
            x=[0.0, u_cand],
            y=[0.0, v_cand],
            z=[0.0, 0.0],
            mode="markers",
            marker=dict(
                size=4,
                color=[ORANGE, BLUE],
                symbol=["diamond", "circle"],
                line=dict(width=1, color="white"),
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[z_pad],
            mode="markers+text",
            text=["OLS minimum"],
            textposition="middle right",
            textfont=dict(size=13, color=ORANGE, family=FONT),
            marker=dict(
                size=11,
                color=ORANGE,
                symbol="diamond",
                line=dict(width=2, color="white"),
            ),
            name="OLS minimum",
            hovertemplate=(
                f"{tip_hover('OLS')}<br>"
                f"Intercept {float(alpha_ols):.4g}<br>"
                f"Slope {float(beta_ols):.4g}<br>"
                f"{tip_hover('SSR')}: {float(sse_min):,.1f}<br>"
                "Excess SSR 0"
                "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[u_cand],
            y=[v_cand],
            z=[current_excess + z_pad],
            mode="markers+text",
            text=["Your line"],
            textposition="middle right",
            textfont=dict(size=13, color=BLUE, family=FONT),
            marker=dict(
                size=10,
                color=BLUE,
                symbol="circle",
                line=dict(width=2, color="white"),
            ),
            name="Your line",
            hovertemplate=(
                "Your line<br>"
                f"Intercept {float(current_alpha):.4g}<br>"
                f"Slope {float(current_beta):.4g}<br>"
                f"{tip_hover('SSR')}: {float(current_sse):,.1f}<br>"
                f"Excess SSR {current_excess:,.1f}"
                "<extra></extra>"
            ),
        )
    )

    u_min, u_max = float(np.nanmin(u)), float(np.nanmax(u))
    v_min, v_max = float(np.nanmin(v)), float(np.nanmax(v))
    u_ticks, u_labels = _surface_tick_labels(alpha_ols, alpha_scale, u_min, u_max)
    v_ticks, v_labels = _surface_tick_labels(beta_ols, beta_scale, v_min, v_max)
    axis_style = dict(
        showbackground=True,
        backgroundcolor="rgba(248, 250, 252, 0.9)",
        gridcolor="rgba(148, 163, 184, 0.22)",
        zeroline=False,
        showspikes=False,
        showline=False,
        ticks="outside",
        tickfont=dict(size=11, color="#334155"),
    )
    title_font = dict(size=13, color=INK, family=FONT)
    fig.update_layout(
        template="plotly_white",
        height=height,
        autosize=True,
        title=dict(
            text=title,
            x=0.0,
            xanchor="left",
            y=0.98,
            yanchor="top",
            font=dict(size=15, color=INK, family=FONT),
        ),
        margin=dict(l=4, r=4, t=36, b=4),
        paper_bgcolor="white",
        showlegend=False,
        uirevision=uirevision,
        scene=dict(
            domain=dict(x=[0.0, 1.0], y=[0.0, 0.96]),
            xaxis=dict(
                **axis_style,
                title=dict(text="Intercept α", font=title_font),
                range=[u_min, u_max],
                tickvals=u_ticks,
                ticktext=u_labels,
            ),
            yaxis=dict(
                **axis_style,
                title=dict(text="Slope β", font=title_font),
                range=[v_min, v_max],
                tickvals=v_ticks,
                ticktext=v_labels,
            ),
            zaxis=dict(
                **axis_style,
                title=dict(
                    text="Extra SSR above OLS",
                    font=dict(size=12, color=INK, family=FONT),
                ),
                range=[-0.02 * z_span, z_top],
            ),
            # Closer side-elevated view fills the panel; floor heatmap readable under surface.
            camera=dict(
                eye=dict(x=1.15, y=-1.3, z=0.95),
                center=dict(x=0.0, y=0.0, z=0.0),
                up=dict(x=0.0, y=0.0, z=1.0),
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.1, y=1.1, z=0.9),
            bgcolor="white",
        ),
    )
    return fig


def fit_metric_mini_chart(
    *,
    values: np.ndarray,
    y_range: tuple[float, float],
    x_max: int,
    title: str,
    height: int = 50,
    width: int = 75,
    uirevision: str = "fit-metric-mini",
) -> go.Figure:
    """Compact SSE/R² chart for lm() animation — curve only, fixed axes.

    Empty / non-finite ``values`` yield a blank plot (labels stay outside).
    """
    series = np.asarray(values, dtype=np.float64)
    empty = series.size == 0 or not np.any(np.isfinite(series))
    y0, y1 = float(y_range[0]), float(y_range[1])
    if not np.isfinite(y0) or not np.isfinite(y1) or y1 <= y0:
        if empty:
            y0, y1 = 0.0, 1.0
        else:
            finite = series[np.isfinite(series)]
            pad = max(0.04 * float(np.ptp(finite)), 1e-6)
            y0, y1 = float(finite.min()) - pad, float(finite.max()) + pad
    x_end = max(int(x_max), 0) if empty else max(int(x_max), max(int(series.size) - 1, 0))

    fig = go.Figure()
    if not empty:
        fig.add_trace(
            go.Scatter(
                x=np.arange(series.size, dtype=np.float64),
                y=series,
                mode="lines",
                line=dict(color=BLUE, width=2.6, shape="spline"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.update_xaxes(range=[-0.15, x_end + 0.15], visible=False, fixedrange=True)
    fig.update_yaxes(range=[y0, y1], visible=False, fixedrange=True)
    fig.update_layout(
        template="plotly_white",
        height=int(height),
        width=int(width),
        margin=dict(l=2, r=2, t=2 if not title else 14, b=2),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=(
            dict(
                text=title,
                x=0,
                y=1.0,
                xanchor="left",
                yanchor="top",
                font=dict(size=10, color=MUTED, family=FONT),
                pad=dict(t=0, b=0, l=0, r=0),
            )
            if title
            else None
        ),
        uirevision=uirevision,
    )
    return fig


def series_colours(count: int) -> list[str]:
    if count <= len(SERIES_COLOURS):
        return list(SERIES_COLOURS[:count])
    extra = count - len(SERIES_COLOURS)
    return list(SERIES_COLOURS) + [BLUE] * extra
