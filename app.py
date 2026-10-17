"""Regression Lab: a teaching app for sampling variability of a fitted line."""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_drawer import (
    DRAWER_INTRO,
    EMPTY_SAMPLE,
    PAGE_SIZE,
    R_COMMANDS,
    SELECTION_NOTE,
    DrawerSnapshot,
    chart_student_id,
    highlights_for,
    page_for_student,
    population_csv,
    population_page,
    sample_count_label,
    sample_csv,
    sample_frame,
    sample_token,
    selection_after_new_sample,
    set_drawer_open,
    state_after_chart_click,
    toggle_student,
)
from src.student_table import student_table
from src.charts import (
    BLUE,
    ORANGE,
    coefficient_histogram,
    distribution_scatter,
    introduction_figure,
    fit_metric_mini_chart,
    fitting_scatter,
    observation_index_from_selection,
    r2_lesson_scatter,
    null_slope_scatter,
    sampling_histogram,
    slope_sampling_figure,
    population_cloud_image,
    t_test_figure,
)
from src.introduction import (
    IntroductionView,
    displayed_coefficient_gap,
    estimate_sample,
    format_sampling_fraction,
    hide_population,
    population_cloud_caption,
    reveal_population,
    select_sample,
)
from src.population import (
    coefficients_match_metadata,
    load_metadata,
    load_population,
)
from src.regression import fit_ols
from src.fitting_line import (
    DEFAULT_SAMPLE_PCT,
    MAX_SAMPLE_PCT,
    MIN_SAMPLE_PCT,
    animation_metric_trajectories,
    animation_steps,
    interactive_n_cap,
    ols_fit,
    poor_candidate_line,
    predicted_scores,
    r_lm_commands,
    r_lm_summary_text,
    r_squared,
    r_standardisation_commands,
    regression_sum_of_squares,
    resolve_sample_n,
    STD_PREDICTOR_KEYS,
    STD_PREDICTOR_META,
    standardisation_student_frame,
    sample_sd,
    standardised_slope,
    subsample_observations,
    sum_of_squared_errors,
    total_sum_of_squares,
    zscore,
)
from src.dataset_import.analyse import (
    build_preview,
    numeric_column_names,
    pair_complete_count,
)
from src.dataset_import.io_files import parse_uploaded_table, read_excel_sheets
from src.dataset_import.registry import (
    activate_dataset,
    download_csv_bytes,
    ensure_registry_state,
    get_catalog,
    get_frame,
    is_synthetic_dataset,
    library_options,
    register_upload,
    resolve_fitting_dataset,
    source_group_label,
)
from src.stat_tips import (
    STAT_TIP_FORMULAS,
    STAT_TIPS,
    resolve_tip_code,
    tip_title,
    tip_visible,
    tipify,
)
from src.null_slope import (
    draw_null_sample,
    null_population,
    slope_t_test,
    t_density_curve,
)
from src.sampling import draw_sample
from src.sampling_distributions import (
    animated_draws,
    can_show_normal,
    estimate_mean,
    estimate_sd,
    experiment_for_sample_size,
    finite_estimates,
    new_experiment,
    normal_frequency,
    one_sample_for_size,
    one_sample_standard_errors,
    population_benchmark,
    benchmark_is_cached,
    relative_se_error,
)
SAMPLE_SIZES = [10, 20, 30, 50, 100, 200, 500, 1000, 5000]

# Main lab navigation (keys are stable; labels match the former st.tabs titles).
MAIN_LAB_TABS: tuple[tuple[str, str], ...] = (
    ("fitting", "Simple Linear Regression"),
    ("sampling", "Sampling & Inference"),
    ("library", "Dataset Library"),
)
MAIN_LAB_TAB_LABELS = dict(MAIN_LAB_TABS)

SAMPLING_INFERENCE_SUBTABS: tuple[tuple[str, str], ...] = (
    ("one_sample", "One Sample"),
    ("repeated", "Repeated Sampling"),
    ("estimate_se", "Estimate SE from One Sample"),
    ("ttest", "The t-test: Is the Slope Zero?"),
)
SAMPLING_INFERENCE_SUB_LABELS = dict(SAMPLING_INFERENCE_SUBTABS)

PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}
FIT_MINI_PLOTLY_CONFIG = {
    "displaylogo": False,
    "displayModeBar": False,
    "staticPlot": False,
    "responsive": False,
}

CSS = """
<style>
.stApp { background: #f4f6f8; }
[data-testid="stHeader"] { display: none !important; }
.block-container { max-width: 1240px; padding-top: 0.25rem; padding-bottom: 0.15rem; }
[data-testid="stMainBlockContainer"] { gap: 0.2rem !important; }
.st-key-main_nav {
  margin: 0 0 0.15rem 0;
  padding: 0;
  overflow: visible !important;
}
.st-key-main_nav [data-testid="stHorizontalBlock"] {
  flex-wrap: nowrap !important;
  align-items: center !important;
  column-gap: 0.45rem !important;
  row-gap: 0 !important;
  overflow: visible !important;
}
.st-key-main_nav [data-testid="stSegmentedControl"] {
  flex: 1 1 auto !important;
  min-width: 0 !important;
  overflow-x: auto !important;
  overflow-y: visible !important;
  scrollbar-width: thin;
}
.st-key-main_nav [data-testid="stSegmentedControl"] [data-testid="stButtonGroup"],
.st-key-main_nav [data-testid="stSegmentedControl"] > div {
  flex-wrap: nowrap !important;
  width: max-content !important;
  min-width: max-content !important;
  box-sizing: content-box !important;
}
.st-key-main_nav [data-testid="stSegmentedControl"] button {
  white-space: nowrap !important;
  overflow: visible !important;
  text-overflow: clip !important;
  flex: 0 0 auto !important;
  padding-left: 0.5rem !important;
  padding-right: 0.5rem !important;
  font-size: 0.82rem !important;
}
.st-key-dataset_status_row {
  margin: 0 0 0.2rem 0 !important;
  padding: 0 !important;
  width: 100% !important;
  flex: 0 0 auto !important;
}
.st-key-dataset_status_row [data-testid="stHorizontalBlock"] {
  justify-content: space-between !important;
  align-items: center !important;
  width: 100% !important;
  margin: 0 !important;
  gap: 0.75rem !important;
  flex-wrap: nowrap !important;
}
.st-key-dataset_status_row [data-testid="stElementContainer"]:has(.dataset-status),
.st-key-dataset_status_row [data-testid="stMarkdownContainer"]:has(.dataset-status) {
  flex: 1 1 auto !important;
  min-width: 0 !important;
  width: auto !important;
  margin: 0 !important;
  padding: 0 !important;
}
.st-key-dataset_status_row [data-testid="stElementContainer"]:has(button),
.st-key-dataset_status_row [data-testid="stLayoutWrapper"]:has(button) {
  flex: 0 0 auto !important;
  width: auto !important;
  margin: 0 0 0 auto !important;
}
.st-key-dataset_status_row .st-key-main_data_drawer_btn button {
  font-size: 0.8rem !important;
  font-weight: 600 !important;
  min-height: 1.85rem !important;
  padding: 0.15rem 0.75rem !important;
  white-space: nowrap !important;
  width: auto !important;
  border: 1px solid #94A3B8 !important;
  background: #ffffff !important;
  color: #334155 !important;
}
.st-key-app_data_drawer {
  margin: 0;
  min-height: 0;
}
.st-key-app_data_drawer > [data-testid="stVerticalBlock"] {
  gap: 0.55rem !important;
}
.st-key-app_data_drawer [data-testid="stMarkdownContainer"]:has(.dm-active-line),
.st-key-app_data_drawer [data-testid="stMarkdownContainer"]:has(.dm-selected-line),
.st-key-app_data_drawer [data-testid="stMarkdownContainer"]:has(.dm-meta) {
  margin: 0 !important;
  padding: 0 !important;
  height: auto !important;
  min-height: 0 !important;
  max-height: none !important;
  overflow: visible !important;
}
.st-key-app_data_drawer [data-testid="stMarkdownContainer"] p {
  white-space: normal !important;
  overflow: visible !important;
  text-overflow: clip !important;
  height: auto !important;
}
.dm-active-line, .dm-selected-line {
  display: block;
  box-sizing: border-box;
  width: 100%;
  margin: 0;
  padding: 0.4rem 0.55rem;
  border-radius: 6px;
  font-size: 0.8rem;
  line-height: 1.4;
  white-space: normal;
  overflow-wrap: anywhere;
  word-break: break-word;
  height: auto;
}
.dm-active-line {
  background: #EEF6FF;
  color: #1E3A5F;
  border-left: 3px solid #3B82F6;
}
.dm-selected-line {
  background: #F8FAFC;
  color: #334155;
  border-left: 3px solid #94A3B8;
}
.dm-selected-line.is-active {
  border-left-color: #3B82F6;
  background: #F0F7FF;
}
.dm-meta {
  display: block;
  box-sizing: border-box;
  width: 100%;
  margin: 0;
  padding: 0;
  font-size: 0.78rem;
  color: #64748B;
  line-height: 1.45;
  white-space: normal;
  overflow-wrap: anywhere;
  word-break: break-word;
  height: auto;
  max-height: none;
  overflow: visible;
}
.card {
    background: #ffffff;
    border: 1px solid #e6eaf0;
    border-radius: 12px;
    padding: 0.85rem 1rem;
    color: #1f2933;
    line-height: 1.45;
    margin: 0.35rem 0 0.7rem 0;
}
.card strong { color: #12263a; }
.feedback { color: #1f2933; font-size: 0.95rem; margin-top: 0.4rem; }
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e6eaf0;
    border-radius: 12px;
    padding: 0.55rem 0.75rem;
}
div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
.st-key-intro_toolbar {
    padding: 0 !important;
    margin: 0 0 0.2rem 0 !important;
    border-bottom: none;
    flex: 0 0 auto !important;
}
.st-key-intro_toolbar > [data-testid="stHorizontalBlock"] {
    align-items: flex-end !important;
    gap: 0.55rem !important;
    flex-wrap: nowrap !important;
}
.st-key-intro_toolbar [data-testid="stElementContainer"]:has([data-testid="stSlider"]),
.st-key-intro_toolbar [data-testid="stLayoutWrapper"]:has([data-testid="stSlider"]) {
    flex: 0 0 230px !important;
    width: 230px !important;
    max-width: 250px !important;
    min-width: 200px !important;
}
.st-key-intro_toolbar [data-testid="stSlider"] {
    width: 100% !important;
    padding-top: 0 !important;
    padding-bottom: 0.1rem !important;
}
.st-key-intro_toolbar [data-testid="stElementContainer"]:has(button),
.st-key-intro_toolbar [data-testid="stLayoutWrapper"]:has(button) {
    flex: 0 0 auto !important;
    width: auto !important;
}
.st-key-intro_toolbar button {
    white-space: nowrap !important;
}
.st-key-intro_header {
    flex: 0 0 auto !important;
    min-height: 0 !important;
    overflow: visible !important;
}
.st-key-intro_header > [data-testid="stVerticalBlock"] {
    gap: 0.25rem !important;
}
.st-key-intro_layout [data-testid="stElementContainer"]:has(.dataset-status),
.st-key-intro_layout [data-testid="stMarkdownContainer"]:has(.dataset-status) {
    margin: 0 0 0.15rem 0 !important;
    padding: 0 !important;
    flex: 0 0 auto !important;
    overflow: visible !important;
    height: auto !important;
    min-height: 1.4rem !important;
}
.st-key-intro_layout [data-testid="stMarkdownContainer"]:has(.intro-stats) {
    margin: 0 0 0.15rem 0 !important;
    padding: 0 !important;
}
.intro-stats {
    height: 6.35rem;
    margin: 0;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.intro-sample-line {
    display: flex;
    gap: 2.75rem;
    height: 1.3rem;
    align-items: baseline;
    font-size: 0.92rem;
}
.intro-sample-line .k { color: #6b7280; font-weight: 500; }
.intro-sample-line .v {
    color: #12263a;
    font-weight: 650;
    font-variant-numeric: tabular-nums;
}
.intro-grid {
    display: grid;
    grid-template-columns: 10.25rem 12.5rem 12.5rem;
    column-gap: 2.25rem;
    align-items: center;
    height: 4.7rem;
    font-size: 0.92rem;
    font-variant-numeric: tabular-nums;
}
.intro-grid .colhead {
    color: #6b7280;
    font-size: 0.68rem;
    font-weight: 650;
    letter-spacing: 0.07em;
}
.intro-grid .rowlabel { color: #6b7280; font-weight: 500; }
.intro-grid .rowlabel.strong { color: #1D4ED8; font-weight: 650; }
.intro-grid .rowlabel.pop-label { color: #EA580C; font-weight: 650; }
.intro-grid .cell {
    display: grid;
    grid-template-columns: 1.7rem 0.9rem 4.8rem;
    align-items: baseline;
    column-gap: 0.15rem;
}
.intro-grid .num { text-align: right; }
.intro-grid .est { color: #1D4ED8; font-weight: 680; }
.intro-grid .pop { color: #EA580C; font-weight: 650; }
.intro-grid .diff { color: #374151; font-weight: 600; }
.line-legend {
    display: flex;
    align-items: center;
    gap: 1.25rem;
    height: 1.15rem;
    margin: 0;
    color: #374151;
    font-size: 0.8rem;
}
.line-legend .swatch {
    display: inline-block;
    width: 1.35rem;
    height: 3px;
    margin-right: 0.4rem;
    border-radius: 2px;
    vertical-align: middle;
}
.line-legend .swatch.pop { background: #EA580C; }
.line-legend .swatch.est { background: #1D4ED8; }
.st-key-intro_view_data { display: flex; justify-content: flex-end; }
.chart-title {
    color: #12263a;
    font-size: 1.05rem;
    font-weight: 650;
    line-height: 1.35rem;
    margin: 0 0 0.1rem 0;
}
.st-key-intro_workspace > [data-testid="stVerticalBlock"] {
    gap: 0.2rem !important;
}
.drawer-kicker {
    color: #6b7280;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    line-height: 1.1rem;
    margin: 0;
}
.drawer-student {
    color: #1f2933;
    font-size: 0.8rem;
    line-height: 1.2rem;
    height: 1.2rem;
    margin: 0;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
}
.st-key-population_settings,
[data-testid="stElementContainer"]:has(.st-key-population_settings) {
    display: none !important;
}
.st-key-intro_layout {
    /* Nav sits above this block; Load Dataset shares the status row. */
    flex: none !important;
    height: calc(100dvh - 6.6rem) !important;
    max-height: calc(100dvh - 6.6rem) !important;
    min-height: 0 !important;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}
.st-key-intro_layout > [data-testid="stVerticalBlock"] {
    gap: 0.45rem !important;
    min-height: 0 !important;
    height: 100% !important;
    max-height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
}
/* Header keeps natural height; never shrink under the chart. */
.st-key-intro_layout [data-testid="stElementContainer"]:has(.st-key-intro_header),
.st-key-intro_layout [data-testid="stLayoutWrapper"]:has(.st-key-intro_header) {
    flex: 0 0 auto !important;
    flex-shrink: 0 !important;
    min-height: auto !important;
    height: auto !important;
    max-height: none !important;
    overflow: visible !important;
}
.st-key-intro_layout [data-testid="stElementContainer"]:has(.st-key-intro_workspace),
.st-key-intro_layout [data-testid="stLayoutWrapper"]:has(.st-key-intro_workspace) {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
}
.st-key-intro_workspace,
.st-key-intro_workspace > [data-testid="stLayoutWrapper"],
.st-key-intro_workspace [data-testid="stHorizontalBlock"],
.st-key-intro_workspace [data-testid="stColumn"],
.st-key-intro_workspace [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {
    flex: 1 1 auto !important;
    min-height: 0 !important;
}
.st-key-intro_workspace {
    display: flex;
    flex-direction: column;
    height: 100%;
}
.st-key-intro_workspace [data-testid="stHorizontalBlock"] {
    align-items: stretch;
    height: 100%;
}
.st-key-intro_workspace [data-testid="stColumn"],
.st-key-intro_workspace [data-testid="stColumn"] > [data-testid="stVerticalBlock"],
.st-key-intro_workspace > [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] {
    display: flex;
    flex-direction: column;
    height: 100%;
}
.st-key-intro_workspace [data-testid="stLayoutWrapper"]:has(.st-key-intro_chart_frame),
.st-key-intro_workspace [data-testid="stElementContainer"]:has(.st-key-intro_chart),
.st-key-intro_workspace [data-testid="stElementContainer"]:has(.st-key-intro_data) {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    display: flex;
    flex-direction: column;
}
.st-key-intro_chart_frame,
.st-key-intro_chart,
.st-key-intro_data {
    flex: 1 1 auto !important;
    height: 100% !important;
    min-height: 0 !important;
}
.st-key-intro_chart_frame [data-testid="stPlotlyChart"],
.st-key-intro_chart_frame [data-testid="stFullScreenFrame"] {
    height: 100% !important;
    min-height: 0 !important;
}
.st-key-intro_workspace [data-testid="stLayoutWrapper"]:has(.st-key-intro_data) {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    height: 100% !important;
    max-height: 100% !important;
    overflow: hidden !important;
}
.st-key-intro_data {
    overflow: hidden !important;
    height: 100% !important;
    max-height: 100% !important;
    min-height: 0 !important;
}
.st-key-intro_data [data-testid="stVerticalBlockBorderWrapper"],
.st-key-intro_data [data-testid="stVerticalBlock"] {
    min-height: 0;
}
.st-key-intro_population_rows,
.st-key-intro_sample_rows {
    flex: 1 1 0% !important;
    min-height: 0 !important;
    overflow: hidden !important;
}
.st-key-intro_population_rows .stBidiComponent,
.st-key-intro_sample_rows .stBidiComponent,
.st-key-intro_population_rows .stBidiComponent div,
.st-key-intro_sample_rows .stBidiComponent div {
    height: 100% !important;
    min-height: 0 !important;
    max-height: 100% !important;
}
.st-key-intro_layout [data-testid="stCaptionContainer"] { margin-bottom: 0 !important; }
.st-key-intro_data [data-testid="stTextInput"] input,
.st-key-intro_data [data-baseweb="select"] > div,
.st-key-intro_data [data-testid="stNumberInput"] input {
    min-height: 1.9rem !important;
    height: 1.9rem !important;
}
.st-key-intro_data [data-testid="stButton"] button,
.st-key-intro_data [data-testid="stDownloadButton"] button,
.st-key-intro_data [data-testid="stPopover"] button {
    min-height: 1.9rem;
    padding-top: 0.1rem;
    padding-bottom: 0.1rem;
}
.st-key-intro_page_line {
    align-items: center;
}
.st-key-intro_page_line [data-testid="stElementContainer"] {
    margin-bottom: 0 !important;
    width: auto !important;
    flex: 0 0 auto !important;
}
.st-key-intro_page_line [data-testid="stCaptionContainer"] {
    width: max-content !important;
}
.st-key-intro_data .st-key-intro_pop_prev button,
.st-key-intro_data .st-key-intro_pop_next button {
    min-height: 1.15rem !important;
    height: 1.15rem !important;
    min-width: 1.2rem !important;
    width: 1.2rem !important;
    padding: 0 !important;
    line-height: 1 !important;
    color: #6b7280 !important;
    font-size: 1.05rem !important;
}
@media (max-width: 980px) {
    .st-key-intro_layout {
        height: auto !important;
        max-height: none !important;
        overflow: visible;
    }
    .st-key-intro_workspace [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        height: auto;
    }
    .st-key-intro_chart_frame,
    .st-key-intro_data {
        flex: none !important;
        height: min(360px, 48dvh) !important;
    }
}
.drawer-student strong { color: #12263a; }
.st-key-intro_layout [data-testid="stMarkdownContainer"]:has(.intro-stats),
.st-key-intro_layout [data-testid="stMarkdownContainer"]:has(.line-legend),
.st-key-intro_layout [data-testid="stMarkdownContainer"]:has(.chart-title),
.st-key-intro_layout [data-testid="stMarkdownContainer"]:has(.drawer-kicker),
.st-key-intro_layout [data-testid="stMarkdownContainer"]:has(.drawer-student) {
    margin-bottom: 0;
}
.dist-meta, .dist-stat {
    margin: 0;
    height: 1.2rem;
    line-height: 1.2rem;
    font-size: 0.82rem;
    color: #374151;
    white-space: nowrap;
}
.dist-meta {
    display: flex;
    gap: 1.75rem;
    align-items: baseline;
    height: 1.45rem;
    line-height: 1.45rem;
}
.dist-count {
    font-size: 1.05rem;
    font-weight: 700;
    color: #12263a;
}
.dist-note, .dist-sd {
    margin: 0;
    height: 1.2rem;
    line-height: 1.2rem;
    font-size: 0.82rem;
    color: #52606D;
    white-space: nowrap;
}
.dist-sd strong { color: #12263a; font-weight: 650; }
.dist-calc, .dist-compare {
    margin: 0;
    font-size: 0.88rem;
    line-height: 1.28rem;
    color: #374151;
}
.dist-calc div, .dist-compare div {
    display: flex;
    justify-content: space-between;
    gap: 0.5rem;
}
.dist-calc-title, .dist-compare-head {
    font-weight: 650;
    color: #12263a;
}
.dist-more { font-size: 0.74rem; color: #52606D; margin: 0; }
.dist-more summary { cursor: pointer; }
.bench, .dist-compare .pop { color: #EA580C; font-weight: 650; }
.ref { color: #4B5563; font-weight: 650; }
.dist-compare .est, .dist-calc .est { color: #1D4ED8; font-weight: 650; }
.st-key-dist_step { margin: 0 0 0.1rem 0; }
.dist-stat { display: flex; gap: 1.1rem; }
.dist-stat.compare {
    display: block;
    height: auto;
    white-space: normal;
    line-height: 1.15rem;
}
.dist-stat .pop { color: #EA580C; font-weight: 650; }
.dist-stat .est { color: #1D4ED8; font-weight: 650; }
.st-key-dist_layout {
    flex: none !important;
    height: calc(100dvh - 6.15rem) !important;
    max-height: calc(100dvh - 6.15rem) !important;
    min-height: 0 !important;
    overflow: hidden;
    display: flex;
    flex-direction: column;
}
.st-key-dist_layout > [data-testid="stLayoutWrapper"]:has(.st-key-dist_workspace),
.st-key-dist_layout > [data-testid="stLayoutWrapper"]:has(.st-key-one_workspace) {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    display: flex;
    flex-direction: column;
}
.st-key-dist_workspace,
.st-key-dist_workspace > [data-testid="stLayoutWrapper"],
.st-key-dist_workspace [data-testid="stHorizontalBlock"],
.st-key-dist_workspace [data-testid="stColumn"],
.st-key-dist_workspace [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    height: 100%;
}
.st-key-dist_workspace [data-testid="stHorizontalBlock"] { align-items: stretch; }
.st-key-dist_workspace [data-testid="stColumn"],
.st-key-dist_workspace [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {
    display: flex;
    flex-direction: column;
}
.st-key-dist_scatter,
.st-key-dist_hists,
.st-key-dist_beta {
    min-height: 0 !important;
    flex: 1 1 auto !important;
}
.st-key-dist_layout [data-testid="stMarkdownContainer"]:has(.dist-meta),
.st-key-dist_layout [data-testid="stMarkdownContainer"]:has(.dist-stat),
.st-key-dist_layout [data-testid="stMarkdownContainer"]:has(.dist-note),
.st-key-dist_layout [data-testid="stMarkdownContainer"]:has(.dist-sd),
.st-key-dist_layout [data-testid="stMarkdownContainer"]:has(.dist-calc),
.st-key-dist_layout [data-testid="stMarkdownContainer"]:has(.dist-compare),
.st-key-dist_layout [data-testid="stMarkdownContainer"]:has(.dist-intro),
.st-key-dist_layout [data-testid="stMarkdownContainer"]:has(.dist-kicker),
.st-key-dist_layout [data-testid="stMarkdownContainer"]:has(.dist-steps),
.st-key-dist_layout [data-testid="stMarkdownContainer"]:has(.dist-legend),
.st-key-dist_layout [data-testid="stMarkdownContainer"]:has(.dist-conclusion),
.st-key-dist_layout [data-testid="stMarkdownContainer"]:has(.se-table),
.st-key-dist_layout [data-testid="stMarkdownContainer"]:has(.t-panel),
.st-key-dist_layout [data-testid="stMarkdownContainer"]:has(.se-discover),
.st-key-dist_layout [data-testid="stMarkdownContainer"]:has(.one-col-head) {
    margin-bottom: 0 !important;
}
.one-col-head { margin: 0 0 0.1rem 0; }
.one-col-head .title { margin: 0; font-size: 0.95rem; font-weight: 750; color: #12263a; }
.one-col-head .sub { margin: 0.05rem 0 0 0; font-size: 0.82rem; color: #52606D; line-height: 1.2rem; }
.se-discover { margin: 0.25rem 0 0 0; }
.se-discover .kicker { margin: 0 0 0.25rem 0; font-size: 0.82rem; font-weight: 750; letter-spacing: 0.04em; color: #12263a; }
.se-cards { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.45rem; }
.se-card { border: 1px solid #E6EAF0; border-radius: 8px; padding: 0.4rem 0.55rem; background: #F8FAFC; }
.se-card .lab { display: block; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.02em; color: #52606D; line-height: 1.15rem; }
.se-card .val { display: block; margin-top: 0.1rem; font-size: 1.35rem; font-weight: 780; font-variant-numeric: tabular-nums; }
.se-card.bench .val { color: #EA580C; }
.se-card.ref .val { color: #4B5563; }
.se-card.est .val { color: #1D4ED8; }
.se-bars { display: flex; flex-direction: column; gap: 0.28rem; margin: 0.35rem 0 0 0; }
.se-bar-row { display: grid; grid-template-columns: 9.5rem 1fr; gap: 0.45rem; align-items: center; }
.se-bar-row .name { font-size: 0.78rem; font-weight: 650; color: #52606D; }
.se-bar-track { height: 13px; background: #EEF1F4; border-radius: 4px; overflow: hidden; }
.se-bar-fill { display: block; height: 100%; border-radius: 4px; min-width: 0; }
.se-bar-fill.bench { background: #EA580C; }
.se-bar-fill.ref { background: #6B7280; }
.se-bar-fill.est { background: #1D4ED8; }
.se-error { margin: 0.28rem 0 0 0; font-size: 0.95rem; font-weight: 700; color: #12263a; }
.se-close { margin: 0.1rem 0 0 0; font-size: 0.88rem; line-height: 1.25rem; color: #374151; }
.dist-intro-sub { margin: 0.05rem 0 0.15rem 0; font-size: 0.9rem; line-height: 1.25rem; color: #52606D; }
.t-panel { margin: 0.15rem 0 0 0; color: #1F2933; }
.t-strip { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)) 1.55fr; gap: 0.35rem; align-items: stretch; }
.t-cell { background: #F8FAFC; border: 1px solid #E6EAF0; border-radius: 6px; padding: 0.28rem 0.45rem; min-width: 0; }
.t-cell .lab { display: block; color: #52606D; font-size: 0.72rem; font-weight: 650; letter-spacing: 0.01em; }
.t-cell .val { display: block; margin-top: 0.05rem; font-variant-numeric: tabular-nums; font-size: 1.05rem; font-weight: 750; color: #12263a; white-space: nowrap; }
.t-cell.emph .val { color: #1D4ED8; font-size: 1.15rem; }
.t-decision { display: flex; align-items: center; justify-content: center; border-radius: 6px; padding: 0.28rem 0.55rem; font-size: 1.05rem; font-weight: 780; text-align: center; }
.t-decision.keep { background: #F3F4F6; color: #1F2933; border: 1px solid #E5E7EB; }
.t-decision.reject { background: #FEE2E2; color: #991B1B; border: 1px solid #FECACA; }
.t-read { margin: 0.2rem 0 0 0; font-size: 0.88rem; line-height: 1.25rem; color: #374151; }
.t-scale { margin: 0.05rem 0 0.05rem 0; font-size: 0.88rem; line-height: 1.2rem; color: #374151; text-align: center; }
.st-key-dist_layout [data-testid="stMarkdownContainer"]:has(.t-scale) { margin-bottom: 0 !important; }
.st-key-dist_layout:has(.dist-step-one),
.st-key-dist_layout:has(.dist-step-test) {
    height: auto !important;
    max-height: none !important;
    overflow: visible;
}
.st-key-fit_shell {
  width: 100%; max-width: 100%; margin: 0; padding: 0;
  box-sizing: border-box;
}
.st-key-fit_shell > [data-testid="stVerticalBlock"] { gap: 0.04rem !important; }
.st-key-app_data_drawer [data-testid="stHorizontalBlock"] {
  flex-wrap: wrap !important;
  row-gap: 0.35rem !important;
  align-items: flex-end !important;
}
.st-key-app_data_drawer [data-testid="stExpander"] {
  margin: 0.15rem 0 0.35rem 0 !important;
}
.st-key-app_data_drawer [data-testid="stSelectbox"] {
  margin-bottom: 0.15rem !important;
}
.dataset-status {
  margin: 0;
  padding: 0;
  font-size: 0.84rem;
  font-weight: 600;
  color: #334155;
  line-height: 1.35rem;
  font-variant-numeric: tabular-nums;
}
.st-key-fit_shell [data-testid="stExpander"] {
  margin: 0 0 0.05rem 0 !important;
  border: none !important;
  background: transparent !important;
}
.st-key-fit_shell [data-testid="stExpander"] details {
  border: none !important;
  background: transparent !important;
}
.fit-slr-intro {
  margin: 0; padding: 0.1rem 0 0 0;
}
.fit-slr-intro .cols {
  display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.35rem 0.85rem;
  margin: 0 0 0.2rem 0;
}
@media (max-width: 700px) {
  .fit-slr-intro .cols { grid-template-columns: 1fr; gap: 0.25rem; }
}
.fit-slr-intro .col .lab {
  margin: 0 0 0.05rem 0; font-size: 0.66rem; font-weight: 780;
  letter-spacing: 0.06em; text-transform: uppercase; color: #64748B;
  line-height: 1.05rem;
}
.fit-slr-intro .col .txt {
  margin: 0; font-size: 0.8rem; color: #334155; line-height: 1.2rem;
}
.fit-slr-intro .eq {
  margin: 0; text-align: center;
  font-size: 0.98rem; font-weight: 700; color: #1E3A8A;
  font-variant-numeric: tabular-nums; line-height: 1.3rem;
}
.fit-slr-intro .eq sub { font-size: 0.7em; }
.fit-slr-intro .cap {
  margin: 0.05rem 0 0 0; text-align: center;
  font-size: 0.74rem; color: #64748B; line-height: 1.15rem;
}
.fit-caption { margin: 0; font-size: 0.74rem; color: #7B8794; }
.fit-sample-n { margin: 0; font-size: 0.78rem; color: #374151; font-variant-numeric: tabular-nums; }
.fit-sample-n .cap { color: #9B6B2F; }
.st-key-fit_lab_tabs {
  margin: 0;
  padding: 0;
}
/* Fit toolbar: one st.columns row — no forced widths / wrap / scroll. */
.st-key-fit_toolbar {
  margin: 0 !important;
  padding: 0 !important;
}
.st-key-fit_toolbar [data-testid="stHorizontalBlock"] {
  align-items: center !important;
  gap: 0.35rem !important;
  margin: 0 !important;
}
/* Suppress fade/flash when Fit toolbar toggles remount siblings. */
.st-key-fit_toolbar [data-testid="stElementContainer"],
.st-key-fit_view_panel [data-testid="stElementContainer"],
.st-key-fit_metrics_box [data-testid="stElementContainer"],
.st-key-fit_toolbar [data-testid="stPlotlyChart"],
.st-key-fit_view_panel [data-testid="stPlotlyChart"],
.st-key-fit_metrics_box [data-testid="stPlotlyChart"] {
  transition: none !important;
  animation: none !important;
  opacity: 1 !important;
}
.fit-toggle-lab {
  margin: 0 0 0.05rem 0;
  font-size: 0.72rem;
  line-height: 0.9rem;
  font-weight: 650;
  color: #64748B;
  text-align: center;
  white-space: nowrap;
}
.st-key-fit_resid_wrap,
.st-key-fit_coefs_wrap {
  margin: 0 !important;
  padding: 0 0.4rem !important;
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  box-sizing: border-box !important;
}
.st-key-fit_resid_wrap [data-testid="stCheckbox"],
.st-key-fit_coefs_wrap [data-testid="stCheckbox"],
.st-key-fit_resid_wrap [data-testid="stWidgetLabel"],
.st-key-fit_coefs_wrap [data-testid="stWidgetLabel"] {
  margin: 0 !important;
}
.st-key-fit_resid_wrap [data-testid="stElementContainer"],
.st-key-fit_coefs_wrap [data-testid="stElementContainer"] {
  display: flex !important;
  justify-content: center !important;
  width: 100% !important;
}
.st-key-fit_view_wrap {
  margin: 0 !important;
  padding: 0 !important;
  display: flex !important;
  justify-content: center !important;
  align-items: center !important;
}
.st-key-fit_view_wrap [data-testid="stElementContainer"] {
  margin: 0 !important;
  width: fit-content !important;
}
.st-key-fit_rcode_wrap {
  margin: 0 !important;
  padding: 0 !important;
}
.st-key-fit_rcode_wrap [data-testid="stPopover"] button {
  width: 85px !important;
  min-width: 85px !important;
  max-width: 85px !important;
  white-space: nowrap !important;
  overflow: visible !important;
  text-overflow: clip !important;
  padding-left: 0.45rem !important;
  padding-right: 0.35rem !important;
}
.st-key-fit_metrics_box {
  margin: 0 !important;
  padding: 6px !important;
  border: 1px solid #D4A5A5 !important;
  border-radius: 6px !important;
  background: transparent !important;
  box-sizing: border-box !important;
}
.st-key-fit_metrics_box [data-testid="stHorizontalBlock"] {
  gap: 0.3rem !important;
  margin: 0 !important;
  align-items: stretch !important;
}
.st-key-fit_metrics_box [data-testid="stColumn"] {
  flex: 1 1 0 !important;
  min-width: 0 !important;
}
.st-key-fit_metrics_box [data-testid="stElementContainer"] {
  margin: 0 !important;
}
.st-key-fit_metrics_box [data-testid="stPlotlyChart"],
.st-key-fit_metrics_box .js-plotly-plot,
.st-key-fit_metrics_box .plot-container {
  width: 75px !important;
  max-width: 75px !important;
  height: 50px !important;
}
.fit-metric-lab {
  margin: 0 0 1px 0;
  height: 0.9rem;
  font-size: 0.72rem;
  line-height: 0.9rem;
  font-weight: 650;
  color: #64748B;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.fit-metric-r2-range {
  margin: 0 0 1px 0;
  height: 0.9rem;
  font-size: 0.72rem;
  line-height: 0.9rem;
  font-weight: 600;
  color: #64748B;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.fit-metric-r2-spacer {
  visibility: hidden;
  pointer-events: none;
  user-select: none;
}
.stat-tip {
  position: relative; display: inline; cursor: help;
  border-bottom: 1px dotted #94A3B8; font: inherit; color: inherit;
  outline: none;
}
.stat-tip:focus, .stat-tip:focus-visible {
  border-bottom-color: #1D4ED8;
  box-shadow: 0 0 0 2px rgba(29, 78, 216, 0.25);
  border-radius: 3px;
}
.stat-tip-bubble {
  display: none; position: absolute; z-index: 40;
  left: 50%; bottom: calc(100% + 0.35rem); transform: translateX(-50%);
  width: max-content; max-width: 15.5rem;
  padding: 0.4rem 0.55rem; border-radius: 7px;
  background: #0F172A; color: #F8FAFC;
  font-size: 0.72rem; font-weight: 500; line-height: 1.2rem;
  letter-spacing: 0; text-transform: none; white-space: normal;
  box-shadow: 0 6px 18px rgba(15, 23, 42, 0.22);
  pointer-events: none;
}
.stat-tip-bubble strong {
  display: block; margin: 0 0 0.12rem 0; font-size: 0.74rem; font-weight: 750;
  color: #FFFFFF;
}
.stat-tip:hover .stat-tip-bubble,
.stat-tip:focus .stat-tip-bubble,
.stat-tip:focus-within .stat-tip-bubble { display: block; }
/* —— What can I use this model for? (one outer box) —— */
.st-key-fit_model_use {
  margin: 0.1rem 0 0 0;
  padding: 0.4rem 0.65rem 0.35rem 0.65rem;
  width: 100%;
  box-sizing: border-box;
  background: #FFFFFF;
  border: 1px solid #E2E8F0;
  border-radius: 10px;
}
.fit-use-title {
  margin: 0 0 4px 0;
  font-size: 0.74rem; font-weight: 800; font-family: inherit;
  letter-spacing: 0.07em; text-transform: uppercase; color: #475569;
  line-height: 1.1rem;
}
/* Equal-width columns that stretch to one shared height (content-driven). */
.st-key-fit_model_use [data-testid="stHorizontalBlock"]:has(.st-key-fit_use_predict) {
  display: flex !important;
  flex-direction: row !important;
  flex-wrap: wrap !important;
  align-items: stretch !important;
  gap: 12px !important;
}
.st-key-fit_model_use [data-testid="stHorizontalBlock"]:has(.st-key-fit_use_predict) > [data-testid="stColumn"] {
  flex: 1 1 calc(50% - 6px) !important;
  width: auto !important;
  max-width: none !important;
  min-width: min(100%, 260px) !important;
  align-self: stretch !important;
  display: flex !important;
  flex-direction: column !important;
  height: auto !important;
}
.st-key-fit_model_use [data-testid="stHorizontalBlock"]:has(.st-key-fit_use_predict) > [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {
  flex: 1 1 auto !important;
  display: flex !important;
  flex-direction: column !important;
  height: 100% !important;
  min-height: 0 !important;
}
/* Streamlit wraps each keyed box in stLayoutWrapper with flex: 0 1 auto — force fill. */
.st-key-fit_model_use [data-testid="stLayoutWrapper"]:has(> .st-key-fit_use_predict),
.st-key-fit_model_use [data-testid="stLayoutWrapper"]:has(> .st-key-fit_use_explain) {
  flex: 1 1 auto !important;
  height: 100% !important;
  min-height: 0 !important;
  display: flex !important;
  flex-direction: column !important;
  align-self: stretch !important;
}
.st-key-fit_use_predict,
.st-key-fit_use_explain {
  margin: 0;
  padding: 12px;
  box-sizing: border-box;
  border-radius: 8px;
  display: flex !important;
  flex-direction: column !important;
  gap: 12px;
  flex: 1 1 auto !important;
  width: 100%;
  height: 100% !important;
  min-height: 100% !important;
  min-width: 0;
  overflow: hidden;
}
.st-key-fit_use_predict {
  background: linear-gradient(180deg, #FFFBF5 0%, #FFF7ED 100%);
  border: 1px solid #FDBA74;
}
.st-key-fit_use_explain {
  background: linear-gradient(180deg, #F8FBFF 0%, #EFF6FF 100%);
  border: 1px solid #93C5FD;
}
.fit-use-lab {
  margin: 0;
  font-size: 0.7rem; font-weight: 780; font-family: inherit;
  letter-spacing: 0.06em; text-transform: uppercase;
  line-height: 1.25rem;
  min-height: 1.35rem;
  display: flex;
  align-items: flex-start;
  white-space: normal;
}
.fit-use-lab.predict { color: #C2410C; min-height: 2.5rem; }
.fit-use-lab.explain { color: #1D4ED8; min-height: 2.5rem; }
.st-key-fit_use_predict [data-testid="stMarkdownContainer"] p,
.st-key-fit_use_explain [data-testid="stMarkdownContainer"] p { margin: 0 !important; }
.st-key-fit_use_predict [data-testid="stHorizontalBlock"],
.st-key-fit_use_explain [data-testid="stHorizontalBlock"] {
  gap: 8px !important;
  align-items: center !important;
  flex-wrap: wrap !important;
}
.st-key-fit_use_predict [data-testid="stHorizontalBlock"] > [data-testid="stElementContainer"],
.st-key-fit_use_predict [data-testid="stHorizontalBlock"] [data-testid="stMarkdownContainer"],
.st-key-fit_use_predict [data-testid="stHorizontalBlock"] [data-testid="stTextInput"],
.st-key-fit_use_predict [data-testid="stHorizontalBlock"] [data-testid="stButton"] {
  margin: 0 !important;
  padding: 0 !important;
  display: flex !important;
  align-items: center !important;
  min-height: 2.15rem !important;
}
.st-key-fit_use_predict [data-testid="stTextInput"] {
  min-width: 6.5rem; max-width: 8.5rem;
}
.st-key-fit_use_predict [data-testid="stTextInput"] label { display: none; }
.st-key-fit_use_predict [data-testid="stTextInput"] > div {
  margin-bottom: 0 !important;
  margin-top: 0 !important;
  display: flex !important;
  align-items: center !important;
}
.st-key-fit_use_predict [data-testid="stTextInput"] input {
  background: #FFF7ED !important; border: 1px solid #FDBA74 !important;
  border-radius: 7px !important; color: #9A3412 !important; font-weight: 650 !important;
  font-variant-numeric: tabular-nums;
  font-size: 0.95rem !important;
  padding: 0 0.55rem !important;
  min-height: 2.15rem !important; height: 2.15rem !important;
  line-height: 2.15rem !important;
  box-shadow: none !important;
  display: flex !important;
  align-items: center !important;
}
.st-key-fit_use_predict [data-testid="stTextInput"] input::placeholder {
  color: #C4A484 !important; font-weight: 500 !important;
  line-height: 2.15rem !important;
  position: relative;
  top: 0 !important;
  transform: none !important;
}
.st-key-fit_use_predict [data-testid="stTextInput"] input:focus {
  border-color: #EA580C !important; box-shadow: 0 0 0 1px #FDBA74 !important;
}
.st-key-fit_use_predict .st-key-fit_predict_btn {
  display: flex !important;
  align-items: center !important;
}
.st-key-fit_use_predict .st-key-fit_predict_btn button {
  background: #EA580C !important; color: #fff !important; border: 1px solid #C2410C !important;
  border-radius: 7px !important; font-weight: 750 !important;
  font-size: 0.95rem !important;
  padding: 0 0.95rem !important;
  min-height: 2.15rem !important; height: 2.15rem !important;
  line-height: 2.15rem !important;
  display: inline-flex !important;
  align-items: center !important;
  margin: 0 !important;
}
.st-key-fit_use_predict .st-key-fit_predict_btn button:hover {
  background: #C2410C !important; border-color: #9A3412 !important;
}
.fit-predict-key {
  margin: 0; font-size: 0.9rem; font-weight: 650; font-family: inherit;
  color: #9A3412; line-height: 2.15rem; height: 2.15rem;
  display: flex; align-items: center; white-space: nowrap;
}
.fit-predict-result {
  margin: 0; min-width: 0;
  display: flex; flex-direction: column; gap: 4px;
}
.fit-predict-yhat,
.fit-predict-eq {
  margin: 0; min-width: 0;
  font-family: inherit; font-variant-numeric: tabular-nums;
  line-height: 1.3rem; white-space: normal; overflow-wrap: anywhere;
}
.fit-predict-yhat { font-size: 0.86rem; color: #1F2937; }
.fit-predict-yhat b { color: #EA580C; font-weight: 780; }
.fit-predict-eq { font-size: 0.82rem; font-weight: 650; color: #9A3412; }
.fit-predict-eq b { color: #EA580C; font-weight: 780; }
.fit-predict-warn {
  margin: 0.15rem 0 0 0;
  padding: 0.35rem 0.5rem;
  font-size: 0.8rem;
  font-weight: 650;
  line-height: 1.25rem;
  color: #9A3412;
  background: #FFF7ED;
  border-left: 3px solid #EA580C;
  border-radius: 4px;
}
.fit-explain-student {
  margin: 0; font-size: 0.9rem; font-weight: 750; color: #0F172A;
  line-height: 1.75rem; white-space: nowrap;
}
.st-key-fit_use_explain .st-key-fit_clear_student button {
  background: #F8FAFC !important; color: #64748B !important; border: 1px solid #CBD5E1 !important;
  border-radius: 6px !important; font-weight: 700 !important; padding: 0.15rem 0.5rem !important;
  min-height: 1.75rem !important; height: 1.75rem !important; line-height: 1.2 !important;
}
.st-key-fit_use_explain .st-key-fit_clear_student button:hover {
  background: #FEE2E2 !important; color: #B91C1C !important; border-color: #FECACA !important;
}
.fit-explain-vals {
  display: flex; flex-wrap: wrap; gap: 4px 14px;
  margin: 0; min-width: 0;
  font-size: 0.84rem; font-variant-numeric: tabular-nums; line-height: 1.25rem;
}
.fit-explain-vals .k { color: #64748B; font-weight: 600; margin-right: 0.2rem; }
.fit-explain-vals .y .v { color: #1D4ED8; font-weight: 780; }
.fit-explain-vals .yhat .v { color: #EA580C; font-weight: 780; }
.fit-explain-vals .mean .v { color: #64748B; font-weight: 780; }
.fit-explain-decomp {
  margin: 0; padding: 6px 8px; min-width: 0;
  background: rgba(255, 255, 255, 0.55); border-radius: 6px;
}
.fit-explain-decomp .words {
  margin: 0 0 4px 0; font-size: 0.88rem; font-weight: 700; color: #475569;
  line-height: 1.25rem;
}
.fit-explain-decomp .words .total { color: #64748B; font-weight: 800; }
.fit-explain-decomp .words .model { color: #1D4ED8; font-weight: 800; }
.fit-explain-decomp .words .resid { color: #EA580C; font-weight: 800; }
.fit-explain-decomp .nums {
  margin: 0; font-size: 1.15rem; font-weight: 820; font-variant-numeric: tabular-nums;
  line-height: 1.4rem; overflow-wrap: anywhere;
}
.fit-explain-decomp .nums .total { color: #64748B; }
.fit-explain-decomp .nums .model { color: #1D4ED8; }
.fit-explain-decomp .nums .resid { color: #EA580C; }
.fit-explain-decomp .nums .op { color: #94A3B8; font-weight: 700; padding: 0 0.15rem; }
.st-key-fit_explain_foot {
  margin-top: auto;
  padding-top: 4px;
  border-top: none !important;
}
.st-key-fit_explain_foot [data-testid="stHorizontalBlock"] {
  gap: 8px !important;
  align-items: center !important;
}
.st-key-fit_explain_foot [data-testid="stElementContainer"],
.st-key-fit_explain_foot [data-testid="stMarkdownContainer"],
.st-key-fit_explain_foot [data-testid="stButton"] {
  margin: 0 !important;
  padding: 0 !important;
  display: flex !important;
  align-items: center !important;
}
.fit-explain-r2line {
  margin: 0; font-size: 0.88rem; font-weight: 800; color: #1D4ED8;
  font-variant-numeric: tabular-nums;
  line-height: 1.75rem;
  height: 1.75rem;
}
.fit-explain-copy {
  margin: 0; font-size: 0.8rem; color: #334155; line-height: 1.35rem;
}
.fit-explain-warn {
  margin: 0; font-size: 0.76rem; color: #9A3412; line-height: 1.25rem;
}
.st-key-fit_use_explain .st-key-fit_open_r2 {
  display: flex !important;
  align-items: center !important;
}
.st-key-fit_use_explain .st-key-fit_open_r2 button {
  background: #EFF6FF !important; color: #1D4ED8 !important; border: 1px solid #93C5FD !important;
  border-radius: 6px !important; font-weight: 700 !important;
  padding: 0 0.55rem !important;
  min-height: 1.75rem !important; height: 1.75rem !important;
  line-height: 1.75rem !important;
  display: inline-flex !important;
  align-items: center !important;
  margin: 0 !important;
}
.st-key-fit_use_explain .st-key-fit_open_r2 button:hover {
  background: #DBEAFE !important; border-color: #60A5FA !important;
}
.st-key-fit_view_panel {
  margin: 0;
  padding: 0;
  width: 100%;
  box-sizing: border-box;
}
.st-key-fit_shell [data-testid="stHorizontalBlock"] {
  width: 100%;
}
.fit-table-sse {
  margin: 0.25rem 0 0.15rem 0; padding: 0.35rem 0.55rem;
  background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px;
  font-size: 0.9rem; font-weight: 650; color: #0F172A;
  font-variant-numeric: tabular-nums;
}
.fit-table-sse b { color: #1D4ED8; font-weight: 780; }
.fit-r2-heading {
  margin: 0 0 0.35rem 0;
  font-size: 1.05rem; font-weight: 750; color: #0F172A;
  line-height: 1.35rem;
}
.fit-std-banner {
  margin: 0 0 0.35rem 0; padding: 0.4rem 0.65rem;
  background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px;
  font-size: 0.92rem; font-weight: 750; color: #1E3A8A; line-height: 1.3rem;
}
.fit-std-insight {
  margin: 0.25rem 0 0.15rem 0; padding: 0.4rem 0.6rem;
  background: #EFF6FF; border: 1px solid #93C5FD; border-radius: 8px;
  font-size: 0.88rem; font-weight: 700; color: #1E3A8A; line-height: 1.3rem;
}
.fit-std-insight b { color: #1D4ED8; font-weight: 820; }
.fit-std-note {
  margin: 0.1rem 0 0.2rem 0; font-size: 0.84rem; font-weight: 650;
  color: #475569; line-height: 1.25rem;
}
.fit-std-formula-pane {
  margin: 0; padding: 0.35rem 0.45rem 0.2rem;
  background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px;
}
.fit-std-formula-pane .fit-std-col-lab { margin-bottom: 0.1rem; }
.fit-std-formula-pane [data-testid="stMarkdownContainer"] {
  margin: 0 !important; padding: 0 !important;
}
.fit-std-formula-pane .katex-display {
  margin: 0.35rem 0 !important;
}
.fit-std-col-lab {
  margin: 0 0 0.2rem 0; font-size: 0.74rem; font-weight: 700;
  letter-spacing: 0.04em; text-transform: uppercase; color: #64748B;
}
.fit-std-slope {
  margin: 0.1rem 0 0.25rem 0; padding: 0.25rem 0.4rem;
  background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 7px;
  font-size: 0.84rem; color: #0F172A; line-height: 1.25rem;
  font-variant-numeric: tabular-nums;
}
.fit-std-slope b { color: #1D4ED8; font-weight: 780; }
.fit-std-how {
  margin: 0.2rem 0 0.15rem 0; padding: 0.4rem 0.55rem;
  background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
  border: 1px solid #E2E8F0; border-radius: 9px;
}
.fit-std-how .formula {
  margin: 0.2rem 0; padding: 0.3rem 0.45rem;
  background: #F1F5F9; border-radius: 6px; border: 1px solid #E2E8F0;
  font-size: 0.88rem; font-weight: 650; color: #1E3A8A;
  font-variant-numeric: tabular-nums;
}
.fit-std-how .formula b { color: #1D4ED8; font-weight: 780; }
.fit-std-how ul {
  margin: 0.25rem 0 0 1.1rem; padding: 0;
  font-size: 0.82rem; color: #334155; line-height: 1.3rem;
}
.fit-std-how li { margin: 0.1rem 0; }
.fit-std-how .pair {
  display: grid; grid-template-columns: 1fr 1fr; gap: 0.45rem;
  margin: 0.35rem 0 0.1rem 0;
}
.fit-std-how .pair .blk {
  margin: 0; padding: 0.4rem 0.5rem;
  background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 7px;
}
.fit-std-how .pair .blk .lab {
  margin: 0 0 0.2rem 0; font-size: 0.72rem; font-weight: 750;
  letter-spacing: 0.04em; text-transform: uppercase; color: #64748B;
}
.fit-std-how .pair .blk p {
  margin: 0; font-size: 0.82rem; color: #334155; line-height: 1.28rem;
}
.fit-std-same { color: #047857; font-weight: 700; }
.fit-r2-lesson {
  margin: 0.35rem 0 0.2rem 0; padding: 0.65rem 0.8rem 0.7rem;
  background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
  border: 1px solid #E2E8F0; border-radius: 10px;
}
.fit-r2-lesson h3 {
  margin: 0 0 0.35rem 0; font-size: 1.05rem; font-weight: 750; color: #0F172A;
}
.fit-r2-lesson p {
  margin: 0 0 0.4rem 0; font-size: 0.9rem; color: #334155; line-height: 1.4rem;
}
.fit-r2-lesson .formula {
  margin: 0.25rem 0 0.45rem 0; padding: 0.4rem 0.55rem;
  background: #F1F5F9; border-radius: 7px; border: 1px solid #E2E8F0;
  font-size: 0.92rem; font-weight: 650; color: #1E3A8A;
  font-variant-numeric: tabular-nums; line-height: 1.4rem;
}
.fit-r2-lesson .formula b { color: #1D4ED8; font-weight: 780; }
.fit-r2-lesson .note {
  margin: 0.15rem 0 0 0; font-size: 0.82rem; color: #64748B; line-height: 1.3rem;
}
.fit-r2-lesson .metrics {
  display: flex; flex-wrap: wrap; gap: 0.55rem 1.1rem; margin: 0.2rem 0 0.35rem 0;
  font-size: 0.9rem; font-variant-numeric: tabular-nums; color: #0F172A;
}
.fit-r2-lesson .metrics b { font-weight: 780; }
.fit-r2-bar {
  display: flex; width: 100%; height: 1.4rem; border-radius: 7px;
  overflow: hidden; margin: 0.35rem 0 0.4rem 0; background: #E2E8F0;
}
.fit-r2-bar .explained {
  background: #3B82F6; color: white; font-size: 0.72rem; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
  font-variant-numeric: tabular-nums; min-width: 0;
}
.fit-r2-bar .unexplained {
  background: #94A3B8; color: white; font-size: 0.72rem; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
  font-variant-numeric: tabular-nums; min-width: 0;
}
.fit-r2-compare {
  display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.45rem;
  margin: 0.35rem 0 0.2rem 0;
}
@media (max-width: 900px) {
  .fit-r2-compare { grid-template-columns: 1fr; }
}
.fit-r2-card {
  padding: 0.45rem 0.55rem; border-radius: 8px; border: 1px solid #E2E8F0;
  background: #FFFFFF;
}
.fit-r2-card .lab {
  margin: 0 0 0.2rem 0; font-size: 0.7rem; font-weight: 750;
  letter-spacing: 0.05em; text-transform: uppercase; color: #64748B;
}
.fit-r2-card .val {
  margin: 0; font-size: 0.88rem; color: #0F172A; font-variant-numeric: tabular-nums;
  line-height: 1.35rem;
}
.fit-r2-card .val b { font-weight: 780; }
.fit-r2-card.best { border-color: #FDBA74; background: #FFF7ED; }
.fit-r2-conclusion {
  margin: 0.45rem 0 0.25rem 0; padding: 0.5rem 0.65rem;
  background: #FFF7ED; border: 1px solid #FDBA74; border-radius: 8px;
}
.fit-r2-conclusion .q {
  margin: 0 0 0.25rem 0; font-size: 0.88rem; font-weight: 780; color: #9A3412;
  line-height: 1.3rem;
}
.fit-r2-conclusion .a {
  margin: 0 0 0.3rem 0; font-size: 0.84rem; color: #334155; line-height: 1.35rem;
}
.fit-r2-conclusion .takeaway {
  margin: 0.3rem 0 0 0; font-size: 0.88rem; font-weight: 800; color: #9A3412;
  line-height: 1.35rem;
}
.fit-r2-warn {
  margin: 0.3rem 0 0 0; font-size: 0.84rem; color: #9A3412; line-height: 1.35rem;
}
.st-key-fit_r_panel { margin: 0; }
.st-key-fit_r_panel [data-testid="stHorizontalBlock"] {
  gap: 0.75rem !important; align-items: stretch !important;
}
.fit-r-col-lab {
  margin: 0 0 0.25rem 0; font-size: 0.7rem; font-weight: 780; letter-spacing: 0.06em;
  text-transform: uppercase; color: #475569;
}
.fit-r-line-note {
  margin: 0 0 0.35rem 0; padding: 0.3rem 0.45rem; border-radius: 6px;
  background: #F8FAFC; border: 1px solid #E2E8F0;
  font-size: 0.78rem; color: #334155; font-variant-numeric: tabular-nums; line-height: 1.25rem;
}
.fit-r-line-note .yours { color: #9A3412; font-weight: 650; }
.fit-r-line-note .ols { color: #1D4ED8; font-weight: 650; }
.fit-r-waiting {
  margin: 0; padding: 0.55rem 0.65rem; border-radius: 6px;
  background: #F3F4F6; color: #9AA5B1; font-size: 0.84rem; line-height: 1.3rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.st-key-fit_r_commands pre, .st-key-fit_r_console pre {
  font-size: 0.78rem !important; line-height: 1.25rem !important;
}
.fit-coef-lab {
  margin: 0 0 0.05rem 0; line-height: 1.2rem; white-space: nowrap;
}
.fit-coef-lab .sym {
  font-size: 1.2rem; font-weight: 780; color: #12263a; font-family: inherit;
}
.fit-coef-lab .desc {
  font-size: 0.72rem; font-weight: 500; color: #64748B; margin-left: 0.28rem;
}
.st-key-fit_alpha_wrap [data-testid="stSlider"] > div:first-child,
.st-key-fit_beta_wrap [data-testid="stSlider"] > div:first-child {
  margin-bottom: 0 !important;
}
.dist-intro { margin: 0 0 0.15rem 0; font-size: 1rem; font-weight: 700; color: #12263a; }
.dist-calc { font-size: 0.92rem; line-height: 1.4rem; color: #374151; }
.se-final { font-size: 1.65rem; font-weight: 780; color: #1D4ED8; }
.dist-note-line { margin: 0.2rem 0 0 0; color: #374151; font-size: 0.9rem; line-height: 1.3rem; }
.dist-formula {
    margin: 0.45rem 0 0.15rem 0;
    padding: 0.55rem 0.7rem;
    font-size: 1.15rem;
    font-weight: 750;
    color: #12263a;
    line-height: 1.5rem;
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-left: 4px solid #1D4ED8;
    border-radius: 6px;
    white-space: nowrap;
}
.dist-formula .se-key { font-weight: 780; }
.se-table { width: 100%; border-collapse: collapse; margin: 0.35rem 0 0.1rem 0; font-size: 0.92rem; }
.se-table caption { caption-side: top; text-align: left; font-weight: 750; color: #12263a; font-size: 1rem; padding-bottom: 0.25rem; }
.se-table th, .se-table td { padding: 0.22rem 0.75rem; text-align: right; border-bottom: 1px solid #e6eaf0; }
.se-table th:first-child, .se-table td:first-child { text-align: left; }
.se-table th { color: #52606D; font-weight: 650; font-size: 0.82rem; }
.se-table td:not(:first-child) { font-variant-numeric: tabular-nums; font-weight: 700; width: 9rem; }
.se-table tr.bench td { color: #EA580C; }
.se-table tr.ref td { color: #4B5563; }
.se-table tr.est td { color: #1D4ED8; }
.se-scale-row {
    display: grid;
    grid-template-columns: 11.2rem 1fr 3.4rem;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.86rem;
    line-height: 1.35rem;
}
.se-track {
    position: relative;
    height: 11px;
    background: #eef1f4;
    border-radius: 6px;
}
.se-track::before {
    content: "";
    position: absolute;
    left: 50%;
    top: -3px;
    bottom: -3px;
    width: 1px;
    background: #9ca3af;
}
.se-interval {
    position: absolute;
    top: 0;
    height: 100%;
    border-radius: 6px;
    left: 50%;
    transform: translateX(-50%);
}
.se-interval.bench { background: #EA580C; }
.se-interval.ref { background: #6B7280; }
.se-interval.est { background: #1D4ED8; }
@media (max-width: 980px) {
    .st-key-dist_layout {
        height: auto !important;
        max-height: none !important;
        overflow: visible;
    }
    .st-key-dist_scatter,
    .st-key-dist_beta {
        flex: none !important;
        height: 320px !important;
    }
    .st-key-one_scatter, .st-key-one_alpha, .st-key-one_beta { height: auto !important; }
}
</style>
"""


def format_number(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:,.{digits}f}"


def format_signed(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:+.{digits}f}"


def session_rng(seed: int | None) -> np.random.Generator:
    """Return the session generator, reseeding only when the seed changes.

    Chart toggles rerun the script. They must not restart the random stream,
    or a second sample would repeat the first.
    """
    if "rng" not in st.session_state or st.session_state.get("rng_seed_key") != seed:
        st.session_state.rng = np.random.default_rng(seed)
        st.session_state.rng_seed_key = seed
    return st.session_state.rng


def simulation_generator(seed: int | None, fallback: np.random.Generator) -> np.random.Generator:
    """Seeded simulations restart from that seed so the same settings repeat.

    Without a seed, the session generator is used and the stream moves on.
    """
    if seed is None:
        return fallback
    return np.random.default_rng(seed)


def init_results() -> None:
    st.session_state.setdefault("intro", IntroductionView())
    st.session_state.setdefault("intro_drawer", DrawerSnapshot())
    st.session_state.setdefault("intro_applied_chart_id", None)
    st.session_state.setdefault("intro_pop_signature", None)
    st.session_state.setdefault("intro_sample_token", None)
    st.session_state.setdefault("simulation", None)
    _ensure_app_data_drawer_state()


def _ensure_app_data_drawer_state() -> None:
    """One shared open flag for the global data drawer."""
    if "app_data_drawer_open" in st.session_state:
        return
    open_drawer = False
    drawer = st.session_state.get("intro_drawer")
    if drawer is not None:
        open_drawer = bool(getattr(drawer, "open", False))
    if st.session_state.pop("fit_data_drawer", False):
        open_drawer = True
    st.session_state.app_data_drawer_open = open_drawer


def _app_data_drawer_open() -> bool:
    _ensure_app_data_drawer_state()
    return bool(st.session_state.app_data_drawer_open)


def _set_app_data_drawer_open(open_drawer: bool) -> None:
    """Open or close the shared drawer without resetting lab state."""
    wanted = bool(open_drawer)
    st.session_state.app_data_drawer_open = wanted
    if "intro_drawer" in st.session_state:
        st.session_state.intro_drawer = set_drawer_open(_current_drawer(), wanted)


def _dismiss_dataset_drawer() -> None:
    """Dialog dismiss / close: hide the overlay only."""
    # Persist fitting drawer choices before dialog widgets are unmounted.
    _persist_fitting_drawer_widgets()
    _set_app_data_drawer_open(False)


def render_main_nav() -> str:
    """Main lab tabs. Load Dataset lives on each statistical tab's status row."""
    _ensure_app_data_drawer_state()
    if "main_lab_tab" not in st.session_state:
        st.session_state.main_lab_tab = "fitting"
    nav = st.container(
        key="main_nav",
        horizontal=True,
        vertical_alignment="center",
        gap="small",
        wrap=False,
    )
    with nav:
        active = st.segmented_control(
            "Lab section",
            options=[key for key, _ in MAIN_LAB_TABS],
            format_func=lambda key: MAIN_LAB_TAB_LABELS[key],
            key="main_lab_tab",
            required=True,
            label_visibility="collapsed",
            width="stretch",
            wrap=False,
        )
    if active == "intro":
        # Former standalone Sampling tab → Sampling & Inference.
        active = "sampling"
        st.session_state.main_lab_tab = active
    if active not in MAIN_LAB_TAB_LABELS:
        active = "fitting"
        st.session_state.main_lab_tab = active
    return str(active)


def _render_dataset_status_row(status_html: str) -> None:
    """Active dataset heading (left) and Load Dataset (right) on one row."""
    row = st.container(
        key="dataset_status_row",
        horizontal=True,
        vertical_alignment="center",
        gap="small",
        wrap=False,
    )
    with row:
        st.markdown(status_html, unsafe_allow_html=True)
        open_drawer = _app_data_drawer_open()
        if st.button(
            "Hide" if open_drawer else "Load Dataset",
            key="main_data_drawer_btn",
            type="secondary",
            help="Choose dataset and X/Y, then Load in Lab.",
        ):
            _set_app_data_drawer_open(not open_drawer)
            st.rerun()


def _render_lab_with_drawer(
    mode: str,
    population,
    body,
    *,
    rng: np.random.Generator | None = None,
) -> None:
    """Render a lab section; Dataset opens as an overlay (chart width unchanged)."""
    body()
    if mode == "intro":
        # Introduction opens the same Load Dataset overlay after highlights are ready.
        return
    if mode == "library":
        # Library is a full page — no load drawer overlay here.
        return
    if _app_data_drawer_open():
        _dataset_overlay_dialog(mode, population, rng=rng)


def render_intro() -> int | None:
    """Keep the population settings available, but out of the page.

    Students will later choose the population intercept, slope, and noise
    from this block. Until then the population stays fixed, so the controls
    are not shown.
    """
    with st.container(key="population_settings"):
        with st.expander("Advanced settings"):
            st.caption(
                "With a seed, repeating a simulation at the same settings "
                "reproduces the same estimates. Leave the seed off for a new "
                "random sequence. Changing a chart option does not restart the generator."
            )
            use_seed = st.checkbox("Use a random seed", value=False)
            seed_value = st.number_input(
                "Seed",
                min_value=0,
                max_value=999_999_999,
                value=42,
                step=1,
                disabled=not use_seed,
            )
    return int(seed_value) if use_seed else None


def progress_runner(label: str):
    """Return a slot that shows how many regressions have been fitted."""
    slot = st.empty()
    bar = slot.progress(0.0, text=label)

    def update(done: int, total: int) -> None:
        fraction = 0.0 if total == 0 else min(done / total, 1.0)
        bar.progress(fraction, text=f"{label}: {done:,} of {total:,}")

    def clear() -> None:
        slot.empty()

    return update, clear


def display_table(
    frame: pd.DataFrame,
    signed: set[str] | None = None,
    digits: dict[str, int] | None = None,
) -> None:
    """Show a table with coefficients rounded for reading."""
    signed_columns = signed or set()
    places = digits or {}
    shown = frame.copy()
    for column in shown.columns:
        if not pd.api.types.is_float_dtype(shown[column]):
            continue
        count = places.get(column, 3)
        if column in signed_columns:
            shown[column] = shown[column].map(
                lambda value, count=count: format_signed(float(value), count)
            )
        else:
            shown[column] = shown[column].map(
                lambda value, count=count: format_number(float(value), count)
            )
    st.dataframe(shown, hide_index=True, width="stretch")


def equation(alpha: float, beta: float) -> str:
    return f"exam score = {alpha:.3f} + {beta:.3f} × hours studied"


_ALPHA_HAT = "α\u0302"
_BETA_HAT = "β\u0302"


def _introduction_step(view: IntroductionView) -> str:
    if view.sample is None:
        return "The cloud is every student. Draw a sample to estimate a regression."
    if view.fit is not None and view.fit.degenerate:
        return "This sample has no variation in hours studied, so a line cannot be fit."
    if view.fit is None and view.population_revealed:
        return "New sample highlighted. The population line stays. Estimate this sample next."
    if view.fit is None:
        return "Blue points are a sample drawn without replacement. Estimate its regression."
    if not view.population_revealed:
        return "The blue line uses only this sample. Show the population regression to compare."
    return f"The orange line is the population regression. The blue line is this sample. {_ALPHA_HAT} − α and {_BETA_HAT} − β are sampling error."


def _signed_label(value: str) -> str:
    """Show an explicit plus or minus. A missing value stays a dash."""
    if value.startswith("-"):
        return "\u2212" + value[1:]
    return value


def _coef_cell(kind: str, symbol: str, value: str) -> str:
    return (
        f'<span class="cell {kind}">'
        f'<span class="sym">{symbol}</span>'
        f'<span class="eq">=</span>'
        f'<span class="num">{value}</span>'
        "</span>"
    )


def _diff_cell(value: str) -> str:
    return (
        '<span class="cell diff">'
        '<span class="sym"></span><span class="eq"></span>'
        f'<span class="num">{value}</span>'
        "</span>"
    )


def _information_strip(population, view: IntroductionView) -> str:
    """Fixed sample line plus a three-row coefficient grid.

    Every row is present before its numbers exist, so the block never
    grows when a line is estimated or the population is revealed.
    """
    if view.sample is None:
        sample = "—"
    else:
        fraction = format_sampling_fraction(view.sample.n, population.n)
        sample = f"{view.sample.n:,} ({fraction})"
    fitted = view.fit is not None and not view.fit.degenerate
    alpha_hat = format_number(view.fit.alpha) if fitted else "—"
    beta_hat = format_number(view.fit.beta) if fitted else "—"
    if view.population_revealed:
        alpha = format_number(population.alpha)
        beta = format_number(population.beta)
        if fitted:
            gap_alpha, gap_beta = displayed_coefficient_gap(
                view.fit, population.alpha, population.beta
            )
            alpha_error = _signed_label(format_signed(gap_alpha))
            beta_error = _signed_label(format_signed(gap_beta))
        else:
            alpha_error = "—"
            beta_error = "—"
    else:
        alpha = "—"
        beta = "—"
        alpha_error = "—"
        beta_error = "—"
    sample_line = (
        '<div class="intro-sample-line">'
        f'<span><span class="k">Population</span> <span class="v">{population.n:,}</span></span>'
        f'<span><span class="k">Sample</span> <span class="v">{sample}</span></span>'
        "</div>"
    )
    cells = [
        "<span></span>",
        '<span class="colhead">INTERCEPT</span>',
        '<span class="colhead">SLOPE</span>',
        '<span class="rowlabel strong">Sample estimate</span>',
        _coef_cell("est", _ALPHA_HAT, alpha_hat),
        _coef_cell("est", _BETA_HAT, beta_hat),
        '<span class="rowlabel pop-label">Population</span>',
        _coef_cell("pop", "α", alpha),
        _coef_cell("pop", "β", beta),
        '<span class="rowlabel">Difference</span>',
        _diff_cell(alpha_error),
        _diff_cell(beta_error),
    ]
    return (
        f'<div class="intro-stats">{sample_line}'
        f'<div class="intro-grid">{"".join(cells)}</div></div>'
    )


_SORT_COLUMNS = {
    "ID": "student_id",
    "Hours": "hours_studied",
    "Score": "exam_score",
}


def _stored_population_controls() -> tuple[str, str, bool, int]:
    query = str(st.session_state.get("intro_pop_query", ""))
    sort_label = str(st.session_state.get("intro_pop_sort", "ID"))
    if sort_label not in _SORT_COLUMNS:
        sort_label = "ID"
    order = str(st.session_state.get("intro_pop_order", "Asc"))
    page = st.session_state.get("intro_pop_page", 1)
    try:
        page_number = int(page)
    except (TypeError, ValueError):
        page_number = 1
    return query, _SORT_COLUMNS[sort_label], order == "Asc", max(page_number, 1)


def _show_chart_student(population, view, tab: str, student_id: int) -> str:
    """Open the table that contains this student and move to their page."""
    in_sample = view.sample is not None and bool(np.any(view.sample.student_id == int(student_id)))
    dataset = tab if tab in ("Population", "Sample") else "Population"
    if dataset == "Sample" and not in_sample:
        dataset = "Population"
    if dataset == "Population":
        query, sort_by, ascending, _page = _stored_population_controls()
        query = query.strip()
        if not (query.isdigit() and int(query) == int(student_id)):
            if query:
                st.session_state.intro_pop_query = ""
            page = page_for_student(population, int(student_id), sort_by, ascending)
            if page is not None:
                st.session_state.intro_pop_page = page + 1
    return dataset


def _sync_intro_selection(population, view) -> None:
    """Apply a chart click. A table click is handled by the table itself."""
    drawer: DrawerSnapshot = st.session_state.intro_drawer
    tab = st.session_state.get("intro_drawer_tab", drawer.dataset)
    if tab not in ("Population", "Sample"):
        tab = drawer.dataset

    chart_id = chart_student_id(st.session_state.get("intro_chart"))
    chart_changed = chart_id is not None and chart_id != st.session_state.intro_applied_chart_id
    if chart_changed:
        drawer = state_after_chart_click(drawer, chart_id)
        st.session_state.intro_applied_chart_id = chart_id
        tab = _show_chart_student(population, view, tab, int(chart_id))
        st.session_state.intro_drawer_tab = tab
        st.session_state.intro_scroll_to = int(chart_id)
        drawer = DrawerSnapshot(True, tab, drawer.selected_ids)
        st.session_state.app_data_drawer_open = True
    elif tab != drawer.dataset:
        drawer = DrawerSnapshot(_app_data_drawer_open(), tab, drawer.selected_ids)

    st.session_state.intro_drawer = drawer


def _mount_student_table(frame: pd.DataFrame, key: str, selected_ids: tuple[int, ...], token: str) -> None:
    """Show ID, Hours, and Score. Clicking a row toggles that student."""
    rows = [
        {
            "id": int(row.student_id),
            "hours": f"{float(row.hours_studied):.2f}",
            "score": f"{float(row.exam_score):.1f}",
        }
        for row in frame.itertuples(index=False)
    ]
    focus = st.session_state.pop("intro_scroll_to", None)
    result = student_table()(
        data={
            "rows": rows,
            "selected": [int(student) for student in selected_ids],
            "focus": None if focus is None else int(focus),
            "token": token,
        },
        key=key,
        width="stretch",
        height="stretch",
        on_pick_change=lambda: None,
    )
    picked = result.get("pick") if result is not None else None
    if isinstance(picked, (int, float)) and not isinstance(picked, bool):
        drawer: DrawerSnapshot = st.session_state.intro_drawer
        st.session_state.intro_drawer = DrawerSnapshot(
            drawer.open,
            drawer.dataset,
            toggle_student(drawer.selected_ids, int(picked)),
        )
        st.session_state.intro_applied_chart_id = chart_student_id(
            st.session_state.get("intro_chart")
        )
        st.rerun()


def _student_line(spec, count: int) -> str:
    text = (
        f"Student {spec.student_id} · {spec.hours_studied:.2f} h · "
        f"score {spec.exam_score:.1f}"
    )
    if count > 1:
        text = f"{count} selected · {text}"
    if spec.show_sample_residual:
        text += f" · predicted {spec.predicted:.1f} · residual {spec.residual:+.1f}"
    elif spec.predicted is not None and not spec.in_sample:
        text += f" · not in the sample; the line would predict {spec.predicted:.1f}"
    elif spec.in_sample:
        text += " · estimate the regression to see the residual"
    return text


def _dataset_status_name(catalog) -> str:
    """Compact label for the read-only status line."""
    if catalog.key == "students":
        return "Students"
    return catalog.short_name


def _dataset_status_html(name: str, actual_n: int, available_n: int, x_label: str, y_label: str) -> str:
    return (
        f'<p class="dataset-status">{name} · n = {actual_n:,} / {available_n:,} · '
        f"{x_label} → {y_label}</p>"
    )


def _fitting_dataset_status_html() -> str:
    ensure_registry_state(st.session_state)
    dataset_key = st.session_state.get("fit_dataset_key", "students")
    catalog = get_catalog(dataset_key, st.session_state)
    x_name, y_name = _ensure_fitting_xy(dataset_key)
    source = resolve_fitting_dataset(
        dataset_key,
        x_name=x_name,
        y_name=y_name,
        session=st.session_state,
        seed=st.session_state.get("fit_seed", 20261018),
    )
    _requested, actual_n, _capped, _pct = _fitting_sample_plan(source)
    name = _dataset_status_name(catalog)
    # Mark observed lab datasets; Students needs no “Synthetic” qualifier.
    if not is_synthetic_dataset(dataset_key):
        name = f"{name} · Observed"
    return _dataset_status_html(
        name,
        int(actual_n),
        int(source.available_n),
        catalog.variable_label(x_name),
        catalog.variable_label(y_name),
    )


def _intro_dataset_status_html(population, view) -> str:
    actual_n = 0 if view.sample is None else int(view.sample.n)
    return _dataset_status_html(
        "Students",
        actual_n,
        int(population.n),
        "Hours studied",
        "Exam score",
    )


def _request_fitting_new_sample() -> None:
    """Redraw the fitting sample while preserving the candidate line when possible."""
    st.session_state.fit_draw = int(st.session_state.get("fit_draw", 0)) + 1
    st.session_state.fit_found = False
    st.session_state.fit_preserve_hats = True
    st.session_state.pop("fit_view", None)
    st.session_state.pop("fit_view_token", None)
    st.session_state.pop("fit_sse_before", None)
    _clear_fitting_prediction()
    _clear_fitting_animation()
    _clear_fitting_metric_paths()
    _clear_fitting_student_selection()


@st.dialog(
    "Dataset",
    position="right",
    width="medium",
    on_dismiss=_dismiss_dataset_drawer,
)
def _dataset_overlay_dialog(
    mode: str,
    population,
    rng: np.random.Generator | None = None,
    intro_view=None,
    intro_highlights=None,
) -> None:
    """Right-side overlay Dataset drawer — does not resize the page or chart.

    Title and close control come from ``@st.dialog``; do not duplicate them here.
    """
    _render_dataset_drawer_body(
        mode,
        population=population,
        rng=rng,
        intro_view=intro_view,
        intro_highlights=intro_highlights,
    )


def _render_dataset_drawer_body(
    mode: str,
    *,
    population,
    intro_view=None,
    intro_highlights=None,
    rng: np.random.Generator | None = None,
) -> None:
    """Shared Dataset drawer: Load Dataset only (same on every lab tab)."""
    del mode, population, intro_view, intro_highlights, rng
    with st.container(gap="small", key="app_data_drawer"):
        _render_load_dataset_panel()


def _apply_intro_new_sample(
    population,
    view: IntroductionView,
    rng: np.random.Generator,
) -> None:
    """Draw a sample and clear the sample regression so Estimate is available again."""
    sample_size = int(st.session_state.get("intro_sample_size", 100))
    drawn = select_sample(view, population, sample_size, rng)
    st.session_state.intro = IntroductionView(
        sample=drawn.sample,
        fit=None,
        population_revealed=False,
    )
    drawer = st.session_state.intro_drawer
    st.session_state.intro_drawer = DrawerSnapshot(
        _app_data_drawer_open(),
        drawer.dataset,
        selection_after_new_sample(drawer.selected_ids, drawn.sample),
    )
    st.session_state.intro_applied_chart_id = chart_student_id(
        st.session_state.get("intro_chart")
    )


def _fitting_persist_bag() -> dict:
    """Non-widget store for fitting dataset choices (survives ``@st.dialog`` close)."""
    bag = st.session_state.get("_fit_persist")
    if not isinstance(bag, dict):
        bag = {}
        st.session_state._fit_persist = bag
    return bag


def _save_fitting_persist() -> None:
    """Write canonical fitting dataset fields into the non-widget persist bag."""
    bag = _fitting_persist_bag()
    bag["dataset"] = st.session_state.get("fit_dataset_key", "students")
    bag["pct"] = float(st.session_state.get("fit_pct", DEFAULT_SAMPLE_PCT))
    bag["use_all"] = bool(st.session_state.get("fit_use_all", False))
    if "fit_x" in st.session_state:
        bag["x"] = st.session_state.fit_x
    if "fit_y" in st.session_state:
        bag["y"] = st.session_state.fit_y


def _restore_fitting_persist(*, sample_controls: bool = True) -> None:
    """Restore fitting fields from the non-widget persist bag.

    ``fit_pct`` / ``fit_use_all`` are lab widgets. Never assign them after those
    widgets have been instantiated (e.g. when the Dataset drawer opens later
    in the same run) — that raises StreamlitWidgetAlreadyInstantiatedError.
    """
    bag = st.session_state.get("_fit_persist")
    if not isinstance(bag, dict) or not bag:
        return
    if "dataset" in bag:
        st.session_state.fit_dataset_key = bag["dataset"]
    if "x" in bag:
        st.session_state.fit_x = bag["x"]
    if "y" in bag:
        st.session_state.fit_y = bag["y"]
    if not sample_controls:
        return
    # Only seed sample widgets before they exist; never overwrite afterward.
    if "pct" in bag and "fit_pct" not in st.session_state:
        st.session_state.fit_pct = float(bag["pct"])
    if "use_all" in bag and "fit_use_all" not in st.session_state:
        st.session_state.fit_use_all = bool(bag["use_all"])


def _seed_fitting_drawer_widgets() -> None:
    """Restore dataset / X/Y for the drawer; do not touch lab sample widgets."""
    _restore_fitting_persist(sample_controls=False)


def _persist_fitting_drawer_widgets() -> None:
    """Persist fitting dataset / sample choices (dialog dismiss safe)."""
    _save_fitting_persist()


def _dm_active_summary() -> str:
    """Short label for the dataset currently driving the SLR tab."""
    key = st.session_state.get("fit_dataset_key", "students")
    try:
        catalog = get_catalog(key, st.session_state)
        x_name, y_name = _ensure_fitting_xy(key)
        return (
            f"{catalog.short_name} · "
            f"{catalog.variable_label(x_name)} → {catalog.variable_label(y_name)}"
        )
    except KeyError:
        return "None"


def _render_dm_active_banner() -> None:
    st.markdown(
        f'<div class="dm-active-line"><b>Active in lab</b> · {_dm_active_summary()}</div>',
        unsafe_allow_html=True,
    )


_LOAD_SOURCES = ("Examples", "R datasets", "My datasets")


def _on_load_dataset_in_lab() -> None:
    """Activate draft X/Y before widgets remount (avoids session-state write errors)."""
    chosen = st.session_state.get("dm_load_select")
    x_choice = st.session_state.get("dm_draft_x_select")
    y_choice = st.session_state.get("dm_draft_y_select")
    st.session_state.pop("dm_load_error", None)
    if not chosen or not x_choice or not y_choice:
        st.session_state.dm_load_error = "Choose a dataset and X/Y first."
        return
    try:
        activate_dataset(
            st.session_state, key=chosen, x_name=x_choice, y_name=y_choice
        )
    except (ValueError, KeyError) as exc:
        st.session_state.dm_load_error = str(exc)
        return
    st.session_state.fit_pct = DEFAULT_SAMPLE_PCT
    st.session_state.fit_use_all = False
    st.session_state.dm_load_source_for = chosen
    try:
        st.session_state.dm_load_source = source_group_label(chosen)
    except KeyError:
        st.session_state.dm_load_source = "Examples"
    _reset_fitting_sample_state()
    _save_fitting_persist()


def _render_load_dataset_panel() -> None:
    """Compact Load Dataset drawer: active, source + dataset, X/Y, Load in Lab."""
    ensure_registry_state(st.session_state)
    _seed_fitting_drawer_widgets()
    _render_dm_active_banner()

    active_key = st.session_state.get("fit_dataset_key", "students")
    try:
        default_source = source_group_label(active_key)
    except KeyError:
        default_source = "Examples"
    if default_source not in _LOAD_SOURCES:
        default_source = "Examples"

    if "dm_load_source" not in st.session_state:
        st.session_state.dm_load_source = default_source
    elif st.session_state.dm_load_source not in _LOAD_SOURCES:
        st.session_state.dm_load_source = default_source

    # Seed source from the active lab dataset on first open / after Load in Lab.
    if st.session_state.get("dm_load_source_for") != active_key:
        st.session_state.dm_load_source_for = active_key
        st.session_state.dm_load_source = default_source

    source = st.selectbox(
        "Source",
        options=list(_LOAD_SOURCES),
        key="dm_load_source",
    )

    options = library_options(st.session_state, group=source, query="")
    if not options:
        if source == "My datasets":
            st.caption("No personal datasets yet. Import one from Dataset Library.")
        else:
            st.caption("No datasets in this source.")
        return

    labels = dict(options)
    keys = [key for key, _ in options]
    draft = st.session_state.get("dm_draft_key") or active_key
    if draft not in keys:
        # Prefer active dataset when switching back to its source; else first item.
        draft = active_key if active_key in keys else keys[0]
    if (
        "dm_load_select" not in st.session_state
        or st.session_state.dm_load_select not in keys
        or st.session_state.get("dm_load_select_source") != source
    ):
        st.session_state.dm_load_select = draft
        st.session_state.dm_load_select_source = source

    chosen = st.selectbox(
        "Dataset",
        options=keys,
        format_func=lambda key: labels[key],
        key="dm_load_select",
    )
    st.session_state.dm_draft_key = chosen
    st.session_state.dm_load_select_source = source

    try:
        catalog = get_catalog(chosen, st.session_state)
        frame = get_frame(chosen, st.session_state)
    except KeyError:
        st.error("That dataset is no longer available.")
        return

    numeric_names = list(catalog.variable_names())
    is_active = chosen == active_key
    if not is_active:
        st.markdown(
            f'<div class="dm-selected-line"><b>Selected</b> · {catalog.short_name} '
            f"· not yet active</div>",
            unsafe_allow_html=True,
        )

    if len(numeric_names) < 2:
        st.warning("Need at least two numeric variables for simple linear regression.")
        return

    prev_key = st.session_state.get("dm_load_xy_for")
    if prev_key != chosen:
        st.session_state.dm_load_xy_for = chosen
        st.session_state.dm_draft_x_select = catalog.default_x
        st.session_state.dm_draft_y_select = catalog.default_y
        if is_active:
            if st.session_state.get("fit_x") in numeric_names:
                st.session_state.dm_draft_x_select = st.session_state.fit_x
            if st.session_state.get("fit_y") in numeric_names:
                st.session_state.dm_draft_y_select = st.session_state.fit_y

    if (
        "dm_draft_x_select" not in st.session_state
        or st.session_state.dm_draft_x_select not in numeric_names
    ):
        st.session_state.dm_draft_x_select = catalog.default_x
    x_choice = st.selectbox(
        "Predictor (X)",
        options=numeric_names,
        key="dm_draft_x_select",
    )
    st.session_state.dm_draft_x = x_choice
    y_opts = [name for name in numeric_names if name != x_choice]
    if (
        "dm_draft_y_select" not in st.session_state
        or st.session_state.dm_draft_y_select not in y_opts
    ):
        preferred = catalog.default_y if catalog.default_y in y_opts else y_opts[0]
        st.session_state.dm_draft_y_select = preferred
    y_choice = st.selectbox(
        "Outcome (Y)",
        options=y_opts,
        key="dm_draft_y_select",
    )
    st.session_state.dm_draft_y = y_choice
    valid_n = pair_complete_count(frame, x_choice, y_choice)
    st.caption(f"Valid observations: **{valid_n:,}**")

    same_xy = (
        is_active
        and st.session_state.get("fit_x") == x_choice
        and st.session_state.get("fit_y") == y_choice
    )
    if same_xy:
        st.button("In use", key="dm_load_in_lab", disabled=True)
    else:
        st.button(
            "Load in Lab",
            type="primary",
            key="dm_load_in_lab",
            on_click=_on_load_dataset_in_lab,
        )
    load_error = st.session_state.pop("dm_load_error", None)
    if load_error:
        st.error(load_error)


def render_dataset_library_tab() -> None:
    """Top-level Dataset Library: browse, import, inspect, export (no activation)."""
    ensure_registry_state(st.session_state)
    st.markdown("**Dataset Library**")
    st.caption(
        "Browse and import datasets here. Open the Dataset drawer on any lab tab "
        "to choose X/Y and Load in Lab."
    )

    group = st.segmented_control(
        "Library source",
        options=["Examples", "R datasets", "My datasets"],
        key="dm_lib_source",
        label_visibility="collapsed",
        width="stretch",
    )
    if group not in ("Examples", "R datasets", "My datasets"):
        group = "Examples"

    st.text_input(
        "Search",
        key="dm_lib_query",
        placeholder="Name or description…",
    )
    query = st.session_state.get("dm_lib_query", "")

    if group == "My datasets":
        _render_library_upload_section()

    opts = library_options(st.session_state, group=group, query=query)
    if not opts:
        if group == "My datasets":
            st.caption("No personal datasets yet. Upload a CSV or Excel file above.")
        else:
            st.caption("No datasets match that search.")
        return

    labels = dict(opts)
    keys = [key for key, _ in opts]
    focus = st.session_state.get("dm_lib_key")
    if focus not in keys:
        focus = keys[0]
        st.session_state.dm_lib_key = focus
    if "dm_lib_select" not in st.session_state or st.session_state.dm_lib_select not in keys:
        st.session_state.dm_lib_select = focus

    chosen = st.selectbox(
        "Browse",
        options=keys,
        format_func=lambda key: labels[key],
        key="dm_lib_select",
    )
    st.session_state.dm_lib_key = chosen

    try:
        catalog = get_catalog(chosen, st.session_state)
        frame = get_frame(chosen, st.session_state)
    except KeyError:
        st.error("Dataset unavailable.")
        return

    summary = build_preview(
        name=catalog.short_name,
        description=catalog.source + (f" {catalog.caveat}" if catalog.caveat else ""),
        frame=frame,
    )
    kind_bit = "" if is_synthetic_dataset(chosen) else " · Observed"
    st.markdown(
        f'<div class="dm-meta"><b>{summary.name}</b>{kind_bit}<br/>'
        f"{summary.description}</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"{summary.n_rows:,} observations · {summary.n_cols} variables · "
        f"numeric: {', '.join(catalog.variable_names())}"
    )
    with st.expander("Preview", expanded=True, key=f"dm_lib_preview_{chosen}"):
        st.dataframe(
            summary.preview,
            width="stretch",
            hide_index=True,
            height=min(280, 36 + 22 * max(len(summary.preview), 1)),
            key=f"dm_lib_preview_table_{chosen}",
        )
    st.download_button(
        "Export CSV",
        data=download_csv_bytes(chosen, st.session_state),
        file_name=f"{catalog.short_name}.csv",
        mime="text/csv",
        key=f"dm_lib_export_{chosen}",
    )


def _render_library_upload_section() -> None:
    """CSV/Excel import into the session library (does not activate in lab)."""
    uploaded = st.file_uploader(
        "Import CSV or Excel",
        type=["csv", "tsv", "txt", "xlsx"],
        key="dm_lib_uploader",
        help="Adds the table to My datasets. Use the Dataset drawer to Load in Lab.",
    )
    if uploaded is None:
        return
    raw = uploaded.getvalue()
    sheet = None
    name_lower = uploaded.name.lower()
    if name_lower.endswith((".xlsx", ".xlsm", ".xls")):
        sheets = read_excel_sheets(raw)
        if "dm_lib_excel_sheet" not in st.session_state or st.session_state.dm_lib_excel_sheet not in sheets:
            st.session_state.dm_lib_excel_sheet = sheets[0]
        sheet = st.selectbox(
            "Worksheet",
            options=sheets,
            key="dm_lib_excel_sheet",
        )
    try:
        parsed = parse_uploaded_table(
            filename=uploaded.name, data=raw, sheet_name=sheet
        )
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        return
    st.caption(
        f"Ready to import **{parsed.name}** · "
        f"{len(parsed.frame):,} rows × {parsed.frame.shape[1]} cols"
    )
    if st.button("Add to library", type="primary", key="dm_lib_register"):
        try:
            key = register_upload(
                st.session_state,
                name=parsed.name,
                frame=parsed.frame,
                source=parsed.source,
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        st.session_state.dm_lib_key = key
        st.session_state.dm_lib_select = key
        st.success(
            f"Added “{parsed.name}” to My datasets. "
            "Open the Dataset drawer to Load in Lab."
        )
        st.rerun()


def _render_fitting_lab_sample_controls(source) -> None:
    """Sample size / redraw controls on the SLR lab (not in the Dataset drawer)."""
    available_for_help = int(source.available_n)
    use_all = bool(st.session_state.get("fit_use_all", False))
    row = st.container(horizontal=True, vertical_alignment="bottom", gap="small", wrap=True)
    with row:
        st.slider(
            "Sample size (%)",
            min_value=MIN_SAMPLE_PCT,
            max_value=MAX_SAMPLE_PCT,
            step=1,
            key="fit_pct",
            format="%d%%",
            disabled=use_all,
            help="Percent of available complete records to draw.",
            width=220,
        )
        st.checkbox(
            "All observations",
            key="fit_use_all",
            help=(
                "Uses every complete record when N is modest. "
                f"For very large datasets the interactive view is capped at "
                f"{interactive_n_cap(available_for_help):,}."
                if available_for_help > interactive_n_cap(available_for_help)
                else "Use every complete record for this X/Y pair."
            ),
        )
        if st.button("Draw New Sample", type="primary", key="fit_draw_btn"):
            _save_fitting_persist()
            _request_fitting_new_sample()
            st.rerun()
    _save_fitting_persist()
    _requested_n, actual_n, capped, pct = _fitting_sample_plan(source)
    pct_label = "all" if st.session_state.get("fit_use_all") else f"{pct:.0f}%"
    note = f" · capped at n = {actual_n:,}" if capped and not st.session_state.get("fit_use_all") else ""
    st.caption(
        f"Sample {pct_label} · n = {actual_n:,} of {available_for_help:,}{note}"
    )


def _render_sampling_lab_data_panel(population) -> None:
    """Show the latest single sample from the sampling-distributions lab when available."""
    experiment = st.session_state.get("dist_experiment")
    one = st.session_state.get("dist_one_sample")
    sample_size = getattr(one, "sample_size", None) or getattr(experiment, "sample_size", None)
    x_values = getattr(one, "x", None)
    y_values = getattr(one, "y", None)
    if x_values is None and experiment is not None:
        x_values = getattr(experiment, "sample_x", None)
        y_values = getattr(experiment, "sample_y", None)
        sample_size = getattr(experiment, "sample_size", sample_size)
    if x_values is None or y_values is None:
        st.caption(
            "Draw a sample in Sampling & Inference to inspect its observations. "
            f"The lab samples from the fixed population of {population.n:,} students."
        )
        return
    x_arr = np.asarray(x_values, dtype=np.float64)
    y_arr = np.asarray(y_values, dtype=np.float64)
    n = int(min(x_arr.size, y_arr.size))
    st.caption(
        f"Latest sample · n = {sample_size or n:,} from {population.n:,} students."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Row": np.arange(1, n + 1, dtype=int),
                "hours_studied": x_arr[:n],
                "exam_score": y_arr[:n],
            }
        ),
        width="stretch",
        hide_index=True,
        height=min(360, 36 + 26 * max(n, 1)),
        key="sampling_tab_data_table",
    )


def _render_population_table(population, selected_ids: tuple[int, ...]) -> None:
    query = st.text_input(
        "Search student ID",
        key="intro_pop_query",
        placeholder="Search student ID",
        label_visibility="collapsed",
    )
    query_text, sort_by, ascending, page_number = _stored_population_controls()
    frame, match_count, page_index = population_page(
        population,
        student_query=query_text,
        sort_by=sort_by,
        ascending=ascending,
        page=page_number - 1,
    )
    controls = st.container(horizontal=True, vertical_alignment="center", gap="xsmall", wrap=False)
    with controls:
        st.selectbox(
            "Sort by",
            list(_SORT_COLUMNS),
            key="intro_pop_sort",
            label_visibility="collapsed",
            width=88,
        )
        st.selectbox(
            "Order",
            ["Asc", "Desc"],
            key="intro_pop_order",
            label_visibility="collapsed",
            width=76,
        )
    if match_count == 0:
        st.caption("No student with that ID.")
    else:
        start = page_index * PAGE_SIZE + 1
        end = start + len(frame) - 1
        page_count = max(1, (match_count + PAGE_SIZE - 1) // PAGE_SIZE)
        show_pages = not query.strip() and page_count > 1
        status = st.container(
            horizontal=True,
            vertical_alignment="center",
            gap="xxsmall",
            key="intro_page_line",
        )
        with status:
            st.caption(f"Showing {start:,}–{end:,} of {match_count:,}.")
            if show_pages:
                previous = st.button(
                    "‹",
                    key="intro_pop_prev",
                    type="tertiary",
                    disabled=page_index <= 0,
                    width=28,
                )
                following = st.button(
                    "›",
                    key="intro_pop_next",
                    type="tertiary",
                    disabled=page_index >= page_count - 1,
                    width=28,
                )
        if show_pages and previous:
            st.session_state.intro_pop_page = page_index
            st.rerun()
        if show_pages and following:
            st.session_state.intro_pop_page = page_index + 2
            st.rerun()
        _mount_student_table(
            frame,
            "intro_population_rows",
            selected_ids,
            f"pop:{query_text}:{sort_by}:{ascending}:{page_index}",
        )
    st.download_button(
        "Download CSV",
        data=population_csv(population),
        file_name="population.csv",
        mime="text/csv",
        key="intro_download_population",
        width="content",
    )


def _render_sample_table(population, view, selected_ids: tuple[int, ...]) -> None:
    if view.sample is None:
        st.caption(EMPTY_SAMPLE)
        return
    st.caption(sample_count_label(view.sample.n, population.n))
    _mount_student_table(
        sample_frame(view.sample),
        "intro_sample_rows",
        selected_ids,
        f"sample:{sample_token(view.sample)}",
    )
    footer = st.container(horizontal=True, vertical_alignment="center", gap="small")
    with footer:
        st.download_button(
            "Download CSV",
            data=sample_csv(view.sample),
            file_name="sample.csv",
            mime="text/csv",
            key="intro_download_sample",
            width="content",
        )
        with st.popover("R commands"):
            st.code(R_COMMANDS, language="r")


def _line_legend() -> str:
    """Fixed caption for the two lines. It does not change height when a line appears."""
    return (
        '<div class="line-legend">'
        '<span><span class="swatch pop"></span>Population regression (true)</span>'
        '<span><span class="swatch est"></span>Sample regression (estimated)</span>'
        "</div>"
    )


def _chart_header() -> None:
    st.markdown(
        '<p class="chart-title">Hours studied and exam score</p>',
        unsafe_allow_html=True,
    )


def _current_drawer() -> DrawerSnapshot:
    """Read the drawer, including a session saved before multi-select."""
    drawer = st.session_state.intro_drawer
    selected = getattr(drawer, "selected_ids", None)
    if isinstance(selected, tuple):
        return drawer
    previous = getattr(drawer, "selected_id", None)
    migrated = DrawerSnapshot(
        bool(getattr(drawer, "open", False)),
        str(getattr(drawer, "dataset", "Population") or "Population"),
        () if previous is None else (int(previous),),
    )
    st.session_state.intro_drawer = migrated
    return migrated


def render_introduction_tab(population, rng: np.random.Generator) -> None:
    view: IntroductionView = st.session_state.intro
    drawer: DrawerSnapshot = _current_drawer()
    drawer_open = _app_data_drawer_open()
    fitted = view.fit is not None
    revealed = view.population_revealed
    has_sample = view.sample is not None
    generator = rng if rng is not None else st.session_state.get("rng")
    with st.container(key="intro_layout", gap="xxsmall"):
        with st.container(key="intro_header", gap="xxsmall"):
            _render_dataset_status_row(_intro_dataset_status_html(population, view))
            toolbar = st.container(
                key="intro_toolbar",
                horizontal=True,
                vertical_alignment="bottom",
                gap="small",
                wrap=False,
            )
            with toolbar:
                st.select_slider(
                    "Sample size",
                    options=SAMPLE_SIZES,
                    value=int(st.session_state.get("intro_sample_size", 100)),
                    key="intro_sample_size",
                )
                draw_clicked = st.button(
                    "Draw New Sample",
                    type="primary",
                    key="intro_draw",
                    disabled=generator is None,
                    wrap=False,
                )
                estimate_clicked = st.button(
                    "Estimate regression",
                    type="primary" if has_sample and not fitted else "secondary",
                    key="intro_estimate",
                    disabled=not (has_sample and not fitted),
                    wrap=False,
                )
                show_clicked = st.button(
                    "Hide population regression" if revealed else "Show population regression",
                    type="primary" if fitted and not revealed else "secondary",
                    key="intro_show",
                    disabled=not fitted,
                    wrap=False,
                )
            st.markdown(_information_strip(population, view), unsafe_allow_html=True)

        if draw_clicked and generator is not None:
            _apply_intro_new_sample(population, view, generator)
            st.rerun()
        if estimate_clicked and view.sample is not None and view.fit is None:
            st.session_state.intro = estimate_sample(view)
            st.rerun()
        if show_clicked and view.fit is not None:
            st.session_state.intro = (
                hide_population(view) if view.population_revealed else reveal_population(view)
            )
            st.rerun()

        view = st.session_state.intro
        _sync_intro_selection(population, view)
        view = st.session_state.intro
        drawer = st.session_state.intro_drawer
        drawer_open = _app_data_drawer_open()

        sample_alpha = None
        sample_beta = None
        show_sample_line = False
        if view.fit is not None and not view.fit.degenerate:
            sample_alpha = view.fit.alpha
            sample_beta = view.fit.beta
            show_sample_line = True

        highlights = highlights_for(population, view.sample, view.fit, drawer.selected_ids)
        focused = highlights[-1] if highlights else None
        cloud = population_cloud_image(
            population.hours_studied,
            population.exam_score,
            population.x_range,
            population.y_range,
            cache_key=population.sha256,
        )
        figure = introduction_figure(
            cloud_image=cloud,
            x_range=population.x_range,
            y_range=population.y_range,
            sample_x=None if view.sample is None else view.sample.hours_studied,
            sample_y=None if view.sample is None else view.sample.exam_score,
            sample_ids=None if view.sample is None else view.sample.student_id,
            sample_alpha=sample_alpha,
            sample_beta=sample_beta,
            population_alpha=population.alpha,
            population_beta=population.beta,
            show_population_line=view.population_revealed,
            show_sample_line=show_sample_line,
            show_title=False,
            selected_ids=None if not highlights else np.array([item.student_id for item in highlights]),
            selected_xs=None if not highlights else np.array([item.hours_studied for item in highlights]),
            selected_ys=None if not highlights else np.array([item.exam_score for item in highlights]),
            selected_membership=None if not highlights else np.array([item.in_sample for item in highlights]),
            show_residual=bool(focused and focused.show_sample_residual),
        )
        figure.update_layout(autosize=True)
        figure.layout.height = None

        with st.container(key="intro_workspace"):
            _chart_header()
            st.markdown(_line_legend(), unsafe_allow_html=True)
            with st.container(key="intro_chart_frame"):
                st.plotly_chart(
                    figure,
                    width="stretch",
                    height="stretch",
                    config=PLOTLY_CONFIG,
                    on_select="rerun",
                    selection_mode="points",
                    key="intro_chart",
                )
            st.caption(_introduction_step(view))
        if drawer_open:
            _dataset_overlay_dialog(
                "intro",
                population,
                rng=rng,
                intro_view=view,
                intro_highlights=highlights,
            )
        slider_n = int(st.session_state.get("intro_sample_size", 100))
        if view.sample is not None and view.sample.n != slider_n:
            st.caption(
                f"The slider is {slider_n:,}. The sample on the chart "
                f"has {view.sample.n:,} students."
            )
        with st.expander("Population & sample tables", expanded=False):
            if "intro_drawer_tab" not in st.session_state:
                st.session_state.intro_drawer_tab = "Population"
            table_tab = st.segmented_control(
                "Table",
                ["Population", "Sample"],
                key="intro_drawer_tab",
                label_visibility="collapsed",
                required=True,
                width="stretch",
            )
            if table_tab not in ("Population", "Sample"):
                table_tab = "Population"
            focused = highlights[-1] if highlights else None
            if focused is None:
                line = "Select a student in the table or on the chart."
            else:
                line = _student_line(focused, len(highlights))
            st.markdown(f'<p class="drawer-student">{line}</p>', unsafe_allow_html=True)
            selected_ids = tuple(spec.student_id for spec in highlights)
            if table_tab == "Population":
                _render_population_table(population, selected_ids)
            else:
                _render_sample_table(population, view, selected_ids)
        with st.expander("How this works"):
            st.markdown(
                f"**{_ALPHA_HAT}** and **{_BETA_HAT}** are the intercept and slope "
                "fitted on the current sample. **α** and **β** are the population "
                "coefficients, fitted on all 100,000 students. "
                f"**{_ALPHA_HAT} − α** and **{_BETA_HAT} − β** are the sample estimate "
                "minus the population coefficient. They fill in when the population "
                "line is revealed."
            )
            st.caption(population_cloud_caption(population.n))
            st.caption(DRAWER_INTRO)
            st.caption(SELECTION_NOTE)


def _dist_pause(added: int, gaps: int) -> float:
    if gaps <= 0:
        return 0.0
    if added <= 10:
        return 0.12
    if added <= 100:
        return 0.07
    return 2.2 / gaps


def _dist_number(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.3f}"


def _dist_status(experiment) -> str:
    """Running total of samples drawn, with the latest slope estimate beside it."""
    latest_beta = None if experiment.n_repetitions == 0 else float(experiment.betas[-1])
    return (
        '<p class="dist-meta">'
        f'<span class="dist-count">Samples drawn: {experiment.n_repetitions:,}</span>'
        f"<span>Latest sample: {_BETA_HAT} {_dist_number(latest_beta)}</span>"
        "</p>"
    )


def _dist_experiment(population):
    experiment = st.session_state.get("dist_experiment")
    if experiment is None:
        experiment = new_experiment(
            100,
            population.hours_studied,
            population.exam_score,
            population.alpha,
            population.beta,
        )
        st.session_state.dist_experiment = experiment
    return experiment


def _distribution_scatter_figure(population, experiment, cloud: str):
    latest_alpha = None if experiment.n_repetitions == 0 else float(experiment.alphas[-1])
    latest_beta = None if experiment.n_repetitions == 0 else float(experiment.betas[-1])
    sample_alpha = latest_alpha if latest_alpha is not None and np.isfinite(latest_alpha) else None
    sample_beta = latest_beta if latest_beta is not None and np.isfinite(latest_beta) else None
    return distribution_scatter(
        cloud_image=cloud,
        x_range=population.x_range,
        y_range=population.y_range,
        population_alpha=population.alpha,
        population_beta=population.beta,
        sample_x=experiment.sample_x,
        sample_y=experiment.sample_y,
        sample_alpha=sample_alpha,
        sample_beta=sample_beta,
        height=_DIST_SCATTER_HEIGHT,
    ), latest_beta


def _paint_scatter_only(population, experiment, cloud: str, *, suffix: str) -> None:
    scatter, _ = _distribution_scatter_figure(population, experiment, cloud)
    chart_key = f"dist_scatter_{suffix}"
    if _DIST_SCATTER is None:
        st.plotly_chart(scatter, width="stretch", config=PLOTLY_CONFIG, key=chart_key)
    else:
        _DIST_SCATTER.plotly_chart(scatter, width="stretch", config=PLOTLY_CONFIG, key=chart_key)


def _paint_distributions(population, experiment, cloud: str, *, highlight: bool, suffix: str) -> None:
    """Step 1: scatter of lines (left) and sampling distribution of β̂ (right)."""
    scatter, latest_beta = _distribution_scatter_figure(population, experiment, cloud)
    chart_key = f"dist_scatter_{suffix}"
    if _DIST_SCATTER is None:
        st.plotly_chart(scatter, width="stretch", config=PLOTLY_CONFIG, key=chart_key)
    else:
        _DIST_SCATTER.plotly_chart(scatter, width="stretch", config=PLOTLY_CONFIG, key=chart_key)
    _paint_histogram(
        _DIST_BETA,
        values=experiment.betas,
        edges=experiment.beta_edges,
        population_value=population.beta,
        title=f"Sampling distribution of {_BETA_HAT}",
        x_title=f"Estimated slope {_BETA_HAT}",
        true_label="True β",
        mean_label=f"Mean {_BETA_HAT}",
        sd_label=f"SD({_BETA_HAT})",
        show_normal=experiment.show_normals,
        highlight=latest_beta if highlight else None,
        suffix=f"beta_{suffix}",
        decimals=3,
        height=_DIST_HIST_HEIGHT,
    )


def _paint_histogram(
    slot,
    *,
    values,
    edges,
    population_value,
    title,
    x_title,
    true_label,
    mean_label,
    sd_label,
    show_normal,
    highlight,
    suffix,
    decimals,
    height: int | None = None,
) -> None:
    mean = estimate_mean(values)
    spread = estimate_sd(values)
    normal_x = None
    normal_y = None
    # Always build the curve when allowed so show/hide only changes opacity.
    if can_show_normal(values):
        normal_x = np.linspace(float(edges[0]), float(edges[-1]), 160)
        normal_y = normal_frequency(
            normal_x,
            mean,
            spread,
            int(finite_estimates(values).size),
            float(edges[1] - edges[0]),
        )
    figure = sampling_histogram(
        values=values,
        edges=edges,
        population=population_value,
        title=title,
        x_title=x_title,
        normal_x=normal_x,
        normal_y=normal_y,
        normal_visible=bool(show_normal),
        highlight=highlight,
        mean=mean,
        standard_error=spread,
        decimals=decimals,
        height=_DIST_HIST_HEIGHT if height is None else int(height),
        uirevision=f"dist-hist-{suffix}",
    )
    stats = (
        '<p class="dist-stat">'
        f'<span class="pop">{true_label} {population_value:.3f}</span>'
        f'<span class="est">{mean_label} {_dist_number(estimate_mean(values))}</span>'
        f"<span>{sd_label} {_dist_number(estimate_sd(values))}</span>"
        "</p>"
    )
    if slot is None:
        st.plotly_chart(
            figure, width="stretch", config=PLOTLY_CONFIG, key=f"dist_hist_{suffix}"
        )
        st.markdown(stats, unsafe_allow_html=True)
        return
    with slot.container():
        st.plotly_chart(
            figure, width="stretch", config=PLOTLY_CONFIG, key=f"dist_hist_{suffix}"
        )
        st.markdown(stats, unsafe_allow_html=True)


_DIST_SCATTER = None
_DIST_SCATTER_NOTE = None
_DIST_BETA = None
_DIST_BETA_NOTE = None
_ONE_ALPHA_NOTE = None
_ONE_BETA_NOTE = None
_ONE_LEGEND = None
_DIST_SCATTER_HEIGHT = 515
_DIST_HIST_HEIGHT = 480
_ONE_SCATTER_HEIGHT = 460
_ONE_BETA_HEIGHT = 460
_ONE_HIST_HEIGHT = 230


def _signed_percent(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:+.1f}%"


def _se_numbers_html(*, title, true_name, true_value, benchmark_se, repeated_sd, repeated_n, estimated) -> str:
    return (
        '<div class="dist-compare">'
        '<div class="dist-compare-head">'
        f"<span>{title}</span>"
        f'<span class="pop">True {true_name} {true_value:.3f}</span>'
        "</div>"
        "<div>"
        '<span class="bench">Population benchmark</span>'
        f'<b class="bench">{_dist_number(benchmark_se)}</b>'
        "</div>"
        "<div>"
        f'<span class="ref">Repeated sampling ({repeated_n:,})</span>'
        f'<b class="ref">{_dist_number(repeated_sd)}</b>'
        "</div>"
        "<div>"
        '<span class="est">One sample</span>'
        f'<b class="est">{_dist_number(estimated)}</b>'
        "</div>"
        "</div>"
    )


def _interval_width(value, scale) -> str:
    if value is None or scale is None or not np.isfinite(value) or not np.isfinite(scale) or scale <= 0:
        return "0%"
    return f"{100.0 * float(value) / float(scale):.1f}%"


def _se_intervals_html(benchmark_se, repeated_sd, estimated) -> str:
    values = [value for value in (benchmark_se, repeated_sd, estimated) if value is not None and np.isfinite(value) and value > 0]
    scale = max(values) if values else None
    gap = relative_se_error(estimated, benchmark_se)
    rows = (
        ("bench", "Population benchmark", benchmark_se),
        ("ref", "Repeated sampling", repeated_sd),
        ("est", "One sample", estimated),
    )
    body = "".join(
        '<div class="se-scale-row">'
        f'<span class="{kind}">{label}</span>'
        '<span class="se-track">'
        f'<span class="se-interval {kind}" style="width:{_interval_width(value, scale)}"></span>'
        "</span>"
        f'<b class="{kind}">{_dist_number(value)}</b>'
        "</div>"
        for kind, label, value in rows
    )
    return (
        '<div class="se-scale">'
        f"{body}"
        f'<div class="est">One-sample SE error: {_signed_percent(gap)}</div>'
        "</div>"
    )


def _slope_calculation_html(error) -> str:
    if error is None or error.residual_sd is None or error.sxx is None or error.se_beta is None:
        residual = sxx_text = "—"
        result = "—"
    else:
        residual = f"{error.residual_sd:.3f}"
        sxx_text = f"{error.sxx:.2f}"
        result = _dist_number(error.se_beta)
    return (
        '<div class="dist-calc">'
        "<p>Residual noise: "
        "s = √[Σ(y<sub>i</sub> − ŷ<sub>i</sub>)<sup>2</sup> / (n − 2)] = "
        f"<b>{residual}</b></p>"
        "<p>Spread of X: "
        "S<sub>xx</sub> = Σ(x<sub>i</sub> − x̄)<sup>2</sup> = "
        f"<b>{sxx_text}</b></p>"
        '<p class="dist-formula">'
        f'<span class="se-key">SE({_BETA_HAT}) = s / √S<sub>xx</sub> = '
        f"{residual} / √{sxx_text} = "
        f'<span class="se-final">{result}</span></span></p>'
        '<p class="dist-note-line">More residual noise increases uncertainty; '
        "more spread in X reduces it.</p>"
        "</div>"
    )


def _discovery_html(error, benchmark, experiment) -> str:
    se_beta = None if error is None else error.se_beta
    repeated = estimate_sd(experiment.betas)
    error_text = _signed_percent(relative_se_error(se_beta, benchmark.se_beta))
    repeated_n = int(experiment.n_repetitions)
    observed_n = f"{repeated_n:,}" if repeated_n > 0 else "—"
    return (
        '<div class="se-discover">'
        '<p class="kicker">HOW CLOSE CAN ONE SAMPLE GET?</p>'
        '<div class="se-cards">'
        '<div class="se-card bench">'
        f'<span class="lab">Simulation benchmark<br>({benchmark.repetitions:,} samples)</span>'
        f'<span class="val">{_dist_number(benchmark.se_beta)}</span>'
        "</div>"
        '<div class="se-card ref">'
        f'<span class="lab">Observed SE<br>({observed_n} samples)</span>'
        f'<span class="val">{_dist_number(repeated)}</span>'
        "</div>"
        '<div class="se-card est">'
        '<span class="lab">Estimated SE<br>(ONE sample)</span>'
        f'<span class="val">{_dist_number(se_beta)}</span>'
        "</div>"
        "</div>"
        f'<p class="se-error">One-sample estimation error: {error_text}</p>'
        f'<p class="se-close">One sample lets us estimate how much {_BETA_HAT} would vary '
        "across repeated samples.</p>"
        "</div>"
    )


_ONE_INTRO = (
    '<div class="dist-step-one"></div>'
    '<p class="dist-intro">With thousands of samples we can see how much β̂ varies. '
    "With one sample we can still estimate that same uncertainty.</p>"
)

_ONE_LEFT_HEAD = (
    '<div class="one-col-head">'
    '<p class="title">WHAT IF WE COULD DRAW 1,000 SAMPLES?</p>'
    '<p class="sub">Repeated samples reveal how much the estimated slope varies.</p>'
    "</div>"
)

_ONE_RIGHT_HEAD = (
    '<div class="one-col-head">'
    '<p class="title">WHAT CAN WE LEARN FROM JUST ONE SAMPLE?</p>'
    '<p class="sub">Residual noise and the spread of X let us estimate the same uncertainty.</p>'
    "</div>"
)


def _paint_one_sample_hist(population, experiment) -> None:
    mean = estimate_mean(experiment.betas)
    spread = estimate_sd(experiment.betas)
    normal_x = normal_y = None
    if can_show_normal(experiment.betas):
        normal_x = np.linspace(float(experiment.beta_edges[0]), float(experiment.beta_edges[-1]), 160)
        normal_y = normal_frequency(
            normal_x,
            mean,
            spread,
            int(experiment.n_repetitions),
            float(experiment.beta_edges[1] - experiment.beta_edges[0]),
        )
    histogram = sampling_histogram(
        values=experiment.betas,
        edges=experiment.beta_edges,
        population=population.beta,
        title=f"Sampling distribution of {_BETA_HAT}",
        x_title=f"Estimated slope {_BETA_HAT}",
        normal_x=normal_x,
        normal_y=normal_y,
        normal_visible=True,
        mean=mean,
        standard_error=spread,
        decimals=3,
        height=_ONE_BETA_HEIGHT,
        se_label="±1 observed SE",
        uirevision="dist-hist-beta-one",
    )
    st.plotly_chart(histogram, width="stretch", config=PLOTLY_CONFIG, key="dist_hist_beta_one")
    if mean is not None and spread is not None and np.isfinite(mean) and np.isfinite(spread):
        st.markdown(
            f'<p class="dist-stat"><span class="pop">β {population.beta:.3f}</span>'
            f'<span class="est">Mean {_BETA_HAT} {_dist_number(mean)}</span>'
            f"<span>Observed SE {_dist_number(spread)}</span></p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p class="dist-stat">Draw repeated samples in Step 1 to see the spread of β̂.</p>',
            unsafe_allow_html=True,
        )


def _paint_one_sample_scatter(population, one, cloud: str):
    error = None
    if one.x is not None and one.y is not None:
        error = one_sample_standard_errors(one.x, one.y)
    sample_alpha = None if error is None or not np.isfinite(error.alpha) else error.alpha
    sample_beta = None if error is None or not np.isfinite(error.beta) else error.beta
    residual_sd = None if error is None else error.residual_sd
    scatter = distribution_scatter(
        cloud_image=cloud,
        x_range=population.x_range,
        y_range=population.y_range,
        population_alpha=population.alpha,
        population_beta=population.beta,
        sample_x=one.x,
        sample_y=one.y,
        sample_alpha=sample_alpha,
        sample_beta=sample_beta,
        height=_ONE_SCATTER_HEIGHT,
        show_residuals=one.show_residuals,
        residual_sd=residual_sd,
        show_residual_band=one.show_band,
    )
    st.plotly_chart(scatter, width="stretch", config=PLOTLY_CONFIG, key="dist_scatter_one")
    st.markdown(_slope_calculation_html(error), unsafe_allow_html=True)
    return error


def _paint_one_sample(population, experiment, one, cloud: str, benchmark) -> None:
    del benchmark
    _paint_one_sample_hist(population, experiment)
    return _paint_one_sample_scatter(population, one, cloud)


def _p_text(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    if value < 0.0001:
        return "< 0.0001"
    return f"{value:.4f}"


def _null_rng() -> np.random.Generator:
    if "null_rng" not in st.session_state:
        st.session_state.null_rng = np.random.default_rng(20261017)
    return st.session_state.null_rng


def _t_result_html(result) -> str:
    beta = _dist_number(result.beta)
    standard_error = _dist_number(result.standard_error)
    statistic = _dist_number(result.t_statistic)
    if result.reject is None:
        decision = '<div class="t-decision keep">No SE</div>'
        reading = ""
    else:
        if result.reject:
            verdict = "Reject H\u2080"
            significance = "It is statistically significant at 5%."
            tone = "reject"
        else:
            verdict = "Fail to reject H\u2080"
            significance = "It is not statistically significant at 5%."
            tone = "keep"
        decision = f'<div class="t-decision {tone}">{verdict}</div>'
        reading = (
            f'<p class="t-read">The observed slope is {_dist_number(abs(result.t_statistic))} '
            f"standard errors from zero. {significance}</p>"
        )
    return (
        '<div class="t-panel">'
        '<div class="t-strip">'
        f'<div class="t-cell"><span class="lab">{_BETA_HAT}</span><span class="val">{beta}</span></div>'
        f'<div class="t-cell"><span class="lab">SE({_BETA_HAT})</span><span class="val">{standard_error}</span></div>'
        f'<div class="t-cell emph"><span class="lab">t</span><span class="val">{statistic}</span></div>'
        f'<div class="t-cell emph"><span class="lab">p-value</span><span class="val">{_p_text(result.p_value)}</span></div>'
        '<div class="t-cell"><span class="lab">&alpha;</span><span class="val">5%</span></div>'
        f"{decision}"
        "</div>"
        f"{reading}"
        "</div>"
    )


def _render_null_slope(sample_size: int, draw: bool, true_beta: float) -> None:
    population = null_population(true_beta)
    stored = st.session_state.get("null_sample")
    if (
        draw
        or not isinstance(stored, tuple)
        or len(stored) != 2
        or int(stored[0]) != int(sample_size)
    ):
        indices = _null_rng().choice(population.hours.size, size=int(sample_size), replace=False)
        st.session_state.null_sample = (int(sample_size), indices)
    else:
        indices = stored[1]
    hours = population.hours[indices]
    scores = population.scores[indices]
    fitted = fit_ols(hours, scores)
    result = slope_t_test(hours, scores)
    degrees = result.degrees_of_freedom if result.degrees_of_freedom > 0 else 1
    grid, density = t_density_curve(degrees, result.t_statistic, result.critical_value)
    st.markdown('<div class="dist-step-test"></div>', unsafe_allow_html=True)
    chart_col, test_col = st.columns([52, 48], gap="small", vertical_alignment="top")
    with chart_col:
        st.plotly_chart(
            null_slope_scatter(
                sample_x=hours,
                sample_y=scores,
                sample_alpha=None if fitted.degenerate else fitted.alpha,
                sample_beta=None if fitted.degenerate else fitted.beta,
                population_alpha=population.alpha,
                population_beta=population.beta,
                height=486,
            ),
            width="stretch",
            config=PLOTLY_CONFIG,
            key="null_scatter",
        )
    with test_col:
        st.plotly_chart(
            slope_sampling_figure(
                beta=result.beta,
                standard_error=result.standard_error,
                height=236,
            ),
            width="stretch",
            config=PLOTLY_CONFIG,
            key="null_slope_dist",
        )
        st.markdown(
            f'<p class="t-scale">Same observation, different scale: '
            f"t = {_BETA_HAT} / SE({_BETA_HAT}).</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            t_test_figure(
                grid=grid,
                density=density,
                statistic=result.t_statistic,
                critical=result.critical_value,
                degrees=result.degrees_of_freedom,
                height=236,
            ),
            width="stretch",
            config=PLOTLY_CONFIG,
            key="null_t",
        )
    st.markdown(_t_result_html(result), unsafe_allow_html=True)


def render_sampling_inference_tab(population, rng: np.random.Generator) -> None:
    """Combined Sampling & Inference: One Sample plus the three distribution steps."""
    sub = st.segmented_control(
        "Sampling & Inference section",
        options=[key for key, _ in SAMPLING_INFERENCE_SUBTABS],
        format_func=lambda key: SAMPLING_INFERENCE_SUB_LABELS[key],
        key="sampling_inference_sub",
        default="one_sample",
        label_visibility="collapsed",
        width="content",
    )
    if sub not in SAMPLING_INFERENCE_SUB_LABELS:
        sub = "one_sample"
    if sub == "one_sample":
        # Introduction owns its Load Dataset overlay.
        render_introduction_tab(population, rng)
        return
    step = {"repeated": "repeated", "estimate_se": "one", "ttest": "test"}[sub]
    render_sampling_tab(population, rng, step=step)
    if _app_data_drawer_open():
        _dataset_overlay_dialog("sampling", population, rng=rng)


def _sampling_status_row(population, experiment) -> None:
    sample_n = (
        int(experiment.sample_size)
        if getattr(experiment, "sample_x", None) is not None
        else 0
    )
    _render_dataset_status_row(
        _dataset_status_html(
            "Students",
            sample_n,
            int(population.n),
            "Hours studied",
            "Exam score",
        )
    )


@st.fragment(key="dist_repeated")
def _render_repeated_sampling(population, rng: np.random.Generator) -> None:
    """Repeated sampling controls + charts (fragment-scoped to avoid page flash)."""
    global _DIST_SCATTER, _DIST_SCATTER_NOTE, _DIST_BETA
    experiment = _dist_experiment(population)
    pending_n = st.session_state.get("dist_sample_size", experiment.sample_size)
    if int(pending_n) != int(experiment.sample_size):
        st.session_state.dist_normals_toggle = False
    experiment = experiment_for_sample_size(
        experiment,
        int(pending_n),
        population.hours_studied,
        population.exam_score,
        population.alpha,
        population.beta,
    )
    if "dist_normals_toggle" not in st.session_state:
        st.session_state.dist_normals_toggle = bool(experiment.show_normals)
    _sampling_status_row(population, experiment)
    curves_allowed = can_show_normal(experiment.alphas) or can_show_normal(experiment.betas)
    curves_on = bool(st.session_state.dist_normals_toggle)
    added = 0
    frames: list = []
    toolbar = st.container(horizontal=True, vertical_alignment="bottom", gap="small", wrap=False)
    with toolbar:
        sample_size = st.select_slider(
            "Sample size (n)",
            options=SAMPLE_SIZES,
            value=experiment.sample_size,
            key="dist_sample_size",
            width=220,
        )
        if st.button("Draw 1", key="dist_draw_1", width=88):
            added = 1
        if st.button("Draw 10", key="dist_draw_10", width=96):
            added = 10
        if st.button("Draw 100", key="dist_draw_100", width=104):
            added = 100
        if st.button("Draw 1,000", key="dist_draw_1000", width=118):
            added = 1000
        reset = st.button(
            "Reset",
            key="dist_reset",
            width=84,
            on_click=lambda: st.session_state.update(dist_normals_toggle=False),
        )
        st.toggle(
            "Normal curves",
            key="dist_normals_toggle",
            disabled=not curves_allowed and not curves_on,
        )
    status = st.empty()
    if reset:
        experiment = new_experiment(
            int(sample_size),
            population.hours_studied,
            population.exam_score,
            population.alpha,
            population.beta,
        )
        added = 0
    else:
        experiment = replace(
            experiment,
            show_normals=bool(st.session_state.dist_normals_toggle),
        )
    if added:
        frames = animated_draws(
            experiment,
            population.hours_studied,
            population.exam_score,
            added,
            rng,
        )
        experiment = frames[-1]
    st.session_state.dist_experiment = experiment
    cloud = population_cloud_image(
        population.hours_studied,
        population.exam_score,
        population.x_range,
        population.y_range,
        cache_key=population.sha256,
    )
    animate = bool(frames)
    snapshots = frames or [experiment]
    with st.container(key="dist_workspace"):
        chart_col, hist_col = st.columns([1, 1], gap="small", vertical_alignment="top")
        if animate:
            with chart_col:
                with st.container(key="dist_scatter"):
                    _DIST_SCATTER = st.empty()
                    _DIST_SCATTER_NOTE = st.empty()
            with hist_col:
                with st.container(key="dist_hists"):
                    with st.container(key="dist_beta"):
                        _DIST_BETA = st.empty()
            _DIST_SCATTER_NOTE.markdown('<p class="dist-sd">&nbsp;</p>', unsafe_allow_html=True)
            for index, snapshot in enumerate(snapshots):
                status.markdown(_dist_status(snapshot), unsafe_allow_html=True)
                _paint_distributions(
                    population,
                    snapshot,
                    cloud,
                    highlight=index == len(snapshots) - 1,
                    suffix=str(index),
                )
                if index < len(snapshots) - 1:
                    time.sleep(_dist_pause(added, len(snapshots) - 1))
            time.sleep(0.35)
            _paint_distributions(population, experiment, cloud, highlight=False, suffix="settled")
        else:
            _DIST_SCATTER = None
            _DIST_BETA = None
            status.markdown(_dist_status(experiment), unsafe_allow_html=True)
            with chart_col:
                with st.container(key="dist_scatter"):
                    _paint_scatter_only(population, experiment, cloud, suffix="live")
                    st.markdown('<p class="dist-sd">&nbsp;</p>', unsafe_allow_html=True)
            with hist_col:
                with st.container(key="dist_hists"):
                    with st.container(key="dist_beta"):
                        _paint_histogram(
                            None,
                            values=experiment.betas,
                            edges=experiment.beta_edges,
                            population_value=population.beta,
                            title=f"Sampling distribution of {_BETA_HAT}",
                            x_title=f"Estimated slope {_BETA_HAT}",
                            true_label="True β",
                            mean_label=f"Mean {_BETA_HAT}",
                            sd_label=f"SD({_BETA_HAT})",
                            show_normal=experiment.show_normals,
                            highlight=None,
                            suffix="beta_live",
                            decimals=3,
                            height=_DIST_HIST_HEIGHT,
                        )
    final_allowed = can_show_normal(experiment.alphas) or can_show_normal(experiment.betas)
    if final_allowed and not curves_allowed and not curves_on:
        st.rerun(scope="fragment")
    st.markdown('<p class="dist-note">&nbsp;</p>', unsafe_allow_html=True)


@st.fragment(key="dist_estimate_se")
def _render_estimate_se(population, rng: np.random.Generator) -> None:
    """Estimate SE controls + charts (fragment-scoped to avoid page flash)."""
    experiment = _dist_experiment(population)
    pending_n = st.session_state.get("dist_sample_size", experiment.sample_size)
    experiment = experiment_for_sample_size(
        experiment,
        int(pending_n),
        population.hours_studied,
        population.exam_score,
        population.alpha,
        population.beta,
    )
    one = one_sample_for_size(
        st.session_state.get("dist_one_sample"),
        experiment.sample_size,
        experiment,
    )
    if "dist_show_residuals" not in st.session_state:
        st.session_state.dist_show_residuals = bool(one.show_residuals)
    if "dist_show_band" not in st.session_state:
        st.session_state.dist_show_band = bool(one.show_band)
    _sampling_status_row(population, experiment)
    toolbar = st.container(horizontal=True, vertical_alignment="bottom", gap="small", wrap=False)
    with toolbar:
        sample_size = st.select_slider(
            "Sample size (n)",
            options=SAMPLE_SIZES,
            value=experiment.sample_size,
            key="dist_sample_size",
            width=220,
        )
        draw_another = st.button("Draw another sample", key="dist_draw_another", width=168)
        st.toggle("Residuals", key="dist_show_residuals")
        st.toggle("Residual SD band", key="dist_show_band")
    experiment = experiment_for_sample_size(
        experiment,
        int(sample_size),
        population.hours_studied,
        population.exam_score,
        population.alpha,
        population.beta,
    )
    one = one_sample_for_size(one, experiment.sample_size, experiment)
    if draw_another:
        sample = draw_sample(population, experiment.sample_size, rng)
        one = replace(one, x=sample.hours_studied, y=sample.exam_score)
    one = replace(
        one,
        show_residuals=bool(st.session_state.dist_show_residuals),
        show_band=bool(st.session_state.dist_show_band),
    )
    st.session_state.dist_experiment = experiment
    st.session_state.dist_one_sample = one
    cloud = population_cloud_image(
        population.hours_studied,
        population.exam_score,
        population.x_range,
        population.y_range,
        cache_key=population.sha256,
    )
    if not benchmark_is_cached(population.sha256, experiment.sample_size):
        with st.spinner("Computing the population benchmark…"):
            benchmark = population_benchmark(
                population.hours_studied,
                population.exam_score,
                experiment.sample_size,
                population.sha256,
            )
    else:
        benchmark = population_benchmark(
            population.hours_studied,
            population.exam_score,
            experiment.sample_size,
            population.sha256,
        )
    with st.container(key="one_workspace"):
        chart_col, hist_col = st.columns([50, 50], gap="small", vertical_alignment="top")
        with chart_col:
            st.markdown(_ONE_RIGHT_HEAD, unsafe_allow_html=True)
            with st.container(key="one_scatter"):
                error = _paint_one_sample_scatter(population, one, cloud)
        with hist_col:
            st.markdown(_ONE_LEFT_HEAD, unsafe_allow_html=True)
            with st.container(key="one_beta"):
                _paint_one_sample_hist(population, experiment)
    st.markdown(_discovery_html(error, benchmark, experiment), unsafe_allow_html=True)


@st.fragment(key="dist_ttest")
def _render_ttest_sampling(population, rng: np.random.Generator) -> None:
    """Null-slope t-test controls (fragment-scoped to avoid page flash)."""
    del rng
    experiment = _dist_experiment(population)
    _sampling_status_row(population, experiment)
    toolbar = st.container(horizontal=True, vertical_alignment="bottom", gap="small", wrap=False)
    with toolbar:
        null_n = st.select_slider(
            "Sample size (n)",
            options=SAMPLE_SIZES,
            value=100,
            key="null_sample_size",
            width=200,
        )
        true_beta = st.slider(
            "True population slope (β)",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            key="null_true_beta",
            width=230,
        )
        draw_null = st.button("Draw another sample", key="null_draw", width=156)
        st.button("Draw 100 samples", key="null_draw_100", disabled=True, width=140)
    _render_null_slope(int(null_n), draw_null, float(true_beta))


def render_sampling_tab(
    population,
    rng: np.random.Generator,
    *,
    step: str,
) -> None:
    """Sampling-distributions body for one inference subtab (repeated / one / test)."""
    if step not in ("repeated", "one", "test"):
        step = "repeated"
    with st.container(key="dist_layout", gap="xxsmall"):
        if step == "one":
            st.markdown(_ONE_INTRO, unsafe_allow_html=True)
            _render_estimate_se(population, rng)
        elif step == "test":
            _render_ttest_sampling(population, rng)
        else:
            _render_repeated_sampling(population, rng)



FIT_CHART_HEIGHT = 340  # Plotting area ~340px; margins stay fixed in fitting_scatter


def _fit_rng() -> np.random.Generator:
    if "fit_rng" not in st.session_state:
        st.session_state.fit_rng = np.random.default_rng(20261019)
    return st.session_state.fit_rng


def _fitting_xy_defaults(dataset_key: str) -> tuple[str, str]:
    catalog = get_catalog(dataset_key, st.session_state)
    return catalog.default_x, catalog.default_y


def _ensure_fitting_xy(dataset_key: str) -> tuple[str, str]:
    catalog = get_catalog(dataset_key, st.session_state)
    names = catalog.variable_names()
    if "fit_x" not in st.session_state or st.session_state.fit_x not in names:
        st.session_state.fit_x = catalog.default_x
    if "fit_y" not in st.session_state or st.session_state.fit_y not in names:
        st.session_state.fit_y = catalog.default_y
    if st.session_state.fit_x == st.session_state.fit_y:
        # Keep predictor; pick a different outcome.
        for name in names:
            if name != st.session_state.fit_x:
                st.session_state.fit_y = name
                break
    return st.session_state.fit_x, st.session_state.fit_y


def _fitting_source():
    ensure_registry_state(st.session_state)
    if "fit_dataset_key" not in st.session_state:
        st.session_state.fit_dataset_key = "students"
    if "fit_seed" not in st.session_state:
        st.session_state.fit_seed = 20261018
    x_name, y_name = _ensure_fitting_xy(st.session_state.fit_dataset_key)
    return resolve_fitting_dataset(
        st.session_state.fit_dataset_key,
        x_name=x_name,
        y_name=y_name,
        session=st.session_state,
        seed=st.session_state.fit_seed,
    )


def _fitting_sample_plan(source) -> tuple[int, int, bool, float]:
    """Return (requested_n, actual_n, capped, percentage)."""
    use_all = bool(st.session_state.get("fit_use_all", False))
    pct = float(st.session_state.get("fit_pct", DEFAULT_SAMPLE_PCT))
    requested, actual, capped = resolve_sample_n(
        source.available_n,
        pct,
        use_all=use_all,
    )
    return requested, actual, capped, pct


def _fitting_view_token(source) -> tuple:
    _, actual_n, _, pct = _fitting_sample_plan(source)
    return (
        source.key,
        source.x_name,
        source.y_name,
        int(source.seed or 0),
        int(actual_n),
        float(pct),
        bool(st.session_state.get("fit_use_all", False)),
        int(st.session_state.get("fit_draw", 0)),
    )


def _ensure_fitting_observations(source):
    token = _fitting_view_token(source)
    stored = st.session_state.get("fit_view")
    if stored is None or st.session_state.get("fit_view_token") != token:
        _, actual_n, _, _ = _fitting_sample_plan(source)
        view = subsample_observations(source, actual_n, _fit_rng())
        st.session_state.fit_view = view
        st.session_state.fit_view_token = token
        # Fingerprint so we can detect accidental sample changes.
        st.session_state.fit_view_fingerprint = (
            float(np.sum(view.x)),
            float(np.sum(view.y)),
            int(view.n),
        )
        # New sample: clear prior Fit OLS mini-chart / animation state.
        st.session_state.fit_found = False
        _clear_fitting_animation()
        _clear_fitting_metric_paths()
    return st.session_state.fit_view


def _queue_fitting_sliders(alpha: float, beta: float) -> None:
    """Store slider values to apply before the widgets are created."""
    st.session_state.fit_pending_alpha = float(alpha)
    st.session_state.fit_pending_beta = float(beta)


def _apply_pending_fitting_sliders() -> None:
    if "fit_pending_alpha" in st.session_state:
        st.session_state.fit_alpha = float(st.session_state.pop("fit_pending_alpha"))
    if "fit_pending_beta" in st.session_state:
        st.session_state.fit_beta = float(st.session_state.pop("fit_pending_beta"))


def _clear_fitting_sliders() -> None:
    for key in ("fit_alpha", "fit_beta", "fit_pending_alpha", "fit_pending_beta"):
        if key in st.session_state:
            del st.session_state[key]


def _reset_fitting_sample_state() -> None:
    st.session_state.fit_draw = 0
    st.session_state.fit_poor_index = 0
    st.session_state.fit_found = False
    _clear_fitting_sliders()
    st.session_state.pop("fit_view", None)
    st.session_state.pop("fit_view_token", None)
    st.session_state.pop("fit_bootstrapped", None)
    st.session_state.pop("fit_view_fingerprint", None)
    st.session_state.pop("fit_sse_before", None)
    _clear_fitting_prediction()
    _clear_fitting_animation()
    _clear_fitting_metric_paths()
    _clear_fitting_student_selection()


def _clear_fitting_animation() -> None:
    for key in ("fit_animating", "fit_anim_frame", "fit_anim_steps"):
        st.session_state.pop(key, None)


def _clear_fitting_metric_paths() -> None:
    for key in (
        "fit_sse_path",
        "fit_r2_path",
        "fit_sse_range",
        "fit_r2_range",
    ):
        st.session_state.pop(key, None)


def _clear_fitting_prediction() -> None:
    for key in (
        "fit_predict_x",
        "fit_predict_input",
        "fit_predict_committed_x",
        "fit_predict_committed_raw",
        "fit_predict_active",
    ):
        st.session_state.pop(key, None)


def _bump_fitting_scatter_nonce() -> None:
    st.session_state.fit_scatter_nonce = int(st.session_state.get("fit_scatter_nonce", 0)) + 1


def _clear_fitting_student_selection() -> None:
    """Deselect the EXPLAIN student (chart highlight, table highlight)."""
    st.session_state.fit_selected = None
    st.session_state.fit_explain_pick = 0
    st.session_state.fit_explain_pick_seen = 0
    st.session_state.fit_applied_chart_idx = None
    # Remount the chart so a stale Plotly selection cannot re-select the student.
    _bump_fitting_scatter_nonce()


def _sync_fitting_prediction_active() -> None:
    """Hide prediction if X is cleared or no longer matches the last Predict click."""
    if not st.session_state.get("fit_predict_active"):
        return
    raw = str(st.session_state.get("fit_predict_input", "")).strip()
    if not raw:
        st.session_state.fit_predict_active = False
        return
    committed = str(st.session_state.get("fit_predict_committed_raw", "")).strip()
    if raw != committed:
        st.session_state.fit_predict_active = False


def _commit_fitting_prediction() -> None:
    """Button callback: commit X before the script reruns (so the chart updates)."""
    raw = str(st.session_state.get("fit_predict_input", "")).strip()
    try:
        value = float(raw.replace(",", ""))
    except ValueError:
        st.session_state.fit_predict_active = False
        st.session_state.pop("fit_predict_committed_x", None)
        st.session_state.pop("fit_predict_committed_raw", None)
        return
    if not np.isfinite(value):
        st.session_state.fit_predict_active = False
        return
    st.session_state.fit_predict_committed_x = float(value)
    st.session_state.fit_predict_committed_raw = raw
    st.session_state.fit_predict_active = True


def _active_fitting_prediction_x() -> float | None:
    if not st.session_state.get("fit_predict_active"):
        return None
    value = st.session_state.get("fit_predict_committed_x")
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _hats_match_ols(
    alpha: float,
    beta: float,
    ols,
    *,
    a_step: float,
    b_step: float,
) -> bool:
    """True when the candidate line is at the OLS estimates (within a half-step)."""
    if ols is None or getattr(ols, "degenerate", False):
        return False
    alpha_tol = max(abs(float(a_step)) * 0.51, 1e-9)
    beta_tol = max(abs(float(b_step)) * 0.51, 1e-9)
    return (
        abs(float(alpha) - float(ols.alpha)) <= alpha_tol
        and abs(float(beta) - float(ols.beta)) <= beta_tol
    )


def _toggle_fit_view_mode() -> None:
    """Chart/Table button callback — flip view without an extra st.rerun()."""
    cur = st.session_state.get("fit_view_mode", "chart")
    if cur not in ("chart", "table"):
        cur = "chart"
    st.session_state.fit_view_mode = "table" if cur == "chart" else "chart"


def _fit_fragment_rerun() -> None:
    """Rerun only the Fit-the-line fragment (avoids full-page flash)."""
    st.rerun(scope="fragment")


def _open_understanding_r2() -> None:
    """Navigate from the model-use EXPLAIN box to the Understanding R² panel."""
    st.session_state.fit_lab_panel = "r2"


def _snap_r2_experiment_to_ols(
    alpha: float,
    beta: float,
    a0: float,
    a1: float,
    b0: float,
    b1: float,
) -> None:
    """Snap R²-lesson sliders to OLS before the widgets are instantiated."""
    st.session_state.r2_exp_alpha = float(np.clip(alpha, a0, a1))
    st.session_state.r2_exp_beta = float(np.clip(beta, b0, b1))


def _reset_fitting_line(view, away_from_ols: bool = True) -> None:
    ols = ols_fit(view.x, view.y)
    (a0, a1, _), (b0, b1, _) = view.slider_bounds()
    if away_from_ols:
        index = int(st.session_state.get("fit_poor_index", 0))
        alpha, beta = poor_candidate_line(
            view.x,
            view.y,
            index=index,
            alpha_bounds=(a0, a1),
            beta_bounds=(b0, b1),
        )
        current = sum_of_squared_errors(view.x, view.y, alpha, beta)
        if current < 1.25 * float(ols.sse):
            for offset in range(1, 20):
                alpha, beta = poor_candidate_line(
                    view.x,
                    view.y,
                    index=index + offset,
                    alpha_bounds=(a0, a1),
                    beta_bounds=(b0, b1),
                )
                current = sum_of_squared_errors(view.x, view.y, alpha, beta)
                if current >= 1.25 * float(ols.sse):
                    index = index + offset
                    break
        _queue_fitting_sliders(alpha, beta)
        st.session_state.fit_poor_index = index + 1
        st.session_state.fit_found = False
    else:
        _queue_fitting_sliders(ols.alpha, ols.beta)
    _clear_fitting_student_selection()
    st.session_state.fit_line_token = _fitting_view_token(_fitting_source())


def _observation_index_from_chart_event(event) -> int | None:
    """First clicked observation index from a Plotly selection event."""
    xs = st.session_state.get("fit_chart_x")
    ys = st.session_state.get("fit_chart_y")
    return observation_index_from_selection(
        event,
        x_values=None if xs is None else np.asarray(xs, dtype=np.float64),
        y_values=None if ys is None else np.asarray(ys, dtype=np.float64),
        observations_curve=0,
    )


def _apply_fitting_chart_selection(index: int | None, *, n: int) -> bool:
    """Update shared chart/EXPLAIN selection. Returns True if it changed."""
    if index is None or n <= 0 or not (0 <= int(index) < n):
        return False
    idx = int(index)
    if idx == st.session_state.get("fit_applied_chart_idx") and idx == st.session_state.get(
        "fit_selected"
    ):
        return False
    st.session_state.fit_applied_chart_idx = idx
    st.session_state.fit_selected = idx
    st.session_state.fit_explain_pick = idx + 1
    st.session_state.fit_explain_pick_seen = idx + 1
    return True


def _on_fit_scatter_select() -> None:
    """Plotly on_select callback: sync EXPLAIN before the script body runs."""
    nonce = int(st.session_state.get("fit_scatter_nonce", 0))
    state = st.session_state.get(f"fit_scatter_{nonce}")
    xs = st.session_state.get("fit_chart_x")
    n = int(len(xs)) if xs is not None else 0
    _apply_fitting_chart_selection(_observation_index_from_chart_event(state), n=n)


def _selected_point_index(event) -> int | None:
    """Backward-compatible helper: chart observation index, else current selection."""
    index = _observation_index_from_chart_event(event)
    if index is not None:
        return index
    return st.session_state.get("fit_selected")


def _selected_dataframe_row(event) -> int | None:
    """Map a Streamlit dataframe row selection to the sample index."""
    if event is None:
        return st.session_state.get("fit_selected")
    selection = getattr(event, "selection", None)
    if selection is None:
        return st.session_state.get("fit_selected")
    rows = selection.get("rows") if hasattr(selection, "get") else getattr(selection, "rows", None)
    if not rows:
        return st.session_state.get("fit_selected")
    return int(rows[0])


def _fitting_residuals_table(view, alpha: float, beta: float) -> pd.DataFrame:
    """Observation-level residuals for the current candidate line."""
    y_hat = predicted_scores(view.x, alpha, beta)
    y = np.asarray(view.y, dtype=np.float64)
    resid = y - y_hat
    return pd.DataFrame(
        {
            "Observation": np.arange(1, int(view.n) + 1, dtype=int),
            "X": np.asarray(view.x, dtype=np.float64),
            "Y": y,
            "Ŷ": y_hat,
            "Residual (Y−Ŷ)": resid,
            "Residual²": resid * resid,
        }
    )


def _style_selected_student(frame: pd.DataFrame, selected: int | None):
    if selected is None or not (0 <= int(selected) < len(frame)):
        return frame
    highlight = int(selected)

    def _row_style(row: pd.Series) -> list[str]:
        if int(row.name) == highlight:
            return ["background-color: #DBEAFE; font-weight: 650"] * len(row)
        return [""] * len(row)

    return frame.style.apply(_row_style, axis=1)


def _format_r_squared(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value:.3f}"


def _format_sse(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.1f}"


def _r2_variation_bar_html(*, explained_frac: float) -> str:
    if not np.isfinite(explained_frac):
        return ""
    pct = max(0.0, min(100.0, 100.0 * float(explained_frac)))
    unexplained_pct = max(0.0, 100.0 - pct)
    explained_label = f"Explained {pct:.1f}%" if pct >= 12 else f"{pct:.0f}%"
    unexplained_label = (
        f"Unexplained {unexplained_pct:.1f}%" if unexplained_pct >= 12 else f"{unexplained_pct:.0f}%"
    )
    return (
        '<div class="fit-r2-bar" role="img" '
        f'aria-label="Explained {pct:.1f} percent, unexplained {unexplained_pct:.1f} percent">'
        f'<div class="explained" style="width:{pct:.2f}%">{explained_label}</div>'
        f'<div class="unexplained" style="width:{unexplained_pct:.2f}%">{unexplained_label}</div>'
        "</div>"
    )


def _format_r2_pct(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{100.0 * float(value):.1f}%"


def _outcome_unit_phrase(y_axis: str) -> str:
    """Plain-language unit phrase for residual SE (e.g. exam-score points)."""
    label = " ".join(str(y_axis).strip().split())
    lowered = label.lower()
    if lowered == "exam score":
        return "exam-score points"
    if "point" in lowered:
        return lowered
    return lowered


def _metric_axis_range(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    pad = max(0.08 * (hi - lo), abs(hi) * 0.02, 1e-6)
    return lo - pad, hi + pad


_FIT_METRIC_RELATION = "lm() minimizes SSR → R² increases."


def _fit_metric_label_html(code: str) -> str:
    """Chart label with tip that includes the SSR/R² relationship."""
    canonical = resolve_tip_code(code)
    full, detail = STAT_TIPS[canonical]
    formula = STAT_TIP_FORMULAS.get(canonical)
    formula_bit = f"<br>{formula}" if formula else ""
    return (
        f'<p class="fit-metric-lab">'
        f'<span class="stat-tip" tabindex="0">'
        f"{tip_visible(code)}"
        f'<span class="stat-tip-bubble" role="tooltip">'
        f"<strong>{full}</strong><br>{detail}{formula_bit}<br>{_FIT_METRIC_RELATION}"
        f"</span></span></p>"
    )


def _fit_r2_range_label_html(r2_start: float, r2_end: float) -> str:
    """Initial → final R² percentages for the current sample (under R² ↑)."""
    visible = f"{_format_r2_pct(r2_start)} → {_format_r2_pct(r2_end)}"
    return f'<p class="fit-metric-r2-range">{visible}</p>'


def _fit_r2_range_spacer_html() -> str:
    """Invisible placeholder matching the R² range-line height."""
    return (
        '<p class="fit-metric-r2-range fit-metric-r2-spacer">0.0% → 0.0%</p>'
    )


_FIT_METRIC_WIDTH = 75
_FIT_METRIC_HEIGHT = 50
_FIT_METRIC_EMPTY_RANGE = (0.0, 1.0)


def _paint_fit_metric_pair(
    *,
    sse_values: np.ndarray,
    r2_values: np.ndarray,
    sse_range: tuple[float, float],
    r2_range: tuple[float, float],
    x_max: int,
    sse_slot,
    r2_slot,
) -> None:
    """Update curve plots only (labels stay mounted outside the empties)."""
    sse_fig = fit_metric_mini_chart(
        values=sse_values,
        y_range=sse_range,
        x_max=x_max,
        title="",
        height=_FIT_METRIC_HEIGHT,
        width=_FIT_METRIC_WIDTH,
        uirevision="fit-sse-mini",
    )
    r2_fig = fit_metric_mini_chart(
        values=r2_values,
        y_range=r2_range,
        x_max=x_max,
        title="",
        height=_FIT_METRIC_HEIGHT,
        width=_FIT_METRIC_WIDTH,
        uirevision="fit-r2-mini",
    )
    sse_slot.plotly_chart(
        sse_fig,
        width=_FIT_METRIC_WIDTH,
        config=FIT_MINI_PLOTLY_CONFIG,
        key="fit_sse_mini",
    )
    r2_slot.plotly_chart(
        r2_fig,
        width=_FIT_METRIC_WIDTH,
        config=FIT_MINI_PLOTLY_CONFIG,
        key="fit_r2_mini",
    )


def _mount_fit_metric_toolbar(
    metrics_col,
    *,
    sse_values: np.ndarray,
    r2_values: np.ndarray,
    sse_range: tuple[float, float],
    r2_range: tuple[float, float],
    x_max: int,
    r2_start: float,
    r2_end: float,
    show_r2_range: bool,
    animating: bool,
) -> tuple:
    """Mount SSR/R² side by side inside one shared bordered box."""
    with metrics_col:
        with st.container(key="fit_metrics_box"):
            sse_col, r2_col = st.columns([1, 1], gap="small")
            with sse_col:
                st.markdown(
                    _fit_metric_label_html("SSR"), unsafe_allow_html=True
                )
                # Match R² range-line height so both curves start on the same row.
                st.markdown(_fit_r2_range_spacer_html(), unsafe_allow_html=True)
                sse_slot = st.empty()
            with r2_col:
                st.markdown(
                    _fit_metric_label_html("R²"), unsafe_allow_html=True
                )
                if show_r2_range:
                    st.markdown(
                        _fit_r2_range_label_html(r2_start, r2_end),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(_fit_r2_range_spacer_html(), unsafe_allow_html=True)
                r2_slot = st.empty()
    if not animating:
        _paint_fit_metric_pair(
            sse_values=sse_values,
            r2_values=r2_values,
            sse_range=sse_range,
            r2_range=r2_range,
            x_max=x_max,
            sse_slot=sse_slot,
            r2_slot=r2_slot,
        )
    return sse_slot, r2_slot


def _render_r_code_panel(view, ols, *, alpha: float, beta: float, lm_run: bool) -> None:
    """Collapsed-by-default R commands + authentic summary(lm) console output."""
    commands = r_lm_commands(
        predictor=view.r_predictor,
        response=view.r_response,
    )
    with st.container(key="fit_r_panel"):
        yours = (
            f'<span class="yours">Your line (sliders): '
            f"α = {float(alpha):.4f}, β = {float(beta):.4f}</span>"
        )
        if lm_run and not ols.degenerate:
            ols_bit = (
                f'<span class="ols">lm() OLS: '
                f"α̂ = {float(ols.alpha):.4f}, β̂ = {float(ols.beta):.4f}</span>"
            )
        else:
            ols_bit = (
                '<span class="ols">lm() OLS: run lm() to reveal the least-squares fit</span>'
            )
        st.markdown(
            f'<p class="fit-r-line-note">{yours} &nbsp;·&nbsp; {ols_bit}</p>',
            unsafe_allow_html=True,
        )
        cmd_col, out_col = st.columns([0.35, 0.65], gap="medium")
        with cmd_col:
            with st.container(key="fit_r_commands"):
                st.markdown(
                    '<p class="fit-r-col-lab">R Commands</p>',
                    unsafe_allow_html=True,
                )
                st.code(commands, language="r", wrap_lines=True)
        with out_col:
            with st.container(key="fit_r_console"):
                st.markdown(
                    '<p class="fit-r-col-lab">R Console Output</p>',
                    unsafe_allow_html=True,
                )
                if lm_run and not ols.degenerate:
                    st.code(
                        r_lm_summary_text(
                            view.x,
                            view.y,
                            predictor=view.r_predictor,
                            response=view.r_response,
                        ),
                        language=None,
                        wrap_lines=True,
                    )
                else:
                    st.markdown(
                        '<p class="fit-r-waiting">summary(model) appears after you run lm().</p>',
                        unsafe_allow_html=True,
                    )


R2_LESSON_STEPS = (
    "baseline",
    "predictor",
    "explained",
    "experiment",
)


def _r2_lesson_step_label(key: str) -> str:
    return {
        "baseline": "1 · Baseline",
        "predictor": "2 · Predictor",
        "explained": "3 · Explained",
        "experiment": "4 · Experiment",
    }.get(key, key)


def render_understanding_r2_subtab(view) -> None:
    """Progressive R² lesson using the current fitting sample."""
    st.markdown(
        tipify('<p class="fit-r2-heading">Understanding R²</p>'),
        unsafe_allow_html=True,
    )
    ols = ols_fit(view.x, view.y)
    sst = total_sum_of_squares(view.y)
    y_bar = float(np.mean(view.y)) if view.n else float("nan")
    (a0, a1, a_step), (b0, b1, b_step) = view.slider_bounds()

    step = st.segmented_control(
        "R² lesson stage",
        options=list(R2_LESSON_STEPS),
        default="baseline",
        format_func=_r2_lesson_step_label,
        key="r2_lesson_step",
        label_visibility="collapsed",
        width="content",
    )
    if step not in R2_LESSON_STEPS:
        step = "baseline"

    # Experiment line starts from the Fit-the-line sliders when available.
    if "r2_exp_alpha" not in st.session_state:
        st.session_state.r2_exp_alpha = float(
            st.session_state.get("fit_alpha", ols.alpha)
        )
    if "r2_exp_beta" not in st.session_state:
        st.session_state.r2_exp_beta = float(
            st.session_state.get("fit_beta", ols.beta)
        )
    st.session_state.r2_exp_alpha = float(
        np.clip(st.session_state.r2_exp_alpha, a0, a1)
    )
    st.session_state.r2_exp_beta = float(
        np.clip(st.session_state.r2_exp_beta, b0, b1)
    )

    show_mean = True
    reg_alpha = reg_beta = None
    ols_alpha = ols_beta = None
    residual_about = None
    regression_name = "Regression"

    if step == "baseline":
        chart_title = "Baseline: horizontal prediction at ȳ"
        show_mean_resid = st.toggle(
            "Show residuals from the mean", key="r2_show_mean_resid", value=True
        )
        residual_about = "mean" if show_mean_resid else None
    elif step == "predictor":
        chart_title = "Add the predictor: regression vs mean"
        reg_alpha, reg_beta = float(ols.alpha), float(ols.beta)
        regression_name = "OLS regression"
        resid_mode = st.radio(
            "Residuals",
            options=["off", "mean", "regression"],
            format_func=lambda k: {
                "off": "Hide residuals",
                "mean": "Around the mean line",
                "regression": "Around the regression line",
            }[k],
            horizontal=True,
            key="r2_resid_about",
        )
        residual_about = None if resid_mode == "off" else resid_mode
    elif step == "explained":
        chart_title = "Variation explained by OLS"
        reg_alpha, reg_beta = float(ols.alpha), float(ols.beta)
        regression_name = "OLS regression"
        show_ols_resid = st.toggle(
            "Show residuals from OLS", key="r2_show_ols_resid", value=False
        )
        residual_about = "regression" if show_ols_resid else None
    else:
        chart_title = "Experiment: mean, your line, and OLS"
        exp_row = st.container(
            horizontal=True, vertical_alignment="bottom", gap="small", wrap=True
        )
        with exp_row:
            exp_alpha = st.slider(
                "Your intercept (α)",
                min_value=float(a0),
                max_value=float(a1),
                step=float(a_step),
                key="r2_exp_alpha",
                width=200,
            )
            exp_beta = st.slider(
                "Your slope (β)",
                min_value=float(b0),
                max_value=float(b1),
                step=float(b_step),
                key="r2_exp_beta",
                width=200,
            )
            st.button(
                "Snap to OLS",
                key="r2_snap_ols",
                width=120,
                help=tip_title("OLS"),
                on_click=_snap_r2_experiment_to_ols,
                args=(float(ols.alpha), float(ols.beta), float(a0), float(a1), float(b0), float(b1)),
            )
            show_exp_resid = st.toggle(
                "Show residuals from your line", key="r2_exp_resid"
            )
        reg_alpha, reg_beta = float(exp_alpha), float(exp_beta)
        regression_name = "Your line"
        ols_alpha, ols_beta = float(ols.alpha), float(ols.beta)
        residual_about = "regression" if show_exp_resid else None

    st.plotly_chart(
        r2_lesson_scatter(
            hours=view.x,
            scores=view.y,
            height=max(FIT_CHART_HEIGHT + 60, 460),
            x_title=view.x_axis,
            y_title=view.y_axis,
            title=chart_title,
            show_mean_line=show_mean,
            regression_alpha=reg_alpha,
            regression_beta=reg_beta,
            regression_name=regression_name,
            ols_alpha=ols_alpha,
            ols_beta=ols_beta,
            residual_about=residual_about,
        ),
        width="stretch",
        config=PLOTLY_CONFIG,
        key=f"r2_lesson_chart_{step}",
    )

    ols_sse = float(ols.sse)
    ols_ssr = float(sst) - ols_sse if np.isfinite(sst) else float("nan")
    ols_r2 = r_squared(ols_sse, sst)

    if step == "baseline":
        y_bar_label = f"{y_bar:.2f}" if np.isfinite(y_bar) else "—"
        st.markdown(
            tipify(
                '<div class="fit-r2-lesson">'
                "<h3>1. The baseline — predicting without a predictor</h3>"
                f"<p>Imagine we know nothing about each student’s "
                f"{view.x_axis.lower()}.</p>"
                f"<p>Our best prediction for every student is simply the average "
                f"{view.y_axis.lower()}: <b>ȳ = {y_bar_label}</b> "
                "(the dashed horizontal line).</p>"
                f"<p>But students don’t all score {y_bar_label}.</p>"
                "<p>The grey vertical lines show the difference between each student’s "
                "actual score and our prediction. These are our prediction errors "
                "(residuals from the mean).</p>"
                "<p>To measure total variation we <b>square</b> each residual and add them. "
                "Signed errors can cancel (e.g. +5 and −5); squaring prevents cancellation "
                "and penalises large misses more:</p>"
                "<p>Residual 5 → squared error 25 &nbsp;·&nbsp; "
                "Residual 10 → squared error 100</p>"
                f'<p class="formula">SS Total = Σ(yᵢ − ȳ)² = <b>{_format_sse(sst)}</b></p>'
                f"<p>SS Total is the squared-error benchmark when we predict everyone by ȳ.</p>"
                "<p><b>The question now becomes:</b><br>"
                f"Can knowing {view.x_axis.lower()} help us predict "
                f"their {view.y_axis.lower()} more accurately?</p>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    elif step == "predictor":
        eliminated = (
            float(sst) - float(ols_sse)
            if np.isfinite(sst) and np.isfinite(ols_sse)
            else float("nan")
        )
        st.markdown(
            tipify(
                '<div class="fit-r2-lesson">'
                "<h3>2. Add the predictor</h3>"
                "<p>Two ways to predict the same outcomes (both lines stay on the chart):</p>"
                f"<p><b>Mean baseline (dashed ȳ):</b> predict every "
                f"{view.y_axis.lower()} with the sample mean — ignore X. "
                f"Squared error = SS Total = <b>{_format_sse(sst)}</b>.</p>"
                f"<p><b>Regression line (solid):</b> predict with "
                f"ŷᵢ = α̂ + β̂ xᵢ = {ols.alpha:.3f} "
                f"{'−' if ols.beta < 0 else '+'} {abs(float(ols.beta)):.3f} × xᵢ. "
                f"Squared error left over = SSR = "
                f"<b>{_format_sse(ols_sse)}</b>.</p>"
                f'<p class="formula">Squared error eliminated = SS Total − SSR '
                f"= <b>{_format_sse(eliminated)}</b></p>"
                "<p>Toggle residuals above to compare gaps to the mean versus to the line.</p>"
                f'<div class="metrics"><span>SS Total = <b>{_format_sse(sst)}</b></span>'
                f"<span>SSR = <b>{_format_sse(ols_sse)}</b></span>"
                f"<span>SSR / SS Total = "
                f"<b>{(ols_sse / sst) if sst > 0 else float('nan'):.3f}</b></span></div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    elif step == "explained":
        explained_frac = (ols_ssr / sst) if sst > 0 and np.isfinite(ols_ssr) else float("nan")
        bar = _r2_variation_bar_html(explained_frac=explained_frac)
        st.markdown(
            tipify(
                '<div class="fit-r2-lesson">'
                "<h3>3. What did we explain?</h3>"
                "<p>The dashed mean line is the no-predictor baseline. The solid OLS line "
                "uses X. For OLS with an intercept, the squared errors decompose exactly:</p>"
                f'<p class="formula">SS Total = SS Model + SSR &nbsp;→&nbsp; '
                f"{_format_sse(sst)} = {_format_sse(ols_ssr)} + {_format_sse(ols_sse)}</p>"
                f'<p class="formula">SS Model = Σ(ŷᵢ − ȳ)² = <b>{_format_sse(ols_ssr)}</b></p>'
                "<p><b>SS Total</b> = squared error if we predict everyone by ȳ.&nbsp; "
                "<b>SS Model</b> = how much of that squared error the regression removes.&nbsp; "
                "<b>SSR</b> = squared error still left after using the line.</p>"
                f"{bar}"
                f'<p class="formula">R² = SS Model / SS Total = 1 − SSR / SS Total '
                f"= <b>{_format_r_squared(ols_r2)}</b></p>"
                "<p><b>Why this formula?</b> SSR / SS Total is the fraction of the "
                "baseline variation that remains unexplained. Subtracting it from 1 gives "
                "the fraction removed by the regression.</p>"
                '<p class="note">R² measures in-sample fit for this sample — not causation, and '
                "not a guarantee of accuracy on new data.</p>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    else:
        cand_sse = sum_of_squared_errors(view.x, view.y, reg_alpha, reg_beta)
        cand_r2 = r_squared(cand_sse, sst)
        cand_ssr = regression_sum_of_squares(view.x, view.y, reg_alpha, reg_beta)
        decomp_gap = abs((cand_ssr + cand_sse) - sst) if np.isfinite(sst) else float("nan")
        mean_sse = float(sst)  # predicting ȳ → SSR = SS Total
        mean_r2 = 0.0 if np.isfinite(sst) and sst > 0 else float("nan")
        warn = ""
        if np.isfinite(cand_r2) and cand_r2 < 0:
            warn = (
                '<p class="fit-r2-warn">Your line has <b>negative R²</b>: it predicts worse than '
                "simply using the sample mean. OLS never does this for a line with an intercept.</p>"
            )
        decomp_note = (
            '<p class="note">For a manually chosen line, report fit as '
            "<b>1 − SSR / SS Total</b>. Do <em>not</em> assume "
            "SS Total = SS Model + SSR — that identity holds for "
            f"OLS with an intercept (here the gap is {_format_sse(decomp_gap)}).</p>"
            if decomp_gap > 1e-4 * max(sst, 1.0)
            else (
                '<p class="note">Your line is close enough to OLS that '
                "SS Model + SSR ≈ SS Total. "
                "In general the exact decomposition is guaranteed only for OLS with an intercept.</p>"
            )
        )
        st.markdown(
            tipify(
                '<div class="fit-r2-lesson">'
                "<h3>4. Experiment on this sample</h3>"
                "<p>Compare three predictors of the same outcomes: the mean-only baseline, "
                "your adjustable line, and the OLS line. OLS maximises R² by minimising "
                "SSR.</p>"
                '<div class="fit-r2-compare">'
                '<div class="fit-r2-card">'
                '<p class="lab">Mean only</p>'
                f'<p class="val">SSR = <b>{_format_sse(mean_sse)}</b><br>'
                f"R² = <b>{_format_r_squared(mean_r2)}</b></p>"
                "</div>"
                '<div class="fit-r2-card">'
                '<p class="lab">Your line</p>'
                f'<p class="val">SSR = <b>{_format_sse(cand_sse)}</b><br>'
                f"R² = <b>{_format_r_squared(cand_r2)}</b><br>"
                f"<span style='color:#64748B'>SS Model* = {_format_sse(cand_ssr)}</span></p>"
                "</div>"
                '<div class="fit-r2-card best">'
                '<p class="lab">OLS (best R²)</p>'
                f'<p class="val">SSR = <b>{_format_sse(ols_sse)}</b><br>'
                f"R² = <b>{_format_r_squared(ols_r2)}</b><br>"
                f"<span style='color:#64748B'>SS Model = {_format_sse(ols_ssr)}</span></p>"
                "</div>"
                "</div>"
                '<div class="fit-r2-conclusion">'
                '<p class="q">Why does OLS produce the highest R²?</p>'
                "<p class=\"a\">OLS finds the line that minimises SSR. "
                "Since SS Total is fixed for a given sample, "
                "minimising SSR also maximises R².</p>"
                '<p class="formula">R² = 1 − SSR / SS Total</p>'
                '<p class="takeaway">OLS minimises SSR ⇔ OLS maximises R²</p>'
                "</div>"
                f'<div class="metrics"><span>SS Total = <b>{_format_sse(sst)}</b> '
                "(fixed for this sample)</span></div>"
                f"{decomp_note}{warn}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def _std_z_axis_range(values: np.ndarray) -> tuple[float, float, list[int]]:
    """Symmetric z-axis range with integer ticks centred on 0."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return -2.0, 2.0, [-2, -1, 0, 1, 2]
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    pad = max(0.2 * (hi - lo), 0.4)
    span = max(abs(lo - pad), abs(hi + pad), 1.0)
    r0, r1 = -span, span
    ticks = list(range(int(np.floor(r0)), int(np.ceil(r1)) + 1))
    if 0 not in ticks:
        ticks.append(0)
        ticks.sort()
    return r0, r1, ticks


def _std_marker_size_opacity(n: int, *, combined: bool) -> tuple[float, float]:
    """Smaller, fainter markers for large samples (especially the overlay)."""
    n = int(n)
    if combined:
        if n > 1500:
            return 3.0, 0.16
        if n > 500:
            return 4.0, 0.24
        if n > 150:
            return 5.0, 0.36
        return 7.0, 0.55
    if n > 1500:
        return 3.5, 0.22
    if n > 500:
        return 5.0, 0.35
    if n > 150:
        return 6.5, 0.48
    return 8.0, 0.72


def _std_rgba(hex_color: str, opacity: float) -> str:
    raw = hex_color.lstrip("#")
    if len(raw) != 6:
        return f"rgba(37, 99, 235, {opacity:.3f})"
    r, g, b = int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)
    return f"rgba({r}, {g}, {b}, {opacity:.3f})"


def _std_compact_scatter(
    *,
    x,
    y,
    alpha: float,
    beta: float,
    x_title: str,
    y_title: str,
    title: str,
    key: str,
    marker_color: str = BLUE,
    line_color: str | None = None,
) -> None:
    """Small teaching scatter for one predictor → outcome pair."""
    line_color = line_color or marker_color
    n = int(np.asarray(x).size)
    size, opacity = _std_marker_size_opacity(n, combined=False)
    fig = fitting_scatter(
        hours=x,
        scores=y,
        alpha=float(alpha),
        beta=float(beta),
        show_residuals=False,
        show_coefficients=False,
        height=168,
        x_title=x_title,
        y_title=y_title,
        title=title,
    )
    if fig.data:
        fig.data[0].marker.color = _std_rgba(marker_color, opacity)
        fig.data[0].marker.size = size
        fig.data[0].marker.line.width = 0
        if len(fig.data) > 1:
            fig.data[1].line.color = line_color
            fig.data[1].line.width = 3.4
    fig.update_layout(showlegend=False, margin=dict(l=40, r=6, t=24, b=32))
    fig.update_xaxes(tickmode="auto", nticks=5, showgrid=True)
    fig.update_yaxes(tickmode="auto", nticks=5, showgrid=True)
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=key)


def _std_dual_z_overlay(
    *,
    z_a,
    z_b,
    z_scores,
    ols_a,
    ols_b,
    label_a: str,
    label_b: str,
    key: str = "fit_std_z",
) -> None:
    """Overlay two different standardised regressions on shared z-axes."""
    za = np.asarray(z_a, dtype=np.float64)
    zb = np.asarray(z_b, dtype=np.float64)
    zy = np.asarray(z_scores, dtype=np.float64)
    n = int(za.size)
    size, opacity = _std_marker_size_opacity(n, combined=True)
    fig = go.Figure()
    fig.add_scatter(
        x=za,
        y=zy,
        mode="markers",
        name=label_a,
        marker=dict(size=size, color=_std_rgba(BLUE, opacity), line=dict(width=0)),
    )
    fig.add_scatter(
        x=zb,
        y=zy,
        mode="markers",
        name=label_b,
        marker=dict(size=size, color=_std_rgba(ORANGE, opacity), line=dict(width=0)),
    )
    for z_x, fit, name, color, dash in (
        (za, ols_a, f"OLS · {label_a}", BLUE, "solid"),
        (zb, ols_b, f"OLS · {label_b}", ORANGE, "dash"),
    ):
        if z_x.size == 0:
            continue
        x_line = np.array([float(np.min(z_x)), float(np.max(z_x))])
        fig.add_scatter(
            x=x_line,
            y=float(fit.alpha) + float(fit.beta) * x_line,
            mode="lines",
            name=name,
            line=dict(color=color, width=3.6, dash=dash),
        )
    z_all = np.concatenate([za, zb]) if zb.size else za
    zx0, zx1, zx_ticks = _std_z_axis_range(z_all)
    zy0, zy1, zy_ticks = _std_z_axis_range(zy)
    fig.update_xaxes(
        title_text="z(predictor)",
        range=[zx0, zx1],
        tickmode="array",
        tickvals=zx_ticks,
        ticktext=[str(t) for t in zx_ticks],
        showgrid=True,
        zeroline=True,
        zerolinewidth=1.4,
    )
    fig.update_yaxes(
        title_text="z(Exam score)",
        range=[zy0, zy1],
        tickmode="array",
        tickvals=zy_ticks,
        ticktext=[str(t) for t in zy_ticks],
        showgrid=True,
        zeroline=True,
        zerolinewidth=1.4,
    )
    fig.update_layout(
        title="Standardised: compare association strength",
        height=360,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.16,
            x=0,
            font=dict(size=10),
        ),
        margin=dict(l=48, r=8, t=26, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=key)


def _std_insight_name(key: str) -> tuple[str, str]:
    """Short display name and verb for the teaching insight sentence."""
    if key == "hours_studied":
        return "Study hours", "have"
    if key == "sleep_hours":
        return "Sleep", "has"
    return "Attendance", "has"


def _std_comparison_insight(
    *,
    key_a: str,
    key_b: str,
    star_a: float,
    star_b: float,
) -> str:
    """Dynamic sentence comparing the two standardised slopes."""
    name_a, verb_a = _std_insight_name(key_a)
    name_b, verb_b = _std_insight_name(key_b)
    a_stronger = abs(float(star_a)) >= abs(float(star_b))
    strong_name = name_a if a_stronger else name_b
    strong_verb = verb_a if a_stronger else verb_b
    star_hi = float(star_a) if a_stronger else float(star_b)
    star_lo = float(star_b) if a_stronger else float(star_a)
    return (
        f"{strong_name} {strong_verb} the stronger association "
        f"(β* = <b>{_format_coef(star_hi)}</b> vs <b>{_format_coef(star_lo)}</b>). "
        "Standardised coefficients are comparable because both predictors "
        "and the outcome are measured in standard-deviation units."
    )


def _std_ensure_predictor_keys() -> tuple[str, str]:
    """Return distinct Predictor A / B keys with defaults Hours / Attendance."""
    keys = list(STD_PREDICTOR_KEYS)
    if "fit_std_pred_a" not in st.session_state or st.session_state.fit_std_pred_a not in keys:
        st.session_state.fit_std_pred_a = "hours_studied"
    if "fit_std_pred_b" not in st.session_state or st.session_state.fit_std_pred_b not in keys:
        st.session_state.fit_std_pred_b = "attendance"
    if st.session_state.fit_std_pred_a == st.session_state.fit_std_pred_b:
        # Keep A; pick the first other option for B.
        for key in keys:
            if key != st.session_state.fit_std_pred_a:
                st.session_state.fit_std_pred_b = key
                break
    return st.session_state.fit_std_pred_a, st.session_state.fit_std_pred_b


def render_standardisation_subtab(view) -> None:
    """Compare two Students predictors of exam score in raw vs standardised units."""
    st.markdown(
        tipify('<p class="fit-r2-heading">Standardisation</p>'),
        unsafe_allow_html=True,
    )
    st.markdown(
        tipify(
            '<p class="fit-std-banner">'
            "Standardisation converts variables into a common scale, "
            "measured in standard deviations rather than their original "
            "units. This makes regression coefficients easier to compare "
            "across predictors."
            "</p>"
        ),
        unsafe_allow_html=True,
    )

    sample = standardisation_student_frame(view)
    if sample is None:
        st.info(
            "Standardisation compares predictors on the Students dataset. "
            "Switch the lab dataset to Students to use this lesson."
        )
        return

    _std_ensure_predictor_keys()
    label_of = {key: STD_PREDICTOR_META[key][0] for key in STD_PREDICTOR_KEYS}
    unit_of = {key: STD_PREDICTOR_META[key][1] for key in STD_PREDICTOR_KEYS}

    pick_a, pick_b = st.columns(2, gap="small")
    with pick_a:
        pred_a = st.selectbox(
            "Predictor A",
            options=list(STD_PREDICTOR_KEYS),
            format_func=lambda key: label_of[key],
            key="fit_std_pred_a",
        )
    options_b = [key for key in STD_PREDICTOR_KEYS if key != pred_a]
    if st.session_state.get("fit_std_pred_b") not in options_b:
        st.session_state.fit_std_pred_b = options_b[0]
    with pick_b:
        pred_b = st.selectbox(
            "Predictor B",
            options=options_b,
            format_func=lambda key: label_of[key],
            key="fit_std_pred_b",
        )

    scores = np.asarray(sample["exam_score"], dtype=np.float64)
    x_a = np.asarray(sample[pred_a], dtype=np.float64)
    x_b = np.asarray(sample[pred_b], dtype=np.float64)
    label_a, label_b = label_of[pred_a], label_of[pred_b]

    ols_a = ols_fit(x_a, scores)
    ols_b = ols_fit(x_b, scores)
    z_a, z_b, z_y = zscore(x_a), zscore(x_b), zscore(scores)
    ols_za = ols_fit(z_a, z_y)
    ols_zb = ols_fit(z_b, z_y)
    sx_a, sx_b, sy = sample_sd(x_a), sample_sd(x_b), sample_sd(scores)
    beta_star_a = standardised_slope(float(ols_a.beta), sx_a, sy)
    beta_star_b = standardised_slope(float(ols_b.beta), sx_b, sy)
    sst = total_sum_of_squares(scores)
    r2_a = r_squared(float(ols_a.sse), float(sst))
    r2_b = r_squared(float(ols_b.sse), float(sst))

    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown(
            '<p class="fit-std-col-lab">Original units</p>',
            unsafe_allow_html=True,
        )
        _std_compact_scatter(
            x=x_a,
            y=scores,
            alpha=float(ols_a.alpha),
            beta=float(ols_a.beta),
            x_title=label_a,
            y_title="Exam score",
            title=f"{label_a} → Exam score",
            key="fit_std_orig_a",
            marker_color=BLUE,
        )
        st.markdown(
            tipify(
                '<p class="fit-std-slope">'
                f"β̂ = <b>{_format_coef(ols_a.beta)}</b> {unit_of[pred_a]} "
                f"| R² = <b>{_format_r_squared(r2_a)}</b>"
                "</p>"
            ),
            unsafe_allow_html=True,
        )
        _std_compact_scatter(
            x=x_b,
            y=scores,
            alpha=float(ols_b.alpha),
            beta=float(ols_b.beta),
            x_title=label_b,
            y_title="Exam score",
            title=f"{label_b} → Exam score",
            key="fit_std_orig_b",
            marker_color=ORANGE,
        )
        st.markdown(
            tipify(
                '<p class="fit-std-slope">'
                f"β̂ = <b>{_format_coef(ols_b.beta)}</b> {unit_of[pred_b]} "
                f"| R² = <b>{_format_r_squared(r2_b)}</b>"
                "</p>"
            ),
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            '<p class="fit-std-col-lab">Standardised units</p>',
            unsafe_allow_html=True,
        )
        _std_dual_z_overlay(
            z_a=z_a,
            z_b=z_b,
            z_scores=z_y,
            ols_a=ols_za,
            ols_b=ols_zb,
            label_a=label_a,
            label_b=label_b,
            key=f"fit_std_z_{pred_a}_{pred_b}",
        )
        st.markdown(
            tipify(
                '<p class="fit-std-slope" style="margin-top:0.05rem;">'
                "Each slope now measures the predicted change in "
                "exam score (SD units) for a 1 SD increase in its predictor."
                "</p>"
                '<p class="fit-std-slope">'
                f"<b>{label_a}</b> · β* = <b>{_format_coef(float(ols_za.beta))}</b> "
                f"| R² = <b>{_format_r_squared(r2_a)}</b><br>"
                f"<b>{label_b}</b> · β* = <b>{_format_coef(float(ols_zb.beta))}</b> "
                f"| R² = <b>{_format_r_squared(r2_b)}</b>"
                "</p>"
            ),
            unsafe_allow_html=True,
        )

    insight = _std_comparison_insight(
        key_a=pred_a,
        key_b=pred_b,
        star_a=float(ols_za.beta),
        star_b=float(ols_zb.beta),
    )
    st.markdown(
        tipify(f'<p class="fit-std-insight">{insight}</p>'),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="fit-std-note">Standardisation changes coefficient values, not R².</p>',
        unsafe_allow_html=True,
    )

    # Concrete observation for the z-score walkthrough (near the sample median).
    x_bar_a = float(np.mean(x_a))
    example_i = int(np.argmin(np.abs(x_a - float(np.median(x_a)))))
    x_obs = float(x_a[example_i])
    z_obs = float(z_a[example_i])

    form_l, form_r = st.columns(2, gap="small")
    with form_l:
        with st.container(border=False):
            st.markdown(
                '<div class="fit-std-formula-pane">'
                '<p class="fit-std-col-lab">Standardising a value</p>'
                "</div>",
                unsafe_allow_html=True,
            )
            st.latex(r"z = \dfrac{x - \overline{x}}{\mathrm{SD}(X)}")
            st.latex(
                rf"z = \dfrac{{{_format_coef(x_obs)} - {_format_coef(x_bar_a)}}}"
                rf"{{{_format_coef(sx_a)}}} = {_format_coef(z_obs)}"
            )
            st.caption(f"One observation from {label_a} in the current sample.")
    with form_r:
        with st.container(border=False):
            st.markdown(
                '<div class="fit-std-formula-pane">'
                '<p class="fit-std-col-lab">Standardising a regression slope</p>'
                "</div>",
                unsafe_allow_html=True,
            )
            st.latex(
                r"\beta^{*} = \hat{\beta}\,\dfrac{\mathrm{SD}(X)}{\mathrm{SD}(Y)}"
            )
            st.latex(
                rf"\beta^{{*}} = {_format_coef(ols_a.beta)}\times"
                rf"\dfrac{{{_format_coef(sx_a)}}}{{{_format_coef(sy)}}} ="
                rf" {_format_coef(beta_star_a)}"
            )
            st.caption(f"Using {label_a} → Exam score.")

    st.markdown(
        tipify(
            '<p class="fit-std-note">'
            "Changes: units, intercept and slope. "
            "Unchanged: R², fitted relationship and predictions "
            "after conversion back to original units."
            "</p>"
        ),
        unsafe_allow_html=True,
    )

    with st.expander("Two methods & R code", expanded=False):
        st.caption(
            f"Both methods use {label_a} and yield β* = {_format_coef(beta_star_a)}."
        )
        st.code(
            r_standardisation_commands(predictor=pred_a, response="exam_score"),
            language="r",
        )


def _format_coef(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{float(value):.3f}"


def render_fitting_tab() -> None:
    # One-time reset so older sessions drop the previous 10% / penguins defaults.
    if not st.session_state.get("fit_defaults_students_2pct"):
        st.session_state.fit_dataset_key = "students"
        st.session_state.fit_pct = DEFAULT_SAMPLE_PCT
        st.session_state.fit_show_residuals = False
        st.session_state.fit_use_all = False
        st.session_state.fit_defaults_students_2pct = True
        st.session_state.pop("fit_source_key", None)
        _reset_fitting_sample_state()
        _save_fitting_persist()
    # Dialog widgets are wiped when the Dataset drawer closes; restore first.
    _restore_fitting_persist()
    if "fit_poor_index" not in st.session_state:
        st.session_state.fit_poor_index = 0
    if "fit_selected" not in st.session_state:
        st.session_state.fit_selected = None
    if "fit_found" not in st.session_state:
        st.session_state.fit_found = False
    if "fit_show_residuals" not in st.session_state:
        st.session_state.fit_show_residuals = False
    if "fit_show_coefs" not in st.session_state:
        st.session_state.fit_show_coefs = False
    if "fit_draw" not in st.session_state:
        st.session_state.fit_draw = 0
    if "fit_dataset_key" not in st.session_state:
        st.session_state.fit_dataset_key = "students"
        _save_fitting_persist()
    if "fit_seed" not in st.session_state:
        st.session_state.fit_seed = 20261018
    if "fit_use_all" not in st.session_state:
        st.session_state.fit_use_all = False
    if "fit_pct" not in st.session_state:
        st.session_state.fit_pct = DEFAULT_SAMPLE_PCT

    with st.container(key="fit_shell"):
        ensure_registry_state(st.session_state)
        try:
            _ensure_fitting_xy(st.session_state.fit_dataset_key)
        except KeyError:
            # Upload purged or unknown key — fall back to students.
            st.session_state.fit_dataset_key = "students"
            _ensure_fitting_xy("students")
        _save_fitting_persist()

        with st.expander(
            "What is linear regression?",
            expanded=False,
            type="compact",
            key="fit_slr_intro",
        ):
            st.markdown(
                '<div class="fit-slr-intro">'
                '<div class="cols">'
                '<div class="col"><p class="lab">Simple</p>'
                "<p class=\"txt\">One predictor (X) to predict an outcome (Y).</p></div>"
                '<div class="col"><p class="lab">Linear</p>'
                "<p class=\"txt\">A straight-line relationship between X and Y.</p></div>"
                '<div class="col"><p class="lab">Regression</p>'
                "<p class=\"txt\">Use observed data to estimate the best-fitting line.</p></div>"
                "</div>"
                '<p class="eq">Ŷ = β̂<sub>0</sub> + β̂<sub>1</sub>X</p>'
                '<p class="cap">One predictor. One straight line. Two estimated coefficients.</p>'
                "</div>",
                unsafe_allow_html=True,
            )

        _render_dataset_status_row(_fitting_dataset_status_html())

        dataset_key = st.session_state.fit_dataset_key
        x_name, y_name = _ensure_fitting_xy(dataset_key)
        pair_key = (dataset_key, x_name, y_name)
        if st.session_state.get("fit_pair_key") != pair_key:
            st.session_state.fit_pair_key = pair_key
            _reset_fitting_sample_state()
            st.rerun()

        source = resolve_fitting_dataset(
            dataset_key,
            x_name=x_name,
            y_name=y_name,
            session=st.session_state,
            seed=st.session_state.fit_seed,
        )
        _render_fitting_lab_sample_controls(source)

        view = _ensure_fitting_observations(source)
        fingerprint_before = st.session_state.get("fit_view_fingerprint")

        ols = ols_fit(view.x, view.y)
        (a0, a1, a_step), (b0, b1, b_step) = view.slider_bounds()

        preserve_hats = bool(st.session_state.pop("fit_preserve_hats", False))
        has_hats = (
            ("fit_alpha" in st.session_state or "fit_pending_alpha" in st.session_state)
            and ("fit_beta" in st.session_state or "fit_pending_beta" in st.session_state)
        )
        needs_poor_start = (
            not has_hats
            or st.session_state.get("fit_line_token") != _fitting_view_token(source)
            or not st.session_state.get("fit_bootstrapped", False)
        )
        if preserve_hats and has_hats:
            # New sample, same candidate line — only refresh the token and selection.
            st.session_state.fit_line_token = _fitting_view_token(source)
            st.session_state.fit_bootstrapped = True
            _clear_fitting_student_selection()
        elif needs_poor_start:
            _reset_fitting_line(view, away_from_ols=True)
            st.session_state.fit_bootstrapped = True

        # Apply queued slider values before instantiating the widgets.
        _apply_pending_fitting_sliders()
        if "fit_alpha" in st.session_state:
            st.session_state.fit_alpha = float(np.clip(st.session_state.fit_alpha, a0, a1))
        if "fit_beta" in st.session_state:
            st.session_state.fit_beta = float(np.clip(st.session_state.fit_beta, b0, b1))

        # Drop legacy number-input default if an old session still carries it.
        if "fit_predict_x" in st.session_state and "fit_predict_input" not in st.session_state:
            st.session_state.pop("fit_predict_x", None)
        _sync_fitting_prediction_active()

        # Lab tabs sit immediately above regression controls / lesson content.
        with st.container(key="fit_lab_tabs"):
            if "fit_lab_panel" not in st.session_state:
                st.session_state.fit_lab_panel = "fit"
            lab_panel = st.segmented_control(
                "Lab panel",
                options=["fit", "r2", "std"],
                format_func=lambda key: {
                    "fit": "Fit the line",
                    "r2": "How Good Is the Model?",
                    "std": "Standardisation",
                }.get(key, key),
                key="fit_lab_panel",
                label_visibility="collapsed",
                width="content",
            )
        if lab_panel not in ("fit", "r2", "std"):
            lab_panel = "fit"

        if lab_panel == "r2":
            render_understanding_r2_subtab(view)
        elif lab_panel == "std":
            render_standardisation_subtab(view)
        else:
            _render_fitting_line_subtab(
                view,
                ols=ols,
                a0=a0,
                a1=a1,
                a_step=a_step,
                b0=b0,
                b1=b1,
                b_step=b_step,
                fingerprint_before=fingerprint_before,
            )


@st.fragment(key="fit_line_panel")
def _render_fitting_line_subtab(
    view,
    *,
    ols,
    a0: float,
    a1: float,
    a_step: float,
    b0: float,
    b1: float,
    b_step: float,
    fingerprint_before,
) -> None:
    # Must run inside the fragment: clearing Predict X only reruns this panel.
    _sync_fitting_prediction_active()
    animating = bool(st.session_state.get("fit_animating"))
    anim_steps = st.session_state.get("fit_anim_steps") or []
    anim_frame = int(st.session_state.get("fit_anim_frame", 0))
    if animating and anim_steps and 0 <= anim_frame < len(anim_steps):
        step_alpha, step_beta = anim_steps[anim_frame]
        # Drive keyed sliders from session state before they mount (smooth per-frame).
        st.session_state.fit_alpha = float(np.clip(step_alpha, a0, a1))
        st.session_state.fit_beta = float(np.clip(step_beta, b0, b1))

    if "fit_view_mode" not in st.session_state:
        st.session_state.fit_view_mode = "chart"

    label_alpha = float(st.session_state.get("fit_alpha", ols.alpha))
    label_beta = float(st.session_state.get("fit_beta", ols.beta))
    show_ols_labels = (
        bool(st.session_state.get("fit_found"))
        and not animating
        and _hats_match_ols(
            label_alpha, label_beta, ols, a_step=a_step, b_step=b_step
        )
    )
    alpha_desc = "(OLS intercept)" if show_ols_labels else "(Your intercept)"
    beta_desc = "(OLS slope)" if show_ols_labels else "(Your slope)"

    # One toolbar: controls are siblings of the SAME st.columns().
    with st.container(key="fit_toolbar"):
        (
            col_run,
            col_resid,
            col_coefs,
            col_rcode,
            col_alpha,
            col_beta,
            col_metrics,
            col_view,
        ) = st.columns(
            [0.95, 0.75, 0.75, 0.85, 1.2, 1.2, 1.9, 0.7],
            gap="small",
            vertical_alignment="center",
        )
        with col_run:
            find = st.button(
                "Fit OLS",
                type="primary",
                key="fit_find",
                width=75,
                help=(
                    "Fit the OLS regression with R's lm(). "
                    f"{tip_title('OLS')}"
                ),
                disabled=animating,
            )
        with col_resid:
            with st.container(key="fit_resid_wrap"):
                st.markdown(
                    '<p class="fit-toggle-lab">Residuals</p>',
                    unsafe_allow_html=True,
                )
                show_residuals = st.toggle(
                    "Residuals",
                    key="fit_show_residuals",
                    label_visibility="collapsed",
                )
        with col_coefs:
            with st.container(key="fit_coefs_wrap"):
                st.markdown(
                    '<p class="fit-toggle-lab">α̂ and β̂</p>',
                    unsafe_allow_html=True,
                )
                show_coefs = st.toggle(
                    "α̂ and β̂",
                    key="fit_show_coefs",
                    label_visibility="collapsed",
                )
        with col_rcode:
            with st.container(key="fit_rcode_wrap"):
                with st.popover("R code", width=85):
                    _render_r_code_panel(
                        view,
                        ols,
                        alpha=float(st.session_state.get("fit_alpha", ols.alpha)),
                        beta=float(st.session_state.get("fit_beta", ols.beta)),
                        lm_run=bool(st.session_state.fit_found),
                    )
        with col_alpha:
            with st.container(key="fit_alpha_wrap"):
                st.markdown(
                    f'<p class="fit-coef-lab"><span class="sym">{_ALPHA_HAT}</span>'
                    f'<span class="desc">{alpha_desc}</span></p>',
                    unsafe_allow_html=True,
                )
                alpha = st.slider(
                    "Your intercept",
                    min_value=float(a0),
                    max_value=float(a1),
                    step=float(a_step),
                    key="fit_alpha",
                    width=100,
                    disabled=animating,
                    label_visibility="collapsed",
                )
        with col_beta:
            with st.container(key="fit_beta_wrap"):
                st.markdown(
                    f'<p class="fit-coef-lab"><span class="sym">{_BETA_HAT}</span>'
                    f'<span class="desc">{beta_desc}</span></p>',
                    unsafe_allow_html=True,
                )
                beta = st.slider(
                    "Your slope",
                    min_value=float(b0),
                    max_value=float(b1),
                    step=float(b_step),
                    key="fit_beta",
                    width=100,
                    disabled=animating,
                    label_visibility="collapsed",
                )

        sst = total_sum_of_squares(view.y)
        if find and not animating:
            st.session_state.fit_sse_before = float(
                sum_of_squared_errors(view.x, view.y, alpha, beta)
            )
            steps = animation_steps(
                float(alpha), float(beta), float(ols.alpha), float(ols.beta), count=18
            )
            sse_path, r2_path = animation_metric_trajectories(
                view.x, view.y, steps, float(sst)
            )
            st.session_state.fit_anim_steps = steps
            st.session_state.fit_sse_path = np.asarray(sse_path, dtype=np.float64)
            st.session_state.fit_r2_path = np.asarray(r2_path, dtype=np.float64)
            st.session_state.fit_sse_range = _metric_axis_range(sse_path)
            st.session_state.fit_r2_range = _metric_axis_range(r2_path)
            st.session_state.fit_anim_frame = 0
            st.session_state.fit_animating = True
            st.session_state.fit_view_mode = "chart"
            # Continue this run as frame 0 (avoid an extra full remount flash).
            animating = True
            anim_steps = steps
            anim_frame = 0

        current_sse = sum_of_squared_errors(view.x, view.y, alpha, beta)
        current_r2 = r_squared(current_sse, sst)

        path_sse = st.session_state.get("fit_sse_path")
        path_r2 = st.session_state.get("fit_r2_path")
        if path_sse is not None and path_r2 is not None:
            sse_path = np.asarray(path_sse, dtype=np.float64)
            r2_path = np.asarray(path_r2, dtype=np.float64)
        else:
            sse_path = np.asarray([], dtype=np.float64)
            r2_path = np.asarray([], dtype=np.float64)
        has_metric_path = int(sse_path.size) >= 2 and int(r2_path.size) >= 2
        at_ols = _hats_match_ols(
            float(alpha), float(beta), ols, a_step=a_step, b_step=b_step
        )

        if has_metric_path:
            sse_range = st.session_state.get("fit_sse_range") or _metric_axis_range(
                sse_path
            )
            r2_range = st.session_state.get("fit_r2_range") or _metric_axis_range(
                r2_path
            )
            paint_x_max = max(int(sse_path.size) - 1, 0)
            r2_start = float(r2_path[0])
            r2_end = float(r2_path[-1])
        else:
            sse_range = _FIT_METRIC_EMPTY_RANGE
            r2_range = _FIT_METRIC_EMPTY_RANGE
            paint_x_max = 0
            r2_start = 0.0
            r2_end = 0.0

        # Show curves while animating, or when hats match OLS (restore completed fit).
        # Off-OLS hats: blank minis (paths kept so returning to OLS restores them).
        if animating and has_metric_path and 0 <= anim_frame < sse_path.size:
            paint_sse = sse_path[: anim_frame + 1]
            paint_r2 = r2_path[: anim_frame + 1]
            show_r2_range = False
        elif has_metric_path and at_ols:
            paint_sse = sse_path
            paint_r2 = r2_path
            show_r2_range = True
        else:
            paint_sse = np.asarray([], dtype=np.float64)
            paint_r2 = np.asarray([], dtype=np.float64)
            paint_x_max = 0
            sse_range = _FIT_METRIC_EMPTY_RANGE
            r2_range = _FIT_METRIC_EMPTY_RANGE
            show_r2_range = False

        _mount_fit_metric_toolbar(
            col_metrics,
            sse_values=paint_sse,
            r2_values=paint_r2,
            sse_range=sse_range,
            r2_range=r2_range,
            x_max=paint_x_max,
            r2_start=r2_start,
            r2_end=r2_end,
            show_r2_range=show_r2_range,
            animating=False,
        )
        view_mode = st.session_state.get("fit_view_mode", "chart")
        if view_mode not in ("chart", "table"):
            view_mode = "chart"
            st.session_state.fit_view_mode = "chart"
        with col_view:
            with st.container(key="fit_view_wrap"):
                toggle_label = "Table" if view_mode == "chart" else "Chart"
                st.button(
                    toggle_label,
                    type="secondary",
                    key="fit_view_toggle",
                    width=65,
                    disabled=animating,
                    help="Switch between chart and table view",
                    on_click=_toggle_fit_view_mode,
                )

    predict_x = _active_fitting_prediction_x()
    y_bar = float(np.mean(view.y)) if view.n else float("nan")

    # Sync EXPLAIN dropdown → selection before the chart/table paint (0 = none).
    if "fit_explain_pick" not in st.session_state:
        st.session_state.fit_explain_pick = 0
    pick = int(st.session_state.fit_explain_pick or 0)
    if view.n <= 0 or pick < 0 or pick > view.n:
        pick = 0
        st.session_state.fit_explain_pick = 0
    if pick == 0:
        st.session_state.fit_selected = None
    else:
        st.session_state.fit_selected = pick - 1

    with st.container(key="fit_view_panel"):
        # Per-frame lm() animation: session-state hats drive the sliders each rerun.
        if animating:
            if fingerprint_before is not None:
                assert st.session_state.get("fit_view_fingerprint") == fingerprint_before
            steps = anim_steps
            if not steps:
                _clear_fitting_animation()
                _fit_fragment_rerun()
            frame = max(0, min(anim_frame, len(steps) - 1))
            step_alpha, step_beta = steps[frame]
            st.plotly_chart(
                fitting_scatter(
                    hours=view.x,
                    scores=view.y,
                    alpha=float(step_alpha),
                    beta=float(step_beta),
                    show_residuals=show_residuals,
                    selected_index=st.session_state.fit_selected,
                    height=FIT_CHART_HEIGHT,
                    x_title=view.x_axis,
                    y_title=view.y_axis,
                    show_coefficients=show_coefs,
                    prediction_x=predict_x,
                    prediction_alpha=(
                        float(step_alpha) if predict_x is not None else None
                    ),
                    prediction_beta=(
                        float(step_beta) if predict_x is not None else None
                    ),
                ),
                width="stretch",
                config=PLOTLY_CONFIG,
                key="fit_scatter_anim",
            )
            if frame < len(steps) - 1:
                time.sleep(0.12)
                st.session_state.fit_anim_frame = frame + 1
                _fit_fragment_rerun()
            _queue_fitting_sliders(ols.alpha, ols.beta)
            st.session_state.fit_found = True
            _clear_fitting_animation()
            _fit_fragment_rerun()

        if view_mode == "table":
            table = _fitting_residuals_table(view, alpha, beta)
            styled = _style_selected_student(table, st.session_state.fit_selected)
            table_event = st.dataframe(
                styled,
                width="stretch",
                hide_index=True,
                height=FIT_CHART_HEIGHT,
                key="fit_resid_table",
                on_select="rerun",
                selection_mode="single-row",
                column_config={
                    "Observation": st.column_config.NumberColumn(
                        "Observation", format="%d"
                    ),
                    "X": st.column_config.NumberColumn("X", format="%.3f"),
                    "Y": st.column_config.NumberColumn("Y", format="%.3f"),
                    "Ŷ": st.column_config.NumberColumn("Ŷ", format="%.3f"),
                    "Residual (Y−Ŷ)": st.column_config.NumberColumn(
                        "Residual (Y−Ŷ)", format="%+.3f"
                    ),
                    "Residual²": st.column_config.NumberColumn("Residual²", format="%.3f"),
                },
            )
            prev_selected = st.session_state.fit_selected
            st.session_state.fit_selected = _selected_dataframe_row(table_event)
            if (
                st.session_state.fit_selected is not None
                and st.session_state.fit_selected != prev_selected
            ):
                pick = int(st.session_state.fit_selected) + 1
                st.session_state.fit_explain_pick = pick
                st.session_state.fit_explain_pick_seen = pick
                st.session_state.fit_applied_chart_idx = int(st.session_state.fit_selected)
                _bump_fitting_scatter_nonce()
                _fit_fragment_rerun()
            st.markdown(
                tipify(
                    f'<p class="fit-table-sse">Total &nbsp;|&nbsp; '
                    f"SSR = Σ Residual² = <b>{_format_sse(current_sse)}</b></p>"
                ),
                unsafe_allow_html=True,
            )
        else:
            if "fit_scatter_nonce" not in st.session_state:
                st.session_state.fit_scatter_nonce = 0
            if "fit_applied_chart_idx" not in st.session_state:
                st.session_state.fit_applied_chart_idx = None
            # Coordinates for click→index fallback inside the on_select callback.
            st.session_state.fit_chart_x = np.asarray(view.x, dtype=np.float64)
            st.session_state.fit_chart_y = np.asarray(view.y, dtype=np.float64)
            event = st.plotly_chart(
                fitting_scatter(
                    hours=view.x,
                    scores=view.y,
                    alpha=alpha,
                    beta=beta,
                    show_residuals=show_residuals,
                    selected_index=st.session_state.fit_selected,
                    height=FIT_CHART_HEIGHT,
                    x_title=view.x_axis,
                    y_title=view.y_axis,
                    show_coefficients=show_coefs,
                    prediction_x=predict_x,
                    prediction_alpha=float(alpha) if predict_x is not None else None,
                    prediction_beta=float(beta) if predict_x is not None else None,
                ),
                width="stretch",
                config=PLOTLY_CONFIG,
                key=f"fit_scatter_{int(st.session_state.fit_scatter_nonce)}",
                on_select=_on_fit_scatter_select,
                selection_mode="points",
            )
            # Backup if the callback did not run (or selection arrived only here).
            if _apply_fitting_chart_selection(
                _observation_index_from_chart_event(event), n=int(view.n)
            ):
                _fit_fragment_rerun()

    selected = st.session_state.fit_selected
    if selected is not None and 0 <= int(selected) < view.n:
        st.session_state.fit_explain_pick = int(selected) + 1
    else:
        st.session_state.fit_selected = None
        st.session_state.fit_explain_pick = 0

    with st.container(key="fit_model_use"):
        st.markdown(
            '<p class="fit-use-title">Using the model</p>',
            unsafe_allow_html=True,
        )
        predict_col, explain_col = st.columns(2, gap="small")
        with predict_col:
            with st.container(key="fit_use_predict"):
                st.markdown(
                    '<p class="fit-use-lab predict">Predict — What outcome can I expect?</p>',
                    unsafe_allow_html=True,
                )
                pred_row = st.container(
                    horizontal=True, vertical_alignment="center", gap="small", wrap=True
                )
                with pred_row:
                    st.markdown(
                        f'<p class="fit-predict-key">X: {view.x_axis}</p>',
                        unsafe_allow_html=True,
                    )
                    if "fit_predict_input" not in st.session_state:
                        st.session_state.fit_predict_input = ""
                    st.text_input(
                        view.x_axis,
                        placeholder="Enter X",
                        key="fit_predict_input",
                        label_visibility="collapsed",
                        width=120,
                    )
                    st.button(
                        "Predict",
                        key="fit_predict_btn",
                        type="primary",
                        width=88,
                        on_click=_commit_fitting_prediction,
                    )
                if predict_x is not None:
                    predict_y = float(alpha) + float(beta) * float(predict_x)
                    sign = "−" if beta < 0 else "+"
                    slope_abs = abs(float(beta))
                    x_obs = np.asarray(view.x, dtype=np.float64)
                    extrapolating = bool(x_obs.size) and (
                        float(predict_x) < float(np.min(x_obs)) - 1e-12
                        or float(predict_x) > float(np.max(x_obs)) + 1e-12
                    )
                    warn = (
                        '<p class="fit-predict-warn">'
                        "Extrapolation: this prediction is outside the observed data. "
                        "The relationship may not continue."
                        "</p>"
                        if extrapolating
                        else ""
                    )
                    st.markdown(
                        '<div class="fit-predict-result">'
                        f'<p class="fit-predict-yhat">Ŷ: Predicted {view.y_axis.lower()} '
                        f"= <b>{predict_y:.2f}</b></p>"
                        f'<p class="fit-predict-eq">'
                        f"Ŷ = {alpha:.3f} {sign} {slope_abs:.3f} × {predict_x:.3f} "
                        f"= <b>{predict_y:.2f}</b>"
                        "</p>"
                        f"{warn}"
                        "</div>",
                        unsafe_allow_html=True,
                    )
        with explain_col:
            with st.container(key="fit_use_explain"):
                st.markdown(
                    '<p class="fit-use-lab explain">Explain</p>',
                    unsafe_allow_html=True,
                )
                if view.n <= 0:
                    st.markdown(
                        '<p class="fit-explain-copy">Draw a sample to explain an observation.</p>',
                        unsafe_allow_html=True,
                    )
                elif (
                    selected is None
                    or not (0 <= int(selected) < view.n)
                ):
                    st.markdown(
                        '<p class="fit-explain-copy">'
                        "Click any point to see how the model explains its outcome."
                        "</p>",
                        unsafe_allow_html=True,
                    )
                else:
                    idx = int(selected)
                    obs_n = idx + 1
                    pick_row = st.container(
                        horizontal=True, vertical_alignment="center", gap="small"
                    )
                    with pick_row:
                        st.markdown(
                            f'<p class="fit-explain-student">Observation {obs_n}</p>',
                            unsafe_allow_html=True,
                        )
                        st.button(
                            "×",
                            key="fit_clear_student",
                            width=40,
                            help="Clear observation selection",
                            on_click=_clear_fitting_student_selection,
                        )
                    y_i = float(view.y[idx])
                    y_hat_i = float(alpha) + float(beta) * float(view.x[idx])
                    total_dev = y_i - y_bar
                    model_comp = y_hat_i - y_bar
                    resid_i = y_i - y_hat_i
                    st.markdown(
                        '<div class="fit-explain-vals">'
                        f'<span class="y"><span class="k">Actual Y</span>'
                        f'<span class="v">{y_i:.2f}</span></span>'
                        f'<span class="yhat"><span class="k">Predicted Ŷ</span>'
                        f'<span class="v">{y_hat_i:.2f}</span></span>'
                        f'<span class="mean"><span class="k">Sample mean Ȳ</span>'
                        f'<span class="v">{y_bar:.2f}</span></span>'
                        "</div>"
                        '<div class="fit-explain-decomp">'
                        '<p class="words">'
                        '<span class="total">Total deviation</span> = '
                        '<span class="model">Model component (blue)</span> + '
                        '<span class="resid">Residual (orange)</span>'
                        "</p>"
                        f'<p class="nums">'
                        f'<span class="total">{total_dev:+.2f}</span>'
                        f'<span class="op">=</span>'
                        f'<span class="model">{model_comp:+.2f}</span>'
                        f'<span class="op">+</span>'
                        f'<span class="resid">{resid_i:+.2f}</span>'
                        "</p>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                with st.container(key="fit_explain_foot"):
                    foot = st.container(
                        horizontal=True,
                        vertical_alignment="center",
                        gap="small",
                        wrap=True,
                    )
                    with foot:
                        if not np.isfinite(current_r2):
                            st.markdown(
                                tipify(
                                    '<p class="fit-explain-copy">R² undefined '
                                    "(no variation in Y).</p>"
                                ),
                                unsafe_allow_html=True,
                            )
                        else:
                            pct = 100.0 * float(current_r2)
                            st.markdown(
                                tipify(
                                    f'<p class="fit-explain-r2line">'
                                    f"R² = {pct:.1f}%</p>"
                                ),
                                unsafe_allow_html=True,
                            )
                            if current_r2 < 0:
                                st.markdown(
                                    tipify(
                                        '<p class="fit-explain-warn">Worse than the sample mean.</p>'
                                    ),
                                    unsafe_allow_html=True,
                                )
                        st.button(
                            "Understand R² →",
                            key="fit_open_r2",
                            width=140,
                            help=tip_title("R²"),
                            on_click=_open_understanding_r2,
                        )

def main() -> None:
    st.set_page_config(page_title="Regression Lab", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    from src.birthday_ines import mount_birthday_surprise

    mount_birthday_surprise()
    init_results()
    seed = render_intro()
    rng = session_rng(seed)

    try:
        population = load_population()
        metadata = load_metadata()
    except FileNotFoundError as exc:
        st.error(f"Could not load the population files. {exc}")
        st.stop()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    if not coefficients_match_metadata(population, metadata, atol=1e-3):
        st.warning(
            "The least-squares line computed from the student file does not "
            "match population_metadata.json. Sampling still uses the line "
            "computed from the file."
        )
    elif population.sha256 != metadata.sha256:
        st.warning("The student file hash does not match the metadata. The file may have changed.")

    active = render_main_nav()
    if active == "fitting":
        _render_lab_with_drawer("fitting", population, render_fitting_tab, rng=rng)
    elif active == "sampling":
        render_sampling_inference_tab(population, rng)
    elif active == "library":
        render_dataset_library_tab()


main()
