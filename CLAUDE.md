# CLAUDE.md — StatsApp Project Memory

## 1. Project Overview
- StatsApp: interactive statistics learning app teaching university-level
  regression and statistical inference.
- Stack: Python, Streamlit, Plotly (3D charts for multiple regression).
- Central pedagogical structure: **The Statistical Journey** homepage
  (`src/statistical_journey.py`), a 3×3 card grid that is also the site map.
- Deployed on Streamlit Community Cloud from GitHub `main`
  (`jelkhoury123/statsapp`); redeploy picks up a push automatically but can lag
  — check the production URL, not just `git push`, before assuming it's live.

## 2. Pedagogical Principles
- Every lesson follows logically from the one before it.
- Explain WHY a concept is needed before HOW it works.
- Lesson shape: pose a question → build intuition interactively → formalize
  the math → conclude with the insight that motivates the next lesson.
- Favor conceptual understanding over formula-dumping; stay statistically
  rigorous; follow the course materials where relevant.

## 3. Design Principles
- Consistent visual language across lessons (shared CSS classes, color
  phases, card chrome in `app.py`'s `<style>` block).
- Spacious, readable layouts; charts large enough to teach from, not
  decorative thumbnails.
- Minimize scrolling and visual clutter.
- Preserve the Journey's 3×3 grid, its zigzag progress arrows (row 1 L→R,
  row 2 R→L, row 3 L→R), and the two-level nav (parent "The Statistical
  Journey" button above the lesson row).
- Design for future expansion (more lesson/placeholder cards).

## 4. Current Project Status

**Architecture**
- `app.py` — single large Streamlit entry point: page CSS, main nav
  (`render_main_nav`, two-level: journey parent + lesson children), tab
  dispatch in `main()`, and the Simple Linear Regression / Sampling &
  Standard Error / t-test lesson bodies.
- `src/statistical_journey.py` — homepage: `STAGE_META`, `PLACEHOLDER_META`,
  `GRID_ORDER`/`VISUAL_ROWS`, mini illustrations, flip-card front/back,
  `open_lesson()` navigation.
- `src/multiple_regression.py` — Multiple Regression lesson (new).
- `src/correlation_causation.py` — Correlation & Causation lesson; also
  exports `fit_multiple()`, the generic OLS-with-intercept helper reused by
  Multiple Regression.
- `src/population.py` / `src/sampling.py` — fixed 100k-student population
  (CSV-backed) and simple-random-sample drawing; never rebuilt/rewritten.
- `src/sampling_distributions.py`, `src/null_slope.py` — sampling
  distributions and the t-test.
- `src/fitting_line.py`, `src/regression.py`, `src/charts.py` — OLS math and
  shared Plotly styling (`BLUE`, `ORANGE`, `GREY`, `INK`, `FONT`, …).

**Implemented lessons / subtabs**
- Simple Linear Regression: Why a Model? · Fit the line · Using the Model ·
  How Good Is the Model? · Standardisation · Understanding R Output.
- Multiple Regression (new): Why & The Equation · Fit the Plane ·
  Interpreting Coefficients · Model Fit. Uses real population columns
  `hours_studied` + `attendance` → `exam_score`; no t-test/F-test content.
- Correlation & Causation: Understanding correlation · Correlation ≠
  causation · Controlling for a third variable (true partial regression /
  FWL).
- Sampling & Standard Error: One Sample · Repeated Sampling · Estimate SE
  from One Sample.
- The t-test: standalone top-level lesson (not nested under Sampling).

**Dataset & sampling infrastructure**
- `data/students_population.csv` is the single source of truth (100k rows:
  hours_studied, sleep_hours, attendance, exam_score, individual_error).
  `src/population.py` loads and caches it; never regenerate or overwrite it.
- `src/sampling.py:draw_sample()` draws simple random samples without
  replacement for all lessons.
- The Journey's mini-chart illustrations use a separate small synthetic
  `JourneySample` (`build_journey_sample`), not the real population — fine,
  they're decorative previews only.

**Navigation structure**
- `MAIN_LAB_TABS` (app.py): journey → fitting → multiple_regression →
  correlation → sampling → ttest → library.
- Nav is two levels: a parent "The Statistical Journey" button, and a child
  lesson row built from `MAIN_LAB_LESSON_KEYS` (excludes journey & library).
  Dataset Library currently has no nav entry (removed "for now").
- Cross-lesson navigators (`open_lesson()`, `_rout_goto_ttest()`) must keep
  `main_lab_tab` and `main_lab_lesson_tab` in sync or the lesson row's
  highlight goes stale.

## 5. Known Issues
- **Bug:** Multiple Regression → Model Fit raises
  `TypeError: evaluate_model() missing required argument 'beta2'` —
  `_render_model_fit()` in `src/multiple_regression.py` calls
  `evaluate_model(*_single_predictor_ols(sample))`, but
  `_single_predictor_ols()` only returns `(alpha, beta1)`.
- Multiple Regression's 3D charts are too small; the fitted plane needs to
  read clearly at the sizes used.
- The Statistical Journey's Multiple Regression card illustration should be
  a more clearly recognizable 3D preview (note: 3D mini-charts must NOT use
  `staticPlot: true` in the Plotly config — it silently blanks WebGL/3D;
  use `MINI_CONFIG_3D` + `scene.dragmode=False` instead, as already done).
- Lesson transitions need stronger narrative links from each concept to the
  next.

## 6. Upcoming Work
- Finish the F-test lesson (still a "Coming soon" placeholder card).
- Review the whole Statistical Journey page by page for pedagogical
  continuity, explanations and layout.
- Preserve existing working functionality while doing so.

## 7. Mandatory Working Instructions
- Minimize token consumption and execution time.
- Make narrowly scoped changes; read only files relevant to the task.
- No exhaustive codebase audits, no elaborate plans, no unrelated
  refactoring.
- No full test suites for minor changes; targeted tests only, and only
  when statistical logic or functionality actually changed. Reserve full
  suite + browser verification for major milestones.
- No repeated browser screenshots/visual audits unless explicitly asked.
- Do not commit or push unless requested.
- Finish each task with a concise summary (what changed, what's unresolved).
