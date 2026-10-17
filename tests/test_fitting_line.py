"""Mathematical checks for the OLS fitting lesson."""

import numpy as np
import pytest

from src.charts import (
    BLUE,
    ORANGE,
    fitting_scatter,
    observation_index_from_point,
    observation_index_from_selection,
    sse_surface_figure,
)
from src.fitting_line import (
    DATASET_CATALOGS,
    DATASET_OPTIONS,
    MIN_SAMPLE_N,
    animation_metric_trajectories,
    animation_steps,
    best_intercept_for_slope,
    interactive_n_cap,
    load_fitting_dataset,
    make_fitting_sample,
    n_from_percentage,
    ols_fit,
    pair_complete_n,
    poor_candidate_line,
    r_lm_commands,
    r_lm_summary_text,
    r_squared,
    residual_standard_error,
    residuals,
    resolve_sample_n,
    sse_curve,
    sse_for_slope_with_best_intercept,
    sse_from_ols_identity,
    sse_surface_grids,
    subsample_observations,
    sum_of_squared_errors,
    total_sum_of_squares,
)


def test_sample_is_fixed_and_reproducible():
    first = make_fitting_sample(20261018)
    second = make_fitting_sample(20261018)
    assert first.hours.shape == (100,)
    assert np.array_equal(first.hours, second.hours)
    assert np.array_equal(first.scores, second.scores)
    other = make_fitting_sample(99)
    assert not (np.array_equal(first.hours, other.hours) and np.array_equal(first.scores, other.scores))


def test_students_fitting_uses_fixed_10k_pool():
    from src.population import load_population

    population = load_population()
    pool = load_fitting_dataset("students")
    assert pool.simulated
    assert pool.n == pool.available_n == 10_000
    assert pool.r_data == "students_10k"
    assert pool.r_command == (
        "model <- lm(exam_score ~ hours_studied, data = students_10k)"
    )
    # Sampling still works — draws from the fixed 10k pool, not the 100k file.
    view = subsample_observations(pool, 100, np.random.default_rng(5))
    assert view.n == 100
    assert view.available_n == 10_000
    assert "sample(" in pool.r_sample_commands(100)
    assert population.n == 100_000
    assert set(zip(pool.x.tolist(), pool.y.tolist())).issubset(
        set(zip(population.hours_studied.tolist(), population.exam_score.tolist()))
    )
    again = load_fitting_dataset("students")
    assert np.array_equal(pool.x, again.x)
    assert np.array_equal(pool.y, again.y)


def test_sse_matches_sum_of_squared_residuals():
    sample = make_fitting_sample()
    alpha, beta = 50.0, 2.0
    errors = residuals(sample.hours, sample.scores, alpha, beta)
    assert sum_of_squared_errors(sample.hours, sample.scores, alpha, beta) == float(np.dot(errors, errors))


def test_sse_curve_minimum_is_the_ols_slope():
    sample = make_fitting_sample()
    ols = ols_fit(sample.hours, sample.scores)
    slopes = np.linspace(ols.beta - 3.0, ols.beta + 3.0, 601)
    values = sse_curve(sample.hours, sample.scores, slopes)
    best = float(slopes[int(np.argmin(values))])
    assert abs(best - ols.beta) < 0.02
    assert abs(sse_for_slope_with_best_intercept(sample.hours, sample.scores, ols.beta) - ols.sse) < 1e-6


def test_best_intercept_matches_ols_at_ols_slope():
    sample = make_fitting_sample()
    ols = ols_fit(sample.hours, sample.scores)
    assert abs(best_intercept_for_slope(sample.hours, sample.scores, ols.beta) - ols.alpha) < 1e-9


def test_ols_sse_is_minimal_among_candidates():
    sample = make_fitting_sample()
    ols = ols_fit(sample.hours, sample.scores)
    candidates = [
        (ols.alpha, ols.beta),
        (ols.alpha + 5, ols.beta),
        (ols.alpha, ols.beta + 1.5),
        (30.0, 0.0),
        (80.0, 8.0),
    ]
    ols_sse = sum_of_squared_errors(sample.hours, sample.scores, ols.alpha, ols.beta)
    for alpha, beta in candidates:
        assert ols_sse <= sum_of_squared_errors(sample.hours, sample.scores, alpha, beta) + 1e-9


def test_animation_ends_at_ols():
    steps = animation_steps(70.0, 0.0, 42.5, 4.2, count=18)
    assert steps[0] == (70.0, 0.0)
    assert abs(steps[-1][0] - 42.5) < 1e-12
    assert abs(steps[-1][1] - 4.2) < 1e-12
    assert len(steps) == 18


def test_poor_candidate_is_plausible_and_imperfect():
    sample = make_fitting_sample()
    ols = ols_fit(sample.hours, sample.scores)
    sst = total_sum_of_squares(sample.scores)
    (a0, a1, _), (b0, b1, _) = sample.slider_bounds()
    x_min, x_max = float(sample.x.min()), float(sample.x.max())
    y_min, y_max = float(sample.y.min()), float(sample.y.max())
    for index in range(10):
        alpha, beta = poor_candidate_line(
            sample.hours,
            sample.scores,
            index=index,
            alpha_bounds=(a0, a1),
            beta_bounds=(b0, b1),
        )
        assert a0 <= alpha <= a1
        assert b0 <= beta <= b1
        sse = sum_of_squared_errors(sample.hours, sample.scores, alpha, beta)
        assert sse >= 1.2 * ols.sse - 1e-6
        assert sse <= 8.0 * ols.sse + 1e-6
        r2 = r_squared(sse, sst)
        assert not np.isfinite(r2) or r2 > -0.5
        # Line crosses the observed x-window inside an expanded y-band.
        y_left = alpha + beta * x_min
        y_right = alpha + beta * x_max
        band = 0.75 * (y_max - y_min)
        assert min(y_left, y_right) < y_max + band
        assert max(y_left, y_right) > y_min - band
    alpha0, beta0 = poor_candidate_line(
        sample.hours,
        sample.scores,
        index=0,
        alpha_bounds=(a0, a1),
        beta_bounds=(b0, b1),
    )
    assert abs(alpha0 - ols.alpha) > 0.5 or abs(beta0 - ols.beta) > 0.05


def test_try_another_line_keeps_the_same_dataset():
    sample = make_fitting_sample(20261018)
    first = poor_candidate_line(sample.hours, sample.scores, index=0)
    second = poor_candidate_line(sample.hours, sample.scores, index=1)
    assert first != second
    again = make_fitting_sample(20261018)
    assert np.array_equal(sample.hours, again.hours)
    assert np.array_equal(sample.scores, again.scores)


def test_current_sse_never_below_ols_minimum():
    sample = make_fitting_sample()
    ols = ols_fit(sample.hours, sample.scores)
    candidates = [
        (50.0, 1.0),
        (70.0, 0.0),
        (ols.alpha, ols.beta),
        *(poor_candidate_line(sample.hours, sample.scores, index=i) for i in range(3)),
    ]
    for alpha, beta in candidates:
        assert sum_of_squared_errors(sample.hours, sample.scores, alpha, beta) >= ols.sse - 1e-9


def test_animation_updates_both_parameters_toward_ols():
    sample = make_fitting_sample()
    ols = ols_fit(sample.hours, sample.scores)
    start = poor_candidate_line(sample.hours, sample.scores, index=0)
    steps = animation_steps(start[0], start[1], ols.alpha, ols.beta, count=18)
    assert steps[0] == start
    assert abs(steps[-1][0] - ols.alpha) < 1e-12
    assert abs(steps[-1][1] - ols.beta) < 1e-12
    mid = steps[len(steps) // 2]
    assert abs(mid[0] - start[0]) > 1e-6 or abs(mid[1] - start[1]) > 1e-6
    assert abs(mid[0] - ols.alpha) > 1e-6 or abs(mid[1] - ols.beta) > 1e-6


def test_authentic_datasets_sizes_and_r_syntax():
    expected = {
        "penguins": (342, "body_mass_g", "flipper_length_mm", "penguins"),
        "auto": (392, "mpg", "horsepower", "Auto"),
        "boston": (506, "medv", "rm", "Boston"),
        "diamonds": (53940, "price", "carat", "diamonds"),
    }
    labels = dict(DATASET_OPTIONS)
    assert labels["penguins"] == "Penguins (N = 342)"
    assert labels["diamonds"] == "Diamonds (N = 53,940)"
    assert labels["students"] == "Students (N = 10,000)"
    assert "—" not in labels["penguins"]
    keys = [key for key, _ in DATASET_OPTIONS]
    assert keys[:4] == ["penguins", "auto", "boston", "diamonds"]
    assert "students" in keys
    for key, (size, response, predictor, data_name) in expected.items():
        dataset = load_fitting_dataset(key)
        assert dataset.n == size
        assert dataset.available_n == size
        assert not dataset.simulated
        assert dataset.r_response == response
        assert dataset.r_predictor == predictor
        assert dataset.r_data == data_name
        assert f"lm({response} ~ {predictor}" in dataset.r_command
        assert "sample_data" in dataset.r_sample_commands(100)
        assert data_name in dataset.r_sample_commands(size)
        ols = ols_fit(dataset.x, dataset.y)
        assert np.isfinite(ols.alpha) and np.isfinite(ols.beta)
        assert ols.sse >= 0.0
    boston = load_fitting_dataset("boston")
    assert boston.caveat is not None and "ethical" in boston.caveat.lower()


def test_variable_switching_updates_pair_and_rejects_same_xy():
    catalog = DATASET_CATALOGS["penguins"]
    assert "bill_length_mm" in catalog.variable_names()
    switched = load_fitting_dataset(
        "penguins",
        x_name="bill_length_mm",
        y_name="body_mass_g",
    )
    assert switched.x_name == "bill_length_mm"
    assert switched.y_name == "body_mass_g"
    assert switched.x_axis == "Bill length (mm)"
    assert switched.y_axis == "Body mass (g)"
    assert switched.r_command == "model <- lm(body_mass_g ~ bill_length_mm, data = penguins)"
    assert switched.n == pair_complete_n("penguins", "bill_length_mm", "body_mass_g")
    assert switched.n >= 300
    ols = ols_fit(switched.x, switched.y)
    assert np.isfinite(ols.beta)
    with pytest.raises(ValueError, match="must be different"):
        load_fitting_dataset("penguins", x_name="body_mass_g", y_name="body_mass_g")


def test_percentage_sample_size_and_cap():
    assert n_from_percentage(342, 10) == 34
    assert n_from_percentage(392, 10) == 39
    requested, actual, capped = resolve_sample_n(53_940, 10, use_all=False)
    assert requested == 5394
    assert actual == 1000
    assert capped
    requested_all, actual_all, capped_all = resolve_sample_n(342, 10, use_all=True)
    assert requested_all == actual_all == 342
    assert not capped_all


def test_subsample_without_replacement_and_defaults():
    source = load_fitting_dataset("penguins")
    assert interactive_n_cap(source.available_n) == source.available_n
    assert interactive_n_cap(10_000) == 10_000
    assert interactive_n_cap(53940) == 1000
    rng = np.random.default_rng(7)
    view = subsample_observations(source, 100, rng)
    assert view.n == 100
    assert view.available_n == source.available_n
    again = subsample_observations(source, 100, np.random.default_rng(8))
    assert again.n == 100
    assert not (np.array_equal(view.x, again.x) and np.array_equal(view.y, again.y))
    full = subsample_observations(source, source.n, np.random.default_rng(1))
    assert full.n == source.n
    tiny = subsample_observations(source, 2, np.random.default_rng(2))
    assert tiny.n == MIN_SAMPLE_N


def test_r_squared_and_residual_se_formulas():
    sample = make_fitting_sample()
    sst = total_sum_of_squares(sample.scores)
    ols = ols_fit(sample.hours, sample.scores)
    ols_r2 = r_squared(ols.sse, sst)
    assert 0.0 <= ols_r2 <= 1.0 + 1e-12
    se = residual_standard_error(ols.sse, sample.n)
    assert abs(se - np.sqrt(ols.sse / (sample.n - 2))) < 1e-12
    poor = poor_candidate_line(sample.hours, sample.scores, index=0)
    poor_sse = sum_of_squared_errors(sample.hours, sample.scores, *poor)
    assert r_squared(poor_sse, sst) < ols_r2
    assert np.isnan(r_squared(0.0, 0.0))


def test_r_lm_commands_are_compact_with_current_names():
    view = subsample_observations(
        load_fitting_dataset("penguins"), 25, np.random.default_rng(11)
    )
    commands = r_lm_commands(
        predictor=view.r_predictor, response=view.r_response
    )
    assert commands == (
        f"model <- lm({view.r_response} ~ {view.r_predictor}, data = sample_data)\n"
        "summary(model)"
    )
    assert "data.frame" not in commands
    assert "sample(" not in commands
    assert "c(" not in commands


def test_r_lm_summary_matches_ols_calculations():
    view = subsample_observations(
        load_fitting_dataset("penguins"), 40, np.random.default_rng(3)
    )
    ols = ols_fit(view.x, view.y)
    text = r_lm_summary_text(
        view.x,
        view.y,
        predictor=view.r_predictor,
        response=view.r_response,
    )
    assert "Call:" in text
    assert "Residuals:" in text
    assert "Coefficients:" in text
    assert "Estimate" in text and "Std. Error" in text and "t value" in text
    assert "Pr(>|t|)" in text
    assert "Residual standard error:" in text
    assert "Multiple R-squared:" in text
    assert "Adjusted R-squared:" in text
    assert "F-statistic:" in text
    assert f"{ols.alpha:.4f}" in text
    assert f"{ols.beta:.4f}" in text
    se = residual_standard_error(ols.sse, view.n)
    assert f"{se:.3f}" in text
    r2 = r_squared(ols.sse, total_sum_of_squares(view.y))
    assert f"{r2:.4f}" in text


def test_sample_fixed_while_line_changes_math():
    source = load_fitting_dataset("auto")
    view = subsample_observations(source, 80, np.random.default_rng(3))
    ols = ols_fit(view.x, view.y)
    a = sum_of_squared_errors(view.x, view.y, ols.alpha + 5, ols.beta - 0.1)
    b = sum_of_squared_errors(view.x, view.y, ols.alpha, ols.beta)
    assert a > b
    assert np.array_equal(view.x, subsample_observations(source, 80, np.random.default_rng(3)).x)


def test_large_sample_performance_smoke():
    source = load_fitting_dataset("diamonds")
    view = subsample_observations(source, 1000, np.random.default_rng(5))
    ols = ols_fit(view.x, view.y)
    slopes = view.slope_grid(81)
    curve = sse_curve(view.x, view.y, slopes)
    assert curve.shape == (81,)
    assert np.isfinite(curve).all()
    assert abs(float(np.min(curve)) - ols.sse) / max(ols.sse, 1.0) < 0.05


def test_missing_dataset_raises():
    with pytest.raises(KeyError):
        load_fitting_dataset("mtcars")


def test_sse_identity_matches_direct_calculation():
    sample = make_fitting_sample(11)
    ols = ols_fit(sample.hours, sample.scores)
    x_mean = float(np.mean(sample.hours))
    centred = sample.hours - x_mean
    sxx = float(np.dot(centred, centred))
    for alpha, beta in (
        (ols.alpha, ols.beta),
        (ols.alpha + 3.0, ols.beta - 0.4),
        (ols.alpha - 1.5, ols.beta + 0.2),
        (ols.alpha + 8.0, ols.beta + 1.1),
    ):
        direct = sum_of_squared_errors(sample.hours, sample.scores, alpha, beta)
        identity = sse_from_ols_identity(
            alpha,
            beta,
            alpha_ols=ols.alpha,
            beta_ols=ols.beta,
            sse_min=ols.sse,
            n=sample.n,
            x_mean=x_mean,
            sxx=sxx,
        )
        assert abs(float(identity) - float(direct)) <= 1e-8 * max(1.0, abs(direct))


@pytest.mark.parametrize("key", ["students", "penguins", "auto", "boston", "diamonds"])
def test_sse_surface_readable_across_datasets(key):
    source = load_fitting_dataset(key)
    n = min(120, source.n)
    view = subsample_observations(source, n, np.random.default_rng(21))
    ols = ols_fit(view.x, view.y)
    cand_alpha = float(ols.alpha + 0.4 * surface_alpha_pad(view))
    cand_beta = float(ols.beta)
    surface = sse_surface_grids(
        view.x,
        view.y,
        candidate_alpha=cand_alpha,
        candidate_beta=cand_beta,
        resolution=25,
    )
    assert abs(float(surface["sse_min"]) - float(ols.sse)) < 1e-9
    assert abs(float(surface["alpha_ols"]) - float(ols.alpha)) < 1e-9
    assert abs(float(surface["beta_ols"]) - float(ols.beta)) < 1e-9
    assert float(np.min(surface["excess_grid"])) >= -1e-6
    # Exact minimum of the continuous surface is zero at OLS.
    assert abs(
        float(
            sse_from_ols_identity(
                ols.alpha,
                ols.beta,
                alpha_ols=ols.alpha,
                beta_ols=ols.beta,
                sse_min=ols.sse,
                n=int(surface["n"]),
                x_mean=float(surface["x_mean"]),
                sxx=float(surface["sxx"]),
            )
        )
        - float(ols.sse)
    ) < 1e-9
    # Bowl is not flattened: interior varies relative to the minimum.
    spread = float(np.max(surface["excess_grid"]) - np.min(surface["excess_grid"]))
    assert spread > 0.05 * max(float(ols.sse), 1.0)
    # Spot-check identity vs direct SSE on the mesh.
    i, j = 3, 7
    alpha = float(surface["alpha_grid"][i, j])
    beta = float(surface["beta_grid"][i, j])
    direct = sum_of_squared_errors(view.x, view.y, alpha, beta)
    assert abs(float(surface["sse_grid"][i, j]) - direct) <= 1e-6 * max(1.0, abs(direct))
    assert abs(float(surface["excess_grid"][i, j]) - (direct - ols.sse)) <= 1e-6 * max(1.0, abs(direct))
    # Normalised axes share comparable ranges (not raw α/β magnitudes).
    u_span = float(np.ptp(surface["u_grid"]))
    v_span = float(np.ptp(surface["v_grid"]))
    assert 0.25 < u_span / v_span < 4.0
    fig = sse_surface_figure(
        u_grid=surface["u_grid"],
        v_grid=surface["v_grid"],
        excess_grid=surface["excess_grid"],
        alpha_grid=surface["alpha_grid"],
        beta_grid=surface["beta_grid"],
        sse_grid=surface["sse_grid"],
        alpha_ols=float(surface["alpha_ols"]),
        beta_ols=float(surface["beta_ols"]),
        sse_min=float(surface["sse_min"]),
        alpha_scale=float(surface["alpha_scale"]),
        beta_scale=float(surface["beta_scale"]),
        current_alpha=cand_alpha,
        current_beta=cand_beta,
        current_sse=sum_of_squared_errors(view.x, view.y, cand_alpha, cand_beta),
        height=400,
    )
    assert len(fig.data) >= 3
    assert fig.layout.title.text == "OLS: Finding the minimum SSR"
    assert "Extra SSR" in fig.layout.scene.zaxis.title.text


def surface_alpha_pad(view) -> float:
    (a0, a1, _), _ = view.slider_bounds()
    return 0.5 * (float(a1) - float(a0))


def test_animation_ends_at_exact_ols():
    sample = make_fitting_sample(5)
    ols = ols_fit(sample.hours, sample.scores)
    steps = animation_steps(ols.alpha + 4, ols.beta - 0.5, ols.alpha, ols.beta, count=12)
    assert abs(steps[-1][0] - ols.alpha) < 1e-12
    assert abs(steps[-1][1] - ols.beta) < 1e-12


def test_fitting_scatter_line_spans_full_displayed_x_range():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 3.0, 5.0, 4.0, 6.0])
    fig = fitting_scatter(
        hours=x,
        scores=y,
        alpha=1.0,
        beta=1.0,
        show_residuals=False,
        x_title="Carat",
        y_title="Price",
    )
    line = next(trace for trace in fig.data if trace.name == "Regression line")
    x0, x1 = fig.layout.xaxis.range
    assert list(line.x) == pytest.approx([x0, x1])
    assert float(line.x[0]) < float(x.min())
    assert float(line.x[1]) > float(x.max())
    obs = next(trace for trace in fig.data if trace.name == "Observations")
    assert "Observation" in str(obs.hovertemplate)
    assert "Student" not in str(obs.hovertemplate)


def test_fitting_scatter_selection_shows_explain_decomposition():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 3.0, 5.0, 4.0, 6.0])
    alpha, beta = 1.0, 1.0
    selected = 2
    y_bar = float(np.mean(y))
    y_hat = alpha + beta * float(x[selected])

    clear = fitting_scatter(
        hours=x, scores=y, alpha=alpha, beta=beta, selected_index=None, show_residuals=False
    )
    names_clear = {trace.name for trace in clear.data}
    assert "Observations" in names_clear
    assert "Actual Y" not in names_clear
    assert "Model component" not in names_clear
    assert "Sample mean Ȳ" not in names_clear
    assert "Total deviation" not in names_clear

    fig = fitting_scatter(
        hours=x,
        scores=y,
        alpha=alpha,
        beta=beta,
        selected_index=selected,
        show_residuals=False,
        prediction_x=10.0,
        prediction_alpha=alpha,
        prediction_beta=beta,
    )
    by_name = {trace.name: trace for trace in fig.data}
    assert "Actual Y" in by_name
    assert "Predicted Ŷ" in by_name
    assert "Sample mean Ȳ" in by_name
    assert "Model component" in by_name
    assert "Residual" in by_name
    assert "Prediction" in by_name  # PREDICT stays independent
    assert "Total deviation" not in by_name

    assert by_name["Sample mean Ȳ"].line.dash == "dash"
    assert by_name["Sample mean Ȳ"].line.color == "#64748B"
    assert by_name["Model component"].line.color == BLUE
    assert by_name["Residual"].line.color == ORANGE
    assert by_name["Predicted Ŷ"].marker.color == ORANGE
    assert by_name["Actual Y"].marker.color == BLUE
    assert list(by_name["Model component"].y) == pytest.approx([y_bar, y_hat])
    assert list(by_name["Residual"].y) == pytest.approx([y_hat, float(y[selected])])
    # Distinct verticals (no shared X) that meet at Ŷ.
    model_x = list(by_name["Model component"].x)
    resid_x = list(by_name["Residual"].x)
    assert model_x[0] == model_x[1]
    assert resid_x[0] == resid_x[1]
    assert model_x[0] < float(x[selected]) < resid_x[0]
    assert "Ŷ join" in by_name
    assert list(by_name["Ŷ join"].y) == pytest.approx([y_hat, y_hat])
    assert float(by_name["Actual Y"].y[0]) == pytest.approx(float(y[selected]))
    assert float(by_name["Predicted Ŷ"].y[0]) == pytest.approx(y_hat)
    assert float(by_name["Predicted Ŷ"].x[0]) == pytest.approx(float(x[selected]))
    assert float(by_name["Actual Y"].x[0]) == pytest.approx(float(x[selected]))
    mean_labels = [
        str(getattr(ann, "text", "") or "")
        for ann in (fig.layout.annotations or [])
        if "Sample mean Ȳ" in str(getattr(ann, "text", "") or "")
    ]
    assert mean_labels and f"{y_bar:.2f}" in mean_labels[0]

    # PREDICT alone must not draw the mean line or student markers.
    predict_only = fitting_scatter(
        hours=x,
        scores=y,
        alpha=alpha,
        beta=beta,
        selected_index=None,
        show_residuals=False,
        prediction_x=10.0,
        prediction_alpha=alpha,
        prediction_beta=beta,
    )
    predict_names = {trace.name for trace in predict_only.data}
    assert "Prediction" in predict_names
    assert "Sample mean Ȳ" not in predict_names
    assert "Actual Y" not in predict_names
    assert "Predicted Ŷ" not in predict_names
    # Unselected points must not be dimmed.
    assert by_name["Observations"].unselected.marker.opacity == 1.0

    obs = by_name["Observations"]
    assert obs.customdata.shape == (5, 2)
    assert int(obs.customdata[selected][0]) == selected


def test_explain_decomposition_segments_above_and_below_mean_and_line():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    # Mean = 5. Line Ŷ = 2 + 2x → at x=0: 2, x=3: 8.
    y = np.array([1.0, 4.0, 6.0, 9.0])
    alpha, beta = 2.0, 2.0
    y_bar = float(np.mean(y))
    assert y_bar == pytest.approx(5.0)

    below = fitting_scatter(
        hours=x, scores=y, alpha=alpha, beta=beta, selected_index=0, show_residuals=False
    )
    below_traces = {trace.name: trace for trace in below.data}
    y_hat0 = alpha + beta * x[0]
    assert y_hat0 < y_bar and y[0] < y_hat0
    assert list(below_traces["Model component"].y) == pytest.approx([y_bar, y_hat0])
    assert list(below_traces["Residual"].y) == pytest.approx([y_hat0, float(y[0])])
    assert below_traces["Model component"].x[0] < x[0] < below_traces["Residual"].x[0]

    above = fitting_scatter(
        hours=x, scores=y, alpha=alpha, beta=beta, selected_index=3, show_residuals=False
    )
    above_traces = {trace.name: trace for trace in above.data}
    y_hat3 = alpha + beta * x[3]
    assert y_hat3 > y_bar and y[3] > y_hat3
    assert list(above_traces["Model component"].y) == pytest.approx([y_bar, y_hat3])
    assert list(above_traces["Residual"].y) == pytest.approx([y_hat3, float(y[3])])
    assert above_traces["Model component"].x[0] < x[3] < above_traces["Residual"].x[0]


def test_explain_decomposition_no_overlap_when_y_between_mean_and_yhat():
    """When Y sits between Ȳ and Ŷ, stacked same-X segments used to hide blue."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 5.5, 6.0])  # mean = 5.125
    alpha, beta = 2.0, 2.0  # at x=2: Ŷ = 6; Y = 5.5 lies between Ȳ and Ŷ
    selected = 2
    y_bar = float(np.mean(y))
    y_hat = alpha + beta * float(x[selected])
    assert y_bar < float(y[selected]) < y_hat

    fig = fitting_scatter(
        hours=x, scores=y, alpha=alpha, beta=beta, selected_index=selected, show_residuals=False
    )
    by_name = {trace.name: trace for trace in fig.data}
    model_x = float(by_name["Model component"].x[0])
    resid_x = float(by_name["Residual"].x[0])
    assert model_x != resid_x
    assert abs(resid_x - model_x) >= 0.2  # clear side-by-side gap
    assert list(by_name["Model component"].y) == pytest.approx([y_bar, y_hat])
    assert list(by_name["Residual"].y) == pytest.approx([y_hat, float(y[selected])])
    # Open Y-intervals would overlap; distinct X keeps both visible.
    model_lo, model_hi = sorted(by_name["Model component"].y)
    resid_lo, resid_hi = sorted(by_name["Residual"].y)
    assert model_lo < resid_hi and resid_lo < model_hi


def test_explain_decomposition_visible_when_residual_crosses_mean():
    """Y above Ȳ and Ŷ below Ȳ: residual contains the model band — blue must stay offset."""
    x = np.array([100.0, 120.0, 140.0, 160.0, 180.0])
    y = np.array([50.0, 40.0, 55.0, 35.0, 45.0])
    alpha, beta = 90.0, -0.4
    selected = 2
    y_bar = float(np.mean(y))
    y_hat = alpha + beta * float(x[selected])
    assert float(y[selected]) > y_bar > y_hat

    fig = fitting_scatter(
        hours=x,
        scores=y,
        alpha=alpha,
        beta=beta,
        selected_index=selected,
        show_residuals=True,
    )
    by_name = {trace.name: trace for trace in fig.data}
    assert list(by_name["Model component"].y) == pytest.approx([y_bar, y_hat])
    assert list(by_name["Residual"].y) == pytest.approx([y_hat, float(y[selected])])
    model_x = float(by_name["Model component"].x[0])
    resid_x = float(by_name["Residual"].x[0])
    assert model_x < float(x[selected]) < resid_x
    # Residual Y-range contains model Y-range; X gap is what keeps blue visible.
    assert min(float(y[selected]), y_hat) <= min(y_bar, y_hat)
    assert max(float(y[selected]), y_hat) >= max(y_bar, y_hat)
    assert abs(resid_x - model_x) >= 0.08 * float(np.ptp(x))
    # Selected point must not also get a grey residual stem on the same column.
    if "Residuals" in by_name:
        xs = [v for v in by_name["Residuals"].x if v is not None]
        assert float(x[selected]) not in xs


def test_animation_metric_trajectories_use_real_sse_and_r2():
    sample = make_fitting_sample()
    ols = ols_fit(sample.hours, sample.scores)
    sst = total_sum_of_squares(sample.scores)
    start = (ols.alpha + 5.0, ols.beta - 0.8)
    steps = animation_steps(start[0], start[1], ols.alpha, ols.beta, count=12)
    sses, r2s = animation_metric_trajectories(sample.hours, sample.scores, steps, sst)
    assert len(sses) == len(steps) == len(r2s)
    for (alpha, beta), sse, r2 in zip(steps, sses, r2s):
        expect_sse = sum_of_squared_errors(sample.hours, sample.scores, alpha, beta)
        assert float(sse) == pytest.approx(expect_sse)
        assert float(r2) == pytest.approx(r_squared(expect_sse, sst))
    assert float(sses[-1]) == pytest.approx(ols.sse)
    assert float(sses[0]) > float(sses[-1])


def test_fit_metric_mini_chart_is_compact_curve_only():
    from src.charts import fit_metric_mini_chart

    values = np.array([1500.0, 1200.0, 900.0, 800.0])
    fig = fit_metric_mini_chart(
        values=values[:2],
        y_range=(700.0, 1600.0),
        x_max=3,
        title="",
        height=50,
        width=75,
        uirevision="fit-sse-mini",
    )
    assert fig.layout.height == 50
    assert fig.layout.width == 75
    assert fig.layout.uirevision == "fit-sse-mini"
    assert fig.layout.xaxis.visible is False
    assert fig.layout.yaxis.visible is False
    assert fig.layout.showlegend is False
    assert len(fig.data) == 1
    assert list(fig.data[0].y) == pytest.approx([1500.0, 1200.0])
    assert fig.layout.yaxis.range == pytest.approx([700.0, 1600.0])
    assert not fig.layout.annotations

    blank = fit_metric_mini_chart(
        values=np.array([], dtype=np.float64),
        y_range=(0.0, 1.0),
        x_max=0,
        title="",
        height=50,
        width=75,
        uirevision="fit-sse-mini",
    )
    assert len(blank.data) == 0
    assert blank.layout.height == 50
    assert blank.layout.width == 75


def test_hats_match_ols_tolerance():
    from app import _hats_match_ols
    from types import SimpleNamespace

    ols = SimpleNamespace(alpha=10.0, beta=2.0, degenerate=False)
    assert _hats_match_ols(10.0, 2.0, ols, a_step=0.1, b_step=0.05)
    assert _hats_match_ols(10.04, 2.02, ols, a_step=0.1, b_step=0.05)
    assert not _hats_match_ols(10.2, 2.0, ols, a_step=0.1, b_step=0.05)
    assert not _hats_match_ols(10.0, 2.1, ols, a_step=0.1, b_step=0.05)


def test_fit_metric_label_html_includes_ssr_r2_relation():
    from app import _fit_metric_label_html, _fit_r2_range_label_html
    from src.stat_tips import tip_html, tipify

    ssr = _fit_metric_label_html("SSR")
    assert "stat-tip" in ssr
    assert "SSR" in ssr
    assert "↓" not in ssr
    assert "Residual Sum of Squares" in ssr
    assert "lm() minimizes SSR" in ssr
    assert "Σ(y" in ssr
    assert "22,727" not in ssr

    r2 = _fit_metric_label_html("R²")
    assert "stat-tip" in r2
    assert "↑" not in r2
    assert "minimizes SSR" in r2

    rng = _fit_r2_range_label_html(0.501, 0.884)
    assert "50.1%" in rng
    assert "88.4%" in rng
    assert "→" in rng
    assert "fit-metric-r2-range" in rng

    tip = tip_html("SSR")
    assert tip.count('class="stat-tip"') == 1
    assert tipify("SSR and R²").count('class="stat-tip"') == 2
    assert tipify("SS Residual").count("SSR") >= 1


def test_observation_index_from_chart_selection_uses_customdata_only():
    point = {"customdata": [3], "point_index": 9, "curve_number": 1}
    assert observation_index_from_point(point) == 3
    assert observation_index_from_point({"customdata": [[3]]}) == 3
    assert observation_index_from_selection({"selection": {"points": [point]}}) == 3
    # Without customdata / coords, do not invent an index.
    assert observation_index_from_point({"point_index": 4, "x": 1.0, "y": 2.0}) is None
    assert observation_index_from_selection(
        {"selection": {"points": [{"point_index": 4}]}}
    ) is None
    assert observation_index_from_selection({"selection": {"points": []}}) is None
    # Residual-line hit: map (x, y) to the nearest observation.
    xs = np.array([1.0, 2.0, 3.0])
    ys = np.array([4.0, 5.0, 6.0])
    assert (
        observation_index_from_selection(
            {"selection": {"points": [{"x": 2.0, "y": 5.5, "point_index": 0}]}},
            x_values=xs,
            y_values=ys,
        )
        == 1
    )
    assert (
        observation_index_from_selection(
            {"selection": {"points": [{"curve_number": 0, "point_index": 2}]}},
            observations_curve=0,
        )
        == 2
    )
