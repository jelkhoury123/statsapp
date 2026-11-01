"""Quiz question bank for the Revision Sheets tab.

Ten multiple-choice questions per week (40 total), keyed to match the
revision-sheet PDFs in data/revision_sheets/. Pure data — no Streamlit
imports — so the UI module can stay separate from the content.

Each question is a dict with keys:
    q           — question text (may include inline code via backticks,
                   rendered as markdown)
    choices     — exactly four answer strings
    correct     — index (0-3) of the correct choice
    explanation — shown after answering, markdown-rendered

Distractors are written to reflect plausible, common misconceptions
(sign flips, conflating standardised/unstandardised quantities, "accept
H0", ignoring the nesting requirement, etc.) rather than obviously
absurd alternatives, so a correct answer requires genuine understanding
rather than elimination by inspection.
"""

from __future__ import annotations

from typing import TypedDict


class QuizQuestion(TypedDict):
    q: str
    choices: list[str]
    correct: int
    explanation: str


_W1: list[QuizQuestion] = [
    {
        "q": "A deterministic model (y = 3 + 2x) and a statistical model "
        "(y = 3 + 2x + ε) both describe a linear relationship between x "
        "and y. What is the key difference between them?",
        "choices": [
            "The statistical model adds an error term ε, so observed points scatter around the line instead of lying exactly on it",
            "The statistical model has a different, data-estimated slope, while the deterministic model's slope is fixed by theory alone",
            "The deterministic model cannot be written as a linear equation",
            "The statistical model guarantees a better fit, because it accounts for every source of variation in y",
        ],
        "correct": 0,
        "explanation": "ε is the only structural difference: it allows "
        "case-by-case variability around the line. Both forms can be "
        "fitted from data or specified by theory — the error term, not "
        "where the slope comes from, is what makes a model \"statistical\". "
        "And a statistical model doesn't guarantee a good fit; it just "
        "allows for imperfect ones.",
    },
    {
        "q": "In yᵢ = β₀ + β₁xᵢ + εᵢ, which symbols carry no subscript *i*, "
        "and why?",
        "choices": [
            "β₀ and β₁, because they define the single line fitted to the whole dataset, shared by every observation",
            "yᵢ and xᵢ, because those are measured without any error",
            "εᵢ, because the residual is assumed to equal zero for every participant",
            "β₀ and β₁, because the intercept and slope don't need to be estimated from the data",
        ],
        "correct": 0,
        "explanation": "β₀ and β₁ are the right symbols, but only option "
        "1 gives the right reason: they're shared across all observations "
        "because they describe the one fitted line — they are still "
        "estimated from the data, just once rather than per observation.",
    },
    {
        "q": "A fitted model has intercept β̂₀ = −5. Which interpretation is correct?",
        "choices": [
            "The model predicts y = −5 when x = 0",
            "The average value of x in the sample is −5",
            "For every one-unit increase in x, y decreases by 5",
            "The model predicts y = −5 for a \"typical\" observation, at the mean of x",
        ],
        "correct": 0,
        "explanation": "The intercept is specifically the prediction at "
        "x = 0 — not the prediction at x̄ (that would need the slope "
        "too, unless x̄ happens to be 0), and not a statement about x's "
        "own average value or about the slope (the rate of change).",
    },
    {
        "q": "A fitted model has slope β̂₁ = −2. Which statement correctly interprets it?",
        "choices": [
            "Each 1-unit increase in x is associated with a 2-unit decrease, on average, in predicted y",
            "x and y have a weak relationship, because the slope's size is small",
            "Each 1-unit increase in x is associated with predicted y being multiplied by −2",
            "A negative slope means a correlation between x and y cannot be computed",
        ],
        "correct": 0,
        "explanation": "The slope is an additive rate of change in the "
        "original units, not a multiplicative one. Its magnitude isn't "
        "directly comparable to \"strength\" of relationship without "
        "accounting for scale — that's what standardising is for. A "
        "negative slope is perfectly ordinary and still has a "
        "well-defined (negative) correlation.",
    },
    {
        "q": "For observation i, the residual is εᵢ = yᵢ − ŷᵢ = +3. What does this tell us?",
        "choices": [
            "The observed value exceeded the model's prediction — the point lies above the line",
            "The model's prediction exceeded the observed value — the point lies below the line",
            "The observation is an outlier and should be removed before fitting",
            "The model's slope must be too steep near that point",
        ],
        "correct": 0,
        "explanation": "εᵢ = observed − predicted, so a positive residual "
        "means the point sits above the line. A single residual, however "
        "large, doesn't by itself justify removing data, or say anything "
        "about the slope specifically.",
    },
    {
        "q": "Why does the least-squares method square the residuals before summing them, rather than summing them directly?",
        "choices": [
            "Unsquared residuals from the best-fitting line sum to exactly zero, so squaring prevents positive and negative deviations from cancelling, and it penalises large errors more",
            "Negative residuals indicate the model is wrong, so squaring eliminates them",
            "Squaring converts the residuals into a probability, which the method then minimises",
            "Squaring ensures every residual has the same value, making the fitted line more stable",
        ],
        "correct": 0,
        "explanation": "A negative residual just means that point lies "
        "below the line — not that the model is \"wrong\". Squaring has "
        "nothing to do with probabilities or with making residuals "
        "equal; it simply removes the exact cancellation and gives a "
        "smooth function with a closed-form minimum.",
    },
    {
        "q": "A 10-student dataset has x̄ = 2.75, ȳ = 3.3, SSₓ = Σ(xᵢ−x̄)² = 20.625 "
        "and SPₓᵥ = Σ(xᵢ−x̄)(yᵢ−ȳ) = 21.75. What is the least-squares slope b₁?",
        "choices": ["1.05 (= SPₓᵥ / SSₓ)", "0.95 (= SSₓ / SPₓᵥ)", "20.63 (= SSₓ alone)", "−1.05 (sign flipped)"],
        "correct": 0,
        "explanation": "b₁ = SPₓᵥ / SSₓ = 21.75 / 20.625 ≈ 1.05. Inverting "
        "the ratio or reporting SSₓ on its own are the two most common "
        "slip-ups here.",
    },
    {
        "q": "Using the same dataset (x̄ = 2.75, ȳ = 3.3, b₁ ≈ 1.05), what is the intercept b₀?",
        "choices": [
            "0.40 (= ȳ − b₁x̄)",
            "3.30 (= ȳ, forgetting to subtract the b₁x̄ term)",
            "2.90 (= b₁x̄, the subtracted term on its own)",
            "1.05 (reporting the slope instead of the intercept)",
        ],
        "correct": 0,
        "explanation": "b₀ = ȳ − b₁·x̄ = 3.3 − (1.05 × 2.75) ≈ 3.3 − 2.90 "
        "= 0.40. The fitted line always passes through the point of means "
        "(x̄, ȳ), which is exactly what this formula encodes.",
    },
    {
        "q": "Which R code correctly fits a model predicting `score` from `hours` using the data frame `test`?",
        "choices": [
            "`lm(score ~ hours, data = test)`",
            "`lm(hours ~ score, data = test)`",
            "`lm(score + hours, data = test)`",
            "`lm(score ~ hours, data = hours)`",
        ],
        "correct": 0,
        "explanation": "The outcome (DV) goes on the left of `~`, the "
        "predictor (IV) on the right, with `data` pointing to the data "
        "frame that holds both columns. The other options either reverse "
        "DV and IV, omit `~` entirely, or point `data` at the wrong object.",
    },
    {
        "q": "A model fitted on babies aged 0–11 months predicts that a "
        "20-year-old (240 months) would be 10 metres tall. What is the "
        "best explanation for this absurd prediction?",
        "choices": [
            "The model was extrapolated far beyond the age range it was fitted on, where a linear relationship no longer holds",
            "The slope must have been calculated incorrectly, since no valid slope could ever produce this result",
            "Linear models should never be used to study human growth, even within the observed age range",
            "The intercept, rather than the slope, is responsible for the unrealistic prediction",
        ],
        "correct": 0,
        "explanation": "The arithmetic can be entirely correct — the "
        "problem is applying the model far outside the range of ages it "
        "was fitted on, where the linear assumption breaks down. Within "
        "the original 0–11 month range the model may fit perfectly well.",
    },
]

_W2: list[QuizQuestion] = [
    {
        "q": "In `lm(score ~ hours, data = test)`, the hours coefficient is 1.0545. Which "
        "sentence correctly interprets it?",
        "choices": [
            "For every one-hour increase in study time, predicted test score increases on average by 1.0545 points",
            "Students who score 1.0545 points higher study one more hour, on average",
            "1.0545% of the variance in score is explained by hours",
            "The odds of passing increase by a factor of 1.0545 for every hour studied",
        ],
        "correct": 0,
        "explanation": "The standard template: \"for every one-[unit of "
        "X] increase in X, Y changes on average by β₁ [units of Y]\" — "
        "here, 1 point of score per 1 hour studied. Variance explained is "
        "R², not the coefficient, and \"odds\" language belongs to "
        "logistic regression, not this linear model.",
    },
    {
        "q": "A model regresses cat tail length (cm) on cat weight (kg) and "
        "finds an intercept of 14 cm. Why is this intercept hard to "
        "interpret meaningfully?",
        "choices": [
            "Because x = 0 (a cat weighing 0 kg) doesn't occur in reality, so the prediction at x = 0 falls outside any meaningful range",
            "Because the intercept must always equal the sample mean of tail length",
            "Because a positive intercept implies the slope must also be positive",
            "Because tail length should be modelled with a quadratic term, not a straight line",
        ],
        "correct": 0,
        "explanation": "The intercept is only a sensible number when "
        "x = 0 is a plausible value of the predictor. It equals ȳ only "
        "in the special case x̄ = 0, so claiming it must always equal "
        "the mean is false in general — and the intercept's sign places "
        "no constraint on the slope's sign.",
    },
    {
        "q": "If both x and y are converted to z-scores before fitting, "
        "what does the resulting slope represent?",
        "choices": [
            "The standardised slope, equal to β₁ · sₓ/s_y",
            "The proportion of variance in y explained by x (R²)",
            "The same numeric value as the unstandardised slope, since standardising doesn't change the relationship between the variables",
            "The correlation between x and the model's residuals",
        ],
        "correct": 0,
        "explanation": "Standardising rescales the coefficient itself — "
        "the classic trap is thinking nothing changes, but that's only "
        "true of the *relationship* (t, p, R², F); the slope's numeric "
        "value does change unless sₓ = s_y. R² is a separate quantity, "
        "and by construction OLS residuals are uncorrelated with x, so "
        "that correlation is ~0, not the slope.",
    },
    {
        "q": "After standardising both the predictor and outcome (converting "
        "both to z-scores), what happens to the intercept of the fitted line?",
        "choices": [
            "It becomes 0, because all variable means become 0",
            "It becomes equal to the standardised slope",
            "It stays exactly the same as the unstandardised intercept, because standardising only affects the slope",
            "It becomes 1, matching the standard deviation of a z-score",
        ],
        "correct": 0,
        "explanation": "β₀ = ȳ − β₁x̄. Once both variables are standardised, "
        "x̄ = ȳ = 0, so β₀ = 0 (up to rounding error) — standardising "
        "affects the intercept too, not just the slope.",
    },
    {
        "q": "Compared with the unstandardised model, what happens to the "
        "t-value, p-value, R² and F-statistic after standardising the variables?",
        "choices": [
            "They are unchanged — standardising only rescales the axes, not the strength of evidence",
            "The p-value becomes smaller, because standardised variables carry less measurement error",
            "R² increases, because standardising removes scale-related noise from the data",
            "The t-value flips sign if the original slope was negative",
        ],
        "correct": 0,
        "explanation": "Standardising is a linear rescaling of the same "
        "fitted relationship, so every test statistic, p-value, R² and F "
        "stays identical. It doesn't change how much noise is in the "
        "data, and the sign of t still matches the sign of the original "
        "slope — it never flips.",
    },
    {
        "q": "In a simple regression with one continuous predictor, the "
        "standardised slope β₁* is exactly equal to…",
        "choices": [
            "Pearson's correlation coefficient r between x and y",
            "The coefficient of determination R²",
            "The unstandardised slope β₁, before any rescaling",
            "The residual standard error, expressed in standard-deviation units",
        ],
        "correct": 0,
        "explanation": "With one predictor, β₁* = r, and consequently "
        "R² = r² — but this equivalence holds for the *standardised* "
        "slope specifically. The raw, unstandardised β₁ only equals r "
        "when sₓ happens to equal s_y.",
    },
    {
        "q": "In multiple regression, a partial regression coefficient βⱼ "
        "represents the change in y for a one-unit increase in xⱼ…",
        "choices": [
            "Holding all other predictors in the model constant (controlling for them)",
            "Ignoring the effect of all other predictors in the model",
            "Averaged across every possible combination of the other predictors' values, rather than holding them fixed",
            "Only when xⱼ is uncorrelated with the other predictors",
        ],
        "correct": 0,
        "explanation": "\"Partial\" / \"controlling for\" specifically "
        "means holding the other predictors fixed, not averaging over "
        "their values — a subtler version of the same error as simply "
        "ignoring them. The interpretation is valid whether or not the "
        "predictors are correlated — correlation among predictors "
        "affects the coefficient's precision, not its meaning.",
    },
    {
        "q": "A model is: Score = 6.87 + 1.38·Hours + 0.92·Motivation "
        "(Motivation is standardised, mean 0). What score does it predict "
        "for a student who studied 3 hours with average motivation?",
        "choices": ["11.01", "9.63 (swaps which coefficient belongs to which predictor)", "4.14 (forgets the intercept)", "6.87 (forgets both slope terms)"],
        "correct": 0,
        "explanation": "6.87 + (1.38 × 3) + (0.92 × 0) = 6.87 + 4.14 = "
        "11.01. Average motivation contributes exactly 0 because that "
        "variable is standardised (mean 0) — it's not that it has no "
        "effect in general.",
    },
    {
        "q": "Using the same model, student ID101 has hours = 2 and "
        "motivation = −1.42. What score does the model predict (before "
        "adding the residual)?",
        "choices": [
            "8.32",
            "10.94 (treats the negative motivation value as if it were positive)",
            "7.00 (this is the student's observed score, not the model's prediction)",
            "5.56 (drops the hours term entirely)",
        ],
        "correct": 0,
        "explanation": "ŷ = 6.87 + (1.38 × 2) + (0.92 × −1.42) = "
        "6.87 + 2.76 − 1.31 ≈ 8.32. The observed score (7) only matches "
        "this once the residual (−1.32) is added back on.",
    },
    {
        "q": "Which of these is NOT a legitimate reason to add predictors "
        "to a regression model?",
        "choices": [
            "To artificially push an existing predictor's p-value below .05",
            "To improve prediction of the outcome",
            "To test a theory that several variables jointly contribute to the outcome",
            "To control for a confound while assessing one predictor's effect",
        ],
        "correct": 0,
        "explanation": "Prediction, theory-testing and covariate control "
        "are the standard motivations for multiple regression. Adding "
        "predictors purely to chase a smaller p-value for an unrelated "
        "coefficient isn't a valid research justification.",
    },
]

_W3: list[QuizQuestion] = [
    {
        "q": "For testing a slope coefficient, which pair of hypotheses is correct?",
        "choices": [
            "H₀: β₁ = 0, H₁: β₁ ≠ 0",
            "H₀: β₁ ≠ 0, H₁: β₁ = 0",
            "H₀: β₁ ≤ 0, H₁: β₁ > 0 (testing only for a positive effect)",
            "H₀: β̂₁ = 0, H₁: β̂₁ ≠ 0 (stated about the sample estimate, not the population parameter)",
        ],
        "correct": 0,
        "explanation": "H₀ is always the precise claim about the "
        "*population* parameter β₁ — that the effect is exactly zero. "
        "Stating the hypothesis about the sample estimate β̂₁ instead is "
        "wrong, since we observe that directly and don't need to test "
        "it. H₁ is the general, two-sided alternative unless a "
        "one-tailed test is explicitly justified in advance.",
    },
    {
        "q": "What does a p-value actually represent?",
        "choices": [
            "The probability of observing a test statistic this extreme or more extreme, assuming H₀ is true",
            "The probability that H₀ is true, given the data",
            "The probability that H₁ is true",
            "The proportion of variance explained by the model",
        ],
        "correct": 0,
        "explanation": "The p-value is a statement about the data "
        "*assuming* H₀ — not a probability attached to H₀ or H₁ "
        "themselves. \"Probability H₀ is true\" is one of the most "
        "common misinterpretations of this quantity.",
    },
    {
        "q": "A test gives p = .12 at α = .05. What is the correct conclusion?",
        "choices": [
            "Fail to reject H₀ — there isn't enough evidence that the effect differs from zero",
            "Accept H₀ — the effect is exactly zero",
            "Reject H₀, because .12 is still fairly small",
            "The predictor should be dropped from the model, since its true effect has been shown to be zero",
        ],
        "correct": 0,
        "explanation": "We never \"accept\" H₀ or treat a non-significant "
        "result as proof the effect is exactly zero — we only fail to "
        "reject it. And p ≥ α means we don't reject H₀, regardless of "
        "whether p \"feels\" small in absolute terms: the comparison is "
        "to α, not to some intuitive sense of smallness.",
    },
    {
        "q": "A coefficient has Estimate = 1.3757 and Std. Error = 0.07989. What is its t-value?",
        "choices": ["17.22", "0.058 (inverts the division)", "1.456 (adds the two values)", "11.02 (divides by the intercept's SE of 1.1111 instead)"],
        "correct": 0,
        "explanation": "t = Estimate / Std. Error = 1.3757 / 0.07989 ≈ "
        "17.22. Each coefficient's t-value uses its own standard error — "
        "mixing in another row's SE from the same table is a common "
        "slip when reading a coefficients table.",
    },
    {
        "q": "A model with n = 150 observations has k = 2 predictors. What "
        "are the residual degrees of freedom for each coefficient's t-test?",
        "choices": ["147", "148 (forgets to subtract 1 for the intercept)", "150", "2"],
        "correct": 0,
        "explanation": "df = n − k − 1 = 150 − 2 − 1 = 147: one df is "
        "lost for each of the 2 slopes plus 1 more for the intercept.",
    },
    {
        "q": "What does R² measure?",
        "choices": [
            "The proportion of variance in the outcome accounted for by the predictors",
            "The proportion of variance in the outcome left unexplained",
            "The correlation between the predicted and observed values",
            "The probability that the model is correctly specified",
        ],
        "correct": 0,
        "explanation": "R² = SSModel / SSTotal = 1 − SSResidual/SSTotal. "
        "\"The correlation between predicted and observed values\" is a "
        "near-miss: that correlation equals √R² (not R² itself), so "
        "stating R² as that correlation directly is off by a square.",
    },
    {
        "q": "A model has SSTotal = 8556.06 and SSResidual = 2826.83. What is R²?",
        "choices": ["0.670", "0.330 (= SSResidual / SSTotal, the unexplained share)", "3.03 (inverts the ratio)", "1.0 (assumes a perfect fit)"],
        "correct": 0,
        "explanation": "R² = 1 − SSResidual/SSTotal = 1 − 2826.83/8556.06 "
        "≈ 1 − 0.330 = 0.670. Reporting the *unexplained* share "
        "(SSResidual/SSTotal) instead of subtracting it from 1 is the "
        "most common error.",
    },
    {
        "q": "Why is adjusted R² generally preferred over plain R² once a "
        "model has several predictors?",
        "choices": [
            "Plain R² never decreases when a predictor is added — even a useless one — while adjusted R² penalises for the number of predictors relative to sample size",
            "Adjusted R² is always larger than plain R², giving a more optimistic picture",
            "Plain R² cannot be computed once a model has more than one predictor",
            "Adjusted R² removes the need to run a significance test on the coefficients",
        ],
        "correct": 0,
        "explanation": "Random sampling fluctuation means R² can only "
        "rise (or stay flat) as predictors are added, regardless of "
        "whether they're genuinely useful — adjusted R² corrects for "
        "this and can decrease. It's typically *smaller*, not larger, "
        "than plain R² (the \"more optimistic\" claim has it backwards), "
        "and both are fully defined for any number of predictors.",
    },
    {
        "q": "A simple regression (n = 10, k = 1) has R² = 0.5201. What is the adjusted R²?",
        "choices": ["0.460", "0.520 (= plain R², unadjusted)", "0.579 (uses n/(n−k) instead of (n−1)/(n−k−1))", "0.400 (uses n/(n−k) and rounds down)"],
        "correct": 0,
        "explanation": "Adjusted R² = 1 − (1 − R²)(n − 1)/(n − k − 1) = "
        "1 − (0.4799 × 9/8) = 1 − 0.540 ≈ 0.460. Using the wrong "
        "denominator pair (n/(n−k) instead of (n−1)/(n−k−1)) is the "
        "classic formula mix-up.",
    },
    {
        "q": "A predictor has t = 2.39 with df = 147, where the two-tailed "
        "critical value at α = .05 is ±1.976. What do we conclude?",
        "choices": [
            "Reject H₀ — the coefficient is statistically significant, since |2.39| exceeds 1.976",
            "Fail to reject H₀ — 2.39 is too close to zero to matter",
            "We cannot decide significance from t alone — R² must also be checked first",
            "The test is invalid, because a t-value should never exceed 2",
        ],
        "correct": 0,
        "explanation": "Because |t| = 2.39 falls beyond the critical "
        "value of 1.976 in either tail, it lands in the rejection region, "
        "so H₀: β = 0 is rejected — this decision about one coefficient "
        "doesn't require consulting R² first, and there's no rule "
        "capping t-values at 2.",
    },
]

_W4: list[QuizQuestion] = [
    {
        "q": "The overall F-test compares your fitted model against which alternative?",
        "choices": [
            "An intercept-only model that predicts ȳ for every observation",
            "A model with every possible predictor included",
            "A model fitted on a different, independent sample",
            "The residual standard error on its own",
        ],
        "correct": 0,
        "explanation": "The F-ratio tests whether your predictors "
        "explain more variance than the \"no information\" baseline of "
        "always guessing the mean, ȳ — not against a maximal or "
        "independently-fitted model.",
    },
    {
        "q": "What is H₀ for the overall F-test in a model with predictors x₁ and x₂?",
        "choices": [
            "β₁ = β₂ = 0 (the intercept is not part of this null)",
            "β₀ = β₁ = β₂ = 0",
            "β₁ = 0 only, ignoring x₂",
            "R² = 1",
        ],
        "correct": 0,
        "explanation": "The overall F-test asks whether all the *slopes* "
        "are jointly zero — i.e. whether the predictors add nothing over "
        "an intercept-only model. The intercept itself isn't part of "
        "this null (folding it in is a common error), and the test "
        "covers the slopes jointly, not just one of them.",
    },
    {
        "q": "A model has SSModel = 5729.23 (dfModel = 2) and SSResidual = "
        "2826.83 (dfResidual = 147). What is the F-statistic?",
        "choices": ["148.9", "2.03 (= SSModel/SSResidual, skipping the df division entirely)", "19.23 (reports MSResidual alone)", "0.13 (inverts MSModel and MSResidual)"],
        "correct": 0,
        "explanation": "F = (SSModel/dfModel) / (SSResidual/dfResidual) "
        "= (5729.23/2) / (2826.83/147) = 2864.6 / 19.23 ≈ 148.9. Skipping "
        "the division by degrees of freedom on either side is the most "
        "common mistake.",
    },
    {
        "q": "An incremental F-test (R's `anova(m1, m2)`) is valid only when…",
        "choices": [
            "m1's predictors are a subset of m2's predictors, and both models are fitted on identical observations",
            "m2 simply has a higher R² than m1, regardless of which predictors each contains",
            "both models have exactly the same number of predictors",
            "m1 and m2 each contain exactly one predictor",
        ],
        "correct": 0,
        "explanation": "The test relies on the restricted model being a "
        "special case (nested within) the full model, fit to the same "
        "rows. A higher R² alone is not sufficient — two models can both "
        "fit well without being nested at all, in which case the F-test "
        "simply doesn't apply.",
    },
    {
        "q": "m0: y ~ x1 + x2 + x4. m1: y ~ x1 + x2 + x3. Are these models nested?",
        "choices": [
            "No — x3 and x4 each appear in only one model, so neither predictor set contains the other",
            "Yes — they share x1 and x2",
            "Yes — because they have the same number of predictors",
            "It depends on which model has the higher R²",
        ],
        "correct": 0,
        "explanation": "Nesting requires one model's entire predictor "
        "set to be contained within the other's. Sharing some predictors, "
        "or having equal predictor *counts*, isn't enough — each model "
        "here has one predictor the other lacks, so neither is a subset "
        "of the other.",
    },
    {
        "q": "Comparing two nested models gives Res.Df = 1758 (RSS = 4740.2) "
        "for the restricted model and Res.Df = 1753 (RSS = 4055.4) for "
        "the full model (5 added predictors). What is the F-statistic "
        "for this incremental test?",
        "choices": ["59.2", "136.97 (reports the numerator mean square alone)", "2.31 (reports the denominator mean square alone)", "684.85 (reports the raw Sum of Sq, without dividing by either df)"],
        "correct": 0,
        "explanation": "Sum of Sq = 4740.2 − 4055.4 = 684.85 over Df = 5 "
        "→ numerator MS = 136.97; residual MS = 4055.4/1753 ≈ 2.313. "
        "F = 136.97 / 2.313 ≈ 59.2 — both mean squares are needed, not "
        "just one of them.",
    },
    {
        "q": "Why can't an incremental F-test be used to compare two non-nested models?",
        "choices": [
            "The test's logic relies on one model being a restricted special case of the other, so the reduction in RSS has a known reference distribution; non-nested models aren't related this way",
            "Because non-nested models always end up fitted on different numbers of observations",
            "Because F-tests can only ever compare models with exactly one predictor",
            "Because R's `anova()` function has no way to accept two model objects at once",
        ],
        "correct": 0,
        "explanation": "It's the missing containment relationship that "
        "breaks the test, not sample size per se — two non-nested models "
        "can perfectly well be fit on exactly the same rows (mismatched "
        "sample sizes from dropped NAs is a real, but separate, pitfall). "
        "`anova()` does accept any two fitted model objects; it just "
        "produces a comparison that isn't statistically valid when "
        "they're non-nested.",
    },
    {
        "q": "Model A has AIC = 500.2 and Model B has AIC = 498.9, fitted on "
        "the same data. What can you correctly conclude?",
        "choices": [
            "Model B is preferred (lower AIC), but a single AIC value in isolation is meaningless — only the difference between models is informative",
            "Model B is preferred, and because the gap is a large negative improvement, it is almost certainly the true model",
            "Model A is preferred, since a higher AIC reflects a more complex and therefore more informative model",
            "Neither model can be preferred unless the two are nested",
        ],
        "correct": 0,
        "explanation": "Smaller AIC is better, but AIC has no absolute "
        "scale and no model is ever \"the true model\" — that's an "
        "overclaim. Higher AIC means a worse fit-versus-complexity "
        "trade-off, not more information. And unlike the F-test, AIC "
        "places no nesting requirement on the comparison.",
    },
    {
        "q": "Why does BIC tend to penalise extra predictors more heavily "
        "than AIC, for typical sample sizes?",
        "choices": [
            "BIC's penalty term k·ln(n) exceeds AIC's 2k whenever n ≥ 8, which covers nearly every real dataset, and the gap widens as n grows",
            "BIC always adds a fixed penalty of 10 points per predictor, regardless of sample size",
            "AIC ignores sample size entirely, while BIC does not use sums of squares at all",
            "BIC is only defined for non-nested models, where stronger penalties are needed",
        ],
        "correct": 0,
        "explanation": "AIC penalises complexity by 2k; BIC penalises by "
        "k·ln(n). Since ln(n) > 2 once n ≥ 8, BIC's penalty is almost "
        "always the larger one, and the gap grows further as the sample "
        "size increases — this is what makes BIC favour simpler models "
        "more strongly; it isn't a fixed per-predictor penalty.",
    },
    {
        "q": "Comparing nested models m1 (age + sex) and m2 (+ five "
        "personality traits) gives BIC(m1) = 6771.1 and BIC(m2) = 6533.7, "
        "so ΔBIC = 237.4. Using Raftery's (1995) guideline, what does "
        "this indicate?",
        "choices": [
            "Very strong evidence favouring m2, the model with the lower BIC",
            "No meaningful evidence of a difference between the models",
            "Very strong evidence favouring m1, since it was fitted first",
            "The comparison is invalid, because an incremental F-test would be needed here instead",
        ],
        "correct": 0,
        "explanation": "Raftery's guideline treats ΔBIC > 10 as very "
        "strong evidence for the model with the smaller BIC — here m2 "
        "(6533.7 < 6771.1). The models being nested doesn't make BIC "
        "invalid or require an F-test instead — AIC/BIC work for nested "
        "or non-nested comparisons alike, and here both methods happen "
        "to agree.",
    },
]

QUIZZES: dict[str, list[QuizQuestion]] = {
    "w1": _W1,
    "w2": _W2,
    "w3": _W3,
    "w4": _W4,
}
