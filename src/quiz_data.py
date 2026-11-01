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
        "(y = 3 + 2x + ε) both describe a linear relationship. What "
        "distinguishes the statistical model?",
        "choices": [
            "It adds an error term ε, so observed points scatter around the line instead of falling exactly on it",
            "It has a different slope",
            "It can only be fitted with lm() in R",
            "It has no intercept term",
        ],
        "correct": 0,
        "explanation": "A deterministic model predicts every point exactly; "
        "a statistical model adds ε to allow case-by-case variability — "
        "points scatter around the line rather than lying exactly on it.",
    },
    {
        "q": "In yᵢ = β₀ + β₁xᵢ + εᵢ, which symbols carry no subscript *i*?",
        "choices": [
            "β₀ and β₁",
            "yᵢ and xᵢ",
            "εᵢ only",
            "xᵢ and εᵢ",
        ],
        "correct": 0,
        "explanation": "β₀ and β₁ describe the single line fitted to the "
        "whole dataset, so they have one value shared by every "
        "observation. yᵢ, xᵢ and εᵢ each take a different value per "
        "participant i.",
    },
    {
        "q": "A fitted model has intercept β̂₀ = −5. What does this tell you?",
        "choices": [
            "The predicted value of y is −5 when x = 0",
            "y decreases by 5 units for every 1-unit increase in x",
            "The model is invalid because an intercept can't be negative",
            "The average value of y across the sample is −5",
        ],
        "correct": 0,
        "explanation": "The intercept is the value the line predicts when "
        "x = 0 — where it crosses the y-axis. It says nothing about the "
        "rate of change (that's the slope) or the mean of y.",
    },
    {
        "q": "A model has slope β̂₁ = −2. What does this mean?",
        "choices": [
            "Each 1-unit increase in x is associated with a 2-unit decrease in predicted y",
            "Each 1-unit increase in x is associated with a 2-unit increase in predicted y",
            "x and y are uncorrelated",
            "The line is flat (no relationship)",
        ],
        "correct": 0,
        "explanation": "The slope is the rate of change: a negative slope "
        "means predicted y falls as x rises. A slope of 0, not −2, would "
        "describe a flat line.",
    },
    {
        "q": "For observation i, the residual is εᵢ = yᵢ − ŷᵢ = +3. What does this tell you about that data point?",
        "choices": [
            "It lies above the fitted line — the model under-predicted it",
            "It lies below the fitted line — the model over-predicted it",
            "It lies exactly on the fitted line",
            "The model's slope is positive",
        ],
        "correct": 0,
        "explanation": "A positive residual means the observed value "
        "exceeded the model's prediction (yᵢ > ŷᵢ), so the point sits "
        "above the line — the model under-predicted it.",
    },
    {
        "q": "Why does the least-squares method square the residuals before summing them, rather than just summing them directly?",
        "choices": [
            "Unsquared residuals from the best-fitting line sum to exactly zero (positive and negative values cancel), and squaring penalises large deviations while giving a smooth, solvable function",
            "Squaring makes the calculation simpler to do by hand",
            "Residuals can only be negative, so squaring is needed to interpret them",
            "It converts the residuals into z-scores",
        ],
        "correct": 0,
        "explanation": "Without squaring, positive and negative residuals "
        "cancel out exactly for the least-squares line. Squaring makes "
        "every deviation positive, penalises bigger misses more, and "
        "yields a sum of squares that has a closed-form minimum.",
    },
    {
        "q": "A 10-student dataset has x̄ = 2.75, ȳ = 3.3, SSₓ = Σ(xᵢ−x̄)² = 20.625 "
        "and SPₓᵥ = Σ(xᵢ−x̄)(yᵢ−ȳ) = 21.75. What is the least-squares slope b₁?",
        "choices": ["1.05", "0.95", "6.25", "21.75"],
        "correct": 0,
        "explanation": "b₁ = SPₓᵥ / SSₓ = 21.75 / 20.625 ≈ 1.05.",
    },
    {
        "q": "Using the same dataset (x̄ = 2.75, ȳ = 3.3, b₁ ≈ 1.05), what is the intercept b₀?",
        "choices": ["0.40", "3.30", "2.90", "−0.40"],
        "correct": 0,
        "explanation": "b₀ = ȳ − b₁·x̄ = 3.3 − (1.05 × 2.75) ≈ 3.3 − 2.90 = 0.40. "
        "The fitted line always passes through the point of means (x̄, ȳ).",
    },
    {
        "q": "Which R code correctly fits a model predicting `score` from `hours` using the data frame `test`?",
        "choices": [
            "`lm(score ~ hours, data = test)`",
            "`lm(hours ~ score, data = test)`",
            "`lm(score + hours, data = test)`",
            "`lm(test ~ score + hours)`",
        ],
        "correct": 0,
        "explanation": "The outcome (DV) goes on the left of `~`, the "
        "predictor (IV) on the right: `lm(DV ~ IV, data = …)`. The other "
        "options either reverse DV and IV, omit `~` entirely, or use the "
        "dataset name as a variable.",
    },
    {
        "q": "A growth model predicts a 20-year-old's height as 10 metres, which is absurd. What statistical lesson does this illustrate?",
        "choices": [
            "Extrapolating a model far beyond the range of the observed data is risky",
            "Linear models should never be used for human growth data",
            "The slope must have been calculated incorrectly",
            "The intercept was negative",
        ],
        "correct": 0,
        "explanation": "A model is only validated within the range of "
        "data it was fitted on. Predictions far outside that range — like "
        "age 240 months for a model fitted on infants — can be nonsensical, "
        "even if the fit within range was good.",
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
            "The probability of passing increases by 1.0545 for every hour studied",
        ],
        "correct": 0,
        "explanation": "The standard template: \"for every one-[unit of X] "
        "increase in X, Y changes on average by β₁ [units of Y]\" — here, "
        "1 point of score per 1 hour studied.",
    },
    {
        "q": "A model regresses cat tail length (cm) on cat weight (kg) and finds an "
        "intercept of 14 cm. Why might this intercept be hard to interpret meaningfully?",
        "choices": [
            "Because x = 0 (a cat weighing 0 kg) doesn't occur in reality, so the prediction at x = 0 is outside any meaningful range",
            "Because intercepts are only meaningful for positive slopes",
            "Because tail length cannot be measured in cm",
            "Because the intercept should always equal the mean of y",
        ],
        "correct": 0,
        "explanation": "The intercept is the prediction at x = 0. If 0 is "
        "outside the plausible range of the predictor (no cat weighs 0 kg, "
        "no one is −2 years old, etc.), the intercept has no real-world "
        "meaning even though the model still needs it mathematically.",
    },
    {
        "q": "If both x and y are converted to z-scores before fitting, what does the "
        "resulting slope represent?",
        "choices": [
            "The standardised slope — equivalently, β₁ multiplied by SD(x)/SD(y)",
            "The same value as R², the proportion of variance explained",
            "The raw sum of squares of x",
            "The same value as the intercept",
        ],
        "correct": 0,
        "explanation": "Standardising before fitting (option 2 in the "
        "sheet) gives the same number as standardising the coefficient "
        "after fitting (option 1): β₁* = β₁ · sₓ/s_y.",
    },
    {
        "q": "After standardising both the predictor and outcome (converting both to "
        "z-scores), what happens to the intercept of the fitted line?",
        "choices": [
            "It becomes 0, because all variable means become 0",
            "It stays exactly the same as the unstandardised intercept",
            "It becomes equal to the slope",
            "It becomes undefined",
        ],
        "correct": 0,
        "explanation": "β₀ = ȳ − β₁x̄. Once both variables are standardised, "
        "x̄ = ȳ = 0, so β₀ = 0 − β₁·0 = 0 (up to rounding error).",
    },
    {
        "q": "Compared with the unstandardised model, what happens to the t-value, "
        "p-value, R² and F-statistic after standardising the variables?",
        "choices": [
            "They are unchanged — standardising only rescales the axes, not the strength of evidence",
            "They all become exactly 1",
            "The p-value increases while R² decreases",
            "They cannot be computed for standardised variables",
        ],
        "correct": 0,
        "explanation": "Standardisation is a linear rescaling of the same "
        "data and the same fitted relationship, so every test statistic, "
        "p-value, R² and F stays identical — only the units of the "
        "coefficients change.",
    },
    {
        "q": "In a simple regression with one continuous predictor, the standardised "
        "slope β₁* is exactly equal to…",
        "choices": [
            "Pearson's correlation coefficient r between x and y",
            "The unstandardised slope β₁",
            "The residual standard error",
            "The adjusted R²",
        ],
        "correct": 0,
        "explanation": "With one predictor, β₁* = r, and consequently "
        "R² = r² — this equivalence holds only for simple (one-predictor) "
        "regression, not multiple regression.",
    },
    {
        "q": "In multiple regression, a partial regression coefficient βⱼ represents "
        "the change in y for a one-unit increase in xⱼ…",
        "choices": [
            "Holding all other predictors in the model constant (controlling for them)",
            "Ignoring the effect of all other predictors",
            "Only when xⱼ is standardised",
            "Averaged across every possible combination of the other predictors' values",
        ],
        "correct": 0,
        "explanation": "\"Partial\" / \"controlling for\" means the model "
        "isolates xⱼ's unique contribution, holding the other predictors' "
        "values fixed — not ignoring them.",
    },
    {
        "q": "A model is: Score = 6.87 + 1.38·Hours + 0.92·Motivation (Motivation is "
        "standardised, mean 0). What score does it predict for a student who "
        "studied 3 hours with average motivation?",
        "choices": ["11.01", "6.87", "4.14", "9.19"],
        "correct": 0,
        "explanation": "6.87 + (1.38 × 3) + (0.92 × 0) = 6.87 + 4.14 = 11.01. "
        "Average motivation contributes 0 because the variable is "
        "standardised (mean 0).",
    },
    {
        "q": "Using the same model, student ID101 has hours = 2 and motivation = −1.42. "
        "What score does the model predict (before adding the residual)?",
        "choices": ["8.32", "7.00", "9.74", "6.87"],
        "correct": 0,
        "explanation": "ŷ = 6.87 + (1.38 × 2) + (0.92 × −1.42) = "
        "6.87 + 2.76 − 1.3064 ≈ 8.32.",
    },
    {
        "q": "Which of these is NOT a legitimate reason to add predictors to a "
        "regression model?",
        "choices": [
            "To artificially push an existing predictor's p-value below .05",
            "To improve prediction of the outcome",
            "To test a theory that several variables jointly contribute to the outcome",
            "To control for a confound while assessing one predictor's effect",
        ],
        "correct": 0,
        "explanation": "Prediction, theory-testing and covariate control "
        "are the standard motivations for multiple regression. Adding "
        "predictors purely to chase a smaller p-value isn't a valid "
        "research justification.",
    },
]

_W3: list[QuizQuestion] = [
    {
        "q": "For testing a slope coefficient, which pair of hypotheses is correct?",
        "choices": [
            "H₀: β₁ = 0, H₁: β₁ ≠ 0",
            "H₀: β₁ ≠ 0, H₁: β₁ = 0",
            "H₀: β₁ = 0, H₁: β₁ > 0 (always one-tailed)",
            "H₀: β₁ = β̂₁, H₁: β₁ ≠ β̂₁",
        ],
        "correct": 0,
        "explanation": "H₀ is always the precise claim — the effect is "
        "exactly zero. H₁ is the non-specific alternative that it isn't "
        "zero (two-tailed, unless a one-tailed test is explicitly justified).",
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
        "explanation": "The p-value is a statement about the data assuming "
        "H₀ — not a probability attached to H₀ or H₁ itself. "
        "\"Probability H₀ is true\" is a classic misinterpretation.",
    },
    {
        "q": "A test gives p = .12 at α = .05. What is the correct conclusion?",
        "choices": [
            "Fail to reject H₀ — there isn't enough evidence that the effect differs from zero",
            "Accept H₀ — the effect is exactly zero",
            "Reject H₀ because .12 rounds to \"small\"",
            "The test is inconclusive and must be re-run with more predictors",
        ],
        "correct": 0,
        "explanation": "We never \"accept\" H₀ — only reject it or fail to "
        "reject it. p ≥ α means the data don't provide sufficient evidence "
        "against H₀, not proof that H₀ is true.",
    },
    {
        "q": "A coefficient has Estimate = 1.3757 and Std. Error = 0.07989. What is its t-value?",
        "choices": ["17.22", "0.058", "1.456", "11.02"],
        "correct": 0,
        "explanation": "t = Estimate / Std. Error = 1.3757 / 0.07989 ≈ 17.22.",
    },
    {
        "q": "A model with n = 150 observations has k = 2 predictors. What are the "
        "residual degrees of freedom for each coefficient's t-test?",
        "choices": ["147", "148", "150", "2"],
        "correct": 0,
        "explanation": "df = n − k − 1 = 150 − 2 − 1 = 147 (one df lost for "
        "each of the k slopes plus the intercept).",
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
        "explanation": "R² = SSModel / SSTotal = 1 − SSResidual/SSTotal — "
        "the share of the outcome's total variability that the model's "
        "predictions account for.",
    },
    {
        "q": "A model has SSTotal = 8556.06 and SSResidual = 2826.83. What is R²?",
        "choices": ["0.670", "0.330", "3.03", "1.0"],
        "correct": 0,
        "explanation": "R² = 1 − SSResidual/SSTotal = 1 − 2826.83/8556.06 ≈ "
        "1 − 0.330 = 0.670.",
    },
    {
        "q": "Why is adjusted R² generally preferred over plain R² once a model has "
        "several predictors?",
        "choices": [
            "Plain R² never decreases when a predictor is added — even a useless one — while adjusted R² penalises for the number of predictors relative to sample size",
            "Adjusted R² is always larger than plain R²",
            "Plain R² cannot be computed with more than one predictor",
            "Adjusted R² removes the need for a significance test",
        ],
        "correct": 0,
        "explanation": "Random sampling fluctuation means R² can only "
        "rise (or stay flat) as predictors are added, regardless of "
        "whether they're genuinely useful. Adjusted R² corrects for this "
        "using both n and k, and can decrease.",
    },
    {
        "q": "A simple regression (n = 10, k = 1) has R² = 0.5201. What is the "
        "adjusted R²?",
        "choices": ["0.460", "0.520", "0.579", "0.400"],
        "correct": 0,
        "explanation": "Adjusted R² = 1 − (1 − R²)(n − 1)/(n − k − 1) = "
        "1 − (0.4799 × 9/8) = 1 − 0.540 ≈ 0.460.",
    },
    {
        "q": "A predictor has t = 2.39 with df = 147, where the two-tailed critical "
        "value at α = .05 is ±1.976. What do we conclude?",
        "choices": [
            "Reject H₀ — the coefficient is statistically significant, since |2.39| exceeds 1.976",
            "Fail to reject H₀ — 2.39 is too small to matter",
            "The test is invalid because t should never exceed 2",
            "We need the F-statistic before we can decide",
        ],
        "correct": 0,
        "explanation": "Because |t| = 2.39 falls beyond the critical value "
        "of 1.976 in either tail, the observed t lands in the rejection "
        "region, so H₀: β = 0 is rejected.",
    },
]

_W4: list[QuizQuestion] = [
    {
        "q": "The overall F-test compares your fitted model against which alternative?",
        "choices": [
            "An intercept-only model that predicts ȳ for every observation",
            "A model with every possible predictor included",
            "A model fitted on a different, independent sample",
            "The residual standard error alone",
        ],
        "correct": 0,
        "explanation": "The F-ratio tests whether your predictors explain "
        "more variance than the \"no information\" baseline of always "
        "guessing the mean, ȳ.",
    },
    {
        "q": "What is H₀ for the overall F-test in a model with predictors x₁ and x₂?",
        "choices": [
            "β₁ = β₂ = 0 (the intercept is not included in this null)",
            "β₀ = β₁ = β₂ = 0",
            "β₁ = 0 only",
            "R² = 1",
        ],
        "correct": 0,
        "explanation": "The overall F-test asks whether all the slopes "
        "are jointly zero — i.e. whether the predictors add nothing over "
        "the intercept-only model. The intercept itself isn't part of "
        "this null.",
    },
    {
        "q": "A model has SSModel = 5729.23 (dfModel = 2) and SSResidual = 2826.83 "
        "(dfResidual = 147). What is the F-statistic?",
        "choices": ["148.9", "2.03", "19.23", "0.13"],
        "correct": 0,
        "explanation": "F = (SSModel/dfModel) / (SSResidual/dfResidual) = "
        "(5729.23/2) / (2826.83/147) = 2864.6 / 19.23 ≈ 148.9.",
    },
    {
        "q": "An incremental F-test (R's `anova(m1, m2)`) is valid only when…",
        "choices": [
            "m1's predictors are a subset of m2's predictors, and both models are fitted on identical observations",
            "both models predict the same outcome variable, regardless of which predictors they use",
            "both models have the same R²",
            "m1 and m2 each contain exactly one predictor",
        ],
        "correct": 0,
        "explanation": "The incremental F-test relies on the restricted "
        "model being a special case (nested within) the full model, fit "
        "to the same rows. Two models predicting the same outcome but "
        "with unrelated predictor sets are non-nested, and the test "
        "doesn't apply.",
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
        "explanation": "Nesting requires one model's predictors to be "
        "entirely contained within the other's. Here each model has a "
        "predictor the other lacks (x4 vs x3), so neither is a subset — "
        "non-nested.",
    },
    {
        "q": "Comparing two nested models gives Res.Df = 1758 (RSS = 4740.2) for the "
        "restricted model and Res.Df = 1753 (RSS = 4055.4) for the full model "
        "(5 added predictors). What is the F-statistic for this incremental test?",
        "choices": ["59.2", "136.97", "2.31", "684.85"],
        "correct": 0,
        "explanation": "Sum of Sq = 4740.2 − 4055.4 = 684.85 over Df = 5 → "
        "MS = 136.97; residual MS = 4055.4/1753 ≈ 2.313. "
        "F = 136.97 / 2.313 ≈ 59.2.",
    },
    {
        "q": "Why can't an incremental F-test be used to compare two non-nested models?",
        "choices": [
            "The test's logic relies on one model being a restricted special case of the other, so the improvement in fit has a known reference distribution; non-nested models aren't related this way",
            "F-tests can only be computed for models with exactly one predictor",
            "Non-nested models never have the same sample size",
            "R's anova() function always errors out, with no exceptions, on any two-model comparison",
        ],
        "correct": 0,
        "explanation": "The incremental F-test's sampling distribution is "
        "derived from the full model containing the restricted model as a "
        "special case. Without that containment relationship, the reduction "
        "in RSS has no comparable reference distribution.",
    },
    {
        "q": "When comparing models with AIC or BIC, which model should you prefer?",
        "choices": [
            "The one with the smaller (lower) value",
            "The one with the larger (higher) value",
            "Whichever model was fitted first",
            "They cannot be compared unless the models are nested",
        ],
        "correct": 0,
        "explanation": "Smaller AIC/BIC indicates a better balance of fit "
        "and complexity. Unlike the F-test, AIC/BIC place no nesting "
        "requirement on the models being compared (though both must be "
        "fit on the same data).",
    },
    {
        "q": "Why does BIC tend to penalise extra predictors more heavily than AIC, "
        "for typical sample sizes?",
        "choices": [
            "BIC's penalty term k·ln(n) exceeds AIC's 2k whenever n ≥ 8, which covers nearly every real dataset, and the gap widens as n grows",
            "BIC always adds a fixed penalty of 10 points per predictor",
            "AIC ignores sample size entirely while BIC does not use sums of squares",
            "BIC is only defined for non-nested models",
        ],
        "correct": 0,
        "explanation": "AIC penalises complexity by 2k; BIC penalises by "
        "k·ln(n). Since ln(n) > 2 once n ≥ 8, BIC's penalty is almost "
        "always the larger one, and it grows further as the sample size "
        "increases — making BIC favour simpler models more strongly.",
    },
    {
        "q": "Comparing nested models m1 (age + sex) and m2 (+ five personality traits) "
        "gives BIC(m1) = 6771.1 and BIC(m2) = 6533.7, so ΔBIC = 237.4. Using Raftery's "
        "(1995) guideline, what does this indicate?",
        "choices": [
            "Very strong evidence favouring m2, the model with the lower BIC",
            "No meaningful evidence of a difference between the models",
            "Very strong evidence favouring m1",
            "The comparison is invalid because the models are non-nested",
        ],
        "correct": 0,
        "explanation": "Raftery's guideline treats ΔBIC > 10 as very "
        "strong evidence for the model with the smaller BIC — here m2 "
        "(6533.7 < 6771.1), consistent with the incremental F-test's "
        "conclusion.",
    },
]

QUIZZES: dict[str, list[QuizQuestion]] = {
    "w1": _W1,
    "w2": _W2,
    "w3": _W3,
    "w4": _W4,
}
