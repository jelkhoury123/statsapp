# Regression Lab

A small teaching app for first-year psychology students. It shows how the
intercept and slope of a simple regression move from sample to sample when
every sample is drawn from the same fixed population.

The population is 100,000 students in `data/students_population.csv`.
Sampling never rebuilds that file and never writes over it.

## Setup

Python 3.11 or newer. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Start the app

```bash
streamlit run app.py
```

Streamlit prints a local address, usually http://localhost:8501.

## The population

Hours studied were generated from a uniform distribution on 0 to 10.
Exam scores were generated from

```text
exam score = 20 + 5 × hours studied + individual error
```

with the error drawn from a normal distribution with mean 0 and standard
deviation 5. Scores are not clipped to 0–100; the noise is narrow enough
that this population stays inside that range.

Those generating values are not the benchmark used in the lab. The file is
one finite population, so its own least-squares line is slightly different.
Computed from the file, that line is about

```text
exam score = 20.018 + 4.994 × hours studied
```

The sampling experiments that follow the introduction judge estimates against
this population line. The app computes it from the CSV at startup and checks
it against `data/population_metadata.json`.

`individual_error` is the error term that was used when the file was created.
Sampling does not draw a new error. It selects rows that already exist.

## Sampling without replacement

A sample of size n is a simple random sample: every subset of n students is
equally likely, and a student appears at most once in that sample.

The next sample is drawn independently from the whole population. A student
who was in the previous sample can appear again. Sample size is never allowed
to exceed 100,000.

Each sample is fit by ordinary least squares with an intercept:

```text
slope = sum((x − x̄)(y − ȳ)) / sum((x − x̄)²)
intercept = ȳ − slope × x̄
```

## Population coefficients and sample coefficients

The population intercept and slope are one fixed pair, computed once on all
100,000 students.

A sample intercept and slope are a different pair, computed on the students
who happened to be drawn. A new sample produces a new pair. The histograms
in Sampling distributions are the sampling distributions of those pairs. The
standard deviation of a histogram approximates the standard error of that
coefficient: how far a single sample estimate typically falls from the
population value.

## Tabs

**Population → Sample → Estimation.** The whole population is on the chart
first, as a cached grey cloud of all 100,000 students. Draw a sample and
those students are highlighted in blue on the same chart. Estimate the
regression from that sample alone, then reveal the population line and the
difference. Another sample keeps the population fixed and clears the sample
line until you estimate again. View Data opens the population and sample
tables beside the chart. Choosing a row highlights that student. It does
not change the fitted line.

**Sampling distributions.** Repeat the sample many times. One histogram shows
the estimated slopes, the other the estimated intercepts. The spread is the
sampling variation. It is not a hypothesis test.

## Random seed

Open Advanced settings and set a seed if you want a simulation to be
repeatable. The same seed and the same settings reproduce the same
simulation. The generator is not restarted when you toggle a chart option.
Without a seed, each run continues a random sequence, so a new sample is
actually new.

## Tests

With the virtual environment active, from this directory:

```bash
pytest
```

The tests check the population line against the metadata, check the
least-squares formulas against NumPy, check that sampled student IDs are
unique, check that the opening experiment fits only the sampled rows and
leaves the population coefficients unchanged, check that repeated samples
are approximately unbiased for the population coefficients, check that the
estimates get more precise as the sample grows, check a zero-variance
predictor, and check that the original CSV is unchanged.
