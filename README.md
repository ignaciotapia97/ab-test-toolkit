# A/B Test Analysis Toolkit

A Python package for rigorous, end-to-end A/B test analysis — from sample size planning to significance testing and results reporting.

---

## Why This Exists

Running A/B tests is straightforward. Running them **correctly** is not.

Common pitfalls:
- Underpowered tests that can't detect real effects
- Peeking at results before reaching the required sample size
- Testing multiple metrics without correcting for false discovery rate
- Reporting statistical significance without practical significance (effect size)

This toolkit standardizes the analysis workflow used by data teams at product companies. It covers every stage from pre-experiment power analysis to post-experiment reporting.

---

## Features

| Module | What it does |
|--------|-------------|
| `power` | Calculate minimum sample size given desired power, significance level, and MDE |
| `tests` | Run significance tests for proportions (conversion rate) and means (revenue, time) |
| `corrections` | Apply Bonferroni or Benjamini-Hochberg correction when testing multiple metrics |
| `report` | Generate a structured summary report with interpretation guidance |

---

## Quick Start

```python
from ab_toolkit.power import sample_size_for_proportion
from ab_toolkit.tests import proportion_test, means_test
from ab_toolkit.report import generate_report

# 1. Pre-experiment: how many users do we need?
n = sample_size_for_proportion(
    baseline_rate=0.05,   # current conversion rate
    mde=0.10,             # minimum detectable effect (relative)
    alpha=0.05,
    power=0.80
)
print(f"Required sample per variant: {n:,}")

# 2. Post-experiment: was the effect significant?
result = proportion_test(
    control_conversions=480,
    control_total=10000,
    treatment_conversions=530,
    treatment_total=10000,
    alpha=0.05
)
print(result)

# 3. Generate a readable report
generate_report(result, metric_name="Checkout Conversion Rate")
```

---

## Worked Examples

See the `examples/` folder for full end-to-end analyses (executed, with plots and outputs):

> 📓 If GitHub's notebook preview is slow or fails to render, open them via
> [**nbviewer**](https://nbviewer.org/github/ignaciotapia97/ab-test-toolkit/tree/main/examples/) ·
> [**Colab**](https://colab.research.google.com/github/ignaciotapia97/ab-test-toolkit) — both render straight from this repo.

- **`01_conversion_rate_test.ipynb`** — Testing whether a new checkout flow increases conversion rate. Covers power analysis, test execution, and result interpretation.
- **`02_revenue_per_user_test.ipynb`** — Testing whether a pricing change affects average revenue per user. Uses means testing and handles skewed distributions with Mann-Whitney.

---

## Project Structure

```
ab_test_toolkit/
├── ab_toolkit/
│   ├── __init__.py
│   ├── power.py           # Sample size and power calculations
│   ├── tests.py           # Significance tests (proportions, means, non-parametric)
│   ├── corrections.py     # Multiple testing corrections (Bonferroni, BH)
│   └── report.py          # Auto-generate a structured summary report
├── tests/                 # pytest suite (cross-checked against scipy / statsmodels)
│   ├── test_power.py
│   ├── test_tests.py
│   ├── test_corrections.py
│   └── test_report.py
├── examples/
│   ├── 01_conversion_rate_test.ipynb
│   └── 02_revenue_per_user_test.ipynb
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Statistical Methods Covered

### Significance Tests
- **Proportions:** Two-proportion z-test (conversion rates, click-through rates)
- **Means:** Welch's t-test for continuous metrics (revenue, session duration)
- **Non-parametric:** Mann-Whitney U test for skewed distributions

### Power Analysis
- Minimum sample size for proportion tests
- Minimum sample size for means tests
- Sensitivity analysis: how MDE changes with sample size

### Multiple Testing Correction
- **Bonferroni** — conservative, controls family-wise error rate
- **Benjamini-Hochberg** — less conservative, controls false discovery rate (recommended for large metric suites)

### Effect Size
- Cohen's d for continuous metrics
- Relative lift (%) for proportion metrics
- Confidence intervals on the estimated effect

---

## Installation

```bash
git clone https://github.com/ignacio-tapia/ab-test-toolkit.git
cd ab-test-toolkit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # editable install + dev tools (pytest, jupyter)
```

The editable install puts `ab_toolkit` on the path so both the test suite and the
example notebooks import it without any path hacking.

---

## Running the Tests

The suite cross-checks the toolkit's statistics against `scipy` and `statsmodels`
(e.g. the z-test against `proportions_ztest`, Welch's t against `scipy.stats.ttest_ind`),
verifies power-analysis round-trips (size → MDE → size), and checks the
multiple-testing corrections against textbook examples.

```bash
pytest -q
# 48 passed
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| scipy | Statistical tests |
| statsmodels | Power analysis, OLS |
| pandas / numpy | Data handling |
| matplotlib | Visualizations |
| pytest | Unit tests |

---

## Author

**Ignacio Tapia** — Senior Data Analyst with 5+ years running and analyzing experiments in marketplace and delivery platforms (Uber, PedidosYa, Front).
[LinkedIn](https://linkedin.com/in/ignacio-tapia) · [GitHub](https://github.com/ignacio-tapia)
