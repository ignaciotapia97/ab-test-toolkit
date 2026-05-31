"""Significance tests for A/B experiments.

Three workhorses cover the vast majority of online experiments:

* ``proportion_test`` — two-proportion z-test for binary metrics
  (conversion rate, click-through rate, retention).
* ``means_test`` — Welch's t-test for continuous metrics
  (revenue per user, session duration, items per order).
* ``mann_whitney_test`` — rank-based fallback for heavily skewed
  continuous metrics where the t-test's normality assumption is shaky.

Every test returns a :class:`TestResult`, a plain dataclass that prints as a
readable summary and is consumed by :func:`ab_toolkit.report.generate_report`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

__all__ = ["TestResult", "proportion_test", "means_test", "mann_whitney_test"]


@dataclass
class TestResult:
    """Outcome of a single A/B significance test.

    Attributes are deliberately generic so the proportion, means, and
    non-parametric tests can all share one container. ``effect_size`` is
    Cohen's d for means tests and ``None`` for proportion tests (which report
    relative lift instead).
    """

    metric_type: str               # "proportion" | "mean"
    test_name: str                 # human-readable name of the test used
    control_estimate: float        # control rate or mean
    treatment_estimate: float      # treatment rate or mean
    absolute_effect: float         # treatment - control
    relative_effect: float         # (treatment - control) / control
    statistic: float               # z or t statistic
    p_value: float
    alpha: float
    ci_low: float                  # CI on the absolute effect
    ci_high: float
    n_control: int
    n_treatment: int
    effect_size: Optional[float] = None   # Cohen's d (means tests only)
    metric_name: str = "metric"
    extra: dict = field(default_factory=dict)

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    @property
    def confidence_level(self) -> float:
        return 1 - self.alpha

    def __str__(self) -> str:
        verdict = "SIGNIFICANT" if self.significant else "not significant"
        pct = self.metric_type == "proportion"
        fmt = (lambda x: f"{x:.4%}") if pct else (lambda x: f"{x:,.4f}")
        lines = [
            f"A/B Test Result — {self.metric_name}",
            f"  Test:          {self.test_name}",
            f"  Control:       {fmt(self.control_estimate)}  (n={self.n_control:,})",
            f"  Treatment:     {fmt(self.treatment_estimate)}  (n={self.n_treatment:,})",
            f"  Abs. effect:   {fmt(self.absolute_effect)}",
            f"  Rel. effect:   {self.relative_effect:+.2%}",
            f"  {int(self.confidence_level * 100)}% CI (abs): "
            f"[{fmt(self.ci_low)}, {fmt(self.ci_high)}]",
            f"  Statistic:     {self.statistic:.4f}",
            f"  p-value:       {self.p_value:.4g}  (alpha={self.alpha})",
        ]
        if self.effect_size is not None:
            lines.append(f"  Cohen's d:     {self.effect_size:.4f}")
        lines.append(f"  Verdict:       {verdict} at {int(self.confidence_level*100)}% confidence")
        return "\n".join(lines)


def proportion_test(
    control_conversions: int,
    control_total: int,
    treatment_conversions: int,
    treatment_total: int,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    metric_name: str = "conversion rate",
) -> TestResult:
    """Two-proportion z-test for a binary metric.

    Parameters
    ----------
    control_conversions, treatment_conversions
        Count of successes (e.g. users who converted) in each arm.
    control_total, treatment_total
        Total users exposed in each arm.
    alpha
        Significance level. The confidence interval is ``1 - alpha``.
    alternative
        ``"two-sided"`` (default), ``"larger"``, or ``"smaller"``. The
        alternative describes the treatment relative to control.
    metric_name
        Label used in the printed summary and report.

    Notes
    -----
    The z-statistic uses the **pooled** proportion for the standard error
    (standard for hypothesis testing), while the confidence interval on the
    absolute difference uses the **unpooled** standard error (standard for
    estimation). This is the conventional pairing and avoids a CI that
    disagrees with the test at the boundary in misleading ways.
    """
    _validate_counts(control_conversions, control_total, "control")
    _validate_counts(treatment_conversions, treatment_total, "treatment")
    _validate_alpha(alpha)

    p_c = control_conversions / control_total
    p_t = treatment_conversions / treatment_total
    diff = p_t - p_c

    # Pooled SE for the test statistic.
    p_pool = (control_conversions + treatment_conversions) / (control_total + treatment_total)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / control_total + 1 / treatment_total))
    if se_pool == 0:
        raise ValueError("Standard error is zero; cannot run the test (no variation in data).")
    z = diff / se_pool

    p_value = _p_from_stat(z, alternative, dist="norm")

    # Unpooled SE for the CI on the difference.
    se_unpooled = np.sqrt(
        p_c * (1 - p_c) / control_total + p_t * (1 - p_t) / treatment_total
    )
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_low, ci_high = diff - z_crit * se_unpooled, diff + z_crit * se_unpooled

    return TestResult(
        metric_type="proportion",
        test_name=f"Two-proportion z-test ({alternative})",
        control_estimate=p_c,
        treatment_estimate=p_t,
        absolute_effect=diff,
        relative_effect=(diff / p_c) if p_c else float("nan"),
        statistic=z,
        p_value=p_value,
        alpha=alpha,
        ci_low=ci_low,
        ci_high=ci_high,
        n_control=control_total,
        n_treatment=treatment_total,
        effect_size=None,
        metric_name=metric_name,
        extra={"pooled_proportion": p_pool},
    )


def means_test(
    control,
    treatment,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    equal_var: bool = False,
    metric_name: str = "mean",
) -> TestResult:
    """Welch's t-test for a continuous metric (unequal variances by default).

    Parameters
    ----------
    control, treatment
        Array-like of per-user observations for each arm.
    alpha
        Significance level.
    alternative
        ``"two-sided"`` (default), ``"larger"``, or ``"smaller"``.
    equal_var
        If ``True`` use Student's t-test (pooled variance). Default ``False``
        (Welch) is the safer choice when arm variances may differ.
    metric_name
        Label used in the printed summary and report.

    Returns a :class:`TestResult` including Cohen's d as the effect size.
    """
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    _validate_sample(control, "control")
    _validate_sample(treatment, "treatment")
    _validate_alpha(alpha)

    n_c, n_t = control.size, treatment.size
    mean_c, mean_t = control.mean(), treatment.mean()
    var_c, var_t = control.var(ddof=1), treatment.var(ddof=1)
    diff = mean_t - mean_c

    scipy_alt = {"two-sided": "two-sided", "larger": "greater", "smaller": "less"}[alternative]
    t_stat, p_value = stats.ttest_ind(
        treatment, control, equal_var=equal_var, alternative=scipy_alt
    )

    # Standard error and degrees of freedom for the CI on the difference.
    if equal_var:
        dof = n_c + n_t - 2
        pooled_var = ((n_c - 1) * var_c + (n_t - 1) * var_t) / dof
        se = np.sqrt(pooled_var * (1 / n_c + 1 / n_t))
    else:
        se = np.sqrt(var_c / n_c + var_t / n_t)
        # Welch–Satterthwaite degrees of freedom.
        dof = (var_c / n_c + var_t / n_t) ** 2 / (
            (var_c / n_c) ** 2 / (n_c - 1) + (var_t / n_t) ** 2 / (n_t - 1)
        )
    t_crit = stats.t.ppf(1 - alpha / 2, dof)
    ci_low, ci_high = diff - t_crit * se, diff + t_crit * se

    # Cohen's d using the pooled standard deviation.
    pooled_sd = np.sqrt(((n_c - 1) * var_c + (n_t - 1) * var_t) / (n_c + n_t - 2))
    cohens_d = diff / pooled_sd if pooled_sd > 0 else float("nan")

    return TestResult(
        metric_type="mean",
        test_name=("Welch's t-test" if not equal_var else "Student's t-test")
        + f" ({alternative})",
        control_estimate=mean_c,
        treatment_estimate=mean_t,
        absolute_effect=diff,
        relative_effect=(diff / mean_c) if mean_c else float("nan"),
        statistic=float(t_stat),
        p_value=float(p_value),
        alpha=alpha,
        ci_low=ci_low,
        ci_high=ci_high,
        n_control=n_c,
        n_treatment=n_t,
        effect_size=cohens_d,
        metric_name=metric_name,
        extra={"dof": float(dof), "control_std": np.sqrt(var_c), "treatment_std": np.sqrt(var_t)},
    )


def mann_whitney_test(
    control,
    treatment,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    metric_name: str = "metric",
) -> TestResult:
    """Mann-Whitney U test for skewed continuous metrics.

    A rank-based test of whether one arm tends to produce larger values than
    the other. Use it when a continuous metric is heavily skewed (revenue,
    time-on-page) and the t-test's normality assumption is questionable.

    The reported "estimate" for each arm is its **median**, and the effect
    size is the rank-biserial correlation derived from the U statistic. No
    closed-form CI on the difference is provided (the test is on rank
    distributions, not means), so the CI bounds are ``nan``.
    """
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    _validate_sample(control, "control")
    _validate_sample(treatment, "treatment")
    _validate_alpha(alpha)

    n_c, n_t = control.size, treatment.size
    med_c, med_t = np.median(control), np.median(treatment)

    scipy_alt = {"two-sided": "two-sided", "larger": "greater", "smaller": "less"}[alternative]
    u_stat, p_value = stats.mannwhitneyu(treatment, control, alternative=scipy_alt)

    # Rank-biserial correlation: effect size in [-1, 1].
    rank_biserial = 1 - (2 * u_stat) / (n_c * n_t)

    return TestResult(
        metric_type="mean",
        test_name=f"Mann-Whitney U test ({alternative})",
        control_estimate=med_c,
        treatment_estimate=med_t,
        absolute_effect=med_t - med_c,
        relative_effect=((med_t - med_c) / med_c) if med_c else float("nan"),
        statistic=float(u_stat),
        p_value=float(p_value),
        alpha=alpha,
        ci_low=float("nan"),
        ci_high=float("nan"),
        n_control=n_c,
        n_treatment=n_t,
        effect_size=float(rank_biserial),
        metric_name=metric_name,
        extra={"effect_size_kind": "rank-biserial correlation", "uses_median": True},
    )


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _p_from_stat(stat: float, alternative: str, dist: str = "norm") -> float:
    d = stats.norm if dist == "norm" else dist
    if alternative == "two-sided":
        return float(2 * d.sf(abs(stat)))
    if alternative == "larger":
        return float(d.sf(stat))
    if alternative == "smaller":
        return float(d.cdf(stat))
    raise ValueError(f"Unknown alternative {alternative!r}; use 'two-sided', 'larger', or 'smaller'.")


def _validate_counts(successes: int, total: int, label: str) -> None:
    if total <= 0:
        raise ValueError(f"{label}_total must be positive, got {total}.")
    if successes < 0 or successes > total:
        raise ValueError(
            f"{label}_conversions must be in [0, {label}_total], got {successes}/{total}."
        )


def _validate_sample(arr: np.ndarray, label: str) -> None:
    if arr.size < 2:
        raise ValueError(f"{label} needs at least 2 observations, got {arr.size}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} contains non-finite values (nan/inf).")


def _validate_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
