"""Power analysis: sample size and minimum detectable effect.

Pre-experiment planning is where most A/B tests are won or lost. These helpers
answer the two questions every experiment owner asks:

* "How many users do I need per arm?"  → :func:`sample_size_for_proportion`,
  :func:`sample_size_for_mean`.
* "Given the traffic I'll actually get, what's the smallest effect I can
  detect?"  → :func:`mde_for_proportion`, :func:`mde_for_mean`.

All functions assume a balanced two-arm design (equal split) and a two-sided
test unless told otherwise.
"""

from __future__ import annotations

import numpy as np
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize

__all__ = [
    "sample_size_for_proportion",
    "sample_size_for_mean",
    "mde_for_proportion",
    "mde_for_mean",
]


def sample_size_for_proportion(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    relative: bool = True,
    alternative: str = "two-sided",
) -> int:
    """Minimum sample size **per arm** for a two-proportion test.

    Parameters
    ----------
    baseline_rate
        Current conversion rate of the control arm, in (0, 1).
    mde
        Minimum detectable effect. By default this is **relative** (e.g.
        ``0.10`` means "detect a 10% lift", so 5% → 5.5%). Set
        ``relative=False`` to pass an absolute lift in rate points.
    alpha, power
        Significance level and desired statistical power.
    relative
        Whether ``mde`` is relative (default) or absolute.
    alternative
        ``"two-sided"`` (default) or ``"one-sided"``.

    Returns
    -------
    int
        Required users per arm, rounded up.
    """
    _validate_rate(baseline_rate, "baseline_rate")
    _validate_alpha_power(alpha, power)

    treatment_rate = baseline_rate * (1 + mde) if relative else baseline_rate + mde
    if not 0 < treatment_rate < 1:
        raise ValueError(
            f"Implied treatment rate {treatment_rate:.4f} is outside (0, 1). "
            "Check the baseline_rate and mde."
        )

    effect = proportion_effectsize(treatment_rate, baseline_rate)
    n = NormalIndPower().solve_power(
        effect_size=abs(effect),
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative=_alt(alternative),
    )
    return int(np.ceil(n))


def sample_size_for_mean(
    std: float,
    mde: float,
    baseline_mean: float | None = None,
    alpha: float = 0.05,
    power: float = 0.80,
    relative: bool = False,
    alternative: str = "two-sided",
) -> int:
    """Minimum sample size **per arm** for a means (t-test) comparison.

    Parameters
    ----------
    std
        Pooled/expected standard deviation of the metric.
    mde
        Minimum detectable effect. Absolute (in metric units) by default. If
        ``relative=True``, ``baseline_mean`` must be supplied and ``mde`` is
        interpreted as a fraction of it.
    baseline_mean
        Required only when ``relative=True``.
    alpha, power
        Significance level and desired statistical power.
    relative
        Whether ``mde`` is relative to ``baseline_mean``.
    alternative
        ``"two-sided"`` (default) or ``"one-sided"``.

    Returns
    -------
    int
        Required users per arm, rounded up.
    """
    _validate_alpha_power(alpha, power)
    if std <= 0:
        raise ValueError(f"std must be positive, got {std}.")

    if relative:
        if baseline_mean is None:
            raise ValueError("baseline_mean is required when relative=True.")
        abs_effect = abs(baseline_mean * mde)
    else:
        abs_effect = abs(mde)
    if abs_effect == 0:
        raise ValueError("mde resolves to a zero effect; cannot size an experiment.")

    cohens_d = abs_effect / std
    n = TTestIndPower().solve_power(
        effect_size=cohens_d,
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative=_alt(alternative),
    )
    return int(np.ceil(n))


def mde_for_proportion(
    baseline_rate: float,
    n_per_arm: int,
    alpha: float = 0.05,
    power: float = 0.80,
    relative: bool = True,
    alternative: str = "two-sided",
) -> float:
    """Smallest detectable lift for a proportion test at a fixed sample size.

    Inverts :func:`sample_size_for_proportion`: given the traffic you can
    realistically get per arm, returns the minimum effect detectable at the
    requested power. Returned as a relative lift by default, absolute if
    ``relative=False``.
    """
    _validate_rate(baseline_rate, "baseline_rate")
    _validate_alpha_power(alpha, power)
    if n_per_arm < 2:
        raise ValueError(f"n_per_arm must be >= 2, got {n_per_arm}.")

    effect = NormalIndPower().solve_power(
        nobs1=n_per_arm,
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative=_alt(alternative),
    )
    # Invert Cohen's h to recover the treatment rate above baseline.
    phi_c = 2 * np.arcsin(np.sqrt(baseline_rate))
    treatment_rate = np.sin((phi_c + effect) / 2) ** 2
    abs_lift = treatment_rate - baseline_rate
    return abs_lift / baseline_rate if relative else abs_lift


def mde_for_mean(
    std: float,
    n_per_arm: int,
    baseline_mean: float | None = None,
    alpha: float = 0.05,
    power: float = 0.80,
    relative: bool = False,
    alternative: str = "two-sided",
) -> float:
    """Smallest detectable effect for a means test at a fixed sample size.

    Inverts :func:`sample_size_for_mean`. Returns an absolute effect in metric
    units by default; if ``relative=True`` (requires ``baseline_mean``) it
    returns the effect as a fraction of the baseline mean.
    """
    _validate_alpha_power(alpha, power)
    if std <= 0:
        raise ValueError(f"std must be positive, got {std}.")
    if n_per_arm < 2:
        raise ValueError(f"n_per_arm must be >= 2, got {n_per_arm}.")

    cohens_d = TTestIndPower().solve_power(
        nobs1=n_per_arm,
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative=_alt(alternative),
    )
    abs_effect = cohens_d * std
    if relative:
        if baseline_mean is None or baseline_mean == 0:
            raise ValueError("baseline_mean (non-zero) is required when relative=True.")
        return abs_effect / baseline_mean
    return abs_effect


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _alt(alternative: str) -> str:
    if alternative == "two-sided":
        return "two-sided"
    if alternative in ("one-sided", "larger", "smaller"):
        return "larger"
    raise ValueError(f"alternative must be 'two-sided' or 'one-sided', got {alternative!r}.")


def _validate_rate(rate: float, label: str) -> None:
    if not 0 < rate < 1:
        raise ValueError(f"{label} must be in (0, 1), got {rate}.")


def _validate_alpha_power(alpha: float, power: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if not 0 < power < 1:
        raise ValueError(f"power must be in (0, 1), got {power}.")
