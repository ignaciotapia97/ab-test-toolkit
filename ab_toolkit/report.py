"""Human-readable reporting for test results.

:func:`generate_report` turns a :class:`~ab_toolkit.tests.TestResult` into a
plain-language summary an experiment owner or stakeholder can act on. It pairs
statistical significance with practical significance and adds an explicit
recommendation, because "p < 0.05" on its own is not a decision.
"""

from __future__ import annotations

from .tests import TestResult

__all__ = ["generate_report", "interpret"]


def interpret(result: TestResult) -> str:
    """Return a one-line plain-language interpretation of a result."""
    direction = "increase" if result.absolute_effect > 0 else "decrease"
    if not result.significant:
        return (
            f"No statistically significant difference detected "
            f"(p = {result.p_value:.4g} ≥ {result.alpha}). "
            f"The observed {result.relative_effect:+.2%} change is within the range "
            f"explainable by chance."
        )
    return (
        f"Statistically significant {direction} of {abs(result.relative_effect):.2%} "
        f"(p = {result.p_value:.4g} < {result.alpha}). The treatment is likely the cause."
    )


def _recommendation(result: TestResult) -> str:
    if not result.significant:
        return (
            "DO NOT SHIP on this metric alone. The data does not support a real effect. "
            "Consider whether the test was adequately powered before concluding 'no effect'."
        )
    if result.absolute_effect > 0:
        return (
            "SHIP CANDIDATE. The treatment shows a significant positive effect. "
            "Confirm the effect is practically meaningful and check guardrail metrics "
            "before rolling out."
        )
    return (
        "DO NOT SHIP. The treatment shows a significant negative effect on this metric."
    )


def generate_report(
    result: TestResult,
    metric_name: str | None = None,
    practical_threshold: float | None = None,
    return_string: bool = False,
) -> str:
    """Build (and by default print) a structured summary report.

    Parameters
    ----------
    result
        A :class:`~ab_toolkit.tests.TestResult` from any test in
        :mod:`ab_toolkit.tests`.
    metric_name
        Override the metric label carried on the result.
    practical_threshold
        Optional minimum **relative** effect considered business-meaningful
        (e.g. ``0.02`` for a 2% lift). When provided, the report flags results
        that are statistically significant but below this bar.
    return_string
        If ``True`` return the report text instead of printing it.
    """
    name = metric_name or result.metric_name
    pct = result.metric_type == "proportion"
    fmt = (lambda x: f"{x:.4%}") if pct else (lambda x: f"{x:,.4f}")
    conf = int(result.confidence_level * 100)

    bar = "=" * 64
    lines = [
        bar,
        f"  A/B TEST REPORT — {name}",
        bar,
        "",
        "  RESULTS",
        f"    Test used .......... {result.test_name}",
        f"    Control ............ {fmt(result.control_estimate)}  (n = {result.n_control:,})",
        f"    Treatment .......... {fmt(result.treatment_estimate)}  (n = {result.n_treatment:,})",
        f"    Absolute effect .... {fmt(result.absolute_effect)}",
        f"    Relative effect .... {result.relative_effect:+.2%}",
    ]
    if result.ci_low == result.ci_low:  # not nan
        lines.append(
            f"    {conf}% CI (absolute) . [{fmt(result.ci_low)}, {fmt(result.ci_high)}]"
        )
    lines += [
        f"    p-value ............ {result.p_value:.4g}  (alpha = {result.alpha})",
    ]
    if result.effect_size is not None:
        kind = result.extra.get("effect_size_kind", "Cohen's d")
        lines.append(f"    Effect size ........ {result.effect_size:.4f}  ({kind})")

    lines += ["", "  INTERPRETATION", f"    {interpret(result)}"]

    if practical_threshold is not None and result.significant:
        if abs(result.relative_effect) < practical_threshold:
            lines += [
                "",
                "  ⚠ PRACTICAL SIGNIFICANCE",
                f"    The effect is statistically significant but the {result.relative_effect:+.2%} "
                f"change is below your {practical_threshold:.2%} practical threshold. "
                "Weigh the implementation cost against a small expected gain.",
            ]
        else:
            lines += [
                "",
                "  PRACTICAL SIGNIFICANCE",
                f"    The {result.relative_effect:+.2%} effect meets your "
                f"{practical_threshold:.2%} practical threshold.",
            ]

    lines += ["", "  RECOMMENDATION", f"    {_recommendation(result)}", "", bar]
    report = "\n".join(lines)

    if return_string:
        return report
    print(report)
    return report
