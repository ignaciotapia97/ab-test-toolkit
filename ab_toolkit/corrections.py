"""Multiple-testing corrections.

When an experiment is evaluated on several metrics at once, the chance of at
least one false positive grows with the number of tests. These functions adjust
for that.

* :func:`bonferroni` — controls the family-wise error rate (FWER). Conservative;
  use when any single false positive is costly.
* :func:`benjamini_hochberg` — controls the false discovery rate (FDR). More
  powerful; the recommended default for a large suite of secondary metrics.

Both accept either a list of raw p-values or a list of :class:`~ab_toolkit.tests.TestResult`
objects, and return a :class:`CorrectionResult`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np

from .tests import TestResult

__all__ = ["CorrectionResult", "bonferroni", "benjamini_hochberg"]

PValues = Union[Sequence[float], Sequence[TestResult]]


@dataclass
class CorrectionResult:
    """Outcome of a multiple-testing correction over a family of tests."""

    method: str
    alpha: float
    labels: list[str]
    p_values: list[float]            # original, in input order
    adjusted_p_values: list[float]   # adjusted, in input order
    rejected: list[bool]             # reject H0 after correction, in input order

    @property
    def n_significant(self) -> int:
        return sum(self.rejected)

    def __str__(self) -> str:
        width = max((len(l) for l in self.labels), default=6)
        header = f"Multiple-testing correction — {self.method} (alpha={self.alpha})"
        rows = [
            f"  {'metric'.ljust(width)}   raw p     adj p     reject",
            f"  {'-' * width}   -------   -------   ------",
        ]
        for label, p, adj, rej in zip(
            self.labels, self.p_values, self.adjusted_p_values, self.rejected
        ):
            rows.append(
                f"  {label.ljust(width)}   {p:<7.4g}   {adj:<7.4g}   "
                f"{'YES' if rej else 'no'}"
            )
        rows.append(f"  → {self.n_significant} of {len(self.labels)} significant after correction")
        return "\n".join([header, *rows])


def _unpack(p_values: PValues) -> tuple[list[float], list[str]]:
    """Accept raw floats or TestResults; return (p-values, labels)."""
    raw, labels = [], []
    for i, item in enumerate(p_values):
        if isinstance(item, TestResult):
            raw.append(item.p_value)
            labels.append(item.metric_name)
        else:
            raw.append(float(item))
            labels.append(f"metric_{i + 1}")
    if not raw:
        raise ValueError("p_values is empty; nothing to correct.")
    if any(not (0 <= p <= 1) for p in raw):
        raise ValueError("All p-values must be in [0, 1].")
    return raw, labels


def bonferroni(p_values: PValues, alpha: float = 0.05) -> CorrectionResult:
    """Bonferroni correction (controls FWER).

    Adjusted p-value is ``min(1, p * m)`` where ``m`` is the number of tests;
    a hypothesis is rejected when its adjusted p-value is below ``alpha``.
    """
    raw, labels = _unpack(p_values)
    m = len(raw)
    adjusted = [min(1.0, p * m) for p in raw]
    rejected = [adj < alpha for adj in adjusted]
    return CorrectionResult("Bonferroni", alpha, labels, raw, adjusted, rejected)


def benjamini_hochberg(p_values: PValues, alpha: float = 0.05) -> CorrectionResult:
    """Benjamini-Hochberg step-up procedure (controls FDR).

    Ranks the p-values, finds the largest ``k`` with ``p_(k) <= (k/m) * alpha``,
    and rejects all hypotheses up to that rank. Adjusted p-values are the
    standard monotone BH q-values, reported in the original input order.
    """
    raw, labels = _unpack(p_values)
    m = len(raw)
    order = np.argsort(raw)                 # indices that sort p ascending
    sorted_p = np.asarray(raw)[order]
    ranks = np.arange(1, m + 1)

    # BH adjusted p-values, enforced monotone non-decreasing from the top.
    adj_sorted = sorted_p * m / ranks
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0, 1)

    # Rejection threshold: largest k with p_(k) <= (k/m)*alpha.
    below = sorted_p <= (ranks / m) * alpha
    k = np.max(np.where(below)[0]) + 1 if below.any() else 0
    reject_sorted = np.zeros(m, dtype=bool)
    if k > 0:
        reject_sorted[:k] = True

    # Scatter back to original order.
    adjusted = np.empty(m)
    rejected = np.empty(m, dtype=bool)
    adjusted[order] = adj_sorted
    rejected[order] = reject_sorted

    return CorrectionResult(
        "Benjamini-Hochberg", alpha, labels, raw, adjusted.tolist(), rejected.tolist()
    )
