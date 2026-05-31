"""ab_toolkit — a small, rigorous toolkit for A/B test analysis.

End-to-end workflow:

1. **Plan** the experiment with :mod:`ab_toolkit.power`
   (sample size / minimum detectable effect).
2. **Test** the result with :mod:`ab_toolkit.tests`
   (proportions, means, non-parametric).
3. **Correct** for multiple metrics with :mod:`ab_toolkit.corrections`
   (Bonferroni, Benjamini-Hochberg).
4. **Report** with :mod:`ab_toolkit.report`.

Example
-------
>>> from ab_toolkit import sample_size_for_proportion, proportion_test, generate_report
>>> n = sample_size_for_proportion(baseline_rate=0.05, mde=0.10)
>>> result = proportion_test(480, 10_000, 530, 10_000)
>>> _ = generate_report(result, metric_name="Checkout Conversion")
"""

from .power import (
    mde_for_mean,
    mde_for_proportion,
    sample_size_for_mean,
    sample_size_for_proportion,
)
from .tests import TestResult, mann_whitney_test, means_test, proportion_test
from .corrections import CorrectionResult, benjamini_hochberg, bonferroni
from .report import generate_report, interpret

__version__ = "0.1.0"

__all__ = [
    # power
    "sample_size_for_proportion",
    "sample_size_for_mean",
    "mde_for_proportion",
    "mde_for_mean",
    # tests
    "TestResult",
    "proportion_test",
    "means_test",
    "mann_whitney_test",
    # corrections
    "CorrectionResult",
    "bonferroni",
    "benjamini_hochberg",
    # report
    "generate_report",
    "interpret",
]
