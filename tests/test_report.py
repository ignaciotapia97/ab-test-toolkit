"""Unit tests for ab_toolkit.report."""

from ab_toolkit.tests import proportion_test, means_test
from ab_toolkit.report import generate_report, interpret


def test_report_contains_key_sections():
    res = proportion_test(480, 10_000, 530, 10_000, metric_name="conversion")
    text = generate_report(res, return_string=True)
    for section in ("A/B TEST REPORT", "RESULTS", "INTERPRETATION", "RECOMMENDATION"):
        assert section in text
    assert "conversion" in text


def test_interpret_significant():
    res = proportion_test(500, 10_000, 900, 10_000)
    msg = interpret(res)
    assert "significant" in msg.lower()
    assert "increase" in msg.lower()


def test_interpret_not_significant():
    res = proportion_test(500, 10_000, 502, 10_000)
    msg = interpret(res)
    assert "no statistically significant" in msg.lower()


def test_practical_threshold_flag():
    # Significant but tiny effect → should warn about practical significance.
    res = proportion_test(50_000, 1_000_000, 51_000, 1_000_000)
    text = generate_report(res, practical_threshold=0.05, return_string=True)
    assert res.significant
    assert "PRACTICAL SIGNIFICANCE" in text


def test_report_handles_means_with_effect_size():
    import numpy as np

    rng = np.random.default_rng(0)
    res = means_test(rng.normal(50, 10, 1000), rng.normal(53, 10, 1000), metric_name="ARPU")
    text = generate_report(res, return_string=True)
    assert "Effect size" in text
    assert "ARPU" in text
