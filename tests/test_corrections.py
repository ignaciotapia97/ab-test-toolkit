"""Unit tests for ab_toolkit.corrections."""

import pytest

from ab_toolkit.corrections import bonferroni, benjamini_hochberg


class TestBonferroni:
    def test_adjusts_by_count(self):
        res = bonferroni([0.01, 0.02, 0.03])
        assert res.adjusted_p_values[0] == pytest.approx(0.03)
        assert res.adjusted_p_values[2] == pytest.approx(0.09)

    def test_caps_at_one(self):
        res = bonferroni([0.5, 0.6])
        assert all(p <= 1.0 for p in res.adjusted_p_values)

    def test_rejection(self):
        res = bonferroni([0.001, 0.5, 0.9], alpha=0.05)
        assert res.rejected == [True, False, False]
        assert res.n_significant == 1


class TestBenjaminiHochberg:
    def test_known_example(self):
        # Classic BH example p-values.
        p = [0.005, 0.011, 0.02, 0.04, 0.13]
        res = benjamini_hochberg(p, alpha=0.05)
        # With m=5, threshold p_(k) <= (k/5)*0.05; first four pass.
        assert res.rejected == [True, True, True, True, False]

    def test_more_powerful_than_bonferroni(self):
        p = [0.001, 0.013, 0.021, 0.04]
        bh = benjamini_hochberg(p, alpha=0.05)
        bonf = bonferroni(p, alpha=0.05)
        assert bh.n_significant >= bonf.n_significant

    def test_adjusted_p_monotone_in_rank(self):
        p = [0.04, 0.001, 0.02]
        res = benjamini_hochberg(p)
        # Smallest raw p should get the smallest (or tied) adjusted p.
        assert res.adjusted_p_values[1] <= res.adjusted_p_values[2]
        assert res.adjusted_p_values[2] <= res.adjusted_p_values[0]

    def test_order_preserved(self):
        p = [0.5, 0.001, 0.2]
        res = benjamini_hochberg(p)
        assert res.p_values == p  # input order untouched


def test_accepts_test_results():
    from ab_toolkit.tests import proportion_test

    r1 = proportion_test(500, 10_000, 800, 10_000, metric_name="conversion")
    r2 = proportion_test(500, 10_000, 505, 10_000, metric_name="retention")
    res = benjamini_hochberg([r1, r2])
    assert res.labels == ["conversion", "retention"]
    assert res.rejected[0] is True


def test_empty_raises():
    with pytest.raises(ValueError):
        bonferroni([])


def test_out_of_range_pvalue_raises():
    with pytest.raises(ValueError):
        bonferroni([0.5, 1.5])
