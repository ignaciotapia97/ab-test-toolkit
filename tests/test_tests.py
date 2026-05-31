"""Unit tests for ab_toolkit.tests."""

import numpy as np
import pytest
from scipy import stats

from ab_toolkit.tests import proportion_test, means_test, mann_whitney_test


class TestProportionTest:
    def test_matches_statsmodels_z_test(self):
        # Cross-check the z-statistic and p-value against statsmodels.
        from statsmodels.stats.proportion import proportions_ztest

        res = proportion_test(480, 10_000, 530, 10_000)
        stat, p = proportions_ztest([530, 480], [10_000, 10_000])
        assert res.statistic == pytest.approx(stat, rel=1e-6)
        assert res.p_value == pytest.approx(p, rel=1e-6)

    def test_clear_winner_is_significant(self):
        res = proportion_test(500, 10_000, 800, 10_000)
        assert res.significant
        assert res.absolute_effect == pytest.approx(0.03)
        assert res.relative_effect == pytest.approx(0.6)

    def test_no_difference_not_significant(self):
        res = proportion_test(500, 10_000, 505, 10_000)
        assert not res.significant
        assert res.p_value > 0.05

    def test_ci_contains_effect(self):
        res = proportion_test(480, 10_000, 530, 10_000)
        assert res.ci_low < res.absolute_effect < res.ci_high

    def test_one_sided_smaller_pvalue(self):
        two = proportion_test(480, 10_000, 530, 10_000, alternative="two-sided")
        one = proportion_test(480, 10_000, 530, 10_000, alternative="larger")
        assert one.p_value == pytest.approx(two.p_value / 2, rel=1e-6)

    @pytest.mark.parametrize(
        "args",
        [
            (480, 0, 530, 10_000),       # zero total
            (-1, 10_000, 530, 10_000),   # negative successes
            (11_000, 10_000, 5, 100),    # successes > total
        ],
    )
    def test_invalid_counts_raise(self, args):
        with pytest.raises(ValueError):
            proportion_test(*args)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            proportion_test(480, 10_000, 530, 10_000, alpha=1.5)


class TestMeansTest:
    def test_matches_scipy_welch(self):
        rng = np.random.default_rng(0)
        a = rng.normal(100, 15, 500)
        b = rng.normal(105, 15, 500)
        res = means_test(a, b)
        t, p = stats.ttest_ind(b, a, equal_var=False)
        assert res.statistic == pytest.approx(t, rel=1e-9)
        assert res.p_value == pytest.approx(p, rel=1e-9)

    def test_detects_real_shift(self):
        rng = np.random.default_rng(1)
        control = rng.normal(50, 10, 2000)
        treatment = rng.normal(53, 10, 2000)
        res = means_test(control, treatment, metric_name="revenue")
        assert res.significant
        assert res.treatment_estimate > res.control_estimate
        assert res.effect_size is not None and res.effect_size > 0

    def test_no_shift_not_significant(self):
        rng = np.random.default_rng(0)
        control = rng.normal(50, 10, 2000)
        treatment = rng.normal(50, 10, 2000)
        res = means_test(control, treatment)
        assert not res.significant

    def test_cohens_d_known_value(self):
        # Two groups separated by exactly one pooled SD → d ≈ 1.
        control = np.array([0.0, 0.0, 0.0, 0.0])  # use larger constructed sample
        rng = np.random.default_rng(3)
        a = rng.normal(0, 1, 5000)
        b = rng.normal(1, 1, 5000)
        res = means_test(a, b)
        assert res.effect_size == pytest.approx(1.0, abs=0.1)

    def test_too_few_observations_raise(self):
        with pytest.raises(ValueError):
            means_test([1.0], [1.0, 2.0, 3.0])

    def test_non_finite_raises(self):
        with pytest.raises(ValueError):
            means_test([1.0, 2.0, np.nan], [1.0, 2.0, 3.0])


class TestMannWhitney:
    def test_matches_scipy(self):
        rng = np.random.default_rng(4)
        a = rng.exponential(1.0, 300)
        b = rng.exponential(1.3, 300)
        res = mann_whitney_test(a, b)
        u, p = stats.mannwhitneyu(b, a, alternative="two-sided")
        assert res.statistic == pytest.approx(u)
        assert res.p_value == pytest.approx(p)

    def test_reports_medians(self):
        a = [1, 2, 3, 4, 5]
        b = [10, 20, 30, 40, 50]
        res = mann_whitney_test(a, b)
        assert res.control_estimate == 3
        assert res.treatment_estimate == 30
        assert res.significant

    def test_rank_biserial_in_range(self):
        rng = np.random.default_rng(5)
        res = mann_whitney_test(rng.normal(0, 1, 200), rng.normal(0.5, 1, 200))
        assert -1 <= res.effect_size <= 1


def test_result_str_is_readable():
    res = proportion_test(480, 10_000, 530, 10_000, metric_name="conversion")
    text = str(res)
    assert "conversion" in text
    assert "p-value" in text
    assert "%" in text  # proportions formatted as percentages
