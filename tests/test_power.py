"""Unit tests for ab_toolkit.power."""

import pytest

from ab_toolkit.power import (
    sample_size_for_proportion,
    sample_size_for_mean,
    mde_for_proportion,
    mde_for_mean,
)


class TestSampleSizeProportion:
    def test_returns_positive_int(self):
        n = sample_size_for_proportion(baseline_rate=0.05, mde=0.10)
        assert isinstance(n, int) and n > 0

    def test_known_ballpark(self):
        # 5% baseline, 10% relative lift (5%→5.5%), alpha=.05, power=.80.
        # Standard references put this near ~31k per arm.
        n = sample_size_for_proportion(0.05, 0.10, alpha=0.05, power=0.80)
        assert 29_000 < n < 33_000

    def test_smaller_effect_needs_more_users(self):
        big = sample_size_for_proportion(0.05, 0.20)
        small = sample_size_for_proportion(0.05, 0.05)
        assert small > big

    def test_higher_power_needs_more_users(self):
        p80 = sample_size_for_proportion(0.05, 0.10, power=0.80)
        p95 = sample_size_for_proportion(0.05, 0.10, power=0.95)
        assert p95 > p80

    def test_absolute_mde(self):
        n_rel = sample_size_for_proportion(0.10, 0.10, relative=True)   # →0.11
        n_abs = sample_size_for_proportion(0.10, 0.01, relative=False)  # →0.11
        assert n_rel == n_abs

    def test_invalid_baseline_raises(self):
        with pytest.raises(ValueError):
            sample_size_for_proportion(1.5, 0.10)

    def test_impossible_treatment_rate_raises(self):
        with pytest.raises(ValueError):
            sample_size_for_proportion(0.9, 0.50)  # implies rate 1.35


class TestSampleSizeMean:
    def test_returns_positive_int(self):
        n = sample_size_for_mean(std=20.0, mde=2.0)
        assert isinstance(n, int) and n > 0

    def test_relative_requires_baseline(self):
        with pytest.raises(ValueError):
            sample_size_for_mean(std=20.0, mde=0.1, relative=True)

    def test_relative_matches_absolute(self):
        n_abs = sample_size_for_mean(std=20.0, mde=5.0)
        n_rel = sample_size_for_mean(std=20.0, mde=0.1, baseline_mean=50.0, relative=True)
        assert n_abs == n_rel  # both → effect of 5.0

    def test_bigger_std_needs_more_users(self):
        small = sample_size_for_mean(std=10.0, mde=2.0)
        big = sample_size_for_mean(std=30.0, mde=2.0)
        assert big > small


class TestMDE:
    def test_proportion_roundtrip(self):
        # Size for a target, then recover ~that MDE from the sample size.
        n = sample_size_for_proportion(0.05, 0.10, power=0.80)
        mde = mde_for_proportion(0.05, n, power=0.80)
        assert mde == pytest.approx(0.10, rel=0.05)

    def test_mean_roundtrip(self):
        n = sample_size_for_mean(std=20.0, mde=2.0, power=0.80)
        mde = mde_for_mean(std=20.0, n_per_arm=n, power=0.80)
        assert mde == pytest.approx(2.0, rel=0.05)

    def test_more_users_smaller_mde(self):
        big_n = mde_for_proportion(0.05, 100_000)
        small_n = mde_for_proportion(0.05, 10_000)
        assert big_n < small_n
