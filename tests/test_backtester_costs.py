"""Tests for the honest backtester cost model (the #9 fix).

The old backtester paid a flat +0.90 per bucket hit and -1.00 per miss, with no
fees/spread/fills, fictional P&L the strategy was tuned against. These tests
pin the central honesty property: a price-taker buying at a fair price has ZERO
expected edge before costs, so costs alone make it lose.
"""
from __future__ import annotations

from weather_edge.analysis.backtester import (
    BacktestCosts,
    _bucket_probability,
    _normal_cdf,
    _pnl_under_costs,
)


class TestNormalAndBucketProb:

    def test_cdf_symmetry(self):
        assert abs(_normal_cdf(0.0, 0.0, 1.0) - 0.5) < 1e-9

    def test_bucket_prob_open_ended_sums_to_one(self):
        below = _bucket_probability(20.0, 3.0, None, 15.0)
        middle = _bucket_probability(20.0, 3.0, 15.0, 25.0)
        above = _bucket_probability(20.0, 3.0, 25.0, None)
        assert abs((below + middle + above) - 1.0) < 1e-9

    def test_centered_bucket_is_most_likely(self):
        centered = _bucket_probability(20.0, 3.0, 18.0, 22.0)
        offset = _bucket_probability(20.0, 3.0, 28.0, 32.0)
        assert centered > offset


class TestPnlUnderCosts:

    def test_fair_price_zero_costs_is_zero_ev(self):
        """Buy at fair price p; win pays (1-p), loss pays -p. EV = p*(1-p) +
        (1-p)*(-p) = 0. No free money for a price-taker before costs."""
        costs = BacktestCosts(spread=0.0, fee_rate=0.0, fill_prob=1.0, payout=1.0)
        p = 0.30
        ev = p * _pnl_under_costs(True, p, costs) + (1 - p) * _pnl_under_costs(False, p, costs)
        assert abs(ev) < 1e-9

    def test_spread_makes_ev_negative(self):
        """Add a spread and the same fair-price bet now loses in expectation."""
        costs = BacktestCosts(spread=0.04, fee_rate=0.0, fill_prob=1.0, payout=1.0)
        p = 0.30
        ev = p * _pnl_under_costs(True, p, costs) + (1 - p) * _pnl_under_costs(False, p, costs)
        assert ev < 0
        assert abs(ev - (-0.04)) < 1e-9   # EV loss equals the spread

    def test_real_edge_can_overcome_costs(self):
        """Buying a bucket the crowd underpriced (entry < true prob) can win."""
        costs = BacktestCosts(spread=0.02, fee_rate=0.0, fill_prob=1.0, payout=1.0,
                              assumed_market_price=0.20)
        true_p = 0.40  # model thinks it's 0.40 but we paid 0.20
        ev = true_p * _pnl_under_costs(True, 0.20, costs) + (1 - true_p) * _pnl_under_costs(False, 0.20, costs)
        assert ev > 0

    def test_fill_probability_scales_pnl(self):
        full = BacktestCosts(spread=0.0, fee_rate=0.0, fill_prob=1.0)
        half = BacktestCosts(spread=0.0, fee_rate=0.0, fill_prob=0.5)
        assert abs(_pnl_under_costs(True, 0.3, half) - 0.5 * _pnl_under_costs(True, 0.3, full)) < 1e-9
