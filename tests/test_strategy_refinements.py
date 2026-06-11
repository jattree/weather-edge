"""Tests for the post-release strategy refinements.

Three changes, each grounded in the proving-run findings:

1. Bias significance gate: the 2026-04-01 validation showed a uniform
   correction helps cities with large real offsets but actively damages
   cities whose raw forecast is already excellent (London raw MAE 0.11C
   made 302% worse). Corrections within sampling noise are now skipped.
2. Kelly at the effective fill price: midpoint Kelly ignores the spread
   and over-sizes every bet.
3. Real spread gate: the HIGH-tier `spread_ok` flag was a hardcoded True
   placeholder; it now applies the same 40%-of-alpha convention as the
   fee gate.

Plus characterization tests for the existing ENSO regime shrinkage,
which previously had no coverage of its actual math.
"""
from __future__ import annotations

from datetime import datetime, timezone

from weather_edge.analysis.bias_correction import (
    MIN_SNAPSHOTS_FOR_BIAS,
    compute_gated_bias,
)
from weather_edge.analysis.edge import calculate_edge
from weather_edge.analysis.enso_regime import ENSOState, get_bias_shrinkage
from weather_edge.models.enums import SignalTier, TradeSide


# ---------------------------------------------------------------------------
# Bias significance gate
# ---------------------------------------------------------------------------

class TestBiasSignificanceGate:

    def test_large_consistent_bias_passes(self):
        """An HKG-style bias (large, consistent) must produce a correction."""
        errors = [-1.3, -1.2, -1.4, -1.3, -1.1, -1.5, -1.3] * 4  # n=28
        corr = compute_gated_bias(errors)
        assert corr.temp_max_offset > 1.0  # correction = -bias
        assert "dynamic" in corr.notes

    def test_small_noisy_bias_is_gated(self):
        """A London-style bias (tiny mean, real scatter) must be skipped."""
        # mean ~ -0.03C against ~0.8C scatter: pure noise
        errors = [0.8, -0.9, 0.7, -0.8, 0.9, -0.7, -0.1, 0.05, -0.2, 0.1,
                  0.75, -0.85, 0.6, -0.6, 0.5, -0.55, 0.3, -0.3, 0.2, -0.25]
        corr = compute_gated_bias(errors)
        assert corr.temp_max_offset == 0.0
        assert "gated" in corr.notes

    def test_insufficient_data_returns_zero(self):
        corr = compute_gated_bias([-1.0] * (MIN_SNAPSHOTS_FOR_BIAS - 1))
        assert corr.temp_max_offset == 0.0
        assert "insufficient" in corr.notes


# ---------------------------------------------------------------------------
# Kelly at the effective fill price + spread gate
# ---------------------------------------------------------------------------

def _edge_signal(spread: float, side_model_prob: float = 0.70):
    return calculate_edge(
        market_id="m1",
        model_prob=side_model_prob,
        market_prob=0.50,
        model_confidence=1.0,
        bankroll=2000.0,
        consensus_id=None,
        hours_to_resolution=2.0,
        city_id="nyc",
        target_date="2026-04-01",
        description="test",
        spread=spread,
    )


class TestKellyAtEffectivePrice:

    def test_spread_reduces_kelly(self):
        """Crossing a wide spread must shrink the Kelly fraction."""
        tight = _edge_signal(spread=0.0)
        wide = _edge_signal(spread=0.10)
        assert tight.recommended_side == TradeSide.YES
        assert wide.kelly_fraction < tight.kelly_fraction

    def test_spread_reduces_kelly_for_no_side(self):
        tight = _edge_signal(spread=0.0, side_model_prob=0.30)
        wide = _edge_signal(spread=0.10, side_model_prob=0.30)
        assert tight.recommended_side == TradeSide.NO
        assert wide.kelly_fraction < tight.kelly_fraction

    def test_zero_spread_keeps_positive_kelly(self):
        """Sanity: a real edge at zero spread still sizes a bet."""
        s = _edge_signal(spread=0.0)
        assert s.kelly_fraction > 0
        assert s.recommended_size > 0


class TestSpreadGate:

    def test_wide_spread_blocks_high_tier(self):
        """Half-spread eating >40% of the raw edge must demote the signal."""
        tight = _edge_signal(spread=0.0)
        assert tight.confidence_tier == SignalTier.HIGH
        # edge ~0.20; half-spread 0.10 > 0.4 * 0.20 = 0.08 -> not HIGH
        wide = _edge_signal(spread=0.20)
        assert wide.confidence_tier != SignalTier.HIGH

    def test_modest_spread_keeps_high_tier(self):
        # half-spread 0.02 <= 0.4 * edge(~0.20) -> HIGH retained
        s = _edge_signal(spread=0.04)
        assert s.confidence_tier == SignalTier.HIGH


# ---------------------------------------------------------------------------
# ENSO regime shrinkage (characterization of existing behavior)
# ---------------------------------------------------------------------------

def _enso(transitioning: bool) -> ENSOState:
    return ENSOState(
        phase="neutral" if transitioning else "la_nina",
        oni_value=-0.3 if transitioning else -1.2,
        transitioning=transitioning,
        confidence=0.6 if transitioning else 0.9,
        fetched_at=datetime.now(timezone.utc),
    )


class TestENSOShrinkage:

    def test_stable_regime_keeps_most_of_correction(self):
        """Stable regime: >= 0.7 of the correction survives everywhere."""
        for city in ("sea", "lon", "den", "unknown_city"):
            assert get_bias_shrinkage(city, _enso(transitioning=False)) >= 0.7

    def test_transition_shrinks_sensitive_cities_more(self):
        """Seattle (sensitivity 0.8) must shrink more than London (0.2)."""
        enso = _enso(transitioning=True)
        assert get_bias_shrinkage("sea", enso) < get_bias_shrinkage("lon", enso)

    def test_transition_floor_is_0_4(self):
        """Even maximum sensitivity never erases the correction entirely."""
        enso = _enso(transitioning=True)
        for city in ("sea", "sfo", "hou", "wlg"):
            assert get_bias_shrinkage(city, enso) >= 0.4

    def test_unknown_city_gets_moderate_default(self):
        """Unmapped cities use the 0.3 default sensitivity, not an extreme."""
        enso = _enso(transitioning=True)
        val = get_bias_shrinkage("xyz", enso)
        assert 0.7 <= val <= 0.9  # 1 - 0.3*0.6 = 0.82
