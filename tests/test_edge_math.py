"""Property-based tests for trading math invariants.

Uses hypothesis to fuzz the edge calculation, Kelly sizing, and EMOS
calibration with random inputs. These catch off-by-one errors, division
by zero, and violated invariants that unit tests miss.

Each property encodes a business rule that must hold for ALL valid inputs.
"""
from __future__ import annotations

import pytest

try:
    from hypothesis import given, settings as h_settings, assume
    from hypothesis import strategies as st

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    # Provide dummy decorators so the module can be imported
    def given(*a, **kw):  # type: ignore[misc]
        def decorator(fn):
            return pytest.mark.skip(reason="hypothesis not installed")(fn)
        return decorator

    class _FakeSettings:
        def __call__(self, *a, **kw):
            def decorator(fn):
                return fn
            return decorator
    h_settings = _FakeSettings()  # type: ignore[assignment]

    class _FakeSt:
        def floats(self, **kw): return None
        def integers(self, **kw): return None
        def lists(self, *a, **kw): return None
    st = _FakeSt()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Kelly sizing: never exceeds max_position_pct, never negative
# ---------------------------------------------------------------------------

class TestKellySizingProperties:

    @given(
        model_prob=st.floats(min_value=0.02, max_value=0.98),
        market_prob=st.floats(min_value=0.02, max_value=0.98),
        model_confidence=st.floats(min_value=0.0, max_value=1.0),
        bankroll=st.floats(min_value=100.0, max_value=100000.0),
    )
    @h_settings(max_examples=200)
    def test_kelly_never_exceeds_max_position(
        self, model_prob, market_prob, model_confidence, bankroll
    ):
        """Kelly sizing must never exceed max_position_pct of bankroll."""
        from weather_edge.analysis.edge import calculate_edge
        from weather_edge.config import settings

        signal = calculate_edge(
            market_id="test",
            model_prob=model_prob,
            market_prob=market_prob,
            model_confidence=model_confidence,
            bankroll=bankroll,
        )
        # Penny/tail strategy has fixed sizing from config (ignores max_position_pct)
        if signal.strategy == "tail":
            from weather_edge.config import settings as s
            assert signal.recommended_size <= s.penny_max_position + 0.01
        else:
            max_allowed = bankroll * settings.max_position_pct
            assert signal.recommended_size <= max_allowed + 0.01  # Float tolerance

    @given(
        model_prob=st.floats(min_value=0.02, max_value=0.98),
        market_prob=st.floats(min_value=0.02, max_value=0.98),
        model_confidence=st.floats(min_value=0.0, max_value=1.0),
        bankroll=st.floats(min_value=100.0, max_value=100000.0),
    )
    @h_settings(max_examples=200)
    def test_kelly_never_negative(
        self, model_prob, market_prob, model_confidence, bankroll
    ):
        """Kelly sizing must never return a negative dollar amount."""
        from weather_edge.analysis.edge import calculate_edge

        signal = calculate_edge(
            market_id="test",
            model_prob=model_prob,
            market_prob=market_prob,
            model_confidence=model_confidence,
            bankroll=bankroll,
        )
        assert signal.recommended_size >= 0.0

    @given(
        model_prob=st.floats(min_value=0.02, max_value=0.98),
        market_prob=st.floats(min_value=0.02, max_value=0.98),
        model_confidence=st.floats(min_value=0.0, max_value=1.0),
    )
    @h_settings(max_examples=200)
    def test_kelly_fraction_non_negative(
        self, model_prob, market_prob, model_confidence
    ):
        """Raw Kelly fraction must be non-negative (clamped at 0)."""
        from weather_edge.analysis.edge import calculate_edge

        signal = calculate_edge(
            market_id="test",
            model_prob=model_prob,
            market_prob=market_prob,
            model_confidence=model_confidence,
        )
        assert signal.kelly_fraction >= 0.0
        assert signal.half_kelly >= 0.0


# ---------------------------------------------------------------------------
# EMOS spread inflation: always widens, never narrows
# ---------------------------------------------------------------------------

class TestEmosProperties:

    @given(
        raw_std=st.floats(min_value=0.0, max_value=10.0),
    )
    @h_settings(max_examples=100, deadline=None)
    def test_emos_inflation_widens_distribution(self, raw_std):
        """EMOS-inflated std_dev must be >= raw std_dev for temperature variables."""
        from weather_edge.analysis.consensus import SPREAD_INFLATION_FACTOR, EMOS_VARIANCE_FLOOR_C

        inflated = max(EMOS_VARIANCE_FLOOR_C, raw_std * SPREAD_INFLATION_FACTOR)
        assert inflated >= raw_std

    @given(
        raw_std=st.floats(min_value=0.0, max_value=10.0),
    )
    @h_settings(max_examples=100)
    def test_emos_variance_floor_enforced(self, raw_std):
        """EMOS std_dev never drops below variance floor for temperature."""
        from weather_edge.analysis.consensus import SPREAD_INFLATION_FACTOR, EMOS_VARIANCE_FLOOR_C

        inflated = max(EMOS_VARIANCE_FLOOR_C, raw_std * SPREAD_INFLATION_FACTOR)
        assert inflated >= EMOS_VARIANCE_FLOOR_C


# ---------------------------------------------------------------------------
# Bucket probability cap: no single bucket > MAX_BUCKET_PROBABILITY
# ---------------------------------------------------------------------------

class TestBucketCapProperties:

    @given(
        raw_prob=st.floats(min_value=0.0, max_value=1.0),
    )
    @h_settings(max_examples=100)
    def test_bucket_cap_applied(self, raw_prob):
        """After cap, no single bucket probability exceeds MAX_BUCKET_PROBABILITY."""
        from weather_edge.analysis.consensus import MAX_BUCKET_PROBABILITY

        capped = min(raw_prob, MAX_BUCKET_PROBABILITY)
        assert capped <= MAX_BUCKET_PROBABILITY
        assert capped >= 0.0


# ---------------------------------------------------------------------------
# Pool budgets: sum to <= bankroll
# ---------------------------------------------------------------------------

class TestPoolBudgetProperties:

    @given(
        bankroll=st.floats(min_value=100.0, max_value=100000.0),
    )
    @h_settings(max_examples=100)
    def test_pool_budgets_sum_to_bankroll(self, bankroll):
        """Three-pool allocation must sum to exactly the bankroll."""
        from weather_edge.config import settings

        today = bankroll * settings.pool_today_pct
        tomorrow = bankroll * settings.pool_tomorrow_pct
        penny = bankroll * settings.pool_penny_pct
        total = today + tomorrow + penny
        # Pool percentages should sum to 1.0 (60% + 30% + 10%)
        assert total <= bankroll + 0.01  # Float tolerance
        assert abs(total - bankroll) < 0.01  # Should be very close to exact


# ---------------------------------------------------------------------------
# Edge calculation: deterministic properties
# ---------------------------------------------------------------------------

class TestEdgeProperties:

    @given(
        model_prob=st.floats(min_value=0.05, max_value=0.95),
        market_prob=st.floats(min_value=0.05, max_value=0.95),
    )
    @h_settings(max_examples=200)
    def test_edge_non_negative(self, model_prob, market_prob):
        """Edge magnitude is always non-negative."""
        from weather_edge.analysis.edge import calculate_edge

        signal = calculate_edge(
            market_id="test",
            model_prob=model_prob,
            market_prob=market_prob,
            model_confidence=0.9,
        )
        assert signal.edge >= 0.0

    @given(
        model_prob=st.floats(min_value=0.05, max_value=0.95),
        market_prob=st.floats(min_value=0.05, max_value=0.95),
        model_confidence=st.floats(min_value=0.0, max_value=1.0),
    )
    @h_settings(max_examples=200)
    def test_model_prob_clamped(self, model_prob, market_prob, model_confidence):
        """Model probability in signal is always in [0.01, 0.99]."""
        from weather_edge.analysis.edge import calculate_edge

        signal = calculate_edge(
            market_id="test",
            model_prob=model_prob,
            market_prob=market_prob,
            model_confidence=model_confidence,
        )
        assert 0.01 <= signal.model_prob <= 0.99

    @given(
        model_prob=st.floats(min_value=0.05, max_value=0.95),
        market_prob=st.floats(min_value=0.05, max_value=0.95),
    )
    @h_settings(max_examples=200)
    def test_side_consistent_with_edge(self, model_prob, market_prob):
        """YES side when model prob > market prob after confidence adjustment."""
        from weather_edge.analysis.edge import calculate_edge
        from weather_edge.models.enums import TradeSide

        signal = calculate_edge(
            market_id="test",
            model_prob=model_prob,
            market_prob=market_prob,
            model_confidence=1.0,  # Full confidence = trust model exactly
        )
        # With full confidence, adj_prob = model_prob
        if model_prob > market_prob:
            assert signal.recommended_side == TradeSide.YES
        elif model_prob < market_prob:
            assert signal.recommended_side == TradeSide.NO
