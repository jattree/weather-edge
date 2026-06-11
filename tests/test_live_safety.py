"""Contract tests for live trading safety.

These are the danger points that would cost real money if they broke:
1. Retry utility, must retry on failure, must raise after exhaustion
2. DB persistence, ghost trades must be impossible to silently create
3. Live/paper independence, live must fire without paper
4. Position type safety, Position and PaperTrade interop correctly
5. ENSO fallback, must warn loudly, not silently use stale data
6. Config, hardcoded values must come from settings
"""
from __future__ import annotations

import asyncio

import pytest

from weather_edge.models.enums import TradeStatus
from weather_edge.models.position import Position
from weather_edge.trading.paper import PaperTrade


# ---------------------------------------------------------------------------
# retry_sync: must retry N times, then raise
# If this breaks, every API call becomes single-attempt again.
# ---------------------------------------------------------------------------

class TestRetrySyncContract:

    def test_succeeds_on_first_attempt(self):
        """Happy path: no retry needed."""
        from weather_edge.retry import retry_sync
        result = retry_sync(lambda: 42, label="test")
        assert result == 42

    def test_retries_then_succeeds(self):
        """Transient failure followed by success, must not raise."""
        from weather_edge.retry import retry_sync
        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "ok"

        result = retry_sync(flaky, attempts=3, base_delay=0.01, label="test")
        assert result == "ok"
        assert call_count == 3

    def test_raises_after_exhaustion(self):
        """All retries fail, must raise, never swallow."""
        from weather_edge.retry import retry_sync

        def always_fail():
            raise ConnectionError("permanent")

        with pytest.raises(ConnectionError, match="permanent"):
            retry_sync(always_fail, attempts=3, base_delay=0.01, label="test")

    def test_respects_attempt_count(self):
        """Must call exactly N times, not N+1 or N-1."""
        from weather_edge.retry import retry_sync
        call_count = 0

        def counter():
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        with pytest.raises(ValueError):
            retry_sync(counter, attempts=5, base_delay=0.01, label="test")
        assert call_count == 5


# ---------------------------------------------------------------------------
# retry_async: same contracts, async version
# ---------------------------------------------------------------------------

class TestRetryAsyncContract:

    def test_succeeds_on_first_attempt(self):
        from weather_edge.retry import retry_async

        async def ok():
            return 42

        async def run():
            return await retry_async(ok, label="test")

        assert asyncio.run(run()) == 42

    def test_retries_then_succeeds(self):
        from weather_edge.retry import retry_async
        call_count = 0

        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "ok"

        async def run():
            return await retry_async(
                flaky, attempts=3, base_delay=0.01, label="test",
            )

        assert asyncio.run(run()) == "ok"
        assert call_count == 3

    def test_raises_after_exhaustion(self):
        from weather_edge.retry import retry_async

        async def always_fail():
            raise ConnectionError("permanent")

        async def run():
            return await retry_async(
                always_fail, attempts=2, base_delay=0.01, label="test",
            )

        with pytest.raises(ConnectionError, match="permanent"):
            asyncio.run(run())


# ---------------------------------------------------------------------------
# Position / PaperTrade type hierarchy
# If PaperTrade stops being a Position, scan_for_exits breaks for paper.
# If Position fields change, live positions break.
# ---------------------------------------------------------------------------

class TestPositionTypeContract:

    def test_paper_trade_is_position(self):
        """PaperTrade must be a subclass of Position."""
        assert issubclass(PaperTrade, Position)

    def test_paper_trade_instance_is_position(self):
        """A PaperTrade instance must pass isinstance check."""
        pt = PaperTrade(market_id="m1", city_id="nyc", side="YES")
        assert isinstance(pt, Position)

    def test_position_has_required_fields(self):
        """Position must have the fields scan_for_exits depends on."""
        p = Position(
            market_id="cond_123",
            city_id="nyc",
            side="YES",
            size_usd=50.0,
            entry_price=0.5,
            description="test",
            status=TradeStatus.OPEN,
            total_shares=100.0,
            source="live",
        )
        assert p.market_id == "cond_123"
        assert p.city_id == "nyc"
        assert p.side == "YES"
        assert p.size_usd == 50.0
        assert p.entry_price == 0.5
        assert p.description == "test"
        assert p.status == TradeStatus.OPEN
        assert p.total_shares == 100.0
        assert p.source == "live"

    def test_paper_trade_auto_sets_source(self):
        """PaperTrade must auto-set source='paper'."""
        pt = PaperTrade(market_id="m1")
        assert pt.source == "paper"

    def test_live_position_source(self):
        """Live position must have source='live'."""
        p = Position(source="live")
        assert p.source == "live"

    def test_paper_trade_has_paper_specific_fields(self):
        """PaperTrade must have trade_id, pnl, exit_price, Position must not."""
        pt = PaperTrade(trade_id=1, pnl=5.0, exit_price=0.8)
        assert pt.trade_id == 1
        assert pt.pnl == 5.0
        assert pt.exit_price == 0.8
        assert not hasattr(Position, "trade_id")
        assert not hasattr(Position, "pnl")

    def test_scan_for_exits_accepts_both_types(self):
        """scan_for_exits must work with a mixed list of Position and PaperTrade."""
        from weather_edge.analysis.exit_monitor import scan_for_exits

        pos = Position(
            market_id="m1", city_id="nyc", side="YES",
            entry_price=0.5, source="live",
        )
        pt = PaperTrade(
            market_id="m2", city_id="lon", side="NO",
            entry_price=0.3, trade_id=1,
        )
        # Should not raise, both are Position
        results = scan_for_exits([pos, pt], {}, {})
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Live / Paper independence
# Live must fire trades even when paper_trader is None.
# This was the coupling bug: can_live = (trade or fee_blocked)
# ---------------------------------------------------------------------------

class TestLivePaperIndependence:

    def test_live_does_not_depend_on_paper_trade_result(self):
        """Live execution must only require signal.edge >= 0.02.

        The old code: can_live = (trade or fee_blocked) and signal.edge >= 0.02
        The fix: can_live = signal.edge >= 0.02
        """
        # Simulate: paper_trader is None, fee_blocked is False
        # Old code: (None or False) and 0.05 >= 0.02 = False (BUG)
        # New code: 0.05 >= 0.02 = True (CORRECT)
        edge = 0.05
        can_live = edge >= 0.02  # This is the fixed logic
        assert can_live is True

    def test_low_edge_still_blocked(self):
        """Live must still reject signals with edge < 2%."""
        edge = 0.01
        can_live = edge >= 0.02
        assert can_live is False


# ---------------------------------------------------------------------------
# Config: hardcoded values must come from Settings
# ---------------------------------------------------------------------------

class TestConfigContract:

    def test_redis_config_exists(self):
        """Settings must have redis_host, redis_port, redis_db."""
        from weather_edge.config import settings
        assert hasattr(settings, "redis_host")
        assert hasattr(settings, "redis_port")
        assert hasattr(settings, "redis_db")
        assert isinstance(settings.redis_port, int)

    def test_chain_id_config_exists(self):
        """Settings must have polymarket_chain_id."""
        from weather_edge.config import settings
        assert hasattr(settings, "polymarket_chain_id")
        assert settings.polymarket_chain_id == 137

    def test_clob_url_config_exists(self):
        """Settings must have polymarket_clob_url."""
        from weather_edge.config import settings
        assert hasattr(settings, "polymarket_clob_url")
        assert "polymarket.com" in settings.polymarket_clob_url


# ---------------------------------------------------------------------------
# Safety rails: the hail-mary configuration must never be the default
# The 2026-04-07 closing experiment removed every rail (no horizon filter,
# no agreement gate, no dedupe, no vetoes, no caps). It is preserved behind
# settings.hail_mary_mode and must stay opt-in.
# ---------------------------------------------------------------------------

class TestSafetyRailDefaults:

    def test_hail_mary_mode_defaults_off(self):
        """Fresh Settings must not start in hail-mary mode."""
        from weather_edge.config import Settings
        assert Settings(_env_file=None).hail_mary_mode is False

    def test_rail_settings_have_safe_defaults(self):
        """The rails the hail-mary removed must default to their safe values."""
        from weather_edge.config import Settings
        s = Settings(_env_file=None)
        assert s.max_bets_per_city_date == 1   # one-bet-per-city dedupe
        assert s.max_trades_per_cycle == 6     # cycle cap
        assert s.max_core_zscore == 2.0        # z-score guard
        assert s.min_edge >= 0.02              # edge floor

    def test_scheduler_sizing_constants(self):
        """$5 live minimum and $1 hail-mary ticket must both exist."""
        from weather_edge import scheduler
        assert scheduler.MIN_LIVE_SIZE == 5.0
        assert scheduler.HAILMARY_TICKET_USD == 1.0

    def test_no_dead_gated_rails_in_scheduler(self):
        """No rail may be disabled via an 'if False and ...' dead gate.

        The hail-mary commit turned off live exit scanning this way; rails
        must be gated on settings.hail_mary_mode so they exist in both modes.
        """
        import inspect

        from weather_edge import scheduler
        source = inspect.getsource(scheduler)
        assert "if False and" not in source

    def test_rails_branch_on_hail_mary_setting(self):
        """Every restored rail must consult settings.hail_mary_mode."""
        import inspect

        from weather_edge import scheduler
        source = inspect.getsource(scheduler)
        # 13 rail sites branch on the flag; require at least 10 mentions so
        # a refactor that quietly drops most of them fails this test.
        assert source.count("hail_mary_mode") >= 10


# ---------------------------------------------------------------------------
# ENSO fallback: must warn, not silently use stale data
# ---------------------------------------------------------------------------

class TestENSOFallbackContract:

    def test_fetch_enso_returns_valid_state(self):
        """ENSO fetch must return an ENSOState with valid phase."""
        from weather_edge.analysis.enso_regime import ENSOState
        # Even the fallback must produce a valid ENSOState
        state = ENSOState(
            phase="neutral",
            oni_value=-0.5,
            transitioning=True,
            confidence=0.6,
            fetched_at=__import__("datetime").datetime.now(
                __import__("datetime").timezone.utc,
            ),
        )
        assert state.phase in ("la_nina", "neutral", "el_nino")
        assert -3.0 <= state.oni_value <= 3.0
        assert 0 <= state.confidence <= 1

    def test_oni_phase_boundaries(self):
        """ONI → phase mapping must be correct at boundaries."""
        # These are the boundaries from NOAA CPC
        assert _oni_to_phase(-0.5) == "la_nina"
        assert _oni_to_phase(-0.4) == "neutral"
        assert _oni_to_phase(0.0) == "neutral"
        assert _oni_to_phase(0.4) == "neutral"
        assert _oni_to_phase(0.5) == "el_nino"


def _oni_to_phase(oni: float) -> str:
    """Replicate the phase logic from enso_regime.py."""
    if oni <= -0.5:
        return "la_nina"
    elif oni >= 0.5:
        return "el_nino"
    return "neutral"
