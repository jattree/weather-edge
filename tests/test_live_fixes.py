"""Regression tests for the live-execution fixes.

No network: the CLOB client, Redis, HTTP and persistence are all faked.
py-clob-client / requests / eth libs are not installed in the dev venv, so
minimal stub modules are registered when the real ones are absent.
"""
from __future__ import annotations

import asyncio
import sqlite3
import sys
import types
from datetime import UTC
from unittest.mock import MagicMock

import pytest


def _ensure_module(name: str, **attrs):
    if name in sys.modules:
        return sys.modules[name]
    try:
        __import__(name)
        return sys.modules[name]
    except ImportError:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod
        return mod


class _ReqExcError(Exception):
    pass


_ensure_module(
    "requests",
    RequestException=_ReqExcError,
    ConnectionError=type("ConnectionError", (_ReqExcError,), {}),
    Timeout=type("Timeout", (_ReqExcError,), {}),
)


class _OrderArgs:
    def __init__(self, token_id, price, size, side):
        self.token_id, self.price, self.size, self.side = token_id, price, size, side


if "py_clob_client" not in sys.modules:
    try:
        import py_clob_client  # noqa: F401
    except ImportError:
        _ensure_module("py_clob_client")
        _ensure_module(
            "py_clob_client.clob_types",
            OrderArgs=_OrderArgs,
            OrderType=types.SimpleNamespace(GTC="GTC"),
        )
        _ensure_module("py_clob_client.order_builder")
        _ensure_module("py_clob_client.order_builder.constants", BUY="BUY", SELL="SELL")

from weather_edge import live_state  # noqa: E402
from weather_edge.analysis import risk_controls  # noqa: E402
from weather_edge.analysis.edge import Signal  # noqa: E402
from weather_edge.models.enums import SignalTier, TradeSide  # noqa: E402
from weather_edge.models.position import PRICE_BASIS_TOKEN, Position, normalize_side  # noqa: E402
from weather_edge.trading import executor as ex  # noqa: E402
from weather_edge.trading import kill_switch  # noqa: E402


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_state(monkeypatch):
    """Fresh in-memory live state, no Redis, paper mode, fresh breakers."""
    monkeypatch.setattr(live_state, "_redis_client", None)
    monkeypatch.setattr(live_state, "_redis_ever_connected", False)
    monkeypatch.setattr(live_state, "_fallback_cache", {})
    monkeypatch.setattr(live_state, "_get_redis", lambda: None)
    from weather_edge.config import settings
    monkeypatch.setattr(settings, "live_mode", False)
    monkeypatch.setattr(risk_controls, "_live_circuit_breaker", risk_controls.CircuitBreakerState())
    monkeypatch.setattr(risk_controls, "_circuit_breaker", risk_controls.CircuitBreakerState())
    monkeypatch.setattr(risk_controls, "_active_profile_name", "balanced")
    yield


class FakeStore:
    saved: list = []
    fail_first = 0

    def __init__(self, *a, **k):
        pass

    def save_live_trade(self, **kw):
        if FakeStore.fail_first > 0:
            FakeStore.fail_first -= 1
            raise sqlite3.OperationalError("database is locked")
        FakeStore.saved.append(kw)

    def close(self):
        pass


@pytest.fixture
def fake_store(monkeypatch):
    import weather_edge.persistence as persistence
    FakeStore.saved = []
    FakeStore.fail_first = 0
    monkeypatch.setattr(persistence, "PersistentStore", FakeStore)
    return FakeStore


@pytest.fixture
def tracked(monkeypatch):
    ids: list = []
    monkeypatch.setattr(ex, "track_open_order", lambda oid: ids.append(oid))
    return ids


def make_client(post_response=None, mid=None):
    client = MagicMock()
    client.create_order.side_effect = lambda args: args
    client.post_order.return_value = (
        post_response if post_response is not None
        else {"success": True, "errorMsg": "", "orderID": "0xabc", "status": "live"}
    )
    if mid is None:
        del client.get_midpoint
    else:
        client.get_midpoint.return_value = {"mid": str(mid)}
    return client


def live_executor(client) -> ex.TradeExecutor:
    e = ex.TradeExecutor(dry_run=False)
    e._client = client
    return e


def make_signal(side="YES", market_prob=0.40, size=10.0, **kw) -> Signal:
    from datetime import datetime
    base = dict(
        market_id="m1", consensus_id=None, computed_at=datetime.now(UTC),
        model_prob=0.6, model_confidence=0.8, market_prob=market_prob,
        edge=0.1, net_edge=0.1, edge_pct=0.2, kelly_fraction=0.1, half_kelly=0.05,
        recommended_side=TradeSide(side), recommended_size=size,
        confidence_tier=SignalTier.HIGH, city_id="nyc", description="test",
    )
    base.update(kw)
    return Signal(**base)


# ---------------------------------------------------------------------------
# 1. Kill switch fails closed
# ---------------------------------------------------------------------------

class TestKillSwitchFailClosed:
    def test_redis_error_blocks_orders(self, monkeypatch):
        r = MagicMock()
        r.get.side_effect = ConnectionError("redis down")
        monkeypatch.setattr(live_state, "_get_redis", lambda: r)
        assert kill_switch.is_kill_switch_active() is True

    def test_redis_lost_after_connect_blocks(self, monkeypatch):
        monkeypatch.setattr(live_state, "_redis_ever_connected", True)
        assert kill_switch.is_kill_switch_active() is True

    def test_redis_clear_flag_allows(self, monkeypatch):
        r = MagicMock()
        r.get.return_value = "0"
        monkeypatch.setattr(live_state, "_get_redis", lambda: r)
        assert kill_switch.is_kill_switch_active() is False

    def test_no_redis_paper_mode_uses_memory(self):
        assert kill_switch.is_kill_switch_active() is False
        kill_switch.activate_kill_switch("test")
        assert kill_switch.is_kill_switch_active() is True
        kill_switch.deactivate_kill_switch()
        assert kill_switch.is_kill_switch_active() is False

    def test_no_redis_live_mode_blocks(self, monkeypatch):
        from weather_edge.config import settings
        monkeypatch.setattr(settings, "live_mode", True)
        assert kill_switch.is_kill_switch_active() is True

    def test_flag_written_during_outage_still_blocks(self, monkeypatch):
        live_state._fallback_cache[kill_switch.KILL_SWITCH_KEY] = "1"
        r = MagicMock()
        r.get.return_value = None
        monkeypatch.setattr(live_state, "_get_redis", lambda: r)
        assert kill_switch.is_kill_switch_active() is True


# ---------------------------------------------------------------------------
# 2. Outcome casing and NO-position math
# ---------------------------------------------------------------------------

class TestSideNormalization:
    def test_normalize(self):
        assert normalize_side("Yes") == "YES"
        assert normalize_side(" no ") == "NO"
        assert normalize_side(TradeSide.NO) == "NO"
        assert normalize_side(None) == ""
        assert Position(side="No").side == "NO"

    def test_yes_exposure_counts_exchange_labels(self):
        profile = risk_controls.RISK_PROFILES["balanced"]
        open_pos = [Position(side="Yes", size_usd=400.0, status="open")]
        allowed, size, reason = risk_controls.check_yes_exposure_limit(
            50.0, open_pos, 1000.0, profile,
        )
        # 35% of 1000 = 350 cap, 400 already deployed in "Yes"
        assert allowed is False and size == 0.0

    def test_yes_exposure_counts_raw_outcome_attr(self):
        profile = risk_controls.RISK_PROFILES["balanced"]
        t = types.SimpleNamespace(side="", outcome="Yes", size_usd=400.0, status="open")
        allowed, _, _ = risk_controls.check_yes_exposure_limit(50.0, [t], 1000.0, profile)
        assert allowed is False

    def test_observation_guard_skips_no_position_regardless_of_case(self):
        from weather_edge.analysis.exit_monitor import _observation_context
        pos = Position(side="No", city_id="nyc", description="x 40-41°F on May 1?")
        assert _observation_context(pos) is None


class TestExitMonitorNoMath:
    def _live_no(self, token_entry=0.30):
        return Position(
            market_id="m1", city_id="nyc", side="No", entry_price=token_entry,
            price_basis=PRICE_BASIS_TOKEN, total_shares=100, size_usd=30,
            description="", source="live",
        )

    def test_live_no_uses_token_entry_basis(self):
        pos = self._live_no(0.30)
        assert pos.yes_entry_price == pytest.approx(0.70)
        assert pos.token_entry_price == pytest.approx(0.30)

    def test_live_no_original_edge_and_token_price(self):
        from weather_edge.analysis.exit_monitor import scan_for_exits
        pos = self._live_no(0.30)
        # Model now says YES 0.9 (NO 0.1), market YES 0.5 -> edge inverted
        cands = scan_for_exits(
            [pos], {"m1": 0.5}, {"m1": 0.9},
            no_prices={"m1": 0.48}, observation_holds=set(),
        )
        assert len(cands) == 1
        c = cands[0]
        assert c.reason == "edge_inversion"
        # NO prob 0.1 vs NO cost 0.30 -> original edge -0.20 (old code: -0.60)
        assert c.original_edge == pytest.approx((1 - 0.9) - 0.30)
        assert c.current_market_price == 0.5  # still the YES price
        assert c.token_price == pytest.approx(0.48)  # NO token's own price

    def test_no_token_price_falls_back_to_one_minus_yes(self):
        from weather_edge.analysis.exit_monitor import scan_for_exits
        cands = scan_for_exits(
            [self._live_no(0.30)], {"m1": 0.5}, {"m1": 0.9}, observation_holds=set(),
        )
        assert cands[0].token_price == pytest.approx(0.5)

    def test_expensive_no_is_not_treated_as_penny(self):
        from weather_edge.analysis.exit_monitor import scan_for_exits

        # Paper NO bought at NO=0.97 (entry_yes 0.03): not a penny ticket
        from weather_edge.trading.paper import PaperTrade
        pt = PaperTrade(market_id="m1", city_id="nyc", side="NO", entry_price=0.03,
                        size_usd=97, total_shares=100)
        cands = scan_for_exits([pt], {"m1": 0.5}, {"m1": 0.9}, observation_holds=set())
        assert len(cands) == 1


# ---------------------------------------------------------------------------
# 3. Sells priced per token with a meaningful slippage guard
# ---------------------------------------------------------------------------

class TestSellSlippageGuard:
    @pytest.fixture(autouse=True)
    def _no_kill(self, monkeypatch):
        monkeypatch.setattr(ex, "is_kill_switch_active", lambda: False)

    def test_blocks_sell_far_below_live_mid(self, fake_store, tracked):
        client = make_client(mid=0.80)
        e = live_executor(client)
        # Selling a NO token worth 0.80 at the YES price 0.20
        res = run(e.place_sell_order("tokNO", 10, 0.20, "m1", reference_price=0.20))
        assert res is None
        client.post_order.assert_not_called()

    def test_allows_sell_at_mid(self, fake_store, tracked):
        client = make_client(mid=0.80)
        e = live_executor(client)
        res = run(e.place_sell_order("tokNO", 10, 0.80, "m1"))
        assert res is not None and res.status == "pending"
        assert tracked == ["0xabc"]
        assert fake_store.saved[0]["side"] == "SELL"

    def test_live_sell_without_any_reference_is_refused(self, fake_store, tracked):
        client = make_client(mid=None)
        e = live_executor(client)
        assert run(e.place_sell_order("tok", 10, 0.50, "m1")) is None
        client.post_order.assert_not_called()

    def test_dry_run_guard_uses_caller_reference(self):
        e = ex.TradeExecutor(dry_run=True)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(ex, "is_kill_switch_active", lambda: False)
            assert run(e.place_sell_order("t", 10, 0.20, "m1", reference_price=0.80)) is None
            ok = run(e.place_sell_order("t", 10, 0.80, "m1", reference_price=0.80))
        assert ok.status == "dry_run"


# ---------------------------------------------------------------------------
# 4. Refresh while stopped: live execution suppressed
# ---------------------------------------------------------------------------

class TestSuppressLiveOrders:
    def test_suppressed_context_blocks_everything(self, monkeypatch, fake_store, tracked):
        monkeypatch.setattr(ex, "is_kill_switch_active", lambda: False)
        client = make_client(mid=0.5)
        e = live_executor(client)

        async def cycle():
            # Work spawned inside the cycle inherits the suppression
            sub = asyncio.create_task(e.place_limit_order(make_signal(), "tok"))
            return (
                await sub,
                await e.place_sell_order("tok", 10, 0.5, "m1"),
                await e.cancel_order("o1"),
                await e.cancel_stale_orders(),
                await e.redeem_positions(),
            )

        async def main():
            with ex.suppress_live_orders():
                return await cycle()

        buy, sell, cancel, stale, redeem = run(main())
        assert (buy, sell, cancel, stale, redeem) == (None, None, False, 0, 0)
        client.post_order.assert_not_called()
        client.cancel.assert_not_called()
        assert not ex.live_orders_suppressed()

    def test_not_suppressed_outside_context(self):
        assert ex.live_orders_suppressed() is False


# ---------------------------------------------------------------------------
# 7. Live circuit breaker trips the kill switch; default profile
# ---------------------------------------------------------------------------

class TestLiveCircuitBreaker:
    def test_default_profile_is_not_aggressive(self):
        assert risk_controls.DEFAULT_PROFILE_NAME == "balanced"
        p = risk_controls.RISK_PROFILES[risk_controls.DEFAULT_PROFILE_NAME]
        assert p.drawdown_kill_pct <= 0.25 and p.max_gross_exposure_multiple <= 2.0

    def test_drawdown_kill_activates_kill_switch(self):
        assert risk_controls.update_live_circuit_breaker(1000.0) == 1.0
        assert kill_switch.is_kill_switch_active() is False
        mult = risk_controls.update_live_circuit_breaker(700.0)  # 30% > 25%
        assert mult == 0.0
        assert kill_switch.is_kill_switch_active() is True
        meta = kill_switch.get_kill_switch_state()
        assert meta["triggered_by"] == "circuit_breaker"

    @staticmethod
    def _redis(monkeypatch, data: dict):
        """Dict-backed Redis; set data['down'] = True to make it unreachable."""
        class R:
            def get(self, k):
                return data.get(k)

            def set(self, k, v, *a, **kw):
                data[k] = v

            def setex(self, k, ttl, v):
                data[k] = v
        client = R()
        monkeypatch.setattr(live_state, "_get_redis",
                            lambda: None if data.get("down") else client)
        return data

    @staticmethod
    def _restart():
        risk_controls._live_circuit_breaker = risk_controls.CircuitBreakerState()

    def test_hwm_survives_restart(self, monkeypatch):
        self._redis(monkeypatch, {})
        risk_controls.update_live_circuit_breaker(1000.0)
        self._restart()
        assert risk_controls.update_live_circuit_breaker(700.0) == 0.0

    def test_restart_during_outage_keeps_persisted_peak(self, monkeypatch):
        # Persisted peak $100; restart while Redis is down; NAV $60 observed.
        data = self._redis(monkeypatch, {risk_controls._LIVE_HWM_KEY: "100.0"})
        self._restart()
        data["down"] = True
        risk_controls.update_live_circuit_breaker(60.0)
        assert data[risk_controls._LIVE_HWM_KEY] == "100.0"   # not overwritten
        data["down"] = False
        # Recovery merges the real peak: 40% drawdown trips the breaker.
        assert risk_controls.update_live_circuit_breaker(60.0) == 0.0
        assert kill_switch.is_kill_switch_active() is True
        assert data[risk_controls._LIVE_HWM_KEY] == "100.0"

    def test_operator_reset_rearms_breaker(self):
        risk_controls.update_live_circuit_breaker(1000.0)
        risk_controls.update_live_circuit_breaker(700.0)
        kill_switch.deactivate_kill_switch("test")
        assert kill_switch.is_kill_switch_active() is False
        assert risk_controls.live_circuit_breaker_multiplier() == 1.0
        # 700 is the new high-water mark, not an immediate re-trip
        assert risk_controls.update_live_circuit_breaker(700.0) == 1.0

    def test_zero_nav_is_ignored(self):
        risk_controls.update_live_circuit_breaker(1000.0)
        assert risk_controls.update_live_circuit_breaker(0.0) == 1.0
        assert kill_switch.is_kill_switch_active() is False

    def test_paper_nav_does_not_touch_live_breaker(self):
        risk_controls._circuit_breaker.update(5000.0, risk_controls.get_active_profile())
        assert risk_controls._live_circuit_breaker.high_water_mark == 0.0

    def test_executor_blocks_when_live_breaker_killed(self, monkeypatch):
        monkeypatch.setattr(ex, "is_kill_switch_active", lambda: False)
        risk_controls._live_circuit_breaker.is_killed = True
        e = ex.TradeExecutor(dry_run=True)
        assert run(e.place_limit_order(make_signal(), "tok")) is None


# ---------------------------------------------------------------------------
# 8. Rejected orders are not tracked or persisted
# ---------------------------------------------------------------------------

class TestRejectedOrders:
    @pytest.fixture(autouse=True)
    def _no_kill(self, monkeypatch):
        monkeypatch.setattr(ex, "is_kill_switch_active", lambda: False)

    @pytest.mark.parametrize("resp,status", [
        ({"success": False, "errorMsg": "invalid post-only order: order crosses book",
          "orderID": "", "status": ""}, "post_only_reject"),
        ({"status": "POST_ONLY_VIOLATION", "orderID": "0x1"}, "post_only_reject"),
        ({"success": False, "errorMsg": "not enough balance / allowance"}, "rejected"),
        ({"status": "REJECTED", "orderID": "0x1"}, "rejected"),
        ({"success": True, "errorMsg": ""}, "rejected"),  # no order id
    ])
    def test_rejected_sell_not_recorded(self, fake_store, tracked, resp, status):
        e = live_executor(make_client(post_response=resp, mid=0.5))
        res = run(e.place_sell_order("tok", 10, 0.5, "m1"))
        assert res.status == status
        assert fake_store.saved == []
        assert tracked == []

    def test_rejected_buy_not_recorded(self, fake_store, tracked):
        e = live_executor(make_client(
            post_response={"success": False, "errorMsg": "post only violation"}))
        res = run(e.place_limit_order(make_signal(), "tok"))
        assert res.status == "post_only_reject"
        assert fake_store.saved == [] and tracked == []

    def test_classify_accepts_normal_response(self):
        assert ex.classify_post_response(
            {"success": True, "errorMsg": "", "orderID": "0x1", "status": "live"}
        ) == (False, "pending", "")


# ---------------------------------------------------------------------------
# 10. Live NO hedge priced correctly
# ---------------------------------------------------------------------------

class TestHedgeSignal:
    def test_no_hedge_market_prob_is_yes_equivalent(self, monkeypatch):
        from weather_edge.trading.market_maker import SpreadOrder, build_hedge_signal
        monkeypatch.setattr(ex, "is_kill_switch_active", lambda: False)
        hedge = SpreadOrder(
            market_id="m1", city_id="nyc", token_id="", side="NO", limit_price=0.40,
            shares=25, cost=10.0, guaranteed_profit=0.5, paired_with="m1",
            description="HEDGE NO",
        )
        sig = build_hedge_signal(make_signal("YES", 0.58), hedge)
        assert sig.recommended_side == TradeSide.NO
        assert sig.market_prob == pytest.approx(0.60)
        assert sig.strategy == "spread"
        e = ex.TradeExecutor(dry_run=True)
        res = run(e.place_limit_order(sig, "tokNO", improve_price_by=0.0))
        assert res.limit_price == pytest.approx(0.40)  # the NO token's own price


# ---------------------------------------------------------------------------
# 6. Spread estimate
# ---------------------------------------------------------------------------

class TestSpreadEstimate:
    def test_complementary_prices_use_conservative_default(self):
        from weather_edge.fetchers.polymarket import MarketInfo
        from weather_edge.trading.market_maker import (
            DEFAULT_SPREAD_ESTIMATE,
            estimate_market_spread,
        )
        m = MarketInfo(market_id="m", yes_price=0.42, no_price=0.58)
        assert estimate_market_spread(m) == DEFAULT_SPREAD_ESTIMATE > 0

    def test_real_top_of_book_used_when_present(self):
        from weather_edge.trading.market_maker import estimate_market_spread
        m = types.SimpleNamespace(best_bid=0.40, best_ask=0.45, yes_price=0.42, no_price=0.58)
        assert estimate_market_spread(m) == pytest.approx(0.05)
        m2 = types.SimpleNamespace(spread=0.07, yes_price=0.42, no_price=0.58)
        assert estimate_market_spread(m2) == pytest.approx(0.07)


# ---------------------------------------------------------------------------
# 11. fill tracker fees and resolution sides
# ---------------------------------------------------------------------------

class TestFillTracker:
    def test_fee_maker_vs_taker(self):
        from weather_edge.trading.fees import calculate_taker_fee
        from weather_edge.trading.fill_tracker import fill_fee_usd
        assert fill_fee_usd({"is_maker": 1}, 0.5, 100) == 0.0
        assert fill_fee_usd({"is_maker": 0}, 0.5, 100) == pytest.approx(
            calculate_taker_fee(0.5, 50.0))

    def test_side_mapping(self):
        from weather_edge.trading.fill_tracker import live_trade_won
        assert live_trade_won("YES", True) is True
        assert live_trade_won("YES", False) is False
        assert live_trade_won("NO", False) is True
        assert live_trade_won("SELL", True) is None

    def test_resolve_live_trades(self, monkeypatch):
        import weather_edge.analysis.resolver as resolver
        import weather_edge.persistence as persistence
        from weather_edge.trading import fill_tracker

        rows = [
            {"order_id": "a", "market_id": "m1", "side": "YES", "filled_shares": 10,
             "cost_basis": 4.05, "fee_usd": 0.05, "status": "filled"},
            {"order_id": "b", "market_id": "m1", "side": "NO", "filled_shares": 10,
             "cost_basis": 6.0, "fee_usd": 0.0, "status": "filled"},
            {"order_id": "c", "market_id": "m1", "side": "SELL", "filled_shares": 10,
             "cost_basis": 0.0, "fee_usd": 0.0, "status": "filled"},
        ]
        resolved = {}

        class Store:
            def __init__(self, *a, **k):
                self.conn = MagicMock()
                self.conn.execute.return_value.fetchall.return_value = rows

            def resolve_live_trade(self, order_id, proceeds, pnl, status):
                resolved[order_id] = (proceeds, pnl, status)

            def close(self):
                pass

        async def fake_resolved():
            return {"m1": True}

        monkeypatch.setattr(persistence, "PersistentStore", Store)
        monkeypatch.setattr(resolver, "fetch_resolved_markets", fake_resolved)
        n = run(fill_tracker.resolve_live_trades(types.SimpleNamespace(dry_run=False)))
        assert n == 2
        assert resolved["a"] == (10.0, pytest.approx(5.95), "won")  # fee only in basis
        assert resolved["b"] == (0.0, -6.0, "lost")
        assert "c" not in resolved


# ---------------------------------------------------------------------------
# 14. Redemption counts only confirmed
# ---------------------------------------------------------------------------

class TestRedemptionCount:
    def test_only_confirmed_counted(self, monkeypatch):
        import httpx

        acct = MagicMock()
        acct.from_key.return_value.address = "0xEOA"
        acct.sign_message.return_value.signature.hex.return_value = "0xsig"
        web3 = MagicMock()
        web3.return_value.eth.contract.return_value.encode_abi.return_value = "0xdata"
        web3.return_value.eth.contract.return_value.functions.payoutDenominator.return_value.call.return_value = 1  # noqa: E501
        stubs = {
            "eth_abi": types.SimpleNamespace(encode=MagicMock()),
            "eth_abi.packed": types.SimpleNamespace(encode_packed=MagicMock(return_value=b"")),
            "eth_account": types.SimpleNamespace(Account=acct),
            "eth_account.messages": types.SimpleNamespace(encode_defunct=MagicMock()),
            "eth_utils": types.SimpleNamespace(
                keccak=MagicMock(return_value=b"\x00" * 32),
                to_bytes=MagicMock(return_value=b""),
                to_checksum_address=lambda x: "0xPROXY",
            ),
            "hexbytes": types.SimpleNamespace(HexBytes=bytes),
            "web3": types.SimpleNamespace(Web3=web3),
        }
        for k, v in stubs.items():
            monkeypatch.setitem(sys.modules, k, v)

        positions = [
            {"redeemable": True, "size": 10, "conditionId": "0x" + "11" * 32, "outcomeIndex": 0},
            {"redeemable": True, "size": 10, "conditionId": "0x" + "22" * 32, "outcomeIndex": 1},
        ]

        class Resp:
            def __init__(self, code, data):
                self.status_code, self._d, self.text = code, data, ""

            def json(self):
                return self._d

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url, **kw):
                if "positions" in url:
                    return Resp(200, positions)
                return Resp(200, {"nonce": 1, "address": "0xRELAY"})

            async def post(self, url, **kw):
                return Resp(200, {"transactionID": "tx"})

        monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: Client())
        outcomes = iter([True, False])  # first confirmed, second failed/pending

        async def poll(self, url, tx, timeout=60):
            return next(outcomes)

        monkeypatch.setattr(ex.TradeExecutor, "_poll_relayer_tx", poll)
        from weather_edge.config import settings
        monkeypatch.setattr(settings, "polymarket_relayer_url", "https://relayer")
        e = ex.TradeExecutor(private_key="0x" + "1" * 64, dry_run=False, relayer_api_key="k")
        assert run(e.redeem_positions()) == 1


# ---------------------------------------------------------------------------
# 15. Price chase moves at least one tick
# ---------------------------------------------------------------------------

class TestPriceChase:
    def test_one_tick_improvement(self):
        assert ex.chase_limit_price(0.45, 0.50) == pytest.approx(0.46)

    def test_capped_at_mid_skips(self):
        assert ex.chase_limit_price(0.45, 0.455) is None
        assert ex.chase_limit_price(0.50, 0.50) is None

    def test_executor_places_at_chased_price(self, monkeypatch):
        monkeypatch.setattr(ex, "is_kill_switch_active", lambda: False)
        e = ex.TradeExecutor(dry_run=True)
        chase = ex.chase_limit_price(0.45, 0.50)
        # scheduler override: YES market_prob = chase + 0.005
        yes = run(e.place_limit_order(make_signal("YES", chase + 0.005), "t"))
        no = run(e.place_limit_order(make_signal("NO", 1.0 - (chase + 0.005)), "t"))
        assert yes.limit_price == pytest.approx(chase) != 0.45
        assert no.limit_price == pytest.approx(chase)


# ---------------------------------------------------------------------------
# 16. No blocking I/O on the async paths
# ---------------------------------------------------------------------------

class TestNonBlocking:
    def test_persist_uses_async_retry_not_time_sleep(self, monkeypatch, fake_store, tracked):
        import time

        monkeypatch.setattr(ex, "is_kill_switch_active", lambda: False)
        monkeypatch.setattr(time, "sleep", MagicMock(side_effect=AssertionError("blocking")))
        fake_store.fail_first = 1
        e = live_executor(make_client(mid=0.4))
        res = run(e.place_limit_order(make_signal(), "tok"))
        assert res.status == "pending"
        assert len(fake_store.saved) == 1

    def test_async_scan_never_calls_sync_observation(self, monkeypatch):
        from weather_edge.analysis import exit_monitor

        monkeypatch.setattr(
            exit_monitor, "_check_observation_confirms_bucket",
            MagicMock(side_effect=AssertionError("sync http in async path")),
        )
        pos = Position(market_id="m1", city_id="nyc", side="YES", entry_price=0.5,
                       total_shares=10, size_usd=5)
        cands = run(exit_monitor.scan_for_exits_async([pos], {"m1": 0.5}, {"m1": 0.1}))
        assert len(cands) == 1


# ---------------------------------------------------------------------------
# 12. Scripts default to dry run
# ---------------------------------------------------------------------------

class TestScriptSafety:
    def test_dry_run_matrix(self):
        assert ex.script_dry_run(False, live_mode=True) is True
        assert ex.script_dry_run(True, live_mode=False) is True
        assert ex.script_dry_run(True, live_mode=True) is False

    def _load(self, name):
        import importlib.util
        from pathlib import Path
        path = Path(__file__).parent.parent / "scripts" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_sell_script_requires_token_and_limit(self):
        mod = self._load("sell_half_chi")
        with pytest.raises(SystemExit):
            mod.parse_args([])
        with pytest.raises(SystemExit):
            mod.parse_args(["--token-id", "t"])
        args = mod.parse_args(["--token-id", "t", "--limit-price", "0.6"])
        assert args.execute is False and args.fraction == 0.5

    def test_dump_dust_prices_at_mark(self):
        mod = self._load("dump_dust")
        assert mod.parse_args([]).execute is False
        assert mod.dust_sell_price(0.034) == 0.03
        assert mod.dust_sell_price(0) is None

    def test_redeem_defaults_dry(self):
        mod = self._load("run_redeem")
        assert mod.parse_args([]).execute is False
