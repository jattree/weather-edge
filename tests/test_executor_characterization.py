"""Characterization tests for TradeExecutor.place_limit_order / place_sell_order.

Pin the current observable behaviour (return value, log lines, exchange calls,
tracking and persistence side effects) of every branch so the two methods can
be refactored safely. No network: the CLOB client and persistence are faked.
"""
from __future__ import annotations

import dataclasses
import logging

import pytest

from tests.test_live_fixes import (
    FakeStore,
    ex,
    live_executor,
    make_client,
    make_signal,
    run,
)
from weather_edge import live_state
from weather_edge.analysis import risk_controls


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


@pytest.fixture
def fake_store(monkeypatch):
    from weather_edge import persistence
    FakeStore.saved = []
    FakeStore.fail_first = 0
    monkeypatch.setattr(persistence, "PersistentStore", FakeStore)
    return FakeStore


@pytest.fixture
def tracked(monkeypatch):
    ids: list = []
    monkeypatch.setattr(ex, "track_open_order", ids.append)
    return ids


LOGGER = "weather_edge.trading.executor"


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)

    def _get():
        return [(r.levelname, r.getMessage()) for r in caplog.records if r.name == LOGGER]

    return _get


@pytest.fixture
def no_kill(monkeypatch):
    monkeypatch.setattr(ex, "is_kill_switch_active", lambda: False)


def snap(result):
    return None if result is None else dataclasses.asdict(result)


class _BrokenStore(FakeStore):
    def save_live_trade(self, **kw):
        raise ValueError("disk gone")


@pytest.fixture
def broken_store(monkeypatch):
    from weather_edge import persistence
    monkeypatch.setattr(persistence, "PersistentStore", _BrokenStore)


# ---------------------------------------------------------------------------
# place_limit_order
# ---------------------------------------------------------------------------

class TestBuyGuards:
    def test_kill_switch_blocks(self, monkeypatch, logs):
        monkeypatch.setattr(ex, "is_kill_switch_active", lambda: True)
        e = live_executor(make_client())
        assert run(e.place_limit_order(make_signal(), "tok")) is None
        assert logs() == [("WARNING", "KILL SWITCH ACTIVE, blocking order for nyc test")]
        e._client.post_order.assert_not_called()

    def test_circuit_breaker_error_fails_closed(self, monkeypatch, no_kill, logs):
        def boom():
            raise RuntimeError("x")
        monkeypatch.setattr(risk_controls, "live_circuit_breaker_multiplier", boom)
        e = ex.TradeExecutor(dry_run=True)
        assert run(e.place_limit_order(make_signal(), "tok")) is None
        assert logs() == [("ERROR", "Circuit breaker check failed, blocking order (fail-closed)")]

    def test_client_not_initialized(self, no_kill, logs):
        e = ex.TradeExecutor(dry_run=False)
        assert run(e.place_limit_order(make_signal(), "tok")) is None
        assert logs()[-1] == ("ERROR", "Client not initialized, cannot place order")

    def test_second_kill_switch_check(self, monkeypatch, logs):
        calls = iter([False, True])
        monkeypatch.setattr(ex, "is_kill_switch_active", lambda: next(calls))
        e = live_executor(make_client())
        assert run(e.place_limit_order(make_signal(), "tok")) is None
        assert logs() == [("WARNING", "KILL SWITCH activated during order prep, aborting")]
        e._client.post_order.assert_not_called()


class TestBuyPricingAndSizing:
    def test_dry_run_yes_maker(self, no_kill, logs):
        e = ex.TradeExecutor(dry_run=True)
        res = run(e.place_limit_order(make_signal(market_prob=0.40, size=10.0), "tok"))
        assert snap(res) == {
            "order_id": "dry_run", "market_id": "m1", "side": "YES",
            "size_usd": 10.0, "size_shares": 25.0, "limit_price": 0.4,
            "status": "dry_run", "is_maker": True,
            "taker_fee_avoided": round(ex.calculate_taker_fee(0.40, 10.0), 4),
            "filled_price": None, "filled_at": None, "tx_hash": None,
            "reject_reason": "", "raw_response": {},
        }
        assert logs() == [("INFO", "DRY RUN: MAKER YES 25 shares @ 0.400 ($10.00) on nyc "
                                   "(post_only=True, taker_fee_avoided=$0.12)")]

    def test_limit_price_rounding_exact(self, no_kill):
        e = ex.TradeExecutor(dry_run=True)
        prices = {}
        for side in ("YES", "NO"):
            for taker in (False, True):
                r = run(e.place_limit_order(
                    make_signal(side=side, market_prob=0.40, size=10.0), "tok",
                    force_taker=taker,
                ))
                prices[(side, taker)] = (r.limit_price, r.size_shares, r.size_usd)
        assert prices == {
            ("YES", False): (0.4, 25.0, 10.0),
            ("YES", True): (0.43, 23.25, 10.0),
            ("NO", False): (0.59, 16.94, 9.99),
            ("NO", True): (0.63, 15.87, 10.0),
        }
        # shares are floored from the rounded price
        for (lp, sh, usd) in prices.values():
            assert sh == ex._floor_shares(10.0 / lp)
            assert usd == round(sh * lp, 2)

    def test_price_clamped(self, no_kill):
        e = ex.TradeExecutor(dry_run=True)
        r = run(e.place_limit_order(make_signal(market_prob=0.99, size=50.0), "tok",
                                    force_taker=True))
        assert r.limit_price == 0.99

    def test_graduated_cap(self, no_kill, logs):
        e = ex.TradeExecutor(dry_run=True, max_shares=7.5)
        r = run(e.place_limit_order(make_signal(market_prob=0.40, size=10.0), "tok",
                                    force_taker=True))
        assert (r.size_shares, r.size_usd) == (7.5, round(7.5 * 0.43, 2))
        assert ("INFO", "GRADUATED CAP: capped nyc to 8 shares (max=7.5)") in logs()

    def test_too_small(self, no_kill, logs):
        e = live_executor(make_client())
        r = run(e.place_limit_order(make_signal(market_prob=0.40, size=1.0), "tok",
                                    force_taker=True))
        assert snap(r) == {
            "order_id": "too_small", "market_id": "m1", "side": "YES",
            "size_usd": 1.0, "size_shares": 2.32, "limit_price": 0.43,
            "status": "rejected", "is_maker": True, "taker_fee_avoided": 0.0,
            "filled_price": None, "filled_at": None, "tx_hash": None,
            "reject_reason": "Size 2.32 < minimum 5.0", "raw_response": {},
        }
        assert logs() == [("INFO", "ORDER TOO SMALL: nyc 2.3 shares < minimum 5, skipping")]
        e._client.post_order.assert_not_called()


class TestBuyLive:
    def test_success_maker(self, no_kill, fake_store, tracked, logs):
        client = make_client()
        e = live_executor(client)
        sig = make_signal(market_prob=0.40, size=10.0)
        r = run(e.place_limit_order(sig, "tok", force_taker=True))
        fee = round(ex.calculate_taker_fee(0.40, 10.0), 4)
        assert snap(r) == {
            "order_id": "0xabc", "market_id": "m1", "side": "YES",
            "size_usd": 10.0, "size_shares": 23.25, "limit_price": 0.43,
            "status": "pending", "is_maker": True, "taker_fee_avoided": fee,
            "filled_price": None, "filled_at": None, "tx_hash": None,
            "reject_reason": "",
            "raw_response": {"success": True, "errorMsg": "", "orderID": "0xabc",
                             "status": "live"},
        }
        args = client.create_order.call_args.args[0]
        assert (args.token_id, args.price, args.size, args.side) == ("tok", 0.43, 23.25, "BUY")
        assert client.post_order.call_args.kwargs == {"orderType": "GTC", "post_only": False}
        assert tracked == ["0xabc"]
        assert fake_store.saved == [{
            "order_id": "0xabc", "market_id": "m1", "token_id": "tok", "city_id": "nyc",
            "side": "YES", "limit_price": 0.43, "size_shares": 23.25, "size_usd": 10.0,
            "description": "test", "strategy": "core", "is_maker": False,
        }]
        assert logs()[-1] == ("INFO",
            "LIVE ORDER PLACED: YES nyc 23 shares @ 0.430 ($10.00) | order_id=0xabc | "
            f"post_only=True | taker_fee_avoided=${ex.calculate_taker_fee(0.40, 10.0):.2f}")

    def test_success_post_only(self, no_kill, fake_store, tracked):
        client = make_client()
        e = live_executor(client)
        r = run(e.place_limit_order(make_signal(side="NO"), "tok"))
        assert r.status == "pending" and r.is_maker is True and r.side == "NO"
        assert client.post_order.call_args.kwargs == {"orderType": "GTC", "post_only": True}
        assert fake_store.saved[0]["is_maker"] is True

    def test_non_dict_response_rejected(self, no_kill, fake_store, tracked, logs):
        e = live_executor(make_client(post_response="garbage"))
        r = run(e.place_limit_order(make_signal(), "tok", force_taker=True))
        assert (r.order_id, r.status, r.is_maker, r.raw_response) == (
            "unknown", "rejected", False, {"raw": "garbage"})
        assert r.reject_reason == "unparseable response: 'garbage'"
        assert logs()[-1] == ("WARNING",
            "ORDER REJECTED: YES nyc 23 shares @ 0.430, unparseable response: 'garbage'")
        assert fake_store.saved == [] and tracked == []

    def test_ghost_trade(self, no_kill, broken_store, tracked, logs):
        e = live_executor(make_client())
        r = run(e.place_limit_order(make_signal(), "tok", force_taker=True))
        assert r.status == "pending" and r.order_id == "0xabc"
        assert tracked == ["0xabc"]
        crit = [m for lvl, m in logs() if lvl == "CRITICAL"]
        assert crit == ["GHOST TRADE: order 0xabc placed on exchange but DB write "
                        "failed after 3 retries, disk gone. Manual reconciliation needed."]

    @pytest.mark.usefixtures("no_kill")
    @pytest.mark.parametrize(("exc_name", "label"), [
        ("ConnectionError", "NETWORK FAILURE"),
        ("Timeout", "NETWORK FAILURE"),
        ("RequestException", "API ERROR"),
        ("KeyError", "PARSE ERROR"),
    ])
    def test_exchange_errors(self, fake_store, tracked, logs, exc_name, label):
        exc_type = KeyError if exc_name == "KeyError" else getattr(ex.requests, exc_name)
        client = make_client()
        client.post_order.side_effect = exc_type("boom")
        e = live_executor(client)
        assert run(e.place_limit_order(make_signal(), "tok")) is None
        lvl, msg = logs()[-1]
        assert lvl == "ERROR" and msg.startswith(f"LIVE ORDER {label}: nyc test, ")
        assert fake_store.saved == [] and tracked == []


# ---------------------------------------------------------------------------
# place_sell_order
# ---------------------------------------------------------------------------

class TestSell:
    def test_kill_switch_blocks(self, monkeypatch, logs):
        monkeypatch.setattr(ex, "is_kill_switch_active", lambda: True)
        e = live_executor(make_client(mid=0.5))
        assert run(e.place_sell_order("tok", 10, 0.5, "m1", city_id="chi")) is None
        assert logs() == [("WARNING", "KILL SWITCH ACTIVE, blocking sell for chi")]

    def test_too_small_after_guard(self, no_kill, logs):
        e = ex.TradeExecutor(dry_run=True)
        assert run(e.place_sell_order("tok", 4.999, 0.5, "m1", city_id="chi")) is None
        assert logs() == [("INFO", "SELL TOO SMALL: chi 5.0 shares < 5 min")]

    def test_slippage_checked_before_size(self, no_kill, logs):
        e = ex.TradeExecutor(dry_run=True)
        assert run(e.place_sell_order("tok", 1, 0.2, "m1", city_id="chi",
                                      reference_price=0.8)) is None
        assert logs()[-1][1].startswith("SELL BLOCKED (slippage vs caller): chi price=0.200")

    def test_caller_reference_checked_before_mid(self, no_kill, logs):
        e = live_executor(make_client(mid=0.30))
        assert run(e.place_sell_order("tok", 10, 0.30, "m1", city_id="chi",
                                      reference_price=0.9)) is None
        assert logs()[-1][1].startswith("SELL BLOCKED (slippage vs caller)")
        e2 = live_executor(make_client(mid=0.9))
        assert run(e2.place_sell_order("tok", 10, 0.30, "m1", city_id="chi",
                                       reference_price=0.3)) is None
        assert logs()[-1][1].startswith("SELL BLOCKED (slippage vs clob_mid)")

    def test_no_reference_message(self, no_kill, logs):
        e = live_executor(make_client(mid=None))
        assert run(e.place_sell_order("tok123456789012345678", 10, 0.5, "m1",
                                      city_id="chi")) is None
        assert logs() == [("WARNING", "SELL BLOCKED (no reference price): chi token "
                                      "tok1234567890123, cannot verify limit 0.500")]

    def test_dry_run(self, no_kill, logs):
        e = ex.TradeExecutor(dry_run=True)
        r = run(e.place_sell_order("tok", 10.129, 0.5049, "m1", city_id="chi"))
        assert snap(r) == {
            "order_id": "dry_run_sell", "market_id": "m1", "side": "SELL",
            "size_usd": round(10.12 * 0.5, 2), "size_shares": 10.12, "limit_price": 0.5,
            "status": "dry_run", "is_maker": True, "taker_fee_avoided": 0.0,
            "filled_price": None, "filled_at": None, "tx_hash": None,
            "reject_reason": "", "raw_response": {},
        }
        assert logs() == [("INFO", "DRY RUN SELL: chi 10 shares @ 0.500")]

    def test_dry_run_when_client_missing(self, no_kill):
        e = ex.TradeExecutor(dry_run=False)
        r = run(e.place_sell_order("tok", 10, 0.5, "m1"))
        assert r.order_id == "dry_run_sell" and r.status == "dry_run"

    def test_live_success(self, no_kill, fake_store, tracked, logs):
        client = make_client(mid=0.5)
        e = live_executor(client)
        r = run(e.place_sell_order("tok", 10.129, 0.5, "m1", city_id="chi",
                                   description="d" * 100))
        assert snap(r) == {
            "order_id": "0xabc", "market_id": "m1", "side": "SELL",
            "size_usd": 5.06, "size_shares": 10.12, "limit_price": 0.5,
            "status": "pending", "is_maker": True, "taker_fee_avoided": 0.0,
            "filled_price": None, "filled_at": None, "tx_hash": None,
            "reject_reason": "",
            "raw_response": {"success": True, "errorMsg": "", "orderID": "0xabc",
                             "status": "live"},
        }
        args = client.create_order.call_args.args[0]
        assert (args.token_id, args.price, args.size, args.side) == ("tok", 0.5, 10.12, "SELL")
        assert client.post_order.call_args.kwargs == {"orderType": "GTC", "post_only": True}
        assert tracked == ["0xabc"]
        assert fake_store.saved == [{
            "order_id": "0xabc", "market_id": "m1", "token_id": "tok", "city_id": "chi",
            "side": "SELL", "limit_price": 0.5, "size_shares": 10.12, "size_usd": 5.06,
            "description": "d" * 80, "strategy": "exit", "is_maker": True,
        }]
        assert logs() == [("INFO",
            "LIVE SELL PLACED: chi 10 shares @ 0.500 ($5.06) | order_id=0xabc")]

    def test_live_force_taker(self, no_kill, fake_store, tracked, logs):
        client = make_client(mid=0.5)
        e = live_executor(client)
        r = run(e.place_sell_order("tok", 10, 0.5, "m1", city_id="chi", force_taker=True))
        assert r.is_maker is False and r.status == "pending"
        assert client.post_order.call_args.kwargs == {"orderType": "GTC", "post_only": False}
        assert fake_store.saved[0]["is_maker"] is False
        assert [m for _, m in logs()] == [
            "EXIT TAKER MODE: chi, post_only=False, will pay taker fees",
            "LIVE SELL PLACED: chi 10 shares @ 0.500 ($5.00) | order_id=0xabc",
        ]

    def test_force_taker_log_precedes_rejection(self, no_kill, fake_store, tracked, logs):
        e = live_executor(make_client(post_response="bad", mid=0.5))
        r = run(e.place_sell_order("tok", 10, 0.5, "m1", city_id="chi", force_taker=True))
        assert snap(r) == {
            "order_id": "unknown", "market_id": "m1", "side": "SELL",
            "size_usd": 5.0, "size_shares": 10.0, "limit_price": 0.5,
            "status": "rejected", "is_maker": False, "taker_fee_avoided": 0.0,
            "filled_price": None, "filled_at": None, "tx_hash": None,
            "reject_reason": "unparseable response: 'bad'", "raw_response": {"raw": "bad"},
        }
        assert [m for _, m in logs()] == [
            "EXIT TAKER MODE: chi, post_only=False, will pay taker fees",
            "SELL REJECTED: chi 10 shares @ 0.500, unparseable response: 'bad'",
        ]
        assert fake_store.saved == [] and tracked == []

    def test_ghost_sell(self, no_kill, broken_store, tracked, logs):
        e = live_executor(make_client(mid=0.5))
        r = run(e.place_sell_order("tok", 10, 0.5, "m1", city_id="chi"))
        assert r.status == "pending" and tracked == ["0xabc"]
        crit = [m for lvl, m in logs() if lvl == "CRITICAL"]
        assert crit == ["GHOST SELL: sell order 0xabc placed on exchange but DB write "
                        "failed, disk gone"]

    @pytest.mark.usefixtures("no_kill")
    @pytest.mark.parametrize(("exc_name", "label"), [
        ("ConnectionError", "NETWORK FAILURE"),
        ("Timeout", "NETWORK FAILURE"),
        ("RequestException", "API ERROR"),
        ("TypeError", "PARSE ERROR"),
    ])
    def test_exchange_errors(self, fake_store, tracked, logs, exc_name, label):
        exc_type = TypeError if exc_name == "TypeError" else getattr(ex.requests, exc_name)
        client = make_client(mid=0.5)
        client.create_order.side_effect = exc_type("boom")
        e = live_executor(client)
        assert run(e.place_sell_order("tok", 10, 0.5, "m1", city_id="chi")) is None
        assert logs()[-1] == ("ERROR", f"LIVE SELL {label}: chi, boom")
        assert fake_store.saved == [] and tracked == []
