"""Regression tests for the post-merge loose ends.

No network: HTTP clients and Redis are faked.
"""
from __future__ import annotations

import asyncio
import sys
import types
from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest

from weather_edge import live_state, scheduler
from weather_edge.models.enums import City, WeatherModel


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """Records GET params and replays one canned payload for every call."""

    calls: list[tuple[str, dict]] = []
    payload = None

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, params=None, **kw):
        type(self).calls.append((url, params or {}))
        return _Resp(type(self).payload)


@pytest.fixture
def fake_client():
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.payload = None
    return _FakeAsyncClient


@pytest.fixture
def no_health(monkeypatch):
    from weather_edge.analysis import service_health
    monkeypatch.setattr(service_health, "record_service_call", lambda *a, **k: None)


# ---------------------------------------------------------------------------
# 3. Gamma top of book reaches MarketInfo and the spread estimate
# ---------------------------------------------------------------------------

def _gamma_event(target: date, **book):
    mkt = {
        "conditionId": "0xabc",
        "question": f"Will the highest temperature in Dallas be between 84-85°F on "
                    f"{target.strftime('%B')} {target.day}?",
        "outcomePrices": '["0.40", "0.60"]',
        "clobTokenIds": '["1", "2"]',
        **book,
    }
    return [{
        "title": f"Highest temperature in Dallas on {target.strftime('%B')} {target.day}?",
        "endDate": f"{target.isoformat()}T12:00:00Z",
        "markets": [mkt],
    }]


def _discover(monkeypatch, fake_client, payload):
    from weather_edge.fetchers import polymarket
    monkeypatch.setattr(polymarket.httpx, "AsyncClient", fake_client)
    fake_client.payload = payload
    return asyncio.run(polymarket.discover_weather_markets())


def test_gamma_book_fields_parsed(monkeypatch, fake_client, no_health):
    from weather_edge.trading.market_maker import estimate_market_spread
    target = date.today()
    markets = _discover(
        monkeypatch, fake_client,
        _gamma_event(target, bestBid="0.38", bestAsk=0.45, spread="0.07"),
    )
    assert markets, "fixture market should parse"
    m = markets[0]
    assert (m.best_bid, m.best_ask, m.spread) == (0.38, 0.45, 0.07)
    assert estimate_market_spread(m) == pytest.approx(0.07)


def test_gamma_book_fields_absent_or_bad(monkeypatch, fake_client, no_health):
    from weather_edge.trading.market_maker import DEFAULT_SPREAD_ESTIMATE, estimate_market_spread
    markets = _discover(
        monkeypatch, fake_client,
        _gamma_event(date.today(), bestBid="n/a", spread=None),
    )
    m = markets[0]
    assert m.best_bid is None and m.best_ask is None and m.spread is None
    assert estimate_market_spread(m) == DEFAULT_SPREAD_ESTIMATE


def test_optional_float():
    from weather_edge.fetchers.polymarket import _optional_float
    assert _optional_float("0.5") == 0.5
    for bad in (None, "", "x", "nan", float("inf"), [1]):
        assert _optional_float(bad) is None


# ---------------------------------------------------------------------------
# 4. Exit reviews use the trade's own target date
# ---------------------------------------------------------------------------

def _fc(model, temp):
    return SimpleNamespace(model_name=model, temp_max_c=temp)


def test_exit_context_uses_market_target_date():
    d1, d2 = date(2026, 9, 26), date(2026, 9, 27)
    # d1 inserted first: the old lookup took the first entry for the city.
    cache = {
        (City.DAL, d1): [_fc("gfs", 10.0), _fc("ecmwf", 12.0)],
        (City.DAL, d2): [_fc("gfs", 30.0), _fc("ecmwf", 34.0)],
    }
    trade = SimpleNamespace(market_id="m2", city_id="dal")
    vals, mean, std = scheduler.exit_model_context(trade, cache, {"m2": d2})
    assert vals == {"gfs": 30.0, "ecmwf": 34.0}
    assert mean == pytest.approx(32.0) and std == pytest.approx(2.0)


def test_exit_context_trade_target_date_wins():
    d1, d2 = date(2026, 9, 26), date(2026, 9, 27)
    cache = {(City.DAL, d1): [_fc("gfs", 10.0)], (City.DAL, d2): [_fc("gfs", 30.0)]}
    trade = SimpleNamespace(market_id="m", city_id="dal", target_date=d1)
    vals, _, _ = scheduler.exit_model_context(trade, cache, {"m": d2})
    assert vals == {"gfs": 10.0}


def test_exit_context_never_borrows_another_date():
    cache = {(City.DAL, date(2026, 9, 26)): [_fc("gfs", 10.0)]}
    trade = SimpleNamespace(market_id="m", city_id="dal")
    # Unknown date, and a known date with no forecasts: both empty context
    assert scheduler.exit_model_context(trade, cache, {}) == ({}, 0.0, 1.0)
    assert scheduler.exit_model_context(
        trade, cache, {"m": date(2026, 9, 27)},
    ) == ({}, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 5. Sniper full probe asks for the city's local day
# ---------------------------------------------------------------------------

def test_sniper_probe_uses_city_timezone(monkeypatch, fake_client):
    from weather_edge.analysis import sniper
    monkeypatch.setattr(sniper.httpx, "AsyncClient", fake_client)
    fake_client.payload = {"daily": {"temperature_2m_max": [20.0]}}
    snap = asyncio.run(
        sniper.ModelSniper().probe_model(WeatherModel.GFS, City.WLG, date(2026, 3, 27)),
    )
    assert snap.temp_max_c == 20.0
    assert fake_client.calls[0][1]["timezone"] == "Pacific/Auckland"


# ---------------------------------------------------------------------------
# 6. Hours to resolution: end of the target date in the city's zone
# ---------------------------------------------------------------------------

def test_hours_to_local_end_of_day():
    now = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)
    d = date(2026, 9, 26)
    # UTC midnight after d is 36h away; local ends differ by the UTC offset.
    assert scheduler.hours_to_local_end_of_day(City.TYO, d, now) == pytest.approx(27.0)  # UTC+9
    assert scheduler.hours_to_local_end_of_day(City.NYC, d, now) == pytest.approx(40.0)  # EDT
    assert scheduler.hours_to_local_end_of_day(City.LON, d, now) == pytest.approx(35.0)  # BST
    # Already past: clamped to zero
    assert scheduler.hours_to_local_end_of_day(City.TYO, date(2026, 9, 24), now) == 0.0


# ---------------------------------------------------------------------------
# 7. Redis connect backoff
# ---------------------------------------------------------------------------

class _FailingRedis:
    constructed = 0

    def __init__(self, *a, **kw):
        type(self).constructed += 1

    def ping(self):
        raise ConnectionError("refused")


class _OkRedis:
    def __init__(self, *a, **kw):
        self.store = {}

    def ping(self):
        return True

    def get(self, key):
        return self.store.get(key)


@pytest.fixture
def redis_state(monkeypatch, no_health):
    clock = {"t": 1000.0}
    monkeypatch.setattr(live_state.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(live_state, "_redis_client", None)
    monkeypatch.setattr(live_state, "_redis_ever_connected", False)
    monkeypatch.setattr(live_state, "_redis_retry_after", 0.0)
    monkeypatch.setattr(live_state, "_fallback_cache", {})
    fake = types.ModuleType("redis")
    fake.Redis = _FailingRedis
    monkeypatch.setitem(sys.modules, "redis", fake)
    _FailingRedis.constructed = 0
    return clock, fake


def test_failed_connect_not_retried_within_backoff(redis_state):
    clock, _ = redis_state
    assert live_state._get_redis() is None
    assert live_state._get_redis() is None
    live_state.set_value("k", "v")
    assert live_state.get_value("k") == "v"
    assert _FailingRedis.constructed == 1

    clock["t"] += live_state.REDIS_RETRY_BACKOFF_S + 1
    assert live_state._get_redis() is None
    assert _FailingRedis.constructed == 2


def test_reconnects_after_backoff(redis_state):
    clock, fake = redis_state
    assert live_state._get_redis() is None
    fake.Redis = _OkRedis
    assert live_state._get_redis() is None  # still backing off
    clock["t"] += live_state.REDIS_RETRY_BACKOFF_S + 1
    assert isinstance(live_state._get_redis(), _OkRedis)
    assert live_state._redis_ever_connected


def test_strict_read_during_backoff_after_prior_connect_raises(redis_state, monkeypatch):
    monkeypatch.setattr(live_state, "_redis_ever_connected", True)
    assert live_state._get_redis() is None  # fails, starts backoff
    with pytest.raises(live_state.LiveStateUnavailableError):
        live_state.get_value_strict("kill")
    with pytest.raises(live_state.LiveStateUnavailableError):
        live_state.get_value_strict("kill")  # within backoff: still raises
    assert _FailingRedis.constructed == 1


def test_strict_read_never_connected_uses_fallback(redis_state):
    live_state._fallback_cache["kill"] = "1"
    assert live_state.get_value_strict("kill") == "1"
    assert live_state.get_value_strict("kill") == "1"
    assert _FailingRedis.constructed == 1
