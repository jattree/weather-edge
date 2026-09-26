"""Price logger: snapshots, book depth, resolutions, resilience (no network)."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime

import httpx
import pytest

from weather_edge import price_logger
from weather_edge.analysis import resolver
from weather_edge.fetchers.polymarket import MarketInfo
from weather_edge.models.enums import City

NOW = datetime(2026, 9, 26, 15, 0, tzinfo=UTC)


def _market(mid: str, target: date, *, yes=0.30, bid=0.29, ask=0.31) -> MarketInfo:
    return MarketInfo(
        market_id=mid, condition_id=mid, token_id_yes=f"{mid}-Y", token_id_no=f"{mid}-N",
        city_id=City.NYC, question=f"Will NYC be 70-71F on {target}?", target_date=target,
        threshold_dir="range", bucket_low_int=70, bucket_high_int=71,
        yes_price=yes, no_price=1 - yes, best_bid=bid, best_ask=ask, spread=ask - bid,
        volume_24h=1000.0, liquidity=500.0,
    )


class _BookClient:
    def __init__(self, books: dict[str, dict]):
        self.books = books

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, params=None, timeout=None):
        token = params["token_id"]
        request = httpx.Request("GET", url)
        if token not in self.books:
            return httpx.Response(404, request=request)
        return httpx.Response(200, json=self.books[token], request=request)


@pytest.fixture
def store(tmp_path):
    s = price_logger.PriceLogStore(tmp_path / "prices.db")
    yield s
    s.close()


def _discover(monkeypatch, markets):
    async def fake():
        return markets
    monkeypatch.setattr(price_logger, "discover_weather_markets", fake)


def test_snapshot_records_markets_prices_and_near_books(store, monkeypatch):
    near, far = _market("m-near", date(2026, 9, 27)), _market("m-far", date(2026, 10, 5))
    _discover(monkeypatch, [near, far])
    book = {"bids": [{"price": "0.28", "size": "10"}, {"price": "0.29", "size": "5"}],
            "asks": [{"price": "0.33", "size": "4"}, {"price": "0.31", "size": "7"},
                     {"price": "bad", "size": "1"}]}
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda *a, **k: _BookClient({"m-near-Y": book, "m-far-Y": book}))

    counts = asyncio.run(price_logger.snapshot_once(store, now=NOW))

    assert counts == {"markets": 2, "books": 1}  # m-far is beyond --book-days
    rows = store.conn.execute(
        "SELECT market_id, yes_price, best_bid, best_ask FROM snapshot_view ORDER BY market_id",
    ).fetchall()
    assert [tuple(r) for r in rows] == [("m-far", 0.30, 0.29, 0.31), ("m-near", 0.30, 0.29, 0.31)]
    levels = store.conn.execute(
        "SELECT side, level, price, size FROM book_levels ORDER BY side, level",
    ).fetchall()
    # best first on each side; the unparseable level is dropped
    assert [tuple(r) for r in levels] == [
        ("ask", 0, 0.31, 7.0), ("ask", 1, 0.33, 4.0),
        ("bid", 0, 0.29, 5.0), ("bid", 1, 0.28, 10.0),
    ]
    meta = store.conn.execute("SELECT * FROM markets WHERE market_id='m-near'").fetchone()
    assert (meta["city_id"], meta["target_date"], meta["bucket_low_int"]) == ("nyc", "2026-09-27", 70)


def test_repeat_snapshots_keep_first_seen(store, monkeypatch):
    m = _market("m1", date(2026, 9, 28))
    _discover(monkeypatch, [m])
    asyncio.run(price_logger.snapshot_once(store, books=False, now=NOW))
    later = NOW.replace(hour=16)
    asyncio.run(price_logger.snapshot_once(store, books=False, now=later))
    row = store.conn.execute("SELECT first_seen, last_seen FROM markets").fetchone()
    assert row["first_seen"].startswith("2026-09-26T15") and row["last_seen"].startswith(
        "2026-09-26T16")
    ts = [r[0] for r in store.conn.execute("SELECT ts_utc FROM snapshot_view ORDER BY ts_utc")]
    assert ts == ["2026-09-26 15:00:00", "2026-09-26 16:00:00"]
    assert store.conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0] == 2


def test_resolutions_only_for_past_logged_markets(store, monkeypatch):
    _discover(monkeypatch, [_market("past", date(2026, 9, 25)),
                            _market("today", date(2026, 9, 26))])
    asyncio.run(price_logger.snapshot_once(store, books=False, now=NOW))

    async def fake_resolved():
        return {"past": True, "today": False, "unlogged": True}
    monkeypatch.setattr(resolver, "fetch_resolved_markets", fake_resolved)

    assert asyncio.run(price_logger.record_resolutions(store, now=NOW)) == 1
    got = dict(store.conn.execute("SELECT market_id, resolved_yes FROM markets").fetchall())
    assert got == {"past": 1, "today": None}


def test_loop_survives_a_failed_pass(store, monkeypatch):
    calls = []

    async def flaky(store, **kw):
        calls.append(kw["books"])
        if len(calls) == 1:
            raise httpx.ConnectError("gamma down")
        return {"markets": 0, "books": 0}

    async def no_resolutions(store, now=None):
        return 0

    async def no_sleep(_):
        return None

    monkeypatch.setattr(price_logger, "snapshot_once", flaky)
    monkeypatch.setattr(price_logger, "record_resolutions", no_resolutions)
    monkeypatch.setattr(price_logger.asyncio, "sleep", no_sleep)
    asyncio.run(price_logger.run_logger(store, interval_min=1, iterations=5, book_every=4))
    assert calls == [True, False, False, False, True]  # books on every 4th pass


# ---------------------------------------------------------------------------
# backfill of resolved markets
# ---------------------------------------------------------------------------

def _event(day: str, city_q: str, yes_won: bool, cid: str) -> dict:
    return {
        "title": f"Highest temperature in NYC on {day}?", "endDate": f"2026-{day}T12:00:00Z",
        "markets": [{
            "question": city_q, "conditionId": cid, "umaResolutionStatus": "resolved",
            "outcomePrices": '["1", "0"]' if yes_won else '["0", "1"]',
            "clobTokenIds": f'["{cid}-Y", "{cid}-N"]',
        }],
    }


class _BackfillClient:
    def __init__(self, events_by_min: dict[str, list], history: dict[str, list]):
        self.events_by_min, self.history, self.history_calls = events_by_min, history, 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, params=None, timeout=None):
        req = httpx.Request("GET", url)
        if url.endswith("/events"):
            events = self.events_by_min.get(params["end_date_min"][:10], [])
            return httpx.Response(200, json=events[params["offset"]:], request=req)
        self.history_calls += 1
        return httpx.Response(200, json={"history": self.history.get(params["market"], [])},
                              request=req)


def test_backfill_stores_history_and_outcomes_idempotently(store, monkeypatch):
    q = "Will the highest temperature in New York City be between 70-71°F on September 20?"
    client = _BackfillClient(
        {"2026-09-20": [_event("09-20", q, True, "c1")],
         "2026-09-21": [_event("09-21", q.replace("20?", "21?"), False, "c2"),
                        {"title": "Will it snow in NYC?", "markets": []}]},
        {"c1-Y": [{"t": 1790000000, "p": 0.3}, {"t": 1790003600, "p": 0.6}],
         "c2-Y": [{"t": 1790090000, "p": "bad"}]},
    )
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: client)
    counts = asyncio.run(price_logger.backfill_history(store, days=6, today=date(2026, 9, 26)))
    assert counts == {"markets": 2, "points": 2, "empty": 1, "skipped": 0}
    got = dict(store.conn.execute("SELECT market_id, resolved_yes FROM markets").fetchall())
    assert got == {"c1": 1, "c2": 0}
    assert store.conn.execute("SELECT COUNT(*) FROM price_history").fetchone()[0] == 2

    calls_before = client.history_calls
    again = asyncio.run(price_logger.backfill_history(store, days=6, today=date(2026, 9, 26)))
    assert again["skipped"] == 1 and client.history_calls == calls_before + 1  # c2 had none


def test_unresolved_or_untracked_markets_are_ignored():
    ev = _event("09-20", "Will the highest temperature in Zhengzhou be 22°C on September 20?",
                True, "z1")
    ev["title"] = "Highest temperature in Zhengzhou on September 20?"
    assert price_logger._resolved_market(ev["markets"][0], ev) is None  # untracked city
    q = "Will the highest temperature in New York City be between 70-71°F on September 20?"
    ev = _event("09-20", q, True, "n1")
    ev["markets"][0]["umaResolutionStatus"] = "proposed"
    assert price_logger._resolved_market(ev["markets"][0], ev) is None
