"""Log Polymarket temperature-market prices over time, for later scoring.

The project never had historical order-book data, so it could not check its
forecasts against the prices it would actually have traded at. This records,
on a schedule, every active daily-high market's metadata, top of book and
(optionally) book depth, plus each market's final outcome once the oracle
resolves it. Nothing here trades.

Storage is its own SQLite file (default ``prices.db`` next to the bot's DB).

    python -m weather_edge log-prices                 # one snapshot
    python -m weather_edge log-prices --interval 15   # every 15 minutes
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx

from weather_edge.fetchers.polymarket import (
    MarketInfo,
    discover_weather_markets,
    parse_market_question,
)

logger = logging.getLogger(__name__)

DEFAULT_PRICE_DB = Path(__file__).parent.parent.parent / "prices.db"
BOOK_LEVELS = 5
BOOK_CONCURRENCY = 5
HISTORY_CONCURRENCY = 8
HISTORY_FIDELITY_MIN = 60  # CLOB prices-history resolution for the backfill
GAMMA_PAGE = 100

_SCHEMA = """
CREATE TABLE IF NOT EXISTS markets (
    id INTEGER PRIMARY KEY,              -- compact key used by the time series
    market_id TEXT NOT NULL UNIQUE,      -- Gamma conditionId
    city_id TEXT,
    target_date TEXT,                    -- station-local resolution day
    market_type TEXT,
    question TEXT,
    event_title TEXT,
    threshold_dir TEXT,                  -- gte / lte / range / exact
    bucket_low_int REAL,                 -- native-unit labels (inclusive)
    bucket_high_int REAL,
    threshold_unit TEXT,
    token_id_yes TEXT,
    token_id_no TEXT,
    first_seen TEXT,
    last_seen TEXT,
    resolved_yes INTEGER,                -- 1 YES won, 0 NO won, NULL pending
    resolved_at TEXT
);
CREATE TABLE IF NOT EXISTS snapshots (
    ts INTEGER NOT NULL,                 -- UTC unix seconds
    mkt INTEGER NOT NULL REFERENCES markets(id),
    yes_price REAL,                      -- Gamma outcomePrices[0]
    no_price REAL,
    best_bid REAL,                       -- YES top of book
    best_ask REAL,
    spread REAL,
    volume_24h REAL,
    liquidity REAL,
    PRIMARY KEY (mkt, ts)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS book_levels (
    ts INTEGER NOT NULL,
    mkt INTEGER NOT NULL REFERENCES markets(id),
    side TEXT NOT NULL,                  -- bid / ask, YES token
    level INTEGER NOT NULL,              -- 0 = best
    price REAL,
    size REAL,
    PRIMARY KEY (mkt, ts, side, level)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS price_history (
    mkt INTEGER NOT NULL REFERENCES markets(id),
    ts INTEGER NOT NULL,                 -- UTC unix seconds
    price REAL NOT NULL,                 -- YES price (CLOB prices-history)
    PRIMARY KEY (mkt, ts)
) WITHOUT ROWID;
CREATE VIEW IF NOT EXISTS snapshot_view AS
    SELECT datetime(s.ts, 'unixepoch') AS ts_utc, m.market_id, m.city_id, m.target_date,
           m.threshold_dir, m.bucket_low_int, m.bucket_high_int, m.threshold_unit,
           s.yes_price, s.best_bid, s.best_ask, s.spread, s.volume_24h, s.liquidity
    FROM snapshots s JOIN markets m ON m.id = s.mkt;
CREATE INDEX IF NOT EXISTS idx_markets_pending ON markets (resolved_yes, target_date);
"""


class PriceLogStore:
    """SQLite store for price snapshots. One writer at a time."""

    def __init__(self, path: Path | str = DEFAULT_PRICE_DB):
        self.path = Path(path)
        # WAL: readers (analysis) never block the logger, nor it them.
        self.conn = sqlite3.connect(self.path, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def upsert_market(self, m: MarketInfo, ts: str) -> int:
        """Insert or touch a market; returns its compact id."""
        self.conn.execute(
            """INSERT INTO markets (market_id, city_id, target_date, market_type, question,
                   event_title, threshold_dir, bucket_low_int, bucket_high_int,
                   threshold_unit, token_id_yes, token_id_no, first_seen, last_seen)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(market_id) DO UPDATE SET last_seen = excluded.last_seen""",
            (m.market_id, m.city_id.value if m.city_id else None, str(m.target_date),
             m.market_type.value, m.question, m.event_title, m.threshold_dir,
             m.bucket_low_int, m.bucket_high_int, m.threshold_unit,
             m.token_id_yes, m.token_id_no, ts, ts),
        )
        return self.conn.execute(
            "SELECT id FROM markets WHERE market_id = ?", (m.market_id,),
        ).fetchone()[0]

    def add_snapshot(self, mkt: int, m: MarketInfo, ts: int) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO snapshots (ts, mkt, yes_price, no_price,
                   best_bid, best_ask, spread, volume_24h, liquidity)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts, mkt, m.yes_price, m.no_price, m.best_bid, m.best_ask,
             m.spread, m.volume_24h, m.liquidity),
        )

    def add_book(self, mkt: int, ts: int, book: dict) -> None:
        rows = []
        for side, key in (("bid", "bids"), ("ask", "asks")):
            levels = _sorted_levels(book.get(key) or [], best_high=(side == "bid"))
            rows += [(ts, mkt, side, i, p, s) for i, (p, s) in enumerate(levels)]
        self.conn.executemany(
            "INSERT OR REPLACE INTO book_levels VALUES (?, ?, ?, ?, ?, ?)", rows,
        )

    def pending_market_ids(self, today: date) -> set[str]:
        """Markets whose day has passed and whose outcome isn't recorded yet."""
        rows = self.conn.execute(
            "SELECT market_id FROM markets WHERE resolved_yes IS NULL AND target_date < ?",
            (str(today),),
        ).fetchall()
        return {r["market_id"] for r in rows}

    def set_resolution(self, market_id: str, yes_won: bool, ts: str) -> None:
        self.conn.execute(
            "UPDATE markets SET resolved_yes = ?, resolved_at = ? WHERE market_id = ?",
            (int(yes_won), ts, market_id),
        )

    def has_history(self, market_id: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM price_history h JOIN markets m ON m.id = h.mkt "
            "WHERE m.market_id = ? LIMIT 1", (market_id,),
        ).fetchone() is not None

    def add_history(self, mkt: int, points: list[dict]) -> int:
        rows = []
        for pt in points:
            try:
                rows.append((mkt, int(pt["t"]), float(pt["p"])))
            except (KeyError, TypeError, ValueError):
                continue
        self.conn.executemany("INSERT OR REPLACE INTO price_history VALUES (?, ?, ?)", rows)
        return len(rows)

    def commit(self) -> None:
        self.conn.commit()


def _sorted_levels(levels: list[dict], *, best_high: bool) -> list[tuple[float, float]]:
    """Top BOOK_LEVELS (price, size) pairs, best first; bad rows are skipped."""
    parsed = []
    for lvl in levels:
        try:
            parsed.append((float(lvl["price"]), float(lvl["size"])))
        except (KeyError, TypeError, ValueError):
            continue
    parsed.sort(key=lambda ps: ps[0], reverse=best_high)
    return parsed[:BOOK_LEVELS]


async def _fetch_books(markets: list[MarketInfo], clob_url: str) -> dict[str, dict]:
    """YES-token order books keyed by market_id (failures are omitted)."""
    sem = asyncio.Semaphore(BOOK_CONCURRENCY)
    books: dict[str, dict] = {}

    async with httpx.AsyncClient() as client:
        async def one(m: MarketInfo) -> None:
            async with sem:
                try:
                    resp = await client.get(
                        f"{clob_url}/book", params={"token_id": m.token_id_yes}, timeout=10.0,
                    )
                    resp.raise_for_status()
                    books[m.market_id] = resp.json()
                except (httpx.HTTPError, ValueError) as e:
                    logger.debug("Book fetch failed for %s: %s", m.market_id[:16], e)

        await asyncio.gather(*(one(m) for m in markets if m.token_id_yes))
    return books


async def snapshot_once(
    store: PriceLogStore, *, books: bool = True, book_days: int = 3,
    now: datetime | None = None,
) -> dict[str, int]:
    """Record one snapshot of every active temperature market."""
    from weather_edge.config import settings

    now = now or datetime.now(UTC)
    iso, ts = now.isoformat(timespec="seconds"), int(now.timestamp())
    markets = [m for m in await discover_weather_markets() if m.city_id is not None]
    ids = {}
    for m in markets:
        ids[m.market_id] = store.upsert_market(m, iso)
        store.add_snapshot(ids[m.market_id], m, ts)
    store.commit()  # don't hold the write lock across the book fetch

    n_books = 0
    if books and markets:
        horizon = now.date() + timedelta(days=book_days)
        near = [m for m in markets if m.target_date <= horizon]
        fetched = await _fetch_books(near, settings.polymarket_clob_url)
        for market_id, book in fetched.items():
            store.add_book(ids[market_id], ts, book)
        n_books = len(fetched)
    store.commit()
    return {"markets": len(markets), "books": n_books}


async def record_resolutions(store: PriceLogStore, now: datetime | None = None) -> int:
    """Record the outcome of logged markets the oracle has resolved."""
    from weather_edge.analysis.resolver import fetch_resolved_markets

    now = now or datetime.now(UTC)
    pending = store.pending_market_ids(now.date())
    if not pending:
        return 0
    resolved = await fetch_resolved_markets()
    ts = now.isoformat(timespec="seconds")
    done = 0
    for market_id in pending & resolved.keys():
        store.set_resolution(market_id, resolved[market_id], ts)
        done += 1
    store.commit()
    return done


async def run_logger(
    store: PriceLogStore, *, interval_min: float, books: bool = True, book_days: int = 3,
    book_every: int = 1, iterations: int | None = None,
) -> None:
    """Snapshot every ``interval_min`` minutes; one failed pass doesn't stop the loop.

    Book depth is recorded on every ``book_every``-th pass (it is ~5x the
    size of a top-of-book snapshot); top of book is recorded on every pass.
    """
    n = 0
    while iterations is None or n < iterations:
        n += 1
        with_books = books and (n - 1) % max(1, book_every) == 0
        try:
            counts = await snapshot_once(store, books=with_books, book_days=book_days)
            resolved = await record_resolutions(store)
            logger.info(
                "PRICE LOG: %d markets, %d books, %d newly resolved",
                counts["markets"], counts["books"], resolved,
            )
        except (httpx.HTTPError, sqlite3.Error, ValueError) as e:
            logger.error("PRICE LOG pass failed: %s", e, exc_info=True)
        if iterations is not None and n >= iterations:
            break
        await asyncio.sleep(interval_min * 60)


# ---------------------------------------------------------------------------
# Backfill: hourly price history of already-resolved markets
# ---------------------------------------------------------------------------
# The CLOB keeps prices-history for resolved markets for roughly a month, so
# anything older than that is lost unless it is fetched while it still exists.


def _json_list(value) -> list:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return value if isinstance(value, list) else []


def _resolved_market(mkt: dict, event: dict) -> tuple[MarketInfo, bool] | None:
    """(parsed market, YES won) for a resolved tracked-city market, else None."""
    from weather_edge.analysis.resolver import _is_market_resolved

    if not _is_market_resolved(mkt):
        return None
    parsed = parse_market_question(
        mkt.get("question", ""), event.get("title", ""), mkt.get("conditionId", ""),
        event.get("endDate"),
    )
    if parsed is None or parsed.city_id is None:
        return None
    tokens, prices = _json_list(mkt.get("clobTokenIds")), _json_list(mkt.get("outcomePrices"))
    if len(tokens) < 2 or len(prices) < 2:
        return None
    parsed.token_id_yes, parsed.token_id_no = tokens[0], tokens[1]
    try:
        return parsed, float(prices[0]) > 0.5
    except (TypeError, ValueError):
        return None


async def _closed_events_for_day(client: httpx.AsyncClient, gamma: str, day: date) -> list:
    """All closed weather events whose end date falls on ``day`` (UTC)."""
    events, offset = [], 0
    while True:
        resp = await client.get(f"{gamma}/events", params={
            "tag_slug": "weather", "closed": "true", "limit": GAMMA_PAGE, "offset": offset,
            "end_date_min": f"{day}T00:00:00Z", "end_date_max": f"{day + timedelta(days=1)}T00:00:00Z",
        }, timeout=30.0)
        resp.raise_for_status()
        page = resp.json()
        events += [e for e in page if "highest temperature" in e.get("title", "").lower()]
        if len(page) < GAMMA_PAGE:
            return events
        offset += GAMMA_PAGE


async def _fetch_history(client: httpx.AsyncClient, clob: str, token: str) -> list[dict]:
    for attempt in range(4):
        resp = await client.get(f"{clob}/prices-history", params={
            "market": token, "interval": "max", "fidelity": HISTORY_FIDELITY_MIN,
        }, timeout=30.0)
        if resp.status_code == 429:
            await asyncio.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        return resp.json().get("history", []) or []
    return []


async def backfill_history(
    store: PriceLogStore, *, days: int = 35, today: date | None = None,
) -> dict[str, int]:
    """Store hourly YES price history and outcomes for resolved markets."""
    from weather_edge.config import settings

    today = today or datetime.now(UTC).date()
    now_iso = datetime.now(UTC).isoformat(timespec="seconds")
    counts = {"markets": 0, "points": 0, "empty": 0, "skipped": 0}
    sem = asyncio.Semaphore(HISTORY_CONCURRENCY)
    async with httpx.AsyncClient() as client:
        todo: list[tuple[MarketInfo, bool]] = []
        for back in range(days, 0, -1):
            events = await _closed_events_for_day(client, settings.polymarket_gamma_url,
                                                  today - timedelta(days=back))
            todo += [r for e in events for mkt in e.get("markets", [])
                     if (r := _resolved_market(mkt, e)) is not None]

        async def one(market: MarketInfo, yes_won: bool) -> None:
            if store.has_history(market.market_id):
                counts["skipped"] += 1
                return
            async with sem:
                try:
                    points = await _fetch_history(client, settings.polymarket_clob_url,
                                                  market.token_id_yes)
                except (httpx.HTTPError, ValueError) as e:
                    logger.warning("History fetch failed for %s: %s", market.market_id[:16], e)
                    return
            mkt = store.upsert_market(market, now_iso)
            store.set_resolution(market.market_id, yes_won, now_iso)
            added = store.add_history(mkt, points)
            counts["markets"] += 1
            counts["points"] += added
            counts["empty"] += added == 0

        await asyncio.gather(*(one(m, won) for m, won in todo))
    store.commit()
    return counts

