"""Regression tests for bugs found while refactoring scheduler.run_cycle."""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pytest

from tests.test_run_cycle_characterization import (
    FakeExecutor,
    Harness,
    events,
    logs,
    seed_live_trade,
    seed_position,
)
from weather_edge import scheduler
from weather_edge.analysis import claude_reasoning
from weather_edge.analysis.edge import Signal
from weather_edge.fetchers import polymarket
from weather_edge.models.enums import City, MarketType, SignalTier, TradeSide


@pytest.fixture
def h(monkeypatch, tmp_path, caplog):
    harness = Harness(monkeypatch, tmp_path, caplog)
    yield harness
    claude_reasoning.clear_decisions()


# ---------------------------------------------------------------------------
# 1. live executor with no store
# ---------------------------------------------------------------------------


async def test_live_executor_without_store_does_not_crash_cycle(h):
    ex = h.executor(balance=500.0)
    h.market("nyc-a", City.NYC, 2, 0.30, value=21.0)
    h.edge("nyc-a", edge=0.05, side="NO")

    await h.run(None, live_executor=ex)

    snap = h.snapshot()
    placed = [e[1]["market_id"] for e in events(snap, "exec.place_limit_order")]
    assert placed == ["nyc-a"]
    assert logs(snap, "NO STORE")


# ---------------------------------------------------------------------------
# 2. live hedges count against the balance and the position cap
# ---------------------------------------------------------------------------


def _signal(mid="m1", side="NO", size=20.0, market_prob=0.30, edge=0.09) -> Signal:
    return Signal(
        market_id=mid, consensus_id=None, computed_at=datetime.now(UTC),
        model_prob=0.2, model_confidence=0.9, market_prob=market_prob, edge=edge,
        net_edge=edge - 0.01, edge_pct=0.3, kelly_fraction=0.1, half_kelly=0.05,
        recommended_side=TradeSide(side), recommended_size=size,
        confidence_tier=SignalTier.HIGH, city_id="nyc", description="d",
        target_date="2026-01-01", strategy="core",
    )


class _FakeMaker:
    def __init__(self, cost=4.0):
        self.cost = cost

    def generate_hedge_orders(self, signal, market_prices, bankroll, is_live=False):
        return SimpleNamespace(market_id=signal.market_id, side="YES", limit_price=0.31,
                               cost=self.cost, description="hedge")


def _hedge_ctx(balance, count=0, max_positions=50, cost=4.0):
    ex = FakeExecutor([], balance=balance)
    ctx = scheduler.CycleContext(
        paper_trader=None, live_executor=ex, store=None, run_ai_reasoning=False,
        target_dates=[], min_horizon_hours=0, now_utc=datetime.now(UTC), forecast_cache={},
    )
    ctx.market_maker = _FakeMaker(cost)
    ctx.market_by_id = {"m1": SimpleNamespace(token_id_yes="m1-Y", token_id_no="m1-N")}
    ctx.live_balance = balance
    ctx.active_position_count = count
    ctx.max_positions = max_positions
    return ctx, ex


async def test_live_hedge_deducts_balance_and_counts_position():
    ctx, ex = _hedge_ctx(100.0)
    await scheduler._place_live_hedge(ctx, _signal())
    assert [e[0] for e in ex.events] == ["exec.place_limit_order"]
    assert ctx.live_balance == pytest.approx(96.0)
    assert ctx.active_position_count == 1


async def test_live_hedge_skipped_when_over_balance():
    ctx, ex = _hedge_ctx(3.0, cost=4.0)
    await scheduler._place_live_hedge(ctx, _signal())
    assert ex.events == []
    assert ctx.live_balance == 3.0 and ctx.active_position_count == 0


async def test_live_hedge_skipped_at_position_cap():
    ctx, ex = _hedge_ctx(100.0, count=50, max_positions=50)
    await scheduler._place_live_hedge(ctx, _signal())
    assert ex.events == []
    assert ctx.active_position_count == 50


# ---------------------------------------------------------------------------
# 3. price chase must not rewrite the signal's observed market price
# ---------------------------------------------------------------------------


async def test_price_chase_keeps_observed_market_prob(h):
    store = h.store()
    ex = h.executor(balance=500.0)
    seed_live_trade(store, "o-old", "chase", "YES", 0.20, placed_ago_min=120)
    h.market("chase", City.ATL, 2, 0.30, value=21.0)
    h.edge("chase", edge=0.05, side="YES")

    signals, _, _ = await h.run(None, store=store, live_executor=ex)

    snap = h.snapshot()
    assert logs(snap, "PRICE CHASE:")
    (order,) = [e[1] for e in events(snap, "exec.place_limit_order")]
    assert order["market_prob"] == pytest.approx(0.215)  # chased to 0.21 + half tick
    (sig,) = signals
    assert sig.market_prob == pytest.approx(0.30)  # observed market price


# ---------------------------------------------------------------------------
# 4. deterministic variable order
# ---------------------------------------------------------------------------


def test_variables_needed_is_canonically_ordered():
    markets = [SimpleNamespace(market_type=t) for t in (
        MarketType.SNOW, MarketType.PRECIP, MarketType.TEMP_LOW,
    )]
    assert scheduler._variables_needed(markets) == [
        "temp_max_c", "temp_min_c", "precip_sum_mm", "snow_sum_cm",
    ]


# ---------------------------------------------------------------------------
# 5. dead-by-construction branches (documents why they were removed / kept)
# ---------------------------------------------------------------------------


def test_fee_gate_only_blocks_edges_below_taker_threshold():
    """validate_fee_alpha_ratio blocks only edge < 0.05 * p(1-p) / 0.4 <= 3.125%.

    Taker entries need edge >= 8%, so a fee-blocked taker cannot exist.
    """
    from weather_edge.analysis.contracts import validate_fee_alpha_ratio

    for p in [i / 100 for i in range(1, 100)]:
        for size in (1.0, 20.0, 500.0):
            assert validate_fee_alpha_ratio(edge=0.08, price=p, size_usd=size).valid
            assert validate_fee_alpha_ratio(edge=0.0313, price=p, size_usd=size).valid


def test_gross_exposure_cannot_bind_after_correlation_limit():
    """Every city is in a correlation group, so size <= group_pct * NAV going in.

    total_at_risk = NAV - cash <= NAV, so gross headroom >= (multiple - 1) * NAV,
    which exceeds group_pct * NAV for every built-in profile.
    """
    from weather_edge.analysis import risk_controls

    assert all(c.value in risk_controls.CITY_TO_GROUP for c in City)
    for profile in risk_controls.RISK_PROFILES.values():
        assert profile.max_gross_exposure_multiple - 1 > profile.max_group_exposure_pct


# ---------------------------------------------------------------------------
# 6. data-API position cleanup must never zero on a truncated response
# ---------------------------------------------------------------------------


class _PagedClient:
    """httpx.AsyncClient stand-in serving /positions in pages."""

    def __init__(self, pages, calls):
        self.pages = pages
        self.calls = calls

    def __call__(self, *a, **k):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, params=None, **kw):
        self.calls.append(dict(params or {}))
        i = len(self.calls) - 1
        spec = self.pages[i] if i < len(self.pages) else (200, [])
        if isinstance(spec, Exception):
            raise spec
        status, payload = spec
        return SimpleNamespace(status_code=status, json=lambda: payload)


def _cleanup_store(tmp_path, cids):
    from weather_edge.persistence import PersistentStore

    store = PersistentStore(tmp_path / "c.db")
    for cid in cids:
        seed_position(store, f"tok-{cid}", cid, city="nyc", outcome="YES", shares=10,
                      avg_price=0.5, cost_basis=5.0)
    return store


def _held(store):
    return sorted(r[0] for r in store.conn.execute(
        "SELECT condition_id FROM positions WHERE total_shares > 0"))


async def _run_cleanup(monkeypatch, tmp_path, pages, cids):
    import httpx

    calls: list = []
    monkeypatch.setattr(httpx, "AsyncClient", _PagedClient(pages, calls))
    store = _cleanup_store(tmp_path, cids)
    ex = SimpleNamespace(wallet_address="0xW")
    try:
        await scheduler._clean_resolved_positions(store, ex)
    except Exception:  # noqa: BLE001 - the caller catches; we only care about the DB
        pass
    return store, calls


def _page(prefix, n):
    return [{"conditionId": f"{prefix}{i}", "size": 1} for i in range(n)]


async def test_cleanup_paginates_and_keeps_positions_on_later_pages(monkeypatch, tmp_path):
    limit = polymarket.DATA_API_POSITIONS_PAGE_LIMIT
    pages = [(200, _page("a", limit)), (200, _page("b", 3))]
    store, calls = await _run_cleanup(
        monkeypatch, tmp_path, pages, ["a0", "b1", "resolved"],
    )
    assert [c["offset"] for c in calls] == [0, limit]
    assert all(c["limit"] == limit for c in calls)
    assert _held(store) == ["a0", "b1"]


@pytest.mark.parametrize("failure", ["status", "raise", "not_list"])
async def test_cleanup_never_zeroes_when_a_page_fails(monkeypatch, tmp_path, failure):
    limit = polymarket.DATA_API_POSITIONS_PAGE_LIMIT
    bad = {"status": (500, []), "raise": RuntimeError("down"),
           "not_list": (200, {"error": "x"})}[failure]
    pages = [(200, _page("a", limit)), bad]
    store, _ = await _run_cleanup(monkeypatch, tmp_path, pages, ["a0", "b1", "resolved"])
    assert _held(store) == ["a0", "b1", "resolved"]


async def test_cleanup_never_zeroes_when_every_page_is_full(monkeypatch, tmp_path):
    limit = polymarket.DATA_API_POSITIONS_PAGE_LIMIT
    pages = [(200, _page(f"p{n}-", limit))
             for n in range(polymarket.DATA_API_POSITIONS_MAX_PAGES)]
    store, calls = await _run_cleanup(monkeypatch, tmp_path, pages, ["p0-1", "resolved"])
    assert len(calls) == polymarket.DATA_API_POSITIONS_MAX_PAGES
    assert _held(store) == ["p0-1", "resolved"]


async def test_cleanup_zeroes_missing_positions_on_complete_response(monkeypatch, tmp_path):
    store, calls = await _run_cleanup(
        monkeypatch, tmp_path, [(200, [{"conditionId": "keep", "size": 2},
                                       {"conditionId": "dust", "size": 0}])],
        ["keep", "dust", "resolved"],
    )
    assert len(calls) == 1
    assert _held(store) == ["keep"]


# ---------------------------------------------------------------------------
# trading_today: the earliest city-local date, independent of the host zone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("now", "expected"), [
    # 23:30 UTC: Asia/Pacific are on the 26th, the Americas still on the 25th
    (datetime(2026, 9, 25, 23, 30, tzinfo=UTC), date(2026, 9, 25)),
    # 06:00 UTC: Los Angeles (UTC-7) is still on the 25th
    (datetime(2026, 9, 26, 6, 0, tzinfo=UTC), date(2026, 9, 25)),
    # 08:00 UTC: every tracked city has reached the 26th
    (datetime(2026, 9, 26, 8, 0, tzinfo=UTC), date(2026, 9, 26)),
])
def test_trading_today_is_earliest_city_local_date(now, expected):
    assert scheduler.trading_today(now) == expected


def test_default_window_starts_at_trading_today():
    today = scheduler.trading_today()
    assert scheduler.default_target_dates(days=2) == [today, today + timedelta(days=1)]
