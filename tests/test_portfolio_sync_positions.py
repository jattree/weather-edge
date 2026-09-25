"""portfolio_sync must never trust a partial Data API position list."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from weather_edge import live_state
from weather_edge.fetchers import polymarket
from weather_edge.trading import portfolio_sync


class _Resp:
    def __init__(self, status: int, body):
        self.status_code = status
        self._body = body

    def json(self):
        return self._body


class _Client:
    """httpx.AsyncClient stand-in serving /positions pages from a list."""

    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        return self.pages.pop(0) if self.pages else _Resp(200, [])


def _executor():
    async def check_balance():
        return 100.0

    return SimpleNamespace(
        check_balance=check_balance,
        _client=SimpleNamespace(get_orders=lambda: []),
    )


def _state_with(monkeypatch, pages):
    client = _Client(pages)
    import httpx
    monkeypatch.setattr(live_state, "_get_redis", lambda: None)  # never touch real Redis
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: client)
    return asyncio.run(portfolio_sync.fetch_polymarket_state(_executor(), "0xABC")), client


def test_complete_multi_page_list_is_used(monkeypatch):
    limit = polymarket.DATA_API_POSITIONS_PAGE_LIMIT
    full = [{"conditionId": f"c{i}", "size": 1, "currentValue": 0} for i in range(limit)]
    state, client = _state_with(monkeypatch, [_Resp(200, full), _Resp(200, full[:3])])
    assert len(state["positions"]) == limit + 3
    assert [c["offset"] for c in client.calls if "offset" in c] == [0, limit]


def test_failed_page_means_nav_not_observed(monkeypatch):
    limit = polymarket.DATA_API_POSITIONS_PAGE_LIMIT
    full = [{"conditionId": f"c{i}", "size": 1, "currentValue": 0} for i in range(limit)]
    state, _ = _state_with(monkeypatch, [_Resp(200, full), _Resp(503, None)])
    assert state["positions"] == []
    assert state["nav_observed"] is False
