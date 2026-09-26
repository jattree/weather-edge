"""Exit monitor: Gemini key never in the URL; Hong Kong bucket rule."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import httpx

from weather_edge.analysis import claude_reasoning, exit_monitor, gemini_reasoning
from weather_edge.analysis.exit_monitor import ExitCandidate
from weather_edge.analysis.resolver import parse_bucket_from_description
from weather_edge.models.position import Position


class _Resp:
    status_code = 200

    def json(self):
        return {"candidates": [{"content": {"parts": [
            {"text": '{"verdict": "HOLD", "rationale": "thin book"}'}]}}]}


def test_gemini_exit_review_sends_key_in_header(monkeypatch):
    calls = []

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, **kw):
            calls.append((url, kw.get("headers", {})))
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: Client())
    monkeypatch.setattr(gemini_reasoning, "GEMINI_API_KEY", "g-secret-123456")
    monkeypatch.setattr(claude_reasoning, "ANTHROPIC_API_KEY", "")
    trade = Position(market_id="m1", city_id="nyc", side="YES", size_usd=10.0,
                     entry_price=0.4, total_shares=25.0, description="NYC 21C?")
    cand = ExitCandidate(trade=trade, reason="edge_inversion", current_model_prob=0.3,
                         current_market_price=0.45, original_edge=0.1,
                         current_edge=-0.05, urgency="medium")
    asyncio.run(exit_monitor.ai_review_exit(cand, {}, 21.0, 1.0))
    assert calls, "Gemini exit review was not called"
    url, headers = calls[0]
    assert "g-secret" not in url and "key=" not in url
    assert headers.get("x-goog-api-key") == "g-secret-123456"
    assert cand.gemini_verdict == "HOLD"


def test_hong_kong_guard_uses_integer_part_rule():
    # HKO 28.7 C is the "28" bucket in Hong Kong, not "29" (which a normal
    # round-half-up would give).
    bucket = parse_bucket_from_description(
        "Will the highest temperature in Hong Kong be 29°C on September 26?")
    assert bucket is not None
    resp = SimpleNamespace(status_code=200,
                           json=lambda: {"daily": {"temperature_2m_max": [28.7]}})
    hkg = Position(city_id="hkg", side="YES")
    nyc = Position(city_id="nyc", side="YES")
    assert exit_monitor._observation_in_bucket(hkg, bucket, resp) is False
    assert exit_monitor._observation_in_bucket(nyc, bucket, resp) is True


def test_dashboard_quiets_httpx_url_logging():
    import pytest
    pytest.importorskip("fastapi")  # dashboard extra
    import weather_edge.dashboard.app  # noqa: F401 - import configures logging
    assert logging.getLogger("httpx").getEffectiveLevel() >= logging.WARNING
