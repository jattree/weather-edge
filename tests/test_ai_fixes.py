"""Regression tests for the AI review gate, fail-closed parsing, retry and caches.

No network: every HTTP call is served by httpx.MockTransport or a monkeypatch.
"""
from __future__ import annotations

import ast
import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from weather_edge import scheduler
from weather_edge.analysis import claude_reasoning, gemini_reasoning
from weather_edge.analysis.edge import Signal
from weather_edge.models.enums import City, SignalTier, TradeSide

SRC = Path(__file__).resolve().parents[1] / "src" / "weather_edge"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_signal(market_id="m1", city="nyc", target=None, size=20.0, edge=0.10,
                side=TradeSide.YES, tier=SignalTier.HIGH) -> Signal:
    target = target or (date.today() + timedelta(days=2))
    return Signal(
        market_id=market_id, consensus_id=None, computed_at=datetime.now(UTC),
        model_prob=0.30, market_prob=0.20, model_confidence=0.9,
        edge=edge, net_edge=edge - 0.01, edge_pct=0.5, kelly_fraction=0.1, half_kelly=0.05,
        recommended_side=side, recommended_size=size, confidence_tier=tier,
        city_id=city, target_date=str(target), description=f"{city} {market_id} question",
        strategy="core",
    )


def fake_forecasts(temps=(20.0, 20.5, 21.0, 19.5, 20.2), fetched_at=None):
    fetched_at = fetched_at or datetime.now(UTC)
    return [
        SimpleNamespace(model_name=f"model{i}", temp_max_c=t, fetched_at=fetched_at)
        for i, t in enumerate(temps)
    ]


def reasoning(signal, should_trade=True, adj=1.0, ok=True):
    return claude_reasoning.TradeReasoning(
        signal=signal, should_trade=should_trade, confidence_adjustment=adj,
        rationale="because weather", risk_factors=[], weather_insight="", review_ok=ok,
    )


def gemini_ok(dissent=0.0, sizing="full", verdict="AGREE"):
    return {
        "dissent_strength": dissent, "verdict": verdict, "primary_risk": "",
        "counter_arguments": [], "risk_the_bull_missed": "",
        "sizing_recommendation": sizing, "model": "gemini",
    }


class SettingsProxy:
    """Delegates to the real settings but accepts settings not yet in config.py."""

    def __init__(self, real):
        object.__setattr__(self, "_real", real)

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.fixture(autouse=True)
def isolated_ai_state(monkeypatch):
    """Fresh, non-persistent review memory and default (non hail-mary) settings."""
    from weather_edge import config, live_state

    proxy = SettingsProxy(config.settings)
    monkeypatch.setattr(scheduler, "settings", proxy)
    monkeypatch.setattr(config, "settings", proxy)
    monkeypatch.setattr(live_state, "_get_redis", lambda: None)  # never touch real Redis
    monkeypatch.setattr(
        claude_reasoning, "_review_memory", claude_reasoning.AIReviewMemory(persist=False),
    )
    monkeypatch.setattr(scheduler.settings, "hail_mary_mode", False)
    monkeypatch.setattr(scheduler, "ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setattr(scheduler, "record_decision", lambda r: None)
    claude_reasoning.clear_decisions()
    yield
    claude_reasoning.clear_decisions()


def cache_for(signal, forecasts=None):
    return {
        (City(signal.city_id), date.fromisoformat(signal.target_date)):
            forecasts if forecasts is not None else fake_forecasts(),
    }


def patch_reviews(monkeypatch, claude=None, gemini=None):
    """claude/gemini: callables (signal) -> result. Returns call log."""
    calls = {"claude": [], "gemini": []}

    async def _analyze(sig, model_vals, mean, std, variable="temp_max_c"):
        calls["claude"].append(sig.market_id)
        return claude(sig) if claude else reasoning(sig)

    async def _red_team(sig, model_vals, mean, std, claude_rationale=""):
        calls["gemini"].append(sig.market_id)
        return gemini(sig) if gemini else None

    monkeypatch.setattr(scheduler, "analyze_trade", _analyze)
    monkeypatch.setattr(gemini_reasoning, "red_team_trade", _red_team)
    return calls


# ---------------------------------------------------------------------------
# Finding 1: Claude SKIP must remove the signal from execution
# ---------------------------------------------------------------------------

async def test_claude_skip_removes_signal_from_execution(monkeypatch):
    keep, skip = make_signal("keep"), make_signal("skip", city="lon")
    patch_reviews(monkeypatch, claude=lambda s: reasoning(s, should_trade=s.market_id == "keep"))
    cache = {**cache_for(keep), **cache_for(skip)}

    out = await scheduler.apply_ai_review([keep, skip], cache, True, store=None)

    assert [s.market_id for s in out] == ["keep"]


async def test_hail_mary_keeps_log_only_claude_skip(monkeypatch):
    monkeypatch.setattr(scheduler.settings, "hail_mary_mode", True)
    sig = make_signal(size=1.0)
    patch_reviews(monkeypatch, claude=lambda s: reasoning(s, should_trade=False))

    out = await scheduler.apply_ai_review([sig], cache_for(sig), True, store=None)

    assert out == [sig]
    assert claude_reasoning.get_review_memory().veto_for(sig) is None


async def test_hail_mary_trades_without_any_review(monkeypatch):
    monkeypatch.setattr(scheduler.settings, "hail_mary_mode", True)
    sig = make_signal()
    out = await scheduler.apply_ai_review([sig], {}, False, store=None)
    assert out == [sig]


# ---------------------------------------------------------------------------
# Finding 2: Gemini skip -> size 0 -> dropped, never bumped to MIN_LIVE_SIZE
# ---------------------------------------------------------------------------

async def test_gemini_skip_drops_signal_and_is_remembered(monkeypatch):
    sig = make_signal()
    patch_reviews(monkeypatch, gemini=lambda s: gemini_ok(0.95, "skip", "DISSENT"))

    out = await scheduler.apply_ai_review([sig], cache_for(sig), True, store=None)

    assert out == []
    assert claude_reasoning.get_review_memory().veto_for(sig).source == "gemini"


async def test_gemini_half_cuts_size(monkeypatch):
    sig = make_signal(size=20.0)
    patch_reviews(monkeypatch, claude=lambda s: reasoning(s, adj=1.0),
                  gemini=lambda s: gemini_ok(0.5, "half", "DISSENT"))

    out = await scheduler.apply_ai_review([sig], cache_for(sig), True, store=None)

    assert out == [sig] and sig.recommended_size == pytest.approx(10.0)


def test_live_min_size_bump_never_applies_to_zero_size():
    src = (SRC / "scheduler.py").read_text()
    bump = src.index("signal.recommended_size = max(signal.recommended_size, MIN_LIVE_SIZE)")
    guard = src.rfind("if signal.recommended_size <= 0:", 0, bump)
    assert guard != -1 and bump - guard < 300, "zero-size guard must precede the $5 bump"


# ---------------------------------------------------------------------------
# Finding 3: Claude parsing fails closed; None review is a veto
# ---------------------------------------------------------------------------

def _resp(text, stop_reason="end_turn"):
    return {"content": [{"type": "text", "text": text}], "stop_reason": stop_reason}


@pytest.mark.parametrize("data", [
    _resp("I think you should trade, it looks fine"),               # no JSON
    _resp('{"should_trade": "false", "confidence_adjustment": 1}'.replace('"false"', '"nope"')),
    _resp('{"should_trade": 1}'),                                    # non-boolean
    _resp('{"should_trade": true, "rationale": "trunc', "max_tokens"),
    _resp('{"should_trade": true}', "refusal"),
    {"content": [], "stop_reason": "end_turn"},
    {"content": [{"type": "thinking", "thinking": ""}], "stop_reason": "end_turn"},
])
def test_unparseable_or_truncated_reply_fails_closed(data):
    r = claude_reasoning.parse_claude_response(make_signal(), data)
    assert r.should_trade is False and r.review_ok is False


def test_string_false_is_not_truthy():
    r = claude_reasoning.parse_claude_response(
        make_signal(), _resp('```json\n{"should_trade": "false", "rationale": "front"}\n```'),
    )
    assert r.should_trade is False and r.review_ok is True


def test_valid_reply_after_thinking_block_parses():
    data = {"content": [{"type": "thinking", "thinking": ""},
                        {"type": "text", "text": '{"should_trade": true, '
                                                 '"confidence_adjustment": 9}'}],
            "stop_reason": "end_turn"}
    r = claude_reasoning.parse_claude_response(make_signal(), data)
    assert r.should_trade is True and r.confidence_adjustment == 1.5


def _install_transport(monkeypatch, module, handler):
    real = httpx.AsyncClient

    def factory(*a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        return real(*a, **kw)

    monkeypatch.setattr(module.httpx, "AsyncClient", factory)


async def test_analyze_trade_uses_configured_model_and_ample_max_tokens(monkeypatch):
    seen = {}

    def handler(request):
        import json
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_resp('{"should_trade": false, "rationale": "fog"}'))

    monkeypatch.setattr(claude_reasoning, "ANTHROPIC_API_KEY", "k")
    _install_transport(monkeypatch, claude_reasoning, handler)
    r = await claude_reasoning.analyze_trade(make_signal(), {"a": 20.0, "b": 21.0}, 20.5, 0.5)

    assert r.should_trade is False
    assert seen["body"]["model"] == claude_reasoning.CLAUDE_MODEL
    assert seen["body"]["max_tokens"] >= 1024


async def test_analyze_trade_api_failure_returns_none_and_scheduler_vetoes(monkeypatch):
    monkeypatch.setattr(claude_reasoning, "ANTHROPIC_API_KEY", "k")
    _install_transport(monkeypatch, claude_reasoning,
                       lambda req: httpx.Response(400, json={"error": "bad"}))
    sig = make_signal()
    assert await claude_reasoning.analyze_trade(sig, {"a": 20.0}, 20.0, 0.5) is None

    patch_reviews(monkeypatch, claude=lambda s: None)
    out = await scheduler.apply_ai_review([sig], cache_for(sig), True, store=None)
    assert out == []
    # A failed review is not a meteorological veto: not remembered.
    assert claude_reasoning.get_review_memory().veto_for(sig) is None


async def test_parse_failure_blocks_but_is_not_remembered(monkeypatch):
    sig = make_signal()
    patch_reviews(monkeypatch, claude=lambda s: reasoning(s, should_trade=False, ok=False))
    assert await scheduler.apply_ai_review([sig], cache_for(sig), True, store=None) == []
    assert claude_reasoning.get_review_memory().veto_for(sig) is None


@pytest.mark.parametrize("payload", [
    '{"dissent_strength": 0.1, "verdict": "AGREE", "sizing_recommendation": "all-in"}',
    '{"dissent_strength": "high", "verdict": "AGREE", "sizing_recommendation": "full"}',
    '{"dissent_strength": 3, "verdict": "AGREE", "sizing_recommendation": "full"}',
    '{"dissent_strength": 0.1, "verdict": "MAYBE", "sizing_recommendation": "full"}',
    "no json here",
])
def test_gemini_parse_rejects_invalid_values(payload):
    data = {"candidates": [{"content": {"parts": [{"text": payload}]}}]}
    assert isinstance(gemini_reasoning.parse_gemini_response(data), str)


async def test_gemini_failure_fails_closed_and_scheduler_drops(monkeypatch):
    monkeypatch.setattr(gemini_reasoning, "GEMINI_API_KEY", "g-key-123456")
    _install_transport(monkeypatch, gemini_reasoning, lambda req: httpx.Response(403))
    sig = make_signal()
    result = await gemini_reasoning.red_team_trade(sig, {"a": 20.0}, 20.0, 0.5, "ok")
    assert result["error"] and result["sizing_recommendation"] == "skip"

    patch_reviews(monkeypatch, gemini=lambda s: result)
    assert await scheduler.apply_ai_review([sig], cache_for(sig), True, store=None) == []
    assert claude_reasoning.get_review_memory().veto_for(sig) is None


async def test_gemini_crash_in_scheduler_fails_closed(monkeypatch):
    sig = make_signal()

    def boom(s):
        raise RuntimeError("kaboom")

    patch_reviews(monkeypatch, gemini=boom)
    assert await scheduler.apply_ai_review([sig], cache_for(sig), True, store=None) == []


# ---------------------------------------------------------------------------
# Finding 4: vetoes/approvals remembered; no review -> no trade by default
# ---------------------------------------------------------------------------

async def test_veto_is_applied_on_later_no_ai_cycle(monkeypatch):
    first = make_signal()
    patch_reviews(monkeypatch, claude=lambda s: reasoning(s, should_trade=False))
    assert await scheduler.apply_ai_review([first], cache_for(first), True) == []

    # Sniper cycle a few minutes later: fresh Signal object, AI skipped.
    later = make_signal()
    monkeypatch.setattr(scheduler.settings, "require_ai_review", False, raising=False)
    assert await scheduler.apply_ai_review([later], cache_for(later), False) == []


async def test_remembered_veto_skips_paid_review(monkeypatch):
    sig = make_signal()
    claude_reasoning.get_review_memory().record_veto(sig, "sea breeze", "claude")
    calls = patch_reviews(monkeypatch)
    assert await scheduler.apply_ai_review([make_signal()], cache_for(sig), True) == []
    assert calls["claude"] == []


async def test_approval_carries_over_with_size_multiplier(monkeypatch):
    first = make_signal(size=20.0)
    patch_reviews(monkeypatch, claude=lambda s: reasoning(s, adj=0.5),
                  gemini=lambda s: gemini_ok(0.3, "reduce_20pct"))
    assert await scheduler.apply_ai_review([first], cache_for(first), True) == [first]
    assert first.recommended_size == pytest.approx(8.0)

    later = make_signal(size=20.0)
    out = await scheduler.apply_ai_review([later], cache_for(later), False)
    assert out == [later] and later.recommended_size == pytest.approx(8.0)

    # Approval is side specific
    other_side = make_signal(side=TradeSide.NO)
    assert await scheduler.apply_ai_review([other_side], {}, False) == []


async def test_no_review_means_no_trade_by_default(monkeypatch):
    sig = make_signal()
    monkeypatch.setattr(scheduler, "ANTHROPIC_API_KEY", "")
    assert await scheduler.apply_ai_review([sig], cache_for(sig), True) == []
    assert await scheduler.apply_ai_review([sig], cache_for(sig), False) == []


async def test_require_ai_review_false_allows_unreviewed(monkeypatch):
    sig = make_signal()
    monkeypatch.setattr(scheduler.settings, "require_ai_review", False, raising=False)
    monkeypatch.setattr(scheduler, "ANTHROPIC_API_KEY", "")
    assert await scheduler.apply_ai_review([sig], {}, False) == [sig]


async def test_review_budget_prioritises_unreviewed_signals(monkeypatch):
    monkeypatch.setattr(scheduler.settings, "max_ai_reviews_per_cycle", 1, raising=False)
    a, b = make_signal("a", edge=0.3), make_signal("b", city="lon", edge=0.1)
    cache = {**cache_for(a), **cache_for(b)}
    calls = patch_reviews(monkeypatch)
    await scheduler.apply_ai_review([a, b], cache, True)
    a2, b2 = make_signal("a", edge=0.3), make_signal("b", city="lon", edge=0.1)
    out = await scheduler.apply_ai_review([a2, b2], cache, True)
    assert calls["claude"] == ["a", "b"]
    assert {s.market_id for s in out} == {"a", "b"}


def test_memory_evicts_past_dates():
    mem = claude_reasoning.AIReviewMemory(persist=False)
    old = make_signal(target=date.today() - timedelta(days=1))
    mem.record_veto(old, "x", "claude")
    mem.record_approval(make_signal("n", target=date.today() - timedelta(days=2)), 1.0)
    assert mem.evict_before(date.today()) == 2
    assert mem.veto_for(old) is None


# ---------------------------------------------------------------------------
# Finding 5: Gemini key in header, never URL; retry logs redact secrets
# ---------------------------------------------------------------------------

async def test_gemini_key_sent_in_header_not_url(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["header"] = request.headers.get("x-goog-api-key")
        text = ('{"dissent_strength": 0.2, "verdict": "AGREE", "primary_risk": "thin book",'
                ' "sizing_recommendation": "full"}')
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": text}]}}]})

    monkeypatch.setattr(gemini_reasoning, "GEMINI_API_KEY", "secret-gemini-key")
    _install_transport(monkeypatch, gemini_reasoning, handler)
    result = await gemini_reasoning.red_team_trade(make_signal(), {"a": 20.0}, 20.0, 0.5, "ok")

    assert "secret-gemini-key" not in seen["url"] and "key=" not in seen["url"]
    assert seen["header"] == "secret-gemini-key"
    # Finding 9: prompt asks for primary_risk and that is what is returned
    assert result["primary_risk"] == "thin book"
    assert "primary_failure_mode" not in result


async def test_retry_logs_redact_query_string_secrets(monkeypatch, caplog):
    from weather_edge import retry

    monkeypatch.setenv("SOME_API_KEY", "supersecretvalue123")
    url = "https://example.test/v1/thing?key=supersecretvalue123&x=1"

    async def fail():
        req = httpx.Request("GET", url)
        raise httpx.HTTPStatusError("503 for url " + url, request=req,
                                    response=httpx.Response(503, request=req))

    caplog.set_level(logging.WARNING, logger="weather_edge.retry")
    with pytest.raises(httpx.HTTPStatusError):
        await retry.retry_async(fail, attempts=2, base_delay=0.0, label="t")
    assert caplog.records
    assert "supersecretvalue123" not in caplog.text
    assert "<redacted>" in caplog.text


def test_redact_known_secret_outside_url(monkeypatch):
    from weather_edge.retry import redact

    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTESTTESTTEST")
    assert "AIzaSyTESTTESTTEST" not in redact("header x-goog-api-key: AIzaSyTESTTESTTEST")


# ---------------------------------------------------------------------------
# Finding 10: only transient errors are retried
# ---------------------------------------------------------------------------

def _status_error(code):
    req = httpx.Request("GET", "https://example.test/")
    return httpx.HTTPStatusError(str(code), request=req, response=httpx.Response(code, request=req))


@pytest.mark.parametrize("exc,expected_calls", [
    (_status_error(400), 1),
    (_status_error(401), 1),
    (_status_error(404), 1),
    (KeyError("content"), 1),
    (ValueError("bad"), 1),
    (_status_error(429), 3),
    (_status_error(503), 3),
    (httpx.ConnectError("refused"), 3),
    (httpx.ReadTimeout("slow"), 3),
    (TimeoutError(), 3),
])
async def test_retry_only_transient(exc, expected_calls):
    from weather_edge.retry import retry_async

    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise exc

    with pytest.raises(type(exc)):
        await retry_async(fn, attempts=3, base_delay=0.0)
    assert calls == expected_calls


def test_retry_sync_retries_sqlite_lock_only():
    import sqlite3

    from weather_edge.retry import retry_sync

    for exc, expected in [(sqlite3.OperationalError("database is locked"), 3),
                          (sqlite3.OperationalError("no such table: x"), 1)]:
        calls = 0

        def fn(e=exc):
            nonlocal calls
            calls += 1
            raise e

        with pytest.raises(sqlite3.OperationalError):
            retry_sync(fn, attempts=3, base_delay=0.0)
        assert calls == expected


# ---------------------------------------------------------------------------
# Finding 6: forecast context keyed by (city, date); eviction; GraphCast per date
# ---------------------------------------------------------------------------

def test_model_context_uses_only_matching_date():
    sig = make_signal(target=date.today() + timedelta(days=3))
    other_day = {(City.NYC, date.today() + timedelta(days=2)): fake_forecasts((30.0, 31.0))}
    assert scheduler.model_context_for_signal(sig, other_day) == ({}, 0.0, 0.0)

    both = {**other_day, **cache_for(sig, fake_forecasts((10.0, 12.0)))}
    vals, mean, std = scheduler.model_context_for_signal(sig, both)
    assert mean == pytest.approx(11.0) and std == pytest.approx(1.0)


async def test_no_forecast_context_means_no_review(monkeypatch):
    sig = make_signal()
    calls = patch_reviews(monkeypatch)
    assert await scheduler.apply_ai_review([sig], {}, True) == []
    assert calls["claude"] == []


def test_evict_stale_forecasts():
    today = date.today()
    cache = {(City.NYC, today - timedelta(days=1)): [1], (City.NYC, today): [2]}
    assert scheduler.evict_stale_forecasts(cache, today) == 1
    assert list(cache) == [(City.NYC, today)]


def test_filter_dates_by_horizon_late_utc_day():
    now = datetime(2026, 9, 25, 13, 0, tzinfo=UTC)
    d0 = date(2026, 9, 25)
    kept, blocked, h = scheduler.filter_dates_by_horizon(
        [d0, d0 + timedelta(days=1), d0 + timedelta(days=2)], now, 36,
    )
    assert h == 36 and blocked == [d0, d0 + timedelta(days=1)]
    assert kept == [d0 + timedelta(days=2)]


# ---------------------------------------------------------------------------
# End-to-end run_cycle with every network call mocked
# ---------------------------------------------------------------------------

@pytest.fixture
def offline_cycle(monkeypatch):
    """Patch run_cycle's external dependencies. Returns a state dict."""
    from weather_edge.analysis import enso_regime, exit_monitor
    from weather_edge.fetchers import gribstream, polymarket

    d2, d3 = date.today() + timedelta(days=2), date.today() + timedelta(days=3)
    state = {"ai_dates": [], "fetch_ok": True, "markets": [], "dates": (d2, d3)}

    def market(mid, city, target):
        return polymarket.MarketInfo(
            market_id=mid, city_id=city, target_date=target, question=f"{mid}?",
            threshold_value=20.0, threshold_dir="gte", yes_price=0.20, no_price=0.80,
        )

    state["markets"] = [market("nyc-d2", City.NYC, d2), market("lon-d3", City.LON, d3)]

    async def discover():
        return state["markets"]

    async def fetch_city(city_id, target_date):
        return fake_forecasts() if state["fetch_ok"] else []

    async def ai_batch(cities, target_date):
        state["ai_dates"].append(target_date)
        return {}

    async def noop(*a, **k):
        return None

    def consensus(city_id, target_date, variable, forecasts, thresholds=None):
        return SimpleNamespace(std_dev=0.5, weighted_mean=21.0, mean_value=21.0,
                               confidence=0.9, model_count=len(forecasts))

    def edge(**kw):
        s = make_signal(kw["market_id"], city=kw["city_id"],
                        target=date.fromisoformat(kw["target_date"]))
        s.hours_to_resolution = kw["hours_to_resolution"]
        return s

    monkeypatch.setattr(scheduler, "discover_weather_markets", discover)
    monkeypatch.setattr(scheduler, "fetch_city_forecasts", fetch_city)
    monkeypatch.setattr(scheduler, "compute_consensus", consensus)
    monkeypatch.setattr(scheduler, "compute_model_prob_for_market", lambda m, c: 0.35)
    monkeypatch.setattr(scheduler, "calculate_edge", edge)
    monkeypatch.setattr(scheduler, "is_golden_window", lambda: False)
    monkeypatch.setattr(scheduler, "detect_patterns", lambda c, f: [])
    monkeypatch.setattr(scheduler, "get_pattern_adjustment", lambda c, a: (1.0, 0.0))
    monkeypatch.setattr(gribstream, "fetch_ai_forecasts_batch", ai_batch)
    monkeypatch.setattr(enso_regime, "fetch_enso_state", noop)
    monkeypatch.setattr(polymarket, "fetch_book_prices", noop)
    monkeypatch.setattr(exit_monitor, "scan_for_exits", lambda *a, **k: [])
    return state


async def test_run_cycle_executes_only_ai_cleared_signals(monkeypatch, offline_cycle):
    from weather_edge.trading.paper import PaperTrader

    patch_reviews(monkeypatch, claude=lambda s: reasoning(s, should_trade=s.city_id == "nyc"))
    trader = PaperTrader(bankroll=1000.0)
    placed = []
    real = trader.place_trade
    monkeypatch.setattr(trader, "place_trade", lambda s: placed.append(s.market_id) or real(s))

    signals, cache, _ = await scheduler.run_cycle(trader, list(offline_cycle["dates"]))

    assert {s.market_id for s in signals} == {"nyc-d2", "lon-d3"}
    assert placed == ["nyc-d2"]
    # GraphCast fetched separately for each target date (not dates[0] for all)
    assert sorted(offline_cycle["ai_dates"]) == sorted(offline_cycle["dates"])


async def test_run_cycle_zero_size_signal_not_traded(monkeypatch, offline_cycle):
    from weather_edge.trading.paper import PaperTrader

    patch_reviews(monkeypatch, claude=lambda s: reasoning(s, adj=1.0))
    monkeypatch.setattr(scheduler.settings, "require_ai_review", False, raising=False)
    orig_edge = scheduler.calculate_edge

    def zero(**kw):
        s = orig_edge(**kw)
        s.recommended_size = 0.0
        return s

    monkeypatch.setattr(scheduler, "calculate_edge", zero)
    monkeypatch.setattr(scheduler, "ANTHROPIC_API_KEY", "")
    trader = PaperTrader(bankroll=1000.0)
    placed = []
    monkeypatch.setattr(trader, "place_trade", lambda s: placed.append(s) or None)
    await scheduler.run_cycle(trader, list(offline_cycle["dates"]))
    assert placed == []


async def test_run_cycle_stale_fallback_uses_passed_cache(monkeypatch, offline_cycle, caplog):
    patch_reviews(monkeypatch)
    cache: dict = {}
    await scheduler.run_cycle(None, list(offline_cycle["dates"]), forecast_cache=cache)
    assert (City.NYC, offline_cycle["dates"][0]) in cache

    offline_cycle["fetch_ok"] = False
    caplog.set_level(logging.WARNING, logger="weather_edge.scheduler")
    signals, _, _ = await scheduler.run_cycle(
        None, list(offline_cycle["dates"]), forecast_cache=cache,
    )
    assert "STALE DATA" in caplog.text
    assert {s.market_id for s in signals} == {"nyc-d2", "lon-d3"}


async def test_run_cycle_rejects_too_old_stale_cache(monkeypatch, offline_cycle, caplog):
    patch_reviews(monkeypatch)
    offline_cycle["fetch_ok"] = False
    old = datetime.now(UTC) - timedelta(hours=30)
    cache = {(City.NYC, d): fake_forecasts(fetched_at=old) for d in offline_cycle["dates"]}
    caplog.set_level(logging.WARNING, logger="weather_edge.scheduler")
    signals, _, _ = await scheduler.run_cycle(
        None, list(offline_cycle["dates"]), forecast_cache=cache,
    )
    assert "STALE DATA REJECTED" in caplog.text
    assert signals == []


async def test_run_cycle_accepts_plain_paper_trader_without_store(monkeypatch, offline_cycle):
    from weather_edge.trading.paper import PaperTrader

    patch_reviews(monkeypatch)
    trader = PaperTrader(bankroll=1000.0)
    assert not hasattr(trader, "store")
    signals, _, _ = await scheduler.run_cycle(trader, list(offline_cycle["dates"]))
    assert signals


async def test_run_loop_honours_days_and_stops(monkeypatch):
    seen = []

    async def fake_cycle(trader, target_dates=None, forecast_cache=None, **kw):
        seen.append(target_dates)
        return [], forecast_cache, {}

    monkeypatch.setattr(scheduler, "run_cycle", fake_cycle)
    await scheduler.run_loop(None, days=3, max_cycles=1)
    assert seen == [scheduler.default_target_dates(scheduler.trading_today(), 3)]


# ---------------------------------------------------------------------------
# Findings 8, 9, 11
# ---------------------------------------------------------------------------

def test_dashboard_passes_forecast_cache_to_run_cycle():
    tree = ast.parse((SRC / "dashboard" / "app.py").read_text())
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "run_cycle"
    ]
    assert calls and all("forecast_cache" in {k.arg for k in c.keywords} for c in calls)


def test_claude_model_default_and_override(monkeypatch):
    from weather_edge.config import settings

    monkeypatch.delenv("CLAUDE_MODEL", raising=False)
    monkeypatch.setattr(settings, "claude_model", "", raising=False)
    assert claude_reasoning._get_model() == "claude-sonnet-5"
    monkeypatch.setattr(settings, "claude_model", "claude-opus-5", raising=False)
    assert claude_reasoning._get_model() == "claude-opus-5"


def test_service_health_does_not_recurse_when_redis_down(monkeypatch):
    """_get_redis reports its failure via record_service_call; must not re-enter Redis."""
    from weather_edge import live_state
    from weather_edge.analysis import service_health

    attempts = 0

    def failing_redis():
        nonlocal attempts
        attempts += 1
        if attempts > 5:
            raise AssertionError("recursed into Redis connect")
        service_health.record_service_call("redis", False)
        return None

    monkeypatch.setattr(live_state, "_get_redis", failing_redis)
    service_health.record_service_call("claude", True)
    assert attempts <= 2
    assert service_health._health_store["redis"]["last_status"] == "error"


def test_batch_analyze_signals_removed():
    assert not hasattr(claude_reasoning, "batch_analyze_signals")
