"""Characterization tests for scheduler.run_cycle.

Each scenario drives one full cycle with every network / external call
replaced by a deterministic fake, then records everything observable:

- the returned signals, forecast_cache keys and city_volume,
- paper trades (sizes, entries, fees, exits),
- every store / live-executor / paper-trader / AI call, in order,
- final live_trades / positions / ai_decisions rows,
- the AI decision history, and
- every log line emitted by ``weather_edge.*`` loggers.

The snapshot is compared to a golden JSON file under
``tests/golden/run_cycle/``. Dates are normalised to ``<D0>`` (today) ..
``<D4>``, clock times to ``<HMS>`` and the temp dir to ``<TMP>`` so the
files do not go stale. Regenerate after an INTENDED
behaviour change with::

    UPDATE_RUN_CYCLE_GOLDEN=1 pytest tests/test_run_cycle_characterization.py

Each test also asserts a few key facts inline so a reader can see what the
scenario is about without opening the JSON.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from weather_edge import scheduler
from weather_edge.analysis import claude_reasoning, model_timing
from weather_edge.analysis import edge as edge_module
from weather_edge.analysis.edge import Signal
from weather_edge.models.enums import City, SignalTier, TradeSide

GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "run_cycle"
UPDATE = os.environ.get("UPDATE_RUN_CYCLE_GOLDEN") == "1"


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class SettingsProxy:
    """Delegates to the real settings; attributes set on it shadow them."""

    def __init__(self, real):
        object.__setattr__(self, "_real", real)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        items = [_jsonable(v) for v in value]
        return sorted(items, key=repr) if isinstance(value, set) else items
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "value") and not callable(value.value):  # enums
        return value.value
    return str(value)


class FakeExecutor:
    """Recording stand-in for TradeExecutor (live, never touches the CLOB)."""

    def __init__(self, events, dry_run=False, balance=500.0):
        self.events = events
        self.dry_run = dry_run
        self.wallet_address = "0xWALLET"
        self.balance = balance
        self.cancel_fail: set[str] = set()
        # (market_id, strategy) -> "ok" | status string | Exception
        self.limit_results: dict[tuple[str, str], object] = {}
        # market_id -> "ok" | status string | Exception
        self.sell_results: dict[str, object] = {}
        self._n = 0

    async def check_balance(self):
        self.events.append(["exec.check_balance"])
        if isinstance(self.balance, Exception):
            raise self.balance
        return self.balance

    async def cancel_order(self, order_id):
        self.events.append(["exec.cancel_order", order_id])
        if order_id in self.cancel_fail:
            raise RuntimeError(f"cancel boom {order_id}")
        return True

    async def place_limit_order(self, signal, token_id, improve_price_by=0.005,
                                force_taker=False):
        self.events.append(["exec.place_limit_order", {
            "market_id": signal.market_id, "token_id": token_id,
            "side": signal.recommended_side.value, "size": signal.recommended_size,
            "market_prob": signal.market_prob, "strategy": signal.strategy,
            "description": signal.description, "improve_price_by": improve_price_by,
            "force_taker": force_taker,
        }])
        spec = self.limit_results.get((signal.market_id, signal.strategy), "ok")
        if isinstance(spec, Exception):
            raise spec
        self._n += 1
        side_price = (
            signal.market_prob if signal.recommended_side == TradeSide.YES
            else 1.0 - signal.market_prob
        )
        limit = round(max(0.01, side_price - improve_price_by), 2)
        return SimpleNamespace(
            status="pending" if spec == "ok" else spec,
            size_usd=round(signal.recommended_size, 2),
            size_shares=round(signal.recommended_size / limit, 1),
            limit_price=limit,
            order_id=f"new-{self._n}",
        )

    async def place_sell_order(self, **kw):
        self.events.append(["exec.place_sell_order", kw])
        spec = self.sell_results.get(kw["market_id"], "ok")
        if isinstance(spec, Exception):
            raise spec
        self._n += 1
        return SimpleNamespace(
            status="pending" if spec == "ok" else spec, order_id=f"sell-{self._n}",
        )


def make_store(path, events):
    """A real PersistentStore on a temp file whose calls are recorded."""
    from weather_edge.persistence import PersistentStore

    store = PersistentStore(path)
    recorded = (
        "save_ai_decision", "save_forecast_snapshot", "get_portfolio_summary",
        "get_positions", "get_open_order_for_market", "cancel_live_trade",
        "get_position_for_market", "commit",
    )
    for name in recorded:
        original = getattr(store, name)

        def wrapper(*a, _name=name, _orig=original, **k):
            if _name == "save_ai_decision":
                k_rec = {key: k.get(key) for key in (
                    "source", "decision", "city_id", "market_id",
                    "confidence_adj", "dissent_strength",
                )}
                events.append(["store." + _name, [], k_rec])
            else:
                events.append(["store." + _name, _jsonable(list(a)), _jsonable(k)])
            return _orig(*a, **k)

        setattr(store, name, wrapper)

    def rebuild_positions():
        # The real one rebuilds from fills; keep the seeded positions instead.
        events.append(["store.rebuild_positions"])

    store.rebuild_positions = rebuild_positions
    return store


def seed_live_trade(store, order_id, market_id, side, limit_price, *, status="open",
                    placed_ago_min=5.0, filled=0.0, description="entry", shares=10.0):
    placed = (datetime.now(UTC) - timedelta(minutes=placed_ago_min)).isoformat()
    store.conn.execute(
        """INSERT INTO live_trades (order_id, market_id, token_id, city_id, side, status,
               limit_price, size_shares, filled_shares, size_usd, placed_at, description)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (order_id, market_id, f"{market_id}-tok", "nyc", side, status, limit_price,
         shares, filled, round(shares * limit_price, 2), placed, description),
    )
    store.conn.commit()


def seed_position(store, asset_id, condition_id, *, city, outcome, shares, avg_price,
                  cost_basis, description="held"):
    store.conn.execute(
        """INSERT INTO positions (asset_id, condition_id, city_id, side, outcome,
               total_shares, avg_price, cost_basis, description)
           VALUES (?, ?, ?, 'BUY', ?, ?, ?, ?, ?)""",
        (asset_id, condition_id, city, outcome, shares, avg_price, cost_basis, description),
    )
    store.conn.commit()


class FakeHttpxClient:
    """httpx.AsyncClient stand-in for the data-api positions call."""

    harness = None  # set per test

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, params=None, **kw):
        h = FakeHttpxClient.harness
        h.events.append(["http.get", url, _jsonable(params)])
        spec = h.data_api
        if isinstance(spec, Exception):
            raise spec
        status, payload = spec
        return SimpleNamespace(status_code=status, json=lambda: payload)


def _divergence(fc, mean):
    if fc == "bad":
        raise ValueError("bad graphcast")
    return {"signal": "strong_diverge", "ai_max_c": 24.0, "physics_mean_c": mean,
            "divergence_c": 24.0 - mean, "confidence_multiplier": 0.8}


def _trend(city, mean):
    if city == "nyc":
        return SimpleNamespace(signal="warming", trend_per_cycle=0.4,
                               stability=0.7, confidence_multiplier=1.1)
    return SimpleNamespace(signal="stable", trend_per_cycle=0.0,
                           stability=1.0, confidence_multiplier=1.0)


# ---------------------------------------------------------------------------
# frozen clock
# ---------------------------------------------------------------------------
# run_cycle's horizon filter and trading_today() depend on the time of day,
# so an unfrozen clock makes the goldens change around UTC midnight. Every
# cycle runs at 15:00 UTC on the real current date (dates are normalised to
# <D0>.. placeholders, so the date itself doesn't matter).

_REAL_DATETIME = datetime
FROZEN_NOW = _REAL_DATETIME.combine(
    _REAL_DATETIME.now(UTC).date(), time(15, 0), tzinfo=UTC,
)


class _FrozenMeta(type):
    # Code under test does isinstance(x, datetime) on real datetimes; keep
    # that true while `datetime` names this subclass.
    def __instancecheck__(cls, obj):
        return isinstance(obj, _REAL_DATETIME)


class FrozenDatetime(_REAL_DATETIME, metaclass=_FrozenMeta):
    @classmethod
    def now(cls, tz=None):
        return FROZEN_NOW.astimezone(tz) if tz else FROZEN_NOW.replace(tzinfo=None)

    @classmethod
    def utcnow(cls):
        return FROZEN_NOW.replace(tzinfo=None)


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------

class Harness:
    def __init__(self, monkeypatch, tmp_path, caplog):
        self.mp = monkeypatch
        for mod in (scheduler, edge_module, model_timing):
            monkeypatch.setattr(mod, "datetime", FrozenDatetime)
        monkeypatch.setattr(sys.modules[__name__], "datetime", FrozenDatetime)
        self.tmp_path = tmp_path
        self.caplog = caplog
        self.d = [scheduler.trading_today() + timedelta(days=i) for i in range(5)]
        self.events: list = []
        self.markets = []
        self.model_prob: dict[str, float | None] = {}
        self.edge_spec: dict[str, dict] = {}
        self.consensus_spec: dict[str, dict | None] = {}
        self.forecasts: dict[tuple, list] = {}
        self.books: dict[str, object] = {}
        self.ai_batch: dict[date, object] = {}
        self.claude: dict[str, object] = {}
        self.gemini: dict[str, object] = {}
        self.cb: list[float] = []
        self.pattern = {}
        self.exit_specs = {"paper": [], "live": []}
        self.exit_decisions: dict[str, str] = {}
        self.exit_crash = False
        self.resolve = 0
        self.enso_error = False
        self.golden = False
        self.market_map_error = False
        self.portfolio_error = False
        self.equity = 1000.0
        self.data_api = (200, [])
        self.edge_calls: list = []
        self.consensus_calls: list = []
        self._install()

    # -- construction helpers ------------------------------------------------

    def market(self, mid, city, day, yes, *, no=None, direction="gte", value=20.0,
               low=None, high=None, lo_int=None, hi_int=None, tokens=True,
               vol=100.0, liq=50.0):
        from weather_edge.fetchers.polymarket import MarketInfo

        m = MarketInfo(
            market_id=mid, city_id=city, target_date=self.d[day], question=f"{mid} bucket?",
            threshold_value=value, threshold_dir=direction,
            threshold_low_c=low, threshold_high_c=high,
            bucket_low_int=lo_int, bucket_high_int=hi_int,
            yes_price=yes, no_price=no if no is not None else round(1.0 - yes, 4),
            token_id_yes=f"{mid}-Y" if tokens else None,
            token_id_no=f"{mid}-N" if tokens is True else None,
            volume_24h=vol, liquidity=liq,
        )
        self.markets.append(m)
        return m

    def edge(self, mid, *, edge=0.10, side="YES", size=20.0, strategy="core",
             tier="high", net_edge=None):
        self.edge_spec[mid] = dict(edge=edge, side=side, size=size, strategy=strategy,
                                   tier=tier, net_edge=net_edge)

    def settings(self, **kw):
        for k, v in kw.items():
            self.mp.setattr(scheduler.settings, k, v, raising=False)

    def store(self):
        return make_store(self.tmp_path / "cycle.db", self.events)

    def executor(self, **kw):
        return FakeExecutor(self.events, **kw)

    def paper(self, bankroll=1000.0):
        from weather_edge.trading.paper import PaperTrader

        trader = PaperTrader(bankroll=bankroll)
        events = self.events
        real_place, real_spread, real_close = (
            trader.place_trade, trader.place_spread_trade, trader.close_position,
        )

        def place_trade(signal):
            t = real_place(signal)
            events.append(["paper.place_trade", signal.market_id,
                           None if t is None else [t.trade_id, t.size_usd, t.fee_usd]])
            return t

        def place_spread_trade(signal, hedge):
            t = real_spread(signal, hedge)
            events.append(["paper.place_spread_trade", signal.market_id, hedge.side,
                           hedge.cost, None if t is None else t.trade_id])
            return t

        def close_position(trade, price, volume_24h=None):
            events.append(["paper.close_position", trade.market_id, price])
            return real_close(trade, price, volume_24h)

        trader.place_trade = place_trade
        trader.place_spread_trade = place_spread_trade
        trader.close_position = close_position
        return trader

    def open_paper_trade(self, trader, mid, *, city="nyc", side="YES", size=10.0,
                         entry=0.30):
        from weather_edge.trading.paper import PaperTrade

        trade = PaperTrade(
            trade_id=900 + len(trader.trades), market_id=mid, city_id=city, side=side,
            size_usd=size, entry_price=entry, description=f"[TOMORROW] {mid} old",
        )
        trader.trades.append(trade)
        return trade

    # -- fakes ----------------------------------------------------------------

    def _install(self):
        self._install_state()
        self._install_signal_fakes()
        self._install_io_fakes()
        self.caplog.set_level(logging.DEBUG, logger="weather_edge")

    def _install_state(self):
        from weather_edge import config, live_state
        from weather_edge.analysis import risk_controls

        mp, h = self.mp, self
        proxy = SettingsProxy(config.settings)
        mp.setattr(scheduler, "settings", proxy)
        mp.setattr(config, "settings", proxy)
        self.settings(
            hail_mary_mode=False, bankroll=1000.0, max_trades_per_cycle=20,
            max_core_zscore=2.0, max_stale_forecast_hours=6.0, require_ai_review=True,
            max_ai_reviews_per_cycle=20, penny_min_position=1.0, penny_max_position=30.0,
            pool_today_pct=0.6, pool_tomorrow_pct=0.3, pool_penny_pct=0.1,
        )
        mp.setattr(live_state, "_get_redis", lambda: None)
        mp.setattr(claude_reasoning, "_review_memory",
                   claude_reasoning.AIReviewMemory(persist=False))
        claude_reasoning.clear_decisions()
        mp.setattr(scheduler, "ANTHROPIC_API_KEY", "test-key")
        mp.setattr(scheduler, "record_decision", lambda r: None)

        profile = risk_controls.RiskProfile(
            name="balanced", drawdown_scale_back_pct=0.15, drawdown_kill_pct=0.25,
            scale_back_factor=0.5, max_group_exposure_pct=0.2, max_yes_exposure_pct=0.35,
            max_gross_exposure_multiple=2.0, kelly_fraction=0.25, max_position_pct=0.03,
            reserve_pct=0.1, penny_max_position=30.0, min_edge=0.05, min_edge_yes=0.06,
            min_edge_no=0.03, fee_alpha_max=0.4, compound_factor=0.5,
        )
        mp.setattr(risk_controls, "get_active_profile", lambda: profile)
        mp.setattr(risk_controls, "_circuit_breaker", risk_controls.CircuitBreakerState())
        mp.setattr(risk_controls, "live_circuit_breaker_multiplier",
                   lambda: h.cb.pop(0) if h.cb else 1.0)

    def _install_signal_fakes(self):
        from weather_edge.analysis import forecast_trends, gemini_reasoning
        from weather_edge.fetchers import gribstream

        mp, h = self.mp, self
        mp.setattr(scheduler, "discover_weather_markets", self._discover)
        mp.setattr(scheduler, "fetch_city_forecasts", self._fetch_city)
        mp.setattr(scheduler, "compute_consensus", self._consensus)
        mp.setattr(scheduler, "compute_model_prob_for_market",
                   lambda m, c: h.model_prob.get(m.market_id, 0.35))
        mp.setattr(scheduler, "calculate_edge", self._calc_edge)
        mp.setattr(scheduler, "is_golden_window", lambda: h.golden)
        mp.setattr(scheduler, "detect_patterns", lambda c, f: [])
        mp.setattr(scheduler, "get_pattern_adjustment",
                   lambda c, a: h.pattern.get(c.value, (1.0, 0.0)))
        mp.setattr(scheduler, "resolve_open_trades", self._resolve_open)
        mp.setattr(scheduler, "analyze_trade", self._analyze)
        mp.setattr(gemini_reasoning, "red_team_trade", self._red_team)
        mp.setattr(gribstream, "fetch_ai_forecasts_batch", self._ai_batch)
        mp.setattr(gribstream, "compute_ai_physics_divergence", _divergence)
        mp.setattr(forecast_trends, "record_forecast", lambda c, m: None)
        mp.setattr(forecast_trends, "compute_trend", _trend)

    def _install_io_fakes(self):
        import httpx

        from weather_edge.analysis import enso_regime, exit_monitor
        from weather_edge.fetchers import polymarket
        from weather_edge.trading import portfolio_sync

        mp = self.mp
        mp.setattr(enso_regime, "fetch_enso_state", self._enso)
        mp.setattr(polymarket, "fetch_book_prices", self._book)
        mp.setattr(portfolio_sync, "sync_market_map_from_discovery", self._market_map)
        mp.setattr(portfolio_sync, "sync_portfolio", self._sync_portfolio)
        mp.setattr(portfolio_sync, "fetch_polymarket_state", self._poly_state)
        mp.setattr(exit_monitor, "scan_for_exits_async", self._scan)
        mp.setattr(exit_monitor, "ai_review_exit", self._review_exit)
        FakeHttpxClient.harness = self
        mp.setattr(httpx, "AsyncClient", FakeHttpxClient)

    async def _discover(self):
        self.events.append(["discover"])
        return list(self.markets)

    async def _fetch_city(self, city_id, target_date):
        key = (city_id, target_date)
        if key in self.forecasts:
            return list(self.forecasts[key])
        return fake_forecasts()

    def _consensus(self, city_id, target_date, variable, forecasts, thresholds=None):
        self.consensus_calls.append([city_id.value, target_date, variable, len(forecasts),
                                     thresholds])
        spec = self.consensus_spec.get(city_id.value, {})
        if spec is None:
            return None
        return SimpleNamespace(
            weighted_mean=spec.get("mean", 21.0), mean_value=spec.get("mean", 21.0),
            std_dev=spec.get("std", 0.5), raw_std_dev=spec.get("raw_std"),
            confidence=spec.get("conf", 0.9), model_count=len(forecasts),
        )

    def _calc_edge(self, **kw):
        rec = {k: v for k, v in kw.items() if k != "hours_to_resolution"}
        rec["hours_ok"] = kw["hours_to_resolution"] >= 0
        self.edge_calls.append(rec)
        spec = self.edge_spec.get(kw["market_id"], {})
        e = spec.get("edge", 0.10)
        net = spec.get("net_edge")
        return Signal(
            market_id=kw["market_id"], consensus_id=None,
            computed_at=datetime.now(UTC),
            model_prob=kw["model_prob"], market_prob=kw["market_prob"],
            model_confidence=kw["model_confidence"], edge=e,
            net_edge=net if net is not None else round(e - 0.01, 4),
            edge_pct=0.5, kelly_fraction=0.1, half_kelly=0.05,
            recommended_side=TradeSide(spec.get("side", "YES")),
            recommended_size=spec.get("size", 20.0),
            confidence_tier=SignalTier(spec.get("tier", "high")),
            city_id=kw["city_id"], description=kw["description"],
            target_date=kw["target_date"], spread=kw["spread"],
            hours_to_resolution=kw["hours_to_resolution"],
            strategy=spec.get("strategy", "core"),
        )

    async def _resolve_open(self, trader):
        self.events.append(["resolve_open_trades"])
        if isinstance(self.resolve, Exception):
            raise self.resolve
        return self.resolve

    async def _analyze(self, sig, model_vals, mean, std, variable="temp_max_c"):
        self.events.append(["claude", sig.market_id, round(mean, 4), round(std, 4)])
        spec = self.claude.get(sig.market_id, {})
        if spec is None:
            return None
        return claude_reasoning.TradeReasoning(
            signal=sig, should_trade=spec.get("trade", True),
            confidence_adjustment=spec.get("adj", 1.0),
            rationale=f"claude on {sig.market_id}", risk_factors=[],
            weather_insight="", review_ok=spec.get("ok", True),
        )

    async def _red_team(self, sig, model_vals, mean, std, claude_rationale=""):
        self.events.append(["gemini", sig.market_id])
        spec = self.gemini.get(sig.market_id)
        if isinstance(spec, Exception):
            raise spec
        return spec

    async def _enso(self):
        if self.enso_error:
            raise RuntimeError("enso down")

    async def _ai_batch(self, cities, target_date):
        self.events.append(["gribstream", [c.value for c in cities], target_date])
        spec = self.ai_batch.get(target_date, {})
        if isinstance(spec, Exception):
            raise spec
        return spec

    async def _book(self, m):
        self.events.append(["book", m.market_id])
        spec = self.books.get(m.market_id)
        if isinstance(spec, BaseException):
            raise spec
        return spec

    async def _market_map(self, store, markets):
        self.events.append(["sync_market_map", len(markets)])
        if self.market_map_error:
            raise RuntimeError("map down")
        return len(markets)

    async def _sync_portfolio(self, executor, store, market_lookup=None):
        self.events.append(["sync_portfolio"])
        if self.portfolio_error:
            raise RuntimeError("sync down")
        return {"positions": 1}

    async def _poly_state(self, executor, wallet):
        self.events.append(["fetch_polymarket_state", wallet])
        return {"portfolio_value": self.equity}

    async def _scan(self, open_trades, market_prices, model_probs, forecast_cache=None,
                    no_prices=None):
        from weather_edge.analysis.exit_monitor import ExitCandidate

        kind = "live" if open_trades and open_trades[0].source == "live" else "paper"
        self.events.append(["scan_exits", kind, [
            [t.market_id, t.side, t.size_usd, t.entry_price, t.total_shares]
            for t in open_trades
        ], sorted(market_prices.items()), sorted(model_probs.items()),
            sorted((no_prices or {}).items()),
            sorted(f"{c.value}|{d}" for c, d in (forecast_cache or {}))])
        if self.exit_crash:
            raise RuntimeError("scan crashed")
        by_id = {t.market_id: t for t in open_trades}
        return [
            ExitCandidate(
                trade=by_id[spec["market_id"]], reason=spec.get("reason", "edge_inversion"),
                current_model_prob=0.2, current_market_price=spec.get("price", 0.40),
                original_edge=0.1, current_edge=spec.get("edge", -0.05),
                urgency=spec.get("urgency", "medium"), token_price=spec.get("token_price"),
            )
            for spec in self.exit_specs[kind]
        ]

    async def _review_exit(self, candidate, model_vals, mean, std):
        self.events.append(["ai_review_exit", candidate.trade.market_id,
                            sorted(model_vals.items()), round(mean, 4), round(std, 4)])
        candidate.final_decision = self.exit_decisions.get(candidate.trade.market_id, "EXIT")
        candidate.claude_rationale = "claude exit view"
        candidate.gemini_rationale = "gemini exit view"
        return candidate

    # -- run + snapshot ---------------------------------------------------------

    async def run(self, trader=None, target_dates="default", **kw):
        if target_dates == "default":
            target_dates = [self.d[0], self.d[2], self.d[3]]
        self.trader = trader
        self.store_obj = kw.get("store")
        self.result = await scheduler.run_cycle(trader, target_dates, **kw)
        return self.result

    def snapshot(self) -> dict:
        signals, cache, city_volume = self.result
        snap = {
            "returned_signals": [
                {"market_id": s.market_id, "city": s.city_id, "side": s.recommended_side.value,
                 "edge": s.edge, "net_edge": s.net_edge, "size": s.recommended_size,
                 "market_prob": s.market_prob, "strategy": s.strategy,
                 "tier": s.confidence_tier.value}
                for s in signals
            ],
            "forecast_cache_keys": sorted(f"{c.value}|{d}" for c, d in cache),
            "city_volume": city_volume,
            "edge_calls": self.edge_calls,
            "consensus_calls": self.consensus_calls,
            "events": self.events,
            "decision_history": [
                {k: v for k, v in e.items() if k != "time"}
                for e in claude_reasoning.get_decisions()
            ],
            "logs": [
                f"{r.levelname} {r.name} {r.getMessage()}"
                for r in self.caplog.records if r.name.startswith("weather_edge")
            ],
        }
        if self.trader is not None:
            snap["paper_trades"] = [
                {"trade_id": t.trade_id, "market_id": t.market_id, "side": t.side,
                 "size_usd": t.size_usd, "entry_price": t.entry_price, "fee_usd": t.fee_usd,
                 "description": t.description, "strategy": t.strategy,
                 "status": _jsonable(t.status), "exit_price": t.exit_price, "pnl": t.pnl}
                for t in self.trader.trades
            ]
        if self.store_obj is not None:
            conn = self.store_obj.conn
            snap["db"] = {
                "live_trades": [list(r) for r in conn.execute(
                    "SELECT order_id, market_id, side, status FROM live_trades ORDER BY order_id")],
                "positions": [list(r) for r in conn.execute(
                    "SELECT asset_id, condition_id, total_shares FROM positions "
                    "ORDER BY asset_id")],
                "ai_decisions": [list(r) for r in conn.execute(
                    "SELECT source, decision, market_id, confidence_adj, dissent_strength "
                    "FROM ai_decisions ORDER BY id")],
                "forecast_snapshots": [list(r) for r in conn.execute(
                    "SELECT city_id, target_date, COUNT(*) FROM forecast_snapshots "
                    "GROUP BY city_id, target_date ORDER BY city_id, target_date")],
            }
        text = json.dumps(_jsonable(snap), indent=1, sort_keys=True, default=str)
        for i, d in enumerate(self.d):
            text = text.replace(d.isoformat(), f"<D{i}>")
        text = text.replace(str(self.tmp_path), "<TMP>")
        text = re.sub(r"\b\d\d:\d\d:\d\d\b", "<HMS>", text)  # forecast fetch times
        return json.loads(text)

    def check_golden(self, name: str) -> dict:
        snap = self.snapshot()
        path = GOLDEN_DIR / f"{name}.json"
        if UPDATE or not path.exists():
            if not UPDATE:
                pytest.fail(f"golden file {path} missing; run with UPDATE_RUN_CYCLE_GOLDEN=1")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n")
        expected = json.loads(path.read_text())
        for key in sorted(set(expected) | set(snap)):
            assert snap.get(key) == expected.get(key), f"{name}: '{key}' differs from golden"
        return snap


def fake_forecasts(n=5, base=20.0, fetched_at=None):
    fetched_at = fetched_at or datetime.now(UTC)
    return [
        SimpleNamespace(model_name=f"model{i}", temp_max_c=base + 0.25 * i,
                        fetched_at=fetched_at)
        for i in range(n)
    ]


@pytest.fixture
def h(monkeypatch, tmp_path, caplog):
    harness = Harness(monkeypatch, tmp_path, caplog)
    yield harness
    claude_reasoning.clear_decisions()


def remember(h, mid, city, day, *, veto=None, approval=None, side="YES"):
    """Record a verdict from an earlier cycle in the review memory."""
    sig = SimpleNamespace(market_id=mid, target_date=str(h.d[day]), city_id=city,
                          recommended_side=TradeSide(side))
    memory = claude_reasoning.get_review_memory()
    if veto:
        memory.record_veto(sig, veto, "claude")
    if approval is not None:
        memory.record_approval(sig, approval, "earlier approval")


def logs(snap, needle):
    return [line for line in snap["logs"] if needle in line]


def events(snap, name):
    return [e for e in snap["events"] if e[0] == name]


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------

async def test_paper_cycle(h):
    """Paper-only cycle through every signal-stage branch, plus a paper exit."""
    store = h.store()
    trader = h.paper()
    h.golden = True
    h.enso_error = True
    h.resolve = 2
    h.settings(max_trades_per_cycle=2)
    # NYC d2: tail_no wins the one-per-city-date filter over a bigger core edge
    h.market("nyc-a", City.NYC, 2, 0.20, value=22.0)
    h.edge("nyc-a", edge=0.20)
    h.market("nyc-b", City.NYC, 2, 0.30, direction="range", low=20.0, high=21.0,
             lo_int=68, hi_int=69)
    h.edge("nyc-b", edge=0.12, strategy="tail_no")
    h.market("nyc-x", City.NYC, 2, 0.995)           # extreme price, skipped
    h.market("nyc-n", City.NYC, 2, 0.25)            # model prob None
    h.model_prob["nyc-n"] = None
    # LON d3: pattern shift, one plain signal
    h.pattern["lon"] = (1.1, 0.5)
    h.market("lon-a", City.LON, 3, 0.40, value=21.0)
    h.edge("lon-a", edge=0.09, side="NO", size=15.0)
    # LON d2: z-score reject and a low-tier signal
    h.market("lon-z", City.LON, 2, 0.20, value=30.0)
    h.market("lon-low", City.LON, 2, 0.20, value=21.0)
    h.edge("lon-low", tier="low")
    # DAL d3: third city-date, trimmed by the cycle limit
    h.market("dal-a", City.DAL, 3, 0.15, value=21.0)
    h.edge("dal-a", edge=0.05, size=8.0)
    # CHI d2: no forecasts at all; SEA d2 stale-but-fresh cache; ATL d2 too old
    h.market("chi-a", City.CHI, 2, 0.20)
    h.forecasts[(City.CHI, h.d[2])] = []
    h.market("sea-a", City.SEA, 2, 0.20)
    h.forecasts[(City.SEA, h.d[2])] = []
    h.market("atl-a", City.ATL, 2, 0.20)
    h.forecasts[(City.ATL, h.d[2])] = []
    cache = {
        (City.SEA, h.d[2]): fake_forecasts(),
        (City.ATL, h.d[2]): fake_forecasts(
            fetched_at=datetime.now(UTC) - timedelta(hours=30)),
        (City.NYC, h.d[0] - timedelta(days=1)): fake_forecasts(),  # evicted
    }
    h.consensus_spec["sea"] = None                  # consensus unavailable
    # MIA d3: model agreement reject; TOR d3: too few models
    h.market("mia-a", City.MIA, 3, 0.20)
    h.consensus_spec["mia"] = {"raw_std": 3.0}
    h.market("tor-a", City.TOR, 3, 0.20)
    h.forecasts[(City.TOR, h.d[3])] = fake_forecasts(n=2)
    # Horizon-blocked date and a market without a city
    h.market("nyc-today", City.NYC, 0, 0.20)
    h.market("nocity", None, 2, 0.20)
    # Parity: two HOU buckets whose YES prices sum way over 1
    h.market("hou-1", City.HOU, 3, 0.70, value=19.0)
    h.market("hou-2", City.HOU, 3, 0.70, direction="lte", value=18.0, high=18.0)
    h.consensus_spec["hou"] = None
    # GraphCast: d2 NYC diverges, d3 fetch fails
    h.ai_batch[h.d[2]] = {City.NYC: "graphcast-nyc", City.LON: "bad"}
    h.ai_batch[h.d[3]] = RuntimeError("gribstream down")
    # Books: nyc-b profitable (paper hedge), lon-a errors
    h.books["nyc-b"] = {"yes_ask": 0.31, "no_ask": 0.60, "yes_bid": 0.29,
                        "profitable": True, "spread_cost": 0.91, "spread_profit": 0.09}
    h.books["lon-a"] = RuntimeError("clob down")
    # Paper exit monitor: one EXIT, one HOLD
    h.open_paper_trade(trader, "old-exit")
    h.open_paper_trade(trader, "old-hold", side="NO", entry=0.70)
    h.exit_specs["paper"] = [{"market_id": "old-exit", "price": 0.25},
                             {"market_id": "old-hold"}]
    h.exit_decisions["old-hold"] = "HOLD"

    signals, cache_out, volume = await h.run(trader, store=store, forecast_cache=cache)

    snap = h.check_golden("paper_cycle")
    assert [s.market_id for s in signals] == ["nyc-b", "lon-a"]
    assert cache_out is cache
    assert volume["nyc"]["markets"] == 5
    assert [e[1] for e in events(snap, "paper.place_trade")] == ["nyc-b", "lon-a"]
    assert events(snap, "paper.place_spread_trade")
    assert [e[1] for e in events(snap, "paper.close_position")] == ["old-exit"]
    for needle in ("HORIZON FILTER", "GOLDEN WINDOW", "ZSCORE REJECT", "MULTI-BUCKET",
                   "CYCLE LIMIT", "MODEL AGREEMENT REJECT", "INSUFFICIENT_MODELS",
                   "STALE DATA:", "STALE DATA REJECTED", "No forecasts for chi",
                   "AI DIVERGE", "PATTERN SHIFT", "TREND", "PARITY ARBITRAGE",
                   "SPREAD OPP", "PAPER EXIT"):
        assert logs(snap, needle), needle


async def test_live_entry_cycle(h):
    """Live (non-dry-run) cycle hitting every entry gate in the execution loop."""
    store = h.store()
    trader = h.paper()
    ex = h.executor(balance=500.0)
    h.equity = 1000.0
    h.settings(max_trades_per_cycle=30)
    # Seeded exchange state
    seed_live_trade(store, "sold-cool", "cool", "SELL", 0.40, status="filled",
                    placed_ago_min=30, description="EXIT: edge_inversion")
    seed_live_trade(store, "sold-bypass", "bypass", "SELL", 0.40, status="filled",
                    placed_ago_min=30, description="EXIT: edge_inversion")
    seed_live_trade(store, "o-keep", "keep", "YES", 0.20, placed_ago_min=5)
    seed_live_trade(store, "o-repl", "repl", "YES", 0.10, placed_ago_min=5)
    seed_live_trade(store, "o-chase", "chase", "YES", 0.15, placed_ago_min=120)
    seed_live_trade(store, "o-cskip", "cskip", "YES", 0.20, placed_ago_min=120)
    seed_live_trade(store, "o-chasey", "chasey", "YES", 0.25, placed_ago_min=120)
    seed_live_trade(store, "o-cfail", "cfail", "NO", 0.50, placed_ago_min=5)
    ex.cancel_fail.add("o-cfail")
    seed_position(store, "tok-held", "held", city="nyc", outcome="YES",
                  shares=30, avg_price=0.30, cost_basis=9.0)
    seed_position(store, "tok-yes-big", "yesbig", city="sel", outcome="YES",
                  shares=1000, avg_price=0.33, cost_basis=330.0)
    seed_position(store, "tok-stale", "gone", city="lon", outcome="NO",
                  shares=50, avg_price=0.50, cost_basis=25.0)
    h.data_api = (200, [{"conditionId": "held", "size": 30}, {"conditionId": "yesbig",
                        "size": 1000}, {"conditionId": "zero", "size": 0}])

    def signal(mid, city, day, yes, **edge):
        h.market(mid, city, day, yes, value=21.0)
        h.edge(mid, **edge)

    signal("cool", City.NYC, 2, 0.20, edge=0.05, side="NO")     # exit cooldown
    signal("bypass", City.NYC, 3, 0.20, edge=0.15, side="NO")   # cooldown bypass, taker
    signal("small", City.LON, 2, 0.30, edge=0.05, size=3.0, side="NO")  # bumped to $5
    # fee-gated for paper (so paper cannot trim it), then over the live balance
    signal("bal", City.LON, 3, 0.50, edge=0.025, size=600.0, side="NO")
    signal("lowedge", City.DAL, 2, 0.30, edge=0.015, size=10.0, side="NO")  # fee gate, live skip
    h.market("notok", City.DAL, 3, 0.30, value=21.0, tokens="yes-only")
    h.edge("notok", edge=0.05, side="NO")                       # no NO token
    signal("held", City.SEA, 2, 0.30, edge=0.05, side="NO")     # position exists
    signal("keep", City.SEA, 3, 0.205, edge=0.05)               # same price, keep
    signal("repl", City.ATL, 2, 0.30, edge=0.05, side="YES", size=15.0)  # replace + YES trim
    signal("chase", City.ATL, 3, 0.30, edge=0.05, side="NO")    # chase (NO side)
    signal("cskip", City.TOR, 2, 0.205, edge=0.05)              # chase skip at mid
    signal("chasey", City.SFO, 2, 0.30, edge=0.05)              # chase (YES side)
    signal("cfail", City.TOR, 3, 0.45, edge=0.05, side="NO")    # cancel fails: no replace
    signal("rej", City.MIA, 3, 0.30, edge=0.05, side="NO")
    signal("boom", City.LAX, 2, 0.30, edge=0.05, side="NO")
    signal("hedge", City.LAX, 3, 0.30, edge=0.09, side="NO")    # live hedge placed
    signal("hfail", City.HOU, 2, 0.30, edge=0.09, side="NO")    # live hedge fails
    signal("hnone", City.HOU, 3, 0.30, edge=0.10, side="NO")    # no hedge (no book)
    signal("tail", City.DEN, 2, 0.04, edge=0.08, side="NO", strategy="tail")
    ex.limit_results[("rej", "core")] = "rejected"
    ex.limit_results[("boom", "core")] = RuntimeError("clob exploded")
    import requests
    ex.limit_results[("hfail", "spread")] = requests.RequestException("hedge down")
    book = {"yes_ask": 0.31, "no_ask": 0.66, "yes_bid": 0.29, "profitable": True,
            "spread_cost": 0.97, "spread_profit": 0.03}
    h.books.update({"hedge": book, "hfail": book, "bypass": TimeoutError()})

    signals, _, _ = await h.run(trader, store=store, live_executor=ex)

    snap = h.check_golden("live_entry_cycle")
    placed = [e[1]["market_id"] for e in events(snap, "exec.place_limit_order")]
    assert "cool" not in placed and "held" not in placed and "keep" not in placed
    assert {"bypass", "small", "repl", "chase", "hedge", "hfail"} <= set(placed)
    # The resting order could not be cancelled, so no replacement is placed.
    assert "cfail" not in placed
    assert ["exec.cancel_order", "o-chase"] in snap["events"]
    assert ["tok-stale", "gone", 0.0] in snap["db"]["positions"]  # cleaned: not on exchange
    for needle in ("EXIT COOLDOWN", "COOLDOWN BYPASS", "BALANCE LIMIT", "LIVE SKIP",
                   "FEE GATE (paper only)", "POSITION EXISTS", "LIVE KEEP", "LIVE REPLACE",
                   "PRICE CHASE:", "PRICE CHASE SKIP", "Failed to cancel old order",
                   "YES EXPOSURE TRIM",
                   "LIVE ORDER FAILED", "LIVE SPREAD HEDGE:", "LIVE SPREAD HEDGE FAILED",
                   "TAKER ENTRY", "PORTFOLIO EQUITY", "USDC balance", "Market map updated"):
        assert logs(snap, needle), needle


async def test_live_circuit_breaker(h):
    """Live drawdown breaker: kill, trim, and a trim that lands below the minimum."""
    store = h.store()
    ex = h.executor(balance=500.0)
    h.market("cb-kill", City.NYC, 2, 0.30, value=21.0)
    h.edge("cb-kill", edge=0.07, side="NO")
    h.market("cb-trim", City.LON, 2, 0.30, value=21.0)
    h.edge("cb-trim", edge=0.06, side="NO", size=16.0)
    h.market("cb-min", City.DAL, 2, 0.30, value=21.0)
    h.edge("cb-min", edge=0.05, side="NO", size=6.0)
    h.cb = [0.0, 0.5, 0.5]  # consumed in net-edge order at the risk block

    await h.run(None, store=store, live_executor=ex)

    snap = h.check_golden("live_circuit_breaker")
    orders = [(e[1]["market_id"], e[1]["size"]) for e in events(snap, "exec.place_limit_order")]
    assert orders == [("cb-trim", 8.0)]
    assert logs(snap, "LIVE CIRCUIT BREAKER: killed, skipping nyc")
    assert logs(snap, "TRIMMED BELOW MIN: dal")


async def test_live_risk_denials_without_balance(h):
    """Balance fetch fails; YES cap, correlation cap and gross trim decide sizes."""
    store = h.store()
    ex = h.executor(balance=RuntimeError("balance down"))
    h.equity = 100.0
    h.data_api = (500, [])
    seed_position(store, "tok-yes", "yesheld", city="sel", outcome="YES",
                  shares=100, avg_price=0.40, cost_basis=40.0)
    seed_position(store, "tok-grp", "grpheld", city="lon", outcome="NO",
                  shares=100, avg_price=0.20, cost_basis=20.0)

    h.market("yesdeny", City.NYC, 2, 0.30, value=21.0)
    h.edge("yesdeny", edge=0.05, side="YES")
    h.market("corr", City.MUC, 2, 0.30, value=21.0)       # uk_europe group full
    h.edge("corr", edge=0.05, side="NO")
    h.market("gross", City.DAL, 2, 0.30, value=21.0)
    h.edge("gross", edge=0.05, side="NO", size=150.0)
    h.market("corrtrim", City.SEA, 2, 0.30, value=21.0)
    h.edge("corrtrim", edge=0.05, side="NO", size=30.0)

    await h.run(None, store=store, live_executor=ex)

    snap = h.check_golden("live_risk_denials")
    for needle in ("Failed to fetch live balance", "YES EXPOSURE CAP", "CORRELATION LIMIT",
                   "CORRELATION TRIM"):
        assert logs(snap, needle), needle


@pytest.mark.parametrize("variant", ["a", "b", "c"])
async def test_live_exit_cycle(h, variant):
    """Live early-exit monitor: YES and NO exits, sell-half, keep/replace, failures."""
    store = h.store()
    ex = h.executor(balance=500.0)
    trader = h.paper()
    h.consensus_spec.update({c.value: None for c in City})  # no new signals
    for mid, city in (("p-yes", City.NYC), ("p-no", City.LON), ("p-half", City.DAL),
                      ("p-trim", City.SEA), ("p-keep", City.ATL), ("p-repl", City.TOR),
                      ("p-hold", City.CHI), ("p-fail", City.MIA), ("p-small", City.LAX)):
        h.market(mid, city, 2, 0.40, value=21.0)
    seed_position(store, "a-yes", "p-yes", city="nyc", outcome="YES",
                  shares=20, avg_price=0.30, cost_basis=6.0)
    seed_position(store, "a-no", "p-no", city="lon", outcome="NO",
                  shares=21, avg_price=0.55, cost_basis=11.55)
    seed_position(store, "a-half", "p-half", city="dal", outcome="YES",
                  shares=30, avg_price=0.20, cost_basis=6.1)
    seed_position(store, "a-trim", "p-trim", city="sea", outcome="NO",
                  shares=40, avg_price=0.50, cost_basis=20.0)
    seed_position(store, "a-keep", "p-keep", city="atl", outcome="YES",
                  shares=10, avg_price=0.35, cost_basis=3.5)
    seed_position(store, "a-repl", "p-repl", city="tor", outcome="NO",
                  shares=12, avg_price=0.45, cost_basis=5.4)
    seed_position(store, "a-hold", "p-hold", city="chi", outcome="YES",
                  shares=8, avg_price=0.25, cost_basis=2.0)
    seed_position(store, "a-fail", "p-fail", city="mia", outcome="NO",
                  shares=9, avg_price=0.60, cost_basis=5.39)
    seed_position(store, "a-small", "p-small", city="lax", outcome="YES",
                  shares=3, avg_price=0.10, cost_basis=0.3)  # < 5 shares
    h.data_api = (200, [{"conditionId": f"p-{x}", "size": 1} for x in (
        "yes", "no", "half", "trim", "keep", "repl", "hold", "fail", "small")])
    seed_live_trade(store, "buy-yes", "p-yes", "YES", 0.30)          # cancelled on exit
    seed_live_trade(store, "buy-no", "p-no", "NO", 0.55)
    ex.cancel_fail.add("buy-no")
    seed_live_trade(store, "sell-half", "p-half", "SELL", 0.30)      # sell-half skip
    seed_live_trade(store, "trimmed", "p-trim", "SELL", 0.50, status="filled",
                    description="SELL_HALF: sell_half")
    seed_live_trade(store, "sell-keep", "p-keep", "SELL", 0.40)      # keep (drift 0)
    seed_live_trade(store, "sell-repl", "p-repl", "SELL", 0.70)      # replace
    ex.cancel_fail.add("sell-repl")                                   # so: no 2nd sell
    ex.sell_results["p-repl"] = "rejected"
    ex.sell_results["p-fail"] = RuntimeError("sell exploded")
    h.exit_decisions["p-hold"] = "HOLD"
    specs = {
        "a": [{"market_id": "p-yes", "urgency": "high", "token_price": 0.38},
              {"market_id": "p-no", "reason": "sell_half"},
              {"market_id": "p-half", "reason": "sell_half"},
              {"market_id": "p-hold"}],                                 # 4th: not reviewed
        "b": [{"market_id": "p-trim", "reason": "sell_half"},
              {"market_id": "p-keep", "price": 0.40},
              {"market_id": "p-repl", "token_price": 0.55}],
        "c": [{"market_id": "p-hold"},
              {"market_id": "p-fail", "token_price": 0.52},
              {"market_id": "p-keep", "token_price": 0.30}],
    }
    h.exit_specs["live"] = specs[variant]
    h.open_paper_trade(trader, "p-yes")
    h.exit_specs["paper"] = [{"market_id": "p-yes"}]

    await h.run(trader, store=store, live_executor=ex)

    snap = h.check_golden(f"live_exit_cycle_{variant}")
    scans = events(snap, "scan_exits")
    assert [s[1] for s in scans] == ["paper", "live"]
    assert "p-small" not in [p[0] for p in scans[1][2]]
    sells = [e[1] for e in events(snap, "exec.place_sell_order")]
    if variant == "a":
        assert [(s["market_id"], s["shares"], s["force_taker"]) for s in sells] == [
            ("p-yes", 20.0, True), ("p-no", 10.0, False)]
        assert logs(snap, "SELL_HALF SKIP: dal has open sell order")
        assert logs(snap, "Failed to cancel BUY order")
    elif variant == "b":
        assert logs(snap, "SELL_HALF SKIP: sea already trimmed")
        assert logs(snap, "LIVE SELL KEEP") and logs(snap, "Failed to cancel old sell order")
        # The old sell could not be cancelled, so no second sell is placed
        # (it could oversell the position).
        assert sells == []
    else:
        assert logs(snap, "LIVE EXIT FAILED")
        assert logs(snap, "LIVE SELL REPLACE: atl cancelled sell-keep")


async def test_hail_mary_cycle(h):
    """HAIL MARY: penny basket, $1 taker tickets, AI log-only, no live exits."""
    store = h.store()
    trader = h.paper()
    ex = h.executor(balance=1.5)
    h.settings(hail_mary_mode=True)
    h.consensus_spec["mia"] = {"raw_std": 3.0}                  # allowed in hail mary
    h.market("pen-yes", City.NYC, 2, 0.05, value=30.0)          # far from consensus: kept
    h.edge("pen-yes", edge=0.03, size=40.0)
    h.market("pen-no", City.NYC, 2, 0.95, value=18.0)
    h.edge("pen-no", edge=0.02, side="NO", size=40.0)
    h.market("pen-mia", City.MIA, 3, 0.06, value=21.0)
    h.edge("pen-mia", edge=0.04, size=0.0)
    h.market("pricey", City.LON, 2, 0.40, value=21.0)
    h.edge("pricey", edge=0.2)
    h.market("noedge", City.LON, 3, 0.05, value=21.0)
    h.edge("noedge", edge=-0.01)
    h.market("today", City.DAL, 0, 0.05, value=21.0)            # horizon 0 in hail mary
    h.edge("today", edge=0.05, side="YES")
    h.claude["pen-yes"] = {"trade": False}
    h.gemini["pen-no"] = {"dissent_strength": 0.95, "verdict": "DISSENT",
                          "sizing_recommendation": "skip", "counter_arguments": ["nope"],
                          "risk_the_bull_missed": "x"}
    h.exit_specs["live"] = [{"market_id": "never"}]

    signals, cache, _ = await h.run(trader, target_dates=None, store=store, live_executor=ex)

    snap = h.check_golden("hail_mary_cycle")
    assert {s.market_id for s in signals} == {"pen-yes", "pen-no", "pen-mia", "today"}
    orders = [e[1] for e in events(snap, "exec.place_limit_order")]
    assert orders and all(o["size"] == 1.0 and o["force_taker"] for o in orders)
    assert logs(snap, "HAILMARY OUT OF CASH")
    assert logs(snap, "HAILMARY override CLAUDE SKIP")
    assert logs(snap, "HAILMARY override GEMINI DISSENT")
    assert logs(snap, "HAILMARY allow")
    assert [s[1] for s in events(snap, "scan_exits")] == ["paper"]


async def test_usdc_floor_blocks_live_entries(h):
    store = h.store()
    trader = h.paper()
    ex = h.executor(balance=10.0)
    h.market("nyc-a", City.NYC, 2, 0.30, value=21.0)
    h.edge("nyc-a", edge=0.05, side="NO")
    h.exit_crash = True

    await h.run(trader, store=store, live_executor=ex)

    snap = h.check_golden("usdc_floor")
    assert not events(snap, "exec.place_limit_order")
    assert events(snap, "paper.place_trade")
    assert logs(snap, "USDC FLOOR")
    assert logs(snap, "EXIT MONITOR CRASHED")


@pytest.mark.parametrize("data_api", ["down", "ok"])
async def test_position_cap_blocks_live_entries(h, data_api):
    store = h.store()
    ex = h.executor(balance=500.0)
    h.market_map_error = True
    for i in range(50):
        seed_position(store, f"cap-{i}", f"cap-cid-{i}", city="nyc", outcome="NO",
                      shares=10, avg_price=0.5, cost_basis=5.0 + i)
    if data_api == "down":
        h.data_api = RuntimeError("data api down")
    else:
        h.data_api = (200, [{"conditionId": f"cap-cid-{i}", "size": 10} for i in range(50)])
    h.market("nyc-a", City.NYC, 2, 0.30, value=21.0)
    h.edge("nyc-a", edge=0.05, side="NO")

    await h.run(None, store=store, live_executor=ex)

    snap = h.check_golden(f"position_cap_{data_api}")
    assert not events(snap, "exec.place_limit_order")
    assert logs(snap, "pos_count=50/50")
    assert bool(logs(snap, "POSITION CAP:")) == (data_api == "ok")


async def test_portfolio_sync_failure_and_dry_run(h):
    """Dry-run executor: no portfolio sync, balance or live orders; then sync failure."""
    store = h.store()
    ex = h.executor(dry_run=True)
    h.market("nyc-a", City.NYC, 2, 0.30, value=21.0)
    await h.run(None, store=store, live_executor=ex)
    snap = h.check_golden("dry_run")
    assert not events(snap, "exec.check_balance")
    assert not events(snap, "sync_portfolio")


async def test_portfolio_sync_failure(h):
    store = h.store()
    ex = h.executor(balance=500.0)
    h.portfolio_error = True
    h.market("nyc-a", City.NYC, 2, 0.30, value=21.0)
    h.edge("nyc-a", edge=0.05, side="NO")
    await h.run(None, store=store, live_executor=ex)
    snap = h.check_golden("portfolio_sync_failure")
    assert logs(snap, "Portfolio sync failed")


async def test_claude_veto_and_gemini_skip(h):
    store = h.store()
    trader = h.paper()
    for mid, city in (("veto", City.NYC), ("gskip", City.LON), ("ghalf", City.DAL),
                      ("gerr", City.SEA), ("gcrash", City.ATL), ("cfail", City.TOR),
                      ("adj", City.CHI), ("notok", City.MIA), ("gcut", City.LAX)):
        h.market(mid, city, 2, 0.30, value=21.0)
        h.edge(mid, edge=0.06, side="NO", size=20.0)
    h.claude["veto"] = {"trade": False}
    h.gemini["gskip"] = {"dissent_strength": 0.95, "verdict": "DISSENT",
                         "sizing_recommendation": "skip", "counter_arguments": ["a", "b", "c"],
                         "risk_the_bull_missed": "fog"}
    h.gemini["ghalf"] = {"dissent_strength": 0.5, "verdict": "DISSENT",
                         "sizing_recommendation": "half", "counter_arguments": [],
                         "risk_the_bull_missed": ""}
    h.gemini["gcut"] = {"dissent_strength": 0.4, "verdict": "AGREE",
                        "sizing_recommendation": "reduce_20pct"}
    h.gemini["gerr"] = {"error": "timeout", "dissent_strength": 1.0}
    h.gemini["gcrash"] = RuntimeError("gemini crashed")
    h.claude["cfail"] = None
    h.claude["adj"] = {"adj": 0.5}
    h.claude["notok"] = {"ok": False}
    h.market("oldveto", City.HOU, 2, 0.30, value=21.0)
    h.edge("oldveto", edge=0.06, side="NO")
    remember(h, "oldveto", "hou", 2, veto="vetoed this morning")

    signals, _, _ = await h.run(trader, store=store)

    snap = h.check_golden("ai_veto")
    assert len(signals) == 10
    assert [e[1] for e in events(snap, "paper.place_trade")] == ["ghalf", "adj", "gcut"]
    assert "oldveto" not in [e[1] for e in events(snap, "claude")]
    for needle in ("CLAUDE SKIP", "GEMINI VETO", "GEMINI DISSENT", "GEMINI REVIEW FAILED",
                   "Gemini red team crashed", "CLAUDE REVIEW UNAVAILABLE", "AI GATE: 3/10",
                   "AI VETO (remembered"):
        assert logs(snap, needle), needle


@pytest.mark.parametrize("why", ["no_key", "ai_off", "budget"])
async def test_ai_review_unavailable_blocks_execution(h, why):
    trader = h.paper()
    h.market("nyc-a", City.NYC, 2, 0.30, value=21.0)
    h.market("lon-a", City.LON, 2, 0.30, value=21.0)
    h.edge("lon-a", edge=0.2)
    kw = {}
    if why == "no_key":
        h.mp.setattr(scheduler, "ANTHROPIC_API_KEY", "")
    elif why == "ai_off":
        kw["run_ai_reasoning"] = False
        h.market("dal-ok", City.DAL, 2, 0.30, value=21.0)
        h.edge("dal-ok", edge=0.05, side="NO")
        remember(h, "dal-ok", "dal", 2, approval=0.5, side="NO")
    else:
        h.settings(max_ai_reviews_per_cycle=1)

    signals, _, _ = await h.run(trader, **kw)

    snap = h.check_golden(f"ai_unavailable_{why}")
    placed = [e[1] for e in events(snap, "paper.place_trade")]
    assert placed == {"no_key": [], "ai_off": ["dal-ok"], "budget": ["lon-a"]}[why]
    assert logs(snap, "NO AI APPROVAL")


async def test_unreviewed_trades_when_review_not_required(h):
    """require_ai_review=False, no key: zero-size and fee-gated signals are skipped."""
    trader = h.paper()
    h.mp.setattr(scheduler, "ANTHROPIC_API_KEY", "")
    h.settings(require_ai_review=False)
    h.market("zero", City.NYC, 2, 0.30, value=21.0)
    h.edge("zero", size=0.0)
    h.market("fee", City.LON, 2, 0.40, value=21.0)
    h.edge("fee", edge=0.02, size=10.0)
    h.market("tail", City.DAL, 2, 0.04, value=21.0)
    h.edge("tail", edge=0.03, size=5.0, strategy="tail")
    h.books["tail"] = {"yes_ask": 0.05, "no_ask": 0.9, "profitable": True,
                       "spread_cost": 0.95, "spread_profit": 0.05}
    h.books["zero"] = None
    h.market("untok", City.SEA, 2, 0.30, value=21.0, tokens=False)
    h.edge("untok", edge=0.3)

    await h.run(trader)

    snap = h.check_golden("unreviewed")
    assert logs(snap, "UNREVIEWED TRADE")
    assert logs(snap, "ZERO SIZE")
    assert logs(snap, "CONTRACT [FEE_EATS_ALPHA]")
    assert not events(snap, "paper.place_spread_trade")  # tail never hedged


async def test_no_markets_cycle(h):
    """No markets: warnings, monitoring refresh for every city, empty results."""
    from weather_edge.trading.paper import PaperTrader

    store = h.store()
    trader = PaperTrader(bankroll=1000.0)
    trader.store = store  # store is picked up from the paper trader
    h.resolve = RuntimeError("resolver down")
    h.forecasts[(City.LON, h.d[3])] = []
    h.mp.setattr(scheduler, "validate_emos_active", lambda *a: SimpleNamespace(
        valid=False, code="EMOS_OFF", error="EMOS calibration disabled"))

    signals, cache, volume = await h.run(trader)

    snap = h.check_golden("no_markets")
    assert signals == [] and volume == {}
    assert logs(snap, "No weather markets found")
    assert logs(snap, "Paper trade resolution failed")
    assert logs(snap, "CONTRACT VIOLATION [EMOS_OFF]")
    assert len(cache) == len(City) - 1
    assert events(snap, "scan_exits") == [["scan_exits", "paper", [], [], [], [], []]]


async def test_monitoring_date_fallbacks(h):
    """Single kept date and an empty date list pick the monitoring date correctly."""
    await h.run(None, target_dates=[h.d[3]])
    first = {k for k in h.result[1]}
    assert first == {(c, h.d[3]) for c in City}
    h.events.clear()
    await h.run(None, target_dates=[h.d[0]])
    assert {k for k in h.result[1]} == {(c, h.d[1]) for c in City}
