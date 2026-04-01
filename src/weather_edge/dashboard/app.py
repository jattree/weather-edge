"""FastAPI web dashboard, dark terminal aesthetic matching ColdMath's Claude Trader."""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone

# Ensure all loggers output to stdout so systemd/journald captures them
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
from pathlib import Path

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from weather_edge.analysis.claude_reasoning import clear_decisions, get_decisions
from weather_edge.analysis.competitor_tracker import CompetitorTracker
from weather_edge.analysis.correlation_matrix import compute_correlation_matrix
from weather_edge.analysis.execution_analytics import compute_execution_analytics
from weather_edge.analysis.regret_tracker import compute_adherence
from weather_edge.analysis.resolver import _extract_target_date_from_trade
from weather_edge.analysis.sniper import ModelSniper
from weather_edge.analysis.weather_alerts import fetch_all_alerts
from weather_edge.config import CITIES, settings
from weather_edge.models.enums import City, TradeStatus
from weather_edge.persistence import PersistentPaperTrader
from weather_edge.scheduler import run_cycle

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
app = FastAPI(title="Weather Edge")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Read the HTML template once at startup (it's entirely client-side JS, no server templating needed)
_index_html = (BASE_DIR / "templates" / "index.html").read_text()

# Global state
paper_trader = PersistentPaperTrader(bankroll=settings.bankroll)
sniper = ModelSniper()
competitor_tracker = CompetitorTracker()
trading_active = False  # Must click START to begin

# Live executor (initialized on startup if LIVE_MODE=true)
live_executor = None

# Load saved risk profile
from weather_edge.analysis.risk_controls import (
    _circuit_breaker,
    set_active_profile,
)

_saved_profile = paper_trader.store.get_state("risk_profile", "aggressive")
try:
    set_active_profile(_saved_profile)
except ValueError:
    set_active_profile("aggressive")
# Init circuit breaker high-water mark from current NAV
_circuit_breaker.high_water_mark = paper_trader.bankroll + paper_trader.total_pnl
_cycle_lock = asyncio.Lock()  # Prevent concurrent cycles from corrupting state
# Cached model probs from last full cycle, used by fast exit loop
_cached_model_probs: dict[str, float] = {}
latest_state: dict = {
    "cities": {},
    "signals": [],
    "markets": [],
    "consensus": {},
    "trade_log": [],
    "bankroll": settings.bankroll,
    "capital_at_risk": 0.0,
    "session_pnl": 0.0,
    "total_pnl": 0.0,
    "win_rate": 0.0,
    "open_positions": 0,
    "best_trade": 0.0,
    "streak": 0,
    "pools": {"core": {"pnl": 0, "at_risk": 0}, "penny": {"pnl": 0, "at_risk": 0}},
    "cycle_count": 0,
    "last_update": None,
    "weather_alerts": [],
    "correlation_matrix": {"cities": [], "matrix": [], "pairs": []},
    "execution_analytics": {},
    "claude_decisions": [],
    "kill_switch": {"active": False, "reason": "", "triggered_by": "", "triggered_at": ""},
    "live": {"enabled": False},
}
connected_websockets: list[WebSocket] = []


try:
    import zoneinfo
    _HAS_ZONEINFO = True
except ImportError:
    _HAS_ZONEINFO = False


def _compute_resolution_time(trade) -> tuple[str | None, str]:
    """Compute resolution timestamp and human-readable countdown for a trade.

    Resolution = midnight in the target city's timezone + 2 hours buffer for
    NWS/observation posting.

    Returns:
        (resolves_at ISO string or None, resolves_in human-readable string)
    """
    from weather_edge.trading.paper import TradeStatus

    if trade.status != TradeStatus.OPEN:
        return None, ""

    target_date = _extract_target_date_from_trade(trade)
    if target_date is None:
        return None, ""

    # Look up city timezone
    city_id_str = trade.city_id if isinstance(trade.city_id, str) else trade.city_id.value
    tz_name = None
    for city_enum, city_config in CITIES.items():
        if city_enum.value == city_id_str.lower():
            tz_name = city_config.timezone
            break

    if tz_name is None or not _HAS_ZONEINFO:
        # Fallback: assume UTC + 2h buffer
        resolution_dt = datetime(
            target_date.year, target_date.month, target_date.day,
            2, 0, 0, tzinfo=timezone.utc,
        ) + timedelta(days=1)
    else:
        tz = zoneinfo.ZoneInfo(tz_name)
        # Midnight local time on the day AFTER target_date + 2h buffer
        local_midnight = datetime(
            target_date.year, target_date.month, target_date.day,
            0, 0, 0, tzinfo=tz,
        ) + timedelta(days=1, hours=2)
        resolution_dt = local_midnight.astimezone(timezone.utc)

    now = datetime.now(timezone.utc)
    delta = resolution_dt - now

    resolves_at = resolution_dt.isoformat()

    if delta.total_seconds() <= 0:
        resolves_in = "OVERDUE"
    elif delta.days > 0:
        resolves_in = f"{delta.days}d {delta.seconds // 3600}h"
    else:
        hours = delta.seconds // 3600
        minutes = (delta.seconds % 3600) // 60
        if hours > 0:
            resolves_in = f"{hours}h {minutes:02d}m"
        else:
            resolves_in = f"{minutes}m"

    return resolves_at, resolves_in


async def broadcast(data: dict) -> None:
    """Send state update to all connected WebSocket clients + cache in Redis."""
    msg = json.dumps(data, default=str)
    # Cache in Redis for instant /api/state responses
    try:
        from weather_edge.live_state import cache_dashboard_state
        cache_dashboard_state(data)
    except Exception:
        pass
    disconnected = []
    for ws in connected_websockets:
        try:
            await ws.send_text(msg)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        connected_websockets.remove(ws)


async def run_dashboard_cycle(run_ai: bool = True) -> None:
    """Run one data cycle and update global state.

    Args:
        run_ai: If True, run Claude + Gemini reasoning. False for sniper-triggered cycles.
    """
    # Prevent concurrent cycles from corrupting paper_trader state
    if _cycle_lock.locked():
        logger.info("Cycle already running, skipping this trigger")
        return
    async with _cycle_lock:
        await _run_dashboard_cycle_inner(run_ai)


async def _run_dashboard_cycle_inner(run_ai: bool = True) -> None:
    """Inner cycle logic, always called under _cycle_lock."""
    global latest_state

    from weather_edge.analysis.consensus import compute_consensus

    today = date.today()
    tomorrow = today + timedelta(days=1)
    target_dates = [today, tomorrow]

    # Track competitor performance
    await competitor_tracker.update_all()

    # Run the core cycle, paper and live are independent systems
    # Either can be None/disabled and the other keeps working
    _paper = paper_trader if settings.paper_mode else None
    _live = live_executor if settings.live_mode else None
    all_signals, forecast_cache, city_volume = await run_cycle(
        _paper, target_dates, run_ai_reasoning=run_ai,
        live_executor=_live,
        store=paper_trader.store,
    )

    # Cache model probs for fast exit loop
    global _cached_model_probs
    for sig in all_signals:
        _cached_model_probs[sig.market_id] = sig.model_prob

    # Build city data from the forecast cache (no re-fetching!)
    city_data = {}
    for city_id in City:
        city_config = CITIES[city_id]
        # Get forecast trend from Redis for sparkline (newest-first, reverse for display)
        trend = []
        try:
            from weather_edge.live_state import get_json
            trend_data = get_json(f"trend:{city_id.value}")
            if trend_data and isinstance(trend_data, list):
                trend = list(reversed(trend_data))  # chronological order for sparkline
        except Exception:
            pass

        city_info = {
            "name": city_config.name,
            "icao": city_config.icao,
            "forecasts": {},
            "models": {},
            "fetched_at": None,
            "trend": trend,
        }

        # Use cached forecasts from run_cycle
        forecasts = forecast_cache.get((city_id, tomorrow), [])
        if not forecasts:
            forecasts = forecast_cache.get((city_id, today), [])
        if forecasts:
            oldest_fetch = min((f.fetched_at for f in forecasts if hasattr(f, 'fetched_at')), default=None)
            if oldest_fetch:
                city_info["fetched_at"] = oldest_fetch.isoformat()

            for f in forecasts:
                if f.temp_max_c is not None:
                    city_info["models"][f.model_name] = {
                        "temp_max_c": f.temp_max_c,
                        "temp_min_c": f.temp_min_c,
                        "precip_mm": f.precip_sum_mm,
                        "snow_cm": f.snow_sum_cm,
                    }

            consensus = compute_consensus(city_id, str(tomorrow), "temp_max_c", forecasts)
            if consensus:
                city_info["forecasts"]["temp_max_c"] = {
                    "mean": consensus.weighted_mean,
                    "confidence": consensus.confidence,
                }

        # Add volume data from market discovery
        vol = city_volume.get(city_id.value, {})
        city_info["volume_24h"] = vol.get("volume_24h", 0)
        city_info["liquidity"] = vol.get("liquidity", 0)
        city_info["market_count"] = vol.get("markets", 0)

        city_data[city_id.value] = city_info

    # Build trade log and stats, from paper or live depending on mode
    trade_log = []
    streak = 0
    best_trade = 0.0
    capital_at_risk = 0.0

    if settings.paper_mode:
        for t in sorted(paper_trader.trades, key=lambda x: x.placed_at, reverse=True)[:100]:
            resolves_at, resolves_in = _compute_resolution_time(t)
            desc = t.description or ""
            if "[PENNY]" in desc:
                tier = "tail"
            elif "[TODAY]" in desc or "[TOMORROW]" in desc:
                tier = "core"
            else:
                tier = "core"
            trade_log.append({
                "side": t.side,
                "city": t.city_id.upper() if isinstance(t.city_id, str) else t.city_id,
                "description": desc,
                "size": t.size_usd,
                "pnl": t.pnl,
                "time": t.placed_at.strftime("%H:%M:%S"),
                "status": t.status.value,
                "tier": tier,
                "resolves_at": resolves_at,
                "resolves_in": resolves_in,
            })
        for t in sorted(paper_trader.closed_trades, key=lambda x: x.placed_at, reverse=True):
            if t.status.value == "won":
                streak += 1
            else:
                break
        best_trade = max(
            (t.pnl for t in paper_trader.trades if t.pnl is not None),
            default=0.0,
        )
        capital_at_risk = sum(t.size_usd for t in paper_trader.open_trades)

    # Build signal dicts for analytics
    signal_dicts = [
        {
            "market_id": s.market_id,
            "city_id": s.city_id,
            "side": s.recommended_side.value,
            "model_prob": s.model_prob,
            "market_prob": s.market_prob,
            "edge": s.edge,
            "edge_pct": s.edge_pct,
            "size": s.recommended_size,
            "tier": s.confidence_tier.value,
            "confidence": s.model_confidence,
            "description": s.description[:60],
            "spread": getattr(s, "spread", 0.0),
        }
        for s in sorted(all_signals, key=lambda x: abs(x.edge), reverse=True)
    ]

    # Compute execution analytics (local computation, no API calls)
    try:
        exec_analytics = compute_execution_analytics(
            trades=paper_trader.trades,
            signals=signal_dicts,
            cycle_count=paper_trader.store.cycle_count if hasattr(paper_trader.store, 'cycle_count') else 0,
            bankroll=settings.bankroll,
            capital_at_risk=capital_at_risk,
        )
    except Exception:
        logger.warning("Execution analytics computation failed", exc_info=True)
        exec_analytics = {}

    # Compute correlation matrix (local computation, no API calls)
    try:
        correlation_data = compute_correlation_matrix(city_data)
    except Exception:
        logger.warning("Correlation matrix computation failed", exc_info=True)
        correlation_data = {"cities": [], "matrix": [], "pairs": []}

    # === BROADCAST TRADING DATA IMMEDIATELY ===
    # Don't wait for weather alerts, get the trading dashboard up fast
    latest_state = {
        "cities": city_data,
        "signals": signal_dicts,
        "consensus": {},
        "trade_log": trade_log,
        "bankroll": settings.bankroll,
        "capital_at_risk": round(capital_at_risk, 2),
        "session_pnl": paper_trader.total_pnl if settings.paper_mode else 0.0,
        "total_pnl": paper_trader.total_pnl if settings.paper_mode else 0.0,
        "win_rate": (paper_trader.win_rate * 100) if settings.paper_mode else 0.0,
        "open_positions": len(paper_trader.open_trades) if settings.paper_mode else 0,
        "best_trade": best_trade,
        "streak": streak,
        "trading_active": trading_active,
        "sniper_events": [
            {
                "model": e.model.value,
                "shift": round(e.shift_c, 1),
                "city": e.city_id,
                "time": e.detected_at.strftime("%H:%M:%S"),
            }
            for e in sniper.recent_events
        ],
        "competitors": competitor_tracker.comparison_summary(
            paper_trader.total_pnl if settings.paper_mode else 0,
            len(paper_trader.trades) if settings.paper_mode else 0,
        ),
        "cycle_count": paper_trader.store.increment_cycle(),
        "last_update": datetime.now(timezone.utc).isoformat(),
        "weather_alerts": [],
        "correlation_matrix": correlation_data,
        "execution_analytics": exec_analytics,
        "claude_decisions": get_decisions(),
        "ai_adherence": compute_adherence(
            paper_trader.closed_trades if settings.paper_mode else [], get_decisions(),
        ).to_dict(),
    }

    # Pool breakdown calculation
    pools = {}
    for strat in ["core", "penny", "spread", "exit"]:
        p_pnl, p_at_risk, p_count = 0.0, 0.0, 0
        if settings.paper_mode:
            p_trades = [t for t in paper_trader.trades if getattr(t, "strategy", "core") == strat]
            p_won = [t for t in p_trades if t.status == TradeStatus.WON]
            p_lost = [t for t in p_trades if t.status == TradeStatus.LOST]
            p_pnl = sum(t.pnl or 0 for t in p_won + p_lost)
            p_at_risk = sum(t.size_usd for t in p_trades if t.status == TradeStatus.OPEN)
            p_count = len(p_trades)

        pools[strat] = {
            "paper_pnl": round(p_pnl, 2),
            "paper_at_risk": round(p_at_risk, 2),
            "live_pnl": 0.0,
            "live_at_risk": 0.0,
            "total_trades": p_count,
        }
    
    latest_state["pools"] = pools

    # Include kill switch state in every broadcast
    try:
        from weather_edge.trading.kill_switch import get_kill_switch_state
        latest_state["kill_switch"] = get_kill_switch_state()
    except Exception:
        pass

    # Include live trading data, Polymarket APIs are source of truth
    if live_executor and not live_executor.dry_run:
        try:
            from weather_edge.trading.portfolio_sync import fetch_polymarket_state
            # Proxy wallet is the on-chain address that holds positions
            proxy = "0xe23940d70793b441c9f949741daa65289947fadb"
            pm = await fetch_polymarket_state(live_executor, proxy)

            # Build position list for dashboard
            import re as _re
            position_list = []
            for p in pm["positions"]:
                size = float(p.get("size", 0))
                if size <= 0:
                    continue
                cur_val = float(p.get("currentValue", 0))
                init_val = float(p.get("initialValue", 0))
                title = p.get("title", "")
                city_match = _re.search(r"in (\w[\w\s]+?) (?:be |on )", title)
                city_name = city_match.group(1) if city_match else ""
                pnl = float(p.get("cashPnl", 0))
                position_list.append({
                    "city": city_name.upper()[:3] if city_name else "",
                    "side": p.get("outcome", ""),
                    "shares": round(size, 1),
                    "avg_price": round(init_val / size, 3) if size > 0 else 0,
                    "cost_basis": round(init_val, 2),
                    "current_value": round(cur_val, 2),
                    "pnl": round(pnl, 2),
                    "description": title,
                })

            # Build open orders list for dashboard
            open_order_list = []
            for o in pm["open_orders"]:
                open_order_list.append({
                    "id": str(o.get("id", ""))[:16],
                    "side": o.get("side", ""),
                    "price": float(o.get("price", 0)),
                    "size": float(o.get("original_size", 0)),
                    "filled": float(o.get("size_matched", 0)),
                    "status": o.get("status", ""),
                })

            # Build trade log from positions (has resolution status) + activity (has timestamps)
            trade_log_live = []

            # Build activity lookup: conditionId → earliest timestamp
            import re as _re
            activity_times = {}
            for a in pm.get("activity", []):
                cid = a.get("conditionId", "")
                ts = a.get("timestamp", 0)
                if cid and ts:
                    if cid not in activity_times or ts < activity_times[cid]:
                        activity_times[cid] = ts

            # Positions → blotter entries with won/lost/open status
            for p in pm["positions"]:
                size = float(p.get("size", 0))
                cur_val = float(p.get("currentValue", 0))
                init_val = float(p.get("initialValue", 0))
                avg_price = float(p.get("avgPrice", 0))
                cash_pnl = float(p.get("cashPnl", 0))
                redeemable = p.get("redeemable", False)
                title = p.get("title", "")
                outcome = p.get("outcome", "")
                cid = p.get("conditionId", "")

                # Determine status
                if redeemable and cur_val == 0:
                    status = "lost"
                elif redeemable:
                    status = "won"
                elif size == 0:
                    status = "sold"
                else:
                    status = "open"

                # Determine tier from entry price
                tier = "tail" if avg_price <= 0.06 else "core"

                # Extract city from title: "Will the highest temperature in Dallas be..."
                city_match = _re.search(r"in ([A-Z][a-z ]+?) (?:be |on )", title)
                city_name = city_match.group(1).strip() if city_match else ""

                # Extract resolution date from title: "on April 1?"
                resolves_in = ""
                resolves_seconds = 999999  # for sorting (large = far away)
                date_match = _re.search(r"on (\w+ \d+)\??$", title)
                if date_match and status == "open":
                    try:
                        from dateutil.parser import parse as _parse_date
                        now = datetime.now(timezone.utc)
                        target_date = _parse_date(date_match.group(1) + f" {now.year}").date()
                        # If parsed date is far in the past, it's probably next year
                        if (now.date() - target_date).days > 180:
                            target_date = _parse_date(date_match.group(1) + f" {now.year + 1}").date()
                        # Resolution = target date + 1 day + 2h buffer
                        resolution_dt = datetime(
                            target_date.year, target_date.month, target_date.day,
                            2, 0, 0, tzinfo=timezone.utc,
                        ) + timedelta(days=1)
                        delta = resolution_dt - now
                        resolves_seconds = delta.total_seconds()
                        if resolves_seconds <= 0:
                            resolves_in = "OVERDUE"
                        elif delta.days > 0:
                            resolves_in = f"{delta.days}d {delta.seconds // 3600}h"
                        else:
                            hours = delta.seconds // 3600
                            minutes = (delta.seconds % 3600) // 60
                            resolves_in = f"{hours}h {minutes:02d}m" if hours > 0 else f"{minutes}m"
                    except Exception:
                        pass

                # Timestamp from activity API
                entry_ts = activity_times.get(cid, 0)
                time_str = ""
                if entry_ts:
                    try:
                        time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).strftime("%H:%M:%S")
                    except Exception:
                        pass

                trade_log_live.append({
                    "side": outcome or "YES",
                    "city": city_name,
                    "description": title,
                    "size": round(init_val, 2),
                    "pnl": round(cash_pnl, 2),
                    "time": time_str,
                    "status": status,
                    "tier": tier,
                    "price": avg_price,
                    "shares": size,
                    "outcome": outcome,
                    "current_value": round(cur_val, 2),
                    "resolves_in": resolves_in,
                    "_sort_seconds": resolves_seconds,
                })

            # Sort: open positions by resolution time (soonest first), then won, then lost
            status_order = {"open": 0, "won": 1, "lost": 2, "sold": 3}
            trade_log_live.sort(key=lambda t: (
                status_order.get(t["status"], 9),
                t.get("_sort_seconds", 999999),
            ))

            latest_state["live"] = {
                "enabled": True,
                "balance": pm["balance"],
                "portfolio_value": pm["portfolio_value"],
                "available_cash": pm["balance"],
                "positions": position_list,
                "position_count": pm["position_count"],
                "open_orders": open_order_list,
                "open_order_count": len(pm["open_orders"]),
                "trade_log": trade_log_live,
                "trade_count": len(trade_log_live),
                "cost_basis": pm["cost_basis"],
                "market_value": pm["market_value"],
                "capital_at_risk": pm["market_value"],
                "pnl": pm["total_pnl"],
                "realized_pnl": pm["realized_pnl"],
                "unrealized_pnl": pm["unrealized_pnl"],
                "total_fees": 0,
                "max_shares": live_executor.max_shares,
            }
        except Exception:
            latest_state["live"] = {"enabled": True, "balance": 0, "portfolio_value": 0}
    else:
        latest_state["live"] = {"enabled": False}

    await broadcast(latest_state)
    logger.info("Dashboard cycle complete, next in %dm", settings.fetch_interval_minutes)

    # === DEFERRED: Fetch weather alerts in background (doesn't block trading) ===
    try:
        weather_alerts = await fetch_all_alerts(forecast_cache)
        latest_state["weather_alerts"] = weather_alerts
        await broadcast(latest_state)
    except Exception:
        logger.warning("Weather alerts fetch failed", exc_info=True)


async def background_loop() -> None:
    """Background task that runs the data cycle periodically."""
    while True:
        if trading_active:
            try:
                await run_dashboard_cycle()
                logger.info("Dashboard cycle complete, next in %dm", settings.fetch_interval_minutes)
            except Exception:
                logger.exception("Dashboard cycle failed")
        await asyncio.sleep(settings.fetch_interval_minutes * 60)


async def sniper_loop() -> None:
    """Sniper runs independently, lightweight metadata probes every 3 min."""
    # Wire the sniper callback to trigger a full dashboard cycle (no AI on sniper)
    async def snipe_trigger():
        if trading_active:
            logger.warning("SNIPER TRIGGERED, running immediate cycle (no AI reasoning)")

            # === GOLDEN WINDOW FLUSH ===
            # Model just dropped, cancel all resting orders priced on old data
            if live_executor and not live_executor.dry_run:
                try:
                    cancelled = await live_executor.cancel_all_orders()
                    if cancelled:
                        # Mark cancelled orders in SQLite
                        open_trades = paper_trader.store.get_open_live_trades()
                        for t in open_trades:
                            oid = t.get("order_id")
                            if oid:
                                paper_trader.store.cancel_live_trade(oid)
                        logger.warning(
                            "GOLDEN WINDOW FLUSH: cancelled %d stale orders, fresh model data incoming",
                            cancelled,
                        )
                except Exception:
                    logger.warning("Golden window flush failed", exc_info=True)

            try:
                await run_dashboard_cycle(run_ai=False)
            except Exception:
                logger.exception("Sniper-triggered cycle failed")

    sniper.set_callback(snipe_trigger)
    await sniper.run_sniper_loop(poll_interval_seconds=180)


async def daily_report_loop() -> None:
    """Save a daily report at midnight UTC."""
    from weather_edge.analysis.daily_report import save_daily_report
    while True:
        now = datetime.now(timezone.utc)
        # Next midnight UTC
        tomorrow = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=5, microsecond=0
        )
        wait_seconds = (tomorrow - now).total_seconds()
        await asyncio.sleep(wait_seconds)
        try:
            save_daily_report(paper_trader, paper_trader.store)
        except Exception:
            logger.exception("Daily report failed")


async def heartbeat_loop() -> None:
    """Send heartbeat every 30s to keep Polymarket session alive."""
    while True:
        if trading_active and live_executor and not live_executor.dry_run:
            await live_executor.send_heartbeat()
        await asyncio.sleep(30)


async def stale_order_loop() -> None:
    """Safety net: cancel truly abandoned orders (4 hour hard cap).

    Smart cleanup happens via cancel-and-replace each cycle + golden window flush.
    This is the backstop for orders that somehow survive both.
    """
    while True:
        await asyncio.sleep(1800)  # Check every 30 minutes
        if trading_active and live_executor and not live_executor.dry_run:
            try:
                cancelled = await live_executor.cancel_stale_orders(max_age_seconds=14400)  # 4 hours
                if cancelled:
                    # Also mark them cancelled in SQLite
                    open_trades = paper_trader.store.get_open_live_trades()
                    for t in open_trades:
                        oid = t.get("order_id")
                        if oid:
                            paper_trader.store.cancel_live_trade(oid)
                    logger.info("STALE CLEANUP: cancelled %d orders (>4h old)", cancelled)
            except Exception:
                logger.debug("Stale order cleanup failed", exc_info=True)


EMERGENCY_EDGE_THRESHOLD = -0.15  # Exit immediately if edge < -15%
FAST_EXIT_INTERVAL = 120  # Check every 2 minutes


async def fast_exit_loop() -> None:
    """Lightweight 5-minute loop: price-only check for emergency exits.

    Uses cached model probs from the last full cycle. No weather fetching,
    no AI review. If edge has inverted past -15%, sell immediately as taker.
    """
    await asyncio.sleep(60)  # Let first full cycle run

    while True:
        await asyncio.sleep(FAST_EXIT_INTERVAL)
        if not trading_active or not live_executor or live_executor.dry_run:
            continue
        if not _cached_model_probs:
            continue

        try:
            # Sync positions from exchange first, keeps dashboard fresh
            # and prevents trading on stale balance data
            from weather_edge.trading.portfolio_sync import sync_portfolio
            try:
                await sync_portfolio(live_executor, paper_trader.store)
            except Exception:
                logger.debug("Fast exit portfolio sync failed", exc_info=True)

            # Update live state from Polymarket (source of truth)
            try:
                from weather_edge.trading.portfolio_sync import fetch_polymarket_state
                proxy = "0xe23940d70793b441c9f949741daa65289947fadb"
                pm = await fetch_polymarket_state(live_executor, proxy)
                if "live" in latest_state:
                    latest_state["live"]["balance"] = pm["balance"]
                    latest_state["live"]["available_cash"] = pm["balance"]
                    latest_state["live"]["portfolio_value"] = pm["portfolio_value"]
                    latest_state["live"]["market_value"] = pm["market_value"]
                    latest_state["live"]["cost_basis"] = pm["cost_basis"]
                    latest_state["live"]["capital_at_risk"] = pm["market_value"]
                    latest_state["live"]["pnl"] = pm["total_pnl"]
                    latest_state["live"]["position_count"] = pm["position_count"]
                    await broadcast(latest_state)
            except Exception:
                pass

            positions = paper_trader.store.get_positions()
            if not positions:
                continue

            # Fetch current prices for our positions from Gamma API
            condition_ids = [p["condition_id"] for p in positions]
            current_prices = await _fetch_position_prices(condition_ids)
            if not current_prices:
                continue

            for pos in positions:
                cid = pos["condition_id"]
                model_prob = _cached_model_probs.get(cid)
                market_price = current_prices.get(cid)
                if model_prob is None or market_price is None:
                    continue

                shares = pos.get("total_shares", 0)
                if shares < 5:  # Below Polymarket minimum
                    continue

                # Calculate edge based on position side
                outcome = pos.get("outcome", "Yes")
                if outcome == "No":
                    # NO position: we profit when event doesn't happen
                    edge = (1 - model_prob) - (1 - market_price)
                else:
                    edge = model_prob - market_price

                if edge < EMERGENCY_EDGE_THRESHOLD:
                    city = pos.get("city_id", "???")
                    asset_id = pos.get("asset_id", "")
                    if not asset_id:
                        continue

                    logger.warning(
                        "EMERGENCY EXIT: %s edge=%.1f%% (threshold=%.0f%%), "
                        "selling %d shares as taker",
                        city, edge * 100, EMERGENCY_EDGE_THRESHOLD * 100, shares,
                    )

                    try:
                        sell_result = await live_executor.place_sell_order(
                            token_id=asset_id,
                            shares=shares,
                            price=market_price,
                            market_id=cid,
                            city_id=city,
                            description=f"EMERGENCY: edge={edge:.1%}",
                            reference_price=market_price,
                            force_taker=True,
                        )
                        if sell_result:
                            logger.warning(
                                "EMERGENCY SELL PLACED: %s %.0f shares @ %.3f, %s",
                                city, shares, market_price, sell_result.order_id,
                            )
                    except Exception as e:
                        logger.error("EMERGENCY SELL FAILED: %s, %s", city, e)

        except Exception:
            logger.error("Fast exit loop failed", exc_info=True)


async def _fetch_position_prices(condition_ids: list[str]) -> dict[str, float]:
    """Fetch current YES prices for a list of condition IDs from Gamma API."""
    prices = {}
    async with httpx.AsyncClient() as client:
        for cid in condition_ids:
            try:
                resp = await client.get(
                    f"{settings.polymarket_gamma_url}/markets",
                    params={"condition_id": cid},
                    timeout=10.0,
                )
                resp.raise_for_status()
                markets = resp.json()
                if markets and isinstance(markets, list):
                    mkt = markets[0]
                    outcome_prices = mkt.get("outcomePrices")
                    if isinstance(outcome_prices, str):
                        import json
                        outcome_prices = json.loads(outcome_prices)
                    if isinstance(outcome_prices, list) and len(outcome_prices) >= 1:
                        prices[cid] = float(outcome_prices[0])
            except (httpx.HTTPError, ValueError, KeyError):
                continue
    return prices


@app.on_event("startup")
async def startup():
    global live_executor

    # Initialize live executor if LIVE_MODE is enabled
    if settings.live_mode:
        from weather_edge.trading.executor import TradeExecutor
        live_executor = TradeExecutor(
            private_key=settings.polymarket_private_key,
            wallet_address=settings.polymarket_wallet,
            api_key=settings.polymarket_api_key,
            api_secret=settings.polymarket_api_secret,
            api_passphrase=settings.polymarket_api_passphrase,
            signature_type=settings.polymarket_signature_type,
            dry_run=False,
            post_only=True,
            max_shares=settings.live_max_shares or None,
        )
        try:
            await live_executor.initialize()
            balance = await live_executor.check_balance()
            logger.warning(
                "LIVE MODE ACTIVE, wallet balance: $%.2f, max_shares: %s",
                balance or 0, settings.live_max_shares or "unlimited",
            )
            # Set initial live state so dashboard shows live view immediately
            latest_state["live"] = {
                "enabled": True,
                "balance": round(balance or 0, 2),
                "portfolio_value": round(balance or 0, 2),
                "available_cash": round(balance or 0, 2),
                "capital_at_risk": 0,
                "positions": [],
                "position_count": 0,
                "trade_log": [],
                "trade_count": 0,
                "pnl": 0,
                "total_fees": 0,
                "max_shares": live_executor.max_shares,
            }
            asyncio.create_task(heartbeat_loop())
            asyncio.create_task(stale_order_loop())
            asyncio.create_task(fast_exit_loop())
            # Fill tracker, poll every 15s for order status updates
            from weather_edge.trading.fill_tracker import poll_fills
            asyncio.create_task(poll_fills(live_executor, interval=15))
        except Exception:
            logger.exception("LIVE EXECUTOR INIT FAILED, falling back to paper only")
            live_executor = None
    else:
        logger.info("Paper mode, live executor not initialized")

    asyncio.create_task(background_loop())
    asyncio.create_task(sniper_loop())
    asyncio.create_task(daily_report_loop())
    # Save initial report on startup
    try:
        from weather_edge.analysis.daily_report import save_daily_report
        save_daily_report(paper_trader, paper_trader.store)
    except Exception:
        pass


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(_index_html)


@app.get("/api/state")
async def api_state():
    # Try Redis cache first (faster, avoids Python dict serialization)
    try:
        from weather_edge.live_state import get_cached_dashboard_state
        cached = get_cached_dashboard_state()
        if cached:
            state = cached
        else:
            state = latest_state
    except Exception:
        state = latest_state

    # Use latest_state["live"] from last cycle broadcast (already has positions/fills)
    # Just refresh the balance which may have changed between cycles
    if live_executor and not live_executor.dry_run:
        try:
            state = dict(state) if not isinstance(state, dict) else state.copy()
            # Use cycle-built live data (has resolution countdowns, city names, etc.)
            if latest_state.get("live", {}).get("enabled"):
                state["live"] = latest_state["live"]
        except Exception as e:
            logger.debug("Live state refresh failed: %s", e)

    return state


@app.post("/api/refresh")
async def api_refresh():
    """Trigger an immediate data refresh (runs in background)."""
    async def _safe_refresh():
        try:
            await run_dashboard_cycle()
        except Exception:
            logger.exception("Manual refresh cycle failed")
    asyncio.create_task(_safe_refresh())
    return {"status": "refreshing"}


@app.post("/api/start")
async def api_start():
    """Start automated trading."""
    global trading_active
    trading_active = True
    latest_state["trading_active"] = True
    logger.info("Trading STARTED")
    # Broadcast the state change immediately so UI updates
    await broadcast(latest_state)
    # Run first cycle in background, don't block the response
    async def _safe_first_cycle():
        try:
            await run_dashboard_cycle()
        except Exception:
            logger.exception("First cycle after START failed")
    asyncio.create_task(_safe_first_cycle())
    return {"status": "started", "trading_active": True}


@app.post("/api/stop")
async def api_stop():
    """Stop automated trading (keeps positions open)."""
    global trading_active
    trading_active = False
    logger.info("Trading STOPPED (positions remain open)")
    latest_state["trading_active"] = False
    await broadcast(latest_state)
    open_count = len(paper_trader.open_trades) if settings.paper_mode else 0
    return {"status": "stopped", "trading_active": False, "open_positions": open_count}


@app.post("/api/kill-switch")
async def api_kill_switch():
    """Emergency kill switch, block all new orders AND cancel open exchange orders."""
    global trading_active
    trading_active = False

    from weather_edge.trading.kill_switch import kill_and_cancel, get_kill_switch_state

    # Get live executor if it exists
    executor = globals().get("live_executor")
    result = await kill_and_cancel(
        executor=executor,
        reason="Manual emergency stop via dashboard",
        triggered_by="api",
    )

    latest_state["trading_active"] = False
    latest_state["kill_switch"] = get_kill_switch_state()
    await broadcast(latest_state)

    return {"status": "killed", **result}


@app.post("/api/kill-switch/reset")
async def api_kill_switch_reset():
    """Reset the kill switch, allow trading to resume."""
    from weather_edge.trading.kill_switch import deactivate_kill_switch, get_kill_switch_state

    deactivate_kill_switch(cleared_by="api")

    latest_state["kill_switch"] = get_kill_switch_state()
    await broadcast(latest_state)

    return {"status": "reset", **get_kill_switch_state()}


@app.get("/api/kill-switch")
async def api_kill_switch_state():
    """Get current kill switch state."""
    from weather_edge.trading.kill_switch import get_kill_switch_state
    return get_kill_switch_state()


@app.post("/api/close-all")
async def api_close_all():
    """Close all open positions at current market prices."""
    global trading_active
    trading_active = False

    # Fetch current prices for open positions
    from weather_edge.fetchers.polymarket import discover_weather_markets
    markets = await discover_weather_markets()
    current_prices = {m.market_id: m.yes_price for m in markets if m.yes_price > 0}

    closed_pnl = 0.0
    if settings.paper_mode:
        closed_pnl = paper_trader.close_all_positions(current_prices)
        logger.info("Closed all positions (paper). P&L from closes: $%.2f", closed_pnl)

    # Also close all live positions on exchange
    live_sells = 0
    if live_executor and not live_executor.dry_run:
        try:
            # Cancel all open orders first
            await live_executor.cancel_all_orders()
            # Place sell orders for all positions
            positions = paper_trader.store.get_positions()
            for pos in positions:
                asset_id = pos.get("asset_id", "")
                shares = pos.get("total_shares", 0)
                city = (pos.get("city_id") or "").upper()
                # Sell at market price (use current YES price from discovery)
                condition_id = pos.get("condition_id", "")
                sell_price = current_prices.get(condition_id, 0.5)
                if asset_id and shares >= 5:
                    try:
                        result = await live_executor.place_sell_order(
                            token_id=asset_id,
                            shares=shares,
                            price=sell_price,
                            market_id=condition_id,
                            city_id=city,
                            description="CLOSE ALL",
                            force_taker=True,
                        )
                        if result:
                            live_sells += 1
                    except Exception as e:
                        logger.error("Failed to sell %s: %s", city, e)
            logger.info("CLOSE ALL: placed %d live sell orders", live_sells)
        except Exception:
            logger.exception("Live close-all failed")

    latest_state["trading_active"] = False
    latest_state["open_positions"] = 0
    latest_state["capital_at_risk"] = 0.0
    latest_state["total_pnl"] = paper_trader.total_pnl if settings.paper_mode else 0.0
    latest_state["session_pnl"] = paper_trader.total_pnl if settings.paper_mode else 0.0
    latest_state["win_rate"] = (paper_trader.win_rate * 100) if settings.paper_mode else 0.0
    await broadcast(latest_state)

    return {
        "status": "closed",
        "positions_closed": len(paper_trader.closed_trades) if settings.paper_mode else live_sells,
        "close_pnl": round(closed_pnl, 2),
        "total_pnl": round(paper_trader.total_pnl, 2) if settings.paper_mode else 0.0,
    }


@app.post("/api/new-session")
async def api_new_session():
    """Reset everything for a new trading session."""
    global trading_active
    trading_active = False
    final_stats = paper_trader.reset_session(settings.bankroll)
    clear_decisions()
    logger.info("New session started. Previous session stats: %s", final_stats)

    latest_state.update({
        "cities": {},
        "signals": [],
        "trade_log": [],
        "bankroll": settings.bankroll,
        "capital_at_risk": 0.0,
        "session_pnl": 0.0,
        "total_pnl": 0.0,
        "win_rate": 0.0,
        "open_positions": 0,
        "best_trade": 0.0,
        "streak": 0,
        "trading_active": False,
        "cycle_count": 0,
        "last_update": None,
        "weather_alerts": [],
        "correlation_matrix": {"cities": [], "matrix": [], "pairs": []},
        "execution_analytics": {},
        "claude_decisions": [],
    })
    await broadcast(latest_state)

    return {"status": "reset", "bankroll": settings.bankroll, "previous_session": final_stats}


@app.get("/api/weather-alerts")
async def api_weather_alerts():
    """Return current weather alerts for all monitored cities."""
    return latest_state.get("weather_alerts", [])


@app.post("/api/backtest")
async def api_backtest(days: int = 7, cities: str | None = None):
    """Run a historical backtest. Optional query params: days (default 7), cities (comma-separated)."""
    from weather_edge.analysis.backtester import run_backtest

    city_list = [c.strip() for c in cities.split(",")] if cities else None
    result = await run_backtest(days=days, cities=city_list)
    return result


@app.get("/api/correlation-matrix")
async def api_correlation_matrix():
    """Return the current city correlation matrix."""
    return latest_state.get("correlation_matrix", {"cities": [], "matrix": [], "pairs": []})


@app.get("/api/execution-analytics")
async def api_execution_analytics():
    """Return execution analytics from paper trading data."""
    return latest_state.get("execution_analytics", {})


@app.get("/api/system-status")
async def api_system_status():
    """Return health status of all external services and API keys."""
    from weather_edge.analysis.service_health import get_service_status
    return get_service_status()


@app.get("/api/daily-report")
async def api_daily_report(date: str | None = None):
    """Get daily report for a specific date (or today)."""
    from weather_edge.analysis.daily_report import generate_daily_report
    return generate_daily_report(paper_trader, report_date=date)


@app.post("/api/daily-report/save")
async def api_save_daily_report():
    """Manually trigger daily report save."""
    from weather_edge.analysis.daily_report import save_daily_report
    report = save_daily_report(paper_trader, paper_trader.store)
    return report


@app.get("/api/daily-reports")
async def api_daily_reports(limit: int = 30):
    """Get historical daily reports."""
    from weather_edge.analysis.daily_report import load_daily_reports
    return load_daily_reports(paper_trader.store, limit=limit)


@app.get("/api/settings")
async def api_get_settings():
    """Return current risk profile and all settings."""
    from weather_edge.analysis.risk_controls import (
        RISK_PROFILES,
        _active_profile_name,
        _circuit_breaker,
    )
    profile = RISK_PROFILES[_active_profile_name]
    return {
        "profile": _active_profile_name,
        "profiles_available": list(RISK_PROFILES.keys()),
        "settings": {
            "drawdown_scale_back_pct": profile.drawdown_scale_back_pct,
            "drawdown_kill_pct": profile.drawdown_kill_pct,
            "scale_back_factor": profile.scale_back_factor,
            "max_group_exposure_pct": profile.max_group_exposure_pct,
            "max_gross_exposure_multiple": profile.max_gross_exposure_multiple,
            "kelly_fraction": profile.kelly_fraction,
            "max_position_pct": profile.max_position_pct,
            "reserve_pct": profile.reserve_pct,
            "penny_max_position": profile.penny_max_position,
            "min_edge": profile.min_edge,
            "fee_alpha_max": profile.fee_alpha_max,
            "compound_factor": profile.compound_factor,
        },
        "circuit_breaker": {
            "high_water_mark": _circuit_breaker.high_water_mark,
            "is_scaled_back": _circuit_breaker.is_scaled_back,
            "is_killed": _circuit_breaker.is_killed,
            "kill_reason": _circuit_breaker.kill_reason,
        },
        "credentials": {
            "anthropic_api_key": bool(settings.anthropic_api_key),
            "gemini_api_key": bool(settings.gemini_api_key),
            "openmeteo_api_key": bool(settings.openmeteo_api_key),
            "gribstream_api_key": bool(settings.gribstream_api_key),
            "polymarket_api_key": bool(settings.polymarket_api_key),
            "polymarket_private_key": bool(settings.polymarket_private_key),
        },
    }


@app.post("/api/settings")
async def api_update_settings(body: dict):
    """Update risk profile or individual settings."""
    from weather_edge.analysis.risk_controls import (
        RISK_PROFILES,
        _circuit_breaker,
        get_active_profile,
        set_active_profile,
    )

    profile_name = body.get("profile")
    if profile_name and profile_name in RISK_PROFILES:
        set_active_profile(profile_name)
        # Persist to SQLite
        try:
            paper_trader.store.set_state(
                "risk_profile", profile_name
            )
        except Exception:
            pass
        logger.info("SETTINGS: Risk profile changed to %s", profile_name)

    # Reset circuit breaker if requested
    if body.get("reset_circuit_breaker"):
        _circuit_breaker.is_scaled_back = False
        _circuit_breaker.is_killed = False
        _circuit_breaker.kill_reason = ""
        nav = (paper_trader.bankroll + paper_trader.total_pnl) if settings.paper_mode else settings.bankroll
        _circuit_breaker.high_water_mark = nav
        logger.info("SETTINGS: Circuit breaker reset, HWM=$%.0f", nav)

    return {"status": "ok", "profile": get_active_profile().name}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_websockets.append(ws)
    # Send current state immediately
    await ws.send_text(json.dumps(latest_state, default=str))
    try:
        while True:
            # Keep connection alive, listen for commands
            data = await ws.receive_text()
            if data == "refresh":
                await run_dashboard_cycle()
    except WebSocketDisconnect:
        if ws in connected_websockets:
            connected_websockets.remove(ws)
