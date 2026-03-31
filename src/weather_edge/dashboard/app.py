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
    _paper = paper_trader if settings.live_mode is False or not live_executor else paper_trader
    all_signals, forecast_cache, city_volume = await run_cycle(
        _paper, target_dates, run_ai_reasoning=run_ai,
        live_executor=live_executor,
        store=paper_trader.store,
    )

    # Build city data from the forecast cache (no re-fetching!)
    city_data = {}
    for city_id in City:
        city_config = CITIES[city_id]
        city_info = {
            "name": city_config.name,
            "icao": city_config.icao,
            "forecasts": {},
            "models": {},
        }

        # Use cached forecasts from run_cycle
        forecasts = forecast_cache.get((city_id, tomorrow), [])
        if not forecasts:
            forecasts = forecast_cache.get((city_id, today), [])
        if forecasts:
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

    # Build trade log entries (with resolution countdown for open trades)
    trade_log = []
    for t in sorted(paper_trader.trades, key=lambda x: x.placed_at, reverse=True)[:100]:
        resolves_at, resolves_in = _compute_resolution_time(t)
        # Derive tier from description tag
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

    # Calculate streak
    streak = 0
    for t in sorted(paper_trader.closed_trades, key=lambda x: x.placed_at, reverse=True):
        if t.status.value == "won":
            streak += 1
        else:
            break

    best_trade = max(
        (t.pnl for t in paper_trader.trades if t.pnl is not None),
        default=0.0,
    )

    # Capital at risk = sum of all open position sizes
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
        "session_pnl": paper_trader.total_pnl,
        "total_pnl": paper_trader.total_pnl,
        "win_rate": paper_trader.win_rate * 100,
        "open_positions": len(paper_trader.open_trades),
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
            paper_trader.total_pnl, len(paper_trader.trades)
        ),
        "cycle_count": paper_trader.store.increment_cycle(),
        "last_update": datetime.now(timezone.utc).isoformat(),
        "weather_alerts": [],
        "correlation_matrix": correlation_data,
        "execution_analytics": exec_analytics,
        "claude_decisions": get_decisions(),
        "ai_adherence": compute_adherence(paper_trader.closed_trades, get_decisions()).to_dict(),
    }

    # Pool breakdown calculation
    pools = {}
    for strat in ["core", "penny", "spread", "exit"]:
        # Paper stats
        p_trades = [t for t in paper_trader.trades if getattr(t, "strategy", "core") == strat]
        p_won = [t for t in p_trades if t.status == TradeStatus.WON]
        p_lost = [t for t in p_trades if t.status == TradeStatus.LOST]
        p_pnl = sum(t.pnl or 0 for t in p_won + p_lost)
        p_at_risk = sum(t.size_usd for t in p_trades if t.status == TradeStatus.OPEN)
        
        pools[strat] = {
            "paper_pnl": round(p_pnl, 2),
            "paper_at_risk": round(p_at_risk, 2),
            "live_pnl": 0.0,
            "live_at_risk": 0.0,
            "total_trades": len(p_trades)
        }
    
    latest_state["pools"] = pools

    # Include kill switch state in every broadcast
    try:
        from weather_edge.trading.kill_switch import get_kill_switch_state
        latest_state["kill_switch"] = get_kill_switch_state()
    except Exception:
        pass

    # Include live trading data from positions + fills (exchange truth)
    if live_executor and not live_executor.dry_run:
        try:
            live_balance = await live_executor.check_balance()
            positions = paper_trader.store.get_positions()
            fills = paper_trader.store.get_fills(limit=100)
            portfolio = paper_trader.store.get_portfolio_summary()

            # Update live pool stats
            for strat in ["core", "penny", "spread", "exit"]:
                l_trades = paper_trader.store.conn.execute(
                    "SELECT pnl, size_usd, status FROM live_trades WHERE strategy = ?",
                    (strat,)
                ).fetchall()
                l_pnl = sum(r[0] or 0 for r in l_trades if r[2] in ("won", "lost"))
                l_at_risk = sum(r[1] or 0 for r in l_trades if r[2] in ("open", "partial"))
                if strat in latest_state["pools"]:
                    latest_state["pools"][strat]["live_pnl"] = round(l_pnl, 2)
                    latest_state["pools"][strat]["live_at_risk"] = round(l_at_risk, 2)

            # Build trade log from fills with resolution countdown
            import re as _re
            live_trade_log = []
            for f in fills:
                filled_at = f.get("filled_at", "")
                try:
                    time_str = filled_at.split("T")[1][:8] if "T" in filled_at else filled_at[-8:]
                except Exception:
                    time_str = filled_at

                # Extract target date from description for resolution countdown
                resolves_at = None
                resolves_in = ""
                desc = f.get("description") or ""
                city_id_str = (f.get("city_id") or "").lower()
                date_match = _re.search(r"on (\w+ \d+)\??$", desc)
                if date_match and not f.get("is_settled"):
                    try:
                        from dateutil.parser import parse as _parse_date
                        target_date = _parse_date(date_match.group(1) + " 2026").date()
                        # Look up city timezone for resolution time
                        tz_name = None
                        for ce, cc in CITIES.items():
                            if ce.value == city_id_str:
                                tz_name = cc.timezone
                                break
                        if tz_name and _HAS_ZONEINFO:
                            tz = zoneinfo.ZoneInfo(tz_name)
                            local_midnight = datetime(
                                target_date.year, target_date.month, target_date.day,
                                0, 0, 0, tzinfo=tz,
                            ) + timedelta(days=1, hours=2)
                            resolution_dt = local_midnight.astimezone(timezone.utc)
                        else:
                            resolution_dt = datetime(
                                target_date.year, target_date.month, target_date.day,
                                2, 0, 0, tzinfo=timezone.utc,
                            ) + timedelta(days=1)
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
                            resolves_in = f"{hours}h {minutes:02d}m" if hours > 0 else f"{minutes}m"
                    except Exception:
                        pass

                live_trade_log.append({
                    "side": f.get("side", ""),
                    "city": (f.get("city_id") or "").upper(),
                    "description": desc,
                    "size": round(f.get("size", 0) * f.get("price", 0), 2),
                    "pnl": None,
                    "time": time_str,
                    "status": "settled" if f.get("is_settled") else "filled",
                    "tier": "core",
                    "price": f.get("price", 0),
                    "shares": f.get("size", 0),
                    "outcome": f.get("outcome", ""),
                    "is_maker": f.get("is_maker", 1),
                    "resolves_at": resolves_at,
                    "resolves_in": resolves_in,
                })

            # Build positions list
            position_list = []
            for p in positions:
                position_list.append({
                    "city": (p.get("city_id") or "").upper(),
                    "side": p.get("outcome") or p.get("side", ""),
                    "shares": round(p.get("total_shares", 0), 1),
                    "avg_price": round(p.get("avg_price", 0), 3),
                    "cost_basis": round(p.get("cost_basis", 0), 2),
                    "description": p.get("description", ""),
                })

            deployed = portfolio.get("total_deployed", 0)
            portfolio_value = round((live_balance or 0) + deployed, 2)

            latest_state["live"] = {
                "enabled": True,
                "balance": round(live_balance or 0, 2),
                "portfolio_value": portfolio_value,
                "available_cash": round(live_balance or 0, 2),
                "positions": position_list,
                "position_count": portfolio.get("position_count", 0),
                "trade_log": live_trade_log,
                "trade_count": len(fills),
                "capital_at_risk": round(deployed, 2),
                "pnl": 0,
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
    return {"status": "stopped", "trading_active": False, "open_positions": len(paper_trader.open_trades)}


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
                            city_id=city,
                            description="CLOSE ALL",
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
    latest_state["total_pnl"] = paper_trader.total_pnl
    latest_state["session_pnl"] = paper_trader.total_pnl
    latest_state["win_rate"] = paper_trader.win_rate * 100
    await broadcast(latest_state)

    return {
        "status": "closed",
        "positions_closed": len(paper_trader.closed_trades),
        "close_pnl": round(closed_pnl, 2),
        "total_pnl": round(paper_trader.total_pnl, 2),
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
        nav = paper_trader.bankroll + paper_trader.total_pnl
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
