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

from weather_edge.analysis.claude_reasoning import get_decisions, clear_decisions
from weather_edge.analysis.competitor_tracker import CompetitorTracker
from weather_edge.analysis.correlation_matrix import compute_correlation_matrix
from weather_edge.analysis.execution_analytics import compute_execution_analytics
from weather_edge.analysis.resolver import _extract_target_date_from_trade
from weather_edge.analysis.sniper import ModelSniper
from weather_edge.analysis.weather_alerts import fetch_all_alerts
from weather_edge.config import CITIES, settings

from weather_edge.models.enums import City
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
    "cycle_count": 0,
    "last_update": None,
    "weather_alerts": [],
    "correlation_matrix": {"cities": [], "matrix": [], "pairs": []},
    "execution_analytics": {},
    "claude_decisions": [],
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
    """Send state update to all connected WebSocket clients."""
    msg = json.dumps(data, default=str)
    disconnected = []
    for ws in connected_websockets:
        try:
            await ws.send_text(msg)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        connected_websockets.remove(ws)


async def run_dashboard_cycle() -> None:
    """Run one data cycle and update global state."""
    global latest_state

    from weather_edge.analysis.consensus import compute_consensus

    today = date.today()
    tomorrow = today + timedelta(days=1)
    target_dates = [today, tomorrow]

    # Track competitor performance
    await competitor_tracker.update_all()

    # Run the core cycle (fetches markets, forecasts, computes signals, places paper trades)
    all_signals, forecast_cache = await run_cycle(paper_trader, target_dates)

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

        city_data[city_id.value] = city_info

    # Build trade log entries (with resolution countdown for open trades)
    trade_log = []
    for t in sorted(paper_trader.trades, key=lambda x: x.placed_at, reverse=True)[:20]:
        resolves_at, resolves_in = _compute_resolution_time(t)
        trade_log.append({
            "side": t.side,
            "city": t.city_id.upper() if isinstance(t.city_id, str) else t.city_id,
            "description": t.description[:50] if t.description else "",
            "size": t.size_usd,
            "pnl": t.pnl,
            "time": t.placed_at.strftime("%H:%M:%S"),
            "status": t.status.value,
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
    }

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
    """Sniper runs independently, lightweight metadata probes every 60s."""
    # Wire the sniper callback to trigger a full dashboard cycle
    async def snipe_trigger():
        if trading_active:
            logger.warning("SNIPER TRIGGERED, running immediate cycle")
            try:
                await run_dashboard_cycle()
            except Exception:
                logger.exception("Sniper-triggered cycle failed")

    sniper.set_callback(snipe_trigger)
    await sniper.run_sniper_loop(poll_interval_seconds=30)


@app.on_event("startup")
async def startup():
    asyncio.create_task(background_loop())
    asyncio.create_task(sniper_loop())


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(_index_html)


@app.get("/api/state")
async def api_state():
    return latest_state


@app.post("/api/refresh")
async def api_refresh():
    """Trigger an immediate data refresh."""
    await run_dashboard_cycle()
    return {"status": "ok", "cycle": latest_state["cycle_count"]}


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
    logger.info("Closed all positions. P&L from closes: $%.2f", closed_pnl)

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
