"""FastAPI web dashboard, dark terminal aesthetic matching ColdMath's Claude Trader."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from weather_edge.analysis.competitor_tracker import CompetitorTracker
from weather_edge.analysis.sniper import ModelSniper
from weather_edge.config import CITIES, settings
from weather_edge.fetchers.openmeteo import fetch_city_forecasts
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
}
connected_websockets: list[WebSocket] = []


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
    all_signals = await run_cycle(paper_trader, target_dates)

    # Build city data for the dashboard display
    city_data = {}
    for city_id in City:
        city_config = CITIES[city_id]
        city_info = {
            "name": city_config.name,
            "icao": city_config.icao,
            "forecasts": {},
            "models": {},
        }

        # Fetch tomorrow's forecasts for display
        forecasts = await fetch_city_forecasts(city_id, tomorrow)
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

    # Build trade log entries
    trade_log = []
    for t in sorted(paper_trader.trades, key=lambda x: x.placed_at, reverse=True)[:20]:
        trade_log.append({
            "side": t.side,
            "city": t.city_id.upper(),
            "description": t.description[:50] if t.description else "",
            "size": t.size_usd,
            "pnl": t.pnl,
            "time": t.placed_at.strftime("%H:%M:%S"),
            "status": t.status.value,
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

    latest_state = {
        "cities": city_data,
        "signals": [
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
        ],
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
    }

    await broadcast(latest_state)


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
    """Sniper runs independently, probes every 2 min even when trading is active."""
    # Wire the sniper callback to trigger a full dashboard cycle
    async def snipe_trigger():
        if trading_active:
            logger.warning("SNIPER TRIGGERED, running immediate cycle")
            await run_dashboard_cycle()

    sniper.set_callback(snipe_trigger)
    await sniper.run_sniper_loop(poll_interval_seconds=120)


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
    logger.info("Trading STARTED")
    # Run first cycle immediately
    await run_dashboard_cycle()
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
    })
    await broadcast(latest_state)

    return {"status": "reset", "bankroll": settings.bankroll, "previous_session": final_stats}


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
