"""Daily report generator, end-of-day snapshot for historical reference.

Runs at midnight UTC (or on demand). Captures everything you'd want to
review the next morning: P&L, trades, AI performance, API health, config.
Stored in SQLite for trend tracking.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def generate_daily_report(
    paper_trader,
    report_date: str | None = None,
) -> dict:
    """Generate a comprehensive daily report.

    Args:
        paper_trader: PersistentPaperTrader instance
        report_date: ISO date string (default: today UTC)

    Returns:
        Report dict suitable for JSON storage and display.
    """
    now = datetime.now(timezone.utc)
    if report_date is None:
        report_date = now.strftime("%Y-%m-%d")

    trades = paper_trader.trades
    bankroll = paper_trader.bankroll

    # Overall stats
    won = [t for t in trades if t.status == "won"]
    lost = [t for t in trades if t.status == "lost"]
    open_trades = [t for t in trades if t.status == "open"]
    total_pnl = sum(t.pnl or 0 for t in won + lost)
    portfolio = bankroll + total_pnl
    at_risk = sum(t.size_usd for t in open_trades)
    free_cash = portfolio - at_risk

    win_count = len(won)
    loss_count = len(lost)
    total_resolved = win_count + loss_count
    win_rate = win_count / total_resolved * 100 if total_resolved > 0 else 0
    avg_win = sum(t.pnl or 0 for t in won) / win_count if won else 0
    avg_loss = sum(t.pnl or 0 for t in lost) / loss_count if lost else 0
    payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # Today's trades (resolved today)
    today_won = [t for t in won if t.resolved_at and report_date in str(t.resolved_at)]
    today_lost = [t for t in lost if t.resolved_at and report_date in str(t.resolved_at)]
    today_pnl = sum(t.pnl or 0 for t in today_won + today_lost)

    # Best/worst trades today
    today_all = today_won + today_lost
    best_trade = max(today_all, key=lambda t: t.pnl or 0) if today_all else None
    worst_trade = min(today_all, key=lambda t: t.pnl or 0) if today_all else None

    # City breakdown
    city_pnl: dict[str, float] = {}
    city_counts: dict[str, dict] = {}
    for t in won + lost:
        city = t.city_id or "unknown"
        city_pnl[city] = city_pnl.get(city, 0) + (t.pnl or 0)
        if city not in city_counts:
            city_counts[city] = {"won": 0, "lost": 0}
        if t.status == "won":
            city_counts[city]["won"] += 1
        else:
            city_counts[city]["lost"] += 1

    top_cities = sorted(city_pnl.items(), key=lambda x: x[1], reverse=True)[:5]
    worst_cities = sorted(city_pnl.items(), key=lambda x: x[1])[:5]

    # Pool breakdown
    pool_stats = {}
    for strat in ["core", "penny", "spread", "exit"]:
        s_trades = [t for t in trades if getattr(t, "strategy", "core") == strat]
        s_won = [t for t in s_trades if t.status == "won"]
        s_lost = [t for t in s_trades if t.status == "lost"]
        s_pnl = sum(t.pnl or 0 for t in s_won + s_lost)
        if s_trades:
            pool_stats[strat] = {
                "total_trades": len(s_trades),
                "pnl": round(s_pnl, 2),
                "win_rate": round(len(s_won) / (len(s_won) + len(s_lost)) * 100, 1) if (len(s_won) + len(s_lost)) > 0 else 0,
                "at_risk": round(sum(t.size_usd for t in s_trades if t.status == "open"), 2),
            }

    # AI stats from decision history
    ai_stats = _get_ai_stats()

    # API health
    api_health = _get_api_health()

    # Risk profile
    risk_config = _get_risk_config()

    # Circuit breaker state
    cb_state = _get_circuit_breaker_state(portfolio)

    report = {
        "date": report_date,
        "generated_at": now.isoformat(),

        # Portfolio
        "portfolio_value": round(portfolio, 2),
        "total_pnl": round(total_pnl, 2),
        "return_pct": round(total_pnl / bankroll * 100, 1),
        "bankroll": bankroll,
        "free_cash": round(free_cash, 2),
        "at_risk": round(at_risk, 2),

        # Today's activity
        "today_pnl": round(today_pnl, 2),
        "today_wins": len(today_won),
        "today_losses": len(today_lost),
        "best_trade": {
            "city": best_trade.city_id if best_trade else None,
            "side": best_trade.side if best_trade else None,
            "pnl": round(best_trade.pnl, 2) if best_trade and best_trade.pnl else None,
            "desc": (best_trade.description or "")[:60] if best_trade else None,
        },
        "worst_trade": {
            "city": worst_trade.city_id if worst_trade else None,
            "side": worst_trade.side if worst_trade else None,
            "pnl": round(worst_trade.pnl, 2) if worst_trade and worst_trade.pnl else None,
            "desc": (worst_trade.description or "")[:60] if worst_trade else None,
        },

        # Overall performance
        "total_wins": win_count,
        "total_losses": loss_count,
        "win_rate": round(win_rate, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "payoff_ratio": round(payoff_ratio, 2),
        "open_positions": len(open_trades),

        # City breakdown
        "top_cities": [
            {"city": c, "pnl": round(p, 2),
             "wins": city_counts.get(c, {}).get("won", 0),
             "losses": city_counts.get(c, {}).get("lost", 0)}
            for c, p in top_cities
        ],
        "worst_cities": [
            {"city": c, "pnl": round(p, 2),
             "wins": city_counts.get(c, {}).get("won", 0),
             "losses": city_counts.get(c, {}).get("lost", 0)}
            for c, p in worst_cities
        ],

        # AI performance
        "ai_stats": ai_stats,

        # Pool performance
        "pool_breakdown": pool_stats,

        # Live performance
        "live_stats": _get_live_stats(paper_trader),

        # API health
        "api_health": api_health,

        # Configuration
        "risk_profile": risk_config,
        "circuit_breaker": cb_state,
    }

    return report


def _get_ai_stats() -> dict:
    """Get AI decision stats from the current session."""
    try:
        from weather_edge.analysis.claude_reasoning import get_decisions
        decisions = get_decisions()
        claude_decisions = [d for d in decisions if d.get("source") != "gemini"]
        gemini_decisions = [d for d in decisions if d.get("source") == "gemini"]

        claude_trades = sum(1 for d in claude_decisions if d.get("decision") == "TRADE")
        claude_skips = sum(1 for d in claude_decisions if d.get("decision") == "SKIP")
        gemini_agrees = sum(1 for d in gemini_decisions if d.get("decision") == "AGREE")
        gemini_dissents = sum(1 for d in gemini_decisions if d.get("decision") == "DISSENT")

        return {
            "claude_trades": claude_trades,
            "claude_skips": claude_skips,
            "claude_skip_rate": round(
                claude_skips / (claude_trades + claude_skips) * 100, 1
            ) if (claude_trades + claude_skips) > 0 else 0,
            "gemini_agrees": gemini_agrees,
            "gemini_dissents": gemini_dissents,
            "gemini_dissent_rate": round(
                gemini_dissents / (gemini_agrees + gemini_dissents) * 100, 1
            ) if (gemini_agrees + gemini_dissents) > 0 else 0,
            "model": "Claude=Meteorologist, Gemini=Quant (v2 2026-03-29)",
        }
    except Exception:
        return {"model": "unknown", "error": "could not load AI stats"}


def _get_live_stats(paper_trader) -> dict:
    """Get live trading stats if available."""
    if hasattr(paper_trader, "store") and paper_trader.store:
        try:
            return paper_trader.store.get_live_stats()
        except Exception as e:
            logger.debug("Failed to fetch live stats for daily report: %s", e)
    return {}


def _get_api_health() -> list[dict]:
    """Get health status of all external APIs."""
    try:
        from weather_edge.analysis.service_health import get_service_status
        data = get_service_status()
        services = data.get("services", [])
        return [
            {
                "name": s.get("name", ""),
                "status": s.get("status", "unknown"),
                "last_call": s.get("last_call", ""),
                "uptime_pct": s.get("uptime_pct", 0),
            }
            for s in services
        ]
    except Exception:
        return []


def _get_risk_config() -> dict:
    """Get current risk profile config."""
    try:
        from weather_edge.analysis.risk_controls import (
            get_active_profile,
        )
        p = get_active_profile()
        return {
            "name": p.name,
            "kelly_fraction": p.kelly_fraction,
            "max_position_pct": p.max_position_pct,
            "reserve_pct": p.reserve_pct,
            "drawdown_scale_back": p.drawdown_scale_back_pct,
            "drawdown_kill": p.drawdown_kill_pct,
            "correlation_limit": p.max_group_exposure_pct,
            "gross_exposure_cap": p.max_gross_exposure_multiple,
        }
    except Exception:
        return {"name": "unknown"}


def _get_circuit_breaker_state(nav: float) -> dict:
    """Get circuit breaker status."""
    try:
        from weather_edge.analysis.risk_controls import _circuit_breaker
        return {
            "status": "KILLED" if _circuit_breaker.is_killed
                      else "SCALED_BACK" if _circuit_breaker.is_scaled_back
                      else "NORMAL",
            "high_water_mark": round(_circuit_breaker.high_water_mark, 2),
            "current_nav": round(nav, 2),
            "drawdown_pct": round(
                (_circuit_breaker.high_water_mark - nav)
                / _circuit_breaker.high_water_mark * 100, 1
            ) if _circuit_breaker.high_water_mark > 0 else 0,
        }
    except Exception:
        return {"status": "unknown"}


def save_daily_report(paper_trader, store) -> dict:
    """Generate and persist a daily report."""
    report = generate_daily_report(paper_trader)
    key = f"daily_report:{report['date']}"
    store.set_state(key, json.dumps(report))
    logger.info(
        "DAILY REPORT saved: %s, portfolio $%.0f, P&L %+.0f, %dW/%dL",
        report["date"], report["portfolio_value"], report["total_pnl"],
        report["total_wins"], report["total_losses"],
    )
    return report


def load_daily_reports(store, limit: int = 30) -> list[dict]:
    """Load recent daily reports from storage."""
    reports = []
    # Scan state table for daily_report keys
    try:
        cur = store.conn.execute(
            "SELECT key, value FROM state WHERE key LIKE 'daily_report:%' "
            "ORDER BY key DESC LIMIT ?",
            (limit,),
        )
        for row in cur.fetchall():
            try:
                reports.append(json.loads(row["value"]))
            except (json.JSONDecodeError, TypeError):
                pass
    except Exception:
        pass
    return reports
