"""Emergency kill switch for live trading.

Two-phase shutdown:
1. BLOCK, Redis flag prevents any new orders from being placed.
2. CANCEL, Actively cancels all open limit orders on Polymarket CLOB.

The kill switch is checked before every order placement. It can be
triggered via:
- POST /api/kill-switch (dashboard emergency button)
- Circuit breaker drawdown kill (automatic)
- Manual: redis-cli SET kill_switch:active 1

The flag persists across restarts (no TTL). Must be explicitly cleared
via POST /api/kill-switch/reset or redis-cli DEL kill_switch:active.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from weather_edge import live_state
from weather_edge.live_state import get_json, set_json, set_value

logger = logging.getLogger(__name__)

KILL_SWITCH_KEY = "kill_switch:active"
KILL_SWITCH_META_KEY = "kill_switch:meta"


def _live_mode_enabled() -> bool:
    try:
        from weather_edge.config import settings
        return bool(settings.live_mode)
    except Exception:
        return True  # Can't tell: assume live, fail closed


def is_kill_switch_active() -> bool:
    """Check if kill switch is engaged. Called before every order.

    Fail-closed semantics (returns True = orders blocked):

    - The in-memory flag is set (activated while Redis was unreachable).
    - Redis is connected and the flag is "1".
    - Redis is in use (connected earlier in this process) but the read
      errors or the connection is lost.
    - Redis has never been reachable in this process AND LIVE_MODE is on:
      a flag persisted by a previous process cannot be verified, so real
      orders are refused. Start Redis to trade live.

    Local no-Redis mode (Redis never reachable, LIVE_MODE off) uses the
    in-memory flag only, so paper/dry-run work without Redis.
    """
    try:
        if live_state.get_fallback_value(KILL_SWITCH_KEY) == "1":
            return True
        val = live_state.get_value_strict(KILL_SWITCH_KEY)
        if val == "1":
            return True
        if not live_state.redis_connected() and _live_mode_enabled():
            logger.error(
                "Kill switch state unverifiable (Redis unreachable) in LIVE_MODE, "
                "blocking orders (fail-closed)",
            )
            return True
        return False
    except Exception:
        # Fail closed, if we can't check, assume killed
        logger.error("Kill switch check failed, blocking orders (fail-closed)")
        return True


def activate_kill_switch(reason: str, triggered_by: str = "manual") -> dict:
    """Activate the kill switch. No new orders will be placed.

    Args:
        reason: Why the kill switch was triggered.
        triggered_by: Who/what triggered it (manual, circuit_breaker, api).

    Returns:
        Kill switch state dict.
    """
    now = datetime.now(timezone.utc)
    # Set the flag (no TTL, must be explicitly cleared)
    set_value(KILL_SWITCH_KEY, "1")

    meta = {
        "active": True,
        "reason": reason,
        "triggered_by": triggered_by,
        "triggered_at": now.isoformat(),
    }
    set_json(KILL_SWITCH_META_KEY, meta)

    logger.warning(
        "KILL SWITCH ACTIVATED by %s: %s", triggered_by, reason,
    )
    return meta


def deactivate_kill_switch(cleared_by: str = "manual") -> dict:
    """Deactivate the kill switch. Trading can resume.

    Args:
        cleared_by: Who cleared it.

    Returns:
        Kill switch state dict.
    """
    set_value(KILL_SWITCH_KEY, "0")
    # Clear any flag written to the in-memory fallback while Redis was down
    live_state._fallback_cache.pop(KILL_SWITCH_KEY, None)
    # Explicit operator reset also re-arms the live drawdown breaker with a
    # fresh high-water mark (otherwise it would keep blocking / re-trip).
    try:
        from weather_edge.analysis.risk_controls import reset_live_circuit_breaker
        reset_live_circuit_breaker()
    except Exception:
        logger.error("Failed to reset live circuit breaker", exc_info=True)

    meta = {
        "active": False,
        "reason": "",
        "triggered_by": "",
        "triggered_at": "",
        "cleared_by": cleared_by,
        "cleared_at": datetime.now(timezone.utc).isoformat(),
    }
    set_json(KILL_SWITCH_META_KEY, meta)

    logger.info("KILL SWITCH DEACTIVATED by %s", cleared_by)
    return meta


def get_kill_switch_state() -> dict:
    """Get current kill switch state for dashboard display."""
    active = is_kill_switch_active()
    meta = get_json(KILL_SWITCH_META_KEY) or {}
    return {
        "active": active,
        "reason": meta.get("reason", ""),
        "triggered_by": meta.get("triggered_by", ""),
        "triggered_at": meta.get("triggered_at", ""),
        "cleared_by": meta.get("cleared_by", ""),
        "cleared_at": meta.get("cleared_at", ""),
    }


async def kill_and_cancel(executor, reason: str, triggered_by: str = "manual") -> dict:
    """Activate kill switch AND cancel all open orders on exchange.

    This is the full emergency shutdown:
    1. Set Redis flag (instant, blocks new orders)
    2. Cancel all open orders via CLOB API

    Args:
        executor: TradeExecutor instance (needs _client for cancel calls).
        reason: Why.
        triggered_by: Who.

    Returns:
        dict with kill switch state and cancel results.
    """
    # Phase 1: Block new orders immediately
    meta = activate_kill_switch(reason, triggered_by)

    # Phase 2: Cancel all open orders on exchange
    cancelled = 0
    cancel_errors = []

    if executor and executor._client is not None:
        try:
            loop = asyncio.get_running_loop()
            # py-clob-client cancel_all() is synchronous
            result = await loop.run_in_executor(
                None, executor._client.cancel_all,
            )
            cancelled = len(result) if isinstance(result, list) else 1
            logger.warning(
                "KILL SWITCH: cancelled %d open orders on exchange", cancelled,
            )
        except Exception as e:
            err = f"Failed to cancel orders: {e}"
            cancel_errors.append(err)
            logger.error("KILL SWITCH CANCEL FAILED: %s", e)

            # Fallback: try cancelling tracked orders individually
            try:
                tracked = get_json("live:open_orders") or []
                for order_id in tracked:
                    try:
                        await loop.run_in_executor(
                            None, executor._client.cancel, order_id,
                        )
                        cancelled += 1
                    except Exception as inner_e:
                        cancel_errors.append(f"Cancel {order_id}: {inner_e}")
            except Exception:
                pass
    else:
        logger.warning("KILL SWITCH: no live client, skipping exchange cancel")

    meta["orders_cancelled"] = cancelled
    meta["cancel_errors"] = cancel_errors
    set_json(KILL_SWITCH_META_KEY, meta)

    return meta


def track_open_order(order_id: str) -> None:
    """Add an order ID to the tracked open orders set.

    Used by kill_and_cancel as fallback if cancel_all fails.
    """
    orders = get_json("live:open_orders") or []
    if order_id not in orders:
        orders.append(order_id)
        set_json("live:open_orders", orders)


def untrack_order(order_id: str) -> None:
    """Remove a filled/cancelled order from tracking."""
    orders = get_json("live:open_orders") or []
    if order_id in orders:
        orders.remove(order_id)
        set_json("live:open_orders", orders)
