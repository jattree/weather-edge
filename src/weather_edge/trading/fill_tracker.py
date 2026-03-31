"""Track order fills and update local state.

Polls Polymarket CLOB for order status updates. Handles:
- Full fills (order complete, update P&L)
- Partial fills (some shares filled, rest still resting)
- Cancellations and expirations
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from weather_edge.live_state import get_json, set_json
from weather_edge.trading.kill_switch import untrack_order

logger = logging.getLogger(__name__)

TRACKED_ORDERS_KEY = "live:open_orders"
FILL_LOG_KEY = "live:fill_log"


@dataclass
class FillEvent:
    """A fill (or partial fill) event."""
    order_id: str
    status: str  # MATCHED, FILLED, CANCELLED, EXPIRED
    filled_shares: float
    total_shares: float
    price: float
    side: str
    token_id: str
    timestamp: str


async def poll_fills(executor, interval: int = 15) -> None:
    """Background loop: poll order status for all tracked open orders.

    Args:
        executor: TradeExecutor with initialized _client.
        interval: Seconds between polls.
    """
    while True:
        await asyncio.sleep(interval)

        if executor is None or executor.dry_run or executor._client is None:
            continue

        tracked = get_json(TRACKED_ORDERS_KEY) or []
        if not tracked:
            continue

        for order_id in list(tracked):  # Copy list, we may modify during iteration
            try:
                status = await executor.get_order_status(order_id)
                if not status:
                    continue

                order_status = status.get("status", "UNKNOWN")
                size_matched = float(status.get("size_matched", 0))
                original_size = float(status.get("original_size", status.get("size", 0)))

                if order_status == "MATCHED" and size_matched > 0:
                    # Partial or full fill
                    fill = FillEvent(
                        order_id=order_id,
                        status="partial" if size_matched < original_size else "filled",
                        filled_shares=size_matched,
                        total_shares=original_size,
                        price=float(status.get("price", 0)),
                        side=status.get("side", ""),
                        token_id=status.get("asset_id", ""),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                    _log_fill(fill)

                    if size_matched >= original_size:
                        untrack_order(order_id)
                        logger.info(
                            "FILL COMPLETE: %s, %.0f shares @ %.3f",
                            order_id[:12], size_matched, fill.price,
                        )
                    else:
                        logger.info(
                            "PARTIAL FILL: %s, %.0f/%.0f shares",
                            order_id[:12], size_matched, original_size,
                        )

                elif order_status in ("CANCELLED", "EXPIRED"):
                    untrack_order(order_id)
                    logger.info("ORDER %s: %s", order_status, order_id[:12])

            except Exception as e:
                logger.debug("Fill check failed for %s: %s", order_id[:12], e)


def _log_fill(fill: FillEvent) -> None:
    """Append fill event to Redis log (last 100 fills)."""
    log = get_json(FILL_LOG_KEY) or []
    log.insert(0, {
        "order_id": fill.order_id,
        "status": fill.status,
        "filled_shares": fill.filled_shares,
        "total_shares": fill.total_shares,
        "price": fill.price,
        "side": fill.side,
        "timestamp": fill.timestamp,
    })
    set_json(FILL_LOG_KEY, log[:100])


def get_fill_log() -> list[dict]:
    """Get recent fill events for dashboard display."""
    return get_json(FILL_LOG_KEY) or []
