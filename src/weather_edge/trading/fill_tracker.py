"""Track order fills and update SQLite live_trades table.

Polls Polymarket CLOB for order status updates. Handles:
- Full fills (order complete, record cost basis)
- Partial fills (some shares filled, rest still resting)
- Cancellations and expirations
- Market resolution (winning = $1/share, losing = $0/share)

All state persisted to SQLite for tax compliance.
"""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


async def poll_fills(executor, interval: int = 15) -> None:
    """Background loop: poll order status for all open live trades.

    Reads open orders from SQLite, checks status via CLOB API,
    updates SQLite with fill data.
    """
    while True:
        await asyncio.sleep(interval)

        if executor is None or executor.dry_run or executor._client is None:
            continue

        try:
            from weather_edge.persistence import PersistentStore
            store = PersistentStore()
            open_trades = store.get_open_live_trades()

            if not open_trades:
                store.close()
                continue

            for trade in open_trades:
                order_id = trade.get("order_id")
                if not order_id:
                    continue

                try:
                    status = await executor.get_order_status(order_id)
                    if not status or not isinstance(status, dict):
                        continue

                    order_status = status.get("status", "UNKNOWN")
                    size_matched = float(status.get("size_matched", 0))
                    original_size = float(status.get("original_size", status.get("size", 0)))
                    price = float(status.get("price", trade.get("limit_price", 0)))

                    if size_matched > 0 and order_status in ("MATCHED", "LIVE"):
                        is_full = size_matched >= original_size * 0.99  # 99% = full (rounding)
                        fill_status = "filled" if is_full else "partial"
                        fee = 0.0  # Maker orders = $0 fee

                        store.update_live_trade_fill(
                            order_id=order_id,
                            avg_fill_price=price,
                            filled_shares=size_matched,
                            fee_usd=fee,
                            status=fill_status,
                        )

                        if is_full:
                            logger.info(
                                "FILL COMPLETE: %s %s %.0f shares @ %.3f ($%.2f)",
                                trade.get("city_id", "?").upper(),
                                trade.get("side", "?"),
                                size_matched, price,
                                size_matched * price,
                            )
                        else:
                            logger.info(
                                "PARTIAL FILL: %s %.0f/%.0f shares",
                                trade.get("city_id", "?").upper(),
                                size_matched, original_size,
                            )

                    elif order_status == "CANCELLED":
                        store.cancel_live_trade(order_id)
                        logger.info(
                            "ORDER CANCELLED: %s %s",
                            trade.get("city_id", "?").upper(), order_id[:16],
                        )

                    elif order_status == "EXPIRED":
                        store.cancel_live_trade(order_id)
                        logger.info(
                            "ORDER EXPIRED: %s %s",
                            trade.get("city_id", "?").upper(), order_id[:16],
                        )

                except Exception as e:
                    logger.debug("Fill check failed for %s: %s", order_id[:16], e)

            store.close()

        except Exception as e:
            logger.warning("Fill tracker cycle failed: %s", e)


async def resolve_live_trades(executor) -> int:
    """Check if any filled live trades have resolved (market settled).

    For weather markets: winning token = $1, losing token = $0.
    P&L = proceeds - cost_basis.

    Called from the main scheduler cycle alongside paper resolution.
    """
    if executor is None or executor.dry_run:
        return 0

    try:
        from weather_edge.persistence import PersistentStore
        from weather_edge.analysis.resolver import fetch_resolved_markets

        store = PersistentStore()
        # Get filled (but not yet resolved) trades
        rows = store.conn.execute(
            "SELECT * FROM live_trades WHERE status = 'filled' ORDER BY placed_at",
        ).fetchall()
        filled_trades = [dict(r) for r in rows]

        if not filled_trades:
            store.close()
            return 0

        # Fetch resolved markets from Polymarket
        try:
            resolved_markets = await fetch_resolved_markets()
        except Exception:
            resolved_markets = {}

        resolved_count = 0

        for trade in filled_trades:
            market_id = trade.get("market_id")
            if market_id not in resolved_markets:
                continue

            outcome_yes = resolved_markets[market_id]
            side = trade.get("side", "")
            filled = trade.get("filled_shares", 0)
            cost_basis = trade.get("cost_basis", 0) or 0
            fee = trade.get("fee_usd", 0) or 0

            # Winning token pays $1/share, losing pays $0
            if (side == "BUY" and outcome_yes) or (side == "NO" and not outcome_yes):
                # We bought the winning side
                proceeds = filled * 1.0 - fee
                status = "won"
            else:
                # We bought the losing side
                proceeds = 0.0
                status = "lost"

            pnl = proceeds - cost_basis

            store.resolve_live_trade(
                order_id=trade["order_id"],
                proceeds=round(proceeds, 4),
                pnl=round(pnl, 4),
                status=status,
            )
            resolved_count += 1

            logger.info(
                "LIVE RESOLVED: %s %s %s | outcome=%s | cost=$%.2f proceeds=$%.2f P&L=$%.2f",
                side, trade.get("city_id", "?").upper(),
                (trade.get("description") or "")[:40],
                "YES" if outcome_yes else "NO",
                cost_basis, proceeds, pnl,
            )

        store.close()
        return resolved_count

    except Exception as e:
        logger.warning("Live trade resolution failed: %s", e)
        return 0
