"""Portfolio sync, reconcile local state with exchange truth.

The CLOB API's get_trades() is the single source of truth.
Orders are intentions. Fills are reality. Positions are aggregated fills.

Every cycle:
1. Fetch all trades from get_trades()
2. INSERT OR IGNORE into fills table (immutable ledger)
3. Rebuild positions table from fills (aggregated view)
4. Update market_map for dashboard display

This replaces the old order-centric fill tracker for position awareness.
The fills table is the tax audit trail (UK CGT Section 104 pooling).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


async def sync_portfolio(executor, store, market_lookup: dict | None = None) -> dict:
    """Full portfolio reconciliation from exchange.

    Args:
        executor: TradeExecutor with initialized _client.
        store: PersistentStore instance.
        market_lookup: Optional dict mapping condition_id → {city_id, description}
                       from current market discovery.

    Returns:
        Summary dict with position_count, total_deployed, fills_synced.
    """
    if executor is None or executor.dry_run or executor._client is None:
        return {"synced": False, "reason": "no live executor"}

    loop = asyncio.get_running_loop()

    # Step 1: Fetch all trades from CLOB API (the truth)
    try:
        all_trades = await loop.run_in_executor(
            None, executor._client.get_trades,
        )
    except Exception as e:
        logger.error("Portfolio sync failed, get_trades error: %s", e)
        return {"synced": False, "reason": str(e)}

    if not isinstance(all_trades, list):
        # Some versions return {data: [...]}
        all_trades = all_trades.get("data", []) if isinstance(all_trades, dict) else []

    fills_synced = 0
    our_address = (executor.wallet_address or "").lower()

    for trade in all_trades:
        trade_id = trade.get("id")
        if not trade_id:
            continue

        condition_id = trade.get("market", "")
        tx_hash = trade.get("transaction_hash", "")
        match_time = trade.get("match_time", "")

        # Convert epoch to ISO
        try:
            filled_at = datetime.fromtimestamp(
                int(match_time), tz=timezone.utc,
            ).isoformat() if match_time else ""
        except (ValueError, TypeError):
            filled_at = match_time

        # Our fills are in maker_orders where our address matches
        maker_orders = trade.get("maker_orders", [])
        for mo in maker_orders:
            maker_addr = (mo.get("maker_address") or "").lower()
            if maker_addr != our_address:
                continue

            asset_id = mo.get("asset_id", "")
            order_id = mo.get("order_id", "")
            matched = float(mo.get("matched_amount", 0))
            price = float(mo.get("price", 0))
            side = mo.get("side", "BUY")
            outcome = mo.get("outcome", "")
            fee_bps = int(mo.get("fee_rate_bps", 0))

            # Unique fill ID: trade_id + order_id (one trade can match multiple of our orders)
            fill_id = f"{trade_id}:{order_id}"

            # Look up city from market_lookup or market_map
            city_id = ""
            description = ""
            if market_lookup and condition_id in market_lookup:
                city_id = market_lookup[condition_id].get("city_id", "")
                description = market_lookup[condition_id].get("description", "")

            store.upsert_fill(
                fill_id=fill_id,
                order_id=order_id,
                asset_id=asset_id,
                condition_id=condition_id,
                city_id=city_id,
                side=side,
                size=matched,
                price=price,
                filled_at=filled_at,
                tx_hash=tx_hash,
                is_maker=1 if trade.get("trader_side") == "MAKER" else 0,
                fee_rate_bps=fee_bps,
                outcome=outcome,
                description=description,
            )

            # Update market_map
            if asset_id:
                store.upsert_market_map(
                    asset_id=asset_id,
                    condition_id=condition_id,
                    city_id=city_id,
                    outcome=outcome,
                    description=description,
                    token_side=side,
                )

            fills_synced += 1

        # We might also be the taker (trader_side == TAKER)
        if trade.get("trader_side") == "TAKER":
            asset_id = trade.get("asset_id", "")
            size = float(trade.get("size", 0))
            price = float(trade.get("price", 0))
            side = trade.get("side", "BUY")
            outcome = trade.get("outcome", "")

            fill_id = f"{trade_id}:taker"

            city_id = ""
            description = ""
            if market_lookup and condition_id in market_lookup:
                city_id = market_lookup[condition_id].get("city_id", "")
                description = market_lookup[condition_id].get("description", "")

            store.upsert_fill(
                fill_id=fill_id,
                order_id="",
                asset_id=asset_id,
                condition_id=condition_id,
                city_id=city_id,
                side=side,
                size=size,
                price=price,
                filled_at=filled_at,
                tx_hash=tx_hash,
                is_maker=0,
                fee_rate_bps=int(trade.get("fee_rate_bps", 0)),
                outcome=outcome,
                description=description,
            )
            fills_synced += 1

    store.commit()

    # Step 1b: Backfill city_id and description on fills from market_map
    store.conn.execute("""
        UPDATE fills SET
            city_id = (SELECT m.city_id FROM market_map m WHERE m.asset_id = fills.asset_id),
            description = (SELECT m.description FROM market_map m WHERE m.asset_id = fills.asset_id)
        WHERE (city_id = '' OR city_id IS NULL)
          AND asset_id IN (SELECT asset_id FROM market_map WHERE city_id != '')
    """)
    store.commit()

    # Step 2: Rebuild positions from fills
    store.rebuild_positions()

    # Step 3: Get summary
    summary = store.get_portfolio_summary()
    summary["fills_synced"] = fills_synced
    summary["total_fills_on_exchange"] = len(all_trades)
    summary["synced"] = True

    logger.info(
        "PORTFOLIO SYNC: %d fills → %d positions, $%.2f deployed",
        fills_synced, summary.get("position_count", 0),
        summary.get("total_deployed", 0),
    )

    return summary


async def sync_market_map_from_discovery(store, markets) -> int:
    """Update market_map from Polymarket market discovery.

    Call this after discover_weather_markets() to map condition_ids
    to city_id and description for the dashboard.
    """
    mapped = 0
    for m in markets:
        if not m.city_id:
            continue
        city = m.city_id.value if hasattr(m.city_id, "value") else str(m.city_id)
        condition_id = m.market_id

        if m.token_id_yes:
            store.upsert_market_map(
                asset_id=m.token_id_yes,
                condition_id=condition_id,
                city_id=city,
                outcome="Yes",
                description=m.question[:80] if m.question else "",
                token_side="YES",
            )
            mapped += 1

        if m.token_id_no:
            store.upsert_market_map(
                asset_id=m.token_id_no,
                condition_id=condition_id,
                city_id=city,
                outcome="No",
                description=m.question[:80] if m.question else "",
                token_side="NO",
            )
            mapped += 1

    store.commit()
    return mapped
