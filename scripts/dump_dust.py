"""Dump all dust positions (value < $5) to free up position slots.

Sells at $0.01 (floor price) as taker to guarantee fill.
Skips positions with < 5 shares (Polymarket minimum).
The goal is clearing position count, not recovering value.
"""
import asyncio
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MIN_SELL_SHARES = 5.0
DUST_VALUE_THRESHOLD = 5.0  # Sell anything worth less than $5


async def main():
    import httpx
    from weather_edge.trading.executor import TradeExecutor
    from weather_edge.config import settings
    from weather_edge.persistence import PersistentStore

    executor = TradeExecutor(
        private_key=settings.polymarket_private_key,
        wallet_address=settings.polymarket_wallet,
        api_key=settings.polymarket_api_key,
        api_secret=settings.polymarket_api_secret,
        api_passphrase=settings.polymarket_api_passphrase,
        signature_type=settings.polymarket_signature_type,
        dry_run=False,
        post_only=True,
    )
    await executor.initialize()

    wallet = "0xe23940d70793b441c9f949741daa65289947fadb"

    # Get positions from Polymarket API (has current prices)
    r = httpx.get(
        "https://data-api.polymarket.com/positions",
        params={"user": wallet.lower(), "sizeThreshold": "0"},
        timeout=15.0,
    )
    positions = r.json()

    store = PersistentStore()

    sellable = []
    too_small = []
    zero_size = []

    for p in positions:
        if not isinstance(p, dict):
            continue
        size = float(p.get("size", 0))
        mv = float(p.get("currentValue", 0))
        cur_price = float(p.get("curPrice", 0))
        # Polymarket API uses "asset" not "asset_id"
        asset_id = p.get("asset", p.get("asset_id", ""))
        condition_id = p.get("conditionId", "")
        title = (p.get("title", "") or "")[:55]
        outcome = p.get("outcome", "")
        slug = p.get("slug", "")

        if size <= 0:
            zero_size.append(title)
            continue

        if mv >= DUST_VALUE_THRESHOLD:
            continue  # Keep non-dust

        # Extract city from slug (e.g. "highest-temperature-in-los-angeles-on-april-2-2026-60-61f")
        city_id = slug.split("in-")[-1].split("-on-")[0].replace("-", "_") if slug else ""

        if size >= MIN_SELL_SHARES:
            sellable.append({
                "asset_id": asset_id,
                "condition_id": condition_id,
                "city_id": city_id,
                "title": title,
                "outcome": outcome,
                "size": size,
                "value": mv,
                "price": cur_price,
            })
        else:
            too_small.append((title, size, mv))

    logger.info("=== DUST DUMP PLAN ===")
    logger.info("Sellable (>= 5 shares): %d positions", len(sellable))
    logger.info("Too small (< 5 shares): %d positions (will resolve naturally)", len(too_small))
    logger.info("Zero size: %d positions", len(zero_size))
    logger.info("")

    sold = 0
    failed = 0
    recovered = 0.0

    for pos in sellable:
        sell_price = max(0.01, round(pos["price"] - 0.02, 2))  # Below market to cross
        shares = pos["size"]

        logger.info(
            "SELLING: %s %s, %.0f shares @ $%.2f (val $%.2f)",
            pos["city_id"], pos["title"], shares, sell_price, pos["value"],
        )

        if not pos["condition_id"]:
            logger.info("  SKIP: no condition_id mapping")
            failed += 1
            continue

        try:
            result = await executor.place_sell_order(
                token_id=pos["asset_id"],
                shares=shares,
                price=sell_price,
                market_id=pos["condition_id"],
                city_id=pos["city_id"],
                description="DUST DUMP: %s" % pos["title"][:40],
                force_taker=True,
            )
            if result and result.status not in ("rejected",):
                sold += 1
                recovered += pos["value"]
                logger.info("  OK: %s", result.order_id)
            else:
                failed += 1
                logger.info("  FAILED: %s", result.status if result else "no result")
        except Exception as e:
            failed += 1
            logger.info("  ERROR: %s", e)

        # Small delay to avoid rate limiting
        await asyncio.sleep(0.5)

    logger.info("")
    logger.info("=== DUST DUMP COMPLETE ===")
    logger.info("Sold: %d positions (~$%.2f recovered)", sold, recovered)
    logger.info("Failed: %d", failed)
    logger.info("Too small to sell: %d (resolve naturally)", len(too_small))
    for title, size, mv in too_small:
        logger.info("  %.1f shares ($%.2f), %s", size, mv, title)

    store.close()


asyncio.run(main())
