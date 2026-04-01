"""One-off script: sell half of Chicago 40-41F position to lock in profit."""
import asyncio


async def main():
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

    store = PersistentStore()
    pos = store.get_positions()
    chi_pos = None
    for p in pos:
        desc = p.get("description", "")
        if "Chicago" in desc and "40-41" in desc:
            chi_pos = p
            break

    if not chi_pos:
        print("Chicago 40-41F position not found!")
        store.close()
        return

    asset_id = chi_pos["asset_id"]
    condition_id = chi_pos["condition_id"]
    total_shares = chi_pos["total_shares"]
    half_shares = round(total_shares / 2, 0)

    # Current market: ~$0.71/share. Sell at $0.69 to guarantee taker fill.
    sell_price = 0.69

    print("Found: {} shares, selling {} @ ${:.2f}".format(total_shares, half_shares, sell_price))

    result = await executor.place_sell_order(
        token_id=asset_id,
        shares=half_shares,
        price=sell_price,
        market_id=condition_id,
        city_id="chi",
        description="TAKE PROFIT: sell half CHI 40-41F",
        force_taker=True,
    )

    if result:
        print("SELL ORDER PLACED: {}".format(result.order_id))
        print("  {} shares @ $0.18 = ${:.2f}".format(half_shares, half_shares * 0.18))
    else:
        print("SELL ORDER FAILED")

    store.close()


asyncio.run(main())
