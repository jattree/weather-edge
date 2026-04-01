"""Test auto-redeem directly."""
import asyncio
import logging

logging.basicConfig(level=logging.DEBUG)

async def main():
    from weather_edge.trading.executor import TradeExecutor
    from weather_edge.config import settings

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

    print("Testing redeem_positions...")
    try:
        result = await executor.redeem_positions()
        print("Result:", result)
    except Exception as e:
        print("Exception:", type(e).__name__, e)

asyncio.run(main())
