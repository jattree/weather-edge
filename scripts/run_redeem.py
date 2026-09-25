"""Redeem resolved winning positions back to USDC.

Dry run by default. Real redemption requires BOTH ``--execute`` and
``LIVE_MODE=true``.

    python scripts/run_redeem.py            # dry run, no transactions
    python scripts/run_redeem.py --execute  # real redemption (LIVE_MODE only)
"""
import argparse
import asyncio
import logging


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--execute", action="store_true",
        help="Submit real redemption transactions (also requires LIVE_MODE=true)",
    )
    return ap.parse_args(argv)


async def main(argv=None):
    args = parse_args(argv)  # before heavy imports so --help always works
    from weather_edge.config import settings
    from weather_edge.trading.executor import TradeExecutor, script_dry_run

    dry_run = script_dry_run(args.execute)

    executor = TradeExecutor(
        private_key=settings.polymarket_private_key,
        wallet_address=settings.polymarket_wallet,
        api_key=settings.polymarket_api_key,
        api_secret=settings.polymarket_api_secret,
        api_passphrase=settings.polymarket_api_passphrase,
        signature_type=settings.polymarket_signature_type,
        dry_run=dry_run,
        post_only=True,
        relayer_api_key=settings.polymarket_relayer_api_key,
    )
    await executor.initialize()

    if dry_run:
        print("DRY RUN: no redemption submitted.")
        return 0

    print("Running redeem_positions...")
    try:
        result = await executor.redeem_positions()
        print("Confirmed redemptions:", result)
    except Exception as e:
        print("Exception:", type(e).__name__, e)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
