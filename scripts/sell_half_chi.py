"""Sell part of one live position at an explicit limit price.

Originally a one-off "sell half of Chicago 40-41F" script that matched the
position by description substring and dumped it at a hardcoded 0.60 taker
limit. It now requires the exact token and an explicit limit price, and is a
dry run unless BOTH ``--execute`` and ``LIVE_MODE=true`` are set.

    python scripts/sell_half_chi.py --token-id <asset_id> --limit-price 0.68
    python scripts/sell_half_chi.py --token-id <asset_id> --limit-price 0.68 \\
        --fraction 0.5 --execute

The executor's slippage guard refuses a limit more than max_slippage_pct
below the token's live midpoint.
"""
import argparse
import asyncio
import logging


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--token-id", required=True, help="Exact asset/token id to sell")
    ap.add_argument(
        "--limit-price", required=True, type=float,
        help="Sell limit price for THIS token (0.01-0.99)",
    )
    ap.add_argument(
        "--fraction", type=float, default=0.5,
        help="Fraction of the held position to sell (default 0.5)",
    )
    ap.add_argument("--taker", action="store_true", help="Cross the spread (pays taker fee)")
    ap.add_argument(
        "--execute", action="store_true",
        help="Place a real order (also requires LIVE_MODE=true)",
    )
    args = ap.parse_args(argv)
    if not 0.0 < args.limit_price < 1.0:
        ap.error("--limit-price must be between 0 and 1")
    if not 0.0 < args.fraction <= 1.0:
        ap.error("--fraction must be in (0, 1]")
    return args


async def main(argv=None):
    args = parse_args(argv)  # before heavy imports so --help always works
    from weather_edge.config import settings
    from weather_edge.persistence import PersistentStore
    from weather_edge.trading.executor import TradeExecutor, script_dry_run

    dry_run = script_dry_run(args.execute)

    store = PersistentStore()
    try:
        pos = next(
            (p for p in store.get_positions() if p.get("asset_id") == args.token_id),
            None,
        )
    finally:
        store.close()
    if not pos:
        print(f"No open position for token {args.token_id}")
        return 1

    total_shares = float(pos["total_shares"])
    shares = round(total_shares * args.fraction, 2)
    print(
        "Position {} ({}): {} shares, selling {} @ ${:.2f}{}".format(
            pos.get("description", "")[:50], pos.get("outcome", ""),
            total_shares, shares, args.limit_price,
            " (DRY RUN)" if dry_run else "",
        )
    )

    executor = TradeExecutor(
        private_key=settings.polymarket_private_key,
        wallet_address=settings.polymarket_wallet,
        api_key=settings.polymarket_api_key,
        api_secret=settings.polymarket_api_secret,
        api_passphrase=settings.polymarket_api_passphrase,
        signature_type=settings.polymarket_signature_type,
        dry_run=dry_run,
        post_only=True,
    )
    await executor.initialize()

    result = await executor.place_sell_order(
        token_id=args.token_id,
        shares=shares,
        price=args.limit_price,
        market_id=pos["condition_id"],
        city_id=pos.get("city_id", ""),
        description=f"MANUAL SELL {args.fraction:.0%}",
        force_taker=args.taker,
    )
    if result and result.status not in ("rejected", "post_only_reject"):
        print(f"SELL {result.status.upper()}: {result.order_id}")
        return 0
    print("SELL NOT PLACED:", result.reject_reason if result else "blocked (see log)")
    return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(asyncio.run(main()))
