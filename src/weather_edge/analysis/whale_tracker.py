"""On-chain whale tracking via Polygon/Polygonscan.

Tracks known weather market traders (ColdMath, etc.) by monitoring their
wallet addresses for CTF (Conditional Token Framework) token transfers
on the Polygon blockchain.

Dormant by default, activate via WHALE_TRACKING=true in .env when going live.

How it works:
1. Query Polygonscan API for recent token transfers from known wallets
2. Parse transfer amounts and token IDs to determine position changes
3. Map token IDs back to Polymarket markets via Gamma API
4. Return a feed of whale trades: "ColdMath bought 500 YES on Dallas 84-85°F"

This is the "pro" way to track competitors, raw on-chain data, not profile scraping.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

# Known whale wallets, add more as we identify them
TRACKED_WALLETS: dict[str, str] = {
    # ColdMath's wallet, find this from his Polymarket profile activity
    # The hex address is visible in the transaction links on his activity page
    # "coldmath": "0x...",  # TODO: populate from Polymarket profile
}

# Polymarket CTF contract on Polygon
CTF_CONTRACT = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

# Polygonscan API (free tier: 5 calls/sec)
POLYGONSCAN_API = "https://api.polygonscan.com/api"
POLYGONSCAN_KEY = os.environ.get("POLYGONSCAN_API_KEY", "")  # Free key from polygonscan.com


@dataclass
class WhaleTrade:
    """A detected trade from a tracked whale."""
    wallet_name: str
    wallet_address: str
    timestamp: datetime
    token_id: str
    amount: float
    direction: str  # "buy" or "sell"
    tx_hash: str
    # Enriched from Polymarket (if we can map token_id -> market)
    market_description: str = ""
    city_id: str = ""
    side: str = ""  # "YES" or "NO"


class WhaleTracker:
    """Tracks on-chain activity of known weather market traders."""

    def __init__(self):
        self.trades: list[WhaleTrade] = []
        self._last_block: dict[str, int] = {}  # wallet -> last scanned block

    @property
    def is_active(self) -> bool:
        """Only active when configured with wallet addresses and API key."""
        return bool(TRACKED_WALLETS) and bool(POLYGONSCAN_KEY)

    async def fetch_recent_transfers(
        self,
        wallet_address: str,
        wallet_name: str,
    ) -> list[WhaleTrade]:
        """Fetch recent ERC-1155 token transfers for a wallet from Polygonscan."""
        if not POLYGONSCAN_KEY:
            return []

        last_block = self._last_block.get(wallet_address, 0)
        trades: list[WhaleTrade] = []

        async with httpx.AsyncClient() as client:
            try:
                # ERC-1155 transfers (CTF tokens are ERC-1155)
                resp = await client.get(
                    POLYGONSCAN_API,
                    params={
                        "module": "account",
                        "action": "token1155tx",
                        "address": wallet_address,
                        "contractaddress": CTF_CONTRACT,
                        "startblock": last_block,
                        "sort": "desc",
                        "apikey": POLYGONSCAN_KEY,
                    },
                    timeout=10.0,
                )
                resp.raise_for_status()
                data = resp.json()
            except (httpx.HTTPError, ValueError) as e:
                logger.debug("Polygonscan query failed for %s: %s", wallet_name, e)
                return []

        results = data.get("result", [])
        if not isinstance(results, list):
            return []

        for tx in results[:20]:  # Last 20 transfers
            try:
                block = int(tx.get("blockNumber", 0))
                timestamp = datetime.fromtimestamp(
                    int(tx.get("timeStamp", 0)), tz=timezone.utc
                )
                token_id = tx.get("tokenID", "")
                amount = float(tx.get("tokenValue", 0))
                from_addr = tx.get("from", "").lower()
                to_addr = tx.get("to", "").lower()
                tx_hash = tx.get("hash", "")

                # Determine direction
                if to_addr == wallet_address.lower():
                    direction = "buy"
                elif from_addr == wallet_address.lower():
                    direction = "sell"
                else:
                    continue

                trade = WhaleTrade(
                    wallet_name=wallet_name,
                    wallet_address=wallet_address,
                    timestamp=timestamp,
                    token_id=token_id,
                    amount=amount,
                    direction=direction,
                    tx_hash=tx_hash,
                )
                trades.append(trade)

                # Update last scanned block
                if block > self._last_block.get(wallet_address, 0):
                    self._last_block[wallet_address] = block

            except (ValueError, KeyError) as e:
                logger.debug("Failed to parse transfer: %s", e)
                continue

        if trades:
            logger.info(
                "WHALE: %s, %d new transfers detected",
                wallet_name, len(trades),
            )

        return trades

    async def scan_all_wallets(self) -> list[WhaleTrade]:
        """Scan all tracked wallets for recent activity."""
        if not self.is_active:
            return []

        all_trades: list[WhaleTrade] = []
        for name, address in TRACKED_WALLETS.items():
            trades = await self.fetch_recent_transfers(address, name)
            all_trades.extend(trades)
            self.trades.extend(trades)

        # Keep only last 100 trades in memory
        if len(self.trades) > 100:
            self.trades = self.trades[-100:]

        return all_trades

    def recent_trades(self, limit: int = 20) -> list[dict]:
        """Return recent whale trades as dicts for the API/dashboard."""
        return [
            {
                "whale": t.wallet_name,
                "direction": t.direction,
                "amount": t.amount,
                "token_id": t.token_id[:20] if t.token_id else "",
                "market": t.market_description or "Unknown market",
                "city": t.city_id,
                "side": t.side,
                "time": t.timestamp.strftime("%H:%M:%S"),
                "tx": t.tx_hash[:16] if t.tx_hash else "",
            }
            for t in sorted(self.trades, key=lambda x: x.timestamp, reverse=True)[:limit]
        ]

    def summary(self) -> dict:
        """Summary for API state."""
        return {
            "active": self.is_active,
            "tracked_wallets": list(TRACKED_WALLETS.keys()),
            "total_trades_observed": len(self.trades),
            "recent": self.recent_trades(10),
        }
