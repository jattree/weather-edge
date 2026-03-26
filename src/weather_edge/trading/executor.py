"""Real trade execution via Polymarket CLOB API.

Uses limit orders (not market orders) for better fill prices on illiquid weather markets.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from weather_edge.analysis.edge import Signal

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    """Result of an order placement."""
    order_id: str
    market_id: str
    side: str
    size_usd: float
    limit_price: float
    status: str  # 'pending', 'filled', 'partial', 'cancelled'
    filled_price: float | None = None
    filled_at: datetime | None = None
    tx_hash: str | None = None


class TradeExecutor:
    """Executes real trades on Polymarket via CLOB API.

    Uses limit orders placed slightly inside the spread for better fills.
    This avoids the slippage of market orders that @ColdMath appears to use.
    """

    def __init__(
        self,
        api_key: str | None = None,
        private_key: str | None = None,
        dry_run: bool = True,
    ):
        self.api_key = api_key
        self.private_key = private_key
        self.dry_run = dry_run
        self._client = None

    async def initialize(self) -> None:
        """Initialize the Polymarket CLOB client."""
        if self.dry_run:
            logger.info("TradeExecutor running in DRY RUN mode")
            return

        if not self.api_key or not self.private_key:
            raise ValueError("API key and private key required for live trading")

        # Lazy import to avoid requiring web3/py-clob-client in paper trading mode
        try:
            from py_clob_client.client import ClobClient
            self._client = ClobClient(
                host="https://clob.polymarket.com",
                key=self.api_key,
                chain_id=137,  # Polygon mainnet
                funder=self.private_key,
            )
            logger.info("TradeExecutor initialized for LIVE trading")
        except ImportError:
            logger.error("py-clob-client not installed. Run: pip install py-clob-client")
            raise

    async def place_limit_order(
        self,
        signal: Signal,
        token_id: str,
        improve_price_by: float = 0.005,
    ) -> OrderResult | None:
        """Place a limit order with slight price improvement over midpoint.

        Instead of market-ordering at the midpoint (like ColdMath seems to do),
        we place a limit order slightly better than the current best bid/ask
        to get a better fill price. On illiquid weather markets, this can
        save 0.5-2% per trade.

        Args:
            signal: The trading signal
            token_id: Polymarket token ID for YES or NO
            improve_price_by: How much to improve the limit price vs midpoint
        """
        if self.dry_run:
            logger.info(
                "DRY RUN: Would place %s limit order for $%.0f @ %.3f on %s",
                signal.recommended_side.value,
                signal.recommended_size,
                signal.market_prob,
                signal.market_id[:30],
            )
            return OrderResult(
                order_id="dry_run",
                market_id=signal.market_id,
                side=signal.recommended_side.value,
                size_usd=signal.recommended_size,
                limit_price=signal.market_prob,
                status="dry_run",
            )

        if self._client is None:
            logger.error("Client not initialized")
            return None

        # Calculate limit price with improvement
        if signal.recommended_side.value == "YES":
            # Buying YES: bid slightly below ask
            limit_price = signal.market_prob - improve_price_by
        else:
            # Buying NO: bid at (1 - market_prob) - improvement
            limit_price = (1.0 - signal.market_prob) - improve_price_by

        limit_price = max(0.01, min(0.99, limit_price))

        # Calculate number of shares (contracts)
        shares = signal.recommended_size / limit_price

        try:
            # Place the order via CLOB client
            order = await self._client.create_and_post_order(
                token_id=token_id,
                price=limit_price,
                size=shares,
                side="BUY",
            )

            order_id = order.get("orderID", "unknown")
            logger.info(
                "LIVE ORDER: %s %s $%.0f @ %.3f (limit) | order_id=%s",
                signal.recommended_side.value,
                signal.city_id,
                signal.recommended_size,
                limit_price,
                order_id,
            )

            return OrderResult(
                order_id=order_id,
                market_id=signal.market_id,
                side=signal.recommended_side.value,
                size_usd=signal.recommended_size,
                limit_price=limit_price,
                status="pending",
            )
        except Exception as e:
            logger.error("Failed to place order: %s", e)
            return None

    async def cancel_stale_orders(self, max_age_seconds: int = 300) -> int:
        """Cancel orders that haven't filled within max_age_seconds."""
        if self.dry_run or self._client is None:
            return 0

        # Implementation would track open orders and cancel stale ones
        logger.info("Checking for stale orders (max age: %ds)", max_age_seconds)
        return 0
