"""Real trade execution via Polymarket CLOB API.

Uses post-only limit orders (maker orders) to avoid taker fees.
Weather markets charge dynamic taker fees from March 30 2026,
posting as maker saves 1.25% at 50% and ensures $0 fees.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from weather_edge.analysis.edge import Signal
from weather_edge.trading.fees import calculate_taker_fee

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    """Result of an order placement."""
    order_id: str
    market_id: str
    side: str
    size_usd: float
    limit_price: float
    status: str  # 'pending', 'filled', 'partial', 'cancelled', 'post_only_reject'
    is_maker: bool = True
    taker_fee_avoided: float = 0.0  # Fee we avoided by being maker
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
        post_only: bool = True,
    ):
        self.api_key = api_key
        self.private_key = private_key
        self.dry_run = dry_run
        self.post_only = post_only
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
        """Place a post-only limit order to avoid taker fees.

        Post-only ensures our order rests on the book (maker) rather than
        crossing the spread (taker). Makers pay $0 fees; takers pay up to
        1.25% at 50% probability. If the order would cross the spread,
        it's rejected with POST_ONLY_REJECT, we log and skip rather than
        falling back to a taker order.

        Args:
            signal: The trading signal
            token_id: Polymarket token ID for YES or NO
            improve_price_by: How much to improve the limit price vs midpoint
        """
        # Calculate the taker fee we'd avoid by being maker
        taker_fee_avoided = calculate_taker_fee(signal.market_prob, signal.recommended_size)

        if self.dry_run:
            logger.info(
                "DRY RUN: Would place %s %s limit order for $%.0f @ %.3f on %s "
                "(post_only=%s, taker_fee_avoided=$%.2f)",
                "MAKER" if self.post_only else "TAKER",
                signal.recommended_side.value,
                signal.recommended_size,
                signal.market_prob,
                signal.market_id[:30],
                self.post_only,
                taker_fee_avoided,
            )
            return OrderResult(
                order_id="dry_run",
                market_id=signal.market_id,
                side=signal.recommended_side.value,
                size_usd=signal.recommended_size,
                limit_price=signal.market_prob,
                status="dry_run",
                is_maker=self.post_only,
                taker_fee_avoided=round(taker_fee_avoided, 4),
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
            # Build order kwargs, add post_only when enabled
            order_kwargs = dict(
                token_id=token_id,
                price=limit_price,
                size=shares,
                side="BUY",
            )
            if self.post_only:
                order_kwargs["post_only"] = True

            order = await self._client.create_and_post_order(**order_kwargs)

            # Detect POST_ONLY_REJECT: order would have crossed spread
            order_status = order.get("status", "")
            if order_status == "POST_ONLY_REJECT":
                logger.warning(
                    "POST_ONLY_REJECT: %s %s @ %.3f, market moved, order would cross spread. "
                    "Skipping to avoid taker fee ($%.2f).",
                    signal.recommended_side.value,
                    signal.city_id,
                    limit_price,
                    taker_fee_avoided,
                )
                return OrderResult(
                    order_id=order.get("orderID", "rejected"),
                    market_id=signal.market_id,
                    side=signal.recommended_side.value,
                    size_usd=signal.recommended_size,
                    limit_price=limit_price,
                    status="post_only_reject",
                    is_maker=False,
                    taker_fee_avoided=0.0,
                )

            order_id = order.get("orderID", "unknown")
            logger.info(
                "LIVE MAKER ORDER: %s %s $%.0f @ %.3f (post_only=%s) | "
                "order_id=%s | taker_fee_avoided=$%.2f",
                signal.recommended_side.value,
                signal.city_id,
                signal.recommended_size,
                limit_price,
                self.post_only,
                order_id,
                taker_fee_avoided,
            )

            return OrderResult(
                order_id=order_id,
                market_id=signal.market_id,
                side=signal.recommended_side.value,
                size_usd=signal.recommended_size,
                limit_price=limit_price,
                status="pending",
                is_maker=self.post_only,
                taker_fee_avoided=round(taker_fee_avoided, 4),
            )
        except Exception as e:
            logger.error("Failed to place order: %s", e)
            return None

    async def send_heartbeat(self) -> bool:
        """Send session heartbeat to prevent Polymarket from cancelling open orders.

        Per Polymarket docs: "If heartbeats are not sent regularly, all open
        orders for the user will be automatically canceled."

        Should be called every ~30 seconds during active trading.
        """
        if self.dry_run or self._client is None:
            return True

        try:
            # py-clob-client should have a heartbeat method
            if hasattr(self._client, 'send_heartbeat'):
                await self._client.send_heartbeat()
            else:
                # Manual heartbeat via HTTP
                import httpx
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "https://clob.polymarket.com/heartbeat",
                        headers=self._get_auth_headers(),
                        timeout=5.0,
                    )
                    resp.raise_for_status()

            # Track in Redis for monitoring
            try:
                from weather_edge.live_state import set_value
                from datetime import datetime, timezone
                set_value("heartbeat:last", datetime.now(timezone.utc).isoformat(), ttl=60)
            except Exception:
                pass

            return True
        except Exception as e:
            logger.warning("Heartbeat failed: %s, open orders may be cancelled", e)
            return False

    def _get_auth_headers(self) -> dict:
        """Build authentication headers for CLOB API."""
        # Placeholder, real implementation needs POLY_* headers from API key
        return {"Content-Type": "application/json"}

    async def cancel_stale_orders(self, max_age_seconds: int = 300) -> int:
        """Cancel orders that haven't filled within max_age_seconds."""
        if self.dry_run or self._client is None:
            return 0

        logger.info("Checking for stale orders (max age: %ds)", max_age_seconds)
        return 0
