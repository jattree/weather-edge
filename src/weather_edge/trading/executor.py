"""Real trade execution via Polymarket CLOB API.

Uses post-only limit orders (maker orders) to avoid taker fees.
Weather markets charge dynamic taker fees from March 30 2026,
posting as maker saves 1.25% at 50% and ensures $0 fees.

Corrected integration based on py-clob-client v0.34.6:
- ClobClient(host, chain_id, key=PRIVATE_KEY, funder=WALLET_ADDRESS)
- set_api_creds(ApiCreds(api_key, api_secret, api_passphrase))
- OrderArgs(token_id, price, size, side), size is in SHARES not USD
- post_order(signed_order, orderType=OrderType.GTC, post_only=True)
- All client methods are synchronous, wrapped with asyncio.run_in_executor
"""
from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from weather_edge.analysis.edge import Signal
from weather_edge.trading.fees import calculate_taker_fee
from weather_edge.trading.kill_switch import (
    is_kill_switch_active,
    track_open_order,
    untrack_order,
)

logger = logging.getLogger(__name__)

# Polymarket constraints
MIN_ORDER_SHARES: float = 5.0  # Minimum order size on Polymarket
TICK_SIZE: float = 0.01  # Price tick size (1 cent)


def _round_price(price: float) -> float:
    """Round price to valid Polymarket tick size."""
    return round(max(0.01, min(0.99, price)), 2)


def _floor_shares(shares: float) -> float:
    """Floor shares to 2 decimal places to prevent API rejection."""
    return math.floor(shares * 100) / 100.0


@dataclass
class OrderResult:
    """Result of an order placement."""
    order_id: str
    market_id: str
    side: str
    size_usd: float
    size_shares: float
    limit_price: float
    status: str  # 'pending', 'filled', 'partial', 'cancelled', 'post_only_reject', 'rejected'
    is_maker: bool = True
    taker_fee_avoided: float = 0.0
    filled_price: float | None = None
    filled_at: datetime | None = None
    tx_hash: str | None = None
    reject_reason: str = ""
    raw_response: dict = field(default_factory=dict)


class TradeExecutor:
    """Executes real trades on Polymarket via CLOB API.

    All py-clob-client methods are synchronous (built on requests).
    We wrap them with asyncio.run_in_executor to avoid blocking the event loop.
    """

    def __init__(
        self,
        private_key: str | None = None,
        wallet_address: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        api_passphrase: str | None = None,
        signature_type: int = 2,
        dry_run: bool = True,
        post_only: bool = True,
        max_shares: float | None = None,
    ):
        self.private_key = private_key
        self.wallet_address = wallet_address
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.signature_type = signature_type
        self.dry_run = dry_run
        self.post_only = post_only
        self.max_shares = max_shares  # For graduated testing (5/20/50/None)
        self._client = None

    async def initialize(self) -> None:
        """Initialize the Polymarket CLOB client."""
        if self.dry_run:
            logger.info("TradeExecutor running in DRY RUN mode")
            return

        if not self.private_key or not self.wallet_address:
            raise ValueError(
                "private_key and wallet_address required for live trading"
            )

        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds

            # ClobClient constructor:
            #   key = wallet private key (0x...)
            #   funder = public wallet address holding funds
            #   signature_type = 2 for EOA wallets, 1 for proxy/Magic Link
            self._client = ClobClient(
                host="https://clob.polymarket.com",
                chain_id=137,  # Polygon mainnet
                key=self.private_key,
                signature_type=self.signature_type,
                funder=self.wallet_address,
            )

            # L2 API credentials, required for order placement
            if self.api_key and self.api_secret and self.api_passphrase:
                creds = ApiCreds(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    api_passphrase=self.api_passphrase,
                )
                self._client.set_api_creds(creds)
            else:
                # Derive API creds from wallet key
                loop = asyncio.get_running_loop()
                creds = await loop.run_in_executor(
                    None, self._client.create_or_derive_api_creds,
                )
                self._client.set_api_creds(creds)
                logger.info("Derived API credentials from wallet key")

            logger.info(
                "TradeExecutor initialized for LIVE trading "
                "(wallet=%s, sig_type=%d, max_shares=%s)",
                self.wallet_address[:10] + "..." if self.wallet_address else "?",
                self.signature_type,
                self.max_shares or "unlimited",
            )

        except ImportError:
            logger.error(
                "py-clob-client not installed. Run: pip install py-clob-client"
            )
            raise

    async def check_balance(self) -> float | None:
        """Check USDC balance on Polygon. Returns balance or None on error."""
        if self.dry_run or self._client is None:
            return None

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                self._client.get_balance_allowance,
                {"asset_type": "COLLATERAL"},
            )
            balance = float(result.get("balance", 0)) / 1e6  # USDC has 6 decimals
            logger.info("USDC balance: $%.2f", balance)
            return balance
        except Exception as e:
            logger.error("Balance check failed: %s", e)
            return None

    async def place_limit_order(
        self,
        signal: Signal,
        token_id: str,
        improve_price_by: float = 0.005,
    ) -> OrderResult | None:
        """Place a post-only limit order to avoid taker fees.

        Args:
            signal: The trading signal.
            token_id: Polymarket token ID for YES or NO.
            improve_price_by: How much to improve the limit price vs midpoint.
        """
        # === KILL SWITCH CHECK ===
        if is_kill_switch_active():
            logger.warning(
                "KILL SWITCH ACTIVE, blocking order for %s %s",
                signal.city_id, signal.description[:40],
            )
            return None

        # Calculate the taker fee we'd avoid by being maker
        taker_fee_avoided = calculate_taker_fee(
            signal.market_prob, signal.recommended_size,
        )

        # Calculate limit price with improvement
        if signal.recommended_side.value == "YES":
            limit_price = signal.market_prob - improve_price_by
        else:
            limit_price = (1.0 - signal.market_prob) - improve_price_by

        limit_price = _round_price(limit_price)

        # Calculate shares from USD size, floor to 2dp
        shares = _floor_shares(signal.recommended_size / limit_price)

        # Enforce graduated testing cap
        if self.max_shares is not None and shares > self.max_shares:
            shares = _floor_shares(self.max_shares)
            logger.info(
                "GRADUATED CAP: capped %s to %.0f shares (max=%s)",
                signal.city_id, shares, self.max_shares,
            )

        # Enforce Polymarket minimum
        if shares < MIN_ORDER_SHARES:
            logger.info(
                "ORDER TOO SMALL: %s %.1f shares < minimum %d, skipping",
                signal.city_id, shares, MIN_ORDER_SHARES,
            )
            return OrderResult(
                order_id="too_small",
                market_id=signal.market_id,
                side=signal.recommended_side.value,
                size_usd=round(shares * limit_price, 2),
                size_shares=shares,
                limit_price=limit_price,
                status="rejected",
                reject_reason=f"Size {shares} < minimum {MIN_ORDER_SHARES}",
            )

        actual_usd = round(shares * limit_price, 2)

        if self.dry_run:
            logger.info(
                "DRY RUN: %s %s %.0f shares @ %.3f ($%.2f) on %s "
                "(post_only=%s, taker_fee_avoided=$%.2f)",
                "MAKER" if self.post_only else "TAKER",
                signal.recommended_side.value,
                shares,
                limit_price,
                actual_usd,
                signal.city_id,
                self.post_only,
                taker_fee_avoided,
            )
            return OrderResult(
                order_id="dry_run",
                market_id=signal.market_id,
                side=signal.recommended_side.value,
                size_usd=actual_usd,
                size_shares=shares,
                limit_price=limit_price,
                status="dry_run",
                is_maker=self.post_only,
                taker_fee_avoided=round(taker_fee_avoided, 4),
            )

        if self._client is None:
            logger.error("Client not initialized, cannot place order")
            return None

        # === SECOND KILL SWITCH CHECK (race condition guard) ===
        if is_kill_switch_active():
            logger.warning("KILL SWITCH activated during order prep, aborting")
            return None

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY

            order_args = OrderArgs(
                token_id=token_id,
                price=limit_price,
                size=shares,
                side=BUY,
            )

            loop = asyncio.get_running_loop()

            # Step 1: Create/sign the order (CPU-bound, fast)
            signed_order = await loop.run_in_executor(
                None, self._client.create_order, order_args,
            )

            # Step 2: Post to exchange (network-bound)
            # post_only=True ensures maker status; orderType=GTC keeps it resting
            response = await loop.run_in_executor(
                None,
                lambda: self._client.post_order(
                    signed_order,
                    orderType=OrderType.GTC,
                    post_only=self.post_only,
                ),
            )

            # Parse response
            if not isinstance(response, dict):
                response = {"raw": str(response)}

            order_id = response.get("orderID", response.get("id", "unknown"))
            status = response.get("status", "pending")

            # Detect rejection
            if status in ("REJECTED", "POST_ONLY_VIOLATION"):
                logger.warning(
                    "ORDER REJECTED: %s %s %.0f shares @ %.3f, %s",
                    signal.recommended_side.value,
                    signal.city_id,
                    shares,
                    limit_price,
                    response.get("reason", status),
                )
                return OrderResult(
                    order_id=order_id,
                    market_id=signal.market_id,
                    side=signal.recommended_side.value,
                    size_usd=actual_usd,
                    size_shares=shares,
                    limit_price=limit_price,
                    status="post_only_reject" if "POST_ONLY" in status else "rejected",
                    is_maker=False,
                    reject_reason=response.get("reason", status),
                    raw_response=response,
                )

            # Track for kill switch mass-cancel
            if order_id and order_id != "unknown":
                track_open_order(order_id)

            logger.info(
                "LIVE ORDER PLACED: %s %s %.0f shares @ %.3f ($%.2f) | "
                "order_id=%s | post_only=%s | taker_fee_avoided=$%.2f",
                signal.recommended_side.value,
                signal.city_id,
                shares,
                limit_price,
                actual_usd,
                order_id,
                self.post_only,
                taker_fee_avoided,
            )

            return OrderResult(
                order_id=order_id,
                market_id=signal.market_id,
                side=signal.recommended_side.value,
                size_usd=actual_usd,
                size_shares=shares,
                limit_price=limit_price,
                status="pending",
                is_maker=self.post_only,
                taker_fee_avoided=round(taker_fee_avoided, 4),
                raw_response=response,
            )

        except Exception as e:
            logger.error(
                "LIVE ORDER FAILED: %s %s, %s",
                signal.city_id, signal.description[:40], e,
            )
            return None

    async def get_order_status(self, order_id: str) -> dict | None:
        """Poll order status for fill tracking."""
        if self.dry_run or self._client is None:
            return None

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, self._client.get_order, order_id,
            )

            status = result.get("status", "unknown") if isinstance(result, dict) else "unknown"

            # If fully filled or cancelled, stop tracking
            if status in ("FILLED", "CANCELLED", "EXPIRED"):
                untrack_order(order_id)

            return result if isinstance(result, dict) else {"raw": str(result)}
        except Exception as e:
            logger.error("Failed to get order %s: %s", order_id, e)
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a single order."""
        if self.dry_run or self._client is None:
            return True

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._client.cancel, order_id)
            untrack_order(order_id)
            logger.info("Cancelled order %s", order_id)
            return True
        except Exception as e:
            logger.error("Failed to cancel %s: %s", order_id, e)
            return False

    async def cancel_all_orders(self) -> int:
        """Cancel all open orders on Polymarket."""
        if self.dry_run or self._client is None:
            return 0

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, self._client.cancel_all,
            )
            count = len(result) if isinstance(result, list) else 1
            logger.warning("Cancelled %d orders via cancel_all", count)
            return count
        except Exception as e:
            logger.error("cancel_all failed: %s", e)
            return 0

    async def cancel_stale_orders(self, max_age_seconds: int = 300) -> int:
        """Cancel orders that haven't filled within max_age_seconds."""
        if self.dry_run or self._client is None:
            return 0

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._client.get_orders({"state": "OPEN"}),
            )

            if not isinstance(result, list):
                result = result.get("data", []) if isinstance(result, dict) else []

            now = datetime.now(timezone.utc)
            cancelled = 0

            for order in result:
                created = order.get("created_at") or order.get("timestamp")
                if not created:
                    continue

                try:
                    if isinstance(created, str):
                        order_time = datetime.fromisoformat(
                            created.replace("Z", "+00:00")
                        )
                    else:
                        order_time = datetime.fromtimestamp(
                            float(created), tz=timezone.utc,
                        )
                except (ValueError, TypeError):
                    continue

                age = (now - order_time).total_seconds()
                if age > max_age_seconds:
                    order_id = order.get("id") or order.get("orderID")
                    if order_id:
                        await self.cancel_order(order_id)
                        cancelled += 1
                        logger.info(
                            "Cancelled stale order %s (age=%ds > %ds)",
                            order_id, int(age), max_age_seconds,
                        )

            return cancelled
        except Exception as e:
            logger.error("Stale order check failed: %s", e)
            return 0

    async def send_heartbeat(self) -> bool:
        """Send session heartbeat to prevent order auto-cancellation.

        Should be called every ~30 seconds during active trading.
        """
        if self.dry_run or self._client is None:
            return True

        try:
            loop = asyncio.get_running_loop()

            if hasattr(self._client, "update_balance_allowance"):
                # Lightweight API call that keeps the session alive
                await loop.run_in_executor(
                    None,
                    self._client.get_balance_allowance,
                    {"asset_type": "COLLATERAL"},
                )
            elif hasattr(self._client, "get_orders"):
                # Fallback: any authenticated L2 call keeps session alive
                await loop.run_in_executor(
                    None,
                    lambda: self._client.get_orders({"state": "OPEN"}),
                )

            # Track in Redis
            try:
                from weather_edge.live_state import set_value
                set_value(
                    "heartbeat:last",
                    datetime.now(timezone.utc).isoformat(),
                    ttl=60,
                )
            except Exception:
                pass

            return True
        except Exception as e:
            logger.warning("Heartbeat failed: %s, orders may be cancelled", e)
            return False
