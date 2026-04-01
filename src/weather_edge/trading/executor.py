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

import requests

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
            from weather_edge.config import settings as cfg
            self._client = ClobClient(
                host=cfg.polymarket_clob_url,
                chain_id=cfg.polymarket_chain_id,
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
            from py_clob_client.clob_types import AssetType, BalanceAllowanceParams

            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                self._client.get_balance_allowance,
                params,
            )
            balance = float(result.get("balance", 0)) / 1e6  # USDC has 6 decimals
            logger.info("USDC balance: $%.2f", balance)
            return balance
        except requests.RequestException as e:
            logger.error("Balance check network error: %s", e)
            return None
        except (ValueError, KeyError, TypeError) as e:
            logger.error("Balance check parse error: %s", e)
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

            # Persist to SQLite for tax compliance and dashboard.
            # This MUST succeed, a ghost trade (on exchange but not in DB)
            # breaks position tracking, duplicate prevention, and tax records.
            from weather_edge.persistence import PersistentStore
            from weather_edge.retry import retry_sync

            def _persist_trade():
                s = PersistentStore()
                try:
                    s.save_live_trade(
                        order_id=order_id,
                        market_id=signal.market_id,
                        token_id=token_id,
                        city_id=signal.city_id,
                        side=signal.recommended_side.value,
                        limit_price=limit_price,
                        size_shares=shares,
                        size_usd=actual_usd,
                        description=signal.description[:80],
                        strategy=getattr(signal, "strategy", "core"),
                        is_maker=self.post_only,
                    )
                finally:
                    s.close()

            try:
                retry_sync(
                    _persist_trade,
                    attempts=3,
                    base_delay=0.5,
                    label=f"persist_trade:{order_id[:16]}",
                )
            except Exception as e:
                # All retries failed. Order is ON the exchange but NOT in our DB.
                # Log at CRITICAL so this is impossible to miss.
                logger.critical(
                    "GHOST TRADE: order %s placed on exchange but DB write "
                    "failed after 3 retries, %s. Manual reconciliation needed.",
                    order_id, e,
                )

            # Record CLOB health
            try:
                from weather_edge.analysis.service_health import record_service_call
                record_service_call("polymarket_clob", True)
            except Exception:
                pass

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

        except (requests.ConnectionError, requests.Timeout) as e:
            logger.error(
                "LIVE ORDER NETWORK FAILURE: %s %s, %s",
                signal.city_id, signal.description[:40], e,
            )
            return None
        except requests.RequestException as e:
            logger.error(
                "LIVE ORDER API ERROR: %s %s, %s",
                signal.city_id, signal.description[:40], e,
            )
            return None
        except (ValueError, KeyError, TypeError) as e:
            logger.error(
                "LIVE ORDER PARSE ERROR: %s %s, %s",
                signal.city_id, signal.description[:40], e,
            )
            return None

    async def get_order_status(self, order_id: str) -> dict | None:
        """Poll order status for fill tracking. Retries on transient errors."""
        if self.dry_run or self._client is None:
            return None

        from weather_edge.retry import retry_async

        async def _poll():
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._client.get_order, order_id,
            )

        try:
            result = await retry_async(
                _poll,
                attempts=3,
                base_delay=2.0,
                label=f"get_order:{order_id[:16]}",
            )

            status = result.get("status", "unknown") if isinstance(result, dict) else "unknown"

            if status in ("FILLED", "CANCELLED", "EXPIRED"):
                untrack_order(order_id)

            return result if isinstance(result, dict) else {"raw": str(result)}
        except Exception as e:
            logger.error("Failed to get order %s after retries: %s", order_id, e)
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
        except requests.RequestException as e:
            logger.error("Failed to cancel %s (network): %s", order_id, e)
            return False
        except (ValueError, KeyError) as e:
            logger.error("Failed to cancel %s (parse): %s", order_id, e)
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
        except requests.RequestException as e:
            logger.error("cancel_all network error: %s", e)
            return 0
        except (ValueError, KeyError) as e:
            logger.error("cancel_all parse error: %s", e)
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
        except requests.RequestException as e:
            logger.error("Stale order check network error: %s", e)
            return 0
        except (ValueError, KeyError, TypeError) as e:
            logger.error("Stale order check parse error: %s", e)
            return 0

    async def place_sell_order(
        self,
        token_id: str,
        shares: float,
        price: float,
        market_id: str, # Required for DB persistence
        city_id: str = "",
        description: str = "",
        reference_price: float | None = None,
        force_taker: bool = False,
    ) -> OrderResult | None:
        """Place a sell order to exit a position.

        Args:
            token_id: The token to sell (YES or NO token we hold).
            shares: Number of shares to sell.
            price: Limit price (sell at this price or better).
            market_id: The market condition_id (for DB persistence).
            city_id: For logging.
            description: For logging.
            reference_price: Current market price for slippage check.
            force_taker: If True, skip post_only to guarantee fill (pays ~2% taker fee).
        """
        if is_kill_switch_active():
            logger.warning("KILL SWITCH ACTIVE, blocking sell for %s", city_id)
            return None

        shares = _floor_shares(shares)
        price = _round_price(price)

        # Slippage guard: reject if limit price drifted too far from mark
        if reference_price is not None:
            from weather_edge.config import settings
            max_slip = settings.max_slippage_pct
            if reference_price > 0 and (reference_price - price) / reference_price > max_slip:
                logger.warning(
                    "SELL BLOCKED (slippage): %s price=%.3f ref=%.3f drift=%.1f%% > max=%.1f%%",
                    city_id, price, reference_price,
                    (reference_price - price) / reference_price * 100,
                    max_slip * 100,
                )
                return None

        if shares < MIN_ORDER_SHARES:
            logger.info(
                "SELL TOO SMALL: %s %.1f shares < %d min",
                city_id, shares, MIN_ORDER_SHARES,
            )
            return None

        if self.dry_run or self._client is None:
            logger.info("DRY RUN SELL: %s %.0f shares @ %.3f", city_id, shares, price)
            return OrderResult(
                order_id="dry_run_sell",
                market_id=market_id,
                side="SELL",
                size_usd=round(shares * price, 2),
                size_shares=shares,
                limit_price=price,
                status="dry_run",
            )

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import SELL

            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=shares,
                side=SELL,
            )

            loop = asyncio.get_running_loop()
            signed_order = await loop.run_in_executor(
                None, self._client.create_order, order_args,
            )
            use_post_only = self.post_only and not force_taker
            response = await loop.run_in_executor(
                None,
                lambda: self._client.post_order(
                    signed_order,
                    orderType=OrderType.GTC,
                    post_only=use_post_only,
                ),
            )

            if force_taker:
                logger.info(
                    "EXIT TAKER MODE: %s, post_only=False, will pay taker fees",
                    city_id,
                )

            if not isinstance(response, dict):
                response = {"raw": str(response)}

            order_id = response.get("orderID", response.get("id", "unknown"))
            status = response.get("status", "pending")

            if order_id and order_id != "unknown":
                track_open_order(order_id)

            # Persist to SQLite
            from weather_edge.persistence import PersistentStore
            from weather_edge.retry import retry_sync

            def _persist_sell():
                s = PersistentStore()
                try:
                    s.save_live_trade(
                        order_id=order_id,
                        market_id=market_id,
                        token_id=token_id,
                        city_id=city_id,
                        side="SELL",
                        limit_price=price,
                        size_shares=shares,
                        size_usd=round(shares * price, 2),
                        description=description[:80],
                        strategy="exit",
                        is_maker=use_post_only,
                    )
                finally:
                    s.close()

            try:
                retry_sync(
                    _persist_sell,
                    attempts=3,
                    base_delay=0.5,
                    label=f"persist_sell:{order_id[:16]}",
                )
            except Exception as e:
                logger.critical(
                    "GHOST SELL: sell order %s placed on exchange but DB write "
                    "failed, %s", order_id, e,
                )

            try:
                from weather_edge.analysis.service_health import record_service_call
                record_service_call("polymarket_clob", True)
            except Exception:
                pass

            logger.info(
                "LIVE SELL PLACED: %s %.0f shares @ %.3f ($%.2f) | order_id=%s",
                city_id, shares, price, round(shares * price, 2), order_id,
            )

            return OrderResult(
                order_id=order_id,
                market_id=market_id,
                side="SELL",
                size_usd=round(shares * price, 2),
                size_shares=shares,
                limit_price=price,
                status="pending" if status != "REJECTED" else "rejected",
                raw_response=response,
            )

        except (requests.ConnectionError, requests.Timeout) as e:
            logger.error("LIVE SELL NETWORK FAILURE: %s, %s", city_id, e)
            return None
        except requests.RequestException as e:
            logger.error("LIVE SELL API ERROR: %s, %s", city_id, e)
            return None
        except (ValueError, KeyError, TypeError) as e:
            logger.error("LIVE SELL PARSE ERROR: %s, %s", city_id, e)
            return None

    async def send_heartbeat(self) -> bool:
        """Send session heartbeat to prevent order auto-cancellation.

        Should be called every ~30 seconds during active trading.
        """
        if self.dry_run or self._client is None:
            return True

        try:
            loop = asyncio.get_running_loop()

            if hasattr(self._client, "get_balance_allowance"):
                # Lightweight API call that keeps the session alive
                from py_clob_client.clob_types import AssetType, BalanceAllowanceParams
                params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
                await loop.run_in_executor(
                    None,
                    self._client.get_balance_allowance,
                    params,
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
                logger.debug("Heartbeat Redis tracking failed", exc_info=True)

            return True
        except requests.RequestException as e:
            logger.warning("Heartbeat network error: %s, orders may be cancelled", e)
            return False
        except (ValueError, KeyError) as e:
            logger.warning("Heartbeat parse error: %s", e)
            return False

    async def redeem_positions(self) -> int:
        """Redeem all resolved (winning) positions back to USDC.

        Calls the Polymarket CTF contract's redeemPositions function
        on Polygon for each redeemable position.

        Returns number of positions redeemed.
        """
        if self.dry_run or not self._private_key:
            return 0

        try:
            import httpx
            from web3 import Web3

            # Find redeemable positions from Data API
            proxy = "0xe23940d70793b441c9f949741daa65289947fadb"
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://data-api.polymarket.com/positions",
                    params={"user": proxy, "sizeThreshold": 0},
                    timeout=15.0,
                )
                if resp.status_code != 200:
                    return 0
                positions = resp.json()

            redeemable = [p for p in positions if p.get("redeemable")]
            if not redeemable:
                return 0

            # Connect to Polygon
            w3 = Web3(Web3.HTTPProvider("https://polygon-rpc.com"))
            account = w3.eth.account.from_key(self._private_key)

            # CTF contract ABI (just redeemPositions)
            CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
            CTF_ABI = [{
                "name": "redeemPositions",
                "type": "function",
                "inputs": [
                    {"name": "collateralToken", "type": "address"},
                    {"name": "parentCollectionId", "type": "bytes32"},
                    {"name": "conditionId", "type": "bytes32"},
                    {"name": "indexSets", "type": "uint256[]"},
                ],
                "outputs": [],
            }]

            ctf = w3.eth.contract(
                address=Web3.to_checksum_address(CTF_ADDRESS),
                abi=CTF_ABI,
            )

            USDC = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
            PARENT_COLLECTION = b"\x00" * 32
            redeemed = 0

            for pos in redeemable:
                condition_id = pos.get("conditionId", "")
                if not condition_id:
                    continue

                try:
                    cid_bytes = Web3.to_bytes(hexstr=condition_id)
                    # Index sets: [1, 2] means redeem both YES (index 0) and NO (index 1)
                    index_sets = [1, 2]

                    tx = ctf.functions.redeemPositions(
                        USDC, PARENT_COLLECTION, cid_bytes, index_sets,
                    ).build_transaction({
                        "from": account.address,
                        "nonce": w3.eth.get_transaction_count(account.address),
                        "gas": 200000,
                        "gasPrice": w3.eth.gas_price,
                        "chainId": 137,
                    })

                    signed = account.sign_transaction(tx)
                    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
                    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)

                    if receipt.status == 1:
                        title = pos.get("title", "")[:50]
                        value = float(pos.get("currentValue", 0))
                        logger.warning(
                            "REDEEMED: %s, $%.2f returned to wallet (tx: %s)",
                            title, value, tx_hash.hex()[:16],
                        )
                        redeemed += 1
                    else:
                        logger.error("REDEEM FAILED: tx reverted for %s", condition_id[:16])

                except Exception as e:
                    logger.error("REDEEM ERROR: %s, %s", condition_id[:16], e)

            return redeemed

        except Exception as e:
            logger.error("Redemption failed: %s", e)
            return 0
