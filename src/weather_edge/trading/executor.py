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
import contextlib
import contextvars
import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime

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


def chase_limit_price(old_price: float, side_mid: float, tick: float = 0.01) -> float | None:
    """Next price-chase limit: one tick above ``old_price``, capped at the mid.

    Returns None when a one-tick improvement is not possible (already at the
    mid cap), so the caller keeps the resting order instead of cancelling and
    re-placing it at the same price.
    """
    cap = math.floor(round(side_mid / tick, 6)) * tick
    new = round(min(old_price + tick, cap, 0.99), 2)
    if new <= round(old_price, 2):
        return None
    return new


def script_dry_run(execute: bool, live_mode: bool | None = None) -> bool:
    """Dry-run decision for operator scripts that can trade.

    Scripts default to dry run. A real order requires BOTH the explicit
    ``--execute`` flag AND ``LIVE_MODE=true`` in settings. Returns True when
    the script must stay in dry run.
    """
    if live_mode is None:
        from weather_edge.config import settings
        live_mode = bool(settings.live_mode)
    if not execute:
        logger.warning("DRY RUN: pass --execute (with LIVE_MODE=true) to place real orders")
        return True
    if not live_mode:
        logger.warning("DRY RUN: --execute given but LIVE_MODE is off, refusing real orders")
        return True
    return False


# When set (per asyncio task/context), every exchange-mutating call on any
# TradeExecutor becomes a no-op. Used so that operator "refresh" requests made
# while trading is STOPPED run analysis/paper only and cannot place, cancel or
# redeem anything live. Context-local: other concurrent requests (close-all,
# kill switch) are unaffected.
_live_orders_suppressed: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "live_orders_suppressed", default=False,
)


@contextlib.contextmanager
def suppress_live_orders():
    """Block live order placement/cancel/redeem inside this context."""
    token = _live_orders_suppressed.set(True)
    try:
        yield
    finally:
        _live_orders_suppressed.reset(token)


def live_orders_suppressed() -> bool:
    return _live_orders_suppressed.get()


_REJECT_STATUSES = {"REJECTED", "POST_ONLY_VIOLATION", "INVALID", "CANCELED", "CANCELLED"}


def classify_post_response(response) -> tuple[bool, str, str]:
    """Decide whether a CLOB post_order response is a rejection.

    py-clob-client returns ``{"success": bool, "errorMsg": str, "orderID": str,
    "status": ...}``. A response is treated as rejected when ``success`` is
    false, ``errorMsg`` is non-empty, the status is a reject status, or no
    order id came back.

    Returns:
        (rejected, result_status, reason) where result_status is
        "post_only_reject", "rejected" or "pending".
    """
    if not isinstance(response, dict):
        return True, "rejected", f"unparseable response: {response!r}"[:200]
    status = str(response.get("status") or "").upper()
    err = str(response.get("errorMsg") or response.get("error") or "")
    reason = response.get("reason") or err or status
    order_id = response.get("orderID") or response.get("id")
    post_only = "POST_ONLY" in status or "post-only" in err.lower() or "post only" in err.lower()
    if post_only:
        return True, "post_only_reject", reason
    if (
        response.get("success") is False
        or err
        or status in _REJECT_STATUSES
        or not order_id
        or order_id == "unknown"
    ):
        return True, "rejected", reason or "no order id returned"
    return False, "pending", ""


def _parse_post_response(response) -> tuple[bool, str, str, dict, str]:
    """Classify a post_order response and normalise it for OrderResult.

    Returns (rejected, result_status, reason, response_dict, order_id); a
    non-dict response is wrapped as ``{"raw": str(response)}`` and a missing
    order id becomes "unknown".
    """
    rejected, result_status, reason = classify_post_response(response)
    if not isinstance(response, dict):
        response = {"raw": str(response)}
    order_id = response.get("orderID") or response.get("id") or "unknown"
    return rejected, result_status, reason, response, order_id


def _live_circuit_breaker_blocks(city_id: str) -> bool:
    """LIVE CIRCUIT BREAKER: True (and logged) when orders must be blocked.

    Tripped breaker also activates the kill switch (risk_controls), this
    is the in-process backstop in case that write failed. Fails closed.
    """
    try:
        from weather_edge.analysis.risk_controls import live_circuit_breaker_multiplier
        if live_circuit_breaker_multiplier() <= 0:
            logger.warning(
                "LIVE CIRCUIT BREAKER KILLED, blocking order for %s", city_id,
            )
            return True
    except Exception:
        logger.error("Circuit breaker check failed, blocking order (fail-closed)")
        return True
    return False


def _buy_limit_price(signal: Signal, improve_price_by: float, force_taker: bool) -> float:
    """Limit price for a buy, rounded to tick.

    Taker: price ABOVE midpoint to cross the spread and guarantee fill.
    Maker: improve below market to rest on the book.
    """
    side_price = (
        signal.market_prob if signal.recommended_side.value == "YES"
        else 1.0 - signal.market_prob
    )
    if force_taker:
        # Price aggressively above midpoint to guarantee crossing the ask.
        # Weather market spreads are typically 1-3¢. Using +3¢ ensures we
        # cross even wider spreads. On Polymarket, limit buys fill at the
        # ask price (not our limit), so overshoot just means guaranteed fill.
        limit_price = side_price + 0.03
    else:
        limit_price = side_price - improve_price_by
    return _round_price(limit_price)


async def _persist_live_trade(label: str, ghost_msg: str, **trade) -> None:
    """Persist a placed live order to SQLite, retrying off the event loop.

    On final failure the order is on the exchange but not in the DB: log
    ``ghost_msg`` (formatted with order id and error) at CRITICAL.
    """
    from weather_edge.persistence import PersistentStore
    from weather_edge.retry import retry_async

    def _persist():
        s = PersistentStore()
        try:
            s.save_live_trade(**trade)
        finally:
            s.close()

    async def _persist_async():
        # SQLite write off the event loop; backoff uses asyncio.sleep
        await asyncio.to_thread(_persist)

    try:
        await retry_async(
            _persist_async,
            attempts=3,
            base_delay=0.5,
            label=label,
        )
    except Exception as e:
        logger.critical(ghost_msg, trade["order_id"], e)


def _record_clob_success() -> None:
    """Record CLOB health (best effort)."""
    try:
        from weather_edge.analysis.service_health import record_service_call
        record_service_call("polymarket_clob", True)
    except Exception:
        pass


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
        relayer_api_key: str | None = None,
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
        self.relayer_api_key = relayer_api_key
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
        force_taker: bool = False,
    ) -> OrderResult | None:
        """Place a limit order (maker by default, taker if forced).

        Args:
            signal: The trading signal.
            token_id: Polymarket token ID for YES or NO.
            improve_price_by: How much to improve the limit price vs midpoint.
            force_taker: If True, cross the spread as taker (pays fee, guarantees fill).
        """
        if live_orders_suppressed():
            logger.info(
                "LIVE ORDERS SUPPRESSED (trading stopped), not placing %s %s",
                signal.city_id, signal.description[:40],
            )
            return None

        # === KILL SWITCH CHECK ===
        if is_kill_switch_active():
            logger.warning(
                "KILL SWITCH ACTIVE, blocking order for %s %s",
                signal.city_id, signal.description[:40],
            )
            return None

        if _live_circuit_breaker_blocks(signal.city_id):
            return None

        # Calculate the taker fee we'd avoid by being maker
        taker_fee_avoided = calculate_taker_fee(
            signal.market_prob, signal.recommended_size,
        )

        limit_price = _buy_limit_price(signal, improve_price_by, force_taker)
        shares = self._buy_shares(signal, limit_price)

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
            return self._dry_run_buy(signal, limit_price, shares, actual_usd, taker_fee_avoided)

        if self._client is None:
            logger.error("Client not initialized, cannot place order")
            return None

        # === SECOND KILL SWITCH CHECK (race condition guard) ===
        if is_kill_switch_active():
            logger.warning("KILL SWITCH activated during order prep, aborting")
            return None

        return await self._place_live_buy(
            signal, token_id,
            limit_price=limit_price, shares=shares, actual_usd=actual_usd,
            taker_fee_avoided=taker_fee_avoided, force_taker=force_taker,
        )

    def _buy_shares(self, signal: Signal, limit_price: float) -> float:
        """Shares for a buy: USD size / price floored to 2dp, graduated cap applied."""
        shares = _floor_shares(signal.recommended_size / limit_price)

        # Enforce graduated testing cap
        if self.max_shares is not None and shares > self.max_shares:
            shares = _floor_shares(self.max_shares)
            logger.info(
                "GRADUATED CAP: capped %s to %.0f shares (max=%s)",
                signal.city_id, shares, self.max_shares,
            )
        return shares

    def _dry_run_buy(
        self,
        signal: Signal,
        limit_price: float,
        shares: float,
        actual_usd: float,
        taker_fee_avoided: float,
    ) -> OrderResult:
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

    async def _place_live_buy(
        self,
        signal: Signal,
        token_id: str,
        *,
        limit_price: float,
        shares: float,
        actual_usd: float,
        taker_fee_avoided: float,
        force_taker: bool,
    ) -> OrderResult | None:
        """Sign, post, track and persist a live BUY. None on exchange errors."""
        try:
            # post_only=True ensures maker status; orderType=GTC keeps it resting
            # force_taker overrides to cross the spread (pays fee, guarantees fill)
            use_post_only = False if force_taker else self.post_only
            response = await self._post_limit_order(
                token_id, limit_price, shares, "BUY", use_post_only,
            )

            # Parse response, detect rejection BEFORE tracking/persisting
            rejected, result_status, reason, response, order_id = _parse_post_response(response)

            if rejected:
                logger.warning(
                    "ORDER REJECTED: %s %s %.0f shares @ %.3f, %s",
                    signal.recommended_side.value,
                    signal.city_id,
                    shares,
                    limit_price,
                    reason,
                )
                return OrderResult(
                    order_id=order_id,
                    market_id=signal.market_id,
                    side=signal.recommended_side.value,
                    size_usd=actual_usd,
                    size_shares=shares,
                    limit_price=limit_price,
                    status=result_status,
                    is_maker=False,
                    reject_reason=reason,
                    raw_response=response,
                )

            # Track for kill switch mass-cancel
            if order_id and order_id != "unknown":
                track_open_order(order_id)

            # Persist to SQLite for tax compliance and dashboard.
            # This MUST succeed, a ghost trade (on exchange but not in DB)
            # breaks position tracking, duplicate prevention, and tax records.
            # If all retries fail the order is ON the exchange but NOT in our
            # DB, logged at CRITICAL so this is impossible to miss.
            await _persist_live_trade(
                f"persist_trade:{order_id[:16]}",
                "GHOST TRADE: order %s placed on exchange but DB write "
                "failed after 3 retries, %s. Manual reconciliation needed.",
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
                is_maker=use_post_only,
            )

            _record_clob_success()

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

    async def _post_limit_order(
        self, token_id: str, price: float, shares: float, side: str, post_only: bool,
    ):
        """Sign and post a GTC limit order, returning the raw post_order response.

        ``side`` is "BUY" or "SELL". Exceptions propagate to the caller.
        """
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY, SELL

        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=shares,
            side={"BUY": BUY, "SELL": SELL}[side],
        )

        loop = asyncio.get_running_loop()

        # Step 1: Create/sign the order (CPU-bound, fast)
        signed_order = await loop.run_in_executor(
            None, self._client.create_order, order_args,
        )

        # Step 2: Post to exchange (network-bound)
        return await loop.run_in_executor(
            None,
            lambda: self._client.post_order(
                signed_order,
                orderType=OrderType.GTC,
                post_only=post_only,
            ),
        )

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
        if live_orders_suppressed():
            logger.info("LIVE ORDERS SUPPRESSED, not cancelling %s", order_id)
            return False
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
        if live_orders_suppressed() or self.dry_run or self._client is None:
            return 0

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._client.get_orders({"state": "OPEN"}),
            )

            if not isinstance(result, list):
                result = result.get("data", []) if isinstance(result, dict) else []

            now = datetime.now(UTC)
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
                            float(created), tz=UTC,
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
            reference_price: An INDEPENDENT mark for the token being sold
                (e.g. last known Gamma price of that token). Must be the
                price of the same token as ``price``. When omitted in live
                mode the executor fetches the token's CLOB midpoint instead;
                if no reference can be obtained the live sell is refused.
            force_taker: If True, skip post_only to guarantee fill (pays ~2% taker fee).
        """
        if live_orders_suppressed():
            logger.info("LIVE ORDERS SUPPRESSED (trading stopped), not selling %s", city_id)
            return None

        if is_kill_switch_active():
            logger.warning("KILL SWITCH ACTIVE, blocking sell for %s", city_id)
            return None

        shares = _floor_shares(shares)
        price = _round_price(price)
        is_live = not (self.dry_run or self._client is None)

        if await self._sell_slippage_blocked(
            token_id, price, city_id, reference_price=reference_price, is_live=is_live,
        ):
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

        return await self._place_live_sell(
            token_id, shares, price,
            market_id=market_id, city_id=city_id, description=description,
            force_taker=force_taker,
        )

    async def _sell_slippage_blocked(
        self,
        token_id: str,
        price: float,
        city_id: str,
        *,
        reference_price: float | None,
        is_live: bool,
    ) -> bool:
        """Slippage guard: True (and logged) when the sell must be refused.

        Rejects a limit too far BELOW an independent mark of this token. Live
        mode also checks the live CLOB midpoint of the token, so a caller that
        passes its own price as the reference (or the wrong token's price)
        cannot silence the guard.
        """
        from weather_edge.config import settings
        max_slip = settings.max_slippage_pct
        references: list[tuple[str, float]] = []
        if reference_price is not None and reference_price > 0:
            references.append(("caller", float(reference_price)))
        if is_live:
            mid = await self.get_token_midpoint(token_id)
            if mid is not None and mid > 0:
                references.append(("clob_mid", mid))
            if not references:
                logger.warning(
                    "SELL BLOCKED (no reference price): %s token %s, cannot "
                    "verify limit %.3f", city_id, token_id[:16], price,
                )
                return True
        for ref_name, ref in references:
            drift = (ref - price) / ref
            if drift > max_slip:
                logger.warning(
                    "SELL BLOCKED (slippage vs %s): %s price=%.3f ref=%.3f "
                    "drift=%.1f%% > max=%.1f%%",
                    ref_name, city_id, price, ref, drift * 100, max_slip * 100,
                )
                return True
        return False

    async def _place_live_sell(
        self,
        token_id: str,
        shares: float,
        price: float,
        *,
        market_id: str,
        city_id: str,
        description: str,
        force_taker: bool,
    ) -> OrderResult | None:
        """Sign, post, track and persist a live SELL. None on exchange errors."""
        try:
            use_post_only = self.post_only and not force_taker
            response = await self._post_limit_order(
                token_id, price, shares, "SELL", use_post_only,
            )

            if force_taker:
                logger.info(
                    "EXIT TAKER MODE: %s, post_only=False, will pay taker fees",
                    city_id,
                )

            rejected, result_status, reason, response, order_id = _parse_post_response(response)

            if rejected:
                # Never track or persist a rejected sell: it is not on the book,
                # and a phantom 'open' SELL row would block re-exit and poison
                # the exit cooldown.
                logger.warning(
                    "SELL REJECTED: %s %.0f shares @ %.3f, %s",
                    city_id, shares, price, reason,
                )
                return OrderResult(
                    order_id=order_id,
                    market_id=market_id,
                    side="SELL",
                    size_usd=round(shares * price, 2),
                    size_shares=shares,
                    limit_price=price,
                    status=result_status,
                    is_maker=False,
                    reject_reason=reason,
                    raw_response=response,
                )

            track_open_order(order_id)

            # Persist to SQLite
            await _persist_live_trade(
                f"persist_sell:{order_id[:16]}",
                "GHOST SELL: sell order %s placed on exchange but DB write "
                "failed, %s",
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

            _record_clob_success()

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
                status="pending",
                is_maker=use_post_only,
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

    async def get_token_midpoint(self, token_id: str) -> float | None:
        """Live CLOB midpoint for one token (YES or NO), or None."""
        if self._client is None or not token_id:
            return None
        fn = getattr(self._client, "get_midpoint", None)
        if fn is None:
            return None
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, fn, token_id)
            if isinstance(result, dict):
                result = result.get("mid", result.get("midpoint"))
            mid = float(result)
            return mid if 0 < mid < 1 else None
        except Exception as e:
            logger.debug("Midpoint fetch failed for %s: %s", token_id[:16], e)
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
                    datetime.now(UTC).isoformat(),
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

        Uses Polymarket's Relayer API with proper PROXY wallet signing flow:
        1. Fetch redeemable positions from Data API
        2. Pre-check on-chain resolution via payoutDenominator
        3. Encode CTF.redeemPositions calldata
        4. Wrap in proxy((uint8,address,uint256,bytes)[]) encoding
        5. Get relay payload (nonce + relay address) from relayer
        6. Build proxy struct hash (rlx: prefix + fields + keccak256)
        7. Sign with EIP-191 personal sign
        8. Submit with builder API key auth headers
        9. Poll for confirmation

        Returns number of positions whose redemption was CONFIRMED on-chain.
        Submitted-but-unconfirmed or failed relayer transactions are not
        counted.
        """
        if live_orders_suppressed() or self.dry_run or not self.private_key:
            return 0

        try:
            import json as _json

            import httpx
            from eth_abi.packed import encode_packed
            from eth_account import Account
            from eth_account.messages import encode_defunct
            from eth_utils import keccak, to_bytes, to_checksum_address
            from web3 import Web3

            from weather_edge.config import settings as cfg

            # ── Constants ──────────────────────────────────────────────
            ctf_addr = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
            usdc_addr = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
            parent_collection = b"\x00" * 32
            proxy_factory = "0xaB45c5A4B0c941a2F231C04C3f49182e1A254052"
            relay_hub = "0xD216153c06E857cD7f72665E0aF1d7D82172F494"
            proxy_init_code_hash = (
                "0xd21df8dc65880a8606f09fe0ce3df9b8869287ab0b058be05aa9e8af6330a00b"
            )
            default_gas_limit = 500_000

            relayer_url = cfg.polymarket_relayer_url.rstrip("/")

            # ── Derive addresses ───────────────────────────────────────
            account = Account.from_key(self.private_key)
            eoa_address = account.address  # checksummed

            # Derive proxy wallet address (CREATE2 with packed salt)
            salt = keccak(encode_packed(["address"], [eoa_address]))
            bytecode_hash = to_bytes(hexstr=proxy_init_code_hash)
            factory_bytes = to_bytes(hexstr=proxy_factory)
            proxy_address = to_checksum_address(
                keccak(b"\xff" + factory_bytes + salt + bytecode_hash)[-20:].hex()
            )

            logger.info(
                "REDEEM: EOA=%s proxy=%s", eoa_address[:10] + "...", proxy_address[:10] + "..."
            )

            # ── Auth config ────────────────────────────────────────────
            if not self.relayer_api_key:
                logger.error("REDEEM: relayer API key not configured")
                return 0

            # ── Fetch redeemable positions ─────────────────────────────
            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    "https://data-api.polymarket.com/positions",
                    params={"user": proxy_address.lower(), "sizeThreshold": 0},
                    timeout=15.0,
                )
                if resp.status_code != 200:
                    logger.error("REDEEM: Data API returned %d", resp.status_code)
                    return 0
                positions = resp.json()

            redeemable = [
                p for p in positions
                if p.get("redeemable") and float(p.get("size", 0)) > 0
            ]
            if not redeemable:
                logger.debug("REDEEM: no redeemable positions found")
                return 0

            logger.info("REDEEM: found %d redeemable positions", len(redeemable))

            # ── On-chain resolution pre-check ──────────────────────────
            w3 = None
            for rpc in ["https://1rpc.io/matic", "https://rpc-mainnet.matic.quiknode.pro"]:
                try:
                    w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 30}))
                    w3.eth.block_number
                    break
                except Exception:
                    w3 = None

            redeem_abi = [{
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
            ctf_check_abi = [{
                "name": "payoutDenominator",
                "type": "function",
                "stateMutability": "view",
                "inputs": [{"name": "conditionId", "type": "bytes32"}],
                "outputs": [{"name": "", "type": "uint256"}],
            }]

            w3_encode = w3 or Web3()
            ctf_contract = w3_encode.eth.contract(
                address=Web3.to_checksum_address(ctf_addr),
                abi=redeem_abi + ctf_check_abi,
            )

            # ── Helper: sign request body for V2 PROXY ──────────────────
            def _sign_request_body(request: dict) -> str:
                """Sign the canonical JSON request body for Relayer V2 PROXY."""
                # Copy to avoid mutating original
                payload = request.copy()
                # Ensure signature field is not in the signed content
                payload.pop("signature", None)
                # Canonical JSON (no spaces, sorted keys)
                json_str = _json.dumps(payload, separators=(",", ":"), sort_keys=True)
                # personal_sign over the JSON string
                msg = encode_defunct(text=json_str)
                sig = Account.sign_message(msg, self.private_key).signature.hex()
                return "0x" + sig if not sig.startswith("0x") else sig

            # ── Process each redeemable position ───────────────────────
            redeemed = 0

            for pos in redeemable:
                condition_id = pos.get("conditionId", "")
                if not condition_id:
                    continue

                try:
                    cid_bytes = Web3.to_bytes(hexstr=condition_id)

                    # Pre-check: is the condition resolved on-chain?
                    if w3:
                        try:
                            denom = ctf_contract.functions.payoutDenominator(
                                cid_bytes
                            ).call()
                            if denom == 0:
                                logger.debug(
                                    "REDEEM SKIP: %s not resolved on-chain "
                                    "(payoutDenominator=0)",
                                    condition_id[:16],
                                )
                                continue
                        except Exception:
                            pass

                    # Build index_sets
                    outcome_index = pos.get("outcomeIndex")
                    if outcome_index is not None:
                        index_sets = [1 << int(outcome_index)]
                    else:
                        index_sets = (
                            [2] if pos.get("outcome", "").lower() == "no" else [1]
                        )

                    # Encode CTF.redeemPositions calldata (target is CTF, not proxy)
                    calldata = ctf_contract.encode_abi(
                        "redeemPositions",
                        [
                            Web3.to_checksum_address(usdc_addr),
                            parent_collection,
                            cid_bytes,
                            index_sets,
                        ],
                    )

                    title = pos.get("title", "")[:50]
                    size = float(pos.get("size", 0))

                    # Get relay payload (nonce + relay address)
                    async with httpx.AsyncClient() as http:
                        relay_resp = await http.get(
                            f"{relayer_url}/relay-payload",
                            params={"address": eoa_address, "type": "PROXY"},
                            timeout=15.0,
                        )
                        if relay_resp.status_code != 200:
                            logger.error(
                                "REDEEM: relay-payload failed %d: %s",
                                relay_resp.status_code,
                                relay_resp.text[:200],
                            )
                            continue
                        relay_payload = relay_resp.json()

                    nonce = str(relay_payload["nonce"])
                    relay_address = relay_payload["address"]
                    gas_limit = str(default_gas_limit)

                    # Build the V2 Relayer PROXY request
                    tx_request = {
                        "type": "PROXY",
                        "from": eoa_address,
                        "to": to_checksum_address(ctf_addr),  # Direct target
                        "proxyWallet": proxy_address,
                        "data": calldata,
                        "nonce": nonce,
                        "value": "0",
                        "signatureParams": {
                            "gasPrice": "0",
                            "gasLimit": gas_limit,
                            "relayerFee": "0",
                            "relayHub": to_checksum_address(relay_hub),
                            "relay": relay_address,
                        },
                        "metadata": f"redeem:{condition_id[:16]}",
                    }

                    # Sign the canonical JSON request body
                    tx_request["signature"] = _sign_request_body(tx_request)

                    # Auth: Relayer API key (simple headers)
                    body_str = _json.dumps(tx_request)
                    submit_headers = {
                        "Content-Type": "application/json",
                        "RELAYER_API_KEY": self.relayer_api_key or "",
                        "RELAYER_API_KEY_ADDRESS": eoa_address,  # Checksummed
                    }

                    # Submit to relayer
                    async with httpx.AsyncClient() as http:
                        submit_resp = await http.post(
                            f"{relayer_url}/submit",
                            content=body_str,
                            headers=submit_headers,
                            timeout=30.0,
                        )

                    if submit_resp.status_code in (200, 201):
                        result = submit_resp.json()
                        tx_id = result.get("transactionID", result.get("id", "?"))
                        logger.warning(
                            "REDEEM SUBMITTED: %s (%.1f shares), relayer tx: %s",
                            title,
                            size,
                            str(tx_id)[:24],
                        )

                        # Poll for confirmation (up to 60s)
                        confirmed = await self._poll_relayer_tx(
                            relayer_url, tx_id, timeout=60
                        )
                        if confirmed:
                            logger.warning(
                                "REDEEM CONFIRMED: %s, tx: %s", title, str(tx_id)[:24]
                            )
                            # Only confirmed (mined) redemptions count
                            redeemed += 1
                        else:
                            logger.warning(
                                "REDEEM NOT CONFIRMED (failed or pending after 60s): %s",
                                title,
                            )
                    else:
                        logger.error(
                            "REDEEM REJECTED: %s, %d %s",
                            condition_id[:16],
                            submit_resp.status_code,
                            submit_resp.text[:300],
                        )

                except Exception as e:
                    logger.error(
                        "REDEEM ERROR: %s, %s", condition_id[:16], e, exc_info=True
                    )

            return redeemed

        except ImportError as e:
            logger.error("Redemption import error (missing dependency): %s", e)
            return 0
        except Exception as e:
            logger.error("Redemption failed: %s", e, exc_info=True)
            return 0

    async def _poll_relayer_tx(
        self, relayer_url: str, tx_id: str, timeout: int = 60
    ) -> bool:
        """Poll relayer for transaction confirmation.

        Returns True if confirmed/mined, False if timed out or failed.
        """
        import httpx

        poll_interval = 3.0
        elapsed = 0.0
        terminal_states = {"STATE_MINED", "STATE_CONFIRMED"}
        fail_states = {"STATE_FAILED", "STATE_INVALID"}

        while elapsed < timeout:
            try:
                async with httpx.AsyncClient() as http:
                    resp = await http.get(
                        f"{relayer_url}/transaction",
                        params={"id": tx_id},
                        timeout=10.0,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        # Response is a list of transactions
                        txns = data if isinstance(data, list) else [data]
                        for txn in txns:
                            state = txn.get("state", "")
                            if state in terminal_states:
                                return True
                            if state in fail_states:
                                tx_hash = txn.get("transactionHash", "?")
                                logger.error(
                                    "REDEEM TX FAILED: %s state=%s hash=%s",
                                    tx_id[:16], state, tx_hash,
                                )
                                return False
            except Exception as e:
                logger.debug("REDEEM poll error: %s", e)

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        return False
