"""Paper trading logger, records what we WOULD have traded."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from weather_edge.analysis.contracts import validate_pool_budget, validate_reserve_pot
from weather_edge.analysis.edge import Signal
from weather_edge.models.enums import SignalTier, TradeStatus

logger = logging.getLogger(__name__)


@dataclass
class PaperTrade:
    """A paper trade record."""
    trade_id: int | None = None
    signal_id: int | None = None
    market_id: str = ""
    city_id: str = ""
    side: str = ""
    size_usd: float = 0.0
    entry_price: float = 0.0
    placed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    description: str = ""
    exit_price: float | None = None
    resolved_at: datetime | None = None
    pnl: float | None = None
    status: TradeStatus = TradeStatus.OPEN


class PaperTrader:
    """Manages paper trades in-memory (persisted to DB via scheduler)."""

    def __init__(self, bankroll: float = 2000.0):
        self.trades: list[PaperTrade] = []
        self._next_id = 1
        self.bankroll = bankroll

        # Pool allocation (Gemini-validated 60/30/10)
        from weather_edge.config import settings
        self._pool_today_pct = settings.pool_today_pct
        self._pool_tomorrow_pct = settings.pool_tomorrow_pct
        self._pool_penny_pct = settings.pool_penny_pct

    @property
    def capital_at_risk(self) -> float:
        """Total capital deployed in open positions."""
        return sum(t.size_usd for t in self.open_trades)

    def _pool_at_risk(self, tag: str) -> float:
        """Capital at risk in a specific pool."""
        return sum(t.size_usd for t in self.open_trades if tag in (t.description or ""))

    @property
    def today_at_risk(self) -> float:
        return self._pool_at_risk("[TODAY]")

    @property
    def tomorrow_at_risk(self) -> float:
        return self._pool_at_risk("[TOMORROW]")

    @property
    def penny_at_risk(self) -> float:
        return self._pool_at_risk("[PENNY]")

    # Legacy aliases for dashboard compatibility
    @property
    def core_at_risk(self) -> float:
        return self.today_at_risk + self.tomorrow_at_risk

    @property
    def tail_at_risk(self) -> float:
        return self.penny_at_risk

    @property
    def available_capital(self) -> float:
        """Capital remaining to deploy.

        Capped at bankroll, profits don't inflate deployment capacity.
        """
        raw = self.bankroll - self.capital_at_risk + self.total_pnl
        return min(raw, self.bankroll - self.capital_at_risk)

    @property
    def today_budget(self) -> float:
        """60%, same-day markets, recycles nightly."""
        return self.bankroll * self._pool_today_pct

    @property
    def tomorrow_budget(self) -> float:
        """30%, tomorrow conviction bets."""
        return self.bankroll * self._pool_tomorrow_pct

    @property
    def penny_budget(self) -> float:
        """10%, penny sweep tail bets."""
        return self.bankroll * self._pool_penny_pct

    # Legacy aliases
    @property
    def core_budget(self) -> float:
        return self.today_budget + self.tomorrow_budget

    @property
    def tail_budget(self) -> float:
        return self.penny_budget

    # Reserve 10% of bankroll for high-conviction sniper trades
    RESERVE_PCT = 0.10

    def should_trade(self, signal: Signal) -> bool:
        """Determine if a signal warrants a paper trade."""
        if signal.confidence_tier == SignalTier.LOW:
            return False
        if signal.recommended_size <= 0:
            return False

        # Contract: total capital at risk must not exceed bankroll
        budget_check = validate_pool_budget(self.capital_at_risk, self.bankroll)
        if not budget_check.valid:
            logger.warning("CONTRACT [%s]: %s", budget_check.code, budget_check.error)
            return False

        # Portfolio-level risk controls
        from weather_edge.analysis.risk_controls import (
            _circuit_breaker,
            check_correlation_limit,
            check_gross_exposure,
            get_active_profile,
        )
        profile = get_active_profile()
        nav = self.bankroll + self.total_pnl

        # Circuit breaker
        _circuit_breaker.update(nav, profile)
        cb_mult = _circuit_breaker.get_size_multiplier(profile)
        if cb_mult <= 0:
            logger.warning("CIRCUIT BREAKER: trading killed, %s",
                           _circuit_breaker.kill_reason)
            return False
        if cb_mult < 1.0:
            signal.recommended_size = round(
                signal.recommended_size * cb_mult, 2
            )

        # Correlation limit
        allowed, max_size, reason = check_correlation_limit(
            signal.city_id, signal.recommended_size,
            self.trades, nav, profile,
        )
        if not allowed:
            logger.warning(reason)
            return False
        if max_size < signal.recommended_size:
            logger.info(reason)
            signal.recommended_size = round(max_size, 2)

        # Gross exposure cap
        allowed, max_size, reason = check_gross_exposure(
            signal.recommended_size, self.capital_at_risk,
            nav, profile,
        )
        if not allowed:
            logger.warning(reason)
            return False
        if max_size < signal.recommended_size:
            logger.info(reason)
            signal.recommended_size = round(max_size, 2)

        # Enforce three-pool budgets
        strategy = getattr(signal, "strategy", "core")
        if strategy == "tail":
            remaining = self.penny_budget - self.penny_at_risk
        else:
            # Core bets: check today + tomorrow combined budget
            remaining = self.core_budget - self.core_at_risk

        # Contract: reserve pot check, uses active risk profile
        tier_name = signal.confidence_tier.value
        effective_reserve = profile.reserve_pct
        reserve_check = validate_reserve_pot(
            self.available_capital, self.bankroll, effective_reserve, tier_name
        )
        available = self.available_capital
        if not reserve_check.valid:
            logger.warning("CONTRACT [%s]: %s", reserve_check.code, reserve_check.error)
            available = 0

        if signal.confidence_tier != SignalTier.HIGH:
            reserve = self.bankroll * effective_reserve
            available = max(0, available - reserve)

        remaining = min(remaining, available)
        if signal.recommended_size > remaining:
            return False
        return True

    def place_trade(self, signal: Signal) -> PaperTrade | None:
        """Log a paper trade from a signal."""
        if not self.should_trade(signal):
            return None

        # Dedup: don't place same market+side twice in same session
        for existing in self.open_trades:
            same_market = existing.market_id == signal.market_id
            same_side = existing.side == signal.recommended_side.value
            if same_market and same_side:
                return None  # Already have this position

        strategy = getattr(signal, "strategy", "core")
        if strategy == "tail":
            remaining = min(self.penny_budget - self.penny_at_risk, self.available_capital)
            pool_tag = "[PENNY]"
        else:
            remaining = min(self.core_budget - self.core_at_risk, self.available_capital)
            # Tag as TODAY or TOMORROW based on hours to resolution
            hours = getattr(signal, "hours_to_resolution", None)
            if hours is not None and hours <= 18:
                pool_tag = "[TODAY]"
            else:
                pool_tag = "[TOMORROW]"

        size = min(signal.recommended_size, remaining)
        if size < 1.0:
            return None

        # Enforce penny sweep min/max from config
        if strategy == "tail":
            from weather_edge.config import settings
            size = max(size, settings.penny_min_position)
            size = min(size, settings.penny_max_position)
            size = min(size, remaining)  # Re-check after clamping
            if size < settings.penny_min_position:
                return None

        # Leverage cap: reject trades where effective cost per share is too low
        # The Dallas incident: NO at market_prob=0.9885 → effective 1.15¢ → 6,261 shares from $72
        # For penny strategy trades, low prices are expected so use a looser cap
        entry = signal.market_prob
        if signal.recommended_side.value == "NO":
            effective_price = 1.0 - entry
        else:
            effective_price = entry
        max_leverage = 50 if strategy == "tail" else 20  # penny: 50x, core: 20x
        if effective_price > 0 and (1.0 / effective_price) > max_leverage:
            logger.warning(
                "LEVERAGE CAP: %s %s, eff price %.4f = %.0fx (max %dx)",
                signal.recommended_side.value, signal.city_id,
                effective_price, 1.0 / effective_price, max_leverage,
            )
            return None

        desc = f"{pool_tag} {signal.description}"

        trade = PaperTrade(
            trade_id=self._next_id,
            market_id=signal.market_id,
            city_id=signal.city_id,
            side=signal.recommended_side.value,
            size_usd=size,
            entry_price=signal.market_prob,
            description=desc,
        )
        self._next_id += 1
        self.trades.append(trade)

        logger.info(
            "PAPER TRADE: %s %s $%.0f @ %.2f | edge=%.1f%% conf=%.0f%% | %s",
            trade.side,
            trade.city_id,
            trade.size_usd,
            trade.entry_price,
            signal.edge * 100,
            signal.model_confidence * 100,
            trade.description[:50],
        )
        return trade

    def place_spread_trade(self, signal: Signal, hedge) -> PaperTrade | None:
        """Log a spread/hedge paper trade paired with a directional trade.

        This simulates ColdMath's spread capture: buy the opposite side
        so if both fill, we can merge for guaranteed profit.
        """
        remaining = self.available_capital
        size = min(hedge.cost, remaining)
        if size < 1.0:
            return None

        trade = PaperTrade(
            trade_id=self._next_id,
            market_id=hedge.market_id,
            city_id=hedge.city_id,
            side=hedge.side,
            size_usd=size,
            entry_price=hedge.limit_price,
            description=f"[SPREAD] {hedge.description}",
        )
        self._next_id += 1
        self.trades.append(trade)

        logger.info(
            "SPREAD TRADE: %s %s $%.0f @ %.2f | guaranteed=$%.2f | %s",
            trade.side, trade.city_id, trade.size_usd,
            trade.entry_price, hedge.guaranteed_profit,
            trade.description[:50],
        )
        return trade

    def resolve_trade(self, trade: PaperTrade, outcome_yes: bool) -> None:
        """Resolve a paper trade based on market outcome."""
        trade.resolved_at = datetime.now(timezone.utc)

        # Spread trades: find the paired directional trade and simulate merge
        if "[SPREAD]" in (trade.description or ""):
            # Find the directional trade on the same market
            paired = None
            for t in self.trades:
                same_mkt = t.market_id == trade.market_id
                diff_id = t.trade_id != trade.trade_id
                not_spread = "[SPREAD]" not in (t.description or "")
                if same_mkt and diff_id and not_spread:
                    paired = t
                    break

            if paired:
                # Merge simulation: YES cost + NO cost, payout = $1/share
                # Both sides together always pay $1, so profit = shares - total_cost
                total_cost = trade.entry_price + paired.entry_price
                if total_cost < 1.0:
                    # Guaranteed spread profit, split evenly between both legs
                    t_shares = trade.size_usd / trade.entry_price if trade.entry_price > 0 else 0
                    p_shares = paired.size_usd / paired.entry_price if paired.entry_price > 0 else 0
                    shares = min(t_shares, p_shares)
                    total_profit = (1.0 - total_cost) * shares
                    half_profit = total_profit / 2

                    trade.pnl = half_profit
                    trade.status = TradeStatus.WON
                    trade.exit_price = 1.0

                    # Also update the paired trade so full P&L is captured
                    paired.pnl = half_profit
                    paired.status = TradeStatus.WON
                    paired.exit_price = 1.0
                    paired.resolved_at = trade.resolved_at

                    logger.info(
                        "MERGE SIM: %s %s, spread profit $%.2f (cost %.2f + %.2f = %.2f < $1)",
                        trade.city_id, trade.market_id[:20],
                        total_profit, trade.entry_price, paired.entry_price, total_cost,
                    )
                    return

        if trade.side == "YES":
            # YES trade: paid entry_price per share, shares = size_usd / entry_price
            shares = trade.size_usd / trade.entry_price if trade.entry_price > 0 else 0
            if outcome_yes:
                # Win: each share pays $1, profit = shares - cost
                trade.pnl = shares - trade.size_usd
                trade.status = TradeStatus.WON
            else:
                # Lose: shares worth $0, lose entire cost
                trade.pnl = -trade.size_usd
                trade.status = TradeStatus.LOST
        else:  # NO
            # NO trade: paid (1 - entry_price) per share, shares = size_usd / (1 - entry_price)
            no_price = 1.0 - trade.entry_price
            shares = trade.size_usd / no_price if no_price > 0 else 0
            if not outcome_yes:
                # Win: each share pays $1, profit = shares - cost
                trade.pnl = shares - trade.size_usd
                trade.status = TradeStatus.WON
            else:
                # Lose: shares worth $0, lose entire cost
                trade.pnl = -trade.size_usd
                trade.status = TradeStatus.LOST

        trade.exit_price = 1.0 if outcome_yes else 0.0

        logger.info(
            "RESOLVED: %s %s => %s P&L=$%.2f",
            trade.side, trade.city_id, trade.status.value, trade.pnl,
        )

    def close_position(self, trade: PaperTrade, current_price: float) -> None:
        """Close an open position at current market price (sell back on Polymarket).

        Smart close logic: only sell if we'd lock in a profit or break even.
        On small-probability bets (entry < $0.10), hold to resolution,
        selling a $0.05 token at $0.02 locks in a 60% loss, but holding
        gives you a shot at the $1.00 payout. The downside is already capped.
        """
        if trade.status != TradeStatus.OPEN:
            return

        # Calculate what P&L would be if we close now
        if trade.side == "YES":
            # Shares = size / entry_price. Sell at current_price.
            shares = trade.size_usd / trade.entry_price if trade.entry_price > 0 else 0
            potential_pnl = (current_price - trade.entry_price) * shares
        else:
            # NO shares = size / (1 - entry_price). Sell at (1 - current_price).
            entry_no_price = 1.0 - trade.entry_price
            exit_no_price = 1.0 - current_price
            shares = trade.size_usd / entry_no_price if entry_no_price > 0 else 0
            potential_pnl = (exit_no_price - entry_no_price) * shares

        # Only close if profitable or breakeven.
        # On losing positions, hold to resolution, downside is capped on binary markets.
        if potential_pnl < 0:
            logger.info(
                "HOLD: %s %s, would lose $%.2f closing now. Holding to resolution.",
                trade.side, trade.city_id, abs(potential_pnl),
            )
            return

        trade.resolved_at = datetime.now(timezone.utc)
        trade.exit_price = current_price
        trade.pnl = potential_pnl
        trade.status = TradeStatus.WON if trade.pnl >= 0 else TradeStatus.LOST

        logger.info(
            "CLOSED: %s %s @ %.3f -> %.3f P&L=$%.2f",
            trade.side, trade.city_id, trade.entry_price, current_price, trade.pnl,
        )

    def close_all_positions(self, current_prices: dict[str, float] | None = None) -> float:
        """Close all open positions at current market prices.

        Args:
            current_prices: Dict of market_id -> current YES midpoint.
                           If None, closes at entry price (breakeven).
        Returns:
            Total P&L from closing all positions.
        """
        total_closed_pnl = 0.0
        for trade in self.open_trades:
            price = trade.entry_price  # Default: close at entry (breakeven)
            if current_prices and trade.market_id in current_prices:
                price = current_prices[trade.market_id]
            self.close_position(trade, price)
            if trade.pnl is not None:
                total_closed_pnl += trade.pnl

        logger.info("Closed all positions. Total P&L from closes: $%.2f", total_closed_pnl)
        return total_closed_pnl

    def reset_session(self, bankroll: float | None = None) -> dict:
        """Reset for a new session. Returns final stats from the old session."""
        final_stats = self.summary()
        self.trades = []
        self._next_id = 1
        if bankroll is not None:
            self.bankroll = bankroll
        logger.info("Session reset. New bankroll: $%.0f", self.bankroll)
        return final_stats

    @property
    def open_trades(self) -> list[PaperTrade]:
        return [t for t in self.trades if t.status == TradeStatus.OPEN]

    @property
    def closed_trades(self) -> list[PaperTrade]:
        return [t for t in self.trades if t.status != TradeStatus.OPEN]

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades if t.pnl is not None)

    @property
    def win_rate(self) -> float:
        closed = self.closed_trades
        if not closed:
            return 0.0
        wins = sum(1 for t in closed if t.status == TradeStatus.WON)
        return wins / len(closed)

    def summary(self) -> dict:
        """Return summary statistics with three-pool breakdown."""
        closed = self.closed_trades
        today_trades = [t for t in self.trades if "[TODAY]" in (t.description or "")]
        tomorrow_trades = [t for t in self.trades if "[TOMORROW]" in (t.description or "")]
        penny_trades = [t for t in self.trades if "[PENNY]" in (t.description or "")]
        today_pnl = sum(t.pnl for t in today_trades if t.pnl is not None)
        tomorrow_pnl = sum(t.pnl for t in tomorrow_trades if t.pnl is not None)
        penny_pnl = sum(t.pnl for t in penny_trades if t.pnl is not None)
        return {
            "total_trades": len(self.trades),
            "open": len(self.open_trades),
            "closed": len(closed),
            "wins": sum(1 for t in closed if t.status == TradeStatus.WON),
            "losses": sum(1 for t in closed if t.status == TradeStatus.LOST),
            "total_pnl": round(self.total_pnl, 2),
            "win_rate": round(self.win_rate * 100, 1),
            # Three-pool breakdown
            "today_trades": len(today_trades),
            "today_pnl": round(today_pnl, 2),
            "today_at_risk": round(self.today_at_risk, 2),
            "today_budget": round(self.today_budget, 2),
            "tomorrow_trades": len(tomorrow_trades),
            "tomorrow_pnl": round(tomorrow_pnl, 2),
            "tomorrow_at_risk": round(self.tomorrow_at_risk, 2),
            "tomorrow_budget": round(self.tomorrow_budget, 2),
            "penny_trades": len(penny_trades),
            "penny_pnl": round(penny_pnl, 2),
            "penny_at_risk": round(self.penny_at_risk, 2),
            "penny_budget": round(self.penny_budget, 2),
            # Legacy aliases for dashboard
            "core_trades": len(today_trades) + len(tomorrow_trades),
            "core_pnl": round(today_pnl + tomorrow_pnl, 2),
            "core_at_risk": round(self.core_at_risk, 2),
            "tail_trades": len(penny_trades),
            "tail_pnl": round(penny_pnl, 2),
            "tail_at_risk": round(self.penny_at_risk, 2),
        }
