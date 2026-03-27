"""Paper trading logger, records what we WOULD have traded."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

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
        """Capital remaining to deploy (bankroll - at risk + resolved P&L)."""
        return self.bankroll - self.capital_at_risk + self.total_pnl

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

        # Enforce three-pool budgets
        strategy = getattr(signal, "strategy", "core")
        if strategy == "tail":
            remaining = self.penny_budget - self.penny_at_risk
        else:
            # Core bets: check today + tomorrow combined budget
            remaining = self.core_budget - self.core_at_risk

        # Reserve: keep 10% of bankroll uncommitted unless this is HIGH tier
        reserve = self.bankroll * self.RESERVE_PCT
        available = self.available_capital
        if signal.confidence_tier != SignalTier.HIGH:
            available = max(0, available - reserve)

        remaining = min(remaining, available)
        if signal.recommended_size > remaining:
            return False
        return True

    def place_trade(self, signal: Signal) -> PaperTrade | None:
        """Log a paper trade from a signal."""
        if not self.should_trade(signal):
            return None

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
                if t.market_id == trade.market_id and t.trade_id != trade.trade_id and "[SPREAD]" not in (t.description or ""):
                    paired = t
                    break

            if paired:
                # Merge simulation: YES cost + NO cost, payout = $1/share
                # Both sides together always pay $1, so profit = shares - total_cost
                total_cost = trade.entry_price + paired.entry_price
                if total_cost < 1.0:
                    # Guaranteed spread profit
                    shares = min(trade.size_usd / trade.entry_price, paired.size_usd / paired.entry_price)
                    trade.pnl = (1.0 - total_cost) * shares / 2  # Split credit between both legs
                    trade.status = TradeStatus.WON
                    trade.exit_price = 1.0
                    logger.info(
                        "MERGE SIM: %s %s, spread profit $%.2f (cost %.2f + %.2f = %.2f < $1)",
                        trade.city_id, trade.market_id[:20],
                        trade.pnl * 2, trade.entry_price, paired.entry_price, total_cost,
                    )
                    return

        if trade.side == "YES":
            if outcome_yes:
                trade.pnl = (1.0 - trade.entry_price) * trade.size_usd
                trade.status = TradeStatus.WON
            else:
                trade.pnl = -trade.entry_price * trade.size_usd
                trade.status = TradeStatus.LOST
        else:  # NO
            if not outcome_yes:
                trade.pnl = trade.entry_price * trade.size_usd
                trade.status = TradeStatus.WON
            else:
                trade.pnl = -(1.0 - trade.entry_price) * trade.size_usd
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
            potential_pnl = (current_price - trade.entry_price) * trade.size_usd
        else:
            entry_no_price = 1.0 - trade.entry_price
            exit_no_price = 1.0 - current_price
            potential_pnl = (exit_no_price - entry_no_price) * trade.size_usd

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
