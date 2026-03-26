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

    def __init__(self, bankroll: float = 1000.0):
        self.trades: list[PaperTrade] = []
        self._next_id = 1
        self.bankroll = bankroll

    @property
    def capital_at_risk(self) -> float:
        """Total capital deployed in open positions."""
        return sum(t.size_usd for t in self.open_trades)

    @property
    def core_at_risk(self) -> float:
        """Capital in core strategy positions."""
        return sum(t.size_usd for t in self.open_trades if not t.description.startswith("[TAIL]"))

    @property
    def tail_at_risk(self) -> float:
        """Capital in tail bet positions."""
        return sum(t.size_usd for t in self.open_trades if t.description.startswith("[TAIL]"))

    @property
    def available_capital(self) -> float:
        """Capital remaining to deploy (bankroll - at risk + resolved P&L)."""
        return self.bankroll - self.capital_at_risk + self.total_pnl

    @property
    def core_budget(self) -> float:
        """70% of bankroll for core bets."""
        return self.bankroll * 0.70

    @property
    def tail_budget(self) -> float:
        """30% of bankroll for tail bets."""
        return self.bankroll * 0.30

    def should_trade(self, signal: Signal) -> bool:
        """Determine if a signal warrants a paper trade."""
        if signal.confidence_tier == SignalTier.LOW:
            return False
        if signal.recommended_size <= 0:
            return False

        # Enforce separate budgets for core vs tail
        is_tail = getattr(signal, "strategy", "core") == "tail"
        if is_tail:
            remaining = self.tail_budget - self.tail_at_risk
        else:
            remaining = self.core_budget - self.core_at_risk
        # Also check overall capital
        remaining = min(remaining, self.available_capital)
        if signal.recommended_size > remaining:
            return False
        return True

    def place_trade(self, signal: Signal) -> PaperTrade | None:
        """Log a paper trade from a signal."""
        if not self.should_trade(signal):
            return None

        is_tail = getattr(signal, "strategy", "core") == "tail"
        if is_tail:
            remaining = min(self.tail_budget - self.tail_at_risk, self.available_capital)
        else:
            remaining = min(self.core_budget - self.core_at_risk, self.available_capital)

        size = min(signal.recommended_size, remaining)
        if size < 1.0:
            return None

        # Tag tail bets in description for tracking
        desc = signal.description
        if is_tail:
            desc = f"[TAIL] {desc}"

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

    def resolve_trade(self, trade: PaperTrade, outcome_yes: bool) -> None:
        """Resolve a paper trade based on market outcome."""
        trade.resolved_at = datetime.now(timezone.utc)

        if trade.side == "YES":
            if outcome_yes:
                # Won: paid entry_price, received 1.0
                trade.pnl = (1.0 - trade.entry_price) * trade.size_usd
                trade.status = TradeStatus.WON
            else:
                # Lost: paid entry_price, received 0
                trade.pnl = -trade.entry_price * trade.size_usd
                trade.status = TradeStatus.LOST
        else:  # NO
            if not outcome_yes:
                # Won: paid (1 - entry_price), received 1.0
                trade.pnl = trade.entry_price * trade.size_usd
                trade.status = TradeStatus.WON
            else:
                # Lost: paid (1 - entry_price), received 0
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
        """Return summary statistics."""
        closed = self.closed_trades
        tail_trades = [t for t in self.trades if t.description and t.description.startswith("[TAIL]")]
        core_trades = [t for t in self.trades if not t.description or not t.description.startswith("[TAIL]")]
        tail_pnl = sum(t.pnl for t in tail_trades if t.pnl is not None)
        core_pnl = sum(t.pnl for t in core_trades if t.pnl is not None)
        return {
            "total_trades": len(self.trades),
            "open": len(self.open_trades),
            "closed": len(closed),
            "wins": sum(1 for t in closed if t.status == TradeStatus.WON),
            "losses": sum(1 for t in closed if t.status == TradeStatus.LOST),
            "total_pnl": round(self.total_pnl, 2),
            "win_rate": round(self.win_rate * 100, 1),
            "core_trades": len(core_trades),
            "core_pnl": round(core_pnl, 2),
            "core_at_risk": round(self.core_at_risk, 2),
            "tail_trades": len(tail_trades),
            "tail_pnl": round(tail_pnl, 2),
            "tail_at_risk": round(self.tail_at_risk, 2),
        }
