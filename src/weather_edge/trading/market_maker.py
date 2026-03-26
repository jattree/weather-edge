"""Market-making spread capture strategy, ColdMath's safety net.

Earns the bid-ask spread on both sides of a bucket while optionally
holding directional exposure. Works alongside the directional strategy.

How it works:
- For a bucket priced at YES=6¢ / NO=94¢ (spread = 0¢, no profit)
- But if we can buy YES at 4¢ and NO at 95¢ = 99¢ total for guaranteed $1 = 1¢ profit
- Or buy YES at 3¢ and NO at 94¢ = 97¢ total = 3¢ guaranteed profit
- Do this hundreds of times = steady income regardless of outcome

Combined with directional:
- We think YES is underpriced (model says 15%, market says 6%)
- Buy extra YES at 4-6¢ as our directional bet (from the main strategy)
- ALSO buy NO at 94-96¢ as spread capture
- If YES wins: directional bet pays huge, NO loses but was cheap insurance
- If NO wins: directional bet loses small (6¢), spread capture earns (96¢ -> $1 = 4¢)
- Net: we're always partially hedged

This module generates market-making orders to pair with directional trades.
Requires real execution (Phase 3), paper trading can only simulate the P&L.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from weather_edge.analysis.edge import Signal

logger = logging.getLogger(__name__)


@dataclass
class SpreadOrder:
    """A market-making order on the opposite side of a directional trade."""
    market_id: str
    city_id: str
    token_id: str  # YES or NO token
    side: str  # "YES" or "NO", the side we're BUYING for spread capture
    limit_price: float  # Our limit price (should be below ask for YES, above bid for NO)
    shares: float
    cost: float  # limit_price * shares
    guaranteed_profit: float  # (1.0 - total_cost_both_sides) * shares
    paired_with: str  # market_id of the directional trade this hedges
    description: str


@dataclass
class SpreadOpportunity:
    """A detected spread capture opportunity on a bucket."""
    market_id: str
    city_id: str
    yes_price: float  # Current YES midpoint
    no_price: float  # Current NO midpoint (1 - yes_price approximately)
    spread: float  # Gap between best bid and ask
    total_cost: float  # yes_limit + no_limit (should be < 1.0 for profit)
    guaranteed_profit_per_share: float
    max_shares: float  # Limited by order book depth
    description: str


class MarketMaker:
    """Generates spread capture orders to pair with directional trades.

    For paper trading: simulates expected spread P&L.
    For live trading: generates actual limit orders for both sides.
    """

    def __init__(
        self,
        spread_budget_pct: float = 0.20,  # 20% of bankroll for market-making
        min_profit_per_trade: float = 0.01,  # Minimum 1¢ guaranteed profit per share
        max_position_per_bucket: float = 100.0,  # Max $ per bucket for spread capture
    ):
        self.spread_budget_pct = spread_budget_pct
        self.min_profit_per_trade = min_profit_per_trade
        self.max_position_per_bucket = max_position_per_bucket
        self.orders: list[SpreadOrder] = []
        self.total_spread_pnl: float = 0.0

    def find_spread_opportunities(
        self,
        signals: list[Signal],
        market_prices: dict[str, dict],  # market_id -> {yes_price, no_price, bid, ask}
    ) -> list[SpreadOpportunity]:
        """Find buckets where buying both sides costs less than $1.00."""
        opportunities: list[SpreadOpportunity] = []

        for signal in signals:
            prices = market_prices.get(signal.market_id)
            if not prices:
                continue

            yes_price = prices.get("yes_price", 0)
            no_price = prices.get("no_price", 0)
            bid = prices.get("bid")
            ask = prices.get("ask")

            if yes_price <= 0 or no_price <= 0:
                continue

            # Can we buy both sides for less than $1.00?
            # Improve price by 1¢ on each side for better queue position
            yes_limit = max(0.01, yes_price - 0.01)
            no_limit = max(0.01, no_price - 0.01)
            total_cost = yes_limit + no_limit

            if total_cost >= 1.0:
                continue  # No spread profit available

            profit_per_share = 1.0 - total_cost

            if profit_per_share < self.min_profit_per_trade:
                continue  # Not worth it

            max_shares = self.max_position_per_bucket / total_cost

            opp = SpreadOpportunity(
                market_id=signal.market_id,
                city_id=signal.city_id,
                yes_price=yes_price,
                no_price=no_price,
                spread=abs(ask - bid) if bid and ask else abs(1.0 - yes_price - no_price),
                total_cost=round(total_cost, 4),
                guaranteed_profit_per_share=round(profit_per_share, 4),
                max_shares=round(max_shares, 1),
                description=signal.description,
            )
            opportunities.append(opp)

            logger.debug(
                "SPREAD: %s, YES@%.2f + NO@%.2f = %.2f (profit %.2f/share, max %d shares)",
                signal.city_id, yes_limit, no_limit, total_cost,
                profit_per_share, int(max_shares),
            )

        return sorted(opportunities, key=lambda o: o.guaranteed_profit_per_share, reverse=True)

    def generate_hedge_orders(
        self,
        signal: Signal,
        market_prices: dict[str, dict],
        bankroll: float,
    ) -> SpreadOrder | None:
        """Generate a spread capture order on the opposite side of a directional trade.

        If our directional trade buys YES, this generates a NO buy order.
        If our directional trade buys NO, this generates a YES buy order.
        """
        prices = market_prices.get(signal.market_id)
        if not prices:
            return None

        budget = bankroll * self.spread_budget_pct

        if signal.recommended_side.value == "YES":
            # Our directional trade is YES, hedge by buying NO
            hedge_side = "NO"
            hedge_price = prices.get("no_price", 0)
            if hedge_price <= 0 or hedge_price >= 0.99:
                return None
            # Place limit slightly below current price for better fill
            limit_price = max(0.01, hedge_price - 0.01)
        else:
            # Our directional trade is NO, hedge by buying YES
            hedge_side = "YES"
            hedge_price = prices.get("yes_price", 0)
            if hedge_price <= 0 or hedge_price >= 0.99:
                return None
            limit_price = max(0.01, hedge_price - 0.01)

        # Size: match the directional trade size but cap at budget
        shares = min(
            signal.recommended_size / limit_price,
            budget / limit_price,
            self.max_position_per_bucket / limit_price,
        )

        if shares < 1:
            return None

        cost = shares * limit_price

        # Calculate guaranteed profit if both sides fill
        directional_cost = signal.recommended_size
        total_both_sides = directional_cost + cost
        # Guaranteed payout is $shares (one side always wins)
        guaranteed_payout = shares  # Each winning share pays $1
        guaranteed_profit = guaranteed_payout - total_both_sides

        order = SpreadOrder(
            market_id=signal.market_id,
            city_id=signal.city_id,
            token_id="",  # Filled in at execution time
            side=hedge_side,
            limit_price=round(limit_price, 4),
            shares=round(shares, 1),
            cost=round(cost, 2),
            guaranteed_profit=round(max(0, guaranteed_profit), 2),
            paired_with=signal.market_id,
            description=f"HEDGE {hedge_side} @ {limit_price:.2f} for {signal.city_id} {signal.description[:40]}",
        )

        self.orders.append(order)

        logger.info(
            "SPREAD ORDER: %s %s %.0f shares @ %.2f ($%.2f), hedges %s %s | guaranteed profit: $%.2f",
            hedge_side, signal.city_id, shares, limit_price, cost,
            signal.recommended_side.value, signal.city_id, max(0, guaranteed_profit),
        )

        return order

    def simulate_spread_pnl(self) -> dict:
        """Simulate expected P&L from spread capture orders.

        In paper trading we can't actually fill both sides, but we can
        estimate what the spread income would have been.
        """
        total_cost = sum(o.cost for o in self.orders)
        total_guaranteed = sum(o.guaranteed_profit for o in self.orders)

        return {
            "spread_orders": len(self.orders),
            "total_cost": round(total_cost, 2),
            "estimated_guaranteed_pnl": round(total_guaranteed, 2),
            "avg_profit_per_order": round(total_guaranteed / len(self.orders), 2) if self.orders else 0,
            "cities": list(set(o.city_id for o in self.orders)),
        }

    def summary(self) -> dict:
        return {
            "total_orders": len(self.orders),
            "total_spread_pnl": round(self.total_spread_pnl, 2),
            "budget_pct": self.spread_budget_pct,
            "simulated": self.simulate_spread_pnl(),
        }
