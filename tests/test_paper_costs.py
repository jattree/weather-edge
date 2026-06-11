"""Tests for microstructure-honest paper fills.

The original paper trader filled at the Gamma midpoint with zero fees, which
is exactly the fiction the post-mortem warns about (lesson 2: "backtests
without fees, slippage, spread cost ... are fiction"). Paper entries now
cross the spread and pay the dynamic taker fee; early exits also pay a taker
fee on sale proceeds. Resolution stays fee-free (redemption is gasless).
"""
from __future__ import annotations

from datetime import datetime, timezone

from weather_edge.analysis.edge import Signal
from weather_edge.models.enums import SignalTier, TradeSide, TradeStatus
from weather_edge.trading.fees import calculate_taker_fee
from weather_edge.trading.paper import PaperTrade, PaperTrader


def _signal(
    side: TradeSide = TradeSide.YES,
    mid: float = 0.50,
    spread: float = 0.04,
    size: float = 100.0,
) -> Signal:
    return Signal(
        market_id="m1",
        consensus_id=None,
        computed_at=datetime.now(timezone.utc),
        model_prob=0.60,
        model_confidence=0.9,
        market_prob=mid,
        edge=0.10,
        net_edge=0.09,
        edge_pct=0.2,
        kelly_fraction=0.1,
        half_kelly=0.05,
        recommended_side=side,
        recommended_size=size,
        confidence_tier=SignalTier.HIGH,
        city_id="nyc",
        description="test market",
        target_date="2026-04-01",
        spread=spread,
        hours_to_resolution=48.0,
        strategy="core",
    )


class TestPaperEntryCosts:

    def test_yes_entry_crosses_spread(self):
        """A YES buyer lifts the ask: entry = mid + half spread."""
        trader = PaperTrader()
        trade = trader.place_trade(_signal(side=TradeSide.YES, mid=0.50, spread=0.04))
        assert trade is not None
        assert trade.entry_price == 0.52

    def test_no_entry_crosses_spread_downward(self):
        """A NO buyer lifting the NO ask maps to a LOWER equivalent YES price."""
        trader = PaperTrader()
        trade = trader.place_trade(_signal(side=TradeSide.NO, mid=0.50, spread=0.04))
        assert trade is not None
        assert trade.entry_price == 0.48

    def test_zero_spread_fills_at_mid(self):
        trader = PaperTrader()
        trade = trader.place_trade(_signal(mid=0.50, spread=0.0))
        assert trade is not None
        assert trade.entry_price == 0.50

    def test_entry_fee_is_dynamic_taker_fee(self):
        """Fee must match the Polymarket dynamic formula at the FILL price."""
        trader = PaperTrader()
        trade = trader.place_trade(_signal(mid=0.50, spread=0.04, size=100.0))
        assert trade is not None
        expected = calculate_taker_fee(0.52, trade.size_usd)
        assert abs(trade.fee_usd - expected) < 1e-6
        assert trade.fee_usd > 0


class TestPaperResolutionCosts:

    def test_win_pnl_deducts_entry_fee(self):
        trader = PaperTrader()
        trade = trader.place_trade(_signal(side=TradeSide.YES, mid=0.50, spread=0.04))
        assert trade is not None
        trader.resolve_trade(trade, outcome_yes=True)
        shares = trade.size_usd / trade.entry_price
        expected = shares - trade.size_usd - trade.fee_usd
        assert trade.status == TradeStatus.WON
        assert abs(trade.pnl - expected) < 1e-6

    def test_loss_pnl_includes_entry_fee(self):
        """Losing costs the stake PLUS the fee paid to enter it."""
        trader = PaperTrader()
        trade = trader.place_trade(_signal(side=TradeSide.YES, mid=0.50, spread=0.04))
        assert trade is not None
        trader.resolve_trade(trade, outcome_yes=False)
        assert trade.status == TradeStatus.LOST
        assert abs(trade.pnl - (-(trade.size_usd + trade.fee_usd))) < 1e-6

    def test_fees_shrink_paper_pnl_vs_frictionless(self):
        """The whole point: paper P&L must be strictly worse than the old
        midpoint/zero-fee model on the same winning trade."""
        trader = PaperTrader()
        trade = trader.place_trade(_signal(side=TradeSide.YES, mid=0.50, spread=0.04))
        assert trade is not None
        trader.resolve_trade(trade, outcome_yes=True)
        frictionless = trade.size_usd / 0.50 - trade.size_usd  # old model
        assert trade.pnl < frictionless


class TestPaperEarlyExitCosts:

    def test_early_exit_charges_entry_and_exit_fees(self):
        """An early exit is a real taker order: entry fee (sunk) + exit fee."""
        trader = PaperTrader()
        no_fee = PaperTrade(
            market_id="m1", city_id="nyc", side="YES", size_usd=100.0,
            entry_price=0.40, description="[TODAY] t", status=TradeStatus.OPEN,
            fee_usd=0.0,
        )
        with_fee = PaperTrade(
            market_id="m2", city_id="nyc", side="YES", size_usd=100.0,
            entry_price=0.40, description="[TODAY] t", status=TradeStatus.OPEN,
            fee_usd=1.20,
        )
        trader.trades.extend([no_fee, with_fee])
        trader.close_position(no_fee, current_price=0.80)
        trader.close_position(with_fee, current_price=0.80)
        # Same exit, but the fee-bearing trade must book exactly $1.20 less
        assert abs((no_fee.pnl - with_fee.pnl) - 1.20) < 1e-6
        # And even the no-entry-fee trade pays an exit taker fee, so its pnl
        # is below the fee-free arithmetic
        shares = 100.0 / 0.40
        exit_price = (0.80 - PaperTrader.PAPER_EXIT_HAIRCUT) * (
            1.0 - PaperTrader.PAPER_EXIT_SLIPPAGE
        )
        frictionless = (exit_price - 0.40) * shares
        assert no_fee.pnl < frictionless

    def test_summary_reports_total_fees(self):
        trader = PaperTrader()
        trade = trader.place_trade(_signal(mid=0.50, spread=0.04))
        assert trade is not None
        assert trader.summary()["total_fees"] == round(trade.fee_usd, 2)
