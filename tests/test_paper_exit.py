"""Tests for the paper exit monitor fix (#10).

Two bugs made the early-exit path dead and dishonest:
  * close_position defaulted volume_24h=0, and callers never pass it, so the
    "$5K volume" gate blocked EVERY exit.
  * It returned early on pnl <= 0, so early exits could only ever book WINS,
    inflating paper P&L versus live (which has no such luxury).
"""
from __future__ import annotations

from weather_edge.models.enums import TradeStatus
from weather_edge.trading.paper import PaperTrade, PaperTrader


def _open_trade(side="YES", entry=0.60, size=100.0) -> PaperTrade:
    t = PaperTrade(
        market_id="m1", city_id="nyc", side=side, size_usd=size,
        entry_price=entry, description="[TODAY] test", status=TradeStatus.OPEN,
    )
    return t


class TestClosePositionExit:

    def test_unknown_volume_does_not_block_exit(self):
        trader = PaperTrader()
        t = _open_trade(side="YES", entry=0.60)
        trader.trades.append(t)
        # YES bought at 0.60, market fell to 0.40 -> this is a LOSS we must take.
        trader.close_position(t, current_price=0.40)
        assert t.status != TradeStatus.OPEN  # exit was NOT blocked by volume gate

    def test_exit_realizes_a_loss(self):
        trader = PaperTrader()
        t = _open_trade(side="YES", entry=0.60)
        trader.trades.append(t)
        trader.close_position(t, current_price=0.40)
        assert t.status == TradeStatus.LOST
        assert t.pnl is not None and t.pnl < 0

    def test_exit_books_a_win_when_profitable(self):
        trader = PaperTrader()
        t = _open_trade(side="YES", entry=0.40)
        trader.trades.append(t)
        trader.close_position(t, current_price=0.80)  # rose well above entry
        assert t.status == TradeStatus.WON
        assert t.pnl is not None and t.pnl > 0

    def test_real_low_volume_still_blocks(self):
        trader = PaperTrader()
        t = _open_trade(side="YES", entry=0.60)
        trader.trades.append(t)
        trader.close_position(t, current_price=0.40, volume_24h=100.0)
        assert t.status == TradeStatus.OPEN  # genuinely illiquid -> held
