"""Shared Position type, the common interface for both paper and live positions.

Both PaperTrade (paper system) and live exchange positions produce Position
objects for the exit monitor and any other system-agnostic consumer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from weather_edge.models.enums import TradeStatus


@dataclass
class Position:
    """A position in a market, paper or live.

    This is the minimal shared interface that scan_for_exits() and other
    system-agnostic code should depend on. Paper trades and live positions
    both satisfy this interface.
    """
    market_id: str = ""
    city_id: str = ""
    side: str = ""              # "YES" or "NO"
    size_usd: float = 0.0      # Total USD invested (cost_basis)
    entry_price: float = 0.0   # Average entry price per share
    description: str = ""
    status: TradeStatus = TradeStatus.OPEN
    placed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_shares: float = 0.0  # Number of shares held (size_usd / entry_price)
    source: str = ""           # "paper" or "live"
    strategy: str = "core"     # "core", "penny", "spread", "exit"
