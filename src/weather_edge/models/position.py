"""Shared Position type, the common interface for both paper and live positions.

Both PaperTrade (paper system) and live exchange positions produce Position
objects for the exit monitor and any other system-agnostic consumer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from weather_edge.models.enums import TradeStatus

# entry_price conventions (Position.price_basis)
PRICE_BASIS_YES = "yes"      # entry_price is the YES-equivalent price (paper)
PRICE_BASIS_TOKEN = "token"  # entry_price is the price paid for the held token (live)


def normalize_side(value) -> str:
    """Canonicalise an outcome/side label to upper case.

    Polymarket labels outcomes "Yes"/"No"; internal code uses the TradeSide
    enum values "YES"/"NO". Accepts enums, None and any casing/whitespace.
    Non YES/NO values (e.g. "SELL") are returned upper-cased.
    """
    if value is None:
        return ""
    raw = getattr(value, "value", value)
    return str(raw).strip().upper()


@dataclass
class Position:
    """A position in a market, paper or live.

    This is the minimal shared interface that scan_for_exits() and other
    system-agnostic code should depend on. Paper trades and live positions
    both satisfy this interface.

    ``side`` is always normalised to "YES"/"NO" on construction.

    ``entry_price`` convention depends on ``price_basis``:
      - "yes" (default, paper): the YES-equivalent price. A NO position's
        cost per share is ``1 - entry_price``.
      - "token" (live): the average price actually paid for the held token,
        so a NO position's cost per share is ``entry_price`` itself.
    Use ``yes_entry_price`` / ``token_entry_price`` instead of reading
    ``entry_price`` directly when the side matters.
    """
    market_id: str = ""
    city_id: str = ""
    side: str = ""              # "YES" or "NO"
    size_usd: float = 0.0      # Total USD invested (cost_basis)
    entry_price: float = 0.0   # Average entry price per share (see price_basis)
    description: str = ""
    status: TradeStatus = TradeStatus.OPEN
    placed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_shares: float = 0.0  # Number of shares held (size_usd / entry_price)
    source: str = ""           # "paper" or "live"
    strategy: str = "core"     # "core", "penny", "spread", "exit"
    price_basis: str = PRICE_BASIS_YES

    def __post_init__(self):
        self.side = normalize_side(self.side)

    @property
    def yes_entry_price(self) -> float:
        """Entry expressed as a YES price, regardless of side/basis."""
        if self.side == "NO" and self.price_basis == PRICE_BASIS_TOKEN:
            return 1.0 - self.entry_price
        return self.entry_price

    @property
    def token_entry_price(self) -> float:
        """Price paid per share of the token actually held."""
        if self.side == "NO" and self.price_basis == PRICE_BASIS_YES:
            return 1.0 - self.entry_price
        return self.entry_price
