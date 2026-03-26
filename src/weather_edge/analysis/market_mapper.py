"""Map Polymarket markets to consensus probabilities."""
from __future__ import annotations

import logging

from weather_edge.fetchers.polymarket import MarketInfo
from weather_edge.models.enums import MarketType

logger = logging.getLogger(__name__)

# Map MarketType to the consensus variable name
MARKET_TYPE_TO_VARIABLE: dict[MarketType, str] = {
    MarketType.TEMP_HIGH: "temp_max_c",
    MarketType.TEMP_LOW: "temp_min_c",
    MarketType.PRECIP: "precip_sum_mm",
    MarketType.SNOW: "snow_sum_cm",
}


def get_required_variable(market: MarketInfo) -> str | None:
    """Get the consensus variable name needed for a market."""
    return MARKET_TYPE_TO_VARIABLE.get(market.market_type)
