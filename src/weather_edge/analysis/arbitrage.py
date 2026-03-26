"""Bucket parity arbitrage detection for Polymarket multi-bucket temperature markets.

In a multi-bucket event (e.g. "Highest temperature in NYC on March 27"),
the sum of all bucket YES prices should theoretically equal 1.0 (minus vig).

When the sum significantly deviates from 1.0, there's an arbitrage opportunity:
- Sum > 1.05: Market is overpriced, look for NO opportunities on inflated buckets
- Sum < 0.95: Market is underpriced, look for YES opportunities on deflated buckets

This check is FREE alpha that most retail traders miss.

Source: Gemini CLI analysis of Polymarket market microstructure.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from weather_edge.fetchers.polymarket import MarketInfo

logger = logging.getLogger(__name__)


@dataclass
class ParityCheck:
    """Result of a bucket parity check for one event."""
    event_title: str
    city_id: str
    target_date: str
    bucket_count: int
    yes_sum: float        # Sum of all YES prices (should be ~1.0)
    overpriced: bool      # True if sum > 1.05
    underpriced: bool     # True if sum < 0.95
    deviation: float      # yes_sum - 1.0
    most_inflated: MarketInfo | None = None   # Bucket with highest YES price vs model
    most_deflated: MarketInfo | None = None   # Bucket with lowest YES price vs model


def check_bucket_parity(
    markets: list[MarketInfo],
) -> list[ParityCheck]:
    """Check parity across all multi-bucket events.

    Groups markets by event (city+date) and checks if YES prices sum to ~1.0.
    """
    # Group by event (city+date)
    events: dict[tuple[str, str], list[MarketInfo]] = {}
    for m in markets:
        if m.city_id is None:
            continue
        key = (m.city_id.value, str(m.target_date))
        events.setdefault(key, []).append(m)

    checks: list[ParityCheck] = []

    for (city, date_str), buckets in events.items():
        if len(buckets) < 3:
            continue  # Need multiple buckets for parity check

        yes_sum = sum(b.yes_price for b in buckets)
        deviation = yes_sum - 1.0

        # Sort by YES price to find most inflated/deflated
        by_price = sorted(buckets, key=lambda b: b.yes_price, reverse=True)
        most_inflated = by_price[0] if by_price else None
        most_deflated = by_price[-1] if by_price else None

        check = ParityCheck(
            event_title=buckets[0].event_title,
            city_id=city,
            target_date=date_str,
            bucket_count=len(buckets),
            yes_sum=round(yes_sum, 4),
            overpriced=yes_sum > 1.05,
            underpriced=yes_sum < 0.95,
            deviation=round(deviation, 4),
            most_inflated=most_inflated,
            most_deflated=most_deflated,
        )
        checks.append(check)

        if check.overpriced:
            logger.info(
                "PARITY ARBI: %s %s, YES sum=%.3f (+%.1f%%), %d buckets. "
                "Most inflated: %s @ %.3f",
                city.upper(), date_str, yes_sum, deviation * 100, len(buckets),
                (most_inflated.question[:50] if most_inflated else ""),
                (most_inflated.yes_price if most_inflated else 0),
            )
        elif check.underpriced:
            logger.info(
                "PARITY ARBI: %s %s, YES sum=%.3f (%.1f%%), %d buckets. "
                "Most deflated: %s @ %.3f",
                city.upper(), date_str, yes_sum, deviation * 100, len(buckets),
                (most_deflated.question[:50] if most_deflated else ""),
                (most_deflated.yes_price if most_deflated else 0),
            )

    return checks


def find_parity_opportunities(
    checks: list[ParityCheck],
    min_deviation: float = 0.05,
) -> list[ParityCheck]:
    """Filter for actionable parity arbitrage opportunities."""
    return [c for c in checks if abs(c.deviation) >= min_deviation]
