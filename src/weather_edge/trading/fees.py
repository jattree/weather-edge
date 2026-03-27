"""Polymarket taker fee calculations for Weather markets.

Starting March 30 2026, Weather markets on Polymarket charge dynamic taker fees.

Fee formula:  fee = peak_rate * 4 * P * (1-P) * trade_value
  - peak_rate: 1.25% (at 50% probability)
  - Scales quadratically: high near 50%, negligible at extremes
  - Makers (limit orders that don't cross spread) pay $0
  - 25% maker rebate distributed daily in USDC

Reference: https://polymarket.com/fees
"""
from __future__ import annotations

# Default peak taker fee rate (1.25% at 50/50 price)
DEFAULT_PEAK_RATE: float = 0.0125

# Default maker rebate percentage (25% of taker fees)
DEFAULT_REBATE_PCT: float = 0.25


def calculate_taker_fee(
    price: float,
    size_usd: float,
    peak_rate: float = DEFAULT_PEAK_RATE,
) -> float:
    """Dynamic taker fee. Higher near 50%, lower at extremes.

    Formula: fee = peak_rate * 4 * P * (1-P) * size_usd

    Examples:
        - At 50%: 0.0125 * 4 * 0.5 * 0.5 * 100 = $1.25 (1.25% of $100)
        - At 95%: 0.0125 * 4 * 0.95 * 0.05 * 100 = $0.2375 (~0.24%)
        - At 5%:  0.0125 * 4 * 0.05 * 0.95 * 100 = $0.2375 (~0.24%)
        - At 99%: 0.0125 * 4 * 0.99 * 0.01 * 100 = $0.0495 (~0.05%)

    Args:
        price: Market probability / price (0-1).
        size_usd: Trade notional in USD.
        peak_rate: Maximum fee rate at 50% probability.

    Returns:
        Taker fee in USD. Always >= 0.
    """
    p = max(0.0, min(1.0, price))
    return peak_rate * 4.0 * p * (1.0 - p) * max(0.0, size_usd)


def calculate_maker_rebate(
    taker_fee: float,
    rebate_pct: float = DEFAULT_REBATE_PCT,
) -> float:
    """25% of taker fees redistributed to makers daily in USDC.

    Args:
        taker_fee: The taker fee amount in USD.
        rebate_pct: Fraction redistributed (default 0.25).

    Returns:
        Maker rebate amount in USD.
    """
    return max(0.0, taker_fee) * rebate_pct


def net_cost_after_fees(
    price: float,
    size_usd: float,
    is_maker: bool = False,
    peak_rate: float = DEFAULT_PEAK_RATE,
) -> float:
    """Total cost including fees.

    Makers (limit orders that don't cross the spread) pay $0 in fees.
    Takers pay the dynamic fee on top of their trade value.

    Args:
        price: Market probability / price (0-1).
        size_usd: Trade notional in USD.
        is_maker: True if this is a maker order (resting limit order).
        peak_rate: Maximum fee rate at 50% probability.

    Returns:
        Total cost = size_usd + taker_fee (or just size_usd for makers).
    """
    if is_maker:
        return size_usd
    fee = calculate_taker_fee(price, size_usd, peak_rate)
    return size_usd + fee


def fee_adjusted_edge(
    edge: float,
    price: float,
    size_usd: float,
    peak_rate: float = DEFAULT_PEAK_RATE,
) -> float:
    """Subtract expected taker fee from raw edge to get net edge.

    Edge is expressed as a probability differential. The fee eats into
    the expected profit, so we convert the fee to an edge-equivalent
    fraction and subtract it.

    Args:
        edge: Raw edge (probability units, e.g. 0.05 for 5%).
        price: Market probability / price (0-1).
        size_usd: Trade notional in USD.
        peak_rate: Maximum fee rate at 50% probability.

    Returns:
        Net edge after subtracting fee impact. Can be negative.
    """
    if size_usd <= 0:
        return edge
    fee = calculate_taker_fee(price, size_usd, peak_rate)
    # Fee as fraction of trade value = fee impact on edge
    fee_as_edge = fee / size_usd
    return edge - fee_as_edge


def fee_eats_alpha(
    edge: float,
    price: float,
    size_usd: float,
    max_fee_pct: float = 0.40,
    peak_rate: float = DEFAULT_PEAK_RATE,
) -> bool:
    """Returns True if taker fee would eat >40% of projected alpha.

    This is the pre-trade gate: if fees consume too much of the edge,
    the trade is not worth taking as a taker. Use maker orders instead.

    Args:
        edge: Raw edge (probability units).
        price: Market probability / price (0-1).
        size_usd: Trade notional in USD.
        max_fee_pct: Maximum acceptable fee as fraction of alpha (default 40%).
        peak_rate: Maximum fee rate at 50% probability.

    Returns:
        True if fee > max_fee_pct * projected_alpha, meaning skip the trade.
    """
    if edge <= 0 or size_usd <= 0:
        return True  # No edge or no size = not worth trading

    fee = calculate_taker_fee(price, size_usd, peak_rate)
    projected_alpha = edge * size_usd  # Expected profit in USD
    return fee > max_fee_pct * projected_alpha
