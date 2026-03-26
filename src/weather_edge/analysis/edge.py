"""Edge detection and Kelly criterion position sizing."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone

from weather_edge.analysis.model_timing import get_confidence_boost
from weather_edge.config import settings
from weather_edge.models.enums import SignalTier, TradeSide

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """A trading signal with edge calculation and sizing."""
    market_id: str
    consensus_id: int | None
    computed_at: datetime

    model_prob: float        # Raw model consensus probability
    model_confidence: float  # 0-1 model agreement
    market_prob: float       # Polymarket implied probability (midpoint)

    edge: float              # adj_prob - market_prob
    edge_pct: float          # edge / market_prob
    kelly_fraction: float    # Optimal Kelly bet fraction
    half_kelly: float        # Conservative (half-Kelly)
    recommended_side: TradeSide
    recommended_size: float  # Dollar amount
    confidence_tier: SignalTier

    # Extra context
    city_id: str = ""
    description: str = ""
    hours_to_resolution: float | None = None
    strategy: str = "core"  # "core" or "tail"


def temporal_decay_factor(hours_to_resolution: float | None) -> float:
    """Scale confidence by how close we are to market resolution.

    Weather forecasts are much more accurate at short range.
    - < 6 hours: full confidence (1.0)
    - 12 hours: 0.95
    - 24 hours: 0.85
    - 48 hours: 0.65
    - 72+ hours: 0.50
    """
    if hours_to_resolution is None:
        return 0.85  # Default if unknown

    if hours_to_resolution <= 12:
        return 1.0
    elif hours_to_resolution <= 24:
        return 0.95
    elif hours_to_resolution <= 48:
        return 0.85
    elif hours_to_resolution <= 72:
        return 0.75
    else:
        return max(0.55, 0.75 * math.exp(-0.015 * (hours_to_resolution - 72)))


def calculate_edge(
    market_id: str,
    model_prob: float,
    market_prob: float,
    model_confidence: float,
    bankroll: float | None = None,
    consensus_id: int | None = None,
    hours_to_resolution: float | None = None,
    city_id: str = "",
    description: str = "",
) -> Signal:
    """Calculate edge and optimal position size.

    Uses confidence-adjusted probability that shrinks toward market price
    when model agreement is low, and temporal decay to reduce confidence
    for longer-horizon forecasts.
    """
    if bankroll is None:
        bankroll = settings.bankroll

    now = datetime.now(timezone.utc)

    # Clamp inputs to valid ranges
    model_prob = max(0.01, min(0.99, model_prob))
    market_prob = max(0.01, min(0.99, market_prob))
    model_confidence = max(0.0, min(1.0, model_confidence))

    # Apply temporal decay and model freshness boost
    t_decay = temporal_decay_factor(hours_to_resolution)
    freshness_boost = get_confidence_boost(now)
    adjusted_confidence = min(1.0, model_confidence * t_decay * freshness_boost)

    # Confidence-adjusted probability: blend model and market based on confidence
    # When models strongly agree (high confidence), trust the model
    # When models disagree (low confidence), shrink toward market price
    adj_prob = adjusted_confidence * model_prob + (1 - adjusted_confidence) * market_prob

    # Determine side: buy YES if model says higher prob than market
    if adj_prob > market_prob:
        side = TradeSide.YES
        edge = adj_prob - market_prob
        # Kelly for YES bet: we're paying market_prob to win (1 - market_prob)
        odds = (1.0 - market_prob) / market_prob
        kelly_f = (adj_prob * odds - (1.0 - adj_prob)) / odds if odds > 0 else 0.0
    else:
        side = TradeSide.NO
        edge = market_prob - adj_prob
        # Kelly for NO bet: we're paying (1 - market_prob) to win market_prob
        no_price = 1.0 - market_prob
        odds = market_prob / no_price if no_price > 0 else 0.0
        no_prob = 1.0 - adj_prob
        kelly_f = (no_prob * odds - adj_prob) / odds if odds > 0 else 0.0

    # Clamp Kelly and apply half-Kelly
    kelly_f = max(0.0, kelly_f)
    half_k = kelly_f * settings.kelly_fraction

    # Position size with max cap
    bet_size = min(half_k * bankroll, bankroll * settings.max_position_pct)
    bet_size = max(0.0, bet_size)

    edge_pct = edge / market_prob if market_prob > 0 else 0.0

    # Detect if this is a tail bet opportunity (ColdMath-style penny sniping)
    # Tail bet: market prices YES at <$0.05 but our model says 3x+ higher probability
    is_tail = (
        side == TradeSide.YES
        and market_prob <= 0.05
        and model_prob >= market_prob * 3
        and adjusted_confidence >= 0.5
    )

    if is_tail:
        # Tail bet sizing: fixed small amount from the tail bankroll (30% of total)
        # Risk is capped at the bet size, but payoff is 20-100x
        tail_bankroll = bankroll * 0.30
        # Flat size per tail bet, spread across many small lotto tickets
        bet_size = min(tail_bankroll * 0.10, bankroll * 0.02)  # Max 2% of total bankroll per tail
        bet_size = max(0.0, bet_size)
        tier = SignalTier.HIGH  # Tail bets are always high priority when they qualify
        strategy = "tail"
    else:
        strategy = "core"
        # Determine confidence tier for core bets
        spread_ok = True  # We'd check spread from price data in practice
        if edge >= 0.05 and adjusted_confidence >= 0.8 and spread_ok:
            tier = SignalTier.HIGH
        elif edge >= 0.03 and adjusted_confidence >= 0.6:
            tier = SignalTier.MEDIUM
        else:
            tier = SignalTier.LOW

    return Signal(
        market_id=market_id,
        consensus_id=consensus_id,
        computed_at=now,
        model_prob=round(model_prob, 4),
        model_confidence=round(adjusted_confidence, 4),
        market_prob=round(market_prob, 4),
        edge=round(edge, 4),
        edge_pct=round(edge_pct, 4),
        kelly_fraction=round(kelly_f, 4),
        half_kelly=round(half_k, 4),
        recommended_side=side,
        recommended_size=round(bet_size, 2),
        confidence_tier=tier,
        city_id=city_id,
        description=description,
        strategy=strategy,
        hours_to_resolution=hours_to_resolution,
    )
