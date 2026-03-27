"""Forecast trend analysis, track how model predictions change across runs.

Professional meteorologists don't just look at the latest model run. They track:
1. Run-to-run consistency: stable forecast = higher confidence
2. Trend direction: 3 consecutive warming runs = likely to warm further
3. Convergence/divergence: models agreeing more over time = higher confidence

This gives us edge because:
- Market prices reflect the CURRENT forecast, not the trajectory
- A forecast trending warmer that hasn't fully arrived in the latest run
  means the next run will likely push further, we can front-run that
- A volatile forecast (jumping between runs) means uncertainty the market
  hasn't priced, and we should reduce size

Stores the last N consensus values per city in Redis for cross-cycle analysis.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

MAX_HISTORY = 10  # Keep last 10 cycle values per city


@dataclass
class ForecastTrend:
    """Trend analysis for a city's temperature forecast."""
    city_id: str
    current_mean: float
    history: list[float]  # Most recent first
    trend_per_cycle: float  # °C change per cycle (positive = warming)
    stability: float  # 0-1, how consistent the forecast has been
    run_count: int  # How many cycles we've tracked
    signal: str  # "warming", "cooling", "stable", "volatile"
    confidence_multiplier: float  # 0.7-1.2 based on stability

    def to_dict(self) -> dict:
        return {
            "city_id": self.city_id,
            "current_mean": round(self.current_mean, 1),
            "trend_per_cycle": round(self.trend_per_cycle, 2),
            "stability": round(self.stability, 2),
            "run_count": self.run_count,
            "signal": self.signal,
            "confidence_multiplier": round(self.confidence_multiplier, 2),
        }


def _get_history(city_id: str) -> list[float]:
    """Load forecast history from Redis."""
    try:
        from weather_edge.live_state import get_json
        data = get_json(f"trend:{city_id}")
        if data and isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _save_history(city_id: str, history: list[float]) -> None:
    """Save forecast history to Redis (1 hour TTL, refreshed each cycle)."""
    try:
        from weather_edge.live_state import set_json
        set_json(f"trend:{city_id}", history[:MAX_HISTORY], ttl=3600)
    except Exception:
        pass


def record_forecast(city_id: str, consensus_mean: float) -> None:
    """Record a new consensus mean for trend tracking."""
    history = _get_history(city_id)
    history.insert(0, round(consensus_mean, 2))
    history = history[:MAX_HISTORY]
    _save_history(city_id, history)


def compute_trend(city_id: str, current_mean: float) -> ForecastTrend:
    """Compute the forecast trend for a city.

    Returns trend analysis with confidence multiplier:
    - Stable forecast (low variance across runs) → 1.1x confidence
    - Trending consistently → 1.15x (strong directional signal)
    - Volatile (jumping around) → 0.8x (market hasn't priced uncertainty)
    """
    history = _get_history(city_id)

    if len(history) < 2:
        return ForecastTrend(
            city_id=city_id,
            current_mean=current_mean,
            history=history,
            trend_per_cycle=0.0,
            stability=0.5,
            run_count=len(history),
            signal="insufficient_data",
            confidence_multiplier=1.0,
        )

    # Trend: average change per cycle (positive = warming)
    deltas = [history[i] - history[i + 1] for i in range(len(history) - 1)]
    avg_delta = sum(deltas) / len(deltas)

    # Stability: inverse of variance in the deltas
    # Low variance in deltas = forecast is moving consistently (stable or trending)
    # High variance = jumping around (volatile)
    if len(deltas) > 1:
        delta_mean = sum(deltas) / len(deltas)
        delta_variance = sum((d - delta_mean) ** 2 for d in deltas) / len(deltas)
        delta_std = delta_variance ** 0.5
    else:
        delta_std = abs(deltas[0]) if deltas else 0

    # Overall value stability (how much has the forecast changed total)
    value_range = max(history) - min(history) if history else 0
    stability = max(0.0, min(1.0, 1.0 - value_range / 5.0))  # 5°C range = 0 stability

    # Determine signal
    if len(history) < 3:
        signal = "insufficient_data"
        confidence_mult = 1.0
    elif value_range < 0.5:
        # Forecast barely moved across runs, very stable
        signal = "stable"
        confidence_mult = 1.1
    elif all(d > 0.1 for d in deltas):
        # Every run warmed, strong consistent trend
        signal = "warming"
        confidence_mult = 1.15
    elif all(d < -0.1 for d in deltas):
        # Every run cooled, strong consistent trend
        signal = "cooling"
        confidence_mult = 1.15
    elif delta_std > 1.0:
        # Deltas are all over the place, volatile
        signal = "volatile"
        confidence_mult = 0.8
    elif abs(avg_delta) > 0.3:
        # General trend but not perfectly consistent
        signal = "warming" if avg_delta > 0 else "cooling"
        confidence_mult = 1.05
    else:
        signal = "stable"
        confidence_mult = 1.0

    return ForecastTrend(
        city_id=city_id,
        current_mean=current_mean,
        history=history,
        trend_per_cycle=avg_delta,
        stability=stability,
        run_count=len(history),
        signal=signal,
        confidence_multiplier=confidence_mult,
    )
