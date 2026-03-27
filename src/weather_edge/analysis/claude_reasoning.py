"""Claude API reasoning layer for trade decisions.

Uses Claude to analyze weather model data, market conditions, and edge signals
before placing trades. Claude provides:
1. Natural language interpretation of model disagreement
2. Assessment of whether the edge is real or a data artifact
3. Identification of weather patterns that models might miss
4. Trade confidence adjustment based on qualitative factors
5. Human-readable trade rationale for the dashboard

This is what makes "Claude Trader" actually use Claude, not just branding.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import httpx

from weather_edge.analysis.edge import Signal

logger = logging.getLogger(__name__)

def _get_api_key() -> str:
    """Get API key from env or .env file via pydantic-settings."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        try:
            from weather_edge.config import settings
            key = getattr(settings, "anthropic_api_key", "")
        except Exception:
            pass
    return key

ANTHROPIC_API_KEY = _get_api_key()
CLAUDE_MODEL = "claude-sonnet-4-20250514"


@dataclass
class TradeReasoning:
    """Claude's analysis of a potential trade."""
    signal: Signal
    should_trade: bool
    confidence_adjustment: float  # Multiplier: 0.5-1.5
    rationale: str  # Human-readable explanation
    risk_factors: list[str]
    weather_insight: str  # What Claude sees in the pattern


SYSTEM_PROMPT = """You are a weather trading analyst. You assess whether a weather prediction market trade has genuine edge.

You receive:
- Model forecasts from 6-8 weather models for a specific city/date
- The computed consensus (mean, std, confidence)
- A Polymarket market with current price
- The calculated edge (model probability vs market price)

Your job:
1. Assess if the edge is REAL or an artifact of bad data/timing
2. Identify weather patterns that models might agree on for the wrong reason
3. Flag risks (frontal boundaries, lake effects, inversions, etc.)
4. Recommend a confidence adjustment (0.5 = halve position, 1.0 = keep as-is, 1.5 = increase)
5. Give a one-sentence trade rationale

Respond in JSON format:
{
  "should_trade": true/false,
  "confidence_adjustment": 0.5-1.5,
  "rationale": "one sentence",
  "risk_factors": ["risk1", "risk2"],
  "weather_insight": "what you see in the pattern"
}

Be concise. Focus on whether the MODEL CONSENSUS is trustworthy for this specific forecast."""


async def analyze_trade(
    signal: Signal,
    model_values: dict[str, float],
    consensus_mean: float,
    consensus_std: float,
    variable: str = "temp_max_c",
) -> TradeReasoning | None:
    """Ask Claude to analyze a potential trade before execution.

    Only called for HIGH/MEDIUM tier signals to avoid API cost on junk signals.
    """
    if not ANTHROPIC_API_KEY:
        logger.debug("No ANTHROPIC_API_KEY set, skipping Claude reasoning")
        return None

    # Build the prompt with trade context
    model_summary = "\n".join(
        f"  {name}: {val:.1f}°C" for name, val in sorted(model_values.items())
    )

    user_prompt = f"""Analyze this weather trade:

CITY: {signal.city_id.upper()}
VARIABLE: {variable}
DATE: tomorrow

MODEL FORECASTS:
{model_summary}

CONSENSUS: mean={consensus_mean:.1f}°C, std={consensus_std:.1f}°C, confidence={signal.model_confidence:.0%}
MODEL PROBABILITY: {signal.model_prob:.1%}
MARKET PRICE: {signal.market_prob:.1%}
EDGE: {signal.edge:+.1%} ({signal.edge_pct:+.0%} relative)
RECOMMENDED: {signal.recommended_side.value} x${signal.recommended_size:.0f}
MARKET: {signal.description[:100]}

Should I take this trade? What weather risks should I watch for?"""

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": CLAUDE_MODEL,
                    "max_tokens": 300,
                    "system": SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": user_prompt}],
                },
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()

        # Parse Claude's response
        content = data["content"][0]["text"]

        # Try to extract JSON from the response
        try:
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            result = json.loads(content)
        except (json.JSONDecodeError, IndexError):
            logger.warning("Could not parse Claude response as JSON: %s", content[:200])
            return TradeReasoning(
                signal=signal,
                should_trade=True,
                confidence_adjustment=1.0,
                rationale=content[:200],
                risk_factors=[],
                weather_insight="",
            )

        reasoning = TradeReasoning(
            signal=signal,
            should_trade=result.get("should_trade", True),
            confidence_adjustment=max(0.5, min(1.5, result.get("confidence_adjustment", 1.0))),
            rationale=result.get("rationale", ""),
            risk_factors=result.get("risk_factors", []),
            weather_insight=result.get("weather_insight", ""),
        )

        logger.info(
            "CLAUDE: %s %s, %s (adj=%.1fx), %s",
            "TRADE" if reasoning.should_trade else "SKIP",
            signal.city_id.upper(),
            reasoning.rationale[:60],
            reasoning.confidence_adjustment,
            ", ".join(reasoning.risk_factors[:2]) or "no risks flagged",
        )

        return reasoning

    except httpx.HTTPError as e:
        logger.warning("Claude API call failed: %s", e)
        return None
    except Exception as e:
        logger.warning("Claude reasoning failed: %s", e)
        return None


async def batch_analyze_signals(
    signals: list[Signal],
    model_data: dict[str, dict[str, float]],
    consensus_data: dict[str, tuple[float, float]],
    max_calls: int = 5,
) -> dict[str, TradeReasoning]:
    """Analyze the top N signals with Claude.

    Only analyzes the highest-edge signals to control API costs.
    At ~$0.003 per call (Sonnet), 5 calls per cycle = ~$0.015/cycle = ~$0.72/day.
    """
    if not ANTHROPIC_API_KEY:
        return {}

    # Sort by edge magnitude, take top N
    top_signals = sorted(signals, key=lambda s: abs(s.edge), reverse=True)[:max_calls]
    results: dict[str, TradeReasoning] = {}

    for signal in top_signals:
        models = model_data.get(signal.city_id, {})
        cons = consensus_data.get(signal.city_id, (0.0, 0.0))

        reasoning = await analyze_trade(
            signal, models, cons[0], cons[1],
        )
        if reasoning:
            results[signal.market_id] = reasoning

    return results
