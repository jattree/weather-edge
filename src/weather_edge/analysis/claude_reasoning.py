"""Claude reasoning layer, the Meteorologist.

Claude is MARKET-BLIND. It assesses the atmosphere, not the trade.
Its job is to answer: "Is this forecast trustworthy?"

Claude provides:
1. Physical plausibility assessment of model consensus
2. Outlier diagnosis (real mesoscale feature vs model error)
3. Forecast confidence based on meteorological factors
4. Risk factors: specific physical bust mechanisms
5. Weather insight for the dashboard

Gemini (the Quant) handles market skepticism, sizing, and risk management.
The two AIs must have UNCORRELATED errors to maximize signal quality.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import httpx

from weather_edge.analysis.edge import Signal

logger = logging.getLogger(__name__)

# ---- Decision history for dashboard AI Decisions tab ----
_decision_history: list[dict] = []
MAX_DECISION_HISTORY = 200


def record_decision(reasoning: "TradeReasoning") -> None:
    """Record a Claude trade decision for the dashboard."""
    from datetime import datetime, timezone

    entry = {
        "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "city": reasoning.signal.city_id.upper() if isinstance(reasoning.signal.city_id, str) else reasoning.signal.city_id,
        "decision": "TRADE" if reasoning.should_trade else "SKIP",
        "signal": reasoning.signal.description[:60] if reasoning.signal.description else "",
        "adjustment": reasoning.confidence_adjustment,
        "rationale": reasoning.rationale,
        "risk_factors": reasoning.risk_factors,
    }
    _decision_history.insert(0, entry)
    # Trim to max size
    while len(_decision_history) > MAX_DECISION_HISTORY:
        _decision_history.pop()


def get_decisions() -> list[dict]:
    """Return a copy of the decision history (most recent first)."""
    return list(_decision_history)


def clear_decisions() -> None:
    """Clear the decision history (e.g. on new session)."""
    _decision_history.clear()


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

# Contract: warn at module load if AI keys are missing
def _check_ai_keys_at_load() -> None:
    from weather_edge.analysis.contracts import validate_ai_keys_present
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        try:
            from weather_edge.config import settings as _s
            gemini_key = getattr(_s, "gemini_api_key", "")
        except Exception:
            pass
    result = validate_ai_keys_present(ANTHROPIC_API_KEY, gemini_key)
    if not result.valid:
        logger.warning("CONTRACT [%s]: %s, AI reasoning will be degraded", result.code, result.error)

_check_ai_keys_at_load()


@dataclass
class TradeReasoning:
    """Claude's analysis of a potential trade."""
    signal: Signal
    should_trade: bool
    confidence_adjustment: float  # Multiplier: 0.5-1.5
    rationale: str  # Human-readable explanation
    risk_factors: list[str]
    weather_insight: str  # What Claude sees in the pattern


SYSTEM_PROMPT = """You are a forensic meteorologist. Your ONLY job is to assess what the atmosphere will do. You are MARKET-BLIND, you do not care about prices, edges, or whether a trade "looks too good." That is someone else's job.

You receive model forecasts from 6-8 weather models for a specific city/date. Assess:

1. PHYSICAL PLAUSIBILITY: Do the models agree for the right physical reasons? Or are they clustering on a shared bias (e.g., all using the same SST boundary condition)?
2. MODEL TRUST: Which models should be weighted more for THIS city? (HRRR for US short-range, UKV for UK, ECMWF for 3+ day global)
3. OUTLIER DIAGNOSIS: If one model disagrees, is it seeing a real mesoscale feature (frontal boundary, sea breeze, orographic effect) or is it just wrong?
4. FORECAST CONFIDENCE: How confident are you in the consensus temperature? Consider: time of year, city microclimate, synoptic pattern.
5. RISKS: What specific physical mechanisms could bust this forecast? (fronts, inversions, lake effect, marine layer, convective initiation)

CRITICAL RULES:
- NEVER say "the edge looks too large" or "massive edge suggests market error." You don't know market prices.
- NEVER skip a trade because "the market knows something." You are the weather expert.
- If models tightly agree and you see no physical bust mechanism, should_trade MUST be true.
- Only set should_trade=false if you have a SPECIFIC meteorological reason (not market skepticism).
- confidence_adjustment reflects YOUR forecast confidence, not market confidence.

Respond in JSON:
{
  "should_trade": true/false,
  "confidence_adjustment": 0.5-1.5,
  "rationale": "one sentence about the WEATHER, not the market",
  "risk_factors": ["specific physical mechanism 1", "mechanism 2"],
  "weather_insight": "what the atmosphere is doing"
}"""


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

    # Extract threshold from description for Claude's context
    desc = signal.description[:120] if signal.description else ""

    user_prompt = f"""Assess this weather forecast:

CITY: {signal.city_id.upper()}
VARIABLE: {variable}
DATE: tomorrow
QUESTION: {desc}

MODEL FORECASTS:
{model_summary}

CONSENSUS: mean={consensus_mean:.1f}°C, std={consensus_std:.1f}°C
MODEL SPREAD: {max(model_values.values()) - min(model_values.values()):.1f}°C range across {len(model_values)} models

Is this consensus trustworthy? What physical mechanisms could bust it?"""

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

        try:
            from weather_edge.analysis.service_health import record_service_call
            existing = {}
            try:
                from weather_edge.live_state import get_json
                existing = get_json("svc:claude") or {}
            except Exception:
                pass
            decisions_today = existing.get("decisions_today", 0) + 1
            record_service_call("claude", True, extra={"decisions_today": decisions_today})
        except Exception:
            pass

        return reasoning

    except httpx.HTTPError as e:
        logger.warning("Claude API call failed: %s", e)
        try:
            from weather_edge.analysis.service_health import record_service_call
            record_service_call("claude", False)
        except Exception:
            pass
        return None
    except Exception as e:
        logger.warning("Claude reasoning failed: %s", e)
        try:
            from weather_edge.analysis.service_health import record_service_call
            record_service_call("claude", False)
        except Exception:
            pass
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
