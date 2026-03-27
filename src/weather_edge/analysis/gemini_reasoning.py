"""Gemini red team layer, finds reasons NOT to trade after Claude approves.

Only runs on signals Claude marked as TRADE. Acts as adversarial dissent:
- Prompted to argue against the trade
- Returns dissent strength (0-1) and counter-arguments
- If dissent > 0.7, we reduce position size or skip
- Logged to AI Decisions tab alongside Claude's reasoning

Uses Gemini Flash for speed and low cost (~$0.10/day).
"""
from __future__ import annotations

import logging
import os
import json

import httpx

from weather_edge.analysis.edge import Signal

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def _get_gemini_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        try:
            from weather_edge.config import settings
            key = getattr(settings, "gemini_api_key", "")
        except Exception:
            pass
    return key


GEMINI_API_KEY = _get_gemini_key()

# Contract: warn at module load if AI keys are missing
def _check_ai_keys_at_load() -> None:
    from weather_edge.analysis.contracts import validate_ai_keys_present
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not anthropic_key:
        try:
            from weather_edge.config import settings as _s
            anthropic_key = getattr(_s, "anthropic_api_key", "")
        except Exception:
            pass
    result = validate_ai_keys_present(anthropic_key, GEMINI_API_KEY)
    if not result.valid:
        logger.warning("CONTRACT [%s]: %s, AI reasoning will be degraded", result.code, result.error)

_check_ai_keys_at_load()

COASTAL_CITIES = {"sf", "la", "seattle", "nyc", "miami", "boston", "san_diego", "portland_or"}

RED_TEAM_PROMPT = """You are a Lead Forensic Meteorologist. Your mission is to find the thermodynamic kill-switch for this proposed weather trade. Ignore the crowd; find why the math fails.

Analyze these failure vectors:

1. CIN Cap: If the bull case relies on convective cooling (storms), check for Convective Inhibition. If CIN is likely high, the cap won't break, expect +3-5F temperature overshoot from uninterrupted solar heating.

2. Marine Layer Trap: For coastal cities (SF, LA, Seattle, NYC), if a strong inversion is present, global models burn off fog 2-4 hours too early. Argue for lower max temps.

3. Soil-Moisture Feedback: In Dallas, Chicago, Houston, Atlanta, if conditions are dry, GFS has a documented 4-8F warm bias. Discount GFS-led heat spikes.

4. Multimodal Ensemble Dissent: If the model forecasts show two distinct clusters (not one peak), the ensemble is bimodal. Assign high dissent, averaging two different weather regimes produces a number that matches neither.

5. Diurnal Timing Error: Models predict daily max temp but the actual peak depends on cloud timing. If afternoon clouds are likely (convective initiation), max temp occurs earlier and lower than models suggest.

Be specific to this city and date. Generic warnings like "weather is uncertain" get a dissent score of 0.

Respond in JSON only:
{
    "dissent_strength": 0.0-1.0,
    "primary_failure_mode": "specific mechanism",
    "counter_arguments": ["specific reason 1", "specific reason 2"],
    "risk_the_bull_missed": "one key risk",
    "sizing_recommendation": "full" or "half" or "skip",
    "verdict": "AGREE" or "DISSENT"
}"""


async def red_team_trade(
    signal: Signal,
    model_values: dict[str, float],
    consensus_mean: float,
    consensus_std: float,
    claude_rationale: str,
) -> dict | None:
    """Run Gemini red team analysis on a Claude-approved trade.

    Returns dict with dissent_strength, counter_arguments, risk, verdict.
    Returns None if Gemini is unavailable or errors.
    """
    if not GEMINI_API_KEY:
        return None

    model_summary = "\n".join(
        f"  {name}: {val:.1f}°C" for name, val in sorted(model_values.items())
    )

    city_lower = signal.city_id.lower().replace(" ", "_")
    is_coastal = city_lower in COASTAL_CITIES
    coastal_label = "YES, marine layer / inversion failure modes apply" if is_coastal else "NO, inland; soil-moisture and CIN failure modes apply"

    user_prompt = f"""APPROVED TRADE to red-team:
City: {signal.city_id}
Coastal city: {coastal_label}
Side: {signal.recommended_side.value}
Market price: {signal.market_prob:.1%}
Model probability: {signal.model_prob:.1%}
Edge: {signal.edge_pct:.1f}%
Size: ${signal.recommended_size:.0f}

Model forecasts:
{model_summary}

Consensus: {consensus_mean:.1f}°C (std={consensus_std:.1f}°C)

Bull case (approved by Claude):
{claude_rationale}

Apply the failure vectors from your system prompt to this specific city. Find the thermodynamic kill-switch for this trade."""

    url = f"{GEMINI_API_URL}/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                json={
                    "contents": [
                        {"role": "user", "parts": [{"text": RED_TEAM_PROMPT + "\n\n" + user_prompt}]}
                    ],
                    "generationConfig": {
                        "temperature": 0.3,
                        "maxOutputTokens": 1000,
                        "thinkingConfig": {"thinkingBudget": 0},
                    },
                },
                timeout=20.0,
            )
            resp.raise_for_status()
            data = resp.json()

        # Parse Gemini response, extract JSON from last text part
        import re
        text = data["candidates"][0]["content"]["parts"][-1]["text"]
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        match = re.search(r"\{.*\}", text.strip(), re.DOTALL)
        if not match:
            logger.warning("Gemini returned non-JSON: %s", text[:100])
            return None
        result = json.loads(match.group())

        dissent = float(result.get("dissent_strength", 0))
        verdict = result.get("verdict", "AGREE")
        counters = result.get("counter_arguments", [])
        risk = result.get("risk_the_bull_missed", "")
        failure_mode = result.get("primary_failure_mode", "")
        sizing = result.get("sizing_recommendation", "full")

        logger.info(
            "GEMINI RED TEAM: %s %s, %s (dissent=%.1f, sizing=%s, failure=%s), %s",
            signal.city_id, signal.recommended_side.value,
            verdict, dissent, sizing, failure_mode,
            "; ".join(counters[:2]) if counters else "no counter-arguments",
        )

        return {
            "dissent_strength": dissent,
            "verdict": verdict,
            "primary_failure_mode": failure_mode,
            "counter_arguments": counters,
            "risk_the_bull_missed": risk,
            "sizing_recommendation": sizing,
            "model": "gemini",
        }

    except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning("Gemini red team failed: %s", e)
        return None
