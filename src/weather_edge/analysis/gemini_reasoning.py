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

RED_TEAM_PROMPT = """You are a weather trading RED TEAM analyst. Your job is to find reasons NOT to take a trade that another analyst has approved.

You receive a weather trading signal that has already been approved. Your task:
1. Find the strongest counter-arguments against this trade
2. Identify risks the approving analyst may have missed
3. Rate your dissent strength from 0.0 (agree with trade) to 1.0 (strongly disagree)

Be specific. Generic warnings like "weather is uncertain" are useless. Point to concrete model disagreements, historical busts, or market microstructure risks.

Respond in JSON only:
{
    "dissent_strength": 0.0-1.0,
    "counter_arguments": ["specific reason 1", "specific reason 2"],
    "risk_the_bull_missed": "one key risk",
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

    user_prompt = f"""APPROVED TRADE to red-team:
City: {signal.city_id}
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

Find the strongest case AGAINST this trade."""

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

        logger.info(
            "GEMINI RED TEAM: %s %s, %s (dissent=%.1f), %s",
            signal.city_id, signal.recommended_side.value,
            verdict, dissent,
            "; ".join(counters[:2]) if counters else "no counter-arguments",
        )

        return {
            "dissent_strength": dissent,
            "verdict": verdict,
            "counter_arguments": counters,
            "risk_the_bull_missed": risk,
            "model": "gemini",
        }

    except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning("Gemini red team failed: %s", e)
        return None
