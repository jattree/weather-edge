"""Gemini Quant layer, the market/execution risk analyst.

Gemini is WEATHER-BLIND. Claude handles meteorology.
Gemini handles: fee efficiency, liquidity, model spread, concentration,
black swan scenarios. Post-fee-cliff aware.

Only runs on signals Claude marked as TRADE. Returns:
- Dissent strength (0-1) based on market/execution risk
- Variable sizing (full / reduce_20pct / half / skip)
- Logged to AI Decisions tab alongside Claude's reasoning

Uses Gemini Flash for speed and low cost (~$0.10/day).
"""
from __future__ import annotations

import json
import logging
import os

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
        logger.warning(
            "CONTRACT [%s]: %s, AI reasoning will be degraded",
            result.code, result.error,
        )

_check_ai_keys_at_load()

COASTAL_CITIES = {
    "sf", "la", "seattle", "nyc", "miami",
    "boston", "san_diego", "portland_or",
}

RED_TEAM_PROMPT = (
    "You are a Quantitative Risk Analyst. You are WEATHER-BLIND, "
    "the meteorologist has already validated the forecast. Your job "
    "is to find reasons this trade fails from a MARKET and "
    "EXECUTION perspective.\n\n"
    "CONTEXT: Polymarket weather markets with taker fees live "
    "(1.25% peak at 50\u00a2, near-zero at tails). Post-fee-cliff, "
    "edges in the 20-80\u00a2 range are mostly dead. Profitable "
    "zones are penny tails (<10\u00a2) and whale NOs (>93\u00a2)."
    "\n\n"
    "ANALYZE THESE RISK VECTORS:\n\n"
    "1. Fee Efficiency: Taker fee = 0.05 \u00d7 P \u00d7 (1-P) per "
    "dollar. At 50\u00a2 = 1.25% fee. At 10\u00a2 = 0.45%. "
    "At 5\u00a2 = 0.24%. At 95\u00a2 = 0.24%. At 99\u00a2 = 0.05%."
    " If total edge < 2x the fee, the trade is a coin flip after "
    "costs. Penny bets (<10\u00a2) and extreme NOs (>93\u00a2) are "
    "essentially fee-free.\n\n"
    "2. Liquidity Reality: Can we actually fill this size at this "
    "price? Thin markets show phantom prices. If the order book "
    "probably has <$50 at this price, flag it.\n\n"
    "3. Model Spread Risk: Look at the range between the highest "
    "and lowest model forecasts. If the models disagree by >3\u00b0C,"
    " the consensus is masking bimodal uncertainty, the \"average\""
    " matches neither outcome.\n\n"
    "4. Concentration: Are we already exposed to this city or "
    "weather system? Multiple bets on the same front/system = "
    "correlated risk.\n\n"
    "5. Black Swan: For NO bets at >93\u00a2, what specific "
    "scenario (sensor error, freak convection, measurement station "
    "issue) could flip this to YES?\n\n"
    "DISSENT STRENGTH CALIBRATION:\n"
    "- 0.0: Trade is in a fee-efficient zone, models agree, "
    "no concentration issue\n"
    "- 0.3: Minor liquidity concern or slight model spread, "
    "but edge survives fees\n"
    "- 0.5: Fee eats significant edge, OR models show >3\u00b0C "
    "spread, OR correlated with existing position\n"
    "- 0.8: Multiple risk vectors compound, fee + liquidity + "
    "model disagreement\n"
    "- 1.0: Trade is mathematically unprofitable after fees, "
    "or extreme concentration risk\n\n"
    "Respond in JSON only:\n"
    "{\n"
    "    \"dissent_strength\": 0.0-1.0,\n"
    "    \"primary_risk\": \"specific market/execution risk\",\n"
    "    \"falsifiable_claim\": \"This trade fails if X because "
    "Y\",\n"
    "    \"counter_arguments\": [\"specific risk citing data "
    "provided\"],\n"
    "    \"risk_the_bull_missed\": \"one key risk\",\n"
    "    \"sizing_recommendation\": \"full\" or \"reduce_20pct\" "
    "or \"half\" or \"skip\",\n"
    "    \"verdict\": \"AGREE\" or \"DISSENT\"\n"
    "}"
)


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
    coastal_label = (
        "YES, marine layer / inversion failure modes apply"
        if is_coastal
        else "NO, inland; soil-moisture and CIN failure modes apply"
    )

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

Apply the failure vectors from your system prompt to this specific city. \
Find the thermodynamic kill-switch for this trade."""

    url = (
        f"{GEMINI_API_URL}/{GEMINI_MODEL}"
        f":generateContent?key={GEMINI_API_KEY}"
    )

    try:
        from weather_edge.retry import retry_async

        async def _call_gemini():
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url,
                    json={
                        "contents": [{
                            "role": "user",
                            "parts": [{
                                "text": RED_TEAM_PROMPT
                                + "\n\n" + user_prompt,
                            }],
                        }],
                        "generationConfig": {
                            "temperature": 0.3,
                            "maxOutputTokens": 1000,
                            "thinkingConfig": {"thinkingBudget": 0},
                        },
                    },
                    timeout=20.0,
                )
                resp.raise_for_status()
                return resp.json()

        data = await retry_async(
            _call_gemini,
            attempts=3,
            base_delay=2.0,
            label=f"gemini:{signal.city_id}",
        )

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

        try:
            from weather_edge.analysis.service_health import record_service_call
            existing = {}
            try:
                from weather_edge.live_state import get_json
                existing = get_json("svc:gemini") or {}
            except Exception:
                logger.debug("Failed to read Gemini service state", exc_info=True)
            dissents_today = existing.get("dissents_today", 0) + (
                1 if verdict == "DISSENT" else 0
            )
            record_service_call(
                "gemini", True,
                extra={"dissents_today": dissents_today},
            )
        except Exception:
            logger.debug("Failed to record Gemini health", exc_info=True)

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
        logger.warning("Gemini red team failed after retries: %s", e)
        try:
            from weather_edge.analysis.service_health import record_service_call
            record_service_call("gemini", False)
        except Exception:
            logger.debug("Failed to record Gemini failure", exc_info=True)
        return None
