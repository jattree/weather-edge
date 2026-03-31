"""Early exit monitor, checks if open positions should be closed before resolution.

Runs each cycle after new model data arrives. Checks:
1. Edge inversion (model flipped against us by >7%)
2. Profit cap (price hit 88%+ on core bets, model <94%)
3. Pattern bust (detected pattern invalidated)
4. Penny bets: NEVER exit (hold to 0 or 1)

Candidates go through Claude (confirm exit) → Gemini (argue for hold).
Both agree → flag for exit. Gemini dissents → reduce urgency, hold.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from weather_edge.analysis.contracts import validate_penny_no_exit
from weather_edge.models.position import Position

logger = logging.getLogger(__name__)

# Thresholds (per Gemini validation)
EDGE_INVERSION_THRESHOLD = -0.07  # Exit when edge flips worse than -7%
PROFIT_CAP_PRICE = 0.88  # Take profit above this price
PROFIT_CAP_MODEL_MAX = 0.94  # Only take profit if model < 94%
PENNY_ENTRY_MAX = 0.06  # Never exit penny bets (entry <= 6¢)
MIN_SPREAD_MULTIPLIER = 1.5  # Don't exit if edge < 1.5x the spread


@dataclass
class ExitCandidate:
    """A trade flagged for potential early exit."""
    trade: Position
    reason: str  # "edge_inversion", "profit_cap", "pattern_bust"
    current_model_prob: float
    current_market_price: float
    original_edge: float
    current_edge: float
    urgency: str  # "high", "medium", "low"
    claude_verdict: str | None = None  # "EXIT" or "HOLD"
    claude_rationale: str | None = None
    gemini_verdict: str | None = None  # "AGREE_EXIT" or "HOLD"
    gemini_rationale: str | None = None
    final_decision: str | None = None  # "EXIT" or "HOLD"


def scan_for_exits(
    open_trades: list[Position],
    market_prices: dict[str, float],
    model_probs: dict[str, float],
) -> list[ExitCandidate]:
    """Scan open positions for early exit candidates.

    Args:
        open_trades: Currently open paper trades
        market_prices: Current market YES prices by market_id
        model_probs: Current model probabilities by market_id

    Returns:
        List of ExitCandidate objects needing AI review
    """
    candidates: list[ExitCandidate] = []

    for trade in open_trades:
        # Contract: never exit penny bets (hold to resolution)
        penny_check = validate_penny_no_exit(trade.entry_price, PENNY_ENTRY_MAX)
        if not penny_check.valid:
            continue

        # Skip spread trades
        if "[SPREAD]" in (trade.description or ""):
            continue

        market_price = market_prices.get(trade.market_id)
        model_prob = model_probs.get(trade.market_id)

        if market_price is None or model_prob is None:
            continue

        # Calculate current edge from our position's perspective
        # Keep raw_market_price for ExitCandidate (close_position needs YES price)
        raw_market_price = market_price
        if trade.side == "YES":
            current_edge = model_prob - market_price
            original_edge = model_prob - trade.entry_price
        else:
            current_edge = (1.0 - model_prob) - (1.0 - market_price)
            original_edge = (1.0 - model_prob) - (1.0 - trade.entry_price)

        # Check exit triggers
        reason = None
        urgency = "medium"

        # 1. Edge inversion: model now says we're wrong
        if current_edge < EDGE_INVERSION_THRESHOLD:
            reason = "edge_inversion"
            urgency = "high" if current_edge < -0.15 else "medium"

        # 2. Profit cap: price ran up, lock in gains on core bets
        elif (trade.side == "YES"
              and market_price >= PROFIT_CAP_PRICE
              and model_prob < PROFIT_CAP_MODEL_MAX):
            reason = "profit_cap"
            urgency = "low"
        elif (trade.side == "NO"
              and (1.0 - market_price) >= PROFIT_CAP_PRICE
              and (1.0 - model_prob) < PROFIT_CAP_MODEL_MAX):
            reason = "profit_cap"
            urgency = "low"

        if reason:
            candidates.append(ExitCandidate(
                trade=trade,
                reason=reason,
                current_model_prob=model_prob,
                # Always YES price, close_position handles NO flip
                current_market_price=raw_market_price,
                original_edge=round(original_edge, 4),
                current_edge=round(current_edge, 4),
                urgency=urgency,
            ))
            logger.info(
                "EXIT CANDIDATE: %s %s %s, %s (edge: %.1f%% → %.1f%%, urgency: %s)",
                trade.side, trade.city_id, trade.market_id[:20],
                reason, original_edge * 100, current_edge * 100, urgency,
            )

    return sorted(candidates, key=lambda c: c.current_edge)  # Worst edge first


async def ai_review_exit(
    candidate: ExitCandidate,
    model_values: dict[str, float],
    consensus_mean: float,
    consensus_std: float,
) -> ExitCandidate:
    """Run Claude + Gemini review on an exit candidate.

    Claude: confirms exit rationale
    Gemini: argues for holding (red team)
    Both agree exit → final_decision = EXIT
    Gemini dissents → final_decision = HOLD (reduce urgency)
    """
    trade = candidate.trade

    # Claude review: should we exit?
    try:
        from weather_edge.analysis.claude_reasoning import ANTHROPIC_API_KEY
        if ANTHROPIC_API_KEY:
            import httpx
            prompt = f"""EXIT REVIEW: Should we close this position early?

Position: {trade.side} {trade.city_id} \
${trade.size_usd:.0f} @ {trade.entry_price:.2f}
Reason flagged: {candidate.reason}
Original edge: {candidate.original_edge:.1%}
Current edge: {candidate.current_edge:.1%}
Current model prob: {candidate.current_model_prob:.1%}
Current market price: {candidate.current_market_price:.1%}

Model forecasts:
{chr(10).join(
    f"  {k}: {v:.1f}°C" for k, v in sorted(model_values.items())
)}
Consensus: {consensus_mean:.1f}°C (std={consensus_std:.1f})

Should we EXIT this position or HOLD to resolution?
Respond JSON only: \
{{"verdict": "EXIT" or "HOLD", "rationale": "brief reason"}}"""

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": ANTHROPIC_API_KEY,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 150,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=15.0,
                )
                if resp.status_code == 200:
                    import json
                    import re

                    text = resp.json()["content"][0]["text"]
                    match = re.search(r"\{.*\}", text, re.DOTALL)
                    if match:
                        result = json.loads(match.group())
                        candidate.claude_verdict = result.get("verdict", "HOLD")
                        candidate.claude_rationale = result.get("rationale", "")
                        logger.info(
                            "CLAUDE EXIT: %s %s, %s: %s",
                            trade.city_id, candidate.reason,
                            candidate.claude_verdict,
                            candidate.claude_rationale,
                        )
    except Exception:
        logger.debug("Claude exit review failed", exc_info=True)

    # Gemini review: can you provide a SPECIFIC catalyst to hold?
    # Default: Claude's EXIT stands. Gemini must earn the hold.
    if candidate.claude_verdict == "EXIT":
        try:
            from weather_edge.analysis.gemini_reasoning import (
                GEMINI_API_KEY,
                GEMINI_API_URL,
                GEMINI_MODEL,
            )
            if GEMINI_API_KEY:
                import json
                import re

                import httpx

                prompt = f"""EXIT CHALLENGE: Claude recommends \
exiting this position. You may argue to HOLD, but ONLY if you \
can provide:
1. A SPECIFIC meteorological catalyst that will occur before \
resolution
2. A SPECIFIC timeframe for when the edge will recover
3. Evidence from the model data that the thesis is still valid

Generic arguments like "edge is still positive" or \
"market might overreact" are NOT sufficient to override an exit.

Position: {trade.side} {trade.city_id} \
${trade.size_usd:.0f}
Exit reason: {candidate.reason} \
(edge: {candidate.original_edge:.1%} \
\u2192 {candidate.current_edge:.1%})
Claude says: EXIT \u2014 {candidate.claude_rationale}

If you cannot provide a specific catalyst with timeframe, \
you MUST agree with the exit.

Respond JSON only:
{{"verdict": "AGREE_EXIT" or "HOLD", \
"catalyst": "specific event or null", \
"timeframe": "hours until catalyst or null", \
"rationale": "brief reason"}}"""

                url = (
                    f"{GEMINI_API_URL}/{GEMINI_MODEL}"
                    f":generateContent?key={GEMINI_API_KEY}"
                )
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        url,
                        json={
                            "contents": [{
                                "role": "user",
                                "parts": [{"text": prompt}],
                            }],
                            "generationConfig": {
                                "maxOutputTokens": 500,
                                "temperature": 0.3,
                                "thinkingConfig": {
                                    "thinkingBudget": 0,
                                },
                            },
                        },
                        timeout=20.0,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        text = (
                            data["candidates"][0]
                            ["content"]["parts"][-1]["text"]
                        )
                        text = re.sub(r"```json\s*", "", text)
                        text = re.sub(r"```\s*", "", text)
                        match = re.search(
                            r"\{.*\}", text.strip(), re.DOTALL,
                        )
                        if match:
                            result = json.loads(match.group())
                            candidate.gemini_verdict = result.get(
                                "verdict", "AGREE_EXIT",
                            )
                            candidate.gemini_rationale = result.get(
                                "rationale", "",
                            )
                            catalyst = result.get("catalyst")
                            logger.info(
                                "GEMINI EXIT: %s, %s "
                                "(catalyst: %s): %s",
                                trade.city_id,
                                candidate.gemini_verdict,
                                catalyst or "none",
                                candidate.gemini_rationale,
                            )
        except Exception:
            logger.debug("Gemini exit review failed", exc_info=True)

    # Final decision: Claude's EXIT is the default
    # Gemini can only override with a specific catalyst + non-high urgency
    if candidate.claude_verdict == "EXIT":
        if candidate.urgency == "high":
            # High urgency exits always go through
            candidate.final_decision = "EXIT"
        elif candidate.gemini_verdict == "HOLD":
            # Gemini earned a hold with specific catalyst
            candidate.final_decision = "HOLD"
            logger.info(
                "EXIT HELD by Gemini (with catalyst): %s %s",
                trade.city_id, candidate.reason,
            )
        else:
            # Gemini agrees, didn't respond, or no specific catalyst
            candidate.final_decision = "EXIT"
    else:
        candidate.final_decision = "HOLD"

    return candidate
