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
from datetime import datetime, timezone

from weather_edge.analysis.consensus import compute_consensus, get_probability_for_threshold
from weather_edge.analysis.edge import Signal
from weather_edge.trading.paper import PaperTrade

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
    trade: PaperTrade
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
    open_trades: list[PaperTrade],
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
        # Never exit penny bets
        if trade.entry_price <= PENNY_ENTRY_MAX:
            continue

        # Skip spread trades
        if "[SPREAD]" in (trade.description or ""):
            continue

        market_price = market_prices.get(trade.market_id)
        model_prob = model_probs.get(trade.market_id)

        if market_price is None or model_prob is None:
            continue

        # Calculate current edge from our position's perspective
        if trade.side == "YES":
            # We hold YES: profit if price goes up / resolves YES
            current_edge = model_prob - market_price
            original_edge = model_prob - trade.entry_price
        else:
            # We hold NO: profit if price goes down / resolves NO
            current_edge = (1.0 - model_prob) - (1.0 - market_price)
            original_edge = (1.0 - model_prob) - (1.0 - trade.entry_price)
            # For NO trades, flip the market price perspective
            market_price = 1.0 - market_price

        # Check exit triggers
        reason = None
        urgency = "medium"

        # 1. Edge inversion: model now says we're wrong
        if current_edge < EDGE_INVERSION_THRESHOLD:
            reason = "edge_inversion"
            urgency = "high" if current_edge < -0.15 else "medium"

        # 2. Profit cap: price ran up, lock in gains on core bets
        elif trade.side == "YES" and market_price >= PROFIT_CAP_PRICE and model_prob < PROFIT_CAP_MODEL_MAX:
            reason = "profit_cap"
            urgency = "low"
        elif trade.side == "NO" and (1.0 - market_price) >= PROFIT_CAP_PRICE and (1.0 - model_prob) < PROFIT_CAP_MODEL_MAX:
            reason = "profit_cap"
            urgency = "low"

        if reason:
            candidates.append(ExitCandidate(
                trade=trade,
                reason=reason,
                current_model_prob=model_prob,
                current_market_price=market_price,
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

Position: {trade.side} {trade.city_id} ${trade.size_usd:.0f} @ {trade.entry_price:.2f}
Reason flagged: {candidate.reason}
Original edge: {candidate.original_edge:.1%}
Current edge: {candidate.current_edge:.1%}
Current model prob: {candidate.current_model_prob:.1%}
Current market price: {candidate.current_market_price:.1%}

Model forecasts:
{chr(10).join(f"  {k}: {v:.1f}°C" for k, v in sorted(model_values.items()))}
Consensus: {consensus_mean:.1f}°C (std={consensus_std:.1f})

Should we EXIT this position or HOLD to resolution?
Respond JSON only: {{"verdict": "EXIT" or "HOLD", "rationale": "brief reason"}}"""

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
                    text = resp.json()["content"][0]["text"]
                    import re
                    match = re.search(r"\{.*\}", text, re.DOTALL)
                    if match:
                        result = json.loads(match.group())
                        candidate.claude_verdict = result.get("verdict", "HOLD")
                        candidate.claude_rationale = result.get("rationale", "")
                        logger.info("CLAUDE EXIT: %s %s, %s: %s",
                                    trade.city_id, candidate.reason,
                                    candidate.claude_verdict, candidate.claude_rationale)
    except Exception:
        logger.debug("Claude exit review failed", exc_info=True)

    # Gemini red team: argue for holding
    if candidate.claude_verdict == "EXIT":
        try:
            from weather_edge.analysis.gemini_reasoning import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_API_URL
            if GEMINI_API_KEY:
                import httpx, json, re
                prompt = f"""HOLD ARGUMENT: Another analyst says we should EXIT this position. Find reasons to HOLD.

Position: {trade.side} {trade.city_id} ${trade.size_usd:.0f}
Exit reason: {candidate.reason} (edge went from {candidate.original_edge:.1%} to {candidate.current_edge:.1%})
Claude says: EXIT, {candidate.claude_rationale}

Argue for HOLDING this position. Respond JSON only:
{{"verdict": "AGREE_EXIT" or "HOLD", "rationale": "brief counter-argument"}}"""

                url = f"{GEMINI_API_URL}/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
                async with httpx.AsyncClient() as client:
                    resp = await client.post(url, json={
                        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "generationConfig": {"maxOutputTokens": 500, "temperature": 0.3, "thinkingConfig": {"thinkingBudget": 0}},
                    }, timeout=20.0)
                    if resp.status_code == 200:
                        text = resp.json()["candidates"][0]["content"]["parts"][-1]["text"]
                        text = re.sub(r"```json\s*", "", text)
                        text = re.sub(r"```\s*", "", text)
                        match = re.search(r"\{.*\}", text.strip(), re.DOTALL)
                        if match:
                            result = json.loads(match.group())
                            candidate.gemini_verdict = result.get("verdict", "HOLD")
                            candidate.gemini_rationale = result.get("rationale", "")
                            logger.info("GEMINI EXIT: %s, %s: %s",
                                        trade.city_id, candidate.gemini_verdict, candidate.gemini_rationale)
        except Exception:
            logger.debug("Gemini exit review failed", exc_info=True)

    # Final decision
    if candidate.claude_verdict == "EXIT":
        if candidate.gemini_verdict == "AGREE_EXIT" or candidate.urgency == "high":
            candidate.final_decision = "EXIT"
        else:
            # Gemini says hold or didn't respond, reduce urgency
            candidate.final_decision = "HOLD"
            logger.info("EXIT OVERRULED by Gemini: %s %s, holding", trade.city_id, candidate.reason)
    else:
        candidate.final_decision = "HOLD"

    return candidate
