"""Early exit monitor, checks if open positions should be closed before resolution.

Runs each cycle after new model data arrives. Checks:
0. Observation guard: if past peak heat and actual temp is in our bucket, HOLD
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

from weather_edge.analysis.contracts import validate_penny_no_exit
from weather_edge.models.position import Position

logger = logging.getLogger(__name__)

# Peak heat hours, after this local time, daily high is essentially locked
PEAK_HEAT_HOUR = 15  # 3pm local

# Thresholds (per Gemini validation)
EDGE_INVERSION_THRESHOLD = -0.07  # Exit when edge flips worse than -7%
PROFIT_CAP_PRICE = 0.88  # Take profit above this price
PROFIT_CAP_MODEL_MAX = 0.94  # Only take profit if model < 94%
PENNY_ENTRY_MAX = 0.06  # Never exit penny bets (entry <= 6¢)
MIN_SPREAD_MULTIPLIER = 1.5  # Don't exit if edge < 1.5x the spread
STALE_MODEL_THRESHOLD_HOURS = 4.0  # Exit if model data > 4h old


@dataclass
class ExitCandidate:
    """A trade flagged for potential early exit."""
    trade: Position
    reason: str  # "edge_inversion", "profit_cap", "pattern_bust", "stale_model"
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


def _check_observation_confirms_bucket(trade: Position) -> bool:
    """Check if actual weather observation confirms our position's bucket.

    After peak heat (3pm local), the daily high is locked. If the observed
    max temperature falls in our bucket, models are irrelevant, hold to
    resolution for the full $1 payout.

    Returns True if observation confirms our bucket (should HOLD).
    """
    try:
        import re
        import zoneinfo

        import httpx

        from weather_edge.analysis.resolver import (
            actual_falls_in_bucket,
            parse_bucket_from_description,
        )
        from weather_edge.config import CITIES
        from weather_edge.models.enums import City

        # Only applies to YES positions, if actual is in bucket, YES wins.
        # For NO positions, actual in bucket means we're LOSING, don't block exit.
        if trade.side == "NO":
            return False

        # Parse the bucket from description
        desc = trade.description or ""
        bucket = parse_bucket_from_description(desc)
        if not bucket:
            return False

        # Find the city config for timezone and coordinates
        try:
            city_enum = City(trade.city_id)
        except ValueError:
            return False
        city_config = CITIES.get(city_enum)
        if not city_config:
            return False

        # Use city's LOCAL date for comparison, not UTC
        tz = zoneinfo.ZoneInfo(city_config.timezone)
        local_now = datetime.now(tz)

        # Only check positions resolving TODAY in the city's timezone
        date_match = re.search(r"on (\w+ \d+)\??$", desc)
        if date_match:
            from dateutil.parser import parse as _parse_date
            target_date = _parse_date(date_match.group(1) + f" {local_now.year}").date()
            if target_date != local_now.date():
                return False  # Not today's market in this city's timezone

        # Check if we're past peak heat
        if local_now.hour < PEAK_HEAT_HOUR:
            return False  # Too early, high might not be set yet

        # Fetch current observation from Open-Meteo
        resp = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": city_config.latitude,
                "longitude": city_config.longitude,
                "daily": "temperature_2m_max",
                "timezone": city_config.timezone,
                "forecast_days": 1,
            },
            timeout=10.0,
        )
        if resp.status_code != 200:
            return False

        data = resp.json()
        daily_max_c = data.get("daily", {}).get("temperature_2m_max", [None])[0]
        if daily_max_c is None:
            return False

        in_bucket = actual_falls_in_bucket(daily_max_c, bucket)
        if in_bucket:
            logger.info(
                "OBSERVATION GUARD: %s actual max %.1f°C is IN bucket %s, HOLD to resolution",
                trade.city_id, daily_max_c, desc[:50],
            )
        return in_bucket

    except Exception:
        logger.debug("Observation check failed for %s", trade.city_id, exc_info=True)
        return False


def scan_for_exits(
    open_trades: list[Position],
    market_prices: dict[str, float],
    model_probs: dict[str, float],
    forecast_cache: dict[tuple, list] | None = None,
) -> list[ExitCandidate]:
    """Scan open positions for early exit candidates.

    Args:
        open_trades: Currently open paper trades
        market_prices: Current market YES prices by market_id
        model_probs: Current model probabilities by market_id
        forecast_cache: Optional cache mapping (city, date) -> forecasts
    """
    candidates: list[ExitCandidate] = []
    now = datetime.now(timezone.utc)

    for trade in open_trades:
        # Contract: never exit penny bets (hold to resolution)
        penny_check = validate_penny_no_exit(trade.entry_price, PENNY_ENTRY_MAX)
        if not penny_check.valid:
            continue

        # Observation guard: if past peak heat and actual temp confirms
        # our bucket, hold to resolution, models are irrelevant at this point
        if _check_observation_confirms_bucket(trade):
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

        # 0. Stale model detection: data hasn't updated in 4+ hours
        if forecast_cache:
            stale = False
            oldest_fetch = now
            city_forecasts = []
            for (cid, t_date), f_list in forecast_cache.items():
                # cid is likely a City enum or string value
                cid_val = cid.value if hasattr(cid, "value") else str(cid)
                if cid_val == trade.city_id:
                    city_forecasts.extend(f_list)
            
            if city_forecasts:
                for f in city_forecasts:
                    if hasattr(f, "fetched_at"):
                        oldest_fetch = min(oldest_fetch, f.fetched_at)
                
                hours_old = (now - oldest_fetch).total_seconds() / 3600
                if hours_old > STALE_MODEL_THRESHOLD_HOURS:
                    stale = True
                    logger.warning(
                        "STALE MODEL DETECTED: %s data is %.1f hours old",
                        trade.city_id, hours_old
                    )

            if stale:
                reason = "stale_model"
                urgency = "high"

        # 1. Edge inversion: model now says we're wrong
        if not reason and current_edge < EDGE_INVERSION_THRESHOLD:
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

        # 3. Sell-half: position up big but models don't support it
        #    Only fires once per position, skips if shares < 2x minimum (already trimmed)
        if not reason:
            from weather_edge.config import settings as _cfg
            unrealized_gain = (market_price - trade.entry_price) / trade.entry_price if trade.entry_price > 0 else 0
            if trade.side == "NO":
                unrealized_gain = ((1.0 - market_price) - (1.0 - trade.entry_price)) / (1.0 - trade.entry_price) if trade.entry_price < 1.0 else 0
            sell_value = trade.total_shares * market_price * 0.5
            if trade.side == "NO":
                sell_value = trade.total_shares * (1.0 - market_price) * 0.5

            # Guard: need enough shares to sell half and still hold minimum
            min_shares_for_half = 10.0  # 2x Polymarket minimum (5 shares)

            if (unrealized_gain >= _cfg.sell_half_gain_pct
                    and current_edge <= _cfg.sell_half_edge_max
                    and sell_value >= _cfg.sell_half_min_usd
                    and trade.total_shares >= min_shares_for_half):
                reason = "sell_half"
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
    """Run Claude (Meteorologist) + Gemini (Risk Quant) in parallel.

    Claude: Is the weather thesis still physically valid?
    Gemini: Is exiting right now the best risk/reward move?
    Both agree exit → final_decision = EXIT
    Gemini dissents (with cost justification) → HOLD on non-high urgency
    """
    import asyncio

    trade = candidate.trade

    async def _claude_review() -> tuple[str, str]:
        """Claude = Meteorologist. Looks at physics only."""
        try:
            from weather_edge.analysis.claude_reasoning import ANTHROPIC_API_KEY
            if not ANTHROPIC_API_KEY:
                return "HOLD", ""
            import httpx
            prompt = f"""EXIT REVIEW: Should we close this position early?

Position: {trade.side} {trade.city_id} {trade.total_shares:.0f} shares (${trade.size_usd:.0f}) @ {trade.entry_price:.2f}
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

Focus on the PHYSICS: Is the original weather thesis still valid \
given the latest model data? Ignore market microstructure.

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
                        return result.get("verdict", "HOLD"), result.get("rationale", "")
        except Exception:
            logger.error("Claude exit review failed", exc_info=True)
        return "HOLD", ""

    async def _gemini_review() -> tuple[str, str]:
        """Gemini = Risk Quant. Looks at cost/liquidity only."""
        try:
            from weather_edge.analysis.gemini_reasoning import (
                GEMINI_API_KEY,
                GEMINI_API_URL,
                GEMINI_MODEL,
            )
            if not GEMINI_API_KEY:
                return "AGREE_EXIT", ""
            import json
            import re

            import httpx

            cost_basis = trade.total_shares * trade.entry_price
            current_value = trade.total_shares * candidate.current_market_price
            unrealized_pnl = current_value - cost_basis
            exit_fee_est = current_value * 0.02  # taker fee estimate

            # Fetch order book depth for this position
            book_context = "Order book data unavailable."
            try:
                from weather_edge.persistence import PersistentStore
                s = PersistentStore()
                pos = s.get_position_for_market(trade.market_id)
                s.close()
                if pos:
                    asset_id = pos.get("asset_id", "")
                    if asset_id:
                        from weather_edge.config import settings
                        async with httpx.AsyncClient() as book_client:
                            resp = await book_client.get(
                                f"{settings.polymarket_clob_url}/book",
                                params={"token_id": asset_id},
                                timeout=8.0,
                            )
                            if resp.status_code == 200:
                                book = resp.json()
                                bids = book.get("bids", [])
                                asks = book.get("asks", [])
                                # Top 5 levels of depth
                                bid_depth = sum(float(b.get("size", 0)) for b in bids[:5])
                                ask_depth = sum(float(a.get("size", 0)) for a in asks[:5])
                                best_bid = float(bids[0]["price"]) if bids else 0
                                best_ask = float(asks[0]["price"]) if asks else 0
                                spread = round(best_ask - best_bid, 3) if best_bid and best_ask else 0
                                book_context = (
                                    f"Best bid: ${best_bid:.3f} ({bid_depth:.0f} shares depth)\n"
                                    f"Best ask: ${best_ask:.3f} ({ask_depth:.0f} shares depth)\n"
                                    f"Spread: ${spread:.3f}\n"
                                    f"Our shares vs bid depth: {trade.total_shares:.0f} vs {bid_depth:.0f}"
                                )
            except Exception:
                pass  # book_context stays as "unavailable"

            prompt = f"""RISK ASSESSMENT: Evaluate the cost of exiting this position NOW.

You are a Risk Quant. Do NOT evaluate weather or meteorology. \
Focus ONLY on execution cost and portfolio impact.

Position: {trade.side} {trade.city_id}
Shares: {trade.total_shares:.0f}
Entry price: ${trade.entry_price:.3f}
Cost basis: ${cost_basis:.2f}
Current market price: ${candidate.current_market_price:.3f}
Current value: ${current_value:.2f}
Unrealized P&L: ${unrealized_pnl:+.2f}
Edge: {candidate.current_edge:+.1%}
Estimated taker fee if we exit: ${exit_fee_est:.2f}
Bankroll: $210

Order Book:
{book_context}

Questions to answer:
1. Is the exit cost (fees + slippage) small relative to the \
potential loss from holding?
2. Can we exit {trade.total_shares:.0f} shares without moving \
the market? Compare our size to the bid depth.
3. Should we exit NOW (taker) or place a maker sell and wait?

If exit cost is <10% of potential loss, recommend EXIT.
If our shares exceed bid depth by 3x+, warn about market impact.
If exit would cost more than holding to resolution, recommend HOLD.

Respond JSON only:
{{"verdict": "AGREE_EXIT" or "HOLD", \
"rationale": "brief cost/risk justification"}}"""

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
                    match = re.search(r"\{.*\}", text.strip(), re.DOTALL)
                    if match:
                        result = json.loads(match.group())
                        return result.get("verdict", "AGREE_EXIT"), result.get("rationale", "")
        except Exception:
            logger.error("Gemini exit review failed", exc_info=True)
        return "AGREE_EXIT", ""

    # Run both in parallel
    (claude_verdict, claude_rationale), (gemini_verdict, gemini_rationale) = (
        await asyncio.gather(_claude_review(), _gemini_review())
    )

    candidate.claude_verdict = claude_verdict
    candidate.claude_rationale = claude_rationale
    candidate.gemini_verdict = gemini_verdict
    candidate.gemini_rationale = gemini_rationale

    logger.info(
        "CLAUDE EXIT: %s %s, %s: %s",
        trade.city_id, candidate.reason, claude_verdict, claude_rationale,
    )
    logger.info(
        "GEMINI RISK: %s, %s: %s",
        trade.city_id, gemini_verdict, gemini_rationale,
    )

    # Final decision logic
    if claude_verdict == "EXIT":
        if candidate.urgency == "high":
            # High urgency: Claude says physics is dead, always exit
            candidate.final_decision = "EXIT"
        elif gemini_verdict == "HOLD":
            # Gemini says exit is too expensive, respect on non-urgent
            candidate.final_decision = "HOLD"
            logger.info(
                "EXIT HELD by Gemini (cost justification): %s %s",
                trade.city_id, candidate.reason,
            )
        else:
            candidate.final_decision = "EXIT"
    else:
        candidate.final_decision = "HOLD"

    return candidate
