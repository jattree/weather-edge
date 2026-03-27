"""Main orchestration loop: fetch → consensus → edge → trade."""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone

from weather_edge.analysis.arbitrage import check_bucket_parity, find_parity_opportunities
from weather_edge.analysis.claude_reasoning import analyze_trade, record_decision, ANTHROPIC_API_KEY
from weather_edge.analysis.contracts import validate_emos_active, validate_model_count
from weather_edge.analysis.pattern_detector import detect_patterns, get_pattern_adjustment
from weather_edge.analysis.consensus import (
    compute_consensus,
    get_probability_for_threshold,
    MAX_BUCKET_PROBABILITY,
    SPREAD_INFLATION_FACTOR,
    EMOS_VARIANCE_FLOOR_C,
)
from weather_edge.analysis.edge import Signal, calculate_edge
from weather_edge.analysis.market_mapper import get_required_variable
from weather_edge.analysis.model_timing import is_golden_window
from weather_edge.analysis.resolver import resolve_open_trades
from weather_edge.config import CITIES, settings
from weather_edge.fetchers.openmeteo import fetch_city_forecasts
from weather_edge.fetchers.polymarket import MarketInfo, discover_weather_markets, get_price_snapshot
from weather_edge.models.enums import City
from weather_edge.trading.paper import PaperTrader

logger = logging.getLogger(__name__)


def compute_model_prob_for_market(market: MarketInfo, consensus) -> float | None:
    """Compute model probability for a market bucket.

    Handles the multi-bucket format with EMOS probability cap:
    - A single 2°F bucket should never exceed 70% at >12h horizon
    - Per Gemini: >90% on a single bucket is "likely broken"
    """
    from weather_edge.analysis.consensus import MAX_BUCKET_PROBABILITY

    prob = None

    if market.threshold_dir == "lte":
        p_gte = get_probability_for_threshold(consensus, market.threshold_high_c or market.threshold_value, "gte")
        prob = 1.0 - p_gte

    elif market.threshold_dir == "range":
        if market.threshold_low_c is not None and market.threshold_high_c is not None:
            p_gte_low = get_probability_for_threshold(consensus, market.threshold_low_c, "gte")
            p_gte_high = get_probability_for_threshold(consensus, market.threshold_high_c + 1.0, "gte")
            prob = max(0.0, p_gte_low - p_gte_high)

    elif market.threshold_dir == "gte":
        prob = get_probability_for_threshold(consensus, market.threshold_value, "gte")

    elif market.threshold_dir == "any":
        prob = get_probability_for_threshold(consensus, 0.0, "any")

    # Apply bucket probability cap for range/lte buckets (narrow temperature ranges)
    if prob is not None and market.threshold_dir in ("range", "lte"):
        prob = min(prob, MAX_BUCKET_PROBABILITY)

    return prob


async def run_cycle(
    paper_trader: PaperTrader,
    target_dates: list[date] | None = None,
    run_ai_reasoning: bool = True,
) -> tuple[list[Signal], dict[tuple, list], dict[str, dict]]:
    """Run one full fetch → analyze → signal cycle.

    Args:
        run_ai_reasoning: If False, skip Claude + Gemini calls (sniper-triggered cycles).

    Returns:
        (signals, forecast_cache) where forecast_cache maps (city_id, date) -> forecasts
    """
    if target_dates is None:
        today = date.today()
        target_dates = [today, today + timedelta(days=1), today + timedelta(days=2)]

    # Contract: verify EMOS calibration is active at cycle start
    emos_check = validate_emos_active(SPREAD_INFLATION_FACTOR, MAX_BUCKET_PROBABILITY, EMOS_VARIANCE_FLOOR_C)
    if not emos_check.valid:
        logger.warning("CONTRACT VIOLATION [%s]: %s", emos_check.code, emos_check.error)

    # Resolve any open trades before placing new ones
    # This frees up capital and updates P&L before new signals are computed
    try:
        resolved_count = await resolve_open_trades(paper_trader)
        if resolved_count > 0:
            logger.info("Resolved %d trades at cycle start", resolved_count)
    except Exception:
        logger.exception("Trade resolution failed, continuing with cycle")

    all_signals: list[Signal] = []
    _forecast_cache: dict[tuple, list] = {}  # (city_id, date) -> forecasts for Claude

    # Check if we're in a golden window (model just updated)
    if is_golden_window():
        logger.info("*** GOLDEN WINDOW: Fresh model data, market may be stale ***")

    # Step 1: Discover active weather markets (prices included from Gamma API)
    logger.info("=== Discovering Polymarket weather markets ===")
    markets = await discover_weather_markets()

    # Aggregate volume and liquidity by city for dashboard
    city_volume: dict[str, dict] = {}
    for m in markets:
        if m.city_id:
            cid = m.city_id.value
            if cid not in city_volume:
                city_volume[cid] = {"volume_24h": 0.0, "liquidity": 0.0, "markets": 0}
            city_volume[cid]["volume_24h"] += m.volume_24h or 0
            city_volume[cid]["liquidity"] += m.liquidity or 0
            city_volume[cid]["markets"] += 1

    # Step 1b: Check bucket parity for arbitrage opportunities
    if markets:
        parity_checks = check_bucket_parity(markets)
        arb_opportunities = find_parity_opportunities(parity_checks)
        if arb_opportunities:
            logger.info("=== %d PARITY ARBITRAGE opportunities ===", len(arb_opportunities))
            for arb in arb_opportunities:
                logger.info(
                    "  %s %s: YES sum=%.3f (%+.1f%%)",
                    arb.city_id.upper(), arb.target_date, arb.yes_sum, arb.deviation * 100,
                )

    if not markets:
        logger.warning("No weather markets found for tracked cities.")

    # Group markets by city+date
    market_groups: dict[tuple[City, date], list[MarketInfo]] = {}
    for m in markets:
        if m.city_id and m.target_date in target_dates:
            key = (m.city_id, m.target_date)
            market_groups.setdefault(key, []).append(m)

    logger.info("Active market groups: %d city-date combos", len(market_groups))

    # Step 2: For each city with markets, fetch forecasts and compute signals
    cities_processed = set()
    for (city_id, target_date), city_markets in market_groups.items():
        city_config = CITIES[city_id]

        if (city_id, target_date) not in cities_processed:
            logger.info("=== %s (%s) on %s, %d markets ===",
                       city_config.name, city_config.icao, target_date, len(city_markets))
            cities_processed.add((city_id, target_date))

        # Fetch multi-model forecasts
        forecasts = await fetch_city_forecasts(city_id, target_date)
        _forecast_cache[(city_id, target_date)] = forecasts
        if not forecasts:
            logger.warning("No forecasts for %s on %s", city_id.value, target_date)
            continue

        # Contract: verify sufficient models for reliable consensus
        has_regional = bool(city_config.regional_models)
        model_check = validate_model_count(len(forecasts), has_regional)
        if not model_check.valid:
            logger.warning(
                "CONTRACT VIOLATION [%s]: %s, skipping %s on %s",
                model_check.code, model_check.error, city_id.value, target_date,
            )
            continue

        # Detect bust-causing weather patterns (Chinook, Foehn, marine layer, etc.)
        pattern_alerts = detect_patterns(city_id, forecasts)
        pattern_conf_mult, pattern_bias = get_pattern_adjustment(city_id, pattern_alerts)

        # Determine which variables we need
        variables_needed = set()
        for m in city_markets:
            var = get_required_variable(m)
            if var:
                variables_needed.add(var)
        variables_needed.add("temp_max_c")  # Always compute

        # Compute consensus per variable
        for variable in variables_needed:
            # Collect thresholds from all markets needing this variable
            thresholds = []
            for m in city_markets:
                if get_required_variable(m) == variable:
                    if m.threshold_low_c is not None:
                        thresholds.append(m.threshold_low_c)
                    if m.threshold_high_c is not None:
                        thresholds.append(m.threshold_high_c)
                    thresholds.append(m.threshold_value)

            consensus = compute_consensus(
                city_id, str(target_date), variable, forecasts,
                sorted(set(thresholds)) if thresholds else None,
            )
            if consensus is None:
                continue

            logger.info(
                "  %s/%s: mean=%.1f°C std=%.1f conf=%.0f%% (%d models)",
                city_id.value, variable,
                consensus.weighted_mean, consensus.std_dev,
                consensus.confidence * 100, consensus.model_count,
            )

            # Compute edge for each matching market bucket
            for market in city_markets:
                if get_required_variable(market) != variable:
                    continue

                # Use price from Gamma API
                market_prob = market.yes_price
                if market_prob <= 0.01 or market_prob >= 0.99:
                    continue  # Skip extreme prices (no edge possible)

                model_prob = compute_model_prob_for_market(market, consensus)
                if model_prob is None:
                    continue

                # Hours to resolution
                now = datetime.now(timezone.utc)
                resolution_dt = datetime.combine(
                    market.target_date + timedelta(days=1),
                    datetime.min.time(),
                ).replace(tzinfo=timezone.utc)
                hours_to = max(0, (resolution_dt - now).total_seconds() / 3600)

                # Apply pattern-based confidence boost
                adjusted_conf = min(1.0, consensus.confidence * pattern_conf_mult)

                signal = calculate_edge(
                    market_id=market.market_id,
                    model_prob=model_prob,
                    market_prob=market_prob,
                    model_confidence=adjusted_conf,
                    consensus_id=None,
                    hours_to_resolution=hours_to,
                    city_id=city_id.value,
                    description=market.question[:80],
                )
                all_signals.append(signal)

    # === Claude + Gemini reasoning layer ===
    # Only on main cycles (not sniper-triggered) to save API costs
    if run_ai_reasoning and ANTHROPIC_API_KEY and all_signals:
        tradeable = sorted(
            [s for s in all_signals if s.confidence_tier.value != "low"],
            key=lambda s: abs(s.edge),
            reverse=True,
        )[:3]

        golden = is_golden_window()
        for signal in tradeable:
            # Build model context for Claude from cached forecasts
            model_vals = {}
            consensus_mean = signal.model_prob * 30  # Rough temp estimate from probability
            consensus_std = 2.0
            for (cid, td), f_list in _forecast_cache.items():
                if cid.value == signal.city_id:
                    model_vals = {f.model_name: f.temp_max_c for f in f_list if f.temp_max_c is not None}
                    if model_vals:
                        vals = list(model_vals.values())
                        consensus_mean = sum(vals) / len(vals)
                        consensus_std = (max(vals) - min(vals)) / 2 if len(vals) > 1 else 0.5
                    break

            reasoning = await analyze_trade(
                signal, model_vals, consensus_mean, consensus_std,
            )
            if reasoning:
                # Record every decision for the AI Decisions dashboard tab
                record_decision(reasoning)

                if not reasoning.should_trade:
                    logger.info("CLAUDE SKIP: %s %s, %s", signal.city_id, signal.description[:40], reasoning.rationale)
                    signal.confidence_tier = signal.confidence_tier  # Keep as-is but don't trade
                    continue
                # Apply Claude's confidence adjustment to position size
                signal.recommended_size = round(signal.recommended_size * reasoning.confidence_adjustment, 2)

                # === Gemini red team on Claude-approved trades ===
                try:
                    from weather_edge.analysis.gemini_reasoning import red_team_trade
                    gemini_result = await red_team_trade(
                        signal, model_vals, consensus_mean, consensus_std,
                        claude_rationale=reasoning.rationale,
                    )
                    if gemini_result:
                        dissent = gemini_result.get("dissent_strength", 0)
                        verdict = gemini_result.get("verdict", "AGREE")
                        # Record Gemini decision for dashboard
                        from weather_edge.analysis.claude_reasoning import _decision_history
                        _decision_history.insert(0, {
                            "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                            "city": signal.city_id.upper() if isinstance(signal.city_id, str) else signal.city_id,
                            "decision": "DISSENT" if verdict == "DISSENT" else "AGREE",
                            "signal": signal.description[:60],
                            "adjustment": round(1.0 - dissent, 2),
                            "rationale": "; ".join(gemini_result.get("counter_arguments", [])[:2]),
                            "risk_factors": [gemini_result.get("risk_the_bull_missed", "")],
                            "source": "gemini",
                        })
                        # High dissent = reduce size
                        if dissent >= 0.7:
                            old_size = signal.recommended_size
                            signal.recommended_size = round(signal.recommended_size * 0.5, 2)
                            logger.info(
                                "GEMINI DISSENT: %s, reducing $%.0f -> $%.0f (dissent=%.1f)",
                                signal.city_id, old_size, signal.recommended_size, dissent,
                            )
                except Exception:
                    logger.debug("Gemini red team skipped", exc_info=True)

    # Place trades for all signals + generate spread capture orders
    from weather_edge.fetchers.polymarket import fetch_book_prices
    from weather_edge.trading.market_maker import MarketMaker
    market_maker = MarketMaker()

    # Build market prices dict using real order book asks (not midpoints)
    # Per Gemini: spread only exists if YES_ask + NO_ask < 1.00
    market_prices: dict[str, dict] = {}
    market_by_id = {m.market_id: m for m in markets}

    # Fetch book prices for top 5 signals only (each = 2 CLOB API calls)
    logger.info("Fetching order book prices for spread detection...")
    top_signals = sorted(all_signals, key=lambda s: abs(s.edge), reverse=True)[:5]
    for signal in top_signals:
        m = market_by_id.get(signal.market_id)
        if m and m.token_id_yes and m.token_id_no:
            try:
                book = await asyncio.wait_for(fetch_book_prices(m), timeout=10.0)
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug("Book fetch timeout/error for %s: %s", signal.city_id, e)
                continue
            if book:
                market_prices[signal.market_id] = {
                    "yes_price": book.get("yes_ask") or m.yes_price,
                    "no_price": book.get("no_ask") or (1.0 - m.yes_price),
                    "bid": book.get("yes_bid") or (m.yes_price - 0.01),
                    "ask": book.get("yes_ask") or (m.yes_price + 0.01),
                    "spread_profitable": book.get("profitable", False),
                }
                if book.get("profitable"):
                    logger.info(
                        "SPREAD OPP: %s, YES_ask=%.3f NO_ask=%.3f total=%.3f profit=%.3f/share",
                        signal.city_id, book["yes_ask"], book["no_ask"],
                        book["spread_cost"], book["spread_profit"],
                    )
    logger.info("Book price fetch complete, placing trades...")

    for signal in all_signals:
        trade = paper_trader.place_trade(signal)
        # Generate hedge/spread order only when book shows real profit
        if trade:
            prices = market_prices.get(signal.market_id, {})
            if prices.get("spread_profitable"):
                hedge = market_maker.generate_hedge_orders(signal, market_prices, settings.bankroll)
                if hedge:
                    paper_trader.place_spread_trade(signal, hedge)

    # Log spread capture summary
    spread_summary = market_maker.simulate_spread_pnl()
    if spread_summary["spread_orders"] > 0:
        logger.info(
            "SPREAD CAPTURE: %d orders, est. guaranteed P&L=$%.2f",
            spread_summary["spread_orders"], spread_summary["estimated_guaranteed_pnl"],
        )

    # === Early exit monitor: check if open positions should be closed ===
    try:
        from weather_edge.analysis.exit_monitor import scan_for_exits, ai_review_exit
        # Build current market prices and model probs for open trades
        current_market_prices = {m.market_id: m.yes_price for m in markets}
        current_model_probs = {}
        for signal in all_signals:
            current_model_probs[signal.market_id] = signal.model_prob

        exit_candidates = scan_for_exits(paper_trader.open_trades, current_market_prices, current_model_probs)
        if exit_candidates:
            logger.info("EXIT MONITOR: %d candidates found, reviewing top 3...", len(exit_candidates))
            for candidate in exit_candidates[:3]:
                # Get model context for AI review
                model_vals = {}
                c_mean, c_std = 0.0, 1.0
                for (cid, td), f_list in _forecast_cache.items():
                    if cid.value == candidate.trade.city_id:
                        model_vals = {f.model_name: f.temp_max_c for f in f_list if f.temp_max_c is not None}
                        if model_vals:
                            vals = list(model_vals.values())
                            c_mean = sum(vals) / len(vals)
                            c_std = (max(vals) - min(vals)) / 2 if len(vals) > 1 else 0.5
                        break

                candidate = await ai_review_exit(candidate, model_vals, c_mean, c_std)

                if candidate.final_decision == "EXIT":
                    logger.warning(
                        "EARLY EXIT: %s %s $%.0f, %s (Claude: %s, Gemini: %s)",
                        candidate.trade.side, candidate.trade.city_id,
                        candidate.trade.size_usd, candidate.reason,
                        candidate.claude_verdict, candidate.gemini_verdict,
                    )
                    # In paper mode: close at current market price
                    paper_trader.close_position(candidate.trade, candidate.current_market_price)

                # Record exit decision to AI Decisions tab
                from weather_edge.analysis.claude_reasoning import _decision_history
                _decision_history.insert(0, {
                    "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                    "city": candidate.trade.city_id.upper(),
                    "decision": "EXIT" if candidate.final_decision == "EXIT" else "HOLD",
                    "signal": f"[EXIT CHECK] {candidate.reason}: {candidate.trade.description[:40]}",
                    "adjustment": round(candidate.current_edge, 2),
                    "rationale": candidate.claude_rationale or "No AI review",
                    "risk_factors": [candidate.gemini_rationale or ""],
                    "source": "exit_monitor",
                })
    except Exception:
        logger.debug("Exit monitor failed", exc_info=True)

    # Also fetch forecasts for cities without active markets (monitoring)
    # But only for tomorrow (not all dates) to save API calls
    tomorrow = target_dates[1] if len(target_dates) > 1 else target_dates[0]
    for city_id in City:
        if (city_id, tomorrow) not in _forecast_cache:
            forecasts = await fetch_city_forecasts(city_id, tomorrow)
            if forecasts:
                _forecast_cache[(city_id, tomorrow)] = forecasts
                consensus = compute_consensus(city_id, str(tomorrow), "temp_max_c", forecasts)
                if consensus:
                    logger.info(
                        "  %s (no markets): mean=%.1f°C conf=%.0f%%",
                        city_id.value, consensus.weighted_mean, consensus.confidence * 100,
                    )

    return all_signals, _forecast_cache, city_volume


async def run_loop(paper_trader: PaperTrader) -> None:
    """Run the fetch-analyze-trade loop continuously."""
    interval = settings.fetch_interval_minutes * 60
    cycle_num = 0

    while True:
        cycle_num += 1
        logger.info("===== CYCLE %d START =====", cycle_num)
        try:
            signals, _, _ = await run_cycle(paper_trader)
            tradeable = [s for s in signals if s.confidence_tier.value != "low"]
            logger.info(
                "Cycle %d complete: %d signals, %d tradeable, P&L=$%.2f",
                cycle_num, len(signals), len(tradeable), paper_trader.total_pnl,
            )
        except Exception:
            logger.exception("Cycle %d failed", cycle_num)

        logger.info("Sleeping %d minutes until next cycle...", settings.fetch_interval_minutes)
        await asyncio.sleep(interval)
