"""Main orchestration loop: fetch → consensus → edge → trade."""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone

from weather_edge.analysis.arbitrage import (
    check_bucket_parity,
    find_parity_opportunities,
)
from weather_edge.analysis.claude_reasoning import (
    ANTHROPIC_API_KEY,
    analyze_trade,
    record_decision,
)
from weather_edge.analysis.consensus import (
    EMOS_VARIANCE_FLOOR_C,
    MAX_BUCKET_PROBABILITY,
    SPREAD_INFLATION_FACTOR,
    compute_consensus,
    get_probability_for_threshold,
)
from weather_edge.analysis.contracts import (
    validate_emos_active,
    validate_fee_alpha_ratio,
    validate_model_count,
)
from weather_edge.analysis.edge import Signal, calculate_edge
from weather_edge.analysis.market_mapper import get_required_variable
from weather_edge.analysis.model_timing import is_golden_window
from weather_edge.analysis.pattern_detector import (
    detect_patterns,
    get_pattern_adjustment,
)
from weather_edge.analysis.resolver import resolve_open_trades
from weather_edge.config import CITIES, settings
from weather_edge.fetchers.openmeteo import fetch_city_forecasts
from weather_edge.fetchers.polymarket import (
    MarketInfo,
    discover_weather_markets,
)
from weather_edge.models.enums import City
from weather_edge.trading.paper import PaperTrader

logger = logging.getLogger(__name__)

MIN_SELL_SHARES = 5.0  # Polymarket minimum order size


def _round_price(price: float) -> float:
    """Round price to valid Polymarket tick size (1 cent)."""
    return round(max(0.01, min(0.99, price)), 2)


def compute_model_prob_for_market(market: MarketInfo, consensus) -> float | None:
    """Compute model probability for a market bucket.

    Handles the multi-bucket format with EMOS probability cap:
    - A single 2°F bucket should never exceed 70% at >12h horizon
    - Per Gemini: >90% on a single bucket is "likely broken"
    """
    from weather_edge.analysis.consensus import MAX_BUCKET_PROBABILITY

    prob = None

    if market.threshold_dir == "lte":
        p_gte = get_probability_for_threshold(
            consensus,
            market.threshold_high_c or market.threshold_value,
            "gte",
        )
        prob = 1.0 - p_gte

    elif market.threshold_dir == "range":
        if (
        market.threshold_low_c is not None
        and market.threshold_high_c is not None
    ):
            p_gte_low = get_probability_for_threshold(
                consensus, market.threshold_low_c, "gte",
            )
            p_gte_high = get_probability_for_threshold(
                consensus, market.threshold_high_c + 1.0, "gte",
            )
            prob = max(0.0, p_gte_low - p_gte_high)

    elif market.threshold_dir == "gte":
        prob = get_probability_for_threshold(consensus, market.threshold_value, "gte")

    elif market.threshold_dir == "any":
        prob = get_probability_for_threshold(consensus, 0.0, "any")

    # Apply bucket probability cap for range/lte buckets (narrow temperature ranges)
    # During extreme events (tight model agreement + anomalous temps), raise the cap
    if prob is not None and market.threshold_dir in ("range", "lte"):
        from weather_edge.analysis.consensus import (
            CITY_CLIMATOLOGY,
            CLIMATOLOGICAL_MEAN,
            CLIMATOLOGICAL_STD,
            MAX_BUCKET_PROBABILITY_EXTREME,
        )
        cap = MAX_BUCKET_PROBABILITY
        if consensus.std_dev < 1.5 and consensus.model_count >= 5:
            # Models tightly clustered, check if extreme for THIS city
            city_key = market.city_id.value if market.city_id else ""
            clim = CITY_CLIMATOLOGY.get(city_key)
            if clim:
                clim_mean, clim_std = clim
            else:
                clim_mean = CLIMATOLOGICAL_MEAN.get("temp_max_c", 15.0)
                clim_std = CLIMATOLOGICAL_STD.get("temp_max_c", 6.0)
            anomaly = (
                abs(consensus.weighted_mean - clim_mean) / clim_std
                if clim_std > 0 else 0
            )
            if anomaly > 2.0:
                cap = MAX_BUCKET_PROBABILITY_EXTREME
                logger.info(
                    "EXTREME EVENT: %s consensus=%.1f°C "
                    "(%.1f sigma from %.1f°C norm), "
                    "std=%.1f, cap raised to %.0f%%",
                    city_key, consensus.weighted_mean,
                    anomaly, clim_mean,
                    consensus.std_dev, cap * 100,
                )
        prob = min(prob, cap)

    return prob


async def run_cycle(
    paper_trader: PaperTrader | None,
    target_dates: list[date] | None = None,
    run_ai_reasoning: bool = True,
    live_executor=None,
    store=None,
    forecast_cache: dict[tuple, list] | None = None,
) -> tuple[list[Signal], dict[tuple, list], dict[str, dict]]:
    """Run one full fetch → analyze → signal cycle.

    Args:
        paper_trader: Paper trader instance (None if paper disabled).
        run_ai_reasoning: If False, skip Claude + Gemini calls (sniper-triggered cycles).
        live_executor: Optional TradeExecutor for real order placement.
        store: PersistentStore instance (shared by both paper and live).
        forecast_cache: Optional persistent cache of forecasts across cycles.

    Returns:
        (signals, forecast_cache) where forecast_cache maps (city_id, date) -> forecasts
    """
    # Store can come from paper_trader or be passed directly
    if store is None and paper_trader is not None:
        store = paper_trader.store

    if target_dates is None:
        today = date.today()
        target_dates = [
            today,
            today + timedelta(days=1),
            today + timedelta(days=2),
            today + timedelta(days=3),
        ]

    # --- SWING BOT: 36h horizon filter ---
    # Block entries on markets resolving too soon. We get front-run by bots
    # with fresher NWS data on short-dated markets. Our bias correction edge
    # is strongest at 48-72h where model ensembles still disagree.
    now_utc = datetime.now(timezone.utc)
    min_horizon = timedelta(hours=36)
    original_dates = list(target_dates)
    target_dates = [
        d for d in target_dates
        if datetime.combine(d + timedelta(days=1), datetime.min.time()).replace(
            tzinfo=timezone.utc,
        ) - now_utc >= min_horizon
    ]
    blocked = set(original_dates) - set(target_dates)
    if blocked:
        logger.info(
            "HORIZON FILTER: blocked %s (resolve < 36h)",
            ", ".join(str(d) for d in sorted(blocked)),
        )

    # Contract: verify EMOS calibration is active at cycle start
    emos_check = validate_emos_active(
        SPREAD_INFLATION_FACTOR, MAX_BUCKET_PROBABILITY,
        EMOS_VARIANCE_FLOOR_C,
    )
    if not emos_check.valid:
        logger.warning(
            "CONTRACT VIOLATION [%s]: %s",
            emos_check.code, emos_check.error,
        )

    # Resolve any open paper trades before placing new ones
    if paper_trader:
        try:
            resolved_count = await resolve_open_trades(paper_trader)
            if resolved_count > 0:
                logger.info("Resolved %d paper trades at cycle start", resolved_count)
        except Exception:
            logger.exception("Paper trade resolution failed, continuing with cycle")

    _portfolio_summary = {}

    all_signals: list[Signal] = []
    _forecast_cache = forecast_cache if forecast_cache is not None else {}
    _ai_divergence_cache: dict[tuple, dict | None] = {}  # (city, variable) -> divergence result

    # Check if we're in a golden window (model just updated)
    if is_golden_window():
        logger.info("*** GOLDEN WINDOW: Fresh model data, market may be stale ***")

    # Refresh ENSO regime state (cached 24h, affects bias correction shrinkage)
    try:
        from weather_edge.analysis.enso_regime import fetch_enso_state
        await fetch_enso_state()
    except Exception:
        logger.debug("ENSO state fetch skipped", exc_info=True)

    # Step 1: Discover active weather markets (prices included from Gamma API)
    logger.info("=== Discovering Polymarket weather markets ===")
    markets = await discover_weather_markets()

    # Update market_map for position tracking (maps asset_id → city_id)
    if live_executor and markets:
        try:
            from weather_edge.trading.portfolio_sync import sync_market_map_from_discovery
            mapped = await sync_market_map_from_discovery(store, markets)
            if mapped:
                logger.info("Market map updated: %d token mappings", mapped)
        except Exception:
            logger.debug("Market map update failed", exc_info=True)

    # === PORTFOLIO SYNC: reconcile with exchange truth ===
    # Runs AFTER market_map so positions get city_id mapping
    if live_executor and not live_executor.dry_run:
        try:
            from weather_edge.trading.portfolio_sync import sync_portfolio
            _portfolio_summary = await sync_portfolio(
                executor=live_executor,
                store=store,
            )
            # Rebuild positions again to pick up market_map city_ids
            # (market_map was updated just before sync, but fills may have empty city_id)
            store.rebuild_positions()
            _portfolio_summary = store.get_portfolio_summary()
        except Exception:
            logger.exception("Portfolio sync failed, continuing with stale positions")

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

    # Step 1c: Fetch AI model forecasts (GraphCast via GribStream) for comparison
    ai_forecasts: dict[str, dict] = {}  # city_id -> AIModelForecast
    try:
        from weather_edge.fetchers.gribstream import (
            compute_ai_physics_divergence,
            fetch_ai_forecasts_batch,
        )
        market_cities = list({city_id for city_id, _ in market_groups})
        # Use earliest target date for GraphCast (not [1] which may be wrong after horizon filter)
        ai_target = target_dates[0] if target_dates else date.today() + timedelta(days=1)
        ai_batch = await fetch_ai_forecasts_batch(market_cities, ai_target)
        ai_forecasts = ai_batch
        if ai_batch:
            logger.info("GraphCast: %d city forecasts fetched", len(ai_batch))
    except Exception:
        logger.debug("GribStream AI fetch skipped", exc_info=True)

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
        if forecasts:
            _forecast_cache[(city_id, target_date)] = forecasts

            # Persist forecast snapshot for self-learning
            try:
                m_vals = {f.model_name: f.temp_max_c for f in forecasts if f.temp_max_c is not None}
                if m_vals:
                    store.save_forecast_snapshot(
                        city_id.value, str(target_date), m_vals,
                    )
            except Exception:
                pass  # Don't break pipeline if persistence fails
        else:
            # Check if we have stale data in cache
            cached = _forecast_cache.get((city_id, target_date))
            if cached:
                forecasts = cached
                logger.warning(
                    "STALE DATA: fetch failed for %s on %s, using cached data from %s",
                    city_id.value, target_date, cached[0].fetched_at.strftime("%H:%M:%S"),
                )

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

            # Track forecast trends (run-to-run consistency)
            trend_mult = 1.0
            try:
                from weather_edge.analysis.forecast_trends import compute_trend, record_forecast
                if variable == "temp_max_c":
                    record_forecast(city_id.value, consensus.weighted_mean)
                    trend = compute_trend(city_id.value, consensus.weighted_mean)
                    trend_mult = trend.confidence_multiplier
                    if trend.signal not in ("stable", "insufficient_data"):
                        logger.info(
                            "  %s TREND: %s (%.2f°C/cycle, stability=%.1f) → conf ×%.2f",
                            city_id.value, trend.signal, trend.trend_per_cycle,
                            trend.stability, trend_mult,
                        )
            except Exception:
                pass

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

                # Apply pattern-based confidence boost + forecast trend stability
                adjusted_conf = min(1.0, consensus.confidence * pattern_conf_mult * trend_mult)

                # Apply AI vs physics divergence (GraphCast comparison), computed once per city
                _ai_div_key = (city_id.value, variable)
                if _ai_div_key not in _ai_divergence_cache:
                    ai_fc = ai_forecasts.get(city_id.value)
                    if ai_fc and variable == "temp_max_c":
                        try:
                            div = compute_ai_physics_divergence(
                                ai_fc, consensus.weighted_mean,
                            )
                            _ai_divergence_cache[_ai_div_key] = div
                            if div["signal"] == "strong_diverge":
                                logger.info(
                                    "AI DIVERGE: %s GraphCast="
                                    "%.1fC vs physics=%.1fC "
                                    "(%+.1fC), conf ×%.2f",
                                    city_id.value,
                                    div["ai_max_c"],
                                    div["physics_mean_c"],
                                    div["divergence_c"],
                                    div["confidence_multiplier"],
                                )
                        except Exception:
                            _ai_divergence_cache[_ai_div_key] = None

                ai_div = _ai_divergence_cache.get(_ai_div_key)
                if ai_div:
                    adjusted_conf = min(1.0, adjusted_conf * ai_div["confidence_multiplier"])

                # Estimate spread from Gamma prices
                estimated_spread = max(0.0, 1.0 - (market.yes_price + market.no_price))

                signal = calculate_edge(
                    market_id=market.market_id,
                    model_prob=model_prob,
                    market_prob=market_prob,
                    model_confidence=adjusted_conf,
                    consensus_id=None,
                    hours_to_resolution=hours_to,
                    city_id=city_id.value,
                    description=market.question[:80],
                    spread=estimated_spread,
                )

                # Z-score guard: reject core bets on buckets too far from consensus
                # Tail/penny bets (entry <=6c) are exempt, they're designed as lottery tickets
                if signal.strategy != "tail" and market_prob > 0.06:
                    bucket_center = None
                    if market.threshold_dir == "range" and market.threshold_low_c is not None and market.threshold_high_c is not None:
                        bucket_center = (market.threshold_low_c + market.threshold_high_c) / 2
                    elif market.threshold_dir in ("gte", "lte") and market.threshold_value is not None:
                        bucket_center = market.threshold_value

                    if bucket_center is not None and consensus.std_dev > 0:
                        zscore = abs(bucket_center - consensus.weighted_mean) / consensus.std_dev
                        if zscore > settings.max_core_zscore:
                            logger.info(
                                "ZSCORE REJECT: %s %s, bucket=%.1f°C mean=%.1f°C std=%.1f z=%.1f (max=%.1f)",
                                city_id.value, market.question[:40],
                                bucket_center, consensus.weighted_mean,
                                consensus.std_dev, zscore, settings.max_core_zscore,
                            )
                            continue

                all_signals.append(signal)

    # === Claude + Gemini reasoning layer ===
    # Only on main cycles (not sniper-triggered) to save API costs
    if run_ai_reasoning and ANTHROPIC_API_KEY and all_signals:
        tradeable = sorted(
            [s for s in all_signals if s.confidence_tier.value != "low"],
            key=lambda s: abs(s.edge),
            reverse=True,
        )[:3]

        _golden = is_golden_window()  # used for future timing logic
        for signal in tradeable:
            # Build model context for Claude from cached forecasts
            model_vals = {}
            consensus_mean = signal.model_prob * 30  # Rough temp estimate from probability
            consensus_std = 2.0
            for (cid, td), f_list in _forecast_cache.items():
                if cid.value == signal.city_id:
                    model_vals = {
                        f.model_name: f.temp_max_c
                        for f in f_list
                        if f.temp_max_c is not None
                    }
                    if model_vals:
                        vals = list(model_vals.values())
                        consensus_mean = sum(vals) / len(vals)
                        consensus_std = (
                            (max(vals) - min(vals)) / 2
                            if len(vals) > 1 else 0.5
                        )
                    break

            reasoning = await analyze_trade(
                signal, model_vals, consensus_mean, consensus_std,
            )
            if reasoning:
                # Record every decision for the AI Decisions dashboard tab
                record_decision(reasoning)

                # Persist AI decision for self-learning
                try:
                    store.save_ai_decision(
                        source="claude",
                        decision="TRADE" if reasoning.should_trade else "SKIP",
                        city_id=signal.city_id,
                        market_id=signal.market_id,
                        rationale=reasoning.rationale[:500],
                        confidence_adj=reasoning.confidence_adjustment,
                    )
                except Exception:
                    pass

                if not reasoning.should_trade:
                    logger.info(
                        "CLAUDE SKIP: %s %s, %s",
                        signal.city_id,
                        signal.description[:40],
                        reasoning.rationale,
                    )
                    signal.confidence_tier = signal.confidence_tier  # Keep as-is but don't trade
                    continue
                # Apply Claude's confidence adjustment to position size
                signal.recommended_size = round(
                    signal.recommended_size
                    * reasoning.confidence_adjustment, 2,
                )

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
                            "city": (
                                signal.city_id.upper()
                                if isinstance(signal.city_id, str)
                                else signal.city_id
                            ),
                            "decision": "DISSENT" if verdict == "DISSENT" else "AGREE",
                            "signal": signal.description[:60],
                            "adjustment": round(1.0 - dissent, 2),
                            "rationale": "; ".join(gemini_result.get("counter_arguments", [])[:2]),
                            "risk_factors": [gemini_result.get("risk_the_bull_missed", "")],
                            "source": "gemini",
                        })
                        # Persist Gemini decision for self-learning
                        try:
                            store.save_ai_decision(
                                source="gemini",
                                decision=verdict,
                                city_id=signal.city_id,
                                market_id=signal.market_id,
                                rationale="; ".join(gemini_result.get("counter_arguments", [])[:2]),
                                dissent_strength=dissent,
                            )
                        except Exception:
                            pass
                        # Variable dissent sizing based on strength
                        sizing = gemini_result.get("sizing_recommendation", "full")
                        if dissent >= 0.7 or sizing in ("half", "skip"):
                            if sizing == "skip" and dissent >= 0.9:
                                multiplier = 0.0
                            elif sizing == "half" or dissent >= 0.7:
                                multiplier = 0.5
                            else:
                                multiplier = 1.0 - (dissent * 0.5)
                            old_size = signal.recommended_size
                            signal.recommended_size = round(
                                signal.recommended_size * multiplier, 2
                            )
                            logger.info(
                                "GEMINI DISSENT: %s, %.0f%% cut $%.0f->$%.0f (d=%.1f %s)",
                                signal.city_id, (1 - multiplier) * 100,
                                old_size, signal.recommended_size, dissent, sizing,
                            )
                        elif dissent >= 0.3 and sizing == "reduce_20pct":
                            old_size = signal.recommended_size
                            signal.recommended_size = round(
                                signal.recommended_size * 0.8, 2
                            )
                            logger.info(
                                "GEMINI MILD DISSENT: %s, 20%% trim $%.0f -> $%.0f (dissent=%.1f)",
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

    # Fetch live balance from exchange once before order loop
    _live_balance: float | None = None
    if live_executor and not live_executor.dry_run:
        try:
            _live_balance = await live_executor.check_balance()
            logger.info("USDC balance: $%.2f", _live_balance or 0)
        except Exception:
            logger.warning("Failed to fetch live balance, will skip balance checks")

    # --- SWING BOT: USDC floor ---
    # Don't place ANY new live orders if balance is below $20.
    # Prevents deploying dust into marginal trades. Capital comes back
    # via resolution/redemption, then we re-enter on high-conviction 48h+ signals.
    USDC_FLOOR = 20.0
    _usdc_floor_block = (
        live_executor
        and not live_executor.dry_run
        and _live_balance is not None
        and _live_balance < USDC_FLOOR
    )
    if _usdc_floor_block:
        logger.warning(
            "USDC FLOOR: $%.2f < $%.0f minimum, blocking all new live entries",
            _live_balance, USDC_FLOOR,
        )

    # --- SWING BOT: Position cap ---
    # Set to 50 to allow new entries alongside existing positions.
    # Most existing positions are small/penny bets that will resolve naturally.
    # Effective concentration is managed by $5 min size + USDC floor ($20).
    MAX_POSITIONS = 50
    _active_position_count = 0
    if store and live_executor and not live_executor.dry_run:
        try:
            # Clean resolved positions before counting. rebuild_positions()
            # re-creates them from fills every cycle, so we must clean every
            # time before reading the count.
            import httpx as _httpx
            _wallet = (live_executor.wallet_address or "").lower()
            async with _httpx.AsyncClient() as _hc:
                _pr = await _hc.get(
                    "https://data-api.polymarket.com/positions",
                    params={"user": _wallet, "sizeThreshold": 0},
                    timeout=15.0,
                )
                if _pr.status_code == 200:
                    _active_cids = {
                        p.get("conditionId")
                        for p in _pr.json()
                        if float(p.get("size", 0)) > 0
                    }
                    _db_pos = store.conn.execute(
                        "SELECT condition_id FROM positions WHERE total_shares > 0"
                    ).fetchall()
                    _cleaned = 0
                    for _row in _db_pos:
                        if _row["condition_id"] not in _active_cids:
                            store.conn.execute(
                                "UPDATE positions SET total_shares = 0 WHERE condition_id = ?",
                                (_row["condition_id"],),
                            )
                            _cleaned += 1
                    if _cleaned:
                        store.commit()

            _active_position_count = store.get_portfolio_summary().get("position_count", 0)
            if _active_position_count >= MAX_POSITIONS:
                logger.warning(
                    "POSITION CAP: %d/%d active positions, blocking new live entries",
                    _active_position_count, MAX_POSITIONS,
                )
        except Exception:
            _active_position_count = store.get_portfolio_summary().get("position_count", 0)

    for signal in all_signals:
        # Contract: verify taker fee doesn't eat >40% of projected alpha
        # This gate applies to TAKER orders only. Live executor uses post_only
        # (maker, $0 fee) so it bypasses this check.
        fee_check = validate_fee_alpha_ratio(
            edge=signal.edge,
            price=signal.market_prob,
            size_usd=signal.recommended_size,
        )
        fee_blocked = not fee_check.valid

        if fee_blocked and not live_executor:
            # Paper-only mode: skip trade entirely
            logger.info(
                "CONTRACT [%s]: %s, skipping %s %s",
                fee_check.code, fee_check.error, signal.city_id, signal.description[:40],
            )
            continue
        elif fee_blocked and live_executor:
            # Live mode: skip paper trade (taker fees eat alpha) but still
            # place live maker order ($0 fee). Log the fee gate for awareness.
            logger.info(
                "FEE GATE (paper only): %s %s, paper skipped, live maker order OK",
                signal.city_id, signal.description[:40],
            )

        trade = paper_trader.place_trade(signal) if (paper_trader and not fee_blocked) else None
        # Generate hedge/spread order only for core trades (not penny bets)
        # Penny bets at 0.1-5c: max loss is the entry cost, hedging is wasteful
        if trade:
            strategy = getattr(signal, "strategy", "core")
            if strategy != "tail":
                prices = market_prices.get(signal.market_id, {})
                if prices.get("spread_profitable"):
                    hedge = market_maker.generate_hedge_orders(
                        signal, market_prices, settings.bankroll,
                    )
                    if hedge and paper_trader:
                        paper_trader.place_spread_trade(signal, hedge)

        # === LIVE EXECUTION (maker orders bypass fee gate, $0 maker fee) ===
        # Still require minimum raw edge, we're bypassing fee check, not edge check
        if (
            live_executor
            and not live_executor.dry_run
            and not _usdc_floor_block
            and _active_position_count < MAX_POSITIONS
        ):
            # Cooldown: don't re-enter a market we recently fully exited
            # Sell-half excluded (still holding shares, POSITION EXISTS catches it)
            # Massive edge (>12%) bypasses cooldown for genuine model shifts
            if store:
                recent_exit = store.conn.execute(
                    """SELECT 1 FROM live_trades
                       WHERE market_id = ? AND side = 'SELL'
                       AND description NOT LIKE 'SELL_HALF%%'
                       AND datetime(placed_at) > datetime('now', '-4 hours')
                       LIMIT 1""",
                    (signal.market_id,),
                ).fetchone()
                if recent_exit:
                    if signal.edge < 0.12:
                        logger.info(
                            "EXIT COOLDOWN: %s, exited <4h ago, edge=%.1f%% (need >12%% to bypass)",
                            signal.city_id, signal.edge * 100,
                        )
                        continue
                    else:
                        logger.warning(
                            "COOLDOWN BYPASS: %s, massive edge %.1f%% overrules 4h window",
                            signal.city_id, signal.edge * 100,
                        )

            # --- SWING BOT: Minimum position size ---
            # Survival tier: $5 min until bankroll > $500, then raise to $10.
            MIN_LIVE_SIZE = 5.0
            if signal.recommended_size < MIN_LIVE_SIZE:
                signal.recommended_size = MIN_LIVE_SIZE

            # Margin check, use tracked live balance, not stale portfolio calc
            # Runs AFTER min size bump so $10 floor is respected
            if _live_balance is not None and signal.recommended_size > _live_balance:
                logger.info(
                    "BALANCE LIMIT: %s needs $%.0f but exchange balance is $%.2f, skipping",
                    signal.city_id, signal.recommended_size, _live_balance,
                )
                continue

            is_spread = getattr(signal, "strategy", "") == "spread"
            can_live = signal.edge >= 0.02 or is_spread
            if not can_live:
                logger.debug(
                    "LIVE SKIP: %s edge=%.3f size=$%.0f (need ≥2%% edge)",
                    signal.city_id, signal.edge, signal.recommended_size,
                )
            if can_live:

                m = market_by_id.get(signal.market_id)
                if m:
                    # Pick the right token
                    if signal.recommended_side.value == "YES":
                        token_id = m.token_id_yes
                    else:
                        token_id = m.token_id_no

                    if token_id:
                        # === POSITION-AWARE DUPLICATE PREVENTION ===
                        # Check POSITIONS (what we actually hold) not orders
                        existing_position = store.get_position_for_market(signal.market_id)
                        if existing_position and existing_position.get("total_shares", 0) > 0:
                            logger.info(
                                "POSITION EXISTS: %s already hold %.0f shares ($%.2f), skipping",
                                signal.city_id,
                                existing_position["total_shares"],
                                existing_position.get("cost_basis", 0),
                            )
                            continue

                        # Also check open orders (not yet filled)
                        existing = store.get_open_order_for_market(signal.market_id)
                        if existing:
                            old_price = existing.get("limit_price", 0)
                            filled = existing.get("filled_shares", 0) or 0
                            # Calculate what our new limit price would be
                            if signal.recommended_side.value == "YES":
                                new_price = round(
                                    max(0.01, min(0.99, signal.market_prob - 0.005)), 2,
                                )
                            else:
                                new_price = round(
                                    max(0.01, min(0.99, (1.0 - signal.market_prob) - 0.005)), 2,
                                )

                            # Check order age for price chase
                            order_age_minutes = 0
                            placed_str = existing.get("placed_at", "")
                            if placed_str:
                                try:
                                    placed_dt = datetime.fromisoformat(placed_str)
                                    order_age_minutes = (
                                        datetime.now(timezone.utc) - placed_dt
                                    ).total_seconds() / 60
                                except (ValueError, TypeError):
                                    pass

                            price_drift = abs(new_price - old_price)
                            should_chase = (
                                order_age_minutes > 60
                                and filled == 0
                                and signal.edge >= 0.02
                            )

                            if should_chase:
                                # Price chase: improve by 0.1c toward midpoint
                                chase_price = round(old_price + 0.001, 3)
                                chase_price = max(0.01, min(0.99, chase_price))
                                old_id = existing.get("order_id")
                                if old_id:
                                    try:
                                        await live_executor.cancel_order(old_id)
                                        store.cancel_live_trade(old_id)
                                        logger.info(
                                            "PRICE CHASE: %s improving "
                                            "%.3f→%.3f after %dm unfilled",
                                            signal.city_id, old_price,
                                            chase_price,
                                            int(order_age_minutes),
                                        )
                                    except Exception as e:
                                        logger.warning(
                                            "Price chase cancel failed %s: %s",
                                            old_id[:16], e,
                                        )
                                # Fall through to place new order at chased price
                                # Override the signal's market_prob to get the chased price
                                if signal.recommended_side.value == "YES":
                                    signal.market_prob = chase_price + 0.005
                                else:
                                    signal.market_prob = 1.0 - (chase_price + 0.005)

                            elif price_drift <= 0.001:
                                # Price unchanged, keep existing order, preserve queue priority
                                logger.info(
                                    "LIVE KEEP: %s @ %.3f "
                                    "(age=%dm, drift=%.4f, filled=%.0f)",
                                    signal.city_id, old_price,
                                    int(order_age_minutes),
                                    price_drift, filled,
                                )
                                continue
                            else:
                                # Price drifted, cancel old, place new
                                old_id = existing.get("order_id")
                                if old_id:
                                    try:
                                        await live_executor.cancel_order(old_id)
                                        store.cancel_live_trade(old_id)
                                        logger.info(
                                            "LIVE REPLACE: %s cancelled "
                                            "%s (price %.3f→%.3f, "
                                            "drift=%.3f)",
                                            signal.city_id, old_id[:16],
                                            old_price, new_price,
                                            price_drift,
                                        )
                                    except Exception as e:
                                        logger.warning(
                                            "Failed to cancel old order %s: %s",
                                            old_id[:16], e,
                                        )

                        try:
                            # --- SWING BOT: Hybrid entry ---
                            # Edge >= 12%: taker (cross spread, pay fee, secure alpha)
                            # Edge 5-12%: maker (rest on book, cancel if unfilled)
                            # At tail prices (<10c), taker fee is negligible anyway
                            use_taker = signal.edge >= 0.08
                            # Re-check fee gate for taker orders, maker is $0 fee
                            # but taker pays real fees that could eat alpha
                            # At $5 bets, taker fee is ~10c, skip fee gate, not worth the miss
                            if use_taker and fee_blocked and signal.recommended_size > 20:
                                logger.info(
                                    "FEE GATE TAKER: %s edge=%.1f%% but fee eats >40%% alpha on $%.0f, using maker",
                                    signal.city_id, signal.edge * 100, signal.recommended_size,
                                )
                                use_taker = False
                            if use_taker:
                                logger.info(
                                    "TAKER ENTRY: %s edge=%.1f%%, crossing spread",
                                    signal.city_id, signal.edge * 100,
                                )

                            result = await live_executor.place_limit_order(
                                signal, token_id, force_taker=use_taker,
                            )
                            if result and result.status not in (
                                "rejected", "post_only_reject", "too_small",
                            ):
                                # Track spending against live balance
                                if _live_balance is not None:
                                    _live_balance -= result.size_usd
                                # Track position count for cap enforcement
                                _active_position_count += 1

                                logger.info(
                                    "LIVE: %s %s %s %s %.0f shares @ %.3f, %s (bal=$%.2f)",
                                    "TAKER" if use_taker else "MAKER",
                                    result.status, signal.recommended_side.value,
                                    signal.city_id, result.size_shares,
                                    result.limit_price, result.order_id,
                                    _live_balance or 0,
                                )

                                # --- LIVE SPREAD CAPTURE HEDGE ---
                                if market_maker and not live_executor.dry_run:
                                    # Use settings.bankroll as baseline for pool-sizing
                                    hedge = market_maker.generate_hedge_orders(
                                        signal, market_prices, settings.bankroll, is_live=True
                                    )
                                    if hedge:
                                        m = market_by_id.get(hedge.market_id)
                                        if m:
                                            from weather_edge.models.enums import MarketType
                                            h_side = MarketType.YES if hedge.side == "YES" else MarketType.NO
                                            h_token_id = m.token_id_yes if hedge.side == "YES" else m.token_id_no
                                            
                                            # Create synthetic signal for the hedge order
                                            h_signal = Signal(
                                                market_id=hedge.market_id,
                                                city_id=signal.city_id,
                                                recommended_side=h_side,
                                                recommended_size=hedge.cost,
                                                edge=0.0, # Spread orders have guaranteed profit, not directional edge
                                                confidence=1.0,
                                                description=f"[SPREAD] {hedge.description}",
                                                strategy="spread",
                                                market_prob=hedge.limit_price,
                                            )
                                            
                                            try:
                                                import requests
                                                h_result = await live_executor.place_limit_order(h_signal, h_token_id)
                                                if h_result:
                                                    logger.info(
                                                        "LIVE SPREAD HEDGE: %s %s %.0f shares @ %.3f, %s",
                                                        h_side.value, signal.city_id,
                                                        h_result.size_shares, h_result.limit_price,
                                                        h_result.order_id,
                                                    )
                                            except (requests.RequestException, ValueError, KeyError) as e:
                                                logger.error("LIVE SPREAD HEDGE FAILED: %s, %s", signal.city_id, e)

                        except Exception as e:
                            logger.error(
                                "LIVE ORDER FAILED: %s, %s", signal.city_id, e,
                            )

    # Log spread capture summary
    spread_summary = market_maker.simulate_spread_pnl()
    if spread_summary["spread_orders"] > 0:
        logger.info(
            "SPREAD CAPTURE: %d orders, est. guaranteed P&L=$%.2f",
            spread_summary["spread_orders"], spread_summary["estimated_guaranteed_pnl"],
        )

    # === Early exit monitor ===
    # Paper and live scan independently, either can run alone
    try:
        from weather_edge.analysis.exit_monitor import ai_review_exit, scan_for_exits
        current_market_prices = {m.market_id: m.yes_price for m in markets}
        current_model_probs = {}
        for signal in all_signals:
            current_model_probs[signal.market_id] = signal.model_prob

        # --- PAPER EXIT SCANNING ---
        if paper_trader:
            paper_candidates = scan_for_exits(
                paper_trader.open_trades,
                current_market_prices, current_model_probs,
                forecast_cache=_forecast_cache,
            )
            if paper_candidates:
                logger.info("PAPER EXIT MONITOR: %d candidates", len(paper_candidates))
                for candidate in paper_candidates[:3]:
                    model_vals = {}
                    c_mean, c_std = 0.0, 1.0
                    for (cid, td), f_list in _forecast_cache.items():
                        if cid.value == candidate.trade.city_id:
                            model_vals = {
                                f.model_name: f.temp_max_c
                                for f in f_list
                                if f.temp_max_c is not None
                            }
                            if model_vals:
                                vals = list(model_vals.values())
                                c_mean = sum(vals) / len(vals)
                                c_std = (
                                    (max(vals) - min(vals)) / 2
                                    if len(vals) > 1 else 0.5
                                )
                            break
                    candidate = await ai_review_exit(
                        candidate, model_vals, c_mean, c_std,
                    )
                    if candidate.final_decision == "EXIT":
                        logger.warning(
                            "PAPER EXIT: %s %s $%.0f, %s",
                            candidate.trade.side,
                            candidate.trade.city_id,
                            candidate.trade.size_usd,
                            candidate.reason,
                        )
                        paper_trader.close_position(
                            candidate.trade,
                            candidate.current_market_price,
                        )

        # --- LIVE EXIT SCANNING (from positions table, independent of paper) ---
        if live_executor and not live_executor.dry_run and store:
            from weather_edge.models.position import Position
            positions = store.get_positions()
            live_positions: list[Position] = []
            for pos in positions:
                shares = pos.get("total_shares", 0)
                avg_price = pos.get("avg_price", 0)
                if shares >= MIN_SELL_SHARES:
                    live_positions.append(Position(
                        market_id=pos.get("condition_id", ""),
                        city_id=pos.get("city_id", ""),
                        side=pos.get("outcome", pos.get("side", "")),
                        size_usd=pos.get("cost_basis", 0),
                        entry_price=avg_price,
                        total_shares=shares,
                        description=pos.get("description", ""),
                        source="live",
                    ))

            if live_positions:
                live_candidates = scan_for_exits(
                    live_positions,
                    current_market_prices, current_model_probs,
                    forecast_cache=_forecast_cache,
                )
                if live_candidates:
                    logger.info("LIVE EXIT MONITOR: %d candidates", len(live_candidates))
                    for candidate in live_candidates[:3]:
                        model_vals = {}
                        c_mean, c_std = 0.0, 1.0
                        for (cid, td), f_list in _forecast_cache.items():
                            if cid.value == candidate.trade.city_id:
                                model_vals = {
                                    f.model_name: f.temp_max_c
                                    for f in f_list
                                    if f.temp_max_c is not None
                                }
                                if model_vals:
                                    vals = list(model_vals.values())
                                    c_mean = sum(vals) / len(vals)
                                    c_std = (
                                        (max(vals) - min(vals)) / 2
                                        if len(vals) > 1 else 0.5
                                    )
                                break
                        candidate = await ai_review_exit(
                            candidate, model_vals, c_mean, c_std,
                        )
                        if candidate.final_decision == "EXIT":
                            city_id_str = candidate.trade.city_id
                            market_id = candidate.trade.market_id

                            # Sell-half guard: skip if already trimmed or has pending sell
                            if candidate.reason == "sell_half":
                                existing = store.get_open_order_for_market(market_id, side="SELL")
                                if existing:
                                    logger.info("SELL_HALF SKIP: %s has open sell order", city_id_str)
                                    continue
                                past_trim = store.conn.execute(
                                    "SELECT 1 FROM live_trades WHERE market_id = ? AND description LIKE 'SELL_HALF%' LIMIT 1",
                                    (market_id,),
                                ).fetchone()
                                if past_trim:
                                    logger.info("SELL_HALF SKIP: %s already trimmed", city_id_str)
                                    continue

                            # 1. Cancel any open BUY orders for this market if we are exiting
                            open_buy = store.get_open_order_for_market(market_id)
                            if open_buy and open_buy.get("side") in ("YES", "NO"):
                                buy_id = open_buy["order_id"]
                                try:
                                    await live_executor.cancel_order(buy_id)
                                    store.cancel_live_trade(buy_id)
                                    logger.info("EXIT: cancelled open BUY order %s for %s", buy_id[:16], city_id_str)
                                except Exception as e:
                                    logger.warning("Failed to cancel BUY order %s: %s", buy_id[:16], e)

                            # 2. Check for existing open SELL order
                            existing_sell = store.get_open_order_for_market(market_id, side="SELL")
                            
                            position = store.get_position_for_market(market_id)
                            if position and position.get("total_shares", 0) >= MIN_SELL_SHARES:
                                asset_id = position.get("asset_id", "")
                                if asset_id:
                                    # If we already have a sell order, check if it's still at the right price
                                    if existing_sell:
                                        old_price = existing_sell.get("limit_price", 0)
                                        new_price = _round_price(candidate.current_market_price)
                                        price_drift = abs(old_price - new_price)
                                        
                                        # Only replace if price drifted by > 1¢
                                        if price_drift <= 0.01:
                                            logger.info(
                                                "LIVE SELL KEEP: %s @ %.3f (drift=%.3f)",
                                                city_id_str, old_price, price_drift
                                            )
                                            continue
                                        else:
                                            # Cancel old, replace with new price
                                            old_id = existing_sell["order_id"]
                                            try:
                                                await live_executor.cancel_order(old_id)
                                                store.cancel_live_trade(old_id)
                                                logger.info(
                                                    "LIVE SELL REPLACE: %s cancelled %s (%.3f->%.3f)",
                                                    city_id_str, old_id[:16], old_price, new_price
                                                )
                                            except Exception as e:
                                                logger.warning("Failed to cancel old sell order %s: %s", old_id[:16], e)

                                    try:
                                        # High urgency: use taker mode to guarantee fill
                                        # Low/medium: use maker (post_only) to save fees
                                        urgent = candidate.urgency == "high"
                                        # Sell half for sell_half reason, full for everything else
                                        sell_shares = position["total_shares"]
                                        if candidate.reason == "sell_half":
                                            sell_shares = max(MIN_SELL_SHARES, round(sell_shares / 2, 0))
                                        sell_result = await live_executor.place_sell_order(
                                            token_id=asset_id,
                                            shares=sell_shares,
                                            price=candidate.current_market_price,
                                            market_id=market_id,
                                            city_id=city_id_str,
                                            description=f"{'SELL_HALF' if candidate.reason == 'sell_half' else 'EXIT'}: {candidate.reason}",
                                            reference_price=candidate.current_market_price,
                                            force_taker=urgent,
                                        )
                                        if sell_result:
                                            logger.warning(
                                                "LIVE EXIT: %s %s %.0f shares @ %.3f, %s",
                                                city_id_str, candidate.trade.side,
                                                position["total_shares"],
                                                candidate.current_market_price,
                                                sell_result.order_id,
                                            )
                                    except Exception as e:
                                        logger.error("LIVE EXIT FAILED: %s, %s", city_id_str, e)

                        # Record exit decision to AI Decisions tab
                        from weather_edge.analysis.claude_reasoning import _decision_history
                        _decision_history.insert(0, {
                            "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                            "city": candidate.trade.city_id.upper(),
                            "decision": "EXIT" if candidate.final_decision == "EXIT" else "HOLD",
                            "signal": (
                                f"[EXIT CHECK] {candidate.reason}: "
                                f"{candidate.trade.description[:40]}"
                            ),
                            "adjustment": round(candidate.current_edge, 2),
                            "rationale": candidate.claude_rationale or "No AI review",
                            "risk_factors": [candidate.gemini_rationale or ""],
                            "source": "exit_monitor",
                        })
    except Exception:
        logger.error("EXIT MONITOR CRASHED, check traceback", exc_info=True)

    # Also fetch forecasts for cities without active markets (monitoring)
    # But only for tomorrow (not all dates) to save API calls
    tomorrow = target_dates[1] if len(target_dates) > 1 else (target_dates[0] if target_dates else date.today() + timedelta(days=1))
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


async def run_loop(paper_trader: PaperTrader | None) -> None:
    """Run the fetch-analyze-trade loop continuously."""
    interval = settings.fetch_interval_minutes * 60
    cycle_num = 0
    forecast_cache: dict[tuple, list] = {}

    while True:
        cycle_num += 1
        logger.info("===== CYCLE %d START =====", cycle_num)
        try:
            # Pass existing cache to run_cycle
            signals, forecast_cache, _ = await run_cycle(
                paper_trader,
                forecast_cache=forecast_cache,
            )
            tradeable = [s for s in signals if s.confidence_tier.value != "low"]
            pnl = paper_trader.total_pnl if paper_trader else 0
            logger.info(
                "Cycle %d complete: %d signals, %d tradeable, P&L=$%.2f",
                cycle_num, len(signals), len(tradeable), pnl,
            )
        except Exception:
            logger.exception("Cycle %d failed", cycle_num)

        logger.info("Sleeping %d minutes until next cycle...", settings.fetch_interval_minutes)
        await asyncio.sleep(interval)
