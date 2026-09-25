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
from weather_edge.fetchers.openmeteo import f_to_c, fetch_city_forecasts
from weather_edge.fetchers.polymarket import (
    MarketInfo,
    discover_weather_markets,
)
from weather_edge.models.enums import City
from weather_edge.trading.paper import PaperTrader

logger = logging.getLogger(__name__)

MIN_SELL_SHARES = 5.0  # Polymarket minimum order size
MIN_LIVE_SIZE = 5.0  # Survival tier: $5 min until bankroll > $500, then $10
HAILMARY_TICKET_USD = 1.00  # Fixed lottery-ticket size in hail-mary mode


def _round_price(price: float) -> float:
    """Round price to valid Polymarket tick size (1 cent)."""
    return round(max(0.01, min(0.99, price)), 2)


def _bucket_celsius_band(market: MarketInfo) -> tuple[float | None, float | None] | None:
    """Half-open °C interval [lo_c, hi_c) covered by a market's temperature bucket.

    Uses round-half-up display semantics: an integer label ``L`` in the market's
    native (displayed) unit covers ``[L-0.5, L+0.5)``. Open-ended buckets return
    ``None`` for the corresponding edge. Returns ``None`` when the market carries
    no native integer labels (e.g. snow / "any" markets), so callers fall back to
    the legacy °C-bound path.

    Hong Kong is the exception: the HK Observatory value is a 0.1°C decimal that
    is not rounded, and label ``L`` covers ``[L, L+1)`` (the decimals whose
    integer part is L). This must match resolver.actual_falls_in_bucket's HKG
    branch, which floors the reading to its label.
    """
    lo = market.bucket_low_int
    hi = market.bucket_high_int
    if lo is None and hi is None:
        return None
    if market.city_id == City.HKG and market.threshold_unit == "celsius":
        return (
            float(lo) if lo is not None else None,
            float(hi) + 1.0 if hi is not None else None,
        )
    conv = (lambda x: x) if market.threshold_unit == "celsius" else f_to_c
    lo_c = conv(lo - 0.5) if lo is not None else None
    hi_c = conv(hi + 0.5) if hi is not None else None
    return lo_c, hi_c


# Model agreement gate: skip a city/variable when the models themselves are
# more than this far apart (sample std of the bias-corrected model values, °C).
MODEL_AGREEMENT_MAX_STD_C = 2.0


def _model_agreement_std(consensus) -> float:
    """Spread the agreement gate compares against MODEL_AGREEMENT_MAX_STD_C.

    This is the RAW model spread (how far apart the models actually are).
    consensus.std_dev is the EMOS-inflated probability std (raw x
    SPREAD_INFLATION_FACTOR, then floored); comparing 2.0 against that silently
    tightened the gate to ~1.54°C of real disagreement. Consensus objects built
    without raw_std_dev fall back to un-inflating std_dev.
    """
    raw = getattr(consensus, "raw_std_dev", None)
    if raw is not None:
        return raw
    return consensus.std_dev / SPREAD_INFLATION_FACTOR


def compute_model_prob_for_market(market: MarketInfo, consensus) -> float | None:
    """Compute model probability for a market bucket.

    Handles the multi-bucket format with EMOS probability cap:
    - A single 2°F bucket should never exceed 70% at >12h horizon
    - Per Gemini: >90% on a single bucket is "likely broken"
    """
    from weather_edge.analysis.consensus import MAX_BUCKET_PROBABILITY

    prob = None

    # Round-half-up display semantics: a market resolves YES when the displayed
    # (rounded) daily high lands on one of the bucket's integer labels. An
    # integer label L therefore covers the continuous half-open interval
    # [L-0.5, L+0.5) in the displayed unit. We integrate the model distribution
    # over exactly that interval (converted to °C) so probability and resolution
    # use the SAME boundary convention. The previous code added a flat 1.0 to a
    # °C-converted bound for Fahrenheit buckets, which mixed units and made every
    # F range bucket ~0.8°F too wide on the top edge (inflating YES probability).
    band = _bucket_celsius_band(market)

    if market.threshold_dir == "lte":
        if band is not None and band[1] is not None:
            p_gte_hi = get_probability_for_threshold(consensus, band[1], "gte")
            prob = 1.0 - p_gte_hi
        else:
            p_gte = get_probability_for_threshold(
                consensus,
                market.threshold_high_c or market.threshold_value,
                "gte",
            )
            prob = 1.0 - p_gte

    elif market.threshold_dir == "range":
        if band is not None:
            lo_c, hi_c = band
            p_lo = (
                get_probability_for_threshold(consensus, lo_c, "gte")
                if lo_c is not None else 1.0
            )
            p_hi = (
                get_probability_for_threshold(consensus, hi_c, "gte")
                if hi_c is not None else 0.0
            )
            prob = max(0.0, p_lo - p_hi)
        elif (
            market.threshold_low_c is not None
            and market.threshold_high_c is not None
        ):
            # Legacy fallback when native integer labels are unavailable.
            p_gte_low = get_probability_for_threshold(
                consensus, market.threshold_low_c, "gte",
            )
            p_gte_high = get_probability_for_threshold(
                consensus, market.threshold_high_c, "gte",
            )
            prob = max(0.0, p_gte_low - p_gte_high)

    elif market.threshold_dir == "gte":
        if band is not None and band[0] is not None:
            prob = get_probability_for_threshold(consensus, band[0], "gte")
        else:
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


# Swing-bot horizon: block entries on markets resolving sooner than this.
MIN_HORIZON_HOURS = 36
# STALE DATA fallback: never reuse a cached forecast older than this.
DEFAULT_MAX_STALE_FORECAST_HOURS = 6.0


def filter_dates_by_horizon(
    target_dates: list[date],
    now_utc: datetime | None = None,
    min_horizon_hours: int | None = None,
) -> tuple[list[date], list[date], int]:
    """Split target dates into (kept, blocked, min_horizon_hours).

    A market for date ``d`` resolves at 00:00 UTC on ``d + 1``; it is kept
    only if that is at least ``min_horizon_hours`` away. HAIL MARY mode uses a
    0h horizon.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    if min_horizon_hours is None:
        min_horizon_hours = 0 if settings.hail_mary_mode else MIN_HORIZON_HOURS
    min_horizon = timedelta(hours=min_horizon_hours)
    kept, blocked = [], []
    for d in target_dates:
        resolves = datetime.combine(d + timedelta(days=1), datetime.min.time()).replace(
            tzinfo=timezone.utc,
        )
        (kept if resolves - now_utc >= min_horizon else blocked).append(d)
    return kept, blocked, min_horizon_hours


def default_target_dates(today: date | None = None, days: int = 4) -> list[date]:
    """``today`` .. ``today + days - 1`` (the scheduler's default window)."""
    today = today or date.today()
    return [today + timedelta(days=i) for i in range(max(1, days))]


def evict_stale_forecasts(forecast_cache: dict[tuple, list], today: date) -> int:
    """Remove cache entries for target dates before ``today``. Returns count."""
    stale = [
        k for k in forecast_cache
        if isinstance(k, tuple) and len(k) == 2 and isinstance(k[1], date) and k[1] < today
    ]
    for k in stale:
        del forecast_cache[k]
    return len(stale)


def _forecast_age_hours(forecasts: list, now_utc: datetime) -> float | None:
    times = [getattr(f, "fetched_at", None) for f in forecasts]
    times = [t for t in times if isinstance(t, datetime)]
    if not times:
        return None
    oldest = min(t if t.tzinfo else t.replace(tzinfo=timezone.utc) for t in times)
    return (now_utc - oldest).total_seconds() / 3600


def model_context_for_signal(
    signal: Signal, forecast_cache: dict[tuple, list],
) -> tuple[dict[str, float], float, float]:
    """(model_values, mean, std) for the signal's OWN city and target date.

    Never falls back to another date's forecasts: an empty dict means "no
    context", and the AI review treats that as no review.
    """
    try:
        key = (City(signal.city_id), date.fromisoformat(str(signal.target_date)))
    except ValueError:
        return {}, 0.0, 0.0
    f_list = forecast_cache.get(key) or []
    model_vals = {f.model_name: f.temp_max_c for f in f_list if f.temp_max_c is not None}
    if not model_vals:
        return {}, 0.0, 0.0
    vals = list(model_vals.values())
    mean = sum(vals) / len(vals)
    std = (max(vals) - min(vals)) / 2 if len(vals) > 1 else 0.5
    return model_vals, mean, std


def _gemini_size_multiplier(dissent: float, sizing: str) -> float:
    """Position-size multiplier from a Gemini red-team verdict (1.0 = no cut)."""
    if dissent >= 0.7 or sizing in ("half", "skip"):
        if sizing == "skip" and dissent >= 0.9:
            return 0.0
        if sizing == "half" or dissent >= 0.7:
            return 0.5
        return 1.0 - (dissent * 0.5)
    if dissent >= 0.3 and sizing == "reduce_20pct":
        return 0.8
    return 1.0


async def _review_signal(signal: Signal, forecast_cache: dict[tuple, list], store) -> str:
    """Claude + Gemini review of one signal: 'approved', 'vetoed' or 'failed'.

    Outside hail-mary, applies the AI sizing to ``signal.recommended_size`` and
    records approvals/vetoes in the shared review memory. In hail-mary mode the
    verdicts are logged only (no cut, no block, nothing remembered).
    """
    from weather_edge.analysis import gemini_reasoning
    from weather_edge.analysis.claude_reasoning import _decision_history, get_review_memory

    hail = settings.hail_mary_mode
    memory = get_review_memory()

    model_vals, consensus_mean, consensus_std = model_context_for_signal(signal, forecast_cache)
    reasoning = None
    if model_vals:
        reasoning = await analyze_trade(signal, model_vals, consensus_mean, consensus_std)
    if reasoning is None:
        logger.warning(
            "CLAUDE REVIEW UNAVAILABLE: %s %s %s (%s)",
            signal.city_id, signal.target_date, signal.description[:40],
            "no forecasts for this date" if not model_vals else "API call failed",
        )
        return "failed"

    record_decision(reasoning)
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

    if not reasoning.review_ok and not hail:
        return "failed"

    size_mult = 1.0
    if not reasoning.should_trade:
        if hail:
            # HAIL MARY: Claude SKIP is logged but does not block
            logger.info(
                "HAILMARY override CLAUDE SKIP: %s %s, %s",
                signal.city_id, signal.description[:40], reasoning.rationale,
            )
        else:
            logger.info(
                "CLAUDE SKIP: %s %s, %s",
                signal.city_id, signal.description[:40], reasoning.rationale,
            )
            memory.record_veto(signal, reasoning.rationale, "claude")
            return "vetoed"
    elif not hail:
        # Apply Claude's confidence adjustment to position size.
        # HAIL MARY ignores it, sizing is fixed downstream.
        size_mult = reasoning.confidence_adjustment
        signal.recommended_size = round(signal.recommended_size * size_mult, 2)

    # === Gemini red team on Claude-approved trades ===
    # None = Gemini not configured (Claude-only review). A configured Gemini
    # that errors returns a fail-closed result carrying "error".
    try:
        gemini_result = await gemini_reasoning.red_team_trade(
            signal, model_vals, consensus_mean, consensus_std,
            claude_rationale=reasoning.rationale,
        )
    except Exception as e:
        logger.warning("Gemini red team crashed for %s: %s", signal.city_id, e)
        gemini_result = {"error": str(e), "dissent_strength": 1.0,
                         "verdict": "DISSENT", "sizing_recommendation": "skip"}

    if gemini_result:
        dissent = float(gemini_result.get("dissent_strength", 1.0))
        verdict = gemini_result.get("verdict", "DISSENT")
        sizing = gemini_result.get("sizing_recommendation", "skip")
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

        if hail:
            if gemini_result.get("error") or dissent >= 0.3:
                logger.info(
                    "HAILMARY override GEMINI DISSENT: %s d=%.1f %s (no cut applied)",
                    signal.city_id, dissent, sizing,
                )
        elif gemini_result.get("error"):
            logger.warning(
                "GEMINI REVIEW FAILED: %s, not executing this cycle (%s)",
                signal.city_id, gemini_result["error"],
            )
            return "failed"
        else:
            multiplier = _gemini_size_multiplier(dissent, sizing)
            if multiplier <= 0:
                logger.info(
                    "GEMINI VETO: %s, sizing=%s dissent=%.1f, dropping signal",
                    signal.city_id, sizing, dissent,
                )
                memory.record_veto(signal, f"gemini {sizing} d={dissent:.1f}", "gemini")
                return "vetoed"
            if multiplier < 1.0:
                old_size = signal.recommended_size
                signal.recommended_size = round(signal.recommended_size * multiplier, 2)
                logger.info(
                    "GEMINI DISSENT: %s, %.0f%% cut $%.0f->$%.0f (d=%.1f %s)",
                    signal.city_id, (1 - multiplier) * 100,
                    old_size, signal.recommended_size, dissent, sizing,
                )
            size_mult *= multiplier

    if not hail:
        memory.record_approval(signal, size_mult, reasoning.rationale)
    return "approved"


async def apply_ai_review(
    signals: list[Signal],
    forecast_cache: dict[tuple, list],
    run_ai_reasoning: bool,
    store=None,
) -> list[Signal]:
    """Gate ``signals`` through the Claude + Gemini review; return the executable ones.

    Outside HAIL MARY mode a signal is executed only if:
    - it has no veto recorded today for its (market, target date), and
    - it was approved in this cycle, or approved earlier (within
      ``claude_reasoning.APPROVAL_TTL_SEC``) on the same side, in which case
      the remembered AI size multiplier is re-applied, and
    - its size after AI adjustments is > 0.

    A failed review (API error, unparseable/truncated reply, no forecast
    context) blocks the signal for this cycle but is not remembered.

    Signals with no review at all (sniper / cooldown-refresh cycles with
    ``run_ai_reasoning=False``, no ANTHROPIC_API_KEY, or beyond the
    ``max_ai_reviews_per_cycle`` budget) are NOT executed unless
    ``settings.require_ai_review`` is False. Explicit vetoes block regardless.

    HAIL MARY mode keeps its log-only behaviour: every signal is returned.
    """
    from weather_edge.analysis.claude_reasoning import get_review_memory

    if not signals:
        return []

    hail = settings.hail_mary_mode
    require_review = bool(getattr(settings, "require_ai_review", True))
    max_reviews = int(getattr(settings, "max_ai_reviews_per_cycle", 3))
    memory = get_review_memory()
    memory.evict_before(date.today())

    outcome: dict[int, str] = {}
    if not hail:
        for s in signals:
            veto = memory.veto_for(s)
            if veto:
                outcome[id(s)] = "vetoed"
                logger.info(
                    "AI VETO (remembered, %s): %s %s %s, %s",
                    veto.source, s.city_id, s.target_date, s.description[:40], veto.reason,
                )

    ai_available = bool(run_ai_reasoning and ANTHROPIC_API_KEY)
    if ai_available:
        candidates = [
            s for s in signals
            if s.confidence_tier.value != "low" and id(s) not in outcome
        ]
        # Unreviewed signals first so the budget eventually covers all of them.
        candidates.sort(key=lambda s: (memory.approval_for(s) is not None, -abs(s.edge)))
        for s in candidates[:max_reviews]:
            outcome[id(s)] = await _review_signal(s, forecast_cache, store)

    if hail:
        return list(signals)

    if not run_ai_reasoning:
        why = "AI skipped on this cycle (sniper / refresh cooldown)"
    elif not ANTHROPIC_API_KEY:
        why = "no ANTHROPIC_API_KEY"
    else:
        why = f"outside the {max_reviews}-review budget"

    executable: list[Signal] = []
    for s in signals:
        state = outcome.get(id(s))
        if state == "vetoed":
            continue
        if state == "approved":
            if s.recommended_size > 0:
                executable.append(s)
            continue
        if state is None:
            approval = memory.approval_for(s)
            if approval:
                s.recommended_size = round(s.recommended_size * approval.size_multiplier, 2)
                if s.recommended_size > 0:
                    logger.info(
                        "AI APPROVAL (remembered): %s %s %s, size x%.2f",
                        s.city_id, s.target_date, s.description[:40],
                        approval.size_multiplier,
                    )
                    executable.append(s)
                continue
        # Failed review, or never reviewed
        if not require_review:
            logger.warning(
                "UNREVIEWED TRADE (require_ai_review=False): %s %s %s",
                s.city_id, s.target_date, s.description[:40],
            )
            executable.append(s)
        else:
            logger.info(
                "NO AI APPROVAL: %s %s %s not executed (%s)",
                s.city_id, s.target_date, s.description[:40],
                "review failed" if state == "failed" else why,
            )

    logger.info(
        "AI GATE: %d/%d signals cleared for execution", len(executable), len(signals),
    )
    return executable


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
        (signals, forecast_cache, city_volume) where forecast_cache maps
        (city_id, date) -> forecasts. ``signals`` is every candidate after
        filtering (for display / exit monitoring); only the AI-cleared subset
        is executed.
    """
    # Store can come from paper_trader or be passed directly. A plain
    # in-memory PaperTrader (CLI) has no store; every store use tolerates None.
    if store is None and paper_trader is not None:
        store = getattr(paper_trader, "store", None)

    if target_dates is None:
        target_dates = default_target_dates()

    # --- SWING BOT: 36h horizon filter ---
    # Block entries on markets resolving too soon. We get front-run by bots
    # with fresher NWS data on short-dated markets. Our bias correction edge
    # is strongest at 48-72h where model ensembles still disagree.
    # HAIL MARY mode drops the filter: penny lottery tickets only need to
    # resolve, they don't need to be ahead of the front-running bots.
    now_utc = datetime.now(timezone.utc)
    target_dates, blocked, min_horizon_hours = filter_dates_by_horizon(
        list(target_dates), now_utc,
    )
    if blocked:
        logger.info(
            "HORIZON FILTER: blocked %s (resolve < %dh)",
            ", ".join(str(d) for d in sorted(blocked)),
            min_horizon_hours,
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
    # Cross-cycle cache: forget target dates that have already passed.
    evict_stale_forecasts(_forecast_cache, date.today())
    _refreshed_this_cycle: set[tuple] = set()  # keys fetched fresh in this cycle
    # (city, target_date, variable) -> divergence result
    _ai_divergence_cache: dict[tuple, dict | None] = {}

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
    total_equity = settings.bankroll
    if live_executor and not live_executor.dry_run:
        try:
            from weather_edge.trading.portfolio_sync import fetch_polymarket_state, sync_portfolio
            _portfolio_summary = await sync_portfolio(
                executor=live_executor,
                store=store,
            )
            # Fetch full state to get balance + current market value of positions
            state = await fetch_polymarket_state(live_executor, live_executor.wallet_address)
            total_equity = state["portfolio_value"]
            logger.info("PORTFOLIO EQUITY: $%.2f (used as bankroll for Kelly sizing)", total_equity)

            # Rebuild positions again to pick up market_map city_ids
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
    # Keyed (city_id, target_date): a GraphCast run for one date must never be
    # compared against the physics consensus for another.
    ai_forecasts: dict[tuple[str, date], object] = {}
    try:
        from weather_edge.fetchers.gribstream import (
            compute_ai_physics_divergence,
            fetch_ai_forecasts_batch,
        )
        for ai_target in sorted({d for _, d in market_groups}):
            market_cities = sorted(
                {c for c, d in market_groups if d == ai_target}, key=lambda c: c.value,
            )
            try:
                ai_batch = await fetch_ai_forecasts_batch(market_cities, ai_target)
            except Exception:
                logger.debug("GribStream AI fetch failed for %s", ai_target, exc_info=True)
                continue
            for cid, fc in (ai_batch or {}).items():
                ai_forecasts[(getattr(cid, "value", cid), ai_target)] = fc
            if ai_batch:
                logger.info(
                    "GraphCast: %d city forecasts fetched for %s", len(ai_batch), ai_target,
                )
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
            _refreshed_this_cycle.add((city_id, target_date))

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
            max_stale_h = float(getattr(
                settings, "max_stale_forecast_hours", DEFAULT_MAX_STALE_FORECAST_HOURS,
            ))
            age_h = _forecast_age_hours(cached, now_utc) if cached else None
            if cached and age_h is not None and age_h <= max_stale_h:
                forecasts = cached
                logger.warning(
                    "STALE DATA: fetch failed for %s on %s, using cached data from %s "
                    "(%.1fh old)",
                    city_id.value, target_date,
                    cached[0].fetched_at.strftime("%H:%M:%S"), age_h,
                )
            elif cached:
                logger.warning(
                    "STALE DATA REJECTED: cached forecast for %s on %s is too old "
                    "(%s h > %.1f h), skipping",
                    city_id.value, target_date,
                    f"{age_h:.1f}" if age_h is not None else "unknown", max_stale_h,
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
                    # Pre-compute on the EXACT band edges that
                    # compute_model_prob_for_market will look up, so the
                    # calibrated blend is served from threshold_probs instead
                    # of missing every key (get_probability_for_threshold also
                    # computes the blend on demand for any other edge).
                    band = _bucket_celsius_band(m)
                    if band is not None:
                        thresholds.extend(e for e in band if e is not None)
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

            # --- MODEL AGREEMENT GATE ---
            # Per README: Skip if models disagree (std > 2.0C). This prevents
            # trading on noise when ensembles are in chaos. HAIL MARY mode lets
            # every signal through: penny tickets are cheap enough that noisy
            # bets beat missed ones.
            # Applied to the RAW model spread, see _model_agreement_std.
            agreement_std = _model_agreement_std(consensus)
            if agreement_std > MODEL_AGREEMENT_MAX_STD_C:
                if settings.hail_mary_mode:
                    logger.info(
                        "HAILMARY allow: %s/%s raw std=%.1f > 2.0 (would have skipped)",
                        city_id.value, variable, agreement_std
                    )
                else:
                    logger.warning(
                        "MODEL AGREEMENT REJECT: %s/%s raw std=%.1f > 2.0, skipping",
                        city_id.value, variable, agreement_std
                    )
                    continue

            # Apply pattern-based bias shift (e.g. haze suppression)
            if pattern_bias != 0:
                old_mean = consensus.weighted_mean
                consensus.weighted_mean += pattern_bias
                consensus.mean_value += pattern_bias
                logger.info(
                    "  %s/%s PATTERN SHIFT: %.1f°C -> %.1f°C (%+.1f°C)",
                    city_id.value, variable, old_mean, consensus.weighted_mean, pattern_bias
                )

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
                _ai_div_key = (city_id.value, target_date, variable)
                if _ai_div_key not in _ai_divergence_cache:
                    ai_fc = ai_forecasts.get((city_id.value, target_date))
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

                # Spread: real top-of-book if the market carries it, else a
                # conservative default. Gamma outcomePrices are complementary
                # (YES + NO ~= 1), so 1-(yes+no) is ~0 and hid the real cost.
                from weather_edge.trading.market_maker import estimate_market_spread
                estimated_spread = estimate_market_spread(market)

                from weather_edge.analysis.risk_controls import get_active_profile
                risk_profile = get_active_profile()

                signal = calculate_edge(
                    market_id=market.market_id,
                    model_prob=model_prob,
                    market_prob=market_prob,
                    model_confidence=adjusted_conf,
                    bankroll=total_equity,
                    consensus_id=None,
                    hours_to_resolution=hours_to,
                    city_id=city_id.value,
                    target_date=str(target_date),
                    description=market.question[:80],
                    spread=estimated_spread,
                    min_edge_yes=risk_profile.min_edge_yes,
                    min_edge_no=risk_profile.min_edge_no,
                )

                # Z-score guard: reject core bets on buckets too far from consensus.
                # Tail/penny bets (entry <=6c) are exempt, they're designed as
                # lottery tickets. HAIL MARY mode skips the guard: penny-only
                # baskets are far-from-consensus by design.
                if (
                    not settings.hail_mary_mode
                    and signal.strategy != "tail"
                    and market_prob > 0.06
                ):
                    bucket_center = None
                    if (
                        market.threshold_dir == "range"
                        and market.threshold_low_c is not None
                        and market.threshold_high_c is not None
                    ):
                        bucket_center = (market.threshold_low_c + market.threshold_high_c) / 2
                    elif (
                        market.threshold_dir in ("gte", "lte")
                        and market.threshold_value is not None
                    ):
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

    filtered_signals: list[Signal] = []
    if settings.hail_mary_mode:
        # === HAIL MARY: penny lottery basket ===
        # The normal strategy filters to one signal per city-date. The hail-mary
        # takes EVERY cheap signal, multi-bucket stacking is fine when each
        # ticket costs $1 because the math is "buy 30 lottery tickets, hope 1 hits."
        hailmary_max_price = 0.10   # only buy stuff priced under 10¢
        for s in all_signals:
            # Penny ceiling, anything above 10¢ is "more of the same bs"
            try:
                if s.recommended_side.value == "YES":
                    entry_price = s.market_prob
                else:
                    entry_price = 1.0 - s.market_prob
            except Exception:
                entry_price = s.market_prob
            if entry_price > hailmary_max_price:
                continue
            # Require some directional edge, we're not buying random things
            if s.edge <= 0:
                continue
            filtered_signals.append(s)

        logger.info(
            "HAILMARY BASKET: %d penny signals (price<=%.2f, edge>0) from %d raw",
            len(filtered_signals), hailmary_max_price, len(all_signals),
        )
    elif all_signals:
        # === STRATEGY REDESIGN: One signal per city-date filter ===
        # Prevents "multi-bucket bleed" by picking only the highest-edge bucket.
        # Prioritizes tail_no (high-prob NO) and tail (penny) strategies.
        signal_groups: dict[tuple[str, str], list[Signal]] = {}
        for s in all_signals:
            if s.confidence_tier.value == "low":
                continue
            key = (s.city_id, s.target_date)
            signal_groups.setdefault(key, []).append(s)

        for key, group in signal_groups.items():
            # Strategy priority: tail_no > tail > core
            # Within same strategy, pick highest absolute net_edge
            def _signal_score(sig: Signal) -> float:
                prio = {"tail_no": 300, "tail": 200, "core": 100}.get(sig.strategy, 0)
                return prio + abs(sig.net_edge)

            best_signal = max(group, key=_signal_score)
            filtered_signals.append(best_signal)

            if len(group) > 1:
                logger.info(
                    "MULTI-BUCKET FILTER: %s %s picked %s (edge=%.1f%%) over %d others",
                    key[0], key[1], best_signal.strategy,
                    best_signal.net_edge * 100, len(group) - 1,
                )

        # Apply global max trades per cycle limit
        # Sort by absolute net_edge descending, take top N
        if len(filtered_signals) > settings.max_trades_per_cycle:
            limited = sorted(
                filtered_signals,
                key=lambda s: abs(s.net_edge),
                reverse=True,
            )[:settings.max_trades_per_cycle]

            logger.info(
                "CYCLE LIMIT: %d signals filtered down to top %d by net_edge (max_trades_per_cycle=%d)",
                len(filtered_signals), len(limited), settings.max_trades_per_cycle,
            )
            filtered_signals = limited

    all_signals = filtered_signals

    # === Claude + Gemini reasoning layer ===
    # Every cycle type (main, sniper, refresh) goes through the same gate: only
    # signals the AI layer approved (this cycle, or earlier today) are executed.
    executable_signals = await apply_ai_review(
        all_signals, _forecast_cache, run_ai_reasoning, store,
    )

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
    top_signals = sorted(executable_signals, key=lambda s: abs(s.edge), reverse=True)[:5]
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
    # Don't place ANY new live orders if balance is below $20. Prevents
    # deploying dust into marginal trades. Capital comes back via
    # resolution/redemption, then we re-enter on high-conviction 48h+ signals.
    # HAIL MARY disables the floor ($1 tickets down to the last dollar).
    usdc_floor = 0.0 if settings.hail_mary_mode else 20.0
    _usdc_floor_block = bool(
        live_executor
        and not live_executor.dry_run
        and _live_balance is not None
        and _live_balance < usdc_floor
    )
    if _usdc_floor_block:
        logger.warning(
            "USDC FLOOR: $%.2f < $%.0f minimum, blocking all new live entries",
            _live_balance, usdc_floor,
        )

    # --- SWING BOT: Position cap ---
    # 50 allows new entries alongside existing positions; most existing
    # positions are small/penny bets that resolve naturally. Effective
    # concentration is managed by the min size + USDC floor. HAIL MARY
    # removes the cap for the penny basket.
    max_positions = 999 if settings.hail_mary_mode else 50
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
            if _active_position_count >= max_positions:
                logger.warning(
                    "POSITION CAP: %d/%d active positions, blocking new live entries",
                    _active_position_count, max_positions,
                )
        except Exception:
            _active_position_count = store.get_portfolio_summary().get("position_count", 0)

    logger.info(
        "EXECUTION LOOP: %d signals, live_executor=%s, dry_run=%s, "
        "usdc_floor_block=%s, pos_count=%d/%d",
        len(executable_signals),
        bool(live_executor),
        live_executor.dry_run if live_executor else "N/A",
        _usdc_floor_block,
        _active_position_count,
        max_positions,
    )
    for signal in executable_signals:
        # A zero/negative size means some layer (Kelly, Claude, Gemini) sized
        # the trade away. Never trade it, and never let the live min-size
        # floor below resurrect it.
        if signal.recommended_size <= 0 and not settings.hail_mary_mode:
            logger.info(
                "ZERO SIZE: %s %s sized to $%.2f, skipping",
                signal.city_id, signal.description[:40], signal.recommended_size,
            )
            continue

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
            and _active_position_count < max_positions
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
                            "EXIT COOLDOWN: %s, exited <4h ago, "
                            "edge=%.1f%% (need >12%% to bypass)",
                            signal.city_id, signal.edge * 100,
                        )
                        continue
                    else:
                        logger.warning(
                            "COOLDOWN BYPASS: %s, massive edge %.1f%% overrules 4h window",
                            signal.city_id, signal.edge * 100,
                        )

            if settings.hail_mary_mode:
                # --- HAIL MARY: fixed $1.00 lottery-ticket sizing ---
                # Override whatever Kelly/Claude/Gemini decided. Every order is
                # exactly $1.00, penny basket, not concentrated bets.
                signal.recommended_size = HAILMARY_TICKET_USD

                # Margin check, only block if we don't even have $1
                if _live_balance is not None and _live_balance < HAILMARY_TICKET_USD:
                    logger.info(
                        "HAILMARY OUT OF CASH: balance $%.2f < $%.2f, skipping %s",
                        _live_balance, HAILMARY_TICKET_USD, signal.city_id,
                    )
                    continue
            else:
                # --- SWING BOT: Minimum position size ---
                # Survival tier: $5 min until bankroll > $500, then raise to $10.
                # Only genuinely small positive sizes are bumped; a zero size
                # (AI veto / Kelly says no) was already dropped above.
                if signal.recommended_size <= 0:
                    continue
                if signal.recommended_size < MIN_LIVE_SIZE:
                    signal.recommended_size = MIN_LIVE_SIZE

                # Margin check, use tracked live balance, not stale portfolio
                # calc. Runs AFTER min size bump so the floor is respected.
                if _live_balance is not None and signal.recommended_size > _live_balance:
                    logger.info(
                        "BALANCE LIMIT: %s needs $%.0f but exchange balance is $%.2f, skipping",
                        signal.city_id, signal.recommended_size, _live_balance,
                    )
                    continue

            is_spread = getattr(signal, "strategy", "") == "spread"
            # Minimum 2% live edge; HAIL MARY takes any positive edge.
            if settings.hail_mary_mode:
                can_live = signal.edge > 0 or is_spread
            else:
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

                            chase_price = None
                            if should_chase:
                                # Price chase: improve by ONE TICK toward the
                                # midpoint. The executor rounds limits to the
                                # 1c tick, so a sub-tick bump would re-place at
                                # the same price (cancel for nothing, losing
                                # queue priority). Never chase above the side's
                                # midpoint (would cross as a post-only buy).
                                side_mid = (
                                    signal.market_prob
                                    if signal.recommended_side.value == "YES"
                                    else 1.0 - signal.market_prob
                                )
                                from weather_edge.trading.executor import chase_limit_price
                                chase_price = chase_limit_price(old_price, side_mid)
                                if chase_price is None:
                                    logger.info(
                                        "PRICE CHASE SKIP: %s @ %.2f already at "
                                        "mid cap %.3f, keeping order",
                                        signal.city_id, old_price, side_mid,
                                    )
                                    continue
                            if chase_price is not None:
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

                        if settings.hail_mary_mode:
                            # --- HAIL MARY: all risk control checks bypassed ---
                            # yes_exposure_cap, correlation_limit, gross_exposure all
                            # skipped. Sizing is fixed $1, max blast radius per
                            # ticket is $1.
                            signal.recommended_size = HAILMARY_TICKET_USD
                        else:
                            # === RISK CONTROL CHECKS ===
                            from weather_edge.analysis.risk_controls import (
                                check_correlation_limit,
                                check_gross_exposure,
                                check_yes_exposure_limit,
                                get_active_profile,
                                live_circuit_breaker_multiplier,
                            )
                            from weather_edge.models.position import Position, normalize_side
                            profile = get_active_profile()

                            # 0. Live drawdown circuit breaker (fed real NAV by
                            # fetch_polymarket_state; a kill also sets the
                            # persistent kill switch).
                            cb_mult = live_circuit_breaker_multiplier()
                            if cb_mult <= 0:
                                logger.warning(
                                    "LIVE CIRCUIT BREAKER: killed, skipping %s",
                                    signal.city_id,
                                )
                                continue
                            if cb_mult < 1.0:
                                signal.recommended_size = round(
                                    signal.recommended_size * cb_mult, 2,
                                )

                            # 1. Yes Exposure Cap
                            if signal.recommended_side.value == "YES":
                                raw_pos = store.get_positions()
                                open_pos = [
                                    Position(
                                        market_id=p.get("condition_id"),
                                        city_id=p.get("city_id"),
                                        side=normalize_side(p.get("outcome") or p.get("side")),
                                        size_usd=p.get("cost_basis", 0),
                                        status="open" if p.get("total_shares", 0) > 0 else "closed"
                                    ) for p in raw_pos
                                ]
                                allowed, new_size, reason = check_yes_exposure_limit(
                                    signal.recommended_size, open_pos, total_equity, profile
                                )
                                if not allowed:
                                    logger.warning(reason)
                                    continue
                                if new_size < signal.recommended_size:
                                    logger.info(reason)
                                    signal.recommended_size = new_size

                            # 2. Correlation Limit
                            raw_pos = store.get_positions()
                            open_pos = [
                                Position(
                                    market_id=p.get("condition_id"),
                                    city_id=p.get("city_id"),
                                    side=normalize_side(p.get("outcome") or p.get("side")),
                                    size_usd=p.get("cost_basis", 0),
                                    status="open" if p.get("total_shares", 0) > 0 else "closed"
                                ) for p in raw_pos
                            ]
                            allowed, new_size, reason = check_correlation_limit(
                                signal.city_id, signal.recommended_size, open_pos, total_equity, profile
                            )
                            if not allowed:
                                logger.warning(reason)
                                continue
                            if new_size < signal.recommended_size:
                                logger.info(reason)
                                signal.recommended_size = new_size

                            # 3. Gross Exposure
                            # Use real position value (total_equity - cash) not inflated
                            # DB cost_basis. DB cost_basis includes resolved positions;
                            # Polymarket value is truth.
                            total_at_risk = max(0, total_equity - (_live_balance or 0))
                            allowed, new_size, reason = check_gross_exposure(
                                signal.recommended_size, total_at_risk, total_equity, profile
                            )
                            if not allowed:
                                logger.warning(reason)
                                continue
                            if new_size < signal.recommended_size:
                                logger.info(reason)
                                signal.recommended_size = new_size

                            # Final min-size check after all trims
                            if signal.recommended_size < MIN_LIVE_SIZE:
                                logger.info("TRIMMED BELOW MIN: %s size $%.2f < $%.2f",
                                            signal.city_id, signal.recommended_size, MIN_LIVE_SIZE)
                                continue

                        try:
                            if settings.hail_mary_mode:
                                # --- HAIL MARY: always taker on penny tickets ---
                                # We need to actually fill before resolution. At
                                # 1-3¢ the taker fee is sub-cent and the maker
                                # order would never get filled.
                                use_taker = True
                            else:
                                # --- SWING BOT: Hybrid entry ---
                                # Edge >= 8%: taker (cross spread, pay fee, secure alpha)
                                # Below: maker (rest on book, cancel if unfilled)
                                # At tail prices (<10c), taker fee is negligible anyway
                                use_taker = signal.edge >= 0.08
                                # Re-check fee gate for taker orders, maker is $0 fee
                                # but taker pays real fees that could eat alpha.
                                # At $5 bets, taker fee is ~10c, skip fee gate, not
                                # worth the miss.
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
                                # HAIL MARY skips hedging: every dollar of
                                # capital is one ticket, no doubling up.
                                if (
                                    market_maker
                                    and not settings.hail_mary_mode
                                    and not live_executor.dry_run
                                ):
                                    # Use settings.bankroll as baseline for pool-sizing
                                    hedge = market_maker.generate_hedge_orders(
                                        signal, market_prices, settings.bankroll, is_live=True
                                    )
                                    if hedge:
                                        m = market_by_id.get(hedge.market_id)
                                        if m:
                                            from weather_edge.trading.market_maker import (
                                                build_hedge_signal,
                                            )
                                            h_token_id = (
                                                m.token_id_yes if hedge.side == "YES"
                                                else m.token_id_no
                                            )
                                            # Synthetic signal: market_prob is the
                                            # YES-equivalent of the hedge token price
                                            h_signal = build_hedge_signal(signal, hedge)
                                            h_side = h_signal.recommended_side

                                            try:
                                                import requests
                                                # improve_price_by=0: limit is exactly
                                                # hedge.limit_price for the hedge token
                                                h_result = await live_executor.place_limit_order(
                                                    h_signal, h_token_id, improve_price_by=0.0,
                                                )
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
        from weather_edge.analysis.exit_monitor import ai_review_exit, scan_for_exits_async
        current_market_prices = {m.market_id: m.yes_price for m in markets}
        # NO token's own price (Gamma outcomePrices[1]); exit_monitor falls
        # back to 1-YES when missing.
        current_no_prices = {
            m.market_id: m.no_price for m in markets if getattr(m, "no_price", 0) > 0
        }
        current_model_probs = {}
        for signal in all_signals:
            current_model_probs[signal.market_id] = signal.model_prob

        # --- PAPER EXIT SCANNING ---
        if paper_trader:
            paper_candidates = await scan_for_exits_async(
                paper_trader.open_trades,
                current_market_prices, current_model_probs,
                forecast_cache=_forecast_cache,
                no_prices=current_no_prices,
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
        # HAIL MARY disables it: penny lottery tickets must hold to resolution,
        # and selling at -7% edge wastes the spread cost on a $1 position.
        if (
            not settings.hail_mary_mode
            and live_executor
            and not live_executor.dry_run
            and store
        ):
            from weather_edge.models.position import PRICE_BASIS_TOKEN, Position
            positions = store.get_positions()
            live_positions: list[Position] = []
            for pos in positions:
                shares = pos.get("total_shares", 0)
                avg_price = pos.get("avg_price", 0)
                if shares >= MIN_SELL_SHARES:
                    live_positions.append(Position(
                        market_id=pos.get("condition_id", ""),
                        city_id=pos.get("city_id", ""),
                        # positions.side is the fill side (BUY); the token
                        # held is the outcome. Normalised to YES/NO.
                        side=pos.get("outcome") or pos.get("side", ""),
                        size_usd=pos.get("cost_basis", 0),
                        # avg_price is what we paid for the held token
                        entry_price=avg_price,
                        price_basis=PRICE_BASIS_TOKEN,
                        total_shares=shares,
                        description=pos.get("description", ""),
                        source="live",
                    ))

            if live_positions:
                live_candidates = await scan_for_exits_async(
                    live_positions,
                    current_market_prices, current_model_probs,
                    forecast_cache=_forecast_cache,
                    no_prices=current_no_prices,
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

                            # Sell-half guard: skip if already trimmed
                            if candidate.reason == "sell_half":
                                existing = store.get_open_order_for_market(
                                    market_id, side="SELL",
                                )
                                if existing:
                                    logger.info(
                                        "SELL_HALF SKIP: %s has open sell order",
                                        city_id_str,
                                    )
                                    continue
                                past_trim = store.conn.execute(
                                    "SELECT 1 FROM live_trades "
                                    "WHERE market_id = ? "
                                    "AND description LIKE 'SELL_HALF%' "
                                    "LIMIT 1",
                                    (market_id,),
                                ).fetchone()
                                if past_trim:
                                    logger.info(
                                        "SELL_HALF SKIP: %s already trimmed",
                                        city_id_str,
                                    )
                                    continue

                            # 1. Cancel any open BUY orders if exiting
                            open_buy = store.get_open_order_for_market(market_id)
                            if open_buy and open_buy.get("side") in ("YES", "NO"):
                                buy_id = open_buy["order_id"]
                                try:
                                    await live_executor.cancel_order(buy_id)
                                    store.cancel_live_trade(buy_id)
                                    logger.info(
                                        "EXIT: cancelled open BUY order %s for %s",
                                        buy_id[:16], city_id_str,
                                    )
                                except Exception as e:
                                    logger.warning(
                                        "Failed to cancel BUY order %s: %s",
                                        buy_id[:16], e,
                                    )

                            # 2. Check for existing open SELL order
                            existing_sell = store.get_open_order_for_market(
                                market_id, side="SELL",
                            )

                            position = store.get_position_for_market(market_id)
                            if (
                                position
                                and position.get("total_shares", 0) >= MIN_SELL_SHARES
                            ):
                                asset_id = position.get("asset_id", "")
                                # Sell the held token at ITS OWN price. The
                                # candidate's current_market_price is always
                                # the YES price, selling a NO token there
                                # would dump it (or never fill).
                                sell_price = (
                                    candidate.token_price
                                    if candidate.token_price is not None
                                    else candidate.current_market_price
                                )
                                if asset_id:
                                    # Replace existing sell only if price drifted >1c
                                    if existing_sell:
                                        old_price = existing_sell.get("limit_price", 0)
                                        new_price = _round_price(sell_price)
                                        price_drift = abs(old_price - new_price)

                                        if price_drift <= 0.01:
                                            logger.info(
                                                "LIVE SELL KEEP: %s @ %.3f "
                                                "(drift=%.3f)",
                                                city_id_str, old_price, price_drift,
                                            )
                                            continue
                                        else:
                                            old_id = existing_sell["order_id"]
                                            try:
                                                await live_executor.cancel_order(old_id)
                                                store.cancel_live_trade(old_id)
                                                logger.info(
                                                    "LIVE SELL REPLACE: %s "
                                                    "cancelled %s (%.3f->%.3f)",
                                                    city_id_str, old_id[:16],
                                                    old_price, new_price,
                                                )
                                            except Exception as e:
                                                logger.warning(
                                                    "Failed to cancel old "
                                                    "sell order %s: %s",
                                                    old_id[:16], e,
                                                )

                                    try:
                                        urgent = candidate.urgency == "high"
                                        sell_shares = position["total_shares"]
                                        if candidate.reason == "sell_half":
                                            sell_shares = max(
                                                MIN_SELL_SHARES,
                                                round(sell_shares / 2, 0),
                                            )
                                        exit_label = (
                                            "SELL_HALF"
                                            if candidate.reason == "sell_half"
                                            else "EXIT"
                                        )
                                        # reference_price=None: the executor
                                        # checks the limit against the token's
                                        # live CLOB midpoint (an independent
                                        # mark), not against itself.
                                        sell_result = await live_executor.place_sell_order(
                                            token_id=asset_id,
                                            shares=sell_shares,
                                            price=sell_price,
                                            market_id=market_id,
                                            city_id=city_id_str,
                                            description=(
                                                f"{exit_label}: {candidate.reason}"
                                            ),
                                            reference_price=None,
                                            force_taker=urgent,
                                        )
                                        if sell_result and sell_result.status not in (
                                            "rejected", "post_only_reject",
                                        ):
                                            logger.warning(
                                                "LIVE EXIT: %s %s %.0f shares "
                                                "@ %.3f, %s",
                                                city_id_str, candidate.trade.side,
                                                sell_shares,
                                                sell_price,
                                                sell_result.order_id,
                                            )
                                    except Exception as e:
                                        logger.error(
                                            "LIVE EXIT FAILED: %s, %s",
                                            city_id_str, e,
                                        )

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
    if len(target_dates) > 1:
        tomorrow = target_dates[1]
    elif target_dates:
        tomorrow = target_dates[0]
    else:
        tomorrow = date.today() + timedelta(days=1)
    for city_id in City:
        # The cache persists across cycles, so "not in cache" would freeze
        # monitoring data after the first cycle; refetch unless fresh this cycle.
        if (city_id, tomorrow) not in _refreshed_this_cycle:
            forecasts = await fetch_city_forecasts(city_id, tomorrow)
            if forecasts:
                _forecast_cache[(city_id, tomorrow)] = forecasts
                _refreshed_this_cycle.add((city_id, tomorrow))
                consensus = compute_consensus(city_id, str(tomorrow), "temp_max_c", forecasts)
                if consensus:
                    logger.info(
                        "  %s (no markets): mean=%.1f°C conf=%.0f%%",
                        city_id.value, consensus.weighted_mean, consensus.confidence * 100,
                    )

    return all_signals, _forecast_cache, city_volume


async def run_loop(
    paper_trader: PaperTrader | None,
    days: int | None = None,
    max_cycles: int | None = None,
) -> None:
    """Run the fetch-analyze-trade loop continuously.

    Args:
        paper_trader: Paper trader (None disables paper trading).
        days: Scan ``today .. today + days - 1`` each cycle (recomputed every
            cycle so the window rolls over at midnight). None = scheduler default.
        max_cycles: Stop after this many cycles (None = forever; for tests).
    """
    from weather_edge.analysis.claude_reasoning import get_review_memory

    interval = settings.fetch_interval_minutes * 60
    cycle_num = 0
    forecast_cache: dict[tuple, list] = {}

    while True:
        cycle_num += 1
        logger.info("===== CYCLE %d START =====", cycle_num)
        try:
            today = date.today()
            evicted = evict_stale_forecasts(forecast_cache, today)
            evicted += get_review_memory().evict_before(today)
            if evicted:
                logger.info("Evicted %d cache entries for past dates", evicted)
            target_dates = default_target_dates(today, days) if days else None
            # Pass existing cache to run_cycle
            signals, forecast_cache, _ = await run_cycle(
                paper_trader,
                target_dates,
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

        if max_cycles is not None and cycle_num >= max_cycles:
            return
        logger.info("Sleeping %d minutes until next cycle...", settings.fetch_interval_minutes)
        await asyncio.sleep(interval)
