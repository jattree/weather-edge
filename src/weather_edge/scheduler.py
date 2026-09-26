"""Main orchestration loop: fetch → consensus → edge → trade."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

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
    CITY_CLIMATOLOGY,
    CLIMATOLOGICAL_MEAN,
    CLIMATOLOGICAL_STD,
    EMOS_VARIANCE_FLOOR_C,
    MAX_BUCKET_PROBABILITY,
    MAX_BUCKET_PROBABILITY_EXTREME,
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
from weather_edge.analysis.market_mapper import MARKET_TYPE_TO_VARIABLE, get_required_variable
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
    fetch_all_data_api_positions,
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


def _lte_probability(market: MarketInfo, consensus, band) -> float:
    """P(displayed high <= the bucket's top label)."""
    hi_c = band[1] if band is not None else None
    if hi_c is None:
        hi_c = market.threshold_high_c or market.threshold_value
    return 1.0 - get_probability_for_threshold(consensus, hi_c, "gte")


def _range_probability(market: MarketInfo, consensus, band) -> float | None:
    """P(displayed high lands in the bucket), None without usable bounds."""
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
        return max(0.0, p_lo - p_hi)
    if market.threshold_low_c is not None and market.threshold_high_c is not None:
        # Legacy fallback when native integer labels are unavailable.
        p_gte_low = get_probability_for_threshold(
            consensus, market.threshold_low_c, "gte",
        )
        p_gte_high = get_probability_for_threshold(
            consensus, market.threshold_high_c, "gte",
        )
        return max(0.0, p_gte_low - p_gte_high)
    return None


def _raw_bucket_probability(market: MarketInfo, consensus) -> float | None:
    """Uncapped model probability for the market's bucket (None if unsupported)."""
    # Round-half-up display semantics: a market resolves YES when the displayed
    # (rounded) daily high lands on one of the bucket's integer labels. An
    # integer label L therefore covers the continuous half-open interval
    # [L-0.5, L+0.5) in the displayed unit. We integrate the model distribution
    # over exactly that interval (converted to °C) so probability and resolution
    # use the SAME boundary convention. The previous code added a flat 1.0 to a
    # °C-converted bound for Fahrenheit buckets, which mixed units and made every
    # F range bucket ~0.8°F too wide on the top edge (inflating YES probability).
    band = _bucket_celsius_band(market)
    direction = market.threshold_dir
    if direction == "lte":
        return _lte_probability(market, consensus, band)
    if direction == "range":
        return _range_probability(market, consensus, band)
    if direction == "gte":
        lo_c = band[0] if band is not None else None
        return get_probability_for_threshold(
            consensus, lo_c if lo_c is not None else market.threshold_value, "gte",
        )
    if direction == "any":
        return get_probability_for_threshold(consensus, 0.0, "any")
    return None


def _bucket_probability_cap(market: MarketInfo, consensus) -> float:
    """EMOS cap for a narrow (range/lte) bucket, raised during extreme events.

    Extreme = tight model agreement and a consensus > 2 sigma from the city's
    climatology.
    """
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
    return cap


def compute_model_prob_for_market(market: MarketInfo, consensus) -> float | None:
    """Compute model probability for a market bucket.

    Handles the multi-bucket format with EMOS probability cap:
    - A single 2°F bucket should never exceed 70% at >12h horizon
    - Per Gemini: >90% on a single bucket is "likely broken"
    """
    prob = _raw_bucket_probability(market, consensus)
    # Apply bucket probability cap for range/lte buckets (narrow temperature ranges)
    # During extreme events (tight model agreement + anomalous temps), raise the cap
    if prob is not None and market.threshold_dir in ("range", "lte"):
        prob = min(prob, _bucket_probability_cap(market, consensus))
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
    0h horizon. This is a coarse, city-agnostic pre-filter; run_cycle also
    applies the horizon per market against the city's local end of day
    (hours_to_local_end_of_day).
    """
    if now_utc is None:
        now_utc = datetime.now(UTC)
    if min_horizon_hours is None:
        min_horizon_hours = 0 if settings.hail_mary_mode else MIN_HORIZON_HOURS
    min_horizon = timedelta(hours=min_horizon_hours)
    kept, blocked = [], []
    for d in target_dates:
        resolves = datetime.combine(d + timedelta(days=1), datetime.min.time()).replace(
            tzinfo=UTC,
        )
        (kept if resolves - now_utc >= min_horizon else blocked).append(d)
    return kept, blocked, min_horizon_hours


def hours_to_local_end_of_day(
    city_id: City, target_date: date, now_utc: datetime | None = None,
) -> float:
    """Hours from ``now_utc`` to the end of ``target_date`` in the city's zone.

    The daily high is fixed at local midnight, which for an Asian or Pacific
    city is up to 13h before 00:00 UTC on the next day (and after it in the
    Americas). Falls back to UTC for a city without config. Never negative.
    """
    if now_utc is None:
        now_utc = datetime.now(UTC)
    city = CITIES.get(city_id)
    tz = ZoneInfo(city.timezone) if city else UTC
    end_local = datetime.combine(target_date + timedelta(days=1), datetime.min.time(), tzinfo=tz)
    return max(0.0, (end_local - now_utc).total_seconds() / 3600)


def trading_today(now_utc: datetime | None = None) -> date:
    """The earliest local calendar date among tracked cities.

    Markets resolve on each city's local date, so a date is still trading
    while any city is on it. This is the oldest such date: the start of the
    scan window and the cut-off for evicting past dates. Independent of the
    host's time zone (it used to be ``date.today()``, the host's local date).
    """
    if now_utc is None:
        now_utc = datetime.now(UTC)
    zones = {c.timezone for c in CITIES.values()}
    return min(
        (now_utc.astimezone(ZoneInfo(tz)).date() for tz in zones),
        default=now_utc.astimezone(UTC).date(),
    )


def default_target_dates(today: date | None = None, days: int = 4) -> list[date]:
    """``today`` .. ``today + days - 1`` (the scheduler's default window)."""
    today = today or trading_today()
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
    oldest = min(t if t.tzinfo else t.replace(tzinfo=UTC) for t in times)
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


def exit_model_context(
    trade, forecast_cache: dict[tuple, list], market_dates: dict[str, date],
) -> tuple[dict[str, float], float, float]:
    """(model_values, mean, std) for an exit review of ``trade``.

    Positions carry no target date, so it comes from the trade itself when
    present, else from the discovered market with the same market_id. Uses
    only that date's forecasts (see model_context_for_signal); with no known
    date or no forecasts it returns empty context and std 1.0.
    """
    target = getattr(trade, "target_date", None) or market_dates.get(trade.market_id)
    model_vals, mean, std = {}, 0.0, 1.0
    if target is not None:
        ctx = SimpleNamespace(city_id=trade.city_id, target_date=target)
        model_vals, mean, std = model_context_for_signal(ctx, forecast_cache)
    if not model_vals:
        return {}, 0.0, 1.0
    return model_vals, mean, std


def _gemini_size_multiplier(dissent: float, sizing: str) -> float:
    """Position-size multiplier from a Gemini red-team verdict (1.0 = no cut).

    An explicit "skip" is a veto whatever the dissent strength: halving it
    instead let the $5 live minimum turn a skipped trade into an order.
    """
    if sizing == "skip":
        return 0.0
    if dissent >= 0.7 or sizing == "half":
        return 0.5
    if dissent >= 0.3 and sizing == "reduce_20pct":
        return 0.8
    return 1.0


def _save_claude_decision(store, signal: Signal, reasoning) -> None:
    """Persist a Claude verdict for accuracy analysis (best effort)."""
    if store is None:
        return
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
        # Analytics only: a failed write must never block the review.
        logger.warning("Failed to persist Claude decision for %s", signal.market_id,
                       exc_info=True)


def _apply_claude_verdict(signal: Signal, reasoning, memory, hail: bool) -> float | None:
    """Apply Claude's verdict. Returns the size multiplier, or None if vetoed."""
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
            return None
    elif not hail:
        # Apply Claude's confidence adjustment to position size.
        # HAIL MARY ignores it, sizing is fixed downstream.
        size_mult = reasoning.confidence_adjustment
        signal.recommended_size = round(signal.recommended_size * size_mult, 2)
    return size_mult


async def _gemini_red_team(
    signal: Signal, model_vals: dict[str, float], consensus_mean: float,
    consensus_std: float, reasoning,
) -> dict | None:
    """Gemini red team on a Claude-approved trade.

    None = Gemini not configured (Claude-only review). A configured Gemini
    that errors returns a fail-closed result carrying "error".
    """
    from weather_edge.analysis import gemini_reasoning

    try:
        return await gemini_reasoning.red_team_trade(
            signal, model_vals, consensus_mean, consensus_std,
            claude_rationale=reasoning.rationale,
        )
    except Exception as e:
        # Any Gemini failure fails closed (below), so catch everything.
        logger.warning("Gemini red team crashed for %s: %s", signal.city_id, e, exc_info=True)
        return {"error": str(e), "dissent_strength": 1.0,
                "verdict": "DISSENT", "sizing_recommendation": "skip"}


def _record_gemini_decision(
    signal: Signal, gemini_result: dict, store, dissent: float, verdict: str,
) -> None:
    """Add a Gemini verdict to the AI Decisions tab and persist it (best effort)."""
    from weather_edge.analysis.claude_reasoning import _decision_history

    _decision_history.insert(0, {
        "time": datetime.now(UTC).strftime("%H:%M:%S"),
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
    if store is None:
        return
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
        # Analytics only: a failed write must never block the review.
        logger.warning("Failed to persist Gemini decision for %s", signal.market_id,
                       exc_info=True)


def _apply_gemini_sizing(signal: Signal, dissent: float, sizing: str, memory) -> float | None:
    """Cut ``signal`` by the Gemini multiplier. Returns it, or None if Gemini vetoes."""
    multiplier = _gemini_size_multiplier(dissent, sizing)
    if multiplier <= 0:
        logger.info(
            "GEMINI VETO: %s, sizing=%s dissent=%.1f, dropping signal",
            signal.city_id, sizing, dissent,
        )
        memory.record_veto(signal, f"gemini {sizing} d={dissent:.1f}", "gemini")
        return None
    if multiplier < 1.0:
        old_size = signal.recommended_size
        signal.recommended_size = round(signal.recommended_size * multiplier, 2)
        logger.info(
            "GEMINI DISSENT: %s, %.0f%% cut $%.0f->$%.0f (d=%.1f %s)",
            signal.city_id, (1 - multiplier) * 100,
            old_size, signal.recommended_size, dissent, sizing,
        )
    return multiplier


def _gemini_gate(
    signal: Signal, gemini_result: dict, store, *, memory, hail: bool,
) -> tuple[str | None, float]:
    """Apply a Gemini verdict: (terminal outcome or None, size multiplier)."""
    dissent = float(gemini_result.get("dissent_strength", 1.0))
    verdict = gemini_result.get("verdict", "DISSENT")
    sizing = gemini_result.get("sizing_recommendation", "skip")
    _record_gemini_decision(signal, gemini_result, store, dissent, verdict)

    if hail:
        if gemini_result.get("error") or dissent >= 0.3:
            logger.info(
                "HAILMARY override GEMINI DISSENT: %s d=%.1f %s (no cut applied)",
                signal.city_id, dissent, sizing,
            )
        return None, 1.0
    if gemini_result.get("error"):
        logger.warning(
            "GEMINI REVIEW FAILED: %s, not executing this cycle (%s)",
            signal.city_id, gemini_result["error"],
        )
        return "failed", 1.0
    multiplier = _apply_gemini_sizing(signal, dissent, sizing, memory)
    if multiplier is None:
        return "vetoed", 1.0
    return None, multiplier


async def _review_signal(signal: Signal, forecast_cache: dict[tuple, list], store) -> str:
    """Claude + Gemini review of one signal: 'approved', 'vetoed' or 'failed'.

    Outside hail-mary, applies the AI sizing to ``signal.recommended_size`` and
    records approvals/vetoes in the shared review memory. In hail-mary mode the
    verdicts are logged only (no cut, no block, nothing remembered).
    """
    from weather_edge.analysis.claude_reasoning import get_review_memory

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
    _save_claude_decision(store, signal, reasoning)

    if not reasoning.review_ok and not hail:
        return "failed"

    size_mult = _apply_claude_verdict(signal, reasoning, memory, hail)
    if size_mult is None:
        return "vetoed"

    gemini_result = await _gemini_red_team(
        signal, model_vals, consensus_mean, consensus_std, reasoning,
    )
    if gemini_result:
        outcome, multiplier = _gemini_gate(
            signal, gemini_result, store, memory=memory, hail=hail,
        )
        if outcome:
            return outcome
        size_mult *= multiplier

    if not hail:
        memory.record_approval(signal, size_mult, reasoning.rationale)
    return "approved"


def _remembered_vetoes(signals: list[Signal], memory) -> dict[int, str]:
    """``{id(signal): "vetoed"}`` for signals with a veto recorded earlier today."""
    outcome: dict[int, str] = {}
    for s in signals:
        veto = memory.veto_for(s)
        if veto:
            outcome[id(s)] = "vetoed"
            logger.info(
                "AI VETO (remembered, %s): %s %s %s, %s",
                veto.source, s.city_id, s.target_date, s.description[:40], veto.reason,
            )
    return outcome


async def _review_candidates(
    signals: list[Signal], outcome: dict[int, str], forecast_cache: dict[tuple, list],
    store, *, memory, max_reviews: int,
) -> None:
    """Review up to ``max_reviews`` signals, recording each result in ``outcome``."""
    candidates = [
        s for s in signals
        if s.confidence_tier.value != "low" and id(s) not in outcome
    ]
    # Unreviewed signals first so the budget eventually covers all of them.
    candidates.sort(key=lambda s: (memory.approval_for(s) is not None, -abs(s.edge)))
    for s in candidates[:max_reviews]:
        outcome[id(s)] = await _review_signal(s, forecast_cache, store)


def _unreviewed_reason(run_ai_reasoning: bool, max_reviews: int) -> str:
    """Why a signal with no review this cycle went unreviewed (for the log)."""
    if not run_ai_reasoning:
        return "AI skipped on this cycle (sniper / refresh cooldown)"
    if not ANTHROPIC_API_KEY:
        return "no ANTHROPIC_API_KEY"
    return f"outside the {max_reviews}-review budget"


def _clears_ai_gate(
    s: Signal, state: str | None, memory, *, require_review: bool, why: str,
) -> bool:
    """True if ``s`` may execute given its review ``state`` this cycle."""
    if state == "vetoed":
        return False
    if state == "approved":
        return s.recommended_size > 0
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
                return True
            return False
    # Failed review, or never reviewed
    if not require_review:
        logger.warning(
            "UNREVIEWED TRADE (require_ai_review=False): %s %s %s",
            s.city_id, s.target_date, s.description[:40],
        )
        return True
    logger.info(
        "NO AI APPROVAL: %s %s %s not executed (%s)",
        s.city_id, s.target_date, s.description[:40],
        "review failed" if state == "failed" else why,
    )
    return False


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
    memory.evict_before(trading_today())

    outcome = {} if hail else _remembered_vetoes(signals, memory)

    ai_available = bool(run_ai_reasoning and ANTHROPIC_API_KEY)
    if ai_available:
        await _review_candidates(
            signals, outcome, forecast_cache, store, memory=memory, max_reviews=max_reviews,
        )

    if hail:
        return list(signals)

    why = _unreviewed_reason(run_ai_reasoning, max_reviews)
    executable = [
        s for s in signals
        if _clears_ai_gate(
            s, outcome.get(id(s)), memory, require_review=require_review, why=why,
        )
    ]

    logger.info(
        "AI GATE: %d/%d signals cleared for execution", len(executable), len(signals),
    )
    return executable


@dataclass
class CycleContext:
    """State shared by the stages of one ``run_cycle`` pass.

    Built by ``_start_cycle``; each stage reads what earlier stages produced
    and fills in its own fields. Only ``forecast_cache`` (the caller's
    cross-cycle cache, when one is passed) outlives the cycle.
    """

    paper_trader: PaperTrader | None
    live_executor: object | None
    store: object | None
    run_ai_reasoning: bool
    target_dates: list[date]
    min_horizon_hours: int
    now_utc: datetime
    forecast_cache: dict[tuple, list]
    # (city, target_date) keys fetched fresh in this cycle
    refreshed: set[tuple] = field(default_factory=set)
    # (city, target_date, variable) -> GraphCast divergence result
    ai_divergence_cache: dict[tuple, dict | None] = field(default_factory=dict)

    # Discovery
    markets: list[MarketInfo] = field(default_factory=list)
    market_groups: dict[tuple[City, date], list[MarketInfo]] = field(default_factory=dict)
    city_volume: dict[str, dict] = field(default_factory=dict)
    total_equity: float = 0.0
    # (city_id, target_date) -> GraphCast forecast
    ai_forecasts: dict[tuple[str, date], object] = field(default_factory=dict)
    # GribStream's compute_ai_physics_divergence, bound when the fetch imports
    ai_divergence_fn: object | None = None

    # Signals: every candidate after filtering, and the AI-cleared subset
    all_signals: list[Signal] = field(default_factory=list)
    executable_signals: list[Signal] = field(default_factory=list)

    # Execution
    market_maker: object | None = None
    market_prices: dict[str, dict] = field(default_factory=dict)
    market_by_id: dict[str, MarketInfo] = field(default_factory=dict)
    live_balance: float | None = None
    usdc_floor_block: bool = False
    max_positions: int = 0
    active_position_count: int = 0

    @property
    def is_live(self) -> bool:
        """A live executor that places real orders (not a dry run) is attached."""
        return bool(self.live_executor and not self.live_executor.dry_run)

    def live_entries_open(self) -> bool:
        """New live entries allowed: live, above the USDC floor, under the position cap."""
        return bool(
            self.is_live
            and not self.usdc_floor_block
            and self.active_position_count < self.max_positions
        )


def _apply_horizon_filter(
    target_dates: list[date], now_utc: datetime,
) -> tuple[list[date], int]:
    """Drop target dates that resolve too soon; returns (kept, min_horizon_hours)."""
    # --- SWING BOT: 36h horizon filter ---
    # Block entries on markets resolving too soon. We get front-run by bots
    # with fresher NWS data on short-dated markets. Our bias correction edge
    # is strongest at 48-72h where model ensembles still disagree.
    # HAIL MARY mode drops the filter: penny lottery tickets only need to
    # resolve, they don't need to be ahead of the front-running bots.
    target_dates, blocked, min_horizon_hours = filter_dates_by_horizon(
        list(target_dates), now_utc,
    )
    if blocked:
        logger.info(
            "HORIZON FILTER: blocked %s (resolve < %dh)",
            ", ".join(str(d) for d in sorted(blocked)),
            min_horizon_hours,
        )
    return target_dates, min_horizon_hours


def _check_emos_contract() -> None:
    """Contract: verify EMOS calibration is active at cycle start."""
    emos_check = validate_emos_active(
        SPREAD_INFLATION_FACTOR, MAX_BUCKET_PROBABILITY,
        EMOS_VARIANCE_FLOOR_C,
    )
    if not emos_check.valid:
        logger.warning(
            "CONTRACT VIOLATION [%s]: %s",
            emos_check.code, emos_check.error,
        )


async def _resolve_paper_trades(paper_trader: PaperTrader | None) -> None:
    """Resolve any open paper trades before placing new ones."""
    if paper_trader:
        try:
            resolved_count = await resolve_open_trades(paper_trader)
            if resolved_count > 0:
                logger.info("Resolved %d paper trades at cycle start", resolved_count)
        except Exception:
            logger.exception("Paper trade resolution failed, continuing with cycle")


async def _refresh_enso_state() -> None:
    """Refresh ENSO regime state (cached 24h, affects bias correction shrinkage)."""
    try:
        from weather_edge.analysis.enso_regime import fetch_enso_state
        await fetch_enso_state()
    except Exception:
        # Optional: bias correction falls back to its cached / neutral state.
        logger.warning("ENSO state fetch skipped", exc_info=True)


async def _start_cycle(
    paper_trader: PaperTrader | None,
    *,
    target_dates: list[date] | None,
    run_ai_reasoning: bool,
    live_executor,
    store,
    forecast_cache: dict[tuple, list] | None,
) -> CycleContext:
    """Stage 0: resolve the store and dates, run the start-of-cycle checks."""
    # Store can come from paper_trader or be passed directly. A plain
    # in-memory PaperTrader (CLI) has no store; every store use tolerates None.
    if store is None and paper_trader is not None:
        store = getattr(paper_trader, "store", None)

    if target_dates is None:
        target_dates = default_target_dates()

    now_utc = datetime.now(UTC)
    target_dates, min_horizon_hours = _apply_horizon_filter(target_dates, now_utc)
    _check_emos_contract()
    await _resolve_paper_trades(paper_trader)

    _forecast_cache = forecast_cache if forecast_cache is not None else {}
    # Cross-cycle cache: forget target dates that have already passed.
    evict_stale_forecasts(_forecast_cache, trading_today(now_utc))
    ctx = CycleContext(
        paper_trader=paper_trader,
        live_executor=live_executor,
        store=store,
        run_ai_reasoning=run_ai_reasoning,
        target_dates=target_dates,
        min_horizon_hours=min_horizon_hours,
        now_utc=now_utc,
        forecast_cache=_forecast_cache,
    )

    # Check if we're in a golden window (model just updated)
    if is_golden_window():
        logger.info("*** GOLDEN WINDOW: Fresh model data, market may be stale ***")

    await _refresh_enso_state()
    return ctx


async def _sync_market_map(ctx: CycleContext) -> None:
    """Update market_map for position tracking (maps asset_id → city_id)."""
    if ctx.live_executor and ctx.markets:
        try:
            from weather_edge.trading.portfolio_sync import sync_market_map_from_discovery
            mapped = await sync_market_map_from_discovery(ctx.store, ctx.markets)
            if mapped:
                logger.info("Market map updated: %d token mappings", mapped)
        except Exception:
            # Optional: positions just keep their previous city mapping.
            logger.warning("Market map update failed", exc_info=True)


async def _reconcile_portfolio(ctx: CycleContext) -> None:
    """PORTFOLIO SYNC: reconcile with exchange truth; sets ``ctx.total_equity``.

    Runs AFTER market_map so positions get city_id mapping.
    """
    ctx.total_equity = settings.bankroll
    if ctx.is_live:
        live_executor, store = ctx.live_executor, ctx.store
        try:
            from weather_edge.trading.portfolio_sync import fetch_polymarket_state, sync_portfolio
            await sync_portfolio(executor=live_executor, store=store)
            # Fetch full state to get balance + current market value of positions
            state = await fetch_polymarket_state(live_executor, live_executor.wallet_address)
            ctx.total_equity = state["portfolio_value"]
            logger.info(
                "PORTFOLIO EQUITY: $%.2f (used as bankroll for Kelly sizing)", ctx.total_equity,
            )

            # Rebuild positions again to pick up market_map city_ids
            store.rebuild_positions()
        except Exception:
            logger.exception("Portfolio sync failed, continuing with stale positions")


def _aggregate_city_volume(markets: list[MarketInfo]) -> dict[str, dict]:
    """Aggregate volume and liquidity by city for the dashboard."""
    city_volume: dict[str, dict] = {}
    for m in markets:
        if m.city_id:
            cid = m.city_id.value
            if cid not in city_volume:
                city_volume[cid] = {"volume_24h": 0.0, "liquidity": 0.0, "markets": 0}
            city_volume[cid]["volume_24h"] += m.volume_24h or 0
            city_volume[cid]["liquidity"] += m.liquidity or 0
            city_volume[cid]["markets"] += 1
    return city_volume


def _log_parity_arbitrage(markets: list[MarketInfo]) -> None:
    """Step 1b: check bucket parity for arbitrage opportunities (log only)."""
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


def _group_markets(
    markets: list[MarketInfo], target_dates: list[date],
) -> dict[tuple[City, date], list[MarketInfo]]:
    """Group markets by city+date, keeping only the cycle's target dates."""
    market_groups: dict[tuple[City, date], list[MarketInfo]] = {}
    for m in markets:
        if m.city_id and m.target_date in target_dates:
            key = (m.city_id, m.target_date)
            market_groups.setdefault(key, []).append(m)
    return market_groups


async def _discover_markets(ctx: CycleContext) -> None:
    """Stage 1: discover markets, sync the portfolio, group markets by city-date."""
    # Step 1: Discover active weather markets (prices included from Gamma API)
    logger.info("=== Discovering Polymarket weather markets ===")
    ctx.markets = await discover_weather_markets()
    await _sync_market_map(ctx)
    await _reconcile_portfolio(ctx)
    ctx.city_volume = _aggregate_city_volume(ctx.markets)
    _log_parity_arbitrage(ctx.markets)

    if not ctx.markets:
        logger.warning("No weather markets found for tracked cities.")

    ctx.market_groups = _group_markets(ctx.markets, ctx.target_dates)
    logger.info("Active market groups: %d city-date combos", len(ctx.market_groups))


async def _fetch_ai_forecasts_for_date(
    ctx: CycleContext, ai_target: date, fetch_ai_forecasts_batch,
) -> None:
    """GraphCast forecasts for every city with markets on ``ai_target``."""
    market_cities = sorted(
        {c for c, d in ctx.market_groups if d == ai_target}, key=lambda c: c.value,
    )
    try:
        ai_batch = await fetch_ai_forecasts_batch(market_cities, ai_target)
    except Exception:
        # Optional: GraphCast only adjusts confidence; the cycle runs without it.
        logger.warning("GribStream AI fetch failed for %s", ai_target, exc_info=True)
        return
    for cid, fc in (ai_batch or {}).items():
        ctx.ai_forecasts[(getattr(cid, "value", cid), ai_target)] = fc
    if ai_batch:
        logger.info(
            "GraphCast: %d city forecasts fetched for %s", len(ai_batch), ai_target,
        )


async def _fetch_ai_forecasts(ctx: CycleContext) -> None:
    """Step 1c: fetch AI model forecasts (GraphCast via GribStream) for comparison.

    Keyed (city_id, target_date): a GraphCast run for one date must never be
    compared against the physics consensus for another. The divergence
    function is bound here, with the fetch, so one failed import disables both.
    """
    try:
        from weather_edge.fetchers.gribstream import (
            compute_ai_physics_divergence,
            fetch_ai_forecasts_batch,
        )
        ctx.ai_divergence_fn = compute_ai_physics_divergence
        for ai_target in sorted({d for _, d in ctx.market_groups}):
            await _fetch_ai_forecasts_for_date(ctx, ai_target, fetch_ai_forecasts_batch)
    except ImportError:
        # Per-date fetch failures are handled in _fetch_ai_forecasts_for_date.
        logger.warning("GribStream AI fetch skipped", exc_info=True)


@dataclass
class SignalGroup:
    """One (city, target date) market group with its forecasts and pattern adjustment."""

    city_id: City
    target_date: date
    markets: list[MarketInfo]
    forecasts: list
    pattern_conf_mult: float
    pattern_bias: float


def _save_forecast_snapshot(store, city_id: City, target_date: date, forecasts: list) -> None:
    """Persist a forecast snapshot for self-learning (best effort)."""
    if store is None:
        return
    try:
        m_vals = {f.model_name: f.temp_max_c for f in forecasts if f.temp_max_c is not None}
        if m_vals:
            store.save_forecast_snapshot(
                city_id.value, str(target_date), m_vals,
            )
    except Exception:
        # Don't break the pipeline if persistence fails
        logger.warning("Failed to save forecast snapshot for %s on %s",
                       city_id.value, target_date, exc_info=True)


def _fresh_cached_forecasts(ctx: CycleContext, city_id: City, target_date: date) -> list | None:
    """Cached forecasts to fall back on after a failed fetch, if young enough."""
    cached = ctx.forecast_cache.get((city_id, target_date))
    max_stale_h = float(getattr(
        settings, "max_stale_forecast_hours", DEFAULT_MAX_STALE_FORECAST_HOURS,
    ))
    age_h = _forecast_age_hours(cached, ctx.now_utc) if cached else None
    if cached and age_h is not None and age_h <= max_stale_h:
        logger.warning(
            "STALE DATA: fetch failed for %s on %s, using cached data from %s "
            "(%.1fh old)",
            city_id.value, target_date,
            cached[0].fetched_at.strftime("%H:%M:%S"), age_h,
        )
        return cached
    if cached:
        logger.warning(
            "STALE DATA REJECTED: cached forecast for %s on %s is too old "
            "(%s h > %.1f h), skipping",
            city_id.value, target_date,
            f"{age_h:.1f}" if age_h is not None else "unknown", max_stale_h,
        )
    return None


async def _group_forecasts(ctx: CycleContext, city_id: City, target_date: date) -> list:
    """Fetch multi-model forecasts for a group, falling back to a fresh-enough cache."""
    forecasts = await fetch_city_forecasts(city_id, target_date)
    if forecasts:
        ctx.forecast_cache[(city_id, target_date)] = forecasts
        ctx.refreshed.add((city_id, target_date))
        _save_forecast_snapshot(ctx.store, city_id, target_date, forecasts)
        return forecasts
    # Check if we have stale data in cache
    return _fresh_cached_forecasts(ctx, city_id, target_date) or forecasts


def _variables_needed(markets: list[MarketInfo]) -> list[str]:
    """Consensus variables the group's markets need (temp_max_c always), in a fixed order.

    A set's iteration order depends on the hash seed, which made the order of
    signals (and log lines) for multi-variable groups vary run to run. Order
    is MARKET_TYPE_TO_VARIABLE's (temp_max_c first), then any others sorted.
    """
    variables_needed = {"temp_max_c"}  # Always compute
    for m in markets:
        var = get_required_variable(m)
        if var:
            variables_needed.add(var)
    canonical = list(MARKET_TYPE_TO_VARIABLE.values())
    return sorted(
        variables_needed,
        key=lambda v: (canonical.index(v) if v in canonical else len(canonical), v),
    )


def _collect_thresholds(markets: list[MarketInfo], variable: str) -> list[float]:
    """Thresholds from all markets needing ``variable``."""
    thresholds = []
    for m in markets:
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
    return thresholds


def _passes_model_agreement(city_id: City, variable: str, consensus) -> bool:
    """MODEL AGREEMENT GATE on the raw model spread.

    Per README: Skip if models disagree (std > 2.0C). This prevents
    trading on noise when ensembles are in chaos. HAIL MARY mode lets
    every signal through: penny tickets are cheap enough that noisy
    bets beat missed ones.
    Applied to the RAW model spread, see _model_agreement_std.
    """
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
            return False
    return True


def _apply_pattern_bias(group: SignalGroup, variable: str, consensus) -> None:
    """Apply pattern-based bias shift (e.g. haze suppression) to the consensus."""
    pattern_bias = group.pattern_bias
    if pattern_bias != 0:
        old_mean = consensus.weighted_mean
        consensus.weighted_mean += pattern_bias
        consensus.mean_value += pattern_bias
        logger.info(
            "  %s/%s PATTERN SHIFT: %.1f°C -> %.1f°C (%+.1f°C)",
            group.city_id.value, variable, old_mean, consensus.weighted_mean, pattern_bias
        )


def _trend_multiplier(city_id: City, variable: str, consensus) -> float:
    """Track forecast trends (run-to-run consistency); returns the confidence multiplier."""
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
        # Optional: trend history lives in Redis / on disk; neutral on failure.
        logger.warning("Forecast trend skipped for %s", city_id.value, exc_info=True)
    return trend_mult


def _ai_divergence(
    ctx: CycleContext, group: SignalGroup, variable: str, consensus,
) -> dict | None:
    """AI vs physics divergence (GraphCast comparison), computed once per city-date-variable."""
    city_id, target_date = group.city_id, group.target_date
    _ai_div_key = (city_id.value, target_date, variable)
    if _ai_div_key not in ctx.ai_divergence_cache:
        ai_fc = ctx.ai_forecasts.get((city_id.value, target_date))
        if ai_fc and variable == "temp_max_c":
            try:
                div = ctx.ai_divergence_fn(
                    ai_fc, consensus.weighted_mean,
                )
                ctx.ai_divergence_cache[_ai_div_key] = div
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
                # Optional: a bad GraphCast payload just means no adjustment.
                logger.warning("AI divergence failed for %s", city_id.value, exc_info=True)
                ctx.ai_divergence_cache[_ai_div_key] = None

    return ctx.ai_divergence_cache.get(_ai_div_key)


def _bucket_center(market: MarketInfo) -> float | None:
    """°C centre of a range bucket, or the threshold of a gte/lte bucket."""
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
    return bucket_center


def _zscore_rejects(signal: Signal, market: MarketInfo, consensus, city_id: City) -> bool:
    """Z-score guard: reject core bets on buckets too far from consensus.

    Tail/penny bets (entry <=6c) are exempt, they're designed as
    lottery tickets. HAIL MARY mode skips the guard: penny-only
    baskets are far-from-consensus by design.
    """
    market_prob = market.yes_price
    if (
        not settings.hail_mary_mode
        and signal.strategy != "tail"
        and market_prob > 0.06
    ):
        bucket_center = _bucket_center(market)
        if bucket_center is not None and consensus.std_dev > 0:
            zscore = abs(bucket_center - consensus.weighted_mean) / consensus.std_dev
            if zscore > settings.max_core_zscore:
                logger.info(
                    "ZSCORE REJECT: %s %s, bucket=%.1f°C mean=%.1f°C std=%.1f z=%.1f (max=%.1f)",
                    city_id.value, market.question[:40],
                    bucket_center, consensus.weighted_mean,
                    consensus.std_dev, zscore, settings.max_core_zscore,
                )
                return True
    return False


def _signal_for_market(
    ctx: CycleContext, group: SignalGroup, market: MarketInfo, *,
    variable: str, consensus, trend_mult: float,
) -> Signal | None:
    """Edge signal for one market bucket, or None if it is skipped or rejected."""
    city_id, target_date = group.city_id, group.target_date
    if get_required_variable(market) != variable:
        return None

    # Use price from Gamma API
    market_prob = market.yes_price
    if market_prob <= 0.01 or market_prob >= 0.99:
        return None  # Skip extreme prices (no edge possible)

    model_prob = compute_model_prob_for_market(market, consensus)
    if model_prob is None:
        return None

    # Hours to resolution: end of the target date in the city's
    # local time, not 00:00 UTC.
    hours_to = hours_to_local_end_of_day(city_id, market.target_date)
    # The date-level horizon filter above assumes UTC midnight;
    # enforce it per city too (Asia/Pacific resolve earlier).
    if hours_to < ctx.min_horizon_hours:
        return None

    # Apply pattern-based confidence boost + forecast trend stability
    adjusted_conf = min(1.0, consensus.confidence * group.pattern_conf_mult * trend_mult)

    ai_div = _ai_divergence(ctx, group, variable, consensus)
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
        bankroll=ctx.total_equity,
        consensus_id=None,
        hours_to_resolution=hours_to,
        city_id=city_id.value,
        target_date=str(target_date),
        description=market.question[:80],
        spread=estimated_spread,
        min_edge_yes=risk_profile.min_edge_yes,
        min_edge_no=risk_profile.min_edge_no,
    )

    if _zscore_rejects(signal, market, consensus, city_id):
        return None
    return signal


def _signals_for_variable(ctx: CycleContext, group: SignalGroup, variable: str) -> None:
    """Consensus for one variable, then a signal per matching market bucket."""
    city_id = group.city_id
    thresholds = _collect_thresholds(group.markets, variable)
    consensus = compute_consensus(
        city_id, str(group.target_date), variable, group.forecasts,
        sorted(set(thresholds)) if thresholds else None,
    )
    if consensus is None:
        return
    if not _passes_model_agreement(city_id, variable, consensus):
        return

    _apply_pattern_bias(group, variable, consensus)
    trend_mult = _trend_multiplier(city_id, variable, consensus)

    logger.info(
        "  %s/%s: mean=%.1f°C std=%.1f conf=%.0f%% (%d models)",
        city_id.value, variable,
        consensus.weighted_mean, consensus.std_dev,
        consensus.confidence * 100, consensus.model_count,
    )

    # Compute edge for each matching market bucket
    for market in group.markets:
        signal = _signal_for_market(
            ctx, group, market, variable=variable, consensus=consensus, trend_mult=trend_mult,
        )
        if signal is not None:
            ctx.all_signals.append(signal)


async def _signals_for_group(
    ctx: CycleContext, city_config, city_id: City, target_date: date,
    city_markets: list[MarketInfo],
) -> None:
    """Forecasts, contract check and pattern adjustment for one (city, date) group."""
    forecasts = await _group_forecasts(ctx, city_id, target_date)
    if not forecasts:
        logger.warning("No forecasts for %s on %s", city_id.value, target_date)
        return

    # Contract: verify sufficient models for reliable consensus
    has_regional = bool(city_config.regional_models)
    model_check = validate_model_count(len(forecasts), has_regional)
    if not model_check.valid:
        logger.warning(
            "CONTRACT VIOLATION [%s]: %s, skipping %s on %s",
            model_check.code, model_check.error, city_id.value, target_date,
        )
        return

    # Detect bust-causing weather patterns (Chinook, Foehn, marine layer, etc.)
    pattern_alerts = detect_patterns(city_id, forecasts)
    pattern_conf_mult, pattern_bias = get_pattern_adjustment(city_id, pattern_alerts)
    group = SignalGroup(
        city_id=city_id, target_date=target_date, markets=city_markets,
        forecasts=forecasts, pattern_conf_mult=pattern_conf_mult, pattern_bias=pattern_bias,
    )

    # Compute consensus per variable
    for variable in _variables_needed(city_markets):
        _signals_for_variable(ctx, group, variable)


async def _compute_signals(ctx: CycleContext) -> None:
    """Stage 2: for each city with markets, fetch forecasts and compute signals."""
    # market_groups keys are unique (city, date) pairs: one header per group.
    for (city_id, target_date), city_markets in ctx.market_groups.items():
        city_config = CITIES[city_id]
        logger.info("=== %s (%s) on %s, %d markets ===",
                    city_config.name, city_config.icao, target_date, len(city_markets))
        await _signals_for_group(ctx, city_config, city_id, target_date, city_markets)


def _hail_mary_entry_price(s: Signal) -> float:
    """Price of the token a signal would buy (YES price, or 1 - YES for NO)."""
    try:
        if s.recommended_side.value == "YES":
            entry_price = s.market_prob
        else:
            entry_price = 1.0 - s.market_prob
    except AttributeError:  # no recommended side
        entry_price = s.market_prob
    return entry_price


def _hail_mary_basket(all_signals: list[Signal]) -> list[Signal]:
    """HAIL MARY: penny lottery basket.

    The normal strategy filters to one signal per city-date. The hail-mary
    takes EVERY cheap signal, multi-bucket stacking is fine when each
    ticket costs $1 because the math is "buy 30 lottery tickets, hope 1 hits."
    """
    hailmary_max_price = 0.10   # only buy stuff priced under 10¢
    filtered_signals: list[Signal] = []
    for s in all_signals:
        # Penny ceiling, anything above 10¢ is "more of the same bs"
        entry_price = _hail_mary_entry_price(s)
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
    return filtered_signals


def _signal_score(sig: Signal) -> float:
    """Strategy priority: tail_no > tail > core; within a strategy, highest |net_edge|."""
    prio = {"tail_no": 300, "tail": 200, "core": 100}.get(sig.strategy, 0)
    return prio + abs(sig.net_edge)


def _best_signal_per_city_date(all_signals: list[Signal]) -> list[Signal]:
    """STRATEGY REDESIGN: one signal per city-date.

    Prevents "multi-bucket bleed" by picking only the highest-edge bucket.
    Prioritizes tail_no (high-prob NO) and tail (penny) strategies.
    """
    signal_groups: dict[tuple[str, str], list[Signal]] = {}
    for s in all_signals:
        if s.confidence_tier.value == "low":
            continue
        key = (s.city_id, s.target_date)
        signal_groups.setdefault(key, []).append(s)

    filtered_signals: list[Signal] = []
    for key, group in signal_groups.items():
        best_signal = max(group, key=_signal_score)
        filtered_signals.append(best_signal)

        if len(group) > 1:
            logger.info(
                "MULTI-BUCKET FILTER: %s %s picked %s (edge=%.1f%%) over %d others",
                key[0], key[1], best_signal.strategy,
                best_signal.net_edge * 100, len(group) - 1,
            )
    return filtered_signals


def _apply_cycle_limit(filtered_signals: list[Signal]) -> list[Signal]:
    """Global max trades per cycle: keep the top N by absolute net_edge."""
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
    return filtered_signals


def _filter_signals(all_signals: list[Signal]) -> list[Signal]:
    """Stage 3: reduce raw signals to the cycle's candidates (hail-mary or normal)."""
    if settings.hail_mary_mode:
        return _hail_mary_basket(all_signals)
    if all_signals:
        return _apply_cycle_limit(_best_signal_per_city_date(all_signals))
    return []


async def _fetch_book_for_signal(ctx: CycleContext, signal: Signal, fetch_book_prices) -> None:
    """Record real order-book asks for one signal's market in ``ctx.market_prices``."""
    m = ctx.market_by_id.get(signal.market_id)
    if m and m.token_id_yes and m.token_id_no:
        try:
            book = await asyncio.wait_for(fetch_book_prices(m), timeout=10.0)
        except Exception as e:  # includes TimeoutError
            # Optional: without a book the signal trades on Gamma prices.
            logger.warning("Book fetch timeout/error for %s: %s", signal.city_id, e,
                           exc_info=True)
            return
        if book:
            ctx.market_prices[signal.market_id] = {
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


async def _fetch_book_prices(ctx: CycleContext, fetch_book_prices) -> None:
    """Fetch book prices for the top 5 signals only (each = 2 CLOB API calls)."""
    logger.info("Fetching order book prices for spread detection...")
    top_signals = sorted(
        ctx.executable_signals, key=lambda s: abs(s.edge), reverse=True,
    )[:5]
    for signal in top_signals:
        await _fetch_book_for_signal(ctx, signal, fetch_book_prices)
    logger.info("Book price fetch complete, placing trades...")


async def _fetch_live_balance(ctx: CycleContext) -> float | None:
    """Fetch the live USDC balance once before the order loop (None if unknown)."""
    _live_balance: float | None = None
    if ctx.is_live:
        try:
            _live_balance = await ctx.live_executor.check_balance()
            logger.info("USDC balance: $%.2f", _live_balance or 0)
        except Exception:
            logger.warning("Failed to fetch live balance, will skip balance checks",
                           exc_info=True)
    return _live_balance


def _usdc_floor_blocks(ctx: CycleContext) -> bool:
    """SWING BOT: USDC floor.

    Don't place ANY new live orders if balance is below $20. Prevents
    deploying dust into marginal trades. Capital comes back via
    resolution/redemption, then we re-enter on high-conviction 48h+ signals.
    HAIL MARY disables the floor ($1 tickets down to the last dollar).
    """
    usdc_floor = 0.0 if settings.hail_mary_mode else 20.0
    _usdc_floor_block = bool(
        ctx.is_live
        and ctx.live_balance is not None
        and ctx.live_balance < usdc_floor
    )
    if _usdc_floor_block:
        logger.warning(
            "USDC FLOOR: $%.2f < $%.0f minimum, blocking all new live entries",
            ctx.live_balance, usdc_floor,
        )
    return _usdc_floor_block


# Kept as a module attribute so tests can patch the scheduler's fetch.
_fetch_all_data_api_positions = fetch_all_data_api_positions


async def _clean_resolved_positions(store, live_executor) -> None:
    """Zero DB positions the exchange no longer reports as held.

    rebuild_positions() re-creates them from fills every cycle, so this must
    run every time before the position count is read. Only a complete
    position list is trusted: on any failed or possibly truncated fetch
    nothing is zeroed.
    """
    import httpx
    wallet = (live_executor.wallet_address or "").lower()
    async with httpx.AsyncClient() as client:
        api_positions = await _fetch_all_data_api_positions(client, wallet)
    if api_positions is None:
        logger.warning("POSITION CLEANUP skipped: Data API position list incomplete")
        return
    active_cids = {
        p.get("conditionId")
        for p in api_positions
        if float(p.get("size", 0)) > 0
    }
    db_pos = store.conn.execute(
        "SELECT condition_id FROM positions WHERE total_shares > 0"
    ).fetchall()
    cleaned = 0
    for row in db_pos:
        if row["condition_id"] not in active_cids:
            store.conn.execute(
                "UPDATE positions SET total_shares = 0 WHERE condition_id = ?",
                (row["condition_id"],),
            )
            cleaned += 1
    if cleaned:
        store.commit()


async def _count_active_positions(ctx: CycleContext) -> int:
    """Live positions currently held, after cleaning resolved ones (0 when not live)."""
    store = ctx.store
    _active_position_count = 0
    if store and ctx.is_live:
        try:
            await _clean_resolved_positions(store, ctx.live_executor)

            _active_position_count = store.get_portfolio_summary().get("position_count", 0)
            if _active_position_count >= ctx.max_positions:
                logger.warning(
                    "POSITION CAP: %d/%d active positions, blocking new live entries",
                    _active_position_count, ctx.max_positions,
                )
        except Exception:
            # Cleanup is best effort; fall back to the uncleaned DB count.
            logger.warning("Position cleanup failed, using the DB position count",
                           exc_info=True)
            _active_position_count = store.get_portfolio_summary().get("position_count", 0)
    return _active_position_count


async def _prepare_execution(ctx: CycleContext) -> None:
    """Stage 5: book prices, live balance, USDC floor and position cap."""
    # Place trades for all signals + generate spread capture orders
    from weather_edge.fetchers.polymarket import fetch_book_prices
    from weather_edge.trading.market_maker import MarketMaker
    ctx.market_maker = MarketMaker()

    # Build market prices dict using real order book asks (not midpoints)
    # Per Gemini: spread only exists if YES_ask + NO_ask < 1.00
    ctx.market_prices = {}
    ctx.market_by_id = {m.market_id: m for m in ctx.markets}
    await _fetch_book_prices(ctx, fetch_book_prices)

    ctx.live_balance = await _fetch_live_balance(ctx)
    ctx.usdc_floor_block = _usdc_floor_blocks(ctx)

    # --- SWING BOT: Position cap ---
    # 50 allows new entries alongside existing positions; most existing
    # positions are small/penny bets that resolve naturally. Effective
    # concentration is managed by the min size + USDC floor. HAIL MARY
    # removes the cap for the penny basket.
    ctx.max_positions = 999 if settings.hail_mary_mode else 50
    ctx.active_position_count = await _count_active_positions(ctx)

    live_executor = ctx.live_executor
    logger.info(
        "EXECUTION LOOP: %d signals, live_executor=%s, dry_run=%s, "
        "usdc_floor_block=%s, pos_count=%d/%d",
        len(ctx.executable_signals),
        bool(live_executor),
        live_executor.dry_run if live_executor else "N/A",
        ctx.usdc_floor_block,
        ctx.active_position_count,
        ctx.max_positions,
    )


def _place_paper_entry(ctx: CycleContext, signal: Signal, fee_blocked: bool) -> None:
    """Paper trade for ``signal`` plus a paper spread hedge when the book allows one."""
    paper_trader = ctx.paper_trader
    trade = paper_trader.place_trade(signal) if (paper_trader and not fee_blocked) else None
    # Generate hedge/spread order only for core trades (not penny bets)
    # Penny bets at 0.1-5c: max loss is the entry cost, hedging is wasteful
    if trade:
        strategy = getattr(signal, "strategy", "core")
        if strategy != "tail":
            prices = ctx.market_prices.get(signal.market_id, {})
            if prices.get("spread_profitable"):
                hedge = ctx.market_maker.generate_hedge_orders(
                    signal, ctx.market_prices, settings.bankroll,
                )
                if hedge and paper_trader:
                    paper_trader.place_spread_trade(signal, hedge)


def _in_exit_cooldown(ctx: CycleContext, signal: Signal) -> bool:
    """Cooldown: don't re-enter a market we recently fully exited.

    Sell-half excluded (still holding shares, POSITION EXISTS catches it).
    Massive edge (>12%) bypasses cooldown for genuine model shifts.
    """
    store = ctx.store
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
                return True
            else:
                logger.warning(
                    "COOLDOWN BYPASS: %s, massive edge %.1f%% overrules 4h window",
                    signal.city_id, signal.edge * 100,
                )
    return False


def _size_hail_mary_ticket(ctx: CycleContext, signal: Signal) -> bool:
    """HAIL MARY: fixed $1.00 lottery-ticket sizing. False if we can't afford it.

    Override whatever Kelly/Claude/Gemini decided. Every order is
    exactly $1.00, penny basket, not concentrated bets.
    """
    signal.recommended_size = HAILMARY_TICKET_USD

    # Margin check, only block if we don't even have $1
    if ctx.live_balance is not None and ctx.live_balance < HAILMARY_TICKET_USD:
        logger.info(
            "HAILMARY OUT OF CASH: balance $%.2f < $%.2f, skipping %s",
            ctx.live_balance, HAILMARY_TICKET_USD, signal.city_id,
        )
        return False
    return True


def _size_swing_entry(ctx: CycleContext, signal: Signal) -> bool:
    """SWING BOT: minimum position size, then the live balance. False to skip."""
    # Survival tier: $5 min until bankroll > $500, then raise to $10.
    # Only genuinely small positive sizes are bumped; a zero size
    # (AI veto / Kelly says no) was already dropped above.
    if signal.recommended_size <= 0:
        return False
    signal.recommended_size = max(signal.recommended_size, MIN_LIVE_SIZE)

    # Margin check, use tracked live balance, not stale portfolio
    # calc. Runs AFTER min size bump so the floor is respected.
    if ctx.live_balance is not None and signal.recommended_size > ctx.live_balance:
        logger.info(
            "BALANCE LIMIT: %s needs $%.0f but exchange balance is $%.2f, skipping",
            signal.city_id, signal.recommended_size, ctx.live_balance,
        )
        return False
    return True


def _live_edge_ok(signal: Signal) -> bool:
    """Minimum 2% live edge; HAIL MARY takes any positive edge. Spread legs always pass."""
    is_spread = getattr(signal, "strategy", "") == "spread"
    if settings.hail_mary_mode:
        can_live = signal.edge > 0 or is_spread
    else:
        can_live = signal.edge >= 0.02 or is_spread
    if not can_live:
        logger.debug(
            "LIVE SKIP: %s edge=%.3f size=$%.0f (need ≥2%% edge)",
            signal.city_id, signal.edge, signal.recommended_size,
        )
    return can_live


@dataclass
class _RestingEntry:
    """An open entry order already on the book for a signal's market."""

    order: dict
    old_price: float
    filled: float
    new_price: float  # what our new limit price would be
    age_minutes: float
    price_drift: float


def _order_age_minutes(existing: dict) -> float:
    """Minutes since an order was placed (0 when unknown); drives the price chase."""
    order_age_minutes = 0
    placed_str = existing.get("placed_at", "")
    if placed_str:
        try:
            placed_dt = datetime.fromisoformat(placed_str)
            order_age_minutes = (
                datetime.now(UTC) - placed_dt
            ).total_seconds() / 60
        except (ValueError, TypeError):
            pass
    return order_age_minutes


def _resting_entry(existing: dict, signal: Signal) -> _RestingEntry:
    """Describe the open order ``existing`` against the price ``signal`` would use."""
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
    order_age_minutes = _order_age_minutes(existing)
    price_drift = abs(new_price - old_price)
    return _RestingEntry(
        order=existing, old_price=old_price, filled=filled, new_price=new_price,
        age_minutes=order_age_minutes, price_drift=price_drift,
    )


def _chase_limit(signal: Signal, resting: _RestingEntry) -> float | None:
    """One-tick price-chase limit for a stale unfilled order, or None to keep it.

    Price chase: improve by ONE TICK toward the midpoint. The executor
    rounds limits to the 1c tick, so a sub-tick bump would re-place at the
    same price (cancel for nothing, losing queue priority). Never chase
    above the side's midpoint (would cross as a post-only buy).
    """
    side_mid = (
        signal.market_prob
        if signal.recommended_side.value == "YES"
        else 1.0 - signal.market_prob
    )
    from weather_edge.trading.executor import chase_limit_price
    chase_price = chase_limit_price(resting.old_price, side_mid)
    if chase_price is None:
        logger.info(
            "PRICE CHASE SKIP: %s @ %.2f already at "
            "mid cap %.3f, keeping order",
            signal.city_id, resting.old_price, side_mid,
        )
    return chase_price


async def _cancel_and_record(ctx: CycleContext, order_id: str, fail_msg: str) -> bool:
    """Cancel ``order_id`` on the exchange; mark it cancelled only if that worked.

    cancel_order returns False (or raises) when the order may still be
    resting, including when live orders are suppressed. Callers must not
    place a replacement then, or two orders would rest on the same market.
    """
    try:
        cancelled = await ctx.live_executor.cancel_order(order_id)
    except Exception as e:  # noqa: BLE001 - one order; the cycle continues
        logger.warning(fail_msg, order_id[:16], e, exc_info=True)
        return False
    if not cancelled:
        logger.warning(fail_msg, order_id[:16], "cancel_order returned False")
        return False
    ctx.store.cancel_live_trade(order_id)
    return True


async def _chase_resting_entry(
    ctx: CycleContext, signal: Signal, resting: _RestingEntry, chase_price: float,
) -> float | None:
    """Cancel the stale order; return the YES-equivalent market_prob to re-place at.

    None if the old order could not be cancelled (keep it, place nothing).

    ``signal.market_prob`` is left alone: it is the observed market price the
    dashboard shows. Only the order placed for this signal uses the chased one.
    """
    old_id = resting.order.get("order_id")
    if old_id:
        if not await _cancel_and_record(ctx, old_id, "Price chase cancel failed %s: %s"):
            return None
        logger.info(
            "PRICE CHASE: %s improving "
            "%.3f→%.3f after %dm unfilled",
            signal.city_id, resting.old_price,
            chase_price,
            int(resting.age_minutes),
        )
    # Fall through to place new order at chased price. place_limit_order
    # improves market_prob by half a tick, so aim half a tick above.
    if signal.recommended_side.value == "YES":
        return chase_price + 0.005
    return 1.0 - (chase_price + 0.005)


async def _replace_resting_entry(
    ctx: CycleContext, signal: Signal, resting: _RestingEntry,
) -> bool:
    """Price drifted: cancel the old order so a new one can be placed.

    False if the old order could not be cancelled (place nothing).
    """
    old_id = resting.order.get("order_id")
    if not old_id:
        return True
    if not await _cancel_and_record(ctx, old_id, "Failed to cancel old order %s: %s"):
        return False
    logger.info(
        "LIVE REPLACE: %s cancelled "
        "%s (price %.3f→%.3f, "
        "drift=%.3f)",
        signal.city_id, old_id[:16],
        resting.old_price, resting.new_price,
        resting.price_drift,
    )
    return True


async def _reprice_resting_entry(
    ctx: CycleContext, signal: Signal, existing: dict,
) -> tuple[bool, float | None]:
    """Chase, keep or replace an open entry order.

    Returns (place a new order?, chased market_prob for that order or None).
    """
    resting = _resting_entry(existing, signal)
    should_chase = (
        resting.age_minutes > 60
        and resting.filled == 0
        and signal.edge >= 0.02
    )

    chase_price = None
    if should_chase:
        chase_price = _chase_limit(signal, resting)
        if chase_price is None:
            return False, None
    if chase_price is not None:
        chased = await _chase_resting_entry(ctx, signal, resting, chase_price)
        return chased is not None, chased
    if resting.price_drift <= 0.001:
        # Price unchanged, keep existing order, preserve queue priority
        logger.info(
            "LIVE KEEP: %s @ %.3f "
            "(age=%dm, drift=%.4f, filled=%.0f)",
            signal.city_id, resting.old_price,
            int(resting.age_minutes),
            resting.price_drift, resting.filled,
        )
        return False, None
    # Price drifted, cancel old, place new
    return await _replace_resting_entry(ctx, signal, resting), None


async def _reconcile_existing_entry(
    ctx: CycleContext, signal: Signal,
) -> tuple[bool, float | None]:
    """POSITION-AWARE DUPLICATE PREVENTION.

    Returns (place a new order?, chased market_prob for that order or None).
    """
    store = ctx.store
    if store is None:
        # No store: nothing to reconcile against. Treat as no position and
        # no resting order rather than crashing the cycle.
        logger.warning(
            "NO STORE: %s, skipping position/open-order duplicate check",
            signal.city_id,
        )
        return True, None
    # Check POSITIONS (what we actually hold) not orders
    existing_position = store.get_position_for_market(signal.market_id)
    if existing_position and existing_position.get("total_shares", 0) > 0:
        logger.info(
            "POSITION EXISTS: %s already hold %.0f shares ($%.2f), skipping",
            signal.city_id,
            existing_position["total_shares"],
            existing_position.get("cost_basis", 0),
        )
        return False, None

    # Also check open orders (not yet filled)
    existing = store.get_open_order_for_market(signal.market_id)
    if existing:
        return await _reprice_resting_entry(ctx, signal, existing)
    return True, None


def _open_positions_for_risk(store) -> list:
    """Stored positions as Position objects for the portfolio risk checks."""
    from weather_edge.models.position import Position, normalize_side

    if store is None:
        return []
    raw_pos = store.get_positions()
    return [
        Position(
            market_id=p.get("condition_id"),
            city_id=p.get("city_id"),
            side=normalize_side(p.get("outcome") or p.get("side")),
            size_usd=p.get("cost_basis", 0),
            status="open" if p.get("total_shares", 0) > 0 else "closed"
        ) for p in raw_pos
    ]


def _apply_risk_limit(signal: Signal, check: tuple[bool, float, str]) -> bool:
    """Apply one (allowed, max_size, reason) risk check to ``signal``. False = blocked."""
    allowed, new_size, reason = check
    if not allowed:
        logger.warning(reason)
        return False
    if new_size < signal.recommended_size:
        logger.info(reason)
        signal.recommended_size = new_size
    return True


def _apply_live_circuit_breaker(signal: Signal) -> bool:
    """Live drawdown circuit breaker. False when trading is killed.

    Fed real NAV by fetch_polymarket_state; a kill also sets the
    persistent kill switch.
    """
    from weather_edge.analysis.risk_controls import live_circuit_breaker_multiplier

    cb_mult = live_circuit_breaker_multiplier()
    if cb_mult <= 0:
        logger.warning(
            "LIVE CIRCUIT BREAKER: killed, skipping %s",
            signal.city_id,
        )
        return False
    if cb_mult < 1.0:
        signal.recommended_size = round(
            signal.recommended_size * cb_mult, 2,
        )
    return True


def _passes_live_risk_controls(ctx: CycleContext, signal: Signal) -> bool:
    """RISK CONTROL CHECKS, trimming ``signal.recommended_size``. False = don't place."""
    if settings.hail_mary_mode:
        # --- HAIL MARY: all risk control checks bypassed ---
        # yes_exposure_cap, correlation_limit, gross_exposure all
        # skipped. Sizing is fixed $1, max blast radius per
        # ticket is $1.
        signal.recommended_size = HAILMARY_TICKET_USD
        return True

    from weather_edge.analysis.risk_controls import (
        check_correlation_limit,
        check_gross_exposure,
        check_yes_exposure_limit,
        get_active_profile,
    )
    profile = get_active_profile()
    total_equity = ctx.total_equity

    # 0. Live drawdown circuit breaker
    if not _apply_live_circuit_breaker(signal):
        return False

    # 1. Yes Exposure Cap
    if signal.recommended_side.value == "YES":
        open_pos = _open_positions_for_risk(ctx.store)
        if not _apply_risk_limit(signal, check_yes_exposure_limit(
            signal.recommended_size, open_pos, total_equity, profile
        )):
            return False

    # 2. Correlation Limit
    open_pos = _open_positions_for_risk(ctx.store)
    if not _apply_risk_limit(signal, check_correlation_limit(
        signal.city_id, signal.recommended_size, open_pos, total_equity, profile
    )):
        return False

    # 3. Gross Exposure
    # Use real position value (total_equity - cash) not inflated
    # DB cost_basis. DB cost_basis includes resolved positions;
    # Polymarket value is truth.
    # With the built-in profiles this cannot bind today: every city is in a
    # correlation group, so size <= max_group_exposure_pct * NAV by now,
    # while total_at_risk <= NAV leaves gross headroom of at least
    # (max_gross_exposure_multiple - 1) * NAV, which is larger. Kept as a
    # backstop for an ungrouped city or a custom profile, not dead code.
    total_at_risk = max(0, total_equity - (ctx.live_balance or 0))
    if not _apply_risk_limit(signal, check_gross_exposure(
        signal.recommended_size, total_at_risk, total_equity, profile
    )):
        return False

    # Final min-size check after all trims
    if signal.recommended_size < MIN_LIVE_SIZE:
        logger.info("TRIMMED BELOW MIN: %s size $%.2f < $%.2f",
                    signal.city_id, signal.recommended_size, MIN_LIVE_SIZE)
        return False
    return True


def _use_taker(signal: Signal) -> bool:
    """Entry style: True = taker (cross the spread), False = resting maker order."""
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
        # No fee-gate re-check for takers: validate_fee_alpha_ratio blocks
        # only when 0.05 * p * (1 - p) / edge > 0.40, i.e. edge < 3.125% at
        # worst (p = 0.5), independent of size. Every taker has edge >= 8%,
        # so a fee-blocked taker cannot exist (the old "FEE GATE TAKER"
        # branch was unreachable).
        if use_taker:
            logger.info(
                "TAKER ENTRY: %s edge=%.1f%%, crossing spread",
                signal.city_id, signal.edge * 100,
            )
    return use_taker


# place_limit_order statuses that mean no order is resting on the book
_REJECTED_ENTRY_STATUSES = ("rejected", "post_only_reject", "too_small")


def _record_live_entry(ctx: CycleContext, signal: Signal, result, use_taker: bool) -> None:
    """Track an accepted live order against the balance and the position cap."""
    # Track spending against live balance
    if ctx.live_balance is not None:
        ctx.live_balance -= result.size_usd
    # Track position count for cap enforcement
    ctx.active_position_count += 1

    logger.info(
        "LIVE: %s %s %s %s %.0f shares @ %.3f, %s (bal=$%.2f)",
        "TAKER" if use_taker else "MAKER",
        result.status, signal.recommended_side.value,
        signal.city_id, result.size_shares,
        result.limit_price, result.order_id,
        ctx.live_balance or 0,
    )


def _hedge_fits(ctx: CycleContext, h_signal: Signal) -> bool:
    """A hedge is a position like any entry: it needs cap room and the cash."""
    if ctx.active_position_count >= ctx.max_positions:
        logger.info(
            "HEDGE SKIP: %s, position cap %d/%d",
            h_signal.city_id, ctx.active_position_count, ctx.max_positions,
        )
        return False
    if ctx.live_balance is not None and h_signal.recommended_size > ctx.live_balance:
        logger.info(
            "HEDGE SKIP: %s needs $%.2f but exchange balance is $%.2f",
            h_signal.city_id, h_signal.recommended_size, ctx.live_balance,
        )
        return False
    return True


def _record_live_hedge(ctx: CycleContext, signal: Signal, h_signal: Signal, h_result) -> None:
    """Track an accepted hedge against the balance and the position cap, like an entry."""
    if ctx.live_balance is not None:
        ctx.live_balance -= h_result.size_usd
    ctx.active_position_count += 1
    logger.info(
        "LIVE SPREAD HEDGE: %s %s %.0f shares @ %.3f, %s (bal=$%.2f)",
        h_signal.recommended_side.value, signal.city_id,
        h_result.size_shares, h_result.limit_price,
        h_result.order_id, ctx.live_balance or 0,
    )


async def _place_live_hedge(ctx: CycleContext, signal: Signal) -> None:
    """LIVE SPREAD CAPTURE HEDGE on the opposite token of a placed entry.

    HAIL MARY skips hedging: every dollar of capital is one ticket, no
    doubling up.
    """
    market_maker, live_executor = ctx.market_maker, ctx.live_executor
    if not (
        market_maker
        and not settings.hail_mary_mode
        and not live_executor.dry_run
    ):
        return
    # Use settings.bankroll as baseline for pool-sizing
    hedge = market_maker.generate_hedge_orders(
        signal, ctx.market_prices, settings.bankroll, is_live=True
    )
    if not hedge:
        return
    m = ctx.market_by_id.get(hedge.market_id)
    if not m:
        return
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
    if not _hedge_fits(ctx, h_signal):
        return

    try:
        import requests
        # improve_price_by=0: limit is exactly
        # hedge.limit_price for the hedge token
        h_result = await live_executor.place_limit_order(
            h_signal, h_token_id, improve_price_by=0.0,
        )
        if h_result and h_result.status not in _REJECTED_ENTRY_STATUSES:
            _record_live_hedge(ctx, signal, h_signal, h_result)
    except (requests.RequestException, ValueError, KeyError) as e:
        logger.error("LIVE SPREAD HEDGE FAILED: %s, %s", signal.city_id, e)


async def _submit_live_entry(ctx: CycleContext, signal: Signal, token_id: str) -> None:
    """Place the live entry order (and its spread hedge); failures are logged."""
    try:
        use_taker = _use_taker(signal)
        result = await ctx.live_executor.place_limit_order(
            signal, token_id, force_taker=use_taker,
        )
        if result and result.status not in _REJECTED_ENTRY_STATUSES:
            _record_live_entry(ctx, signal, result, use_taker)
            await _place_live_hedge(ctx, signal)

    except Exception as e:
        # One failed order must not stop the rest of the execution loop.
        logger.error(
            "LIVE ORDER FAILED: %s, %s", signal.city_id, e, exc_info=True,
        )


async def _place_live_entry(ctx: CycleContext, signal: Signal) -> None:
    """LIVE EXECUTION (maker orders bypass fee gate, $0 maker fee).

    Still require minimum raw edge, we're bypassing fee check, not edge check.
    """
    if _in_exit_cooldown(ctx, signal):
        return
    if settings.hail_mary_mode:
        sized = _size_hail_mary_ticket(ctx, signal)
    else:
        sized = _size_swing_entry(ctx, signal)
    if not sized or not _live_edge_ok(signal):
        return

    m = ctx.market_by_id.get(signal.market_id)
    if not m:
        return
    # Pick the right token
    if signal.recommended_side.value == "YES":
        token_id = m.token_id_yes
    else:
        token_id = m.token_id_no
    if not token_id:
        return

    place, chased_prob = await _reconcile_existing_entry(ctx, signal)
    if not place or not _passes_live_risk_controls(ctx, signal):
        return
    # Order on a copy at the chased price; the signal keeps the observed price.
    order_signal = (
        signal if chased_prob is None else replace(signal, market_prob=chased_prob)
    )
    await _submit_live_entry(ctx, order_signal, token_id)


async def _execute_signal(ctx: CycleContext, signal: Signal) -> None:
    """Paper and live entries for one AI-cleared signal."""
    # A zero/negative size means some layer (Kelly, Claude, Gemini) sized
    # the trade away. Never trade it, and never let the live min-size
    # floor below resurrect it.
    if signal.recommended_size <= 0 and not settings.hail_mary_mode:
        logger.info(
            "ZERO SIZE: %s %s sized to $%.2f, skipping",
            signal.city_id, signal.description[:40], signal.recommended_size,
        )
        return

    # Contract: verify taker fee doesn't eat >40% of projected alpha
    # This gate applies to TAKER orders only. Live executor uses post_only
    # (maker, $0 fee) so it bypasses this check.
    fee_check = validate_fee_alpha_ratio(
        edge=signal.edge,
        price=signal.market_prob,
        size_usd=signal.recommended_size,
    )
    fee_blocked = not fee_check.valid

    if fee_blocked and not ctx.live_executor:
        # Paper-only mode: skip trade entirely
        logger.info(
            "CONTRACT [%s]: %s, skipping %s %s",
            fee_check.code, fee_check.error, signal.city_id, signal.description[:40],
        )
        return
    elif fee_blocked and ctx.live_executor:
        # Live mode: skip paper trade (taker fees eat alpha) but still
        # place live maker order ($0 fee). Log the fee gate for awareness.
        logger.info(
            "FEE GATE (paper only): %s %s, paper skipped, live maker order OK",
            signal.city_id, signal.description[:40],
        )

    _place_paper_entry(ctx, signal, fee_blocked)

    if ctx.live_entries_open():
        await _place_live_entry(ctx, signal)


async def _execute_signals(ctx: CycleContext) -> None:
    """Stage 6: place paper and live orders for every AI-cleared signal."""
    for signal in ctx.executable_signals:
        await _execute_signal(ctx, signal)


def _log_spread_summary(market_maker) -> None:
    """Log the spread capture summary."""
    spread_summary = market_maker.simulate_spread_pnl()
    if spread_summary["spread_orders"] > 0:
        logger.info(
            "SPREAD CAPTURE: %d orders, est. guaranteed P&L=$%.2f",
            spread_summary["spread_orders"], spread_summary["estimated_guaranteed_pnl"],
        )


@dataclass
class _ExitMarks:
    """Current market marks shared by the paper and live exit scans."""

    market_prices: dict[str, float]
    no_prices: dict[str, float]
    market_dates: dict[str, date]
    model_probs: dict[str, float]


def _exit_marks(markets: list[MarketInfo], all_signals: list[Signal]) -> _ExitMarks:
    """YES / NO prices, target dates and model probabilities for the exit scans."""
    current_market_prices = {m.market_id: m.yes_price for m in markets}
    # NO token's own price (Gamma outcomePrices[1]); exit_monitor falls
    # back to 1-YES when missing.
    current_no_prices = {
        m.market_id: m.no_price for m in markets if getattr(m, "no_price", 0) > 0
    }
    # Target date per market, so exit reviews use the trade's own date
    market_dates = {m.market_id: m.target_date for m in markets}
    current_model_probs = {}
    for signal in all_signals:
        current_model_probs[signal.market_id] = signal.model_prob
    return _ExitMarks(
        market_prices=current_market_prices, no_prices=current_no_prices,
        market_dates=market_dates, model_probs=current_model_probs,
    )


async def _review_exit_candidate(ctx: CycleContext, candidate, marks: _ExitMarks):
    """Claude + Gemini review of one exit candidate, on its own date's forecasts."""
    from weather_edge.analysis.exit_monitor import ai_review_exit

    model_vals, c_mean, c_std = exit_model_context(
        candidate.trade, ctx.forecast_cache, marks.market_dates,
    )
    return await ai_review_exit(
        candidate, model_vals, c_mean, c_std,
    )


async def _scan_paper_exits(ctx: CycleContext, marks: _ExitMarks) -> None:
    """PAPER EXIT SCANNING: review up to 3 candidates and close the EXITs."""
    from weather_edge.analysis.exit_monitor import scan_for_exits_async

    paper_trader = ctx.paper_trader
    paper_candidates = await scan_for_exits_async(
        paper_trader.open_trades,
        marks.market_prices, marks.model_probs,
        forecast_cache=ctx.forecast_cache,
        no_prices=marks.no_prices,
    )
    if paper_candidates:
        logger.info("PAPER EXIT MONITOR: %d candidates", len(paper_candidates))
        for candidate in paper_candidates[:3]:
            reviewed = await _review_exit_candidate(ctx, candidate, marks)
            if reviewed.final_decision == "EXIT":
                logger.warning(
                    "PAPER EXIT: %s %s $%.0f, %s",
                    reviewed.trade.side,
                    reviewed.trade.city_id,
                    reviewed.trade.size_usd,
                    reviewed.reason,
                )
                paper_trader.close_position(
                    reviewed.trade,
                    reviewed.current_market_price,
                )


def _live_exit_positions(store) -> list:
    """Live positions large enough to sell, from the positions table."""
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
    return live_positions


def _sell_half_done(store, market_id: str, city_id_str: str) -> bool:
    """Sell-half guard: True (skip) if a sell is open or the position was already trimmed."""
    existing = store.get_open_order_for_market(
        market_id, side="SELL",
    )
    if existing:
        logger.info(
            "SELL_HALF SKIP: %s has open sell order",
            city_id_str,
        )
        return True
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
        return True
    return False


async def _cancel_open_buy(ctx: CycleContext, market_id: str, city_id_str: str) -> None:
    """Cancel any open BUY order on a market we are exiting."""
    store = ctx.store
    open_buy = store.get_open_order_for_market(market_id)
    if open_buy and open_buy.get("side") in ("YES", "NO"):
        buy_id = open_buy["order_id"]
        # A buy that can't be cancelled stays tracked as open; the exit
        # still sells the tokens already held.
        if await _cancel_and_record(ctx, buy_id, "Failed to cancel BUY order %s: %s"):
            logger.info(
                "EXIT: cancelled open BUY order %s for %s",
                buy_id[:16], city_id_str,
            )


async def _replace_existing_sell(
    ctx: CycleContext, existing_sell: dict, sell_price: float, city_id_str: str,
) -> bool:
    """Replace an open sell only if the price drifted >1c and the old one cancelled.

    False = keep it (skip placing a new sell).
    """
    old_price = existing_sell.get("limit_price", 0)
    new_price = _round_price(sell_price)
    price_drift = abs(old_price - new_price)

    if price_drift <= 0.01:
        logger.info(
            "LIVE SELL KEEP: %s @ %.3f "
            "(drift=%.3f)",
            city_id_str, old_price, price_drift,
        )
        return False
    old_id = existing_sell["order_id"]
    # A second sell while the first may still rest could oversell the position.
    if not await _cancel_and_record(ctx, old_id, "Failed to cancel old sell order %s: %s"):
        return False
    logger.info(
        "LIVE SELL REPLACE: %s "
        "cancelled %s (%.3f->%.3f)",
        city_id_str, old_id[:16],
        old_price, new_price,
    )
    return True


async def _place_exit_sell(
    ctx: CycleContext, candidate, position: dict, asset_id: str, sell_price: float,
) -> None:
    """Sell the held token (half of it for sell_half); failures are logged."""
    city_id_str = candidate.trade.city_id
    market_id = candidate.trade.market_id
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
        sell_result = await ctx.live_executor.place_sell_order(
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
        # One failed exit must not stop the remaining exit candidates.
        logger.error(
            "LIVE EXIT FAILED: %s, %s",
            city_id_str, e, exc_info=True,
        )


async def _execute_live_exit(ctx: CycleContext, candidate) -> bool:
    """Act on a live EXIT decision. False = skipped (the decision is not recorded)."""
    store = ctx.store
    city_id_str = candidate.trade.city_id
    market_id = candidate.trade.market_id

    # Sell-half guard: skip if already trimmed
    if candidate.reason == "sell_half" and _sell_half_done(store, market_id, city_id_str):
        return False

    # 1. Cancel any open BUY orders if exiting
    await _cancel_open_buy(ctx, market_id, city_id_str)

    # 2. Check for existing open SELL order
    existing_sell = store.get_open_order_for_market(
        market_id, side="SELL",
    )

    position = store.get_position_for_market(market_id)
    if not (
        position
        and position.get("total_shares", 0) >= MIN_SELL_SHARES
    ):
        return True
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
    if not asset_id:
        return True
    # Replace existing sell only if price drifted >1c
    if existing_sell and not await _replace_existing_sell(
        ctx, existing_sell, sell_price, city_id_str,
    ):
        return False
    await _place_exit_sell(ctx, candidate, position, asset_id, sell_price)
    return True


def _record_exit_decision(candidate) -> None:
    """Record an exit decision to the AI Decisions tab."""
    from weather_edge.analysis.claude_reasoning import _decision_history
    _decision_history.insert(0, {
        "time": datetime.now(UTC).strftime("%H:%M:%S"),
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


async def _scan_live_exits(ctx: CycleContext, marks: _ExitMarks) -> None:
    """LIVE EXIT SCANNING from the positions table, independent of paper."""
    from weather_edge.analysis.exit_monitor import scan_for_exits_async

    live_positions = _live_exit_positions(ctx.store)
    if live_positions:
        live_candidates = await scan_for_exits_async(
            live_positions,
            marks.market_prices, marks.model_probs,
            forecast_cache=ctx.forecast_cache,
            no_prices=marks.no_prices,
        )
        if live_candidates:
            logger.info("LIVE EXIT MONITOR: %d candidates", len(live_candidates))
            for candidate in live_candidates[:3]:
                reviewed = await _review_exit_candidate(ctx, candidate, marks)
                if (
                    reviewed.final_decision == "EXIT"
                    and not await _execute_live_exit(ctx, reviewed)
                ):
                    continue
                _record_exit_decision(reviewed)


async def _monitor_exits(ctx: CycleContext) -> None:
    """Stage 7: early exit monitor. Paper and live scan independently."""
    try:
        marks = _exit_marks(ctx.markets, ctx.all_signals)

        # --- PAPER EXIT SCANNING ---
        if ctx.paper_trader:
            await _scan_paper_exits(ctx, marks)

        # --- LIVE EXIT SCANNING (from positions table, independent of paper) ---
        # HAIL MARY disables it: penny lottery tickets must hold to resolution,
        # and selling at -7% edge wastes the spread cost on a $1 position.
        if (
            not settings.hail_mary_mode
            and ctx.is_live
            and ctx.store
        ):
            await _scan_live_exits(ctx, marks)
    except Exception:
        logger.error("EXIT MONITOR CRASHED, check traceback", exc_info=True)


def _monitoring_date(target_dates: list[date]) -> date:
    """The one date monitored for cities without markets ("tomorrow" of the window)."""
    if len(target_dates) > 1:
        tomorrow = target_dates[1]
    elif target_dates:
        tomorrow = target_dates[0]
    else:
        tomorrow = trading_today() + timedelta(days=1)
    return tomorrow


async def _refresh_monitoring_forecasts(ctx: CycleContext) -> None:
    """Stage 8: also fetch forecasts for cities without active markets (monitoring).

    But only for tomorrow (not all dates) to save API calls.
    """
    tomorrow = _monitoring_date(ctx.target_dates)
    for city_id in City:
        # The cache persists across cycles, so "not in cache" would freeze
        # monitoring data after the first cycle; refetch unless fresh this cycle.
        if (city_id, tomorrow) not in ctx.refreshed:
            forecasts = await fetch_city_forecasts(city_id, tomorrow)
            if forecasts:
                ctx.forecast_cache[(city_id, tomorrow)] = forecasts
                ctx.refreshed.add((city_id, tomorrow))
                consensus = compute_consensus(city_id, str(tomorrow), "temp_max_c", forecasts)
                if consensus:
                    logger.info(
                        "  %s (no markets): mean=%.1f°C conf=%.0f%%",
                        city_id.value, consensus.weighted_mean, consensus.confidence * 100,
                    )


async def run_cycle(
    paper_trader: PaperTrader | None,
    target_dates: list[date] | None = None,
    *,
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
    ctx = await _start_cycle(
        paper_trader,
        target_dates=target_dates,
        run_ai_reasoning=run_ai_reasoning,
        live_executor=live_executor,
        store=store,
        forecast_cache=forecast_cache,
    )
    await _discover_markets(ctx)
    await _fetch_ai_forecasts(ctx)
    await _compute_signals(ctx)
    ctx.all_signals = _filter_signals(ctx.all_signals)

    # === Claude + Gemini reasoning layer ===
    # Every cycle type (main, sniper, refresh) goes through the same gate: only
    # signals the AI layer approved (this cycle, or earlier today) are executed.
    ctx.executable_signals = await apply_ai_review(
        ctx.all_signals, ctx.forecast_cache, ctx.run_ai_reasoning, ctx.store,
    )

    await _prepare_execution(ctx)
    await _execute_signals(ctx)
    _log_spread_summary(ctx.market_maker)
    await _monitor_exits(ctx)
    await _refresh_monitoring_forecasts(ctx)

    return ctx.all_signals, ctx.forecast_cache, ctx.city_volume


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
            today = trading_today()
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
