"""Trade resolution system, settles open paper trades against actual outcomes.

Checks Polymarket Gamma API for resolved markets, with METAR/HKO station
observations as a fallback for markets whose target date has passed but haven't
resolved on-chain yet. There is no reanalysis fallback: without station data a
trade stays open and is retried.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date

import httpx

from weather_edge.config import CITIES
from weather_edge.fetchers.openmeteo import c_to_f
from weather_edge.fetchers.polymarket import (
    LOWER_TAIL_WORDS,
    MONTH_MAP,
    UPPER_TAIL_WORDS,
    nearest_year_date,
)
from weather_edge.models.enums import City
from weather_edge.trading.paper import PaperTrade, PaperTrader

logger = logging.getLogger(__name__)

# Regex to parse temperature ranges from trade descriptions.
# Capture groups use (-?\d+) so subzero winter buckets parse correctly, an
# unsigned (\d+) drops the minus sign and resolves "-2°C or below" as "2°C or
# below", a guaranteed mis-resolution on every freezing-weather market.
# Tail wording ("or below"/"or lower"/..., "or above"/"or higher"/...) is shared
# with the market parser in fetchers/polymarket.py so the two never disagree.
# Fahrenheit patterns
RANGE_PATTERN = re.compile(r"(-?\d+)\s*[-–]\s*(-?\d+)\s*°?\s*F", re.IGNORECASE)
BELOW_PATTERN = re.compile(
    r"(-?\d+)\s*°?\s*F\s+or\s+" + LOWER_TAIL_WORDS + r"\b", re.IGNORECASE,
)
ABOVE_PATTERN = re.compile(
    r"(-?\d+)\s*°?\s*F\s+or\s+" + UPPER_TAIL_WORDS + r"\b", re.IGNORECASE,
)
# Celsius patterns (Asian/international cities)
RANGE_PATTERN_C = re.compile(r"(-?\d+)\s*[-–]\s*(-?\d+)\s*°\s*C", re.IGNORECASE)
BELOW_PATTERN_C = re.compile(
    r"(-?\d+)\s*°?\s*C\s+or\s+" + LOWER_TAIL_WORDS + r"\b", re.IGNORECASE,
)
ABOVE_PATTERN_C = re.compile(
    r"(-?\d+)\s*°?\s*C\s+or\s+" + UPPER_TAIL_WORDS + r"\b", re.IGNORECASE,
)
# Exact Celsius: "be 8°C on" (single-value buckets)
EXACT_PATTERN_C = re.compile(r"be\s+(-?\d+)\s*°\s*C\s+on\s+", re.IGNORECASE)

# Cache native Fahrenheit max from METAR to avoid F↔C conversion rounding errors.
# Key: (city_id, target_date_str) → native max_f value
_metar_native_f_cache: dict[tuple[str, str], float | None] = {}


# A resolved market pays out at exactly 1/0; allow float noise only.
RESOLVED_PRICE_THRESHOLD = 0.99


def _is_market_resolved(mkt: dict) -> bool:
    """True only when Gamma reports the market's oracle resolution as final.

    ``closed`` alone is NOT resolution: it just means trading stopped. Gamma
    signals final resolution via ``umaResolutionStatus == "resolved"`` (or a
    ``resolved`` flag on some payloads).
    """
    if mkt.get("resolved") is True:
        return True
    status = str(mkt.get("umaResolutionStatus") or "").strip().lower()
    return status == "resolved"


async def fetch_resolved_markets() -> dict[str, bool]:
    """Query Polymarket Gamma API for recently resolved weather markets.

    Returns:
        Dict mapping market condition_id -> outcome_yes (True if YES won).
    """
    resolved: dict[str, bool] = {}

    async with httpx.AsyncClient() as client:
        for offset in range(0, 500, 100):
            try:
                resp = await client.get(
                    "https://gamma-api.polymarket.com/events",
                    params={
                        "tag_slug": "weather",
                        "closed": "true",
                        "limit": 100,
                        "offset": offset,
                    },
                    timeout=15.0,
                )
                resp.raise_for_status()
                events = resp.json()
            except (httpx.HTTPError, ValueError) as e:
                logger.error("Failed to fetch resolved events at offset %d: %s", offset, e)
                break

            if not events:
                break

            for event in events:
                event_markets = event.get("markets", [])
                for mkt in event_markets:
                    condition_id = mkt.get("conditionId", "")
                    if not condition_id:
                        continue

                    # Only an ACTUALLY resolved market settles a trade. A market
                    # that is merely closed (trading halted, oracle not yet
                    # reported, or a disputed proposal) can sit at 0.95+ and
                    # still resolve the other way; treating it as settled
                    # booked P&L that the oracle could reverse.
                    if not _is_market_resolved(mkt):
                        continue

                    # Determine outcome: check outcomePrices for resolved state
                    # Resolved markets show [1.0, 0.0] (YES won) or [0.0, 1.0] (NO won)
                    outcome_prices = mkt.get("outcomePrices")
                    if outcome_prices:
                        try:
                            if isinstance(outcome_prices, str):
                                import json
                                outcome_prices = json.loads(outcome_prices)
                            if isinstance(outcome_prices, list) and len(outcome_prices) >= 2:
                                yes_price = float(outcome_prices[0])
                                no_price = float(outcome_prices[1])
                                # Resolved markets have prices at 0 or 1
                                if yes_price >= RESOLVED_PRICE_THRESHOLD:
                                    resolved[condition_id] = True
                                elif no_price >= RESOLVED_PRICE_THRESHOLD:
                                    resolved[condition_id] = False
                        except (ValueError, TypeError):
                            pass

                    # Also check the "outcome" or "winner" field if present
                    outcome = mkt.get("outcome")
                    if outcome is not None:
                        if outcome == "Yes" or outcome == "yes" or outcome is True:
                            resolved[condition_id] = True
                        elif outcome == "No" or outcome == "no" or outcome is False:
                            resolved[condition_id] = False

            logger.debug("Fetched %d resolved events at offset %d", len(events), offset)

            if len(events) < 100:
                break

    logger.info("Found %d resolved weather markets from Polymarket", len(resolved))
    return resolved


async def check_nws_observations(city_id: str, target_date: date) -> float | None:
    """Fetch actual observed high temperature from METAR station data.

    Uses IEM (Iowa Environmental Mesonet) ASOS archive as primary source,
    this is the same raw METAR data that Weather Underground displays, and
    Polymarket resolves weather markets against Wunderground.

    Returns None (never a reanalysis substitute) when METAR is unavailable,
    so the caller leaves the trade open and retries later.

    Args:
        city_id: City enum value (e.g., "nyc", "den").
        target_date: The date to look up observations for.

    Returns:
        Observed high temperature in Celsius, or None if unavailable.
    """
    try:
        city_enum = City(city_id)
    except ValueError:
        logger.warning("Unknown city_id for observation lookup: %s", city_id)
        return None

    if city_enum not in CITIES:
        return None

    city_config = CITIES[city_enum]

    # --- IEM METAR station observation (the resolution source) ---
    # Fetch both native C and F to avoid conversion rounding errors.
    # Store native_f on the trade for Fahrenheit markets so the resolver
    # can round(max_f) directly instead of round(c_to_f(max_c)).
    #
    # There is deliberately NO Open-Meteo reanalysis fallback here. Gridded
    # reanalysis differs from the station sensor by 0.5-1.5°C, enough to flip a
    # whole-degree bucket most of the time; settling a trade from it booked a
    # permanent, possibly wrong outcome AND wrote reanalysis values into the
    # bias-training data as if they were station truth. When METAR is
    # unavailable we return None and the trade stays open, to be retried next
    # cycle (or settled by Polymarket's own resolution).
    try:
        from weather_edge.fetchers.metar import fetch_station_tmax_both
        temp_c, temp_f = await fetch_station_tmax_both(
            city_config.icao, target_date, station_tz=city_config.timezone,
        )
    except Exception as e:
        logger.warning(
            "METAR fetch failed for %s on %s: %s; leaving trade unresolved",
            city_id, target_date, e,
        )
        return None

    # fetch_station_tmax_both always returns (c, f) together or (None, None):
    # every station row contributes to both candidate lists, so there is no
    # Fahrenheit-only case to reconstruct here.
    if temp_c is None:
        logger.warning(
            "No usable METAR observation for %s on %s [station %s]; "
            "leaving trade unresolved",
            city_id, target_date, city_config.icao,
        )
        return None

    # Store native F max in a module-level cache for the resolver
    _metar_native_f_cache[(city_id, str(target_date))] = temp_f
    logger.info(
        "METAR obs for %s on %s: %.1f°C / %.1f°F [station %s]",
        city_id, target_date, temp_c,
        temp_f if temp_f is not None else c_to_f(temp_c), city_config.icao,
    )
    # Backfill actual value in forecast snapshots (station truth only)
    try:
        from weather_edge.dashboard.app import paper_trader
        updated = paper_trader.store.backfill_actual(
            city_id, str(target_date), temp_c,
        )
        if updated:
            logger.info("Backfilled %d forecast snapshots for %s %s",
                       updated, city_id, target_date)
    except Exception:
        logger.debug("Snapshot backfill skipped", exc_info=True)
    return temp_c


def _c_to_f(c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return c * 9.0 / 5.0 + 32.0


def _round_half_up(x: float) -> int:
    """Round to the nearest integer, ties going up (toward +inf).

    Weather Underground displays temperatures rounded half-up, and Polymarket
    resolves against that displayed value. Python's built-in round() uses
    banker's rounding (round-half-to-even), so round(72.5) == 72 while
    Wunderground shows 73, flipping the resolution bucket on exact-half
    readings. math.floor(x + 0.5) matches the displayed-value convention for
    both positive and negative temperatures.
    """
    import math
    return math.floor(x + 0.5)


def _hkg_label(x: float) -> int:
    """Whole-degree bucket label for an HK Observatory 0.1°C reading: [L, L+1)."""
    import math
    return math.floor(x + 1e-9)


@dataclass
class BucketInfo:
    """Parsed temperature bucket with native unit tracking."""
    low: float | None   # Lower bound (None = unbounded)
    high: float | None  # Upper bound (None = unbounded)
    unit: str           # "fahrenheit" or "celsius"
    exclusive_upper: bool = False  # True for exact C buckets [X, X+1)


def parse_bucket_from_description(
    description: str,
) -> BucketInfo | None:
    """Parse temperature bucket boundaries from a trade description.

    Returns BucketInfo in native units (no conversion).
    Fahrenheit ranges: inclusive both bounds [76, 78)
    Celsius exact: exclusive upper [20, 21)
    """
    # --- Fahrenheit patterns ---
    m = BELOW_PATTERN.search(description)
    if m:
        return BucketInfo(None, float(m.group(1)), "fahrenheit")

    m = RANGE_PATTERN.search(description)
    if m:
        # "76-77°F" means [76, 78) per Polymarket rules
        return BucketInfo(float(m.group(1)), float(m.group(2)) + 1.0,
                          "fahrenheit", exclusive_upper=True)

    m = ABOVE_PATTERN.search(description)
    if m:
        return BucketInfo(float(m.group(1)), None, "fahrenheit")

    # --- Celsius patterns (keep in native °C) ---
    m = BELOW_PATTERN_C.search(description)
    if m:
        return BucketInfo(None, float(m.group(1)), "celsius")

    m = RANGE_PATTERN_C.search(description)
    if m:
        return BucketInfo(float(m.group(1)), float(m.group(2)) + 1.0,
                          "celsius", exclusive_upper=True)

    m = ABOVE_PATTERN_C.search(description)
    if m:
        return BucketInfo(float(m.group(1)), None, "celsius")

    # Exact Celsius: "be 8°C on" → [8, 9)
    m = EXACT_PATTERN_C.search(description)
    if m:
        val_c = float(m.group(1))
        return BucketInfo(val_c, val_c + 1.0, "celsius", exclusive_upper=True)

    return None


def actual_falls_in_bucket(
    actual_temp_c: float,
    bucket: BucketInfo,
    *,
    native_f: float | None = None,
    is_hkg: bool = False,
) -> bool:
    """Check if an actual temperature falls within a bucket's range.

    Polymarket resolves against Wunderground's displayed value, which is
    rounded to whole degrees. We round in the market's NATIVE unit to
    avoid F↔C conversion rounding errors.

    HKG is special: HK Observatory publishes the Absolute Daily Max to 0.1°C
    and that decimal value is NOT rounded to a whole degree. A whole-degree
    label L covers the decimal values whose integer part is L, i.e. [L, L+1):
    "28°C" is 28.0-28.9, "28°C or below" is anything under 29.0, "28°C or
    above" is 28.0 and up. We floor the decimal to its whole-degree label and
    then apply the same integer bucket test as every other city. The
    probability side (scheduler._bucket_celsius_band) integrates the same
    [L, L+1) interval for HKG. (The raw-decimal comparison this replaced got
    the "or below" tail wrong: 28.5 is inside "28°C or below" under this rule
    but failed ``28.5 <= 28``.)

    Args:
        actual_temp_c: Actual observed temperature in Celsius (raw METAR).
        bucket: BucketInfo from parse_bucket_from_description.
        native_f: Native Fahrenheit max from METAR (avoids C→F→round error).
        is_hkg: True for Hong Kong (0.1°C value, whole-degree label = floor).

    Returns:
        True if the actual temperature falls in this bucket.
    """
    if bucket.unit == "fahrenheit":
        # Use native F reading when available to avoid conversion rounding
        if native_f is not None:
            actual = _round_half_up(native_f)
        else:
            actual = _round_half_up(_c_to_f(actual_temp_c))
    elif is_hkg:
        # HK Observatory: 0.1°C value; its whole-degree label is the floor.
        # The epsilon absorbs float noise like 28.999999 from a "29.0" reading.
        actual = _hkg_label(actual_temp_c)
    else:
        actual = _round_half_up(actual_temp_c)

    low, high = bucket.low, bucket.high

    if low is None and high is not None:
        # "X or below", actual <= high
        return actual <= high
    elif low is not None and high is None:
        # "X or above", actual >= low
        return actual >= low
    elif low is not None and high is not None:
        if bucket.exclusive_upper:
            # Range/exact: low <= actual < high
            return low <= actual < high
        else:
            # Inclusive: low <= actual <= high
            return low <= actual <= high

    return False


def _extract_target_date_from_trade(trade: PaperTrade) -> date | None:
    """Try to extract the target date from a trade's description or market context.

    Looks for date patterns like "on March 27" in the description.
    """
    desc = trade.description or ""
    # Match "on March 27" pattern
    date_pattern = re.compile(
        r"on\s+(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+(\d{1,2})",
        re.IGNORECASE,
    )
    m = date_pattern.search(desc)
    if m:
        month = MONTH_MAP.get(m.group(1).lower(), 1)
        day = int(m.group(2))
        # Year: nearest to when the trade was placed (the market was live then),
        # falling back to today. Fixes the year rollover where a "December 31"
        # market checked on Jan 2 parsed as the future Dec 31.
        placed = getattr(trade, "placed_at", None)
        reference = placed.date() if placed is not None else date.today()
        return nearest_year_date(month, day, reference)
    return None


async def resolve_open_trades(paper_trader: PaperTrader) -> int:
    """Resolve any open trades whose markets have settled.

    Checks Polymarket for resolved markets first, then falls back to
    METAR/HKO station observations for trades past their target date. A trade
    with no station observation stays open.

    Args:
        paper_trader: The paper trader instance (PaperTrader or PersistentPaperTrader).

    Returns:
        Number of trades resolved this cycle.
    """
    open_trades = paper_trader.open_trades
    if not open_trades:
        return 0

    logger.info("=== RESOLVER: Checking %d open trades ===", len(open_trades))

    # Step 1: Fetch resolved markets from Polymarket
    try:
        resolved_markets = await fetch_resolved_markets()
    except Exception as e:
        logger.error("Failed to fetch resolved markets: %s", e)
        resolved_markets = {}

    resolved_count = 0
    today = date.today()

    for trade in open_trades:
        # --- Try Polymarket resolution first ---
        if trade.market_id in resolved_markets:
            outcome_yes = resolved_markets[trade.market_id]
            paper_trader.resolve_trade(trade, outcome_yes=outcome_yes)
            resolved_count += 1
            logger.info(
                "RESOLVED (Polymarket): %s %s %s | outcome=%s | P&L=$%.2f | %s",
                trade.side,
                trade.city_id.upper() if isinstance(trade.city_id, str) else trade.city_id,
                trade.description[:50] if trade.description else "",
                "YES" if outcome_yes else "NO",
                trade.pnl or 0.0,
                trade.status.value,
            )
            continue

        # --- Fallback: check if target date has passed and use observations ---
        target_date = _extract_target_date_from_trade(trade)
        if target_date is None:
            continue

        # Use city's local timezone to determine if the day is over
        # Wellington (UTC+13) finishes 13h before UTC midnight
        city_id = trade.city_id
        try:
            city_enum = City(city_id)
            city_tz_name = CITIES[city_enum].timezone
            from datetime import datetime
            from zoneinfo import ZoneInfo
            city_now = datetime.now(ZoneInfo(city_tz_name))
            city_today = city_now.date()
        except Exception:
            city_today = today

        # Resolve if the target date has passed in the city's timezone
        if target_date >= city_today:
            continue

        # Fetch actual observation
        city_id = trade.city_id
        actual_temp_c = await check_nws_observations(city_id, target_date)
        if actual_temp_c is None:
            logger.debug(
                "No observation available yet for %s on %s",
                city_id, target_date,
            )
            continue

        # Check bucket, compare in native units (no unnecessary conversion)
        bucket = parse_bucket_from_description(trade.description or "")
        if bucket is None:
            logger.warning(
                "Could not parse bucket from trade description: %s",
                trade.description[:80] if trade.description else "(empty)",
            )
            continue

        # Determine if YES won (actual temp falls in this bucket)
        # Use native F max and HKG flag for correct rounding
        native_f = _metar_native_f_cache.get((city_id, str(target_date)))
        is_hkg = city_id == "hkg"
        yes_won = actual_falls_in_bucket(
            actual_temp_c, bucket, native_f=native_f, is_hkg=is_hkg,
        )
        paper_trader.resolve_trade(trade, outcome_yes=yes_won)
        resolved_count += 1

        unit = bucket.unit
        unit_sym = "°C" if unit == "celsius" else "°F"
        bucket_str = (
            f"[{bucket.low}-{bucket.high}){unit_sym}"
            if bucket.low is not None and bucket.high is not None
            else f"<={bucket.high}{unit_sym}" if bucket.high is not None
            else f">={bucket.low}{unit_sym}" if bucket.low is not None
            else "unknown"
        )
        logger.info(
            "RESOLVED (observation): %s %s | bucket=%s | actual=%.1f°C | "
            "YES_won=%s | P&L=$%.2f | %s",
            trade.side,
            trade.city_id.upper() if isinstance(trade.city_id, str) else trade.city_id,
            bucket_str,
            actual_temp_c,
            yes_won,
            trade.pnl or 0.0,
            trade.status.value,
        )

    if resolved_count > 0:
        logger.info(
            "=== RESOLVER: Settled %d trades | Total P&L now $%.2f | Win rate %.1f%% ===",
            resolved_count,
            paper_trader.total_pnl,
            paper_trader.win_rate * 100,
        )
    else:
        logger.debug("RESOLVER: No trades resolved this cycle")

    return resolved_count
