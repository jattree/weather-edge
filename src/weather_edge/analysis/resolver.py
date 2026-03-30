"""Trade resolution system, settles open paper trades against actual outcomes.

Checks Polymarket Gamma API for resolved markets, with Open-Meteo archive
as a fallback for markets whose target date has passed but haven't resolved
on-chain yet.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date

import httpx

from weather_edge.config import CITIES
from weather_edge.fetchers.openmeteo import c_to_f
from weather_edge.models.enums import City
from weather_edge.trading.paper import PaperTrade, PaperTrader

logger = logging.getLogger(__name__)

# Regex to parse temperature ranges from trade descriptions
# Fahrenheit patterns
RANGE_PATTERN = re.compile(r"(\d+)\s*[-–]\s*(\d+)\s*°?\s*F", re.IGNORECASE)
BELOW_PATTERN = re.compile(r"(\d+)\s*°?\s*F\s+or\s+below", re.IGNORECASE)
ABOVE_PATTERN = re.compile(r"(\d+)\s*°?\s*F\s+or\s+above", re.IGNORECASE)
# Celsius patterns (Asian/international cities)
RANGE_PATTERN_C = re.compile(r"(\d+)\s*[-–]\s*(\d+)\s*°\s*C", re.IGNORECASE)
BELOW_PATTERN_C = re.compile(r"(\d+)\s*°?\s*C\s+or\s+below", re.IGNORECASE)
ABOVE_PATTERN_C = re.compile(r"(\d+)\s*°?\s*C\s+or\s+above", re.IGNORECASE)
# Exact Celsius: "be 8°C on" (single-value buckets)
EXACT_PATTERN_C = re.compile(r"be\s+(\d+)\s*°\s*C\s+on\s+", re.IGNORECASE)

# Open-Meteo archive API, always use free tier (customer archive needs Professional plan)
ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"


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

                    # Check resolution status
                    # Polymarket marks resolved markets with "resolved" flag or
                    # outcome data showing which side won
                    is_resolved = mkt.get("resolved", False)
                    if not is_resolved:
                        # Also check if the market is closed with a clear outcome
                        if not mkt.get("closed", False):
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
                                if yes_price >= 0.95:
                                    resolved[condition_id] = True
                                elif no_price >= 0.95:
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
    """Fetch actual observed high temperature from Open-Meteo archive API.

    Args:
        city_id: City enum value (e.g., "nyc", "den").
        target_date: The date to look up observations for.

    Returns:
        Observed high temperature in Celsius, or None if unavailable.
    """
    # Look up city coordinates
    try:
        city_enum = City(city_id)
    except ValueError:
        logger.warning("Unknown city_id for observation lookup: %s", city_id)
        return None

    if city_enum not in CITIES:
        return None

    city_config = CITIES[city_enum]

    async with httpx.AsyncClient() as client:
        try:
            _params = {
                    "latitude": city_config.latitude,
                    "longitude": city_config.longitude,
                    "start_date": target_date.isoformat(),
                    "end_date": target_date.isoformat(),
                    "daily": "temperature_2m_max",
                    "timezone": "auto",
                }
            # Always use free archive tier (no apikey needed)
            resp = await client.get(
                ARCHIVE_API_URL,
                params=_params,
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
            try:
                from weather_edge.analysis.service_health import record_service_call
                record_service_call("openmeteo_archive", True)
            except Exception:
                pass
        except (httpx.HTTPError, ValueError) as e:
            logger.warning("Failed to fetch observations for %s on %s: %s", city_id, target_date, e)
            try:
                from weather_edge.analysis.service_health import record_service_call
                record_service_call("openmeteo_archive", False)
            except Exception:
                pass
            return None

    # Parse response
    daily = data.get("daily", {})
    temps = daily.get("temperature_2m_max", [])
    if temps and temps[0] is not None:
        temp_c = float(temps[0])
        logger.info(
            "Observed high for %s on %s: %.1f°C (%.1f°F)",
            city_id, target_date, temp_c, c_to_f(temp_c),
        )
        # Backfill actual value in forecast snapshots for self-learning
        try:
            from weather_edge.dashboard.app import paper_trader
            updated = paper_trader.store.backfill_actual(
                city_id, str(target_date), temp_c,
            )
            if updated:
                logger.info("Backfilled %d forecast snapshots for %s %s",
                           updated, city_id, target_date)
        except Exception:
            pass
        return temp_c

    return None


def _c_to_f(c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return c * 9.0 / 5.0 + 32.0


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


def actual_falls_in_bucket(actual_temp_c: float, bucket: BucketInfo) -> bool:
    """Check if an actual temperature falls within a bucket's range.

    Compares in native units to avoid floating-point conversion errors.

    Args:
        actual_temp_c: Actual observed temperature in Celsius.
        bucket: BucketInfo from parse_bucket_from_description.

    Returns:
        True if the actual temperature falls in this bucket.
    """
    # Convert actual to bucket's native unit
    if bucket.unit == "fahrenheit":
        actual = _c_to_f(actual_temp_c)
    else:
        actual = actual_temp_c

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
        month_map = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
        }
        month_name = m.group(1).lower()
        day = int(m.group(2))
        month = month_map.get(month_name, 1)
        year = date.today().year
        try:
            d = date(year, month, day)
            # If the date is more than 6 months in the past, it's probably next year
            if (date.today() - d).days > 180:
                d = date(year + 1, month, day)
            return d
        except ValueError:
            pass
    return None


async def resolve_open_trades(paper_trader: PaperTrader) -> int:
    """Resolve any open trades whose markets have settled.

    Checks Polymarket for resolved markets first, then falls back to
    Open-Meteo archive observations for trades past their target date.

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
            from zoneinfo import ZoneInfo
            from datetime import datetime
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
        yes_won = actual_falls_in_bucket(actual_temp_c, bucket)
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
