"""Polymarket market discovery and price fetching.

Real Polymarket weather market format (from Gamma API):
- Events: "Highest temperature in Denver on March 27?"
- Markets (per event): ~11 buckets like "49°F or below", "50-51°F", "52-53°F", etc.
- Each market has outcomePrices [YES_price, NO_price] and clobTokenIds [YES_token, NO_token]
- Resolution: NWS observations for US, Met Office for London
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import NamedTuple

import httpx

from weather_edge.config import Settings
from weather_edge.fetchers.openmeteo import f_to_c
from weather_edge.models.enums import City, MarketType

logger = logging.getLogger(__name__)

# Map city names in Polymarket descriptions to our city IDs
CITY_NAME_MAP: dict[str, City] = {
    "new york": City.NYC,
    "nyc": City.NYC,
    "london": City.LON,
    "dallas": City.DAL,
    "seattle": City.SEA,
    "atlanta": City.ATL,
    "toronto": City.TOR,
    "buenos aires": City.BUE,
    "seoul": City.SEL,
    "chicago": City.CHI,
    "miami": City.MIA,
    "los angeles": City.LAX,
    "houston": City.HOU,
    "denver": City.DEN,
    "san francisco": City.SFO,
    "shanghai": City.SHA,
    "madrid": City.MAD,
    "tokyo": City.TYO,
    "hong kong": City.HKG,
    "munich": City.MUC,
    "warsaw": City.WAR,
    "shenzhen": City.SZN,
    "austin": City.AUS,
}

# Parse date from event titles: "... on March 27?"
DATE_PATTERN = re.compile(
    r"on\s+(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2})",
    re.IGNORECASE,
)

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

# Parse bucket ranges from market questions
# "Will the highest temperature in Denver be 49°F or below on March 27?"
BUCKET_BELOW_PATTERN = re.compile(
    r"(?:highest|high)\s+temperature\s+in\s+(.+?)\s+be\s+(\d+)\s*°?\s*(F|C)\s+or\s+below",
    re.IGNORECASE,
)
# "Will the highest temperature in Denver be between 50-51°F on March 27?"
BUCKET_RANGE_PATTERN = re.compile(
    r"(?:highest|high)\s+temperature\s+in\s+(.+?)\s+be\s+(?:between\s+)?(\d+)\s*[-–]\s*(\d+)\s*°?\s*(F|C)",
    re.IGNORECASE,
)
# "Will the highest temperature in Denver be 60°F or above on March 27?"
BUCKET_ABOVE_PATTERN = re.compile(
    r"(?:highest|high)\s+temperature\s+in\s+(.+?)\s+be\s+(\d+)\s*°?\s*(F|C)\s+or\s+above",
    re.IGNORECASE,
)
# Snow pattern
SNOW_PATTERN = re.compile(
    r"(?:will\s+it\s+)?snow\s+in\s+(.+?)\s+(?:on|tomorrow)",
    re.IGNORECASE,
)


class TempBucket(NamedTuple):
    """A temperature range bucket from a Polymarket event."""
    low_f: float | None  # None means -inf
    high_f: float | None  # None means +inf
    low_c: float | None
    high_c: float | None


@dataclass
class MarketInfo:
    """Parsed Polymarket weather market."""
    market_id: str
    condition_id: str = ""
    token_id_yes: str | None = None
    token_id_no: str | None = None
    city_id: City | None = None
    city_name: str = ""  # Raw city name from Polymarket
    market_type: MarketType = MarketType.TEMP_HIGH
    description: str = ""
    question: str = ""
    event_title: str = ""
    target_date: date = field(default_factory=date.today)
    # For temperature buckets
    threshold_value: float = 0.0  # Celsius
    threshold_dir: str = "gte"  # gte, lte, range, any
    threshold_low_c: float | None = None  # For range buckets
    threshold_high_c: float | None = None
    threshold_unit: str = "fahrenheit"
    # Price from Gamma API (avoids CLOB calls)
    yes_price: float = 0.0
    no_price: float = 0.0
    resolution_source: str = "nws"
    slug: str = ""
    volume_24h: float = 0.0
    liquidity: float = 0.0


@dataclass
class PriceSnapshot:
    """Market price at a point in time."""
    market_id: str
    fetched_at: datetime
    bid: float | None = None
    ask: float | None = None
    midpoint: float | None = None
    spread: float | None = None
    volume_24h: float | None = None
    liquidity: float | None = None


def parse_city_from_text(text: str) -> tuple[City | None, str]:
    """Extract city from market description text. Returns (city_id, raw_city_name)."""
    text_lower = text.lower()
    for name, city in sorted(CITY_NAME_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if name in text_lower:
            return city, name.title()
    return None, ""


def parse_target_date(text: str, fallback_end_date: str | None = None) -> date:
    """Extract target date from event title or endDate."""
    m = DATE_PATTERN.search(text)
    if m:
        month_name = m.group(1).lower()
        day = int(m.group(2))
        month = MONTH_MAP.get(month_name, 1)
        year = date.today().year
        try:
            d = date(year, month, day)
            # If the date is more than 6 months in the past, it's probably next year
            if (date.today() - d).days > 180:
                d = date(year + 1, month, day)
            return d
        except ValueError:
            pass

    if fallback_end_date:
        try:
            return datetime.fromisoformat(
                fallback_end_date.replace("Z", "+00:00")
            ).date()
        except (ValueError, AttributeError):
            pass

    return date.today()


def parse_market_question(
    question: str,
    event_title: str,
    condition_id: str,
    end_date: str | None = None,
) -> MarketInfo | None:
    """Parse a Polymarket market question into structured data."""

    target_date = parse_target_date(event_title, end_date)

    # Try "X°F or below" pattern
    m = BUCKET_BELOW_PATTERN.search(question)
    if m:
        city_id, city_name = parse_city_from_text(m.group(1))
        if city_id is None:
            city_id, city_name = parse_city_from_text(event_title)
        high_f = float(m.group(2))
        unit = m.group(3)
        high_c = f_to_c(high_f) if unit.upper() == "F" else high_f

        return MarketInfo(
            market_id=condition_id,
            condition_id=condition_id,
            city_id=city_id,
            city_name=city_name,
            market_type=MarketType.TEMP_HIGH,
            question=question,
            event_title=event_title,
            target_date=target_date,
            threshold_value=high_c,
            threshold_dir="lte",
            threshold_low_c=None,
            threshold_high_c=high_c,
            threshold_unit="fahrenheit" if unit.upper() == "F" else "celsius",
        )

    # Try "between X-Y°F" pattern
    m = BUCKET_RANGE_PATTERN.search(question)
    if m:
        city_id, city_name = parse_city_from_text(m.group(1))
        if city_id is None:
            city_id, city_name = parse_city_from_text(event_title)
        low_f = float(m.group(2))
        high_f = float(m.group(3))
        unit = m.group(4)
        if unit.upper() == "F":
            low_c = f_to_c(low_f)
            high_c = f_to_c(high_f)
        else:
            low_c = low_f
            high_c = high_f

        return MarketInfo(
            market_id=condition_id,
            condition_id=condition_id,
            city_id=city_id,
            city_name=city_name,
            market_type=MarketType.TEMP_HIGH,
            question=question,
            event_title=event_title,
            target_date=target_date,
            threshold_value=(low_c + high_c) / 2,  # midpoint for display
            threshold_dir="range",
            threshold_low_c=low_c,
            threshold_high_c=high_c,
            threshold_unit="fahrenheit" if unit.upper() == "F" else "celsius",
        )

    # Try "X°F or above" pattern
    m = BUCKET_ABOVE_PATTERN.search(question)
    if m:
        city_id, city_name = parse_city_from_text(m.group(1))
        if city_id is None:
            city_id, city_name = parse_city_from_text(event_title)
        low_f = float(m.group(2))
        unit = m.group(3)
        low_c = f_to_c(low_f) if unit.upper() == "F" else low_f

        return MarketInfo(
            market_id=condition_id,
            condition_id=condition_id,
            city_id=city_id,
            city_name=city_name,
            market_type=MarketType.TEMP_HIGH,
            question=question,
            event_title=event_title,
            target_date=target_date,
            threshold_value=low_c,
            threshold_dir="gte",
            threshold_low_c=low_c,
            threshold_high_c=None,
            threshold_unit="fahrenheit" if unit.upper() == "F" else "celsius",
        )

    # Try snow
    m = SNOW_PATTERN.search(question)
    if m:
        city_id, city_name = parse_city_from_text(m.group(1))
        if city_id is None:
            city_id, city_name = parse_city_from_text(event_title)
        return MarketInfo(
            market_id=condition_id,
            condition_id=condition_id,
            city_id=city_id,
            city_name=city_name,
            market_type=MarketType.SNOW,
            question=question,
            event_title=event_title,
            target_date=target_date,
            threshold_value=0.0,
            threshold_dir="any",
            threshold_unit="cm",
        )

    return None


async def discover_weather_markets(
    settings: Settings | None = None,
) -> list[MarketInfo]:
    """Discover active weather markets from Polymarket's Gamma API.

    Paginates through weather-tagged events and parses multi-bucket temperature markets.
    Uses outcomePrices from Gamma API directly (no CLOB calls needed for price discovery).
    """
    if settings is None:
        from weather_edge.config import settings as _settings
        settings = _settings

    markets: list[MarketInfo] = []
    today = date.today()

    async with httpx.AsyncClient() as client:
        for offset in range(0, 500, 100):
            try:
                resp = await client.get(
                    f"{settings.polymarket_gamma_url}/events",
                    params={
                        "tag_slug": "weather",
                        "active": "true",
                        "closed": "false",
                        "limit": 100,
                        "offset": offset,
                    },
                    timeout=15.0,
                )
                resp.raise_for_status()
                events = resp.json()
            except (httpx.HTTPError, ValueError) as e:
                logger.error("Failed to fetch Polymarket events at offset %d: %s", offset, e)
                break

            if not events:
                break

            for event in events:
                event_title = event.get("title", "")
                end_date = event.get("endDate")

                # Only process temperature/snow events for our tracked cities
                title_lower = event_title.lower()
                if "highest temperature" not in title_lower and "snow" not in title_lower:
                    continue

                event_markets = event.get("markets", [])
                for mkt in event_markets:
                    condition_id = mkt.get("conditionId", "")
                    question = mkt.get("question", "")
                    if not question:
                        continue

                    parsed = parse_market_question(question, event_title, condition_id, end_date)
                    if parsed is None:
                        continue

                    # Skip markets for dates that have already passed
                    if parsed.target_date < today:
                        continue

                    # Extract token IDs
                    tokens = mkt.get("clobTokenIds") or []
                    if len(tokens) >= 2:
                        parsed.token_id_yes = tokens[0]
                        parsed.token_id_no = tokens[1]

                    # Get prices directly from Gamma (avoids CLOB rate limits)
                    outcome_prices = mkt.get("outcomePrices")
                    if outcome_prices and isinstance(outcome_prices, list) and len(outcome_prices) >= 2:
                        try:
                            parsed.yes_price = float(outcome_prices[0])
                            parsed.no_price = float(outcome_prices[1])
                        except (ValueError, TypeError):
                            pass
                    elif isinstance(outcome_prices, str):
                        # Sometimes it's a JSON string
                        try:
                            import json
                            prices = json.loads(outcome_prices)
                            if isinstance(prices, list) and len(prices) >= 2:
                                parsed.yes_price = float(prices[0])
                                parsed.no_price = float(prices[1])
                        except (ValueError, TypeError):
                            pass

                    parsed.description = question
                    parsed.slug = mkt.get("slug", "")
                    # Volume and liquidity from Gamma API
                    try:
                        parsed.volume_24h = float(mkt.get("volumeNum") or mkt.get("volume") or 0)
                    except (ValueError, TypeError):
                        pass
                    try:
                        parsed.liquidity = float(mkt.get("liquidityNum") or mkt.get("liquidity") or 0)
                    except (ValueError, TypeError):
                        pass
                    markets.append(parsed)

            logger.info("Fetched %d events at offset %d, %d markets so far", len(events), offset, len(markets))

            if len(events) < 100:
                break

    # Filter to only cities we track
    tracked_cities = set(City)
    tracked = [m for m in markets if m.city_id in tracked_cities]

    logger.info(
        "Discovered %d total markets, %d for tracked cities",
        len(markets), len(tracked),
    )
    return tracked


def get_price_snapshot(market: MarketInfo) -> PriceSnapshot:
    """Convert a MarketInfo's embedded price to a PriceSnapshot."""
    return PriceSnapshot(
        market_id=market.market_id,
        fetched_at=datetime.now(timezone.utc),
        midpoint=market.yes_price,
        bid=None,
        ask=None,
        spread=None,
    )


async def fetch_market_price(
    token_id: str,
    settings: Settings | None = None,
) -> PriceSnapshot | None:
    """Fetch current price for a single market token from CLOB API."""
    if settings is None:
        from weather_edge.config import settings as _settings
        settings = _settings

    now = datetime.now(timezone.utc)

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                f"{settings.polymarket_clob_url}/midpoint",
                params={"token_id": token_id},
                timeout=10.0,
            )
            resp.raise_for_status()
            mid_data = resp.json()
            midpoint = float(mid_data.get("mid", 0))
        except (httpx.HTTPError, ValueError, KeyError) as e:
            logger.warning("Failed to fetch midpoint for %s: %s", token_id[:20], e)
            return None

        bid = None
        ask = None
        spread = None
        try:
            resp = await client.get(
                f"{settings.polymarket_clob_url}/book",
                params={"token_id": token_id},
                timeout=10.0,
            )
            resp.raise_for_status()
            book = resp.json()
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            if bids:
                bid = float(bids[0].get("price", 0))
            if asks:
                ask = float(asks[0].get("price", 0))
            if bid is not None and ask is not None:
                spread = ask - bid
        except (httpx.HTTPError, ValueError) as e:
            logger.debug("Failed to fetch book for %s: %s", token_id[:20], e)

    return PriceSnapshot(
        market_id=token_id,
        fetched_at=now,
        bid=bid,
        ask=ask,
        midpoint=midpoint,
        spread=spread,
    )


async def fetch_book_prices(market: "MarketInfo") -> dict | None:
    """Fetch top-of-book ask prices for both YES and NO sides of a market.

    Returns dict with yes_ask, no_ask, spread_cost (yes_ask + no_ask).
    Spread capture is profitable when spread_cost < 1.0.
    """
    if not market.token_id_yes or not market.token_id_no:
        return None

    from weather_edge.config import settings

    result = {"yes_ask": None, "no_ask": None, "yes_bid": None, "no_bid": None}

    async with httpx.AsyncClient() as client:
        for side, token_id in [("yes", market.token_id_yes), ("no", market.token_id_no)]:
            try:
                resp = await client.get(
                    f"{settings.polymarket_clob_url}/book",
                    params={"token_id": token_id},
                    timeout=10.0,
                )
                resp.raise_for_status()
                book = resp.json()
                bids = book.get("bids", [])
                asks = book.get("asks", [])
                if asks:
                    result[f"{side}_ask"] = float(asks[0].get("price", 0))
                if bids:
                    result[f"{side}_bid"] = float(bids[0].get("price", 0))
            except (httpx.HTTPError, ValueError) as e:
                logger.debug("Book fetch failed for %s %s: %s", side, token_id[:20], e)

    if result["yes_ask"] and result["no_ask"]:
        result["spread_cost"] = result["yes_ask"] + result["no_ask"]
        result["spread_profit"] = 1.0 - result["spread_cost"]
        result["profitable"] = result["spread_cost"] < 0.97  # 3% buffer to absorb round-trip taker fees
    else:
        result["spread_cost"] = None
        result["spread_profit"] = None
        result["profitable"] = False

    return result
