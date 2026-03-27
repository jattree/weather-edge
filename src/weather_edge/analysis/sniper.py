"""Model-drop sniper, triggers trades immediately when fresh forecast data appears.

Like eBay sniping: sit quietly monitoring model release schedules, then the instant
new data drops (especially ECMWF), fire off a cycle before the market adjusts.

The key insight: Polymarket weather prices lag model updates by 5-15 minutes.
If we detect fresh data within 60 seconds, we get first-mover advantage on
every consensus shift.

Detection strategy (v2, metadata-based):
1. Poll Open-Meteo with minimal payload (single variable, lat=0/lon=0)
2. Track `generationtime_ms` per model, changes indicate new data ingested
3. When change detected, wait 60s for cluster consistency across Open-Meteo CDN
4. Then trigger the full cycle via callback
5. Prioritize cities by Polymarket volume: SEA, DAL, ATL, NYC, DEN first

This replaces the old approach of fetching full forecast data every 2 minutes.
The metadata probe is ~10x lighter and detects updates faster.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import httpx

from weather_edge.config import CITIES, settings
from weather_edge.models.enums import City, WeatherModel

logger = logging.getLogger(__name__)

# Which models to monitor and their expected update frequency
SNIPE_TARGETS: dict[WeatherModel, dict] = {
    WeatherModel.ECMWF: {
        "priority": 1,        # Highest priority, moves all markets
        "shift_threshold_c": 1.0,  # Trigger on 1°C+ shift
        "label": "ECMWF",
    },
    WeatherModel.GFS: {
        "priority": 2,
        "shift_threshold_c": 1.5,
        "label": "GFS",
    },
    WeatherModel.HRRR: {
        "priority": 3,
        "shift_threshold_c": 1.5,
        "label": "HRRR",
    },
}

# Expected model availability times (model run time + computation + Open-Meteo ingest)
# Used to focus polling around expected windows and avoid wasting cycles
MODEL_AVAILABILITY: dict[WeatherModel, list[dict]] = {
    WeatherModel.ECMWF: [
        {"run": "00z", "available_utc_hour": 7, "available_utc_minute": 0},   # T+7h
        {"run": "12z", "available_utc_hour": 19, "available_utc_minute": 0},  # T+7h
    ],
    WeatherModel.GFS: [
        {"run": "00z", "available_utc_hour": 4, "available_utc_minute": 0},   # T+4h
        {"run": "06z", "available_utc_hour": 10, "available_utc_minute": 0},
        {"run": "12z", "available_utc_hour": 16, "available_utc_minute": 0},
        {"run": "18z", "available_utc_hour": 22, "available_utc_minute": 0},
    ],
    WeatherModel.HRRR: [
        # Every hour, available ~T+2h, generate dynamically
        {"run": f"{h:02d}z", "available_utc_hour": (h + 2) % 24, "available_utc_minute": 0}
        for h in range(24)
    ],
}

# City priority order by Polymarket volume, fetch these first after a model drop
CITY_PRIORITY: list[City] = [
    City.SEA, City.DAL, City.ATL, City.NYC, City.DEN,
    # Remaining cities in default order
    City.CHI, City.MIA, City.LAX, City.HOU, City.SFO,
    City.LON, City.TOR, City.BUE, City.SEL, City.SHA,
    City.MAD, City.TYO, City.HKG, City.MUC, City.WAR, City.SZN,
]

# Seconds to wait after detecting new data before triggering cycle
# Allows Open-Meteo's CDN cluster to reach consistency
CLUSTER_CONSISTENCY_DELAY_SECONDS = 60


@dataclass
class ModelSnapshot:
    """Cached forecast from a model, used to detect when new data appears."""
    model: WeatherModel
    temp_max_c: float | None = None
    fetched_at: datetime | None = None
    reference_time: str | None = None  # Open-Meteo's generationtime_ms or similar


@dataclass
class MetadataProbe:
    """Lightweight probe result tracking generationtime_ms for update detection."""
    model: WeatherModel
    generationtime_ms: float
    probed_at: datetime


@dataclass
class SniperEvent:
    """A detected model-drop event."""
    model: WeatherModel
    detected_at: datetime
    old_temp_c: float | None
    new_temp_c: float | None
    shift_c: float
    city_id: str
    target_date: str
    priority: int


def _is_near_availability_window(
    model: WeatherModel,
    now: datetime | None = None,
    window_minutes: int = 90,
) -> bool:
    """Check if current time is within `window_minutes` of any expected availability.

    Used to focus polling effort. Outside these windows, we can skip probes
    for models that are unlikely to have new data.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    windows = MODEL_AVAILABILITY.get(model, [])
    if not windows:
        return True  # Unknown schedule, always probe

    for w in windows:
        avail = now.replace(
            hour=w["available_utc_hour"],
            minute=w["available_utc_minute"],
            second=0,
            microsecond=0,
        )
        # Check if we're within the window before or after the expected time
        delta = (now - avail).total_seconds() / 60
        if -window_minutes <= delta <= window_minutes:
            return True

    return False


class ModelSniper:
    """Monitors model updates via lightweight metadata probes and triggers trades."""

    def __init__(self):
        self._cache: dict[str, ModelSnapshot] = {}  # model_name -> last known snapshot
        self._generation_cache: dict[str, float] = {}  # model_name -> last generationtime_ms
        self._events: list[SniperEvent] = []
        self._snipe_callback = None  # async callable to run when snipe triggers

    def set_callback(self, callback) -> None:
        """Set the async function to call when a snipe triggers."""
        self._snipe_callback = callback

    @property
    def recent_events(self) -> list[SniperEvent]:
        """Events from the last hour."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        return [e for e in self._events if e.detected_at > cutoff]

    @property
    def city_priority(self) -> list[City]:
        """Cities ordered by Polymarket trading volume for fetch prioritization."""
        return CITY_PRIORITY

    async def probe_metadata(
        self,
        model: WeatherModel,
    ) -> MetadataProbe | None:
        """Ultra-lightweight probe, fetch minimal payload to check generationtime_ms.

        Uses lat=0, lon=0 with only temperature_2m for 1 day. The key data point
        is `generationtime_ms` in the response, which changes when Open-Meteo
        ingests a new model run.
        """
        model_id = model.value

        async with httpx.AsyncClient() as client:
            try:
                _params = {
                        "latitude": 0,
                        "longitude": 0,
                        "hourly": "temperature_2m",
                        "forecast_days": 1,
                        "models": model_id,
                    }
                if settings.openmeteo_api_key:
                    _params["apikey"] = settings.openmeteo_api_key
                resp = await client.get(
                    settings.effective_openmeteo_url,
                    params=_params,
                    timeout=10.0,
                )
                resp.raise_for_status()
                data = resp.json()
            except (httpx.HTTPError, ValueError) as e:
                logger.debug("Metadata probe failed for %s: %s", model_id, e)
                return None

        gen_time = data.get("generationtime_ms")
        if gen_time is None:
            logger.debug("No generationtime_ms in response for %s", model_id)
            return None

        return MetadataProbe(
            model=model,
            generationtime_ms=float(gen_time),
            probed_at=datetime.now(timezone.utc),
        )

    async def probe_model(
        self,
        model: WeatherModel,
        city_id: City | None = None,
        target_date: date | None = None,
    ) -> ModelSnapshot | None:
        """Full probe, fetch daily max temp for one city/model (used after detection).

        This is only called when metadata probe detects a change, to measure
        the actual temperature shift for snipe event logging.
        """
        if city_id is None:
            city_id = CITY_PRIORITY[0] if CITY_PRIORITY else City.NYC
        if target_date is None:
            target_date = date.today() + timedelta(days=1)

        city = CITIES[city_id]
        model_id = model.value

        async with httpx.AsyncClient() as client:
            try:
                _params2 = {
                        "latitude": city.latitude,
                        "longitude": city.longitude,
                        "daily": "temperature_2m_max",
                        "models": model_id,
                        "timezone": "UTC",
                        "start_date": str(target_date),
                        "end_date": str(target_date),
                    }
                if settings.openmeteo_api_key:
                    _params2["apikey"] = settings.openmeteo_api_key
                resp = await client.get(
                    settings.effective_openmeteo_url,
                    params=_params2,
                    timeout=10.0,
                )
                resp.raise_for_status()
                data = resp.json()
            except (httpx.HTTPError, ValueError) as e:
                logger.debug("Full probe failed for %s: %s", model_id, e)
                return None

        daily = data.get("daily", {})
        temps = daily.get("temperature_2m_max", [])
        gen_time = str(data.get("generationtime_ms", ""))

        return ModelSnapshot(
            model=model,
            temp_max_c=temps[0] if temps else None,
            fetched_at=datetime.now(timezone.utc),
            reference_time=gen_time,
        )

    async def check_for_updates(self) -> list[SniperEvent]:
        """Probe all monitored models via lightweight metadata check.

        Two-phase detection:
        1. Metadata probe: check generationtime_ms for each model (cheap)
        2. Full probe: only if metadata changed, fetch actual temp to log shift

        Returns list of snipe events (models with new data detected).
        """
        now = datetime.now(timezone.utc)
        tomorrow = date.today() + timedelta(days=1)
        events: list[SniperEvent] = []
        models_with_new_data: list[WeatherModel] = []

        # Phase 1: Lightweight metadata probes for all models
        for model, config in SNIPE_TARGETS.items():
            # Skip models outside their expected availability window
            if not _is_near_availability_window(model, now):
                logger.debug(
                    "Skipping %s probe, outside availability window",
                    config["label"],
                )
                continue

            probe = await self.probe_metadata(model)
            if probe is None:
                continue

            cache_key = model.value
            old_gen = self._generation_cache.get(cache_key)

            if old_gen is not None and probe.generationtime_ms != old_gen:
                logger.info(
                    "NEW DATA DETECTED: %s generationtime_ms changed %.2f -> %.2f",
                    config["label"],
                    old_gen,
                    probe.generationtime_ms,
                )
                models_with_new_data.append(model)

            # Update generation cache
            self._generation_cache[cache_key] = probe.generationtime_ms

        # Phase 2: For models with new data, wait for cluster consistency then full probe
        if models_with_new_data:
            logger.info(
                "Waiting %ds for CDN cluster consistency before full probe...",
                CLUSTER_CONSISTENCY_DELAY_SECONDS,
            )
            await asyncio.sleep(CLUSTER_CONSISTENCY_DELAY_SECONDS)

            for model in models_with_new_data:
                config = SNIPE_TARGETS[model]
                snapshot = await self.probe_model(model, target_date=tomorrow)
                if snapshot is None or snapshot.temp_max_c is None:
                    continue

                temp_cache_key = f"{model.value}_{tomorrow}"
                old = self._cache.get(temp_cache_key)

                shift_c = 0.0
                if old is not None and old.temp_max_c is not None:
                    shift_c = snapshot.temp_max_c - old.temp_max_c

                # Always create an event when new data is detected, the model drop
                # itself is the trigger, not just the temperature shift
                event = SniperEvent(
                    model=model,
                    detected_at=datetime.now(timezone.utc),
                    old_temp_c=old.temp_max_c if old else None,
                    new_temp_c=snapshot.temp_max_c,
                    shift_c=shift_c,
                    city_id=CITY_PRIORITY[0].value if CITY_PRIORITY else "nyc",
                    target_date=str(tomorrow),
                    priority=config["priority"],
                )
                events.append(event)
                self._events.append(event)

                if abs(shift_c) >= config["shift_threshold_c"]:
                    logger.warning(
                        "SNIPE DETECTED: %s shifted %+.1f°C for %s on %s "
                        "(%.1f -> %.1f), TRIGGERING TRADE CYCLE",
                        config["label"],
                        event.shift_c,
                        CITY_PRIORITY[0].value.upper() if CITY_PRIORITY else "NYC",
                        tomorrow,
                        old.temp_max_c if old and old.temp_max_c else 0.0,
                        snapshot.temp_max_c,
                    )
                else:
                    logger.info(
                        "Model drop: %s new data, shift %+.1f°C (below threshold %.1f°C)",
                        config["label"],
                        shift_c,
                        config["shift_threshold_c"],
                    )

                # Update temp cache
                self._cache[temp_cache_key] = snapshot

        return events

    async def run_sniper_loop(self, poll_interval_seconds: int = 60) -> None:
        """Main sniper loop, polls metadata every 60 seconds, triggers on model drops.

        This runs alongside the regular 30-minute cycle. The regular cycle
        handles steady-state trading. The sniper handles time-sensitive
        opportunities when models shift.

        v2: Uses lightweight metadata probes (generationtime_ms comparison)
        instead of full forecast fetches. ~10x less API load.
        """
        logger.info(
            "Sniper armed (v2 metadata-based), monitoring %s every %ds",
            ", ".join(cfg["label"] for cfg in SNIPE_TARGETS.values()),
            poll_interval_seconds,
        )

        while True:
            try:
                events = await self.check_for_updates()

                if events and self._snipe_callback:
                    # Sort by priority, ECMWF triggers first
                    events.sort(key=lambda e: e.priority)
                    best = events[0]

                    logger.warning(
                        "SNIPING: %s new data detected (shift %+.1f°C), "
                        "executing immediate cycle (cities: %s first)",
                        best.model.value,
                        best.shift_c,
                        ", ".join(c.value.upper() for c in CITY_PRIORITY[:5]),
                    )

                    # Fire the callback (should be run_dashboard_cycle or similar)
                    await self._snipe_callback()

            except Exception:
                logger.exception("Sniper probe failed")

            await asyncio.sleep(poll_interval_seconds)
