"""Model-drop sniper, triggers trades immediately when fresh forecast data appears.

Like eBay sniping: sit quietly monitoring model release schedules, then the instant
new data drops (especially ECMWF), fire off a cycle before the market adjusts.

The key insight: Polymarket weather prices lag model updates by 5-15 minutes.
If we detect fresh data within 60 seconds, we get first-mover advantage on
every consensus shift.

Flow:
1. Poll Open-Meteo every 2 minutes (lightweight, just check headers/timestamps)
2. When a model's reference_time changes → new data detected
3. Compare new forecast to cached previous forecast
4. If shift > threshold → SNIPE: trigger immediate full cycle
5. Log the shift for the dashboard
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

# City to monitor for each model (use one representative city to detect updates)
PROBE_CITY = City.NYC  # NYC has all models available


@dataclass
class ModelSnapshot:
    """Cached forecast from a model, used to detect when new data appears."""
    model: WeatherModel
    temp_max_c: float | None = None
    fetched_at: datetime | None = None
    reference_time: str | None = None  # Open-Meteo's generationtime_ms or similar


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


class ModelSniper:
    """Monitors model updates and triggers trades on significant shifts."""

    def __init__(self):
        self._cache: dict[str, ModelSnapshot] = {}  # model_name -> last known snapshot
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

    async def probe_model(
        self,
        model: WeatherModel,
        city_id: City = PROBE_CITY,
        target_date: date | None = None,
    ) -> ModelSnapshot | None:
        """Lightweight probe, fetch just the daily max temp for one city/model."""
        if target_date is None:
            target_date = date.today() + timedelta(days=1)

        city = CITIES[city_id]
        model_id = model.value

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(
                    settings.openmeteo_base_url,
                    params={
                        "latitude": city.latitude,
                        "longitude": city.longitude,
                        "daily": "temperature_2m_max",
                        "models": model_id,
                        "timezone": "UTC",
                        "start_date": str(target_date),
                        "end_date": str(target_date),
                    },
                    timeout=10.0,
                )
                resp.raise_for_status()
                data = resp.json()
            except (httpx.HTTPError, ValueError) as e:
                logger.debug("Probe failed for %s: %s", model_id, e)
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
        """Probe all monitored models and detect shifts.

        Returns list of snipe events (models with significant forecast changes).
        """
        tomorrow = date.today() + timedelta(days=1)
        events: list[SniperEvent] = []

        for model, config in SNIPE_TARGETS.items():
            snapshot = await self.probe_model(model, target_date=tomorrow)
            if snapshot is None or snapshot.temp_max_c is None:
                continue

            cache_key = f"{model.value}_{tomorrow}"
            old = self._cache.get(cache_key)

            if old is not None and old.temp_max_c is not None:
                shift = abs(snapshot.temp_max_c - old.temp_max_c)

                if shift >= config["shift_threshold_c"]:
                    event = SniperEvent(
                        model=model,
                        detected_at=datetime.now(timezone.utc),
                        old_temp_c=old.temp_max_c,
                        new_temp_c=snapshot.temp_max_c,
                        shift_c=snapshot.temp_max_c - old.temp_max_c,
                        city_id=PROBE_CITY.value,
                        target_date=str(tomorrow),
                        priority=config["priority"],
                    )
                    events.append(event)
                    self._events.append(event)

                    logger.warning(
                        "SNIPE DETECTED: %s shifted %+.1f°C for %s on %s "
                        "(%.1f -> %.1f), TRIGGERING TRADE CYCLE",
                        config["label"],
                        event.shift_c,
                        PROBE_CITY.value.upper(),
                        tomorrow,
                        old.temp_max_c,
                        snapshot.temp_max_c,
                    )

            # Update cache
            self._cache[cache_key] = snapshot

        return events

    async def run_sniper_loop(self, poll_interval_seconds: int = 120) -> None:
        """Main sniper loop, polls every 2 minutes, triggers on model drops.

        This runs alongside the regular 30-minute cycle. The regular cycle
        handles steady-state trading. The sniper handles time-sensitive
        opportunities when models shift.
        """
        logger.info(
            "Sniper armed, monitoring %s every %ds",
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
                        "SNIPING: %s %+.1f°C shift, executing immediate cycle",
                        best.model.value, best.shift_c,
                    )

                    # Fire the callback (should be run_dashboard_cycle or similar)
                    await self._snipe_callback()

            except Exception:
                logger.exception("Sniper probe failed")

            await asyncio.sleep(poll_interval_seconds)
