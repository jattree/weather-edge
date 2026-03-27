"""GribStream AI model fetcher, GraphCast, AIFS, FourCastNet.

Supplements Open-Meteo physics models with AI weather models.
When AI and physics diverge by >3°C, signals reduced confidence.
When they agree, signals increased confidence.

Free tier: 1,200 credits/day. Each city fetch = ~4 credits.
At 22 cities per cycle, 2 cycles/hour = ~352 credits/day (29% of quota).
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from dataclasses import dataclass

import httpx

from weather_edge.config import CITIES
from weather_edge.models.enums import City

logger = logging.getLogger(__name__)

GRIBSTREAM_API_URL = "https://gribstream.com/api/v2"
KELVIN_OFFSET = 273.15

# AI models available on GribStream
AI_MODELS = {
    "graphcast": "GraphCast (Google DeepMind)",
}


def _get_api_key() -> str:
    import os
    key = os.environ.get("GRIBSTREAM_API_KEY", "")
    if not key:
        try:
            from weather_edge.config import settings
            key = getattr(settings, "gribstream_api_key", "")
        except Exception:
            pass
    return key


@dataclass
class AIModelForecast:
    """Forecast from an AI weather model."""
    city_id: str
    model_name: str  # "graphcast", "aifs", etc.
    target_date: date
    temp_max_c: float | None
    temp_min_c: float | None
    fetched_at: datetime
    hourly_temps_c: list[float]  # 6-hourly temps in Celsius

    @property
    def temp_max_f(self) -> float | None:
        return self.temp_max_c * 9 / 5 + 32 if self.temp_max_c is not None else None


async def fetch_ai_forecast(
    city_id: City,
    target_date: date,
    model: str = "graphcast",
) -> AIModelForecast | None:
    """Fetch AI model forecast for a single city from GribStream.

    Returns AIModelForecast with daily max/min from 6-hourly data.
    """
    api_key = _get_api_key()
    if not api_key:
        return None

    city = CITIES[city_id]
    start = datetime(target_date.year, target_date.month, target_date.day, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=23)

    payload = {
        "fromTime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "untilTime": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "coordinates": [{"lat": city.latitude, "lon": city.longitude, "name": city_id.value}],
        "variables": [{"name": "TMP", "level": "2 m above ground", "info": "", "alias": "temp_2m"}],
    }

    url = f"{GRIBSTREAM_API_URL}/{model}/timeseries"

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "text/csv",
                },
                timeout=15.0,
            )
            resp.raise_for_status()
            csv_text = resp.text
    except (httpx.HTTPError, Exception) as e:
        logger.debug("GribStream %s fetch failed for %s: %s", model, city_id.value, e)
        try:
            from weather_edge.analysis.service_health import record_service_call
            record_service_call("gribstream", False)
        except Exception:
            pass
        return None

    try:
        from weather_edge.analysis.service_health import record_service_call
        record_service_call("gribstream", True)
    except Exception:
        pass

    # Parse CSV response
    lines = csv_text.strip().split("\n")
    if len(lines) < 2:
        return None

    header = lines[0].split(",")
    temp_col = None
    for i, h in enumerate(header):
        if "temp_2m" in h or "TMP" in h:
            temp_col = i
            break

    if temp_col is None:
        return None

    temps_k = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) > temp_col:
            try:
                temps_k.append(float(parts[temp_col]))
            except ValueError:
                pass

    if not temps_k:
        return None

    temps_c = [t - KELVIN_OFFSET for t in temps_k]

    return AIModelForecast(
        city_id=city_id.value,
        model_name=model,
        target_date=target_date,
        temp_max_c=max(temps_c),
        temp_min_c=min(temps_c),
        fetched_at=datetime.now(timezone.utc),
        hourly_temps_c=temps_c,
    )


async def fetch_ai_forecasts_batch(
    cities: list[City],
    target_date: date,
    model: str = "graphcast",
) -> dict[str, AIModelForecast]:
    """Fetch AI model forecasts for multiple cities in one API call.

    GribStream supports multiple coordinates per request, saves credits.
    """
    api_key = _get_api_key()
    if not api_key:
        return {}

    start = datetime(target_date.year, target_date.month, target_date.day, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=23)

    coordinates = []
    for city_id in cities:
        city = CITIES[city_id]
        coordinates.append({"lat": city.latitude, "lon": city.longitude, "name": city_id.value})

    payload = {
        "fromTime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "untilTime": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "coordinates": coordinates,
        "variables": [{"name": "TMP", "level": "2 m above ground", "info": "", "alias": "temp_2m"}],
    }

    url = f"{GRIBSTREAM_API_URL}/{model}/timeseries"

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "text/csv",
                },
                timeout=20.0,
            )
            resp.raise_for_status()
            csv_text = resp.text
    except (httpx.HTTPError, Exception) as e:
        logger.warning("GribStream batch %s fetch failed: %s", model, e)
        return {}

    # Parse CSV, group by city name
    lines = csv_text.strip().split("\n")
    if len(lines) < 2:
        return {}

    header = lines[0].split(",")
    name_col = None
    temp_col = None
    for i, h in enumerate(header):
        if h == "name":
            name_col = i
        if "temp_2m" in h or "TMP" in h:
            temp_col = i

    if name_col is None or temp_col is None:
        return {}

    # Collect temps per city
    city_temps: dict[str, list[float]] = {}
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) > max(name_col, temp_col):
            city_name = parts[name_col]
            try:
                temp_k = float(parts[temp_col])
                city_temps.setdefault(city_name, []).append(temp_k - KELVIN_OFFSET)
            except ValueError:
                pass

    now = datetime.now(timezone.utc)
    results = {}
    for city_name, temps in city_temps.items():
        if temps:
            results[city_name] = AIModelForecast(
                city_id=city_name,
                model_name=model,
                target_date=target_date,
                temp_max_c=max(temps),
                temp_min_c=min(temps),
                fetched_at=now,
                hourly_temps_c=temps,
            )

    logger.info(
        "GribStream %s: fetched %d/%d cities for %s",
        model, len(results), len(cities), target_date,
    )
    return results


def compute_ai_physics_divergence(
    ai_forecast: AIModelForecast,
    physics_mean_c: float,
) -> dict:
    """Compare AI model forecast to physics consensus.

    Returns divergence info for dashboard and confidence adjustment.
    """
    if ai_forecast.temp_max_c is None:
        return {"divergence_c": 0, "signal": "no_data"}

    divergence = ai_forecast.temp_max_c - physics_mean_c

    if abs(divergence) < 1.5:
        signal = "agree"  # AI and physics within 1.5°C, high confidence
    elif abs(divergence) < 3.0:
        signal = "mild_diverge"  # Notable but not alarming
    else:
        signal = "strong_diverge"  # AI and physics disagree significantly, reduce confidence

    return {
        "ai_model": ai_forecast.model_name,
        "ai_max_c": round(ai_forecast.temp_max_c, 1),
        "physics_mean_c": round(physics_mean_c, 1),
        "divergence_c": round(divergence, 1),
        "signal": signal,
        "confidence_multiplier": 1.1 if signal == "agree" else 0.9 if signal == "mild_diverge" else 0.7,
    }
