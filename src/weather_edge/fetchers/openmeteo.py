"""Open-Meteo multi-model weather forecast fetcher."""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timezone

import httpx

from weather_edge.config import CITIES, Settings, get_models_for_city
from weather_edge.models.enums import City, WeatherModel

logger = logging.getLogger(__name__)

# Open-Meteo model name mapping
OPENMETEO_MODEL_IDS: dict[WeatherModel, str] = {
    WeatherModel.ECMWF: "ecmwf_ifs025",
    WeatherModel.GFS: "gfs_seamless",
    WeatherModel.ICON: "icon_seamless",
    WeatherModel.GEM: "gem_seamless",
    WeatherModel.JMA: "jma_seamless",
    WeatherModel.METEOFRANCE: "meteofrance_seamless",
    WeatherModel.HRRR: "ncep_hrrr_conus",
    WeatherModel.NAM: "ncep_nam_conus",
    WeatherModel.UKV: "ukmo_seamless",
    WeatherModel.HRDPS: "gem_hrdps_continental",
    WeatherModel.KMA: "kma_seamless",
}


def c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def f_to_c(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


class ForecastResult:
    """Parsed forecast for one model + one city + one target date."""

    def __init__(
        self,
        city_id: str,
        model_name: str,
        target_date: date,
        fetched_at: datetime,
        temperature_2m_hourly: list[float | None],
        precipitation_hourly: list[float | None],
        snowfall_hourly: list[float | None],
        wind_speed_10m_hourly: list[float | None],
        temp_max_c: float | None,
        temp_min_c: float | None,
        precip_sum_mm: float | None,
        snow_sum_cm: float | None,
        wind_max_kmh: float | None,
        raw_response: dict | None = None,
    ):
        self.city_id = city_id
        self.model_name = model_name
        self.target_date = target_date
        self.fetched_at = fetched_at
        self.temperature_2m_hourly = temperature_2m_hourly
        self.precipitation_hourly = precipitation_hourly
        self.snowfall_hourly = snowfall_hourly
        self.wind_speed_10m_hourly = wind_speed_10m_hourly
        self.temp_max_c = temp_max_c
        self.temp_min_c = temp_min_c
        self.precip_sum_mm = precip_sum_mm
        self.snow_sum_cm = snow_sum_cm
        self.wind_max_kmh = wind_max_kmh
        self.raw_response = raw_response

    @property
    def temp_max_f(self) -> float | None:
        return c_to_f(self.temp_max_c) if self.temp_max_c is not None else None

    @property
    def temp_min_f(self) -> float | None:
        return c_to_f(self.temp_min_c) if self.temp_min_c is not None else None


async def fetch_model_forecast(
    client: httpx.AsyncClient,
    city_id: City,
    model: WeatherModel,
    target_date: date,
    base_url: str,
) -> ForecastResult | None:
    """Fetch forecast from a single model for a single city."""
    city = CITIES[city_id]
    model_id = OPENMETEO_MODEL_IDS[model]

    # Request 3 days centered on target to ensure we get the full day
    params = {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "hourly": "temperature_2m,precipitation,snowfall,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,snowfall_sum,wind_speed_10m_max",
        "timezone": "UTC",
        "models": model_id,
        "start_date": str(target_date),
        "end_date": str(target_date),
    }

    try:
        resp = await client.get(base_url, params=params, timeout=15.0)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        logger.warning("HTTP %s fetching %s for %s: %s", e.response.status_code, model_id, city_id, e)
        return None
    except httpx.RequestError as e:
        logger.warning("Request error fetching %s for %s: %s", model_id, city_id, e)
        return None

    now = datetime.now(timezone.utc)

    # Parse hourly data, Open-Meteo returns arrays keyed by variable name
    hourly = data.get("hourly", {})
    daily = data.get("daily", {})

    temp_hourly = hourly.get("temperature_2m", [])
    precip_hourly = hourly.get("precipitation", [])
    snow_hourly = hourly.get("snowfall", [])
    wind_hourly = hourly.get("wind_speed_10m", [])

    # Daily aggregates
    temp_max_list = daily.get("temperature_2m_max", [])
    temp_min_list = daily.get("temperature_2m_min", [])
    precip_sum_list = daily.get("precipitation_sum", [])
    snow_sum_list = daily.get("snowfall_sum", [])
    wind_max_list = daily.get("wind_speed_10m_max", [])

    temp_max_c = temp_max_list[0] if temp_max_list else None
    temp_min_c = temp_min_list[0] if temp_min_list else None
    precip_sum_mm = precip_sum_list[0] if precip_sum_list else None
    snow_sum_cm = snow_sum_list[0] if snow_sum_list else None
    wind_max_kmh = wind_max_list[0] if wind_max_list else None

    return ForecastResult(
        city_id=city_id.value,
        model_name=model.value,
        target_date=target_date,
        fetched_at=now,
        temperature_2m_hourly=temp_hourly,
        precipitation_hourly=precip_hourly,
        snowfall_hourly=snow_hourly,
        wind_speed_10m_hourly=wind_hourly,
        temp_max_c=temp_max_c,
        temp_min_c=temp_min_c,
        precip_sum_mm=precip_sum_mm,
        snow_sum_cm=snow_sum_cm,
        wind_max_kmh=wind_max_kmh,
        raw_response=data,
    )


async def fetch_city_forecasts(
    city_id: City,
    target_date: date,
    settings: Settings | None = None,
) -> list[ForecastResult]:
    """Fetch forecasts from all applicable models for a city, concurrently."""
    if settings is None:
        from weather_edge.config import settings as _settings
        settings = _settings

    models = get_models_for_city(city_id)
    results: list[ForecastResult] = []

    if settings.openmeteo_paid_tier:
        # Paid tier (600 req/min): fetch all models in parallel, no delay
        async with httpx.AsyncClient() as client:
            tasks = [
                fetch_model_forecast(client, city_id, model, target_date, settings.openmeteo_base_url)
                for model in models
            ]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    else:
        # Free tier: fetch models sequentially within a shared client
        # (concurrent requests from gather are fine, they share one connection)
        async with httpx.AsyncClient() as client:
            tasks = [
                fetch_model_forecast(client, city_id, model, target_date, settings.openmeteo_base_url)
                for model in models
            ]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    for r in raw_results:
        if isinstance(r, ForecastResult):
            results.append(r)
        elif isinstance(r, Exception):
            logger.warning("Forecast fetch failed: %s", r)

    logger.info(
        "Fetched %d/%d model forecasts for %s on %s",
        len(results), len(models), city_id.value, target_date,
    )
    return results


async def fetch_all_cities(
    target_date: date,
    settings: Settings | None = None,
    city_order: list[City] | None = None,
) -> dict[City, list[ForecastResult]]:
    """Fetch forecasts for all cities.

    Args:
        target_date: Date to fetch forecasts for.
        settings: Config settings (uses global if None).
        city_order: Optional priority ordering of cities. If provided, cities
            are fetched in this order. Defaults to City enum order.

    On free tier: 0.5s delay between cities to respect rate limits.
    On paid tier ($30/month, 600 req/min): no delay, cuts full cycle from
    ~14 minutes to ~15 seconds.
    """
    if settings is None:
        from weather_edge.config import settings as _settings
        settings = _settings

    all_forecasts: dict[City, list[ForecastResult]] = {}
    inter_city_delay = 0.0 if settings.openmeteo_paid_tier else 0.5

    # Use provided city order, or default to all cities
    cities = city_order if city_order is not None else list(City)

    for city_id in cities:
        forecasts = await fetch_city_forecasts(city_id, target_date, settings)
        all_forecasts[city_id] = forecasts
        if inter_city_delay > 0:
            await asyncio.sleep(inter_city_delay)

    return all_forecasts
