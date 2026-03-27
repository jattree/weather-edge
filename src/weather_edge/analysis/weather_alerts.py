"""Real-time weather alerts from NWS (US cities) and synthetic alerts from Open-Meteo (international)."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from weather_edge.config import CITIES, CityConfig
from weather_edge.models.enums import City

logger = logging.getLogger(__name__)

# US-territory cities that the NWS API covers (continental US + territories)
_US_CITIES = {
    City.NYC, City.DAL, City.SEA, City.ATL, City.CHI, City.MIA,
    City.LAX, City.HOU, City.DEN, City.SFO,
}

# NWS requires a descriptive User-Agent
_NWS_USER_AGENT = "WeatherEdgeDashboard/1.0 (weather-edge-trading; contact@weatheredge.dev)"

# Rate-limit: max concurrent NWS requests
_NWS_SEMAPHORE = asyncio.Semaphore(3)
_INTL_SEMAPHORE = asyncio.Semaphore(5)

# Open-Meteo thresholds for synthetic alerts
_PRECIP_HEAVY_MM = 10.0
_PRECIP_MODERATE_MM = 5.0
_WIND_STRONG_KMH = 60.0
_WIND_MODERATE_KMH = 40.0
_TEMP_EXTREME_HIGH_C = 38.0
_TEMP_EXTREME_LOW_C = -15.0
_SNOW_HEAVY_CM = 10.0
_SNOW_MODERATE_CM = 3.0


@dataclass
class WeatherAlert:
    """A weather alert for a monitored city."""
    city_id: str
    city_name: str
    severity: str  # "warning", "watch", "advisory", "info"
    headline: str
    description: str
    expires: str | None  # ISO timestamp or None
    source: str  # "nws" or "open-meteo"

    def to_dict(self) -> dict:
        return {
            "city_id": self.city_id,
            "city_name": self.city_name,
            "severity": self.severity,
            "headline": self.headline,
            "description": self.description,
            "expires": self.expires,
            "source": self.source,
        }


async def _fetch_nws_alerts(city_id: City, config: CityConfig) -> list[WeatherAlert]:
    """Fetch active NWS alerts for a US city by lat/lon point."""
    alerts: list[WeatherAlert] = []
    url = f"https://api.weather.gov/alerts/active?point={config.latitude},{config.longitude}"

    async with _NWS_SEMAPHORE:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    url,
                    headers={
                        "User-Agent": _NWS_USER_AGENT,
                        "Accept": "application/geo+json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            for feature in data.get("features", []):
                props = feature.get("properties", {})
                severity_raw = (props.get("severity") or "").lower()
                # Map NWS severity to our levels
                if severity_raw in ("extreme", "severe"):
                    severity = "warning"
                elif severity_raw == "moderate":
                    severity = "watch"
                elif severity_raw == "minor":
                    severity = "advisory"
                else:
                    severity = "info"

                headline = props.get("headline") or props.get("event") or "Weather Alert"
                description = props.get("description") or ""
                # Truncate long descriptions
                if len(description) > 300:
                    description = description[:297] + "..."
                expires = props.get("expires")

                alerts.append(WeatherAlert(
                    city_id=city_id.value,
                    city_name=config.name,
                    severity=severity,
                    headline=headline,
                    description=description,
                    expires=expires,
                    source="nws",
                ))

            try:
                from weather_edge.analysis.service_health import record_service_call
                record_service_call("nws", True)
            except Exception:
                pass

        except httpx.HTTPStatusError as e:
            logger.warning("NWS API error for %s: HTTP %d", config.name, e.response.status_code)
            try:
                from weather_edge.analysis.service_health import record_service_call
                record_service_call("nws", False)
            except Exception:
                pass
        except Exception:
            logger.warning("NWS alert fetch failed for %s", config.name, exc_info=True)
            try:
                from weather_edge.analysis.service_health import record_service_call
                record_service_call("nws", False)
            except Exception:
                pass

    return alerts


async def _fetch_openmeteo_synthetic_alerts(city_id: City, config: CityConfig) -> list[WeatherAlert]:
    """Generate synthetic alerts from Open-Meteo forecast data for international cities."""
    from weather_edge.config import settings
    alerts: list[WeatherAlert] = []
    url = settings.effective_openmeteo_url
    params = {
        "latitude": config.latitude,
        "longitude": config.longitude,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,snowfall_sum,wind_speed_10m_max",
        "forecast_days": 2,
        "timezone": "auto",
    }
    if settings.openmeteo_api_key:
        params["apikey"] = settings.openmeteo_api_key

    async with _INTL_SEMAPHORE:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

            daily = data.get("daily", {})
            dates = daily.get("time", [])
            temp_maxes = daily.get("temperature_2m_max", [])
            temp_mins = daily.get("temperature_2m_min", [])
            precip_sums = daily.get("precipitation_sum", [])
            snow_sums = daily.get("snowfall_sum", [])
            wind_maxes = daily.get("wind_speed_10m_max", [])

            for i, d in enumerate(dates):
                t_max = temp_maxes[i] if i < len(temp_maxes) else None
                t_min = temp_mins[i] if i < len(temp_mins) else None
                precip = precip_sums[i] if i < len(precip_sums) else None
                snow = snow_sums[i] if i < len(snow_sums) else None
                wind = wind_maxes[i] if i < len(wind_maxes) else None

                # Heavy precipitation
                if precip is not None and precip >= _PRECIP_HEAVY_MM:
                    alerts.append(WeatherAlert(
                        city_id=city_id.value,
                        city_name=config.name,
                        severity="warning",
                        headline=f"Heavy rain expected: {precip:.0f}mm on {d}",
                        description=f"Forecast precipitation of {precip:.1f}mm for {config.name} on {d}. "
                                    "This may impact temperature markets and outdoor conditions.",
                        expires=f"{d}T23:59:59Z",
                        source="open-meteo",
                    ))
                elif precip is not None and precip >= _PRECIP_MODERATE_MM:
                    alerts.append(WeatherAlert(
                        city_id=city_id.value,
                        city_name=config.name,
                        severity="advisory",
                        headline=f"Moderate rain expected: {precip:.0f}mm on {d}",
                        description=f"Forecast precipitation of {precip:.1f}mm for {config.name} on {d}.",
                        expires=f"{d}T23:59:59Z",
                        source="open-meteo",
                    ))

                # Heavy snowfall
                if snow is not None and snow >= _SNOW_HEAVY_CM:
                    alerts.append(WeatherAlert(
                        city_id=city_id.value,
                        city_name=config.name,
                        severity="warning",
                        headline=f"Heavy snow expected: {snow:.0f}cm on {d}",
                        description=f"Forecast snowfall of {snow:.1f}cm for {config.name} on {d}. "
                                    "Significant impact on travel and temperatures expected.",
                        expires=f"{d}T23:59:59Z",
                        source="open-meteo",
                    ))
                elif snow is not None and snow >= _SNOW_MODERATE_CM:
                    alerts.append(WeatherAlert(
                        city_id=city_id.value,
                        city_name=config.name,
                        severity="advisory",
                        headline=f"Snow expected: {snow:.0f}cm on {d}",
                        description=f"Forecast snowfall of {snow:.1f}cm for {config.name} on {d}.",
                        expires=f"{d}T23:59:59Z",
                        source="open-meteo",
                    ))

                # Strong wind
                if wind is not None and wind >= _WIND_STRONG_KMH:
                    alerts.append(WeatherAlert(
                        city_id=city_id.value,
                        city_name=config.name,
                        severity="watch",
                        headline=f"Strong winds expected: {wind:.0f} km/h on {d}",
                        description=f"Forecast wind gusts up to {wind:.0f} km/h for {config.name} on {d}.",
                        expires=f"{d}T23:59:59Z",
                        source="open-meteo",
                    ))
                elif wind is not None and wind >= _WIND_MODERATE_KMH:
                    alerts.append(WeatherAlert(
                        city_id=city_id.value,
                        city_name=config.name,
                        severity="advisory",
                        headline=f"Elevated winds: {wind:.0f} km/h on {d}",
                        description=f"Forecast wind speeds of {wind:.0f} km/h for {config.name} on {d}.",
                        expires=f"{d}T23:59:59Z",
                        source="open-meteo",
                    ))

                # Extreme temperatures
                if t_max is not None and t_max >= _TEMP_EXTREME_HIGH_C:
                    alerts.append(WeatherAlert(
                        city_id=city_id.value,
                        city_name=config.name,
                        severity="warning",
                        headline=f"Extreme heat: {t_max:.0f}C on {d}",
                        description=f"Forecast high of {t_max:.1f}C for {config.name} on {d}. "
                                    "Temperature markets may see significant movement.",
                        expires=f"{d}T23:59:59Z",
                        source="open-meteo",
                    ))

                if t_min is not None and t_min <= _TEMP_EXTREME_LOW_C:
                    alerts.append(WeatherAlert(
                        city_id=city_id.value,
                        city_name=config.name,
                        severity="warning",
                        headline=f"Extreme cold: {t_min:.0f}C on {d}",
                        description=f"Forecast low of {t_min:.1f}C for {config.name} on {d}. "
                                    "Temperature markets may see significant movement.",
                        expires=f"{d}T23:59:59Z",
                        source="open-meteo",
                    ))

        except Exception:
            logger.warning("Open-Meteo synthetic alert fetch failed for %s", config.name, exc_info=True)

    return alerts


async def fetch_city_alerts(city_id: City) -> list[WeatherAlert]:
    """Fetch alerts for a single city (NWS for US, synthetic for international)."""
    config = CITIES[city_id]
    if city_id in _US_CITIES:
        return await _fetch_nws_alerts(city_id, config)
    else:
        return await _fetch_openmeteo_synthetic_alerts(city_id, config)


def generate_synthetic_alerts_from_cache(city_id: City, config: CityConfig, forecasts: list) -> list[WeatherAlert]:
    """Generate synthetic alerts from already-fetched forecast data (no API call)."""
    alerts: list[WeatherAlert] = []
    for f in forecasts:
        d = str(f.forecast_date) if hasattr(f, 'forecast_date') else "upcoming"

        if f.precip_sum_mm is not None and f.precip_sum_mm >= _PRECIP_HEAVY_MM:
            alerts.append(WeatherAlert(
                city_id=city_id.value, city_name=config.name, severity="warning",
                headline=f"Heavy rain expected: {f.precip_sum_mm:.0f}mm on {d}",
                description=f"Forecast precipitation of {f.precip_sum_mm:.1f}mm for {config.name}.",
                expires=None, source="forecast-cache",
            ))
        elif f.precip_sum_mm is not None and f.precip_sum_mm >= _PRECIP_MODERATE_MM:
            alerts.append(WeatherAlert(
                city_id=city_id.value, city_name=config.name, severity="advisory",
                headline=f"Moderate rain expected: {f.precip_sum_mm:.0f}mm on {d}",
                description=f"Forecast precipitation of {f.precip_sum_mm:.1f}mm for {config.name}.",
                expires=None, source="forecast-cache",
            ))

        if f.snow_sum_cm is not None and f.snow_sum_cm >= _SNOW_HEAVY_CM:
            alerts.append(WeatherAlert(
                city_id=city_id.value, city_name=config.name, severity="warning",
                headline=f"Heavy snow expected: {f.snow_sum_cm:.0f}cm on {d}",
                description=f"Forecast snowfall of {f.snow_sum_cm:.1f}cm for {config.name}.",
                expires=None, source="forecast-cache",
            ))
        elif f.snow_sum_cm is not None and f.snow_sum_cm >= _SNOW_MODERATE_CM:
            alerts.append(WeatherAlert(
                city_id=city_id.value, city_name=config.name, severity="advisory",
                headline=f"Snow expected: {f.snow_sum_cm:.0f}cm on {d}",
                description=f"Forecast snowfall of {f.snow_sum_cm:.1f}cm for {config.name}.",
                expires=None, source="forecast-cache",
            ))

        if f.temp_max_c is not None and f.temp_max_c >= _TEMP_EXTREME_HIGH_C:
            alerts.append(WeatherAlert(
                city_id=city_id.value, city_name=config.name, severity="warning",
                headline=f"Extreme heat: {f.temp_max_c:.0f}C on {d}",
                description=f"Forecast high of {f.temp_max_c:.1f}C for {config.name}.",
                expires=None, source="forecast-cache",
            ))

        if f.temp_min_c is not None and f.temp_min_c <= _TEMP_EXTREME_LOW_C:
            alerts.append(WeatherAlert(
                city_id=city_id.value, city_name=config.name, severity="warning",
                headline=f"Extreme cold: {f.temp_min_c:.0f}C on {d}",
                description=f"Forecast low of {f.temp_min_c:.1f}C for {config.name}.",
                expires=None, source="forecast-cache",
            ))

    # Deduplicate by headline (multiple models may produce same alert)
    seen = set()
    unique = []
    for a in alerts:
        if a.headline not in seen:
            seen.add(a.headline)
            unique.append(a)
    return unique


async def fetch_all_alerts(forecast_cache: dict | None = None) -> list[dict]:
    """Fetch weather alerts for all 21 monitored cities.

    US cities: NWS API (concurrent, rate-limited).
    International cities: generated from forecast_cache (zero API calls).
    Falls back to Open-Meteo API only if no cache provided.

    Returns a list of alert dicts sorted by severity (warning > watch > advisory > info).
    """
    all_alerts: list[WeatherAlert] = []

    # NWS alerts for US cities (these are a different API, no rate-limit issue)
    nws_tasks = []
    for city_id in City:
        if city_id in _US_CITIES:
            nws_tasks.append(fetch_city_alerts(city_id))
    if nws_tasks:
        results = await asyncio.gather(*nws_tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.warning("NWS alert fetch failed: %s", result)
                continue
            all_alerts.extend(result)

    # International cities: use forecast cache if available (no API calls)
    for city_id in City:
        if city_id in _US_CITIES:
            continue
        config = CITIES[city_id]
        if forecast_cache:
            # Find any forecasts for this city in the cache
            city_forecasts = []
            for (cid, td), f_list in forecast_cache.items():
                if cid == city_id:
                    city_forecasts.extend(f_list)
                    break
            if city_forecasts:
                all_alerts.extend(generate_synthetic_alerts_from_cache(city_id, config, city_forecasts))
                continue
        # Fallback: fetch from Open-Meteo (only if no cache)
        try:
            intl_alerts = await _fetch_openmeteo_synthetic_alerts(city_id, config)
            all_alerts.extend(intl_alerts)
        except Exception:
            logger.debug("Synthetic alert fallback failed for %s", config.name)

    # Sort by severity priority
    severity_order = {"warning": 0, "watch": 1, "advisory": 2, "info": 3}
    all_alerts.sort(key=lambda a: severity_order.get(a.severity, 99))

    return [a.to_dict() for a in all_alerts]
