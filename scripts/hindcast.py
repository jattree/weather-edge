"""Hindcast script, bootstrap Brier weights from historical data.

Pulls 90 days of historical model forecasts and actual observations,
populates forecast_snapshots table. This gives the adaptive weighting
engine thousands of data points instantly instead of waiting months
for live trading to accumulate them.

Usage:
    .venv/bin/python scripts/hindcast.py [--days 90] [--city nyc]

Observation sources (in priority order):
    1. IEM ASOS (METAR station data), same source as Wunderground,
       which Polymarket resolves against. This is the correct ground
       truth for bias correction.
    2. Open-Meteo archive API (fallback), gridded reanalysis data.
       ~0.9°C MAE vs station readings with 67% rounding mismatch.
       Only used when METAR data is unavailable.

Open-Meteo historical forecast API:
    https://historical-forecast-api.open-meteo.com/v1/forecast
    - Free tier, no API key needed
    - Archives model runs for the past ~3 months
    - Returns what each model predicted on a given date
"""
from __future__ import annotations

import argparse
import csv
import io
import logging
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import httpx

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from weather_edge.config import CITIES, GLOBAL_MODELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Historical forecast API (what did models predict?)
# Paid tier gets higher rate limits, free tier is 5K req/day
HIST_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
# Archive API (what actually happened?), always free, but gridded reanalysis
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
# IEM ASOS, actual METAR station observations (same as Wunderground)
IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
# Paid API key (optional, speeds up requests, no rate-limit risk)
OPENMETEO_API_KEY = "W8ZEnxSiyS61KSh8"

# Models to hindcast (use the Open-Meteo model IDs)
MODELS_TO_HINDCAST = [m.value for m in GLOBAL_MODELS]

DB_PATH = Path(__file__).parent.parent / "weather_edge.db"

# Historical forecast API only goes back ~2.5 years reliably
# Beyond that, model versions change (ECMWF IFS upgrade 2023)
MAX_HINDCAST_DAYS = 912  # ~2.5 years


def fetch_historical_forecast(
    lat: float, lon: float, target_date: date, model_id: str,
) -> float | None:
    """Fetch what a model predicted for a specific date.

    Uses the historical forecast API which archives past model runs.
    We ask for the 1-day-ahead forecast (what did the model say
    the day before about tomorrow's high?).
    """
    try:
        resp = httpx.get(
            HIST_FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max",
                "start_date": str(target_date),
                "end_date": str(target_date),
                "past_days": 0,
                "forecast_days": 1,
                "models": model_id,
            },
            timeout=15,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        temps = data.get("daily", {}).get("temperature_2m_max", [])
        if temps and temps[0] is not None:
            return float(temps[0])
    except Exception:
        pass
    return None


def fetch_metar_observation(
    icao: str, target_date: date,
) -> float | None:
    """Fetch actual observed daily high from METAR station data.

    Uses IEM ASOS archive, the same raw METAR/ASOS data that
    Weather Underground displays. Polymarket resolves against
    Wunderground, so this is the correct ground truth.

    Returns daily max temperature in Celsius, or None if unavailable.
    """
    # US stations: strip K prefix (KDAL -> DAL). International: use as-is.
    station_id = icao[1:] if icao.startswith("K") else icao

    try:
        resp = httpx.get(
            IEM_ASOS_URL,
            params={
                "station": station_id,
                "data": "tmpf",
                "year1": target_date.year,
                "month1": target_date.month,
                "day1": target_date.day,
                "year2": target_date.year,
                "month2": target_date.month,
                "day2": target_date.day,
                "tz": "Etc/UTC",
                "format": "onlycomma",
                "latlon": "no",
                "elev": "no",
                "missing": "M",
                "trace": "T",
                "direct": "no",
                "report_type": "3",
            },
            timeout=20,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            return None

        reader = csv.DictReader(io.StringIO(resp.text))
        temps_f: list[float] = []
        for row in reader:
            tmpf = row.get("tmpf", "M")
            if tmpf and tmpf != "M":
                try:
                    temps_f.append(float(tmpf))
                except ValueError:
                    continue

        if not temps_f:
            return None

        max_f = max(temps_f)
        return (max_f - 32.0) * 5.0 / 9.0
    except Exception:
        return None


def fetch_batch_metar_observations(
    icao: str, start: date, end: date,
) -> dict[str, float]:
    """Fetch daily max temperatures from METAR station data for a date range.

    Returns dict mapping ISO date string -> tmax in Celsius.
    Single API call for the whole range, much more efficient than per-day.
    """
    station_id = icao[1:] if icao.startswith("K") else icao

    try:
        resp = httpx.get(
            IEM_ASOS_URL,
            params={
                "station": station_id,
                "data": "tmpf",
                "year1": start.year,
                "month1": start.month,
                "day1": start.day,
                "year2": end.year,
                "month2": end.month,
                "day2": end.day,
                "tz": "Etc/UTC",
                "format": "onlycomma",
                "latlon": "no",
                "elev": "no",
                "missing": "M",
                "trace": "T",
                "direct": "no",
                "report_type": "3",
            },
            timeout=30,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            return {}

        reader = csv.DictReader(io.StringIO(resp.text))
        daily_temps: dict[str, list[float]] = defaultdict(list)

        for row in reader:
            ts = row.get("valid", "")
            tmpf = row.get("tmpf", "M")
            if ts and tmpf and tmpf != "M":
                d = ts[:10]  # Extract YYYY-MM-DD from timestamp
                try:
                    daily_temps[d].append(float(tmpf))
                except ValueError:
                    continue

        result: dict[str, float] = {}
        for d, temps in daily_temps.items():
            max_f = max(temps)
            result[d] = (max_f - 32.0) * 5.0 / 9.0

        return result
    except Exception:
        return {}


def fetch_actual_observation(
    lat: float, lon: float, target_date: date,
    icao: str | None = None,
) -> float | None:
    """Fetch actual observed high temperature for a date.

    Tries METAR station data first (correct ground truth for Polymarket),
    falls back to Open-Meteo archive (gridded reanalysis) if unavailable.
    """
    # Primary: METAR station observation
    if icao:
        result = fetch_metar_observation(icao, target_date)
        if result is not None:
            return result
        logger.debug("METAR unavailable for %s on %s, falling back to Open-Meteo", icao, target_date)

    # Fallback: Open-Meteo archive (gridded reanalysis)
    try:
        resp = httpx.get(
            ARCHIVE_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max",
                "start_date": str(target_date),
                "end_date": str(target_date),
            },
            timeout=15,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        temps = data.get("daily", {}).get("temperature_2m_max", [])
        if temps and temps[0] is not None:
            return float(temps[0])
    except Exception:
        pass
    return None


def fetch_batch_observations(
    lat: float, lon: float, start: date, end: date,
    icao: str | None = None,
) -> dict[str, float]:
    """Fetch actual observations for a date range.

    Tries METAR station data first (correct ground truth for Polymarket),
    fills gaps from Open-Meteo archive (gridded reanalysis) if needed.
    """
    result: dict[str, float] = {}

    # Primary: METAR station observations
    if icao:
        result = fetch_batch_metar_observations(icao, start, end)
        if result:
            logger.info("  METAR (%s): %d days of station data", icao, len(result))

    # Fallback: fill any missing days from Open-Meteo archive
    # Generate expected date range to check for gaps
    expected_days = (end - start).days + 1
    if len(result) < expected_days:
        try:
            resp = httpx.get(
                ARCHIVE_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": "temperature_2m_max",
                    "start_date": str(start),
                    "end_date": str(end),
                },
                timeout=30,
                follow_redirects=True,
            )
            if resp.status_code == 200:
                data = resp.json()
                dates = data.get("daily", {}).get("time", [])
                temps = data.get("daily", {}).get("temperature_2m_max", [])
                om_filled = 0
                for d, t in zip(dates, temps):
                    if t is not None and d not in result:
                        result[d] = float(t)
                        om_filled += 1
                if om_filled:
                    logger.info("  Open-Meteo fallback: filled %d missing days", om_filled)
        except Exception:
            pass

    return result


def fetch_batch_forecasts(
    lat: float, lon: float, start: date, end: date, model_id: str,
) -> dict[str, float]:
    """Fetch historical model forecasts for a date range."""
    try:
        # No API key for historical, paid tier doesn't cover it
        # Free tier: 5K req/day (we need ~216 total)
        resp = httpx.get(
            HIST_FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max",
                "start_date": str(start),
                "end_date": str(end),
                "models": model_id,
            },
            timeout=30,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            return {}
        data = resp.json()
        # Key format: temperature_2m_max or temperature_2m_max_{model_id}
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        # Try model-specific key first, then generic
        key = f"temperature_2m_max_{model_id}"
        temps = daily.get(key) or daily.get("temperature_2m_max", [])
        result = {}
        for d, t in zip(dates, temps):
            if t is not None:
                result[d] = float(t)
        return result
    except Exception:
        return {}


def run_hindcast(
    days: int = 90,
    city_filter: str | None = None,
    db_path: Path = DB_PATH,
):
    """Run hindcast for all cities, populating forecast_snapshots."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Ensure table exists
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS forecast_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER,
            city_id TEXT NOT NULL,
            target_date TEXT NOT NULL,
            model_name TEXT NOT NULL,
            forecast_value REAL,
            actual_value REAL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_forecast_city_model
            ON forecast_snapshots(city_id, model_name);
    """)

    end_date = date.today() - timedelta(days=1)  # Yesterday (most recent available)
    start_date = end_date - timedelta(days=days)

    cities = CITIES
    if city_filter:
        cities = {k: v for k, v in CITIES.items() if k.value == city_filter}
        if not cities:
            logger.error("City %s not found", city_filter)
            return

    total_inserted = 0
    now_str = date.today().isoformat() + "T00:00:00+00:00"

    for city_id, config in cities.items():
        city_str = city_id.value
        logger.info("=== %s (%s), %d days ===", config.name, city_str, days)

        # Check what we already have
        existing = conn.execute(
            "SELECT COUNT(DISTINCT target_date) FROM forecast_snapshots WHERE city_id = ?",
            (city_str,),
        ).fetchone()[0]
        if existing >= days * 0.8:
            logger.info("  Already have %d dates, skipping", existing)
            continue

        # Fetch actual observations in one batch (METAR primary, Open-Meteo fallback)
        actuals = fetch_batch_observations(
            config.latitude, config.longitude, start_date, end_date,
            icao=config.icao,
        )
        logger.info("  Fetched %d actual observations", len(actuals))
        time.sleep(0.5)  # Rate limit

        # Fetch forecasts per model
        all_models = GLOBAL_MODELS + config.regional_models
        for model in all_models:
            model_id = model.value
            forecasts = fetch_batch_forecasts(
                config.latitude, config.longitude,
                start_date, end_date, model_id,
            )
            logger.info("  %s: %d forecasts", model_id, len(forecasts))

            # Insert into DB
            batch = []
            for date_str, forecast_val in forecasts.items():
                actual_val = actuals.get(date_str)
                batch.append((
                    None,  # trade_id (hindcast, no trade)
                    city_str,
                    date_str,
                    model_id,
                    forecast_val,
                    actual_val,
                    now_str,
                ))

            if batch:
                conn.executemany(
                    """INSERT INTO forecast_snapshots
                       (trade_id, city_id, target_date, model_name,
                        forecast_value, actual_value, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    batch,
                )
                conn.commit()
                total_inserted += len(batch)

            time.sleep(0.3)  # Rate limit between models

        time.sleep(1.0)  # Rate limit between cities

    logger.info("=== HINDCAST COMPLETE: %d forecast snapshots inserted ===", total_inserted)

    # Quick stats
    cur = conn.execute("""
        SELECT model_name, COUNT(*) as cnt,
            ROUND(AVG(ABS(forecast_value - actual_value)), 2) as mae
        FROM forecast_snapshots
        WHERE actual_value IS NOT NULL
        GROUP BY model_name ORDER BY mae
    """)
    logger.info("=== MODEL ACCURACY (MAE in °C) ===")
    for r in cur.fetchall():
        logger.info("  %s: %d forecasts, MAE=%.2f°C", r[0], r[1], r[2])

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hindcast: bootstrap Brier weights")
    parser.add_argument("--days", type=int, default=90, help="Days to hindcast")
    parser.add_argument("--city", type=str, default=None, help="Single city (e.g. nyc)")
    parser.add_argument("--db", type=str, default=None, help="DB path")
    args = parser.parse_args()

    db = Path(args.db) if args.db else DB_PATH
    run_hindcast(days=args.days, city_filter=args.city, db_path=db)
