"""METAR station observations from Iowa Environmental Mesonet (IEM).

IEM archives raw METAR/ASOS observations from airports worldwide,
the same data that Weather Underground displays. Polymarket weather
markets resolve against Wunderground, so this is the authoritative
ground truth for trade resolution and bias correction.

Open-Meteo archive returns gridded reanalysis (ERA5) interpolated to
a 9km grid cell. Even at airport coordinates, it can differ from the
actual runway sensor by 0.5-1.5°C, enough to flip a whole-degree
resolution bucket ~67% of the time.
"""
from __future__ import annotations

import csv
import io
import logging
from collections import defaultdict
from datetime import date

import httpx

logger = logging.getLogger(__name__)

IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


async def fetch_station_tmax(
    icao: str,
    target_date: date,
    *,
    timeout: float = 20.0,
) -> float | None:
    """Fetch observed daily high temperature from METAR station data.

    Returns temperature in Celsius, or None if unavailable.
    The ICAO code should match the Polymarket resolution station
    (e.g., KLGA for NYC, ZGSZ for Shenzhen, RJTT for Tokyo).

    IEM uses the raw ICAO code for international stations and
    strips the K prefix for US stations internally, but accepts
    full ICAO codes for all.
    """
    # IEM accepts ICAO codes directly (KLGA, EGLC, RJTT, etc.)
    # For US stations, strip the K prefix
    station_id = icao[1:] if icao.startswith("K") else icao

    params = {
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
    }

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(IEM_ASOS_URL, params=params, timeout=timeout)
            resp.raise_for_status()
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            logger.warning("IEM METAR fetch failed for %s on %s: %s", icao, target_date, e)
            return None

    # Parse CSV: columns are station, valid, tmpf
    content = resp.text
    reader = csv.DictReader(io.StringIO(content))

    temps_f: list[float] = []
    for row in reader:
        tmpf = row.get("tmpf", "M")
        if tmpf and tmpf != "M":
            try:
                temps_f.append(float(tmpf))
            except ValueError:
                continue

    if not temps_f:
        logger.debug("No METAR observations for %s on %s", icao, target_date)
        return None

    max_f = max(temps_f)
    max_c = (max_f - 32.0) * 5.0 / 9.0

    logger.info(
        "METAR obs for %s on %s: %.1f°F (%.1f°C) from %d readings",
        icao, target_date, max_f, max_c, len(temps_f),
    )
    return max_c


async def fetch_station_tmax_range(
    icao: str,
    start_date: date,
    end_date: date,
    *,
    timeout: float = 30.0,
) -> dict[str, float]:
    """Fetch daily high temperatures for a date range.

    Returns dict mapping ISO date string -> tmax in Celsius.
    More efficient than calling fetch_station_tmax per day.
    """
    station_id = icao[1:] if icao.startswith("K") else icao

    params = {
        "station": station_id,
        "data": "tmpf",
        "year1": start_date.year,
        "month1": start_date.month,
        "day1": start_date.day,
        "year2": end_date.year,
        "month2": end_date.month,
        "day2": end_date.day,
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "elev": "no",
        "missing": "M",
        "trace": "T",
        "direct": "no",
        "report_type": "3",
    }

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(IEM_ASOS_URL, params=params, timeout=timeout)
            resp.raise_for_status()
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            logger.warning("IEM METAR range fetch failed for %s: %s", icao, e)
            return {}

    reader = csv.DictReader(io.StringIO(resp.text))
    daily_temps: dict[str, list[float]] = defaultdict(list)

    for row in reader:
        ts = row.get("valid", "")
        tmpf = row.get("tmpf", "M")
        if ts and tmpf and tmpf != "M":
            d = ts[:10]
            try:
                daily_temps[d].append(float(tmpf))
            except ValueError:
                continue

    result: dict[str, float] = {}
    for d, temps in daily_temps.items():
        max_f = max(temps)
        result[d] = (max_f - 32.0) * 5.0 / 9.0

    logger.info(
        "METAR range for %s: %d days with data (%s to %s)",
        icao, len(result), start_date, end_date,
    )
    return result
