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


# Minimum hourly readings required to trust a daily max.
# Prevents bad resolution from sparse/incomplete station data.
MIN_READINGS = 12


async def _fetch_hko_range(start_date: date, end_date: date, timeout: float = 30.0) -> dict[str, float]:
    """Fetch daily max temps from HK Observatory Open Data API."""
    url = "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMMAXT&rformat=json&station=HKO"
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, timeout=timeout)
            resp.raise_for_status()
        except Exception as e:
            logger.warning("HKO fetch failed: %s", e)
            return {}
    
    daily = {}
    data = resp.json().get("data", [])
    for row in data:
        try:
            y, m, d = int(row[0]), int(row[1]), int(row[2])
            obs_date = date(y, m, d)
            if start_date <= obs_date <= end_date:
                daily[str(obs_date)] = float(row[3])
        except (ValueError, IndexError):
            pass
    return daily


async def fetch_station_tmax(
    icao: str,
    target_date: date,
    *,
    timeout: float = 20.0,
) -> float | None:
    """Fetch observed daily high temperature from METAR station data.

    Returns temperature in Celsius, or None if unavailable.
    Fetches both tmpf and tmpc from IEM to avoid F→C conversion
    errors that can flip whole-degree rounding buckets.
    """
    if icao == "45005":
        daily = await _fetch_hko_range(target_date, target_date, timeout=timeout)
        val = daily.get(str(target_date))
        if val is not None:
            logger.info("HKO obs for 45005 on %s: %.1f°C", target_date, val)
        return val

    station_id = icao[1:] if icao.startswith("K") else icao

    params = {
        "station": station_id,
        "data": "tmpf,tmpc",
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

    content = resp.text
    reader = csv.DictReader(io.StringIO(content))

    temps_f: list[float] = []
    temps_c: list[float] = []
    for row in reader:
        tmpf = row.get("tmpf", "M")
        tmpc = row.get("tmpc", "M")
        if tmpf and tmpf != "M":
            try:
                temps_f.append(float(tmpf))
            except ValueError:
                pass
        if tmpc and tmpc != "M":
            try:
                temps_c.append(float(tmpc))
            except ValueError:
                pass

    readings = max(len(temps_f), len(temps_c))
    if readings < MIN_READINGS:
        logger.debug(
            "Insufficient METAR readings for %s on %s: %d < %d",
            icao, target_date, readings, MIN_READINGS,
        )
        return None

    # Use native Celsius when available (avoids F→C conversion rounding errors)
    if temps_c:
        max_c = max(temps_c)
    elif temps_f:
        max_c = (max(temps_f) - 32.0) * 5.0 / 9.0
    else:
        return None

    max_f = max(temps_f) if temps_f else max_c * 9.0 / 5.0 + 32.0

    logger.info(
        "METAR obs for %s on %s: %.1f°F / %.1f°C from %d readings",
        icao, target_date, max_f, max_c, readings,
    )
    return max_c


async def fetch_station_tmax_both(
    icao: str,
    target_date: date,
    *,
    timeout: float = 20.0,
) -> tuple[float | None, float | None]:
    """Fetch daily max in both native Celsius AND native Fahrenheit.

    Returns (max_c, max_f) from their respective native METAR fields.
    This avoids F↔C conversion rounding errors: for Fahrenheit markets,
    round(max_f) directly instead of round(c_to_f(max_c)).
    """
    if icao == "45005":
        daily = await _fetch_hko_range(target_date, target_date, timeout=timeout)
        val_c = daily.get(str(target_date))
        if val_c is not None:
            return val_c, val_c * 9.0 / 5.0 + 32.0
        return None, None

    station_id = icao[1:] if icao.startswith("K") else icao

    params = {
        "station": station_id,
        "data": "tmpf,tmpc",
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
            return None, None

    reader = csv.DictReader(io.StringIO(resp.text))
    temps_f: list[float] = []
    temps_c: list[float] = []
    for row in reader:
        tmpf = row.get("tmpf", "M")
        tmpc = row.get("tmpc", "M")
        if tmpf and tmpf != "M":
            try:
                temps_f.append(float(tmpf))
            except ValueError:
                pass
        if tmpc and tmpc != "M":
            try:
                temps_c.append(float(tmpc))
            except ValueError:
                pass

    readings = max(len(temps_f), len(temps_c))
    if readings < MIN_READINGS:
        return None, None

    max_c = max(temps_c) if temps_c else None
    max_f = max(temps_f) if temps_f else None
    return max_c, max_f


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
    if icao == "45005":
        return await _fetch_hko_range(start_date, end_date, timeout=timeout)

    station_id = icao[1:] if icao.startswith("K") else icao

    params = {
        "station": station_id,
        "data": "tmpf,tmpc",
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
    daily_temps_c: dict[str, list[float]] = defaultdict(list)
    daily_temps_f: dict[str, list[float]] = defaultdict(list)

    for row in reader:
        ts = row.get("valid", "")
        if not ts:
            continue
        d = ts[:10]
        tmpc = row.get("tmpc", "M")
        tmpf = row.get("tmpf", "M")
        if tmpc and tmpc != "M":
            try:
                daily_temps_c[d].append(float(tmpc))
            except ValueError:
                pass
        if tmpf and tmpf != "M":
            try:
                daily_temps_f[d].append(float(tmpf))
            except ValueError:
                pass

    result: dict[str, float] = {}
    all_dates = set(daily_temps_c.keys()) | set(daily_temps_f.keys())
    for d in all_dates:
        c_temps = daily_temps_c.get(d, [])
        f_temps = daily_temps_f.get(d, [])
        readings = max(len(c_temps), len(f_temps))
        if readings < MIN_READINGS:
            continue
        # Prefer native Celsius to avoid conversion rounding errors
        if c_temps:
            result[d] = max(c_temps)
        elif f_temps:
            result[d] = (max(f_temps) - 32.0) * 5.0 / 9.0

    logger.info(
        "METAR range for %s: %d days with data (%s to %s)",
        icao, len(result), start_date, end_date,
    )
    return result
