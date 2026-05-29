"""METAR station observations from Iowa Environmental Mesonet (IEM).

IEM archives raw METAR/ASOS observations from airports worldwide,
the same data that Weather Underground displays. Polymarket weather
markets resolve against Wunderground, so this is the authoritative
ground truth for trade resolution and bias correction.

Open-Meteo archive returns gridded reanalysis (ERA5) interpolated to
a 9km grid cell. Even at airport coordinates, it can differ from the
actual runway sensor by 0.5-1.5°C, enough to flip a whole-degree
resolution bucket ~67% of the time.

Daily-max accuracy notes (these match the Wunderground oracle):
  * The day is the station's LOCAL civil day, not the UTC day, we pass the
    station timezone to IEM so observations bucket by local midnight.
  * We include SPECI (special, off-hour) reports, not just routine hourly
    METARs, via report_type=3,4.
  * The displayed daily high can exceed the max of the hourly spot readings
    because the true peak fell between reports. We recover it from two METAR
    encodings: the precise hourly temperature group (``Tsnnnsnnn``, 0.1°C) and
    the 6-hour maximum-temperature group (``1snnn``) in the remarks.
"""
from __future__ import annotations

import csv
import io
import logging
import re
from collections import defaultdict
from datetime import date

import httpx

logger = logging.getLogger(__name__)

IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


# Minimum hourly readings required to trust a daily max.
# Prevents bad resolution from sparse/incomplete station data.
MIN_READINGS = 12

# Precise hourly temperature group: T s nnn s nnn (tenths of °C; s 0=+, 1=-).
# Example: T02110017 -> +21.1°C air temp, +1.7°C dewpoint.
_T_GROUP = re.compile(r"\bT([01])(\d{3})([01])(\d{3})\b")
# 6-hour maximum-temperature group in the remarks: 1 s nnn (tenths of °C).
# Example: 10217 -> +21.7°C max over the past 6 hours.
_SIXHR_MAX = re.compile(r"\b1([01])(\d{3})\b")


def _point_temp_c_from_metar(metar: str) -> list[float]:
    """Precise hourly temperature (T-group) from a METAR, in °C.

    This is timestamped to its own observation hour, so it is always safe to
    attribute to that observation's local day.
    """
    if not metar:
        return []
    m = _T_GROUP.search(metar)
    if not m:
        return []
    sign = -1 if m.group(1) == "1" else 1
    return [sign * int(m.group(2)) / 10.0]


def _sixhr_max_c_from_metar(metar: str) -> list[float]:
    """6-hour maximum-temperature group(s) from a METAR's remarks, in °C.

    The 6-hr max covers the PRECEDING ~6 hours, so the CALLER must only
    attribute it to the report's local day when the report is from the
    afternoon/evening (otherwise an after-midnight report imports the previous
    evening's high). Only the remarks section is scanned to avoid colliding with
    numeric groups in the METAR body (e.g. "10SM" visibility).
    """
    if not metar:
        return []
    parts = metar.split(" RMK", 1)
    if len(parts) != 2:
        return []
    out: list[float] = []
    for mm in _SIXHR_MAX.finditer(parts[1]):
        sign = -1 if mm.group(1) == "1" else 1
        out.append(sign * int(mm.group(2)) / 10.0)
    return out


def _temps_c_from_metar(metar: str) -> list[float]:
    """All Celsius temperatures encoded in a METAR (T-group + 6-hr max groups).

    Convenience wrapper. _daily_max_from_rows uses the two component helpers
    directly so it can apply the time-of-day gate to the 6-hr max group.
    """
    return _point_temp_c_from_metar(metar) + _sixhr_max_c_from_metar(metar)


def _daily_max_from_rows(
    rows: list[dict[str, str]],
) -> dict[str, tuple[float, float, int]]:
    """Group IEM rows by (local) date -> (max_c, max_f, readings).

    ``readings`` counts observation rows that carried a usable temperature, for
    the MIN_READINGS sufficiency gate. The max combines native tmpc, native
    tmpf, and METAR-encoded temperatures so the daily high matches Wunderground.
    """
    c_vals: dict[str, list[float]] = defaultdict(list)
    f_vals: dict[str, list[float]] = defaultdict(list)
    metar_c: dict[str, list[float]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)

    for row in rows:
        valid = row.get("valid") or ""
        d = valid[:10]
        if not d:
            continue
        # Local hour of this report (valid is in the requested station tz).
        try:
            hour = int(valid[11:13])
        except (ValueError, IndexError):
            hour = 12  # unknown -> treat as daytime (don't drop data)
        got = False
        tmpc = row.get("tmpc", "M")
        tmpf = row.get("tmpf", "M")
        if tmpc and tmpc != "M":
            try:
                c_vals[d].append(float(tmpc))
                got = True
            except ValueError:
                pass
        if tmpf and tmpf != "M":
            try:
                f_vals[d].append(float(tmpf))
                got = True
            except ValueError:
                pass
        metar = row.get("metar", "")
        # The T-group is timestamped to this hour -> always safe.
        metar_c[d].extend(_point_temp_c_from_metar(metar))
        # The 6-hr max covers the preceding ~6h. Only credit it to this local day
        # for afternoon/evening reports (hour >= 12); an after-midnight report
        # would otherwise import the previous evening's high into the new day.
        if hour >= 12:
            metar_c[d].extend(_sixhr_max_c_from_metar(metar))
        if got:
            counts[d] += 1

    result: dict[str, tuple[float, float, int]] = {}
    all_dates = set(c_vals) | set(f_vals) | set(metar_c)
    for d in all_dates:
        candidates_c = list(c_vals.get(d, []))
        candidates_c += [(f - 32.0) * 5.0 / 9.0 for f in f_vals.get(d, [])]
        candidates_c += metar_c.get(d, [])
        if not candidates_c:
            continue
        max_c = max(candidates_c)

        candidates_f = list(f_vals.get(d, []))
        candidates_f += [c * 9.0 / 5.0 + 32.0 for c in c_vals.get(d, [])]
        candidates_f += [c * 9.0 / 5.0 + 32.0 for c in metar_c.get(d, [])]
        max_f = max(candidates_f) if candidates_f else max_c * 9.0 / 5.0 + 32.0

        result[d] = (max_c, max_f, counts.get(d, 0))
    return result


async def _fetch_iem_rows(
    station_id: str,
    start_date: date,
    end_date: date,
    *,
    station_tz: str,
    timeout: float,
) -> list[dict[str, str]] | None:
    """Fetch raw IEM ASOS rows (tmpf, tmpc, metar) for a station + date range.

    Dates and timestamps are interpreted in ``station_tz`` so daily grouping is
    by the local civil day. Includes routine (3) and special (4) reports.
    Returns None on transport error.
    """
    params = {
        "station": station_id,
        "data": "tmpf,tmpc,metar",
        "year1": start_date.year,
        "month1": start_date.month,
        "day1": start_date.day,
        "year2": end_date.year,
        "month2": end_date.month,
        "day2": end_date.day,
        "tz": station_tz,
        "format": "onlycomma",
        "latlon": "no",
        "elev": "no",
        "missing": "M",
        "trace": "T",
        "direct": "no",
        "report_type": ["3", "4"],
    }
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(IEM_ASOS_URL, params=params, timeout=timeout)
            resp.raise_for_status()
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            logger.warning(
                "IEM METAR fetch failed for %s (%s to %s): %s",
                station_id, start_date, end_date, e,
            )
            return None
    return list(csv.DictReader(io.StringIO(resp.text)))


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
    station_tz: str = "Etc/UTC",
) -> float | None:
    """Fetch observed daily high temperature from METAR station data.

    Returns temperature in Celsius, or None if unavailable. Pass ``station_tz``
    (the station's IANA timezone) so the daily max is taken over the local civil
    day rather than the UTC day.
    """
    if icao == "45005":
        daily = await _fetch_hko_range(target_date, target_date, timeout=timeout)
        val = daily.get(str(target_date))
        if val is not None:
            logger.info("HKO obs for 45005 on %s: %.1f°C", target_date, val)
        return val

    station_id = icao[1:] if icao.startswith("K") else icao
    rows = await _fetch_iem_rows(
        station_id, target_date, target_date, station_tz=station_tz, timeout=timeout,
    )
    if rows is None:
        return None

    daily = _daily_max_from_rows(rows)
    entry = daily.get(str(target_date))
    if entry is None:
        return None
    max_c, max_f, readings = entry
    if readings < MIN_READINGS:
        logger.debug(
            "Insufficient METAR readings for %s on %s: %d < %d",
            icao, target_date, readings, MIN_READINGS,
        )
        return None
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
    station_tz: str = "Etc/UTC",
) -> tuple[float | None, float | None]:
    """Fetch daily max in both native Celsius AND native Fahrenheit.

    Returns (max_c, max_f). Either may be None if no data. For Fahrenheit
    markets the resolver rounds max_f directly to avoid C->F conversion error.
    """
    if icao == "45005":
        daily = await _fetch_hko_range(target_date, target_date, timeout=timeout)
        val_c = daily.get(str(target_date))
        if val_c is not None:
            return val_c, val_c * 9.0 / 5.0 + 32.0
        return None, None

    station_id = icao[1:] if icao.startswith("K") else icao
    rows = await _fetch_iem_rows(
        station_id, target_date, target_date, station_tz=station_tz, timeout=timeout,
    )
    if rows is None:
        return None, None

    daily = _daily_max_from_rows(rows)
    entry = daily.get(str(target_date))
    if entry is None:
        return None, None
    max_c, max_f, readings = entry
    if readings < MIN_READINGS:
        return None, None
    return max_c, max_f


async def fetch_station_tmax_range(
    icao: str,
    start_date: date,
    end_date: date,
    *,
    timeout: float = 30.0,
    station_tz: str = "Etc/UTC",
) -> dict[str, float]:
    """Fetch daily high temperatures (°C) for a date range, keyed by local date.

    More efficient than calling fetch_station_tmax per day.
    """
    if icao == "45005":
        return await _fetch_hko_range(start_date, end_date, timeout=timeout)

    station_id = icao[1:] if icao.startswith("K") else icao
    rows = await _fetch_iem_rows(
        station_id, start_date, end_date, station_tz=station_tz, timeout=timeout,
    )
    if rows is None:
        return {}

    daily = _daily_max_from_rows(rows)
    result: dict[str, float] = {
        d: max_c
        for d, (max_c, _max_f, readings) in daily.items()
        if readings >= MIN_READINGS
    }
    logger.info(
        "METAR range for %s: %d days with data (%s to %s)",
        icao, len(result), start_date, end_date,
    )
    return result
