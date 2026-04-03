"""Build station_offsets table: Open-Meteo archive vs IEM METAR per city.

Compares 90 days of Open-Meteo gridded archive data against actual METAR
station observations from IEM. The offset captures the systematic gap
between the two sources per city.

This is Layer 2 of bias correction:
  Layer 1: model forecast vs Open-Meteo archive (existing hindcast)
  Layer 2: Open-Meteo archive vs METAR station (this script)
  Total correction = Layer 1 + Layer 2

Usage:
    .venv/bin/python scripts/build_station_offsets.py [--days 90]
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from weather_edge.config import CITIES
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "weather_edge.db"
IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_iem_daily_max(
    icao: str, start: date, end: date,
) -> dict[str, float]:
    """Fetch daily max temps from IEM METAR archive. Returns {date: tmax_celsius}."""
    station_id = icao[1:] if icao.startswith("K") else icao

    params = {
        "station": station_id,
        "data": "tmpf",
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": end.year, "month2": end.month, "day2": end.day,
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no", "elev": "no",
        "missing": "M", "trace": "T",
        "direct": "no", "report_type": "3",
    }

    resp = httpx.get(IEM_ASOS_URL, params=params, timeout=30.0)
    resp.raise_for_status()

    reader = csv.DictReader(io.StringIO(resp.text))
    daily: dict[str, list[float]] = defaultdict(list)

    for row in reader:
        ts = row.get("valid", "")
        tmpf = row.get("tmpf", "M")
        if ts and tmpf and tmpf != "M":
            try:
                daily[ts[:10]].append(float(tmpf))
            except ValueError:
                continue

    return {d: (max(vs) - 32.0) * 5.0 / 9.0 for d, vs in daily.items()}


def fetch_hko_daily_max(start: date, end: date) -> dict[str, float]:
    """Fetch daily max temps from HK Observatory Open Data API. Returns {date: tmax_celsius}."""
    url = "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMMAXT&rformat=json&station=HKO"
    resp = httpx.get(url, timeout=30.0)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    
    daily: dict[str, float] = {}
    for row in data:
        try:
            y, m, d = int(row[0]), int(row[1]), int(row[2])
            obs_date = date(y, m, d)
            if start <= obs_date <= end:
                daily[str(obs_date)] = float(row[3])
        except (ValueError, IndexError):
            continue
    return daily


def fetch_openmeteo_daily_max(
    lat: float, lon: float, start: date, end: date,
) -> dict[str, float]:
    """Fetch daily max temps from Open-Meteo archive. Returns {date: tmax_celsius}."""
    resp = httpx.get(
        ARCHIVE_URL,
        params={
            "latitude": lat, "longitude": lon,
            "start_date": str(start), "end_date": str(end),
            "daily": "temperature_2m_max",
            "timezone": "UTC",
        },
        timeout=15.0,
    )
    resp.raise_for_status()
    data = resp.json()

    dates = data.get("daily", {}).get("time", [])
    temps = data.get("daily", {}).get("temperature_2m_max", [])
    return {d: t for d, t in zip(dates, temps) if t is not None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90, help="Days of history to compare")
    args = parser.parse_args()

    # Use the most recent 90 days ending yesterday.
    # IEM ASOS may not have the very latest days, that's OK, fewer samples.
    # Updated 2026-04-01: was Oct-Dec 2025, now Jan-Mar 2026 for seasonal relevance.
    end = date.today() - timedelta(days=1)  # yesterday
    start = end - timedelta(days=args.days)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Create table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS station_offsets (
            city_id TEXT PRIMARY KEY,
            icao TEXT NOT NULL,
            avg_offset REAL NOT NULL,
            mae REAL NOT NULL,
            std_dev REAL NOT NULL,
            max_gap REAL NOT NULL,
            rounding_mismatch_pct REAL NOT NULL,
            sample_count INTEGER NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    logger.info("Building station offsets: %s to %s (%d days)", start, end, args.days)

    for city_enum, config in CITIES.items():
        city_id = city_enum.value
        icao = config.icao

        logger.info("Processing %s (%s)...", config.name, icao)

        try:
            if icao == "45005":
                iem = fetch_hko_daily_max(start, end)
            else:
                iem = fetch_iem_daily_max(icao, start, end)
        except Exception as e:
            logger.error("  IEM failed: %s", e)
            continue

        time.sleep(0.5)

        try:
            om = fetch_openmeteo_daily_max(config.latitude, config.longitude, start, end)
        except Exception as e:
            logger.error("  Open-Meteo failed: %s", e)
            continue

        time.sleep(0.5)

        # Compare overlapping days
        gaps = []
        rounding_mismatches = 0
        matched = 0
        for d in iem:
            if d in om:
                gap = om[d] - iem[d]  # positive = OM reads higher than station
                gaps.append(gap)
                matched += 1
                if round(om[d]) != round(iem[d]):
                    rounding_mismatches += 1

        if matched < 14:
            logger.warning("  Only %d overlapping days, skipping", matched)
            continue

        avg = sum(gaps) / len(gaps)
        mae = sum(abs(g) for g in gaps) / len(gaps)
        std = (sum((g - avg) ** 2 for g in gaps) / len(gaps)) ** 0.5
        max_gap = max(abs(g) for g in gaps)
        rnd_pct = rounding_mismatches / matched * 100

        logger.info(
            "  %s: %d days, bias=%+.2f°C, MAE=%.2f°C, rounding_miss=%.1f%%",
            icao, matched, avg, mae, rnd_pct,
        )

        from datetime import datetime
        conn.execute(
            """INSERT OR REPLACE INTO station_offsets
               (city_id, icao, avg_offset, mae, std_dev, max_gap,
                rounding_mismatch_pct, sample_count, start_date, end_date, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (city_id, icao, round(avg, 4), round(mae, 4), round(std, 4),
             round(max_gap, 2), round(rnd_pct, 1), matched,
             str(start), str(end), datetime.utcnow().isoformat()),
        )
        conn.commit()

    # Summary
    logger.info("\n=== STATION OFFSET SUMMARY ===")
    for row in conn.execute(
        "SELECT city_id, icao, avg_offset, mae, rounding_mismatch_pct, sample_count "
        "FROM station_offsets ORDER BY mae DESC"
    ).fetchall():
        logger.info(
            "  %s (%s): offset=%+.2f°C, MAE=%.2f°C, rounding_miss=%.1f%%, n=%d",
            row["city_id"], row["icao"], row["avg_offset"], row["mae"],
            row["rounding_mismatch_pct"], row["sample_count"],
        )

    conn.close()
    logger.info("Done. Station offsets saved to %s", DB_PATH)


if __name__ == "__main__":
    main()
