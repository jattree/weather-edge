#!/usr/bin/env python3
"""Report per-model, per-city forecast bias against the resolution source.

Compares Open-Meteo historical model forecasts with METAR/HKO station
observations (the data Polymarket resolves on) and prints, or writes as JSON,
the per-(city, model) bias together with the SAME significance-gated correction
the live bot would derive (analysis.bias_correction.compute_gated_bias).

This script is a diagnostic. It never rewrites source code. The live bot
computes its corrections dynamically from the forecast_snapshots table
(see scripts/hindcast.py and analysis/bias_correction.py); this report lets you
inspect what those corrections look like over an independent window.

What changed from the original generator, and why:
  * It used to OVERWRITE analysis/bias_correction.py from a hardcoded
    /Volumes/... path, replacing the dynamic, significance-gated module with a
    static table and silently undoing the gate.
  * It emitted a per-model table AND a per-city average of the same
    corrections, and the generated getter added both, so every model was
    corrected roughly twice.
  * It measured against the Open-Meteo reanalysis archive on the UTC day,
    not the station observation on the station's local civil day.

Observations now go through weather_edge.fetchers.metar (local civil day,
SPECI, T-group and 6-hour max groups, MIN_READINGS gate, HKO for Hong Kong),
and forecasts are requested in the city's IANA timezone.

Usage:
    python scripts/build_bias_table.py [--days 30] [--city nyc ...] [--out bias.json]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from weather_edge.analysis.bias_correction import compute_gated_bias  # noqa: E402
from weather_edge.config import CITIES, get_models_for_city  # noqa: E402
from weather_edge.fetchers.metar import fetch_station_tmax_range  # noqa: E402
from weather_edge.models.enums import City  # noqa: E402

FORECAST_HISTORY_API = "https://historical-forecast-api.open-meteo.com/v1/forecast"


@dataclass
class BiasRow:
    city: str
    model: str
    samples: int            # distinct local dates with both forecast and obs
    bias_c: float           # mean(forecast - observed); + = model runs warm
    correction_c: float     # gated correction the live bot would apply (0 if gated)
    notes: str


def compute_errors(
    observations: dict[str, float], forecasts: dict[str, float],
) -> list[float]:
    """One (forecast - observed) error per local date present in both."""
    return [
        forecasts[d] - observations[d]
        for d in sorted(observations)
        if d in forecasts
    ]


def bias_row(city: str, model: str, errors: list[float]) -> BiasRow:
    """Summarise one (city, model) error sample, applying the live gate.

    There is exactly one row per (city, model): no separate city-level term
    that would be added on top (the old double count).
    """
    gated = compute_gated_bias(errors)
    mean = sum(errors) / len(errors) if errors else 0.0
    return BiasRow(
        city=city,
        model=model,
        samples=len(errors),
        bias_c=round(mean, 3),
        correction_c=gated.temp_max_offset,
        notes=gated.notes,
    )


async def fetch_model_forecasts(
    client: httpx.AsyncClient,
    lat: float, lon: float, tz: str,
    start: date, end: date, model_id: str,
) -> dict[str, float]:
    """Historical daily-max forecasts keyed by station-local date."""
    resp = await client.get(
        FORECAST_HISTORY_API,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": "temperature_2m_max",
            "models": model_id,
            "timezone": tz,
        },
    )
    resp.raise_for_status()
    daily = resp.json().get("daily", {})
    temps = daily.get(f"temperature_2m_max_{model_id}") or daily.get("temperature_2m_max", [])
    return {
        d: float(t) for d, t in zip(daily.get("time", []), temps) if t is not None
    }


async def build(days: int, cities: list[City]) -> list[BiasRow]:
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=days - 1)
    rows: list[BiasRow] = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for city in cities:
            cfg = CITIES[city]
            obs = await fetch_station_tmax_range(
                cfg.icao, start, end, station_tz=cfg.timezone,
            )
            print(f"{city.value}: {len(obs)} station days ({cfg.icao})", file=sys.stderr)
            if not obs:
                continue
            for model in get_models_for_city(city):
                try:
                    fc = await fetch_model_forecasts(
                        client, cfg.latitude, cfg.longitude, cfg.timezone,
                        start, end, model.value,
                    )
                except Exception as e:  # noqa: BLE001 - report and keep going
                    print(f"  {model.value}: ERROR {e}", file=sys.stderr)
                    continue
                rows.append(bias_row(city.value, model.value, compute_errors(obs, fc)))
    return rows


def format_table(rows: list[BiasRow]) -> str:
    lines = [f"{'city':<5} {'model':<24} {'n':>4} {'bias':>7} {'corr':>7}  notes"]
    for r in sorted(rows, key=lambda r: (r.city, r.model)):
        lines.append(
            f"{r.city:<5} {r.model:<24} {r.samples:>4} "
            f"{r.bias_c:>+7.2f} {r.correction_c:>+7.2f}  {r.notes}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--days", type=int, default=30, help="Lookback window in days")
    parser.add_argument("--city", action="append", default=None,
                        help="City id (repeatable); default all cities")
    parser.add_argument("--out", type=Path, default=None,
                        help="Write JSON rows to this path instead of printing a table")
    args = parser.parse_args(argv)

    cities = [City(c) for c in args.city] if args.city else list(CITIES)
    rows = asyncio.run(build(args.days, cities))
    if not rows:
        print("No bias data collected.", file=sys.stderr)
        return 1

    if args.out:
        args.out.write_text(json.dumps([asdict(r) for r in rows], indent=2) + "\n")
        print(f"Wrote {len(rows)} rows to {args.out}", file=sys.stderr)
    else:
        print(format_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
