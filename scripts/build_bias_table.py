#!/usr/bin/env python3
"""Build NWS station bias correction tables from real historical data.

Uses Open-Meteo APIs to compare actual observed temperatures against model
forecasts over the past 30 days, computing per-model per-city bias corrections.

Usage:
    python scripts/build_bias_table.py
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_HISTORY_API = "https://historical-forecast-api.open-meteo.com/v1/forecast"

LOOKBACK_DAYS = 30

# End date is yesterday (latest full day of observations), start is 30 days before that.
END_DATE = date.today() - timedelta(days=1)
START_DATE = END_DATE - timedelta(days=LOOKBACK_DAYS - 1)


@dataclass
class StationConfig:
    city_enum: str    # e.g. "NYC"
    city_value: str   # e.g. "nyc"
    station: str      # e.g. "KLGA"
    lat: float
    lon: float
    models: list[str]


STATIONS: list[StationConfig] = [
    StationConfig("NYC", "nyc", "LGA", 40.7743, -73.8726, [
        "ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless", "ncep_hrrr_conus",
    ]),
    StationConfig("LON", "lon", "EGLC", 51.5053, 0.0553, [
        "ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless", "ukmo_seamless",
    ]),
    StationConfig("DAL", "dal", "KDAL", 32.8471, -96.8518, [
        "ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless", "ncep_hrrr_conus",
    ]),
    StationConfig("SEA", "sea", "KSEA", 47.4502, -122.3088, [
        "ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless", "ncep_hrrr_conus",
    ]),
    StationConfig("ATL", "atl", "KATL", 33.6407, -84.4277, [
        "ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless", "ncep_hrrr_conus",
    ]),
    StationConfig("TOR", "tor", "CYYZ", 43.6777, -79.6248, [
        "ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless",
    ]),
    StationConfig("SEL", "sel", "RKSI", 37.4691, 126.4505, [
        "ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless",
    ]),
]

# Map model API names to WeatherModel enum names for code generation
MODEL_ENUM_MAP: dict[str, str] = {
    "ecmwf_ifs025": "ECMWF",
    "gfs_seamless": "GFS",
    "icon_seamless": "ICON",
    "gem_seamless": "GEM",
    "ncep_hrrr_conus": "HRRR",
    "ukmo_seamless": "UKV",
}

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

async def fetch_observations(
    client: httpx.AsyncClient, station: StationConfig
) -> dict[str, list[float | None]]:
    """Fetch actual observed temperatures from Open-Meteo Archive API."""
    params = {
        "latitude": station.lat,
        "longitude": station.lon,
        "start_date": START_DATE.isoformat(),
        "end_date": END_DATE.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": "UTC",
    }
    resp = await client.get(ARCHIVE_API, params=params)
    resp.raise_for_status()
    data = resp.json()
    daily = data["daily"]
    return {
        "dates": daily["time"],
        "obs_max": daily["temperature_2m_max"],
        "obs_min": daily["temperature_2m_min"],
    }


async def fetch_model_forecast(
    client: httpx.AsyncClient,
    station: StationConfig,
    model: str,
) -> dict[str, list[float | None]]:
    """Fetch historical model forecasts from Open-Meteo Historical Forecast API."""
    params = {
        "latitude": station.lat,
        "longitude": station.lon,
        "start_date": START_DATE.isoformat(),
        "end_date": END_DATE.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min",
        "models": model,
        "timezone": "UTC",
    }
    resp = await client.get(FORECAST_HISTORY_API, params=params)
    resp.raise_for_status()
    data = resp.json()
    daily = data["daily"]
    return {
        "dates": daily["time"],
        "fcst_max": daily["temperature_2m_max"],
        "fcst_min": daily["temperature_2m_min"],
    }


# ---------------------------------------------------------------------------
# Bias computation
# ---------------------------------------------------------------------------

@dataclass
class BiasResult:
    city_enum: str
    city_value: str
    model: str
    bias_max: float      # mean(forecast - observed) for daily max
    bias_min: float      # mean(forecast - observed) for daily min
    sample_count: int
    notes: str = ""


def compute_bias(
    obs: dict[str, list[float | None]],
    fcst: dict[str, list[float | None]],
) -> tuple[float, float, int]:
    """Compute mean bias (forecast - observed) for max and min temps.

    Returns (bias_max, bias_min, sample_count).
    Positive bias = model over-predicts (runs too warm).
    We negate this for the correction offset (subtract the bias).
    """
    diffs_max: list[float] = []
    diffs_min: list[float] = []

    obs_max = obs["obs_max"]
    obs_min = obs["obs_min"]
    fcst_max = fcst["fcst_max"]
    fcst_min = fcst["fcst_min"]

    for i in range(min(len(obs_max), len(fcst_max))):
        if obs_max[i] is not None and fcst_max[i] is not None:
            diffs_max.append(fcst_max[i] - obs_max[i])
        if obs_min[i] is not None and fcst_min[i] is not None:
            diffs_min.append(fcst_min[i] - obs_min[i])

    if not diffs_max:
        return 0.0, 0.0, 0

    bias_max = sum(diffs_max) / len(diffs_max)
    bias_min = sum(diffs_min) / len(diffs_min) if diffs_min else 0.0
    return bias_max, bias_min, len(diffs_max)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run() -> list[BiasResult]:
    """Fetch all data and compute biases."""
    results: list[BiasResult] = []
    sem = asyncio.Semaphore(5)  # Rate limit concurrent requests

    async with httpx.AsyncClient(timeout=30.0) as client:
        for station in STATIONS:
            print(f"\n{'='*60}")
            print(f"  Station: {station.station} ({station.city_enum})")
            print(f"  Coords: {station.lat}, {station.lon}")
            print(f"  Period: {START_DATE} to {END_DATE} ({LOOKBACK_DAYS} days)")
            print(f"{'='*60}")

            # Fetch observations
            try:
                obs = await fetch_observations(client, station)
                obs_count = sum(1 for v in obs["obs_max"] if v is not None)
                print(f"  Observations fetched: {obs_count} days with data")
            except httpx.HTTPStatusError as e:
                print(f"  ERROR fetching observations: {e}")
                continue
            except Exception as e:
                print(f"  ERROR fetching observations: {e}")
                continue

            # Fetch each model's forecasts
            for model in station.models:
                async with sem:
                    try:
                        fcst = await fetch_model_forecast(client, station, model)
                        bias_max, bias_min, n = compute_bias(obs, fcst)

                        enum_name = MODEL_ENUM_MAP.get(model, model)
                        print(
                            f"  {enum_name:>8s}: bias_max={bias_max:+.2f}C, "
                            f"bias_min={bias_min:+.2f}C  (n={n})"
                        )

                        # For the correction offset, we NEGATE the bias:
                        # If model over-predicts by +1C, correction = -1C
                        correction_max = -bias_max
                        correction_min = -bias_min

                        results.append(BiasResult(
                            city_enum=station.city_enum,
                            city_value=station.city_value,
                            model=model,
                            bias_max=round(correction_max, 2),
                            bias_min=round(correction_min, 2),
                            sample_count=n,
                        ))
                    except httpx.HTTPStatusError as e:
                        print(f"  {model}: ERROR {e.response.status_code} - {e}")
                    except Exception as e:
                        print(f"  {model}: ERROR - {e}")

    return results


def print_bias_table(results: list[BiasResult]) -> None:
    """Print a formatted bias table."""
    print("\n")
    print("=" * 80)
    print("  STATION BIAS CORRECTION TABLE")
    print(f"  Period: {START_DATE} to {END_DATE} ({LOOKBACK_DAYS} days)")
    print(f"  Positive offset = model under-predicts (add to forecast)")
    print(f"  Negative offset = model over-predicts (subtract from forecast)")
    print("=" * 80)
    print(f"  {'City':<6s} {'Model':<22s} {'Max Offset':>10s} {'Min Offset':>10s} {'N':>4s}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: (x.city_enum, x.model)):
        enum_name = MODEL_ENUM_MAP.get(r.model, r.model)
        print(
            f"  {r.city_enum:<6s} {enum_name:<22s} "
            f"{r.bias_max:>+10.2f} {r.bias_min:>+10.2f} {r.sample_count:>4d}"
        )
    print("=" * 80)


def generate_bias_correction_module(results: list[BiasResult]) -> str:
    """Generate the updated bias_correction.py source code."""
    lines: list[str] = []

    def w(line: str = "") -> None:
        lines.append(line)

    # --- Module docstring ---
    w('"""NWS station bias correction for model forecasts.')
    w("")
    w(f"AUTO-GENERATED by scripts/build_bias_table.py on {date.today().isoformat()}.")
    w(f"Data period: {START_DATE} to {END_DATE} ({LOOKBACK_DAYS} days).")
    w("")
    w("Bias corrections are computed as: -(mean(model_forecast - actual_observation))")
    w("Positive offset = model under-predicts (actual runs hotter than model)")
    w("Negative offset = model over-predicts (actual runs cooler than model)")
    w("")
    w("These corrections are applied BEFORE consensus computation to improve")
    w("the accuracy of model-derived probabilities vs NWS resolution data.")
    w("")
    w("Source: Open-Meteo Archive API (observations) + Historical Forecast API (model hindcasts).")
    w("Re-run the script periodically to keep corrections fresh.")
    w('"""')
    w("from __future__ import annotations")
    w("")
    w("from dataclasses import dataclass")
    w("")
    w("from weather_edge.models.enums import City, WeatherModel")
    w("")
    w("")
    w("@dataclass(frozen=True)")
    w("class BiasCorrection:")
    w('    """Temperature bias correction in degC for a model at a station."""')
    w("")
    w("    temp_max_offset: float = 0.0  # Add this to model forecast (positive = model under-predicts)")
    w("    temp_min_offset: float = 0.0")
    w('    notes: str = ""')
    w("")
    w("")
    w("# Known biases: model_name -> city_id -> BiasCorrection")
    w("# Positive offset means model typically under-predicts (actual runs hotter)")
    w("# Negative offset means model typically over-predicts (actual runs cooler)")
    w("STATION_BIAS: dict[str, dict[City, BiasCorrection]] = {")

    # Group results by model
    model_biases: dict[str, list[BiasResult]] = {}
    for r in results:
        model_biases.setdefault(r.model, []).append(r)

    for model in sorted(model_biases.keys()):
        enum_name = MODEL_ENUM_MAP.get(model, model)
        city_entries: list[str] = []
        for r in sorted(model_biases[model], key=lambda x: x.city_enum):
            # Skip tiny biases (< 0.05C) to keep table clean
            if abs(r.bias_max) < 0.05 and abs(r.bias_min) < 0.05:
                continue
            note = f"{r.sample_count}-day mean bias correction"
            city_entries.append(
                f"        City.{r.city_enum}: BiasCorrection("
                f"temp_max_offset={r.bias_max}, "
                f"temp_min_offset={r.bias_min}, "
                f'notes="{note}"),'
            )
        if city_entries:
            w(f"    WeatherModel.{enum_name}.value: {{")
            for entry in city_entries:
                w(entry)
            w("    },")

    w("}")
    w("")

    # --- City-level biases ---
    w("# Generic city-level bias (applied to ALL models for a city)")
    w("# These capture station-level biases independent of any particular model")
    w("# Computed as the average correction across all models analyzed for that city")
    w("CITY_BIAS: dict[City, BiasCorrection] = {")

    city_agg: dict[str, list[BiasResult]] = {}
    for r in results:
        city_agg.setdefault(r.city_enum, []).append(r)

    for city in sorted(city_agg.keys()):
        city_results = city_agg[city]
        if not city_results:
            continue
        avg_max = sum(r.bias_max for r in city_results) / len(city_results)
        avg_min = sum(r.bias_min for r in city_results) / len(city_results)
        avg_max = round(avg_max, 2)
        avg_min = round(avg_min, 2)
        if abs(avg_max) < 0.05 and abs(avg_min) < 0.05:
            continue
        n_models = len(city_results)
        w(f"    City.{city}: BiasCorrection(")
        w(f"        temp_max_offset={avg_max},")
        w(f"        temp_min_offset={avg_min},")
        w(f'        notes="Avg across {n_models} models, {LOOKBACK_DAYS}-day window",')
        w("    ),")

    w("}")
    w("")
    w("")

    # --- Functions ---
    w("def get_bias_correction(model_name: str, city_id: City) -> BiasCorrection:")
    w('    """Get the total bias correction for a model at a city.')
    w("")
    w("    Combines model-specific and city-level biases.")
    w('    """')
    w("    model_bias = STATION_BIAS.get(model_name, {}).get(city_id, BiasCorrection())")
    w("    city_bias = CITY_BIAS.get(city_id, BiasCorrection())")
    w("")
    w("    return BiasCorrection(")
    w("        temp_max_offset=model_bias.temp_max_offset + city_bias.temp_max_offset,")
    w("        temp_min_offset=model_bias.temp_min_offset + city_bias.temp_min_offset,")
    w("    )")
    w("")
    w("")
    w("def apply_bias_correction(")
    w("    value: float,")
    w("    variable: str,")
    w("    model_name: str,")
    w("    city_id: City,")
    w(") -> float:")
    w('    """Apply bias correction to a model forecast value.')
    w("")
    w("    Returns the corrected value that should better match NWS observations.")
    w('    """')
    w("    correction = get_bias_correction(model_name, city_id)")
    w("")
    w('    if "max" in variable:')
    w("        return value + correction.temp_max_offset")
    w('    elif "min" in variable:')
    w("        return value + correction.temp_min_offset")
    w("")
    w("    return value")
    w("")

    return "\n".join(lines)


def write_bias_correction(results: list[BiasResult]) -> Path:
    """Write the updated bias_correction.py module."""
    output_path = Path("/Volumes/2TB_HD/weather/src/weather_edge/analysis/bias_correction.py")
    source = generate_bias_correction_module(results)
    output_path.write_text(source)
    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Building bias correction table from real historical data...")
    print(f"Period: {START_DATE} to {END_DATE} ({LOOKBACK_DAYS} days)")

    results = asyncio.run(run())

    if not results:
        print("\nERROR: No bias data collected. Check API connectivity.")
        sys.exit(1)

    print_bias_table(results)

    output_path = write_bias_correction(results)
    print(f"\nUpdated bias correction module written to:")
    print(f"  {output_path}")
    print(f"\nTotal corrections: {len(results)} (model x city pairs)")


if __name__ == "__main__":
    main()
