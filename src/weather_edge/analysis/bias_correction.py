"""Dynamic bias correction from hindcast data.

Replaces static hardcoded bias tables with data-driven corrections
computed from the forecast_snapshots table. Uses a rolling window
(default 30 days) so corrections adapt as seasons change.

Bias = mean(forecast - actual) per model per city.
Correction = -bias (subtract the systematic error).

Falls back to zero correction if insufficient data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from weather_edge.models.enums import City

logger = logging.getLogger(__name__)

# Minimum snapshots needed to trust a bias correction
MIN_SNAPSHOTS_FOR_BIAS = 14


@dataclass(frozen=True)
class BiasCorrection:
    """Temperature bias correction in °C for a model at a station."""
    temp_max_offset: float = 0.0
    temp_min_offset: float = 0.0
    notes: str = ""


# Cache to avoid hitting DB every forecast
_bias_cache: dict[tuple[str, str], BiasCorrection] = {}
_cache_age: float = 0


def _load_dynamic_bias(model_name: str, city_id: City) -> BiasCorrection:
    """Compute bias correction from hindcast data.

    Queries forecast_snapshots for this model+city, computes
    mean(forecast - actual) over the most recent data, and returns
    the negative as the correction offset.
    """
    try:
        from weather_edge.persistence import PersistentStore
        store = PersistentStore()

        rows = store.conn.execute(
            """SELECT forecast_value, actual_value
               FROM forecast_snapshots
               WHERE model_name = ? AND city_id = ?
               AND actual_value IS NOT NULL
               ORDER BY target_date DESC
               LIMIT 90""",
            (model_name, city_id.value),
        ).fetchall()
        store.close()

        if len(rows) < MIN_SNAPSHOTS_FOR_BIAS:
            return BiasCorrection(notes="insufficient data")

        errors = [r["forecast_value"] - r["actual_value"] for r in rows]
        mean_bias = sum(errors) / len(errors)

        # Correction = negative of bias (if model runs warm, subtract)
        correction = -mean_bias

        return BiasCorrection(
            temp_max_offset=round(correction, 3),
            temp_min_offset=round(correction, 3),
            notes=f"dynamic {len(rows)}-sample, bias={mean_bias:+.2f}°C",
        )
    except Exception as e:
        logger.debug("Dynamic bias lookup failed: %s", e)
        return BiasCorrection()


def get_bias_correction(model_name: str, city_id: City) -> BiasCorrection:
    """Get bias correction for a model at a city.

    Uses dynamic corrections from hindcast data when available.
    Caches results to avoid repeated DB queries within a cycle.
    """
    import time
    global _cache_age

    cache_key = (model_name, city_id.value)

    # Refresh cache every 30 minutes
    now = time.time()
    if now - _cache_age > 1800:
        _bias_cache.clear()
        _cache_age = now

    if cache_key in _bias_cache:
        return _bias_cache[cache_key]

    correction = _load_dynamic_bias(model_name, city_id)
    _bias_cache[cache_key] = correction
    return correction


def get_station_offset(city_id: City) -> float:
    """Get Layer 2 offset: Open-Meteo archive vs METAR station observation.

    The hindcast bias correction (Layer 1) calibrates model forecasts against
    Open-Meteo archive observations. But Polymarket resolves against Wunderground
    (real airport METAR data), which differs from Open-Meteo by ~0.9°C MAE.

    This offset corrects for that gap: station_offset = mean(OM_archive - METAR)
    per city, stored in the station_offsets table. A negative offset means OM reads
    colder than the station, so we add a positive correction to shift predictions
    toward the station reading.
    """
    cache_key = ("_station_offset", city_id.value)
    if cache_key in _bias_cache:
        return _bias_cache[cache_key].temp_max_offset

    try:
        from weather_edge.persistence import PersistentStore
        store = PersistentStore()

        row = store.conn.execute(
            """SELECT avg_offset, sample_count
               FROM station_offsets
               WHERE city_id = ?""",
            (city_id.value,),
        ).fetchone()
        store.close()

        if row and row["sample_count"] >= MIN_SNAPSHOTS_FOR_BIAS:
            offset = -row["avg_offset"]  # Negate: if OM reads low, add positive
            _bias_cache[cache_key] = BiasCorrection(
                temp_max_offset=round(offset, 3),
                notes=f"station offset {row['sample_count']}-sample",
            )
            return round(offset, 3)
    except Exception as e:
        logger.debug("Station offset lookup failed for %s: %s", city_id.value, e)

    _bias_cache[cache_key] = BiasCorrection()
    return 0.0


def apply_bias_correction(
    value: float,
    variable: str,
    model_name: str,
    city_id: City,
) -> float:
    """Apply two-layer bias correction to a model forecast value.

    Layer 1: Model vs Open-Meteo archive (from hindcast data)
    Layer 2: Open-Meteo archive vs METAR station (station offset)

    Returns the corrected value that should match the actual airport
    sensor reading used by Polymarket/Wunderground for resolution.
    """
    correction = get_bias_correction(model_name, city_id)
    station_offset = get_station_offset(city_id)

    if "max" in variable:
        return value + correction.temp_max_offset + station_offset
    elif "min" in variable:
        return value + correction.temp_min_offset + station_offset

    return value


def get_all_biases(limit_cities: list[str] | None = None) -> list[dict]:
    """Get all current bias corrections for reporting.

    Returns list of dicts with model, city, bias, correction, sample_size.
    """
    try:
        from weather_edge.persistence import PersistentStore
        store = PersistentStore()

        query = """SELECT model_name, city_id,
            COUNT(*) as n,
            ROUND(AVG(forecast_value - actual_value), 3) as bias,
            ROUND(AVG(ABS(forecast_value - actual_value)), 3) as mae
            FROM forecast_snapshots
            WHERE actual_value IS NOT NULL
            GROUP BY model_name, city_id
            ORDER BY city_id, model_name"""

        results = []
        for r in store.conn.execute(query).fetchall():
            if limit_cities and r["city_id"] not in limit_cities:
                continue
            results.append({
                "model": r["model_name"],
                "city": r["city_id"],
                "samples": r["n"],
                "bias": r["bias"],
                "mae": r["mae"],
                "correction": round(-r["bias"], 3),
            })
        store.close()
        return results
    except Exception:
        return []
