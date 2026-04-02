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


def apply_bias_correction(
    value: float,
    variable: str,
    model_name: str,
    city_id: City,
) -> float:
    """Apply bias correction to a model forecast value.

    Single-layer: model forecast vs METAR station observation.
    Once the hindcast is rebuilt with METAR actuals, this directly
    calibrates models against what Polymarket resolves on.

    The station_offsets table is kept for diagnostics but no longer
    applied as a correction layer, it was a patch for the old
    Open-Meteo-based hindcast and would compound errors if applied
    on top of METAR-calibrated biases.
    """
    correction = get_bias_correction(model_name, city_id)

    if "max" in variable:
        return value + correction.temp_max_offset
    elif "min" in variable:
        return value + correction.temp_min_offset

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
