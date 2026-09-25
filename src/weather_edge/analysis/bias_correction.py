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

# Minimum independent samples (distinct target dates, NOT raw snapshots)
# needed to trust a bias correction.
MIN_SNAPSHOTS_FOR_BIAS = 14

# Rolling window: the most recent N distinct target dates.
BIAS_WINDOW_DATES = 90

# Significance gate: only apply a correction whose mean bias is
# distinguishable from sampling noise (|bias| > t * standard error).
# This gate is a precaution; it has not itself been validated. It is
# motivated by the 2026-04-01 validation of a DIFFERENT layer, the
# static station offsets (Layer 2): applied uniformly, those offsets
# helped cities with large real offsets (HKG -1.32C, +49% MAE) but
# damaged cities whose raw forecast was already excellent (London raw
# MAE 0.11C, 302% worse). The same failure mode is plausible for this
# dynamic per-model correction (Layer 1), so only significant biases
# are applied.
BIAS_SIGNIFICANCE_T = 2.0


@dataclass(frozen=True)
class BiasCorrection:
    """Temperature bias correction in °C for a model at a station."""
    temp_max_offset: float = 0.0
    temp_min_offset: float = 0.0
    notes: str = ""


# Cache to avoid hitting DB every forecast
_bias_cache: dict[tuple[str, str], BiasCorrection] = {}
_cache_age: float = 0


def compute_gated_bias(errors: list[float]) -> BiasCorrection:
    """Turn a sample of forecast errors into a (possibly gated) correction.

    errors must be INDEPENDENT samples: one (forecast - actual) per target
    date, never one per raw snapshot (see PersistentStore.
    get_daily_forecast_history). Returns a zero correction
    when the sample is too small or the mean bias is not statistically
    distinguishable from noise.
    """
    if len(errors) < MIN_SNAPSHOTS_FOR_BIAS:
        return BiasCorrection(notes="insufficient data")

    n = len(errors)
    mean_bias = sum(errors) / n

    # Significance gate: a bias within ~2 standard errors of zero is
    # indistinguishable from noise; correcting for it just degrades
    # cities whose forecast is already good (the London/Miami failure
    # in the 2026-04-01 validation).
    variance = sum((e - mean_bias) ** 2 for e in errors) / max(1, n - 1)
    std_err = (variance ** 0.5) / (n ** 0.5)
    if abs(mean_bias) < BIAS_SIGNIFICANCE_T * std_err:
        return BiasCorrection(
            notes=(
                f"gated: bias {mean_bias:+.2f}°C within noise "
                f"(±{BIAS_SIGNIFICANCE_T:.0f}·SE={std_err:.2f}°C, n={n})"
            ),
        )

    # Correction = negative of bias (if model runs warm, subtract)
    correction = -mean_bias

    return BiasCorrection(
        temp_max_offset=round(correction, 3),
        temp_min_offset=round(correction, 3),
        notes=f"dynamic {n}-sample, bias={mean_bias:+.2f}°C",
    )


def compute_bias_from_store(store, model_name: str, city_id: City) -> BiasCorrection:
    """Gated correction from one-error-per-target-date snapshot history.

    The old query read ``LIMIT 90`` raw rows. With a snapshot per model per
    30-minute cycle, that was ~2 target dates counted as ~90 independent
    samples, so the 2-SE significance gate almost always passed.
    """
    rows = store.get_daily_forecast_history(
        model_name, city_id.value, limit=BIAS_WINDOW_DATES,
    )
    errors = [r["forecast_value"] - r["actual_value"] for r in rows]
    return compute_gated_bias(errors)


def _load_dynamic_bias(model_name: str, city_id: City) -> BiasCorrection:
    """Compute bias correction from hindcast data.

    Queries forecast_snapshots for this model+city, collapsed to one error
    per target date, over the most recent BIAS_WINDOW_DATES dates, and returns
    the negative mean as the correction offset (gated on significance).
    """
    try:
        from weather_edge.persistence import PersistentStore
        store = PersistentStore()
        try:
            return compute_bias_from_store(store, model_name, city_id)
        finally:
            store.close()
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

        # One error per (model, city, target_date) first, so "samples" is the
        # number of independent days, not the number of 30-min snapshots.
        query = """SELECT model_name, city_id,
            COUNT(*) as n,
            ROUND(AVG(err), 3) as bias,
            ROUND(AVG(ABS(err)), 3) as mae
            FROM (
                SELECT model_name, city_id, target_date,
                       AVG(forecast_value) - AVG(actual_value) AS err
                FROM forecast_snapshots
                WHERE actual_value IS NOT NULL AND forecast_value IS NOT NULL
                GROUP BY model_name, city_id, target_date
            )
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
