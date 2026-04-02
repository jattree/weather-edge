"""Multi-model consensus computation with weighted averaging and probability estimation."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from weather_edge.analysis.bias_correction import apply_bias_correction
from weather_edge.config import get_model_weights
from weather_edge.fetchers.openmeteo import ForecastResult
from weather_edge.models.enums import City, WeatherModel

logger = logging.getLogger(__name__)

# === EMOS / CALIBRATION CONSTANTS ===
# Per Gemini analysis: raw ensemble spread underestimates true uncertainty.
# Models are correlated (shared physics + initial conditions) so spread is artificially narrow.

# Spread inflation factor: multiply raw std_dev by this before computing probabilities.
# Professional forecasting uses 1.5-2.2x. Reduced from 2.0 to 1.3 after METAR
# recalibration showed we were over-inflating uncertainty, spreading probability
# across too many buckets and killing real edges.
SPREAD_INFLATION_FACTOR = 1.3

# Bias shrinkage: apply this fraction of the bias correction.
# Raised from 0.5 to 0.9, now that the hindcast is calibrated against METAR
# (the actual resolution source), we can trust the bias numbers much more.
# Old value (0.5) was leaving ~half the systematic error uncorrected.
BIAS_SHRINKAGE = 0.9

# Probability cap: never assign more than this to a single 2°F bucket >12h out.
# A >90% single-bucket probability is "likely broken" per Gemini.
MAX_BUCKET_PROBABILITY = 0.70

# Extreme event override: when models are tightly clustered (low std) AND
# consensus is >2 std deviations from climatological normal, raise the cap.
# This catches heat domes, cold snaps, etc. where 80%+ on a bucket is genuine.
MAX_BUCKET_PROBABILITY_EXTREME = 0.85

# Per-city March climatological normals (°C) for extreme event detection.
# A forecast is "extreme" when it's >2 sigma from that city's normal.
# Source: approximate 30-year March daily high averages.
CITY_CLIMATOLOGY = {
    # US cities
    "nyc": (12.0, 4.0),   # NYC March avg high 12°C, std 4°C
    "chi": (8.0, 5.0),    # Chicago, volatile spring
    "dal": (20.0, 5.0),   # Dallas
    "hou": (22.0, 4.0),   # Houston
    "atl": (18.0, 4.0),   # Atlanta
    "mia": (28.0, 2.5),   # Miami, small variance, warm baseline
    "den": (12.0, 6.0),   # Denver, Chinook swings
    "sea": (11.0, 3.5),   # Seattle
    "lax": (20.0, 3.0),   # LA
    "sfo": (16.0, 3.0),   # SF
    "aus": (21.0, 5.0),   # Austin
    "tor": (5.0, 5.0),    # Toronto
    # Europe
    "lon": (10.0, 3.5),   # London
    "muc": (8.0, 4.5),    # Munich, Foehn swings
    "mad": (15.0, 4.0),   # Madrid
    "war": (6.0, 5.0),    # Warsaw, continental extremes
    # Asia-Pacific
    "tyo": (13.0, 3.5),   # Tokyo
    "sel": (10.0, 4.5),   # Seoul
    "hkg": (22.0, 3.0),   # Hong Kong
    "sha": (13.0, 4.0),   # Shanghai
    "szn": (23.0, 3.0),   # Shenzhen
    # South America
    "bue": (25.0, 3.5),   # Buenos Aires (southern hemisphere autumn)
    # Oceania
    "wlg": (17.0, 3.0),   # Wellington, maritime, windy, moderate
    # South Asia
    "lko": (32.0, 4.0),   # Lucknow, hot, March is pre-monsoon transition
}

# Fallback for unknown cities
CLIMATOLOGICAL_MEAN = {"temp_max_c": 15.0}
CLIMATOLOGICAL_STD = {"temp_max_c": 6.0}

# EMOS minimum variance floor: even when models perfectly agree,
# instrument error + microclimate fluctuations create irreducible uncertainty.
# Reduced from 1.2 to 0.7, the old floor was wider than the bucket itself
# (1°C), guaranteeing that even a perfect forecast got diluted across buckets.
# 0.7°C reflects actual METAR measurement noise without over-inflating.
EMOS_VARIANCE_FLOOR_C = 0.7

# Normalization ranges for confidence calculation
CONFIDENCE_NORMALIZATION: dict[str, float] = {
    "temp_max_c": 8.0,
    "temp_min_c": 8.0,
    "precip_sum_mm": 15.0,
    "snow_sum_cm": 10.0,
    "wind_max_kmh": 30.0,
}


@dataclass
class ConsensusResult:
    """Result of multi-model consensus computation."""
    city_id: str
    target_date: str  # ISO format
    variable: str
    model_count: int
    mean_value: float
    median_value: float
    std_dev: float
    min_value: float
    max_value: float
    weighted_mean: float
    model_values: dict[str, float]  # model_name -> value
    model_weights: dict[str, float]  # model_name -> normalized weight
    confidence: float  # 0-1, how much models agree
    threshold_probs: dict[str, float] = field(default_factory=dict)  # ">=20.0" -> 0.85


def _compute_threshold_probs_normal(
    mean: float,
    std: float,
    thresholds: list[float],
) -> dict[str, float]:
    """Compute P(X >= threshold) using a normal distribution fit."""
    probs = {}
    if std <= 0:
        # All models agree exactly
        for t in thresholds:
            probs[f">={t}"] = 1.0 if mean >= t else 0.0
        return probs

    for t in thresholds:
        prob = 1.0 - stats.norm.cdf(t, loc=mean, scale=std)
        probs[f">={t}"] = round(float(prob), 4)
    return probs


def _compute_threshold_probs_kde(
    values: list[float],
    weights: list[float],
    thresholds: list[float],
) -> dict[str, float]:
    """Compute P(X >= threshold) using Kernel Density Estimation.

    KDE handles multimodal distributions (when models disagree on front timing)
    and asymmetric distributions (cold tails, heat caps) better than normal fit.
    Uses Scott's bandwidth with weight support.
    """
    probs = {}
    arr = np.array(values)

    if len(arr) < 3:
        # Fall back to normal for very few models
        return _compute_threshold_probs_normal(float(np.mean(arr)), float(np.std(arr, ddof=1)), thresholds)

    try:
        # Gaussian KDE with weighted samples
        # Repeat samples by weight to approximate weighted KDE
        w_arr = np.array(weights)
        w_arr = w_arr / w_arr.sum()

        # Use scipy's gaussian_kde
        kde = stats.gaussian_kde(arr, weights=w_arr)

        # Evaluate CDF by integration
        for t in thresholds:
            # P(X >= t) = integral from t to +inf
            # Approximate with integration range
            x_max = float(np.max(arr)) + 20  # Generous upper bound
            x_grid = np.linspace(t, x_max, 500)
            pdf_vals = kde(x_grid)
            dx = (x_max - t) / 500
            prob = float(np.sum(pdf_vals) * dx)
            probs[f">={t}"] = round(max(0.0, min(1.0, prob)), 4)
    except Exception:
        # Fall back to normal if KDE fails
        return _compute_threshold_probs_normal(
            float(np.average(arr, weights=weights)),
            float(np.std(arr, ddof=1)),
            thresholds,
        )

    return probs


def _compute_threshold_probs_empirical(
    values: list[float],
    thresholds: list[float],
) -> dict[str, float]:
    """Compute P(X >= threshold) using empirical fraction of models."""
    n = len(values)
    probs = {}
    for t in thresholds:
        count = sum(1 for v in values if v >= t)
        probs[f">={t}"] = round(count / n, 4) if n > 0 else 0.0
    return probs


def _compute_snow_probability(values: list[float]) -> float:
    """Hurdle model: P(any snow) from model forecasts.

    Snow is zero-inflated, so we first estimate P(snow > 0),
    then for the amount we'd fit a distribution to nonzero values.
    """
    n = len(values)
    if n == 0:
        return 0.0
    nonzero = sum(1 for v in values if v > 0)
    return nonzero / n


def compute_consensus(
    city_id: City,
    target_date: str,
    variable: str,
    forecasts: list[ForecastResult],
    thresholds: list[float] | None = None,
) -> ConsensusResult | None:
    """Compute weighted multi-model consensus for a variable.

    Args:
        city_id: The city
        target_date: ISO date string
        variable: One of 'temp_max_c', 'temp_min_c', 'precip_sum_mm', 'snow_sum_cm', 'wind_max_kmh'
        forecasts: List of ForecastResult from different models
        thresholds: Specific thresholds to compute P(X >= t) for
    """
    # Extract the value for the requested variable from each forecast
    # Apply NWS station bias correction with shrinkage (50% of full correction)
    # ENSO regime-aware bias shrinkage
    # During La Nina → Neutral transition, reduce bias corrections for sensitive cities
    enso_shrinkage = 1.0
    try:
        from weather_edge.analysis.enso_regime import _cached_enso, get_bias_shrinkage
        if _cached_enso:
            enso_shrinkage = get_bias_shrinkage(city_id.value if hasattr(city_id, 'value') else str(city_id), _cached_enso)
    except Exception:
        pass

    effective_shrinkage = BIAS_SHRINKAGE * enso_shrinkage

    model_values: dict[str, float] = {}
    for f in forecasts:
        val = getattr(f, variable, None)
        if val is not None:
            full_correction = apply_bias_correction(val, variable, f.model_name, city_id)
            # Shrinkage: blend raw and corrected, modulated by ENSO regime
            corrected = val + (full_correction - val) * effective_shrinkage
            model_values[f.model_name] = corrected

    if not model_values:
        logger.warning("No model values for %s/%s/%s", city_id.value, target_date, variable)
        return None

    # Get weights for this city
    all_weights = get_model_weights(city_id)
    # Filter to only models we have data for
    active_weights: dict[str, float] = {}
    for model_name, value in model_values.items():
        try:
            wm = WeatherModel(model_name)
            active_weights[model_name] = all_weights.get(wm, 1.0 / len(model_values))
        except ValueError:
            active_weights[model_name] = 1.0 / len(model_values)

    # Normalize weights
    total_w = sum(active_weights.values())
    if total_w > 0:
        active_weights = {k: v / total_w for k, v in active_weights.items()}

    values = list(model_values.values())
    weights = [active_weights.get(m, 1.0 / len(values)) for m in model_values.keys()]

    values_arr = np.array(values)
    weights_arr = np.array(weights)

    # Core statistics
    mean_val = float(np.mean(values_arr))
    median_val = float(np.median(values_arr))
    raw_std = float(np.std(values_arr, ddof=1)) if len(values_arr) > 1 else 0.0
    min_val = float(np.min(values_arr))
    max_val = float(np.max(values_arr))
    weighted_mean = float(np.average(values_arr, weights=weights_arr))

    # EMOS: inflate spread to account for correlated models + add variance floor
    # Raw ensemble spread underestimates true uncertainty (Gemini-validated)
    # σ_emos² = c + d * σ_raw² where c = VARIANCE_FLOOR², d = INFLATION²
    if "temp" in variable:
        std_val = max(EMOS_VARIANCE_FLOOR_C, raw_std * SPREAD_INFLATION_FACTOR)
    else:
        # Non-temp variables: apply inflation with a small floor to avoid zero-std step functions
        NON_TEMP_VARIANCE_FLOOR = 0.5  # mm for precip, cm for snow
        std_val = max(NON_TEMP_VARIANCE_FLOOR, raw_std * SPREAD_INFLATION_FACTOR)

    # Confidence: 1 - (std / normalization_range), clamped to [0, 1]
    norm_range = CONFIDENCE_NORMALIZATION.get(variable, 10.0)
    confidence = max(0.0, min(1.0, 1.0 - (std_val / norm_range)))

    # Threshold probabilities
    if thresholds is None:
        # Generate reasonable thresholds based on variable
        if "temp" in variable:
            # Generate thresholds around the mean in 2°C steps
            center = round(weighted_mean)
            thresholds = [float(center + offset) for offset in range(-10, 12, 2)]
        elif "precip" in variable:
            thresholds = [0.0, 0.1, 1.0, 2.5, 5.0, 10.0, 25.0]
        elif "snow" in variable:
            thresholds = [0.0, 0.1, 1.0, 2.5, 5.0, 10.0]
        else:
            thresholds = []

    # Blend parametric and empirical threshold probabilities
    if variable.startswith("snow") or variable.startswith("precip"):
        # Use hurdle model for precipitation-type variables
        # P(X >= t) for t=0 is just the fraction of models predicting any
        empirical_probs = _compute_threshold_probs_empirical(values, thresholds)
        # For non-zero thresholds on zero-inflated data, parametric is unreliable
        threshold_probs = empirical_probs

        # For nonzero values, try parametric if we have enough
        nonzero_vals = [v for v in values if v > 0]
        if len(nonzero_vals) >= 3:
            nz_mean = np.mean(nonzero_vals)
            nz_std = np.std(nonzero_vals, ddof=1)
            p_any = len(nonzero_vals) / len(values)
            parametric = _compute_threshold_probs_normal(float(nz_mean), float(nz_std), thresholds)
            # Blend: P(X >= t) = P(any) * P(X >= t | X > 0)
            for key in threshold_probs:
                t = float(key.replace(">=", ""))
                if t > 0 and key in parametric:
                    hurdle_prob = p_any * parametric[key]
                    threshold_probs[key] = round(
                        0.4 * empirical_probs.get(key, 0) + 0.6 * hurdle_prob, 4
                    )
    else:
        # EMOS-calibrated probability computation:
        # 1. KDE with inflated bandwidth (captures multimodal patterns)
        # 2. Normal with EMOS-inflated std (captures spread underestimation)
        # 3. Empirical (reality check from raw model counts)
        # Weight toward parametric since EMOS inflation handles the calibration
        kde_probs = _compute_threshold_probs_kde(values, weights, thresholds)
        parametric = _compute_threshold_probs_normal(weighted_mean, std_val, thresholds)
        empirical = _compute_threshold_probs_empirical(values, thresholds)
        threshold_probs = {}
        for key in parametric:
            # Blend: 40% parametric (EMOS-calibrated) + 30% KDE + 30% empirical
            p = (0.4 * parametric.get(key, 0)
                 + 0.3 * kde_probs.get(key, 0)
                 + 0.3 * empirical.get(key, 0))
            threshold_probs[key] = round(p, 4)

    return ConsensusResult(
        city_id=city_id.value,
        target_date=target_date,
        variable=variable,
        model_count=len(model_values),
        mean_value=round(mean_val, 2),
        median_value=round(median_val, 2),
        std_dev=round(std_val, 2),
        min_value=round(min_val, 2),
        max_value=round(max_val, 2),
        weighted_mean=round(weighted_mean, 2),
        model_values=model_values,
        model_weights=active_weights,
        confidence=round(confidence, 4),
        threshold_probs=threshold_probs,
    )


def get_probability_for_threshold(
    consensus: ConsensusResult,
    threshold_c: float,
    direction: str = "gte",
) -> float:
    """Get the probability that the value meets the threshold condition.

    Args:
        consensus: Computed consensus result
        threshold_c: Threshold value (in the variable's unit, Celsius for temp)
        direction: 'gte' (>=), 'lte' (<=), or 'any' (> 0)
    """
    if direction == "any":
        # For snow/precip: P(value > 0)
        key = ">=0.0"
        if key in consensus.threshold_probs:
            return consensus.threshold_probs[key]
        # Fallback: fraction of models predicting nonzero
        vals = list(consensus.model_values.values())
        return sum(1 for v in vals if v > 0) / len(vals) if vals else 0.0

    # Find the closest threshold in our pre-computed set
    target_key = f">={threshold_c}"
    if target_key in consensus.threshold_probs:
        prob = consensus.threshold_probs[target_key]
    else:
        # Interpolate or compute on the fly
        vals = list(consensus.model_values.values())
        if len(vals) >= 2 and consensus.std_dev > 0:
            prob = float(1.0 - stats.norm.cdf(
                threshold_c, loc=consensus.weighted_mean, scale=consensus.std_dev
            ))
        else:
            prob = 1.0 if consensus.weighted_mean >= threshold_c else 0.0

    if direction == "lte":
        prob = 1.0 - prob

    # Clamp to valid range (no cap here, cap is applied at the bucket level
    # in the scheduler when computing P(low <= X <= high) for range buckets)
    return max(0.0, min(1.0, prob))
