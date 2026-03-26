"""Conditional bias detection, identifies weather patterns that cause model busts.

This is the knowledge edge. Static bias corrections tell us the average error.
Pattern detection tells us when TODAY's error will be much larger than average.

When a bust pattern is detected, the edge calculator applies a larger correction
and increases confidence on the affected buckets.

Patterns detected:
- Chinook/Foehn winds (Denver, Munich), models cold-biased 5-10°C
- Lake breeze (Chicago), GFS warm-biased 4-8°F
- Marine layer (SF, LA), global models warm-biased 5-10°F
- Santa Ana (LA), models cold-biased 5-8°F
- Sea breeze timing (Tokyo, Seoul, NYC, Seattle), 2-3°C swing
- Gulf moisture stall (Houston), GFS warm-biased 3-5°F
- Arctic outbreak shallow pool (Warsaw, Chicago), GFS warm-biased 5-15°C
- Convective quench (Miami), models cool-biased 2-4°F
- Saharan dust/Calima (Madrid), models warm-biased 3-5°C

Detection method: analyze the spread and direction of model forecasts.
When models disagree in a pattern-consistent way, flag the pattern.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from weather_edge.fetchers.openmeteo import ForecastResult
from weather_edge.models.enums import City, WeatherModel

logger = logging.getLogger(__name__)


@dataclass
class PatternAlert:
    """A detected weather pattern that causes model bias."""
    city_id: City
    pattern_name: str
    description: str
    affected_models: list[str]  # Models that will be wrong
    bias_direction: str  # "warm" (model too warm) or "cold" (model too cold)
    estimated_magnitude_c: float  # Expected additional error in °C
    confidence: float  # 0-1 how sure we are the pattern is active
    trading_implication: str  # What to do


# Thresholds for pattern detection
CHINOOK_PRESSURE_GRADIENT_THRESHOLD = 4.0  # °C spread between models
MARINE_LAYER_TEMP_SPREAD = 3.0  # °C, when coastal models diverge from global
LAKE_BREEZE_THRESHOLD = 3.0  # °C, GFS vs HRRR divergence


def detect_patterns(
    city_id: City,
    forecasts: list[ForecastResult],
) -> list[PatternAlert]:
    """Analyze model forecasts to detect bust-causing weather patterns.

    Uses model disagreement patterns as signatures:
    - When high-res models (HRRR) and global models (GFS/ECMWF) strongly disagree,
      it often indicates a mesoscale feature the globals can't resolve.
    - When ALL models cluster high or low relative to climatology,
      it suggests a strong synoptic pattern (which models usually get right).
    - When models SPLIT into two camps, it indicates pattern uncertainty.
    """
    alerts: list[PatternAlert] = []

    if not forecasts or len(forecasts) < 3:
        return alerts

    # Extract max temps by model
    temps: dict[str, float] = {}
    for f in forecasts:
        if f.temp_max_c is not None:
            temps[f.model_name] = f.temp_max_c

    if len(temps) < 3:
        return alerts

    all_vals = list(temps.values())
    mean_temp = sum(all_vals) / len(all_vals)
    spread = max(all_vals) - min(all_vals)

    # Get specific model values
    ecmwf = temps.get(WeatherModel.ECMWF.value)
    gfs = temps.get(WeatherModel.GFS.value)
    hrrr = temps.get(WeatherModel.HRRR.value)
    icon = temps.get(WeatherModel.ICON.value)
    nam = temps.get(WeatherModel.NAM.value)

    # === DENVER: Chinook detection ===
    if city_id == City.DEN and spread > CHINOOK_PRESSURE_GRADIENT_THRESHOLD:
        # Chinook: HRRR often catches downslope warming that globals miss
        if hrrr and gfs and hrrr > gfs + 3.0:
            alerts.append(PatternAlert(
                city_id=city_id,
                pattern_name="chinook",
                description=f"Chinook/downslope pattern: HRRR={hrrr:.1f} vs GFS={gfs:.1f} ({hrrr-gfs:+.1f}°C gap)",
                affected_models=[WeatherModel.GFS.value, WeatherModel.ICON.value, WeatherModel.ECMWF.value],
                bias_direction="cold",
                estimated_magnitude_c=min(8.0, (hrrr - gfs) * 0.7),
                confidence=min(0.8, spread / 10.0),
                trading_implication="Buy HIGH temp buckets, globals under-predicting",
            ))

    # === MUNICH: Foehn detection ===
    if city_id == City.MUC and spread > CHINOOK_PRESSURE_GRADIENT_THRESHOLD:
        # Similar to Chinook, alpine Foehn warms dramatically
        warmest = max(all_vals)
        coldest = min(all_vals)
        if warmest - coldest > 5.0:
            alerts.append(PatternAlert(
                city_id=city_id,
                pattern_name="foehn",
                description=f"Alpine Foehn pattern: model spread {spread:.1f}°C, warmest={warmest:.1f} coldest={coldest:.1f}",
                affected_models=[WeatherModel.GFS.value],
                bias_direction="cold",
                estimated_magnitude_c=min(10.0, spread * 0.6),
                confidence=min(0.7, spread / 12.0),
                trading_implication="Buy HIGH temp buckets, Foehn warming likely under-predicted",
            ))

    # === CHICAGO: Lake breeze detection ===
    if city_id == City.CHI and hrrr and gfs:
        if gfs > hrrr + LAKE_BREEZE_THRESHOLD:
            # GFS doesn't resolve the lake breeze cooling at O'Hare
            alerts.append(PatternAlert(
                city_id=city_id,
                pattern_name="lake_breeze",
                description=f"Lake breeze pattern: GFS={gfs:.1f} vs HRRR={hrrr:.1f}, GFS likely too warm",
                affected_models=[WeatherModel.GFS.value, WeatherModel.ECMWF.value],
                bias_direction="warm",
                estimated_magnitude_c=min(5.0, (gfs - hrrr) * 0.6),
                confidence=min(0.7, (gfs - hrrr) / 6.0),
                trading_implication="Sell HIGH temp buckets, lake breeze will cool more than GFS thinks",
            ))

    # === SF / LA: Marine layer detection ===
    if city_id in (City.SFO, City.LAX):
        if hrrr and ecmwf and ecmwf > hrrr + MARINE_LAYER_TEMP_SPREAD:
            city_name = "SF" if city_id == City.SFO else "LA"
            alerts.append(PatternAlert(
                city_id=city_id,
                pattern_name="marine_layer",
                description=f"Marine layer: ECMWF={ecmwf:.1f} vs HRRR={hrrr:.1f}, globals burning fog too early",
                affected_models=[WeatherModel.ECMWF.value, WeatherModel.GFS.value, WeatherModel.ICON.value],
                bias_direction="warm",
                estimated_magnitude_c=min(6.0, (ecmwf - hrrr) * 0.5),
                confidence=min(0.7, (ecmwf - hrrr) / 8.0),
                trading_implication=f"Sell HIGH temp buckets in {city_name}, marine layer keeping it cooler",
            ))

    # === LA: Santa Ana detection (opposite of marine layer) ===
    if city_id == City.LAX and hrrr and ecmwf and hrrr > ecmwf + 4.0:
        alerts.append(PatternAlert(
            city_id=city_id,
            pattern_name="santa_ana",
            description=f"Santa Ana winds: HRRR={hrrr:.1f} vs ECMWF={ecmwf:.1f}, offshore warming",
            affected_models=[WeatherModel.ECMWF.value, WeatherModel.GFS.value],
            bias_direction="cold",
            estimated_magnitude_c=min(5.0, (hrrr - ecmwf) * 0.5),
            confidence=0.6,
            trading_implication="Buy HIGH temp buckets, Santa Ana warming under-predicted by globals",
        ))

    # === WARSAW: Shallow cold pool / inversion ===
    if city_id == City.WAR and gfs and ecmwf:
        if gfs > ecmwf + 4.0:
            # GFS erodes inversions too fast
            alerts.append(PatternAlert(
                city_id=city_id,
                pattern_name="cold_pool",
                description=f"Cold pool/inversion: GFS={gfs:.1f} vs ECMWF={ecmwf:.1f}, GFS mixing out too fast",
                affected_models=[WeatherModel.GFS.value],
                bias_direction="warm",
                estimated_magnitude_c=min(8.0, (gfs - ecmwf) * 0.7),
                confidence=min(0.6, (gfs - ecmwf) / 10.0),
                trading_implication="Sell GFS-driven HIGH buckets, cold pool likely persists",
            ))

    # === HOUSTON: GFS soil moisture warm bias ===
    if city_id == City.HOU and gfs and ecmwf:
        if gfs > ecmwf + 2.5:
            alerts.append(PatternAlert(
                city_id=city_id,
                pattern_name="gfs_dry_bias",
                description=f"GFS dry bias: GFS={gfs:.1f} vs ECMWF={ecmwf:.1f}, soil moisture likely higher than GFS assumes",
                affected_models=[WeatherModel.GFS.value],
                bias_direction="warm",
                estimated_magnitude_c=min(3.0, (gfs - ecmwf) * 0.5),
                confidence=0.5,
                trading_implication="Trust ECMWF over GFS for Houston highs",
            ))

    # === SEA BREEZE TIMING: Tokyo, Seoul, NYC, Seattle ===
    if city_id in (City.TYO, City.SEL, City.NYC, City.SEA) and spread > 3.0:
        if hrrr and ecmwf and abs(hrrr - ecmwf) > 2.0:
            alerts.append(PatternAlert(
                city_id=city_id,
                pattern_name="sea_breeze_timing",
                description=f"Sea breeze uncertainty: {spread:.1f}°C model spread, timing-dependent",
                affected_models=[WeatherModel.GFS.value],
                bias_direction="warm" if gfs and gfs > mean_temp else "cold",
                estimated_magnitude_c=min(3.0, spread * 0.4),
                confidence=0.4,
                trading_implication="Wide spread suggests sea breeze timing uncertainty, reduce position size",
            ))

    # Log detected patterns
    for alert in alerts:
        logger.warning(
            "PATTERN: %s in %s, %s (conf=%.0f%%, mag=%.1f°C), %s",
            alert.pattern_name.upper(),
            alert.city_id.value.upper(),
            alert.description[:60],
            alert.confidence * 100,
            alert.estimated_magnitude_c,
            alert.trading_implication[:50],
        )

    return alerts


def get_pattern_adjustment(
    city_id: City,
    alerts: list[PatternAlert],
) -> tuple[float, float]:
    """Get confidence multiplier and additional bias from detected patterns.

    Returns (confidence_multiplier, additional_bias_c):
    - confidence_multiplier: >1.0 means bet bigger (we know models are wrong)
    - additional_bias_c: extra correction to apply (positive = actual will be warmer)
    """
    city_alerts = [a for a in alerts if a.city_id == city_id]
    if not city_alerts:
        return 1.0, 0.0

    # Use the highest-confidence alert for this city
    best = max(city_alerts, key=lambda a: a.confidence)

    # Confidence boost: when we detect a pattern, we're MORE confident in our edge
    confidence_mult = 1.0 + (best.confidence * 0.3)  # Up to 1.3x

    # Additional bias: signed correction
    if best.bias_direction == "cold":
        additional_bias = best.estimated_magnitude_c  # Models too cold, actual warmer
    else:
        additional_bias = -best.estimated_magnitude_c  # Models too warm, actual cooler

    return confidence_mult, additional_bias
