"""Wet bulb temperature calculation for humidity-sensitive forecasting.

Wet bulb temperature accounts for both heat AND humidity, which matters for:
1. Humid city model bias (Houston, Miami, Hong Kong), models predict dry-bulb
   max but actual perceived/measured conditions depend on moisture
2. Energy demand forecasting, cooling load scales with wet bulb, not dry bulb.
   Data centers hit critical thresholds at specific wet-bulb temps.
3. Heat stress thresholds, cities with high wet bulb have different
   temperature-to-impact curves than dry cities

Uses the Stull (2011) approximation: accurate within 0.3°C for RH 5-99%.
"""
from __future__ import annotations

import math


def wet_bulb_stull(temp_c: float, rh_pct: float) -> float:
    """Calculate wet bulb temperature using Stull (2011) formula.

    Args:
        temp_c: Dry bulb temperature in Celsius
        rh_pct: Relative humidity in percent (0-100)

    Returns:
        Wet bulb temperature in Celsius

    Accurate within 0.3°C for RH 5-99% and temp -20 to 50°C.
    """
    t = temp_c
    r = rh_pct

    tw = (
        t * math.atan(0.151977 * math.sqrt(r + 8.313659))
        + math.atan(t + r)
        - math.atan(r - 1.676331)
        + 0.00391838 * r ** 1.5 * math.atan(0.023101 * r)
        - 4.686035
    )
    return round(tw, 1)


def humidity_bias_factor(temp_c: float, rh_pct: float) -> float:
    """Calculate how much humidity affects model bias for this city/condition.

    High wet-bulb depression (dry air) = models are more accurate
    Low wet-bulb depression (humid air) = models tend to over-predict max temp
    because evaporative cooling suppresses the actual peak.

    Returns a multiplier:
    - 1.0 = dry conditions, no humidity bias
    - 0.85-0.95 = moderate humidity, slight cooling bias
    - 0.7-0.85 = high humidity, models likely over-predict max temp
    """
    wb = wet_bulb_stull(temp_c, rh_pct)
    depression = temp_c - wb  # Wet bulb depression

    if depression > 10:
        # Very dry, models are reliable for max temp
        return 1.0
    elif depression > 5:
        # Moderate humidity
        return 0.95
    elif depression > 2:
        # Humid, models may over-predict by 1-2°C
        return 0.88
    else:
        # Extremely humid (tropical), models significantly over-predict
        return 0.80


# Cities where humidity significantly affects temperature forecast accuracy
HUMIDITY_SENSITIVE_CITIES = {
    "hou": True,  # Houston: Gulf moisture, GFS warm bias amplified by humidity
    "mia": True,  # Miami: perpetually humid, models over-predict max
    "hkg": True,  # Hong Kong: subtropical humidity
    "sha": True,  # Shanghai: coastal humid
    "szn": True,  # Shenzhen: Pearl River Delta humidity
    "atl": True,  # Atlanta: summer humidity affects convective initiation
    "nyc": False,  # NYC: coastal but not consistently humid enough to bias
    "dal": False,  # Dallas: generally dry heat
    "den": False,  # Denver: very dry
    "sea": False,  # Seattle: marine influence but cool
}
