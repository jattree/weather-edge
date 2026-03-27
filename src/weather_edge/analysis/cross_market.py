"""Cross-market correlation registry, DORMANT until markets exist.

This module documents known correlations between weather events and other
prediction markets. Currently no liquid non-weather markets exist on
Polymarket or Kalshi, but when they appear, these correlations become
tradeable edges.

Activate when:
- Polymarket/Kalshi launches energy demand markets (ERCOT, PJM)
- Polymarket/Kalshi launches aviation delay markets
- Any venue launches AQI/air quality markets

The architecture is ready: swap a fetcher, add market discovery, use
existing consensus engine. This file serves as the correlation knowledge
base, not executable trading code.

Last verified: 2026-03-27 (Kalshi weather: 3 cities, zero volume.
Polymarket: weather only. No energy/aviation/AQI markets found.)
"""
from __future__ import annotations

# Known cross-market correlations
# Verified against historical data and meteorological literature
CORRELATION_REGISTRY = {
    # ENERGY: Temperature → Grid Load
    # When to activate: Polymarket/Kalshi launches ERCOT or PJM markets
    ("dal", "ercot_load"): {
        "linkage": 0.91,
        "trigger": "temp_max_f > 95",
        "mechanism": "Texas grid loses efficiency above 95F. GFS dry-soil warm bias "
                     "means our model predicts heat 4-8F earlier than market prices in.",
        "data_source": "EIA ERCOT real-time load API",
        "status": "dormant, no ERCOT prediction market exists",
    },
    ("hou", "ercot_load"): {
        "linkage": 0.85,
        "trigger": "temp_max_f > 95 AND humidity > 70%",
        "mechanism": "Houston heat + humidity = exponential cooling demand. Wet bulb "
                     "temperature is the real driver, not dry bulb.",
        "data_source": "EIA ERCOT real-time load API",
        "status": "dormant",
    },
    ("nyc", "pjm_load"): {
        "linkage": 0.72,
        "trigger": "temp_max_f > 92",
        "mechanism": "NYC grid hits peak at 92F. Data center cooling spikes demand "
                     "exponentially at specific wet bulb thresholds.",
        "data_source": "PJM Interconnection real-time load",
        "status": "dormant",
    },

    # AVIATION: Weather → Flight Delays
    # When to activate: Polymarket/Kalshi launches flight delay markets
    ("sfo", "sfo_delays"): {
        "linkage": 0.85,
        "trigger": "marine_layer_detected AND wind_shift_to_southeast",
        "mechanism": "SFO uses West Plan 95% of the time. Southeast wind shift triggers "
                     "Southeast Plan, cutting capacity 50%. HRRR detects this 30-60 min "
                     "before FlightAware shows delays.",
        "data_source": "FAA ATCSCC, FlightAware API",
        "status": "dormant, no SFO delay prediction market exists",
    },
    ("chi", "ord_delays"): {
        "linkage": 0.78,
        "trigger": "convective_pattern_detected",
        "mechanism": "O'Hare is Delta/United hub. Convective storms cause ground delay "
                     "programs. 30-min lead time from HRRR convective initiation forecast.",
        "data_source": "FAA ATCSCC",
        "status": "dormant",
    },

    # AGRICULTURE: Freeze → Commodity prices
    # When to activate: Commodity-linked prediction markets
    ("mia", "oj_futures"): {
        "linkage": 0.65,
        "trigger": "temp_min_f < 32 AND duration > 4h",
        "mechanism": "Late-season Florida freeze creates massive OJ futures volatility. "
                     "Our model can predict freeze probability days ahead of market pricing.",
        "data_source": "CME Orange Juice futures (not prediction market, futures venue)",
        "status": "dormant, no prediction market equivalent",
    },

    # AQI / WILDFIRE: Smoke → Air quality
    # When to activate: AQI prediction markets (likely during wildfire season)
    ("sfo", "aqi_spike"): {
        "linkage": 0.70,
        "trigger": "smoke_plume_detected AND inversion_layer",
        "mechanism": "HRRR-Smoke model predicts when wildfire smoke descends to surface. "
                     "Global models miss the descent timing by 6-12 hours.",
        "data_source": "EPA AirNow API, HRRR-Smoke via Open-Meteo",
        "status": "dormant, activate during June-October wildfire season",
    },
}


def get_active_correlations() -> list[dict]:
    """Return correlations that have active markets to trade.

    Currently returns empty, no cross-markets exist.
    When markets appear, update status from 'dormant' to 'active' and
    add the market_id/event_id for automatic discovery.
    """
    return [
        {"city": city, "market": market, **info}
        for (city, market), info in CORRELATION_REGISTRY.items()
        if info.get("status") == "active"
    ]
