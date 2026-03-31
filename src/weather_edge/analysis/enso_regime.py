"""ENSO regime awareness for bias correction.

The 30-day rolling bias table is calibrated against recent observations, but
ENSO regime transitions can shift weather patterns faster than 30 days.

Current state (March 2026): La Nina → Neutral transition
- La Nina (winter 2025-26): warmer South US, cooler North, wetter Pacific NW
- Neutral (spring 2026): patterns normalize
- El Nino (summer 2026): warmer everywhere, wetter South, drier Pacific NW

When the regime shifts, historical bias corrections become less reliable.
This module:
1. Fetches current ENSO state from NOAA
2. Flags cities whose bias corrections are regime-sensitive
3. Applies a shrinkage factor to bias corrections during transitions
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx

logger = logging.getLogger(__name__)

# ENSO sensitivity by city, how much does ENSO affect model bias?
# High: bias correction built during La Nina may be wrong during Neutral/El Nino
# Low: bias correction is stable across regimes
ENSO_SENSITIVITY = {
    "sea": 0.8,   # Seattle: La Nina = wetter/cooler, El Nino = drier/warmer, big swing
    "sfo": 0.7,   # SF: marine layer behavior shifts with Pacific SSTs
    "lax": 0.6,   # LA: Santa Ana frequency changes with ENSO
    "hou": 0.7,   # Houston: Gulf moisture patterns shift significantly
    "dal": 0.5,   # Dallas: moderate ENSO sensitivity via jet stream
    "mia": 0.6,   # Miami: subtropical jet position changes
    "atl": 0.4,   # Atlanta: moderate, more affected by local convection
    "den": 0.3,   # Denver: Chinook is elevation-driven, less ENSO-dependent
    "chi": 0.5,   # Chicago: jet stream position affects lake breeze patterns
    "nyc": 0.4,   # NYC: moderate coastal influence
    "aus": 0.5,   # Austin: similar to Dallas
    # International cities: lower ENSO sensitivity for temperature
    "lon": 0.2, "muc": 0.2, "war": 0.2, "mad": 0.3,
    "tyo": 0.4, "sel": 0.4, "hkg": 0.3, "sha": 0.3,
    "szn": 0.3, "bue": 0.5, "tor": 0.4,
    "wlg": 0.6,  # Wellington, strong ENSO sensitivity, Southern Hemisphere
    "lko": 0.4,  # Lucknow, moderate, monsoon onset timing shifts with ENSO
}


@dataclass
class ENSOState:
    """Current ENSO regime state."""
    phase: str  # "la_nina", "neutral", "el_nino"
    oni_value: float  # Oceanic Nino Index (negative = La Nina, positive = El Nino)
    transitioning: bool  # True if in transition between phases
    confidence: float  # 0-1, how certain is the current phase
    fetched_at: datetime


# Cache the ENSO state, it changes monthly, not per-cycle
_cached_enso: ENSOState | None = None
_cache_expiry: datetime | None = None


async def fetch_enso_state() -> ENSOState:
    """Fetch current ENSO state from NOAA CPC.

    Uses the ONI (Oceanic Nino Index) to determine phase.
    Caches for 24 hours since ENSO changes slowly.
    """
    global _cached_enso, _cache_expiry

    now = datetime.now(timezone.utc)
    if _cached_enso and _cache_expiry and now < _cache_expiry:
        return _cached_enso

    # Try NOAA's ENSO data (with retry, transient failures are common)
    from weather_edge.retry import retry_async

    async def _fetch_oni() -> float:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
                timeout=10.0,
            )
            resp.raise_for_status()
            lines = resp.text.strip().split("\n")
            last_line = lines[-1].split()
            if len(last_line) >= 4:
                return float(last_line[-1])
            return 0.0

    try:
        oni = await retry_async(
            _fetch_oni,
            attempts=3,
            base_delay=2.0,
            label="noaa_enso",
        )
        try:
            from weather_edge.analysis.service_health import record_service_call
            record_service_call("noaa_cpc", True)
        except Exception:
            logger.debug("Failed to record NOAA health", exc_info=True)
    except Exception as e:
        logger.warning(
            "NOAA ENSO fetch failed after retries: %s, using stale "
            "fallback ONI=-0.5 (March 2026). This will drift over time.",
            e,
        )
        try:
            from weather_edge.analysis.service_health import record_service_call
            record_service_call("noaa_cpc", False)
        except Exception:
            logger.debug("Failed to record NOAA failure", exc_info=True)
        # Fallback: known March 2026 state from CPC.
        # WARNING: this value will become stale. If this log line fires
        # persistently, the ONI value needs manual update.
        oni = -0.5

    # Determine phase from ONI
    if oni <= -0.5:
        phase = "la_nina"
    elif oni >= 0.5:
        phase = "el_nino"
    else:
        phase = "neutral"

    # Detect transition: ONI between -0.8 and 0.3 with recent decline = transitioning
    transitioning = -0.8 < oni < 0.3

    state = ENSOState(
        phase=phase,
        oni_value=oni,
        transitioning=transitioning,
        confidence=0.9 if not transitioning else 0.6,
        fetched_at=now,
    )

    _cached_enso = state
    _cache_expiry = (
        now.replace(hour=0, minute=0, second=0)
        + timedelta(days=1)
    )

    logger.info(
        "ENSO state: %s (ONI=%.2f, transitioning=%s)",
        state.phase, state.oni_value, state.transitioning,
    )
    return state


def get_bias_shrinkage(city_id: str, enso_state: ENSOState) -> float:
    """Calculate how much to shrink bias corrections for a city given ENSO state.

    Returns a multiplier 0-1:
    - 1.0 = use full bias correction (stable regime, low ENSO sensitivity)
    - 0.5 = use half the bias correction (transitioning regime, high sensitivity)
    - 0.0 = ignore bias correction entirely (would never do this)

    During transitions, high-sensitivity cities get their bias corrections
    reduced because the 30-day calibration window spans two different regimes.
    """
    sensitivity = ENSO_SENSITIVITY.get(city_id.lower(), 0.3)

    if not enso_state.transitioning:
        # Stable regime: full bias correction, slight reduction for high-sensitivity cities
        return max(0.7, 1.0 - sensitivity * 0.1)

    # Transitioning: reduce bias corrections proportional to city sensitivity
    # High sensitivity (0.8) → shrinkage = 0.5 (half the bias)
    # Low sensitivity (0.2) → shrinkage = 0.85 (almost full bias)
    shrinkage = 1.0 - sensitivity * 0.6
    return max(0.4, shrinkage)
