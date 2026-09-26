"""Portfolio-level risk controls, the seatbelt.

Three controls that prevent wipeout without choking returns:
1. Circuit breaker: scale back or kill trading on drawdown from peak
2. Correlation limits: cap exposure to a single weather system
3. Gross exposure cap: max total capital deployed as multiple of NAV

All thresholds configurable via risk profiles (aggressive/balanced/conservative).
Controls are invisible on good days, save your arse on bad ones.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Weather system correlation groups, cities affected by the same synoptic patterns
WEATHER_SYSTEM_GROUPS = {
    "us_northeast": ["nyc", "tor"],
    "us_southeast": ["atl", "mia"],
    "us_gulf": ["hou", "dal", "aus"],
    "us_west": ["sea", "sfo", "lax", "den"],
    "us_midwest": ["chi"],
    "uk_europe": ["lon", "mad", "muc", "war"],
    "east_asia": ["sel", "tyo", "sha", "szn", "hkg"],
    "south_asia": ["lko"],
    "southern_hemisphere": ["bue", "wlg"],
}

# Reverse lookup: city -> group
CITY_TO_GROUP: dict[str, str] = {}
for group, cities in WEATHER_SYSTEM_GROUPS.items():
    for city in cities:
        CITY_TO_GROUP[city] = group


@dataclass
class RiskProfile:
    """Configurable risk thresholds."""
    name: str  # "aggressive", "balanced", "conservative"

    # Circuit breaker
    drawdown_scale_back_pct: float  # Scale back position sizes at this drawdown
    drawdown_kill_pct: float  # Stop all trading at this drawdown
    scale_back_factor: float  # Multiply sizes by this when scaled back

    # Correlation
    max_group_exposure_pct: float  # Max % of NAV in one weather system group
    max_yes_exposure_pct: float    # Max % of NAV in ALL Yes bets combined

    # Gross exposure
    max_gross_exposure_multiple: float  # Max total at-risk as multiple of NAV

    # Position sizing
    kelly_fraction: float
    max_position_pct: float
    reserve_pct: float

    # Penny
    penny_max_position: float

    # Edge thresholds
    min_edge: float
    min_edge_yes: float
    min_edge_no: float
    fee_alpha_max: float

    # Compounding
    compound_factor: float  # 0.0=original bankroll, 0.5=half profits, 1.0=full NAV


RISK_PROFILES: dict[str, RiskProfile] = {
    "aggressive": RiskProfile(
        name="aggressive",
        drawdown_scale_back_pct=0.25,
        drawdown_kill_pct=0.40,
        scale_back_factor=0.5,
        max_group_exposure_pct=0.30,
        max_yes_exposure_pct=0.50,
        max_gross_exposure_multiple=3.0,
        kelly_fraction=0.50,
        max_position_pct=0.05,
        reserve_pct=0.05,
        penny_max_position=50.0,
        min_edge=0.03,
        min_edge_yes=0.04,
        min_edge_no=0.02,
        fee_alpha_max=0.50,
        compound_factor=1.0,
    ),
    "balanced": RiskProfile(
        name="balanced",
        drawdown_scale_back_pct=0.15,
        drawdown_kill_pct=0.25,
        scale_back_factor=0.5,
        max_group_exposure_pct=0.20,
        max_yes_exposure_pct=0.35,
        max_gross_exposure_multiple=2.0,
        kelly_fraction=0.25,
        max_position_pct=0.03,
        reserve_pct=0.10,
        penny_max_position=30.0,
        min_edge=0.05,
        min_edge_yes=0.06,
        min_edge_no=0.03,
        fee_alpha_max=0.40,
        compound_factor=0.5,
    ),
    "conservative": RiskProfile(
        name="conservative",
        drawdown_scale_back_pct=0.10,
        drawdown_kill_pct=0.15,
        scale_back_factor=0.5,
        max_group_exposure_pct=0.10,
        max_yes_exposure_pct=0.20,
        max_gross_exposure_multiple=1.5,
        kelly_fraction=0.125,
        max_position_pct=0.015,
        reserve_pct=0.20,
        penny_max_position=15.0,
        min_edge=0.08,
        min_edge_yes=0.10,
        min_edge_no=0.05,
        fee_alpha_max=0.30,
        compound_factor=0.0,
    ),
}


@dataclass
class CircuitBreakerState:
    """Tracks high-water mark and drawdown state."""
    high_water_mark: float = 0.0
    is_scaled_back: bool = False
    is_killed: bool = False
    kill_reason: str = ""
    # Live breaker only: True once the persisted high-water mark has been
    # read from Redis (see _sync_live_hwm). Until then it is never written.
    hwm_synced: bool = False

    def update(self, nav: float, profile: RiskProfile) -> None:
        """Update circuit breaker state based on current NAV."""
        if nav > self.high_water_mark:
            self.high_water_mark = nav
            # Recovery: if we were scaled back and recovered, reset
            if self.is_scaled_back and not self.is_killed:
                self.is_scaled_back = False
                logger.info(
                    "CIRCUIT BREAKER: recovered to new high $%.0f, resuming full size",
                    nav,
                )

        if self.high_water_mark <= 0:
            return

        drawdown = (self.high_water_mark - nav) / self.high_water_mark

        # Kill switch
        if drawdown >= profile.drawdown_kill_pct and not self.is_killed:
            self.is_killed = True
            self.kill_reason = (
                f"Drawdown {drawdown * 100:.1f}% from peak ${self.high_water_mark:.0f} "
                f"(threshold {profile.drawdown_kill_pct * 100:.0f}%)"
            )
            logger.warning("CIRCUIT BREAKER KILL: %s", self.kill_reason)

        # Scale back
        elif drawdown >= profile.drawdown_scale_back_pct and not self.is_scaled_back:
            self.is_scaled_back = True
            logger.warning(
                "CIRCUIT BREAKER SCALE-BACK: drawdown %.1f%% from peak $%.0f, halving positions",
                drawdown * 100, self.high_water_mark,
            )

    def get_size_multiplier(self, profile: RiskProfile) -> float:
        """Return position size multiplier (1.0 = normal, 0.5 = scaled back, 0.0 = killed)."""
        if self.is_killed:
            return 0.0
        if self.is_scaled_back:
            return profile.scale_back_factor
        return 1.0


# Module-level state
# Default is the moderate "balanced" profile (25% drawdown kill, 2x NAV gross
# cap). "aggressive" (40% kill, 3x NAV) must be chosen explicitly.
DEFAULT_PROFILE_NAME = "balanced"
_active_profile_name: str = DEFAULT_PROFILE_NAME
# Paper circuit breaker: tracks PAPER NAV and gates paper trades only.
_circuit_breaker = CircuitBreakerState()
# Live circuit breaker: tracks the real exchange NAV. Kept separate from the
# paper breaker so paper NAV never pollutes the live high-water mark. When it
# trips it activates the persistent kill switch (see update_live_circuit_breaker).
_live_circuit_breaker = CircuitBreakerState()
_LIVE_HWM_KEY = "circuit_breaker:live_hwm"


def get_active_profile() -> RiskProfile:
    """Return the currently active risk profile."""
    return RISK_PROFILES[_active_profile_name]


def set_active_profile(name: str) -> None:
    """Change the active risk profile."""
    global _active_profile_name
    if name not in RISK_PROFILES:
        raise ValueError(f"Unknown profile: {name}")
    _active_profile_name = name
    logger.info("Risk profile set to: %s", name)


def _is_yes(t) -> bool:
    side = str(getattr(t, "side", "") or getattr(t, "outcome", "") or "")
    return side.strip().upper() == "YES"


def _sync_live_hwm(cb: CircuitBreakerState) -> bool:
    """Merge the persisted high-water mark into ``cb`` once Redis is readable.

    Until it has been read, the HWM lives in memory only and is NOT written
    back: writing it after a restart during a Redis outage would replace the
    real peak with a lower one and hide the drawdown. Retried every call.
    """
    if cb.hwm_synced:
        return True
    try:
        from weather_edge.live_state import get_value_persisted
        stored = get_value_persisted(_LIVE_HWM_KEY)
    except Exception as e:  # noqa: BLE001 - Redis down or erroring: retry later
        logger.warning("Live HWM not loaded yet (%s); keeping it in memory", e)
        return False
    if stored:
        try:
            cb.high_water_mark = max(cb.high_water_mark, float(stored))
        except ValueError:
            logger.warning("Live HWM in Redis unreadable: %r", stored)
    cb.hwm_synced = True
    return True


def update_live_circuit_breaker(nav: float) -> float:
    """Feed the live exchange NAV to the live circuit breaker.

    Must only be called with a NAV that was actually observed from the
    exchange (not a fallback). The high-water mark is persisted in live state
    so a restart does not reset the drawdown reference. When the drawdown
    crosses the profile's kill threshold, the persistent kill switch is
    activated, which blocks every subsequent live order until an operator
    resets it.

    Returns:
        Position size multiplier for live orders (1.0, scale-back factor, or 0.0).
    """
    profile = get_active_profile()
    cb = _live_circuit_breaker
    synced = _sync_live_hwm(cb)

    was_killed = cb.is_killed
    if nav <= 0:
        # A zero/negative NAV is a data problem, not a 100% drawdown signal
        logger.warning("Live circuit breaker: ignoring non-positive NAV %.2f", nav)
        return cb.get_size_multiplier(profile)
    cb.update(nav, profile)
    if synced:
        try:
            from weather_edge.live_state import set_value
            set_value(_LIVE_HWM_KEY, str(cb.high_water_mark))
        except Exception:  # noqa: BLE001 - best-effort persistence
            logger.warning("Live HWM save failed", exc_info=True)

    if cb.is_killed and not was_killed:
        from weather_edge.trading.kill_switch import activate_kill_switch
        activate_kill_switch(
            f"Live circuit breaker: {cb.kill_reason}", triggered_by="circuit_breaker",
        )
    return cb.get_size_multiplier(profile)


def reset_live_circuit_breaker() -> None:
    """Operator reset: clear the trip and restart the high-water mark.

    The next observed NAV becomes the new high-water mark. Called when the
    kill switch is explicitly deactivated, otherwise a tripped breaker would
    keep blocking orders (or immediately re-trip from the stale HWM).
    """
    global _live_circuit_breaker
    # The operator chose a fresh reference: never re-merge the old peak.
    _live_circuit_breaker = CircuitBreakerState(hwm_synced=True)
    try:
        from weather_edge.live_state import set_value
        set_value(_LIVE_HWM_KEY, "0")
    except Exception:
        logger.debug("Live HWM reset failed", exc_info=True)


def live_circuit_breaker_multiplier() -> float:
    """Current live size multiplier without updating NAV."""
    return _live_circuit_breaker.get_size_multiplier(get_active_profile())


def check_correlation_limit(
    city_id: str,
    size_usd: float,
    open_trades: list,
    nav: float,
    profile: RiskProfile,
) -> tuple[bool, float, str]:
    """Check if adding a trade would breach the correlation group limit.

    Returns:
        (allowed, max_allowed_size, reason)
    """
    group = CITY_TO_GROUP.get(city_id.lower())
    if not group:
        return (True, size_usd, "")

    # Sum exposure in the same weather system group
    group_cities = set(WEATHER_SYSTEM_GROUPS.get(group, []))
    group_exposure = sum(
        t.size_usd for t in open_trades
        if getattr(t, "city_id", "").lower() in group_cities
        and getattr(t, "status", "") == "open"
    )

    max_group = nav * profile.max_group_exposure_pct
    remaining = max_group - group_exposure

    if remaining <= 0:
        return (
            False,
            0.0,
            f"CORRELATION LIMIT: {group} group at ${group_exposure:.0f} / "
            f"${max_group:.0f} max ({profile.max_group_exposure_pct * 100:.0f}%)",
        )

    if size_usd > remaining:
        return (
            True,
            remaining,
            f"CORRELATION TRIM: {group} group ${size_usd:.0f}→${remaining:.0f} (cap "
            f"${max_group:.0f})",
        )

    return (True, size_usd, "")


def check_yes_exposure_limit(
    size_usd: float,
    open_trades: list,
    nav: float,
    profile: RiskProfile,
) -> tuple[bool, float, str]:
    """Check if adding a YES trade would breach the total YES exposure cap.

    Returns:
        (allowed, max_allowed_size, reason)
    """
    # Sum exposure in all YES positions
    yes_exposure = sum(
        t.size_usd for t in open_trades
        if _is_yes(t)
        and getattr(t, "status", "") == "open"
    )

    max_yes = nav * profile.max_yes_exposure_pct
    remaining = max_yes - yes_exposure

    if remaining <= 0:
        return (
            False,
            0.0,
            f"YES EXPOSURE CAP: ${yes_exposure:.0f} / ${max_yes:.0f} max "
            f"({profile.max_yes_exposure_pct * 100:.0f}% NAV)",
        )

    if size_usd > remaining:
        return (
            True,
            remaining,
            f"YES EXPOSURE TRIM: ${size_usd:.0f}→${remaining:.0f} (cap ${max_yes:.0f})",
        )

    return (True, size_usd, "")


def check_gross_exposure(
    size_usd: float,
    total_at_risk: float,
    nav: float,
    profile: RiskProfile,
) -> tuple[bool, float, str]:
    """Check if adding a trade would breach the gross exposure cap.

    Returns:
        (allowed, max_allowed_size, reason)
    """
    max_exposure = nav * profile.max_gross_exposure_multiple
    remaining = max_exposure - total_at_risk

    if remaining <= 0:
        return (
            False,
            0.0,
            f"GROSS EXPOSURE CAP: ${total_at_risk:.0f} / ${max_exposure:.0f} max "
            f"({profile.max_gross_exposure_multiple:.1f}x NAV)",
        )

    if size_usd > remaining:
        return (
            True,
            remaining,
            f"GROSS EXPOSURE TRIM: ${size_usd:.0f}→${remaining:.0f} (cap ${max_exposure:.0f})",
        )

    return (True, size_usd, "")
