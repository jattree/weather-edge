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
    fee_alpha_max: float


RISK_PROFILES: dict[str, RiskProfile] = {
    "aggressive": RiskProfile(
        name="aggressive",
        drawdown_scale_back_pct=0.25,
        drawdown_kill_pct=0.40,
        scale_back_factor=0.5,
        max_group_exposure_pct=0.30,
        max_gross_exposure_multiple=3.0,
        kelly_fraction=0.50,
        max_position_pct=0.05,
        reserve_pct=0.05,
        penny_max_position=50.0,
        min_edge=0.03,
        fee_alpha_max=0.50,
    ),
    "balanced": RiskProfile(
        name="balanced",
        drawdown_scale_back_pct=0.15,
        drawdown_kill_pct=0.25,
        scale_back_factor=0.5,
        max_group_exposure_pct=0.20,
        max_gross_exposure_multiple=2.0,
        kelly_fraction=0.25,
        max_position_pct=0.03,
        reserve_pct=0.10,
        penny_max_position=30.0,
        min_edge=0.05,
        fee_alpha_max=0.40,
    ),
    "conservative": RiskProfile(
        name="conservative",
        drawdown_scale_back_pct=0.10,
        drawdown_kill_pct=0.15,
        scale_back_factor=0.5,
        max_group_exposure_pct=0.10,
        max_gross_exposure_multiple=1.5,
        kelly_fraction=0.125,
        max_position_pct=0.015,
        reserve_pct=0.20,
        penny_max_position=15.0,
        min_edge=0.08,
        fee_alpha_max=0.30,
    ),
}


@dataclass
class CircuitBreakerState:
    """Tracks high-water mark and drawdown state."""
    high_water_mark: float = 0.0
    is_scaled_back: bool = False
    is_killed: bool = False
    kill_reason: str = ""

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
                "Drawdown %.1f%% from peak $%.0f (threshold %.0f%%)"
                % (drawdown * 100, self.high_water_mark, profile.drawdown_kill_pct * 100)
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
_active_profile_name: str = "aggressive"  # Current default, we're paper trading
_circuit_breaker = CircuitBreakerState()


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
            "CORRELATION LIMIT: %s group at $%.0f / $%.0f max (%.0f%%)"
            % (group, group_exposure, max_group, profile.max_group_exposure_pct * 100),
        )

    if size_usd > remaining:
        return (
            True,
            remaining,
            "CORRELATION TRIM: %s group $%.0f→$%.0f (cap $%.0f)"
            % (group, size_usd, remaining, max_group),
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
            "GROSS EXPOSURE CAP: $%.0f / $%.0f max (%.1fx NAV)"
            % (total_at_risk, max_exposure, profile.max_gross_exposure_multiple),
        )

    if size_usd > remaining:
        return (
            True,
            remaining,
            "GROSS EXPOSURE TRIM: $%.0f→$%.0f (cap $%.0f)"
            % (size_usd, remaining, max_exposure),
        )

    return (True, size_usd, "")
