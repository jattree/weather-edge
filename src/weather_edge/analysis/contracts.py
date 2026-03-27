"""Contract validation functions for silent failure detection.

Pure functions, no side effects, no API calls, no file I/O.
Each returns a ContractResult indicating whether a business invariant holds.

These catch the 8 dangerous silent failures that would cost real money:
1. EMOS calibration disabled (edges go from 5% to 86%)
2. Pool budget exceeded (betting more than the bankroll)
3. Reserve pot breached (no capital left for high-conviction trades)
4. Penny bets flagged for exit (selling a $0.03 token locks in loss)
5. AI keys missing when reasoning expected (trading blind)
6. Insufficient models for consensus (garbage in, garbage out)
7. Spread profit calculated from midpoints instead of asks (phantom edge)
8. Taker fee eats too much of projected alpha (fees > 40% of edge)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContractResult:
    """Result of a contract validation check."""
    valid: bool
    error: str = ""
    code: str = ""  # e.g. "EMOS_DISABLED", "BUDGET_EXCEEDED"


def validate_emos_active(
    spread_inflation: float,
    bucket_cap: float,
    variance_floor: float,
) -> ContractResult:
    """Verify EMOS calibration is applied.

    If EMOS is off (spread_inflation=1.0, bucket_cap=1.0, variance_floor=0),
    raw ensemble spread is trusted directly. Models are correlated, so raw
    spread underestimates uncertainty by ~2x. This turns 5% edges into 86%
    apparent edges, pure hallucination.

    Args:
        spread_inflation: Must be >1.0 (typically 2.0). Inflates raw std_dev.
        bucket_cap: Must be <1.0 (typically 0.70). Caps single-bucket probability.
        variance_floor: Must be >0 (typically 1.2°C). Minimum uncertainty.
    """
    problems = []

    if spread_inflation <= 1.0:
        problems.append(f"spread_inflation={spread_inflation} (must be >1.0)")
    if bucket_cap >= 1.0:
        problems.append(f"bucket_cap={bucket_cap} (must be <1.0)")
    if variance_floor <= 0:
        problems.append(f"variance_floor={variance_floor} (must be >0)")

    if problems:
        return ContractResult(
            valid=False,
            error=f"EMOS calibration disabled: {'; '.join(problems)}",
            code="EMOS_DISABLED",
        )
    return ContractResult(valid=True)


def validate_pool_budget(
    capital_at_risk: float,
    bankroll: float,
) -> ContractResult:
    """Verify total capital at risk doesn't exceed bankroll.

    This is the absolute ceiling, no trade should push total exposure
    past what we actually have. Pool-level limits are enforced elsewhere;
    this catches the case where pool math has a bug.

    Args:
        capital_at_risk: Sum of size_usd across all open positions.
        bankroll: Total capital available for trading.
    """
    if capital_at_risk > bankroll:
        return ContractResult(
            valid=False,
            error=f"Capital at risk ${capital_at_risk:.2f} exceeds bankroll ${bankroll:.2f}",
            code="BUDGET_EXCEEDED",
        )
    return ContractResult(valid=True)


def validate_reserve_pot(
    available_capital: float,
    bankroll: float,
    reserve_pct: float = 0.10,
    signal_tier: str = "low",
) -> ContractResult:
    """Verify available capital stays above reserve unless signal is HIGH.

    Reserve exists so we always have dry powder for high-conviction
    opportunities. LOW/MEDIUM signals should not drain the last 10%.
    HIGH tier signals can dip into reserve, that's the whole point.

    Args:
        available_capital: Bankroll minus capital at risk plus realized P&L.
        bankroll: Total bankroll.
        reserve_pct: Fraction of bankroll kept as reserve (default 10%).
        signal_tier: "high", "medium", or "low".
    """
    if signal_tier.lower() == "high":
        return ContractResult(valid=True)

    reserve = bankroll * reserve_pct
    if available_capital < reserve:
        return ContractResult(
            valid=False,
            error=(
                f"Available capital ${available_capital:.2f} below reserve "
                f"${reserve:.2f} ({reserve_pct:.0%} of ${bankroll:.2f}) "
                f"for {signal_tier} tier signal"
            ),
            code="RESERVE_BREACHED",
        )
    return ContractResult(valid=True)


def validate_penny_no_exit(
    entry_price: float,
    penny_threshold: float = 0.06,
) -> ContractResult:
    """Verify penny bets are never flagged for early exit.

    Selling a $0.03 token at $0.01 locks in a 67% loss. Holding gives
    a shot at the $1.00 payout. The downside is already capped at the
    entry cost. This contract fires when something tries to exit a
    position that should be held to resolution.

    Returns INVALID when entry_price <= penny_threshold (meaning: this IS
    a penny bet and should NOT be exited).

    Args:
        entry_price: Price at which we entered the position.
        penny_threshold: Maximum entry price to qualify as penny bet (default $0.06).
    """
    if entry_price <= penny_threshold:
        return ContractResult(
            valid=False,
            error=f"Penny bet (entry=${entry_price:.2f}) must not be exited early",
            code="PENNY_NO_EXIT",
        )
    return ContractResult(valid=True)


def validate_ai_keys_present(
    anthropic_key: str,
    gemini_key: str,
) -> ContractResult:
    """Verify AI API keys are non-empty when reasoning is expected.

    Without keys, the Claude reasoning layer and Gemini red team are
    silently skipped. Trades go through without any AI review, which
    defeats the purpose of the multi-model validation pipeline.

    Args:
        anthropic_key: Anthropic API key string.
        gemini_key: Gemini API key string.
    """
    missing = []
    if not anthropic_key or not anthropic_key.strip():
        missing.append("ANTHROPIC_API_KEY")
    if not gemini_key or not gemini_key.strip():
        missing.append("GEMINI_API_KEY")

    if missing:
        return ContractResult(
            valid=False,
            error=f"AI keys missing: {', '.join(missing)}",
            code="AI_KEY_MISSING",
        )
    return ContractResult(valid=True)


def validate_model_count(
    model_count: int,
    city_has_regional_models: bool,
    min_models: int | None = None,
) -> ContractResult:
    """Verify consensus is based on sufficient models.

    US cities with HRRR/NAM should have 6 global + 2 regional = 8 models.
    Requiring >= 4 catches cases where half the models failed to respond.
    International cities without regional models need >= 3.

    Args:
        model_count: Number of models that returned data for this city/date.
        city_has_regional_models: Whether this city has regional models available.
        min_models: Override minimum (default: 4 for regional, 3 for international).
    """
    if min_models is None:
        min_models = 4 if city_has_regional_models else 3

    if model_count < min_models:
        return ContractResult(
            valid=False,
            error=(
                f"Only {model_count} models available (need >= {min_models}). "
                f"Consensus unreliable."
            ),
            code="INSUFFICIENT_MODELS",
        )
    return ContractResult(valid=True)


def validate_spread_uses_asks(
    yes_price_source: str,
    no_price_source: str,
) -> ContractResult:
    """Verify spread capture profit uses ask prices, not midpoints.

    Spread only exists if YES_ask + NO_ask < $1.00. If we compute spread
    from midpoints, we see phantom profit that evaporates when we actually
    try to buy at the ask. This is the difference between a guaranteed
    arbitrage and a losing trade.

    Args:
        yes_price_source: Must be "ask" (not "mid").
        no_price_source: Must be "ask" (not "mid").
    """
    problems = []
    if yes_price_source != "ask":
        problems.append(f"YES price source is '{yes_price_source}' not 'ask'")
    if no_price_source != "ask":
        problems.append(f"NO price source is '{no_price_source}' not 'ask'")

    if problems:
        return ContractResult(
            valid=False,
            error=f"Spread calculated from midpoints: {'; '.join(problems)}",
            code="SPREAD_MIDPOINT_RISK",
        )
    return ContractResult(valid=True)


def validate_fee_alpha_ratio(
    edge: float,
    price: float,
    size_usd: float,
    max_fee_pct: float = 0.40,
    peak_rate: float = 0.0125,
) -> ContractResult:
    """Verify taker fee doesn't eat >40% of projected alpha.

    Dynamic taker fee = peak_rate * 4 * P * (1-P) * size_usd.
    Projected alpha = edge * size_usd.
    If fee / alpha > max_fee_pct, the trade's not worth taking.

    This catches trades where fees erode most of the edge, especially
    dangerous near 50% probability where fees peak at 1.25%.

    Args:
        edge: Raw edge in probability units (e.g. 0.05 = 5%).
        price: Market probability / price (0-1).
        size_usd: Trade notional in USD.
        max_fee_pct: Maximum acceptable fee as fraction of alpha (default 40%).
        peak_rate: Taker fee peak rate (default 1.25%).
    """
    if edge <= 0 or size_usd <= 0:
        return ContractResult(
            valid=False,
            error="No positive edge or size, trade has no alpha",
            code="FEE_EATS_ALPHA",
        )

    fee = peak_rate * 4.0 * price * (1.0 - price) * size_usd
    projected_alpha = edge * size_usd
    fee_ratio = fee / projected_alpha if projected_alpha > 0 else float("inf")

    if fee_ratio > max_fee_pct:
        return ContractResult(
            valid=False,
            error=(
                f"Taker fee ${fee:.2f} eats {fee_ratio:.0%} of "
                f"projected alpha ${projected_alpha:.2f} "
                f"(max allowed: {max_fee_pct:.0%})"
            ),
            code="FEE_EATS_ALPHA",
        )
    return ContractResult(valid=True)
