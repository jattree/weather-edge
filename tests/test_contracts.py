"""Exhaustive tests for contract validation functions.

Each test justifies its existence: these are the 7 silent failures
that would cost real money if they broke in production.
"""
from __future__ import annotations

import pytest

from weather_edge.analysis.contracts import (
    ContractResult,
    validate_ai_keys_present,
    validate_emos_active,
    validate_fee_alpha_ratio,
    validate_leverage_cap,
    validate_model_count,
    validate_penny_no_exit,
    validate_pool_budget,
    validate_reserve_pot,
    validate_spread_uses_asks,
)
from weather_edge.trading.fees import (
    calculate_maker_rebate,
    calculate_taker_fee,
    fee_eats_alpha,
    net_cost_after_fees,
)

# ---------------------------------------------------------------------------
# validate_emos_active
# Without EMOS, raw ensemble spread is trusted. Models share physics,
# so spread is artificially narrow. A 5% real edge becomes 86% apparent.
# ---------------------------------------------------------------------------

class TestValidateEmosActive:

    def test_normal_emos_params(self):
        """Production defaults: inflation=2.0, cap=0.70, floor=1.2."""
        result = validate_emos_active(2.0, 0.70, 1.2)
        assert result.valid is True
        assert result.error == ""

    def test_spread_inflation_disabled(self):
        """inflation=1.0 means raw std_dev used directly, underestimates uncertainty."""
        result = validate_emos_active(1.0, 0.70, 1.2)
        assert result.valid is False
        assert result.code == "EMOS_DISABLED"
        assert "spread_inflation" in result.error

    def test_bucket_cap_disabled(self):
        """cap=1.0 allows 100% on a single 2F bucket, 'likely broken' per Gemini."""
        result = validate_emos_active(2.0, 1.0, 1.2)
        assert result.valid is False
        assert result.code == "EMOS_DISABLED"
        assert "bucket_cap" in result.error

    def test_variance_floor_disabled(self):
        """floor=0 means perfect model agreement = zero uncertainty, impossible."""
        result = validate_emos_active(2.0, 0.70, 0)
        assert result.valid is False
        assert result.code == "EMOS_DISABLED"
        assert "variance_floor" in result.error

    def test_all_disabled(self):
        """All three off at once, total calibration failure."""
        result = validate_emos_active(1.0, 1.0, 0)
        assert result.valid is False
        assert result.code == "EMOS_DISABLED"
        # All three problems should be mentioned
        assert "spread_inflation" in result.error
        assert "bucket_cap" in result.error
        assert "variance_floor" in result.error

    def test_barely_valid_inflation(self):
        """inflation=1.01 is technically valid but marginal."""
        result = validate_emos_active(1.01, 0.70, 1.2)
        assert result.valid is True

    def test_negative_variance_floor(self):
        """Negative variance floor is even worse than zero."""
        result = validate_emos_active(2.0, 0.70, -0.5)
        assert result.valid is False
        assert result.code == "EMOS_DISABLED"


# ---------------------------------------------------------------------------
# validate_pool_budget
# Capital at risk must never exceed bankroll. Period.
# ---------------------------------------------------------------------------

class TestValidatePoolBudget:

    def test_under_budget(self):
        """$1000 risk on $2000 bankroll, plenty of room."""
        result = validate_pool_budget(1000.0, 2000.0)
        assert result.valid is True

    def test_over_budget(self):
        """$2100 risk on $2000, this should never happen."""
        result = validate_pool_budget(2100.0, 2000.0)
        assert result.valid is False
        assert result.code == "BUDGET_EXCEEDED"
        assert "2100" in result.error
        assert "2000" in result.error

    def test_exact_match(self):
        """$2000 risk on $2000, fully deployed but not over."""
        result = validate_pool_budget(2000.0, 2000.0)
        assert result.valid is True

    def test_zero_risk(self):
        """No open positions, always valid."""
        result = validate_pool_budget(0.0, 2000.0)
        assert result.valid is True

    def test_zero_bankroll_with_risk(self):
        """Edge case: zero bankroll but somehow have risk."""
        result = validate_pool_budget(1.0, 0.0)
        assert result.valid is False
        assert result.code == "BUDGET_EXCEEDED"

    def test_floating_point_just_over(self):
        """Floating point edge: slightly over due to rounding."""
        result = validate_pool_budget(2000.01, 2000.0)
        assert result.valid is False
        assert result.code == "BUDGET_EXCEEDED"


# ---------------------------------------------------------------------------
# validate_reserve_pot
# Keep 10% of bankroll uncommitted unless signal is HIGH tier.
# HIGH tier signals can dip into reserve, that's the whole point.
# ---------------------------------------------------------------------------

class TestValidateReservePot:

    def test_above_reserve(self):
        """$500 available on $2000 bankroll (reserve=$200), fine."""
        result = validate_reserve_pot(500.0, 2000.0, 0.10, "low")
        assert result.valid is True

    def test_below_reserve_low_tier(self):
        """$150 available, LOW tier, reserve breached, block the trade."""
        result = validate_reserve_pot(150.0, 2000.0, 0.10, "low")
        assert result.valid is False
        assert result.code == "RESERVE_BREACHED"
        assert "150" in result.error

    def test_below_reserve_high_tier(self):
        """$150 available, HIGH tier, reserve waived, trade allowed."""
        result = validate_reserve_pot(150.0, 2000.0, 0.10, "high")
        assert result.valid is True

    def test_zero_available_high_tier(self):
        """$0 available, HIGH tier, still valid (reserve waived)."""
        result = validate_reserve_pot(0.0, 2000.0, 0.10, "high")
        assert result.valid is True

    def test_below_reserve_medium_tier(self):
        """MEDIUM tier follows same rules as LOW, blocked."""
        result = validate_reserve_pot(150.0, 2000.0, 0.10, "medium")
        assert result.valid is False
        assert result.code == "RESERVE_BREACHED"

    def test_exactly_at_reserve(self):
        """$200 available = exactly 10% of $2000, should pass."""
        result = validate_reserve_pot(200.0, 2000.0, 0.10, "low")
        assert result.valid is True

    def test_case_insensitive_tier(self):
        """HIGH in any case should waive reserve."""
        result = validate_reserve_pot(0.0, 2000.0, 0.10, "HIGH")
        assert result.valid is True


# ---------------------------------------------------------------------------
# validate_penny_no_exit
# Penny bets (entry <= $0.06) must NEVER be exited early.
# Selling a $0.03 token at $0.01 locks in 67% loss.
# Holding to resolution: downside capped at entry, upside is $1.00.
# ---------------------------------------------------------------------------

class TestValidatePennyNoExit:

    def test_penny_bet_blocked(self):
        """$0.02 entry, this is a penny bet, must not exit."""
        result = validate_penny_no_exit(0.02)
        assert result.valid is False
        assert result.code == "PENNY_NO_EXIT"

    def test_boundary_penny(self):
        """$0.06 entry, exactly at threshold, still a penny bet."""
        result = validate_penny_no_exit(0.06)
        assert result.valid is False
        assert result.code == "PENNY_NO_EXIT"

    def test_just_above_penny(self):
        """$0.07 entry, not a penny bet, exit is allowed."""
        result = validate_penny_no_exit(0.07)
        assert result.valid is True

    def test_normal_position(self):
        """$0.50 entry, standard position, exit allowed."""
        result = validate_penny_no_exit(0.50)
        assert result.valid is True

    def test_custom_threshold(self):
        """Custom threshold of $0.10."""
        result = validate_penny_no_exit(0.08, penny_threshold=0.10)
        assert result.valid is False
        assert result.code == "PENNY_NO_EXIT"

    def test_zero_entry(self):
        """$0.00 entry, definitely a penny bet."""
        result = validate_penny_no_exit(0.00)
        assert result.valid is False
        assert result.code == "PENNY_NO_EXIT"


# ---------------------------------------------------------------------------
# validate_ai_keys_present
# Without keys, Claude and Gemini are silently skipped.
# Trades go through without AI review, trading blind.
# ---------------------------------------------------------------------------

class TestValidateAiKeysPresent:

    def test_both_keys_set(self):
        """Normal operation: both keys present."""
        result = validate_ai_keys_present("sk-ant-xxx", "AIza-xxx")
        assert result.valid is True

    def test_anthropic_empty(self):
        """Anthropic key missing, Claude reasoning skipped."""
        result = validate_ai_keys_present("", "AIza-xxx")
        assert result.valid is False
        assert result.code == "AI_KEY_MISSING"
        assert "ANTHROPIC" in result.error

    def test_gemini_empty(self):
        """Gemini key missing, red team skipped."""
        result = validate_ai_keys_present("sk-ant-xxx", "")
        assert result.valid is False
        assert result.code == "AI_KEY_MISSING"
        assert "GEMINI" in result.error

    def test_both_empty(self):
        """Both keys missing, completely blind trading."""
        result = validate_ai_keys_present("", "")
        assert result.valid is False
        assert result.code == "AI_KEY_MISSING"
        assert "ANTHROPIC" in result.error
        assert "GEMINI" in result.error

    def test_whitespace_only_key(self):
        """Key that's just spaces, effectively empty."""
        result = validate_ai_keys_present("   ", "AIza-xxx")
        assert result.valid is False
        assert result.code == "AI_KEY_MISSING"

    def test_none_like_empty(self):
        """Empty string is the sentinel for 'not configured'."""
        result = validate_ai_keys_present("", "")
        assert result.valid is False


# ---------------------------------------------------------------------------
# validate_model_count
# Consensus from too few models is unreliable. US cities should have
# 6 global + 2 regional = 8 models. Require >=4 for US, >=3 international.
# ---------------------------------------------------------------------------

class TestValidateModelCount:

    def test_us_city_full_models(self):
        """8 models for US city, all models responding."""
        result = validate_model_count(8, city_has_regional_models=True)
        assert result.valid is True

    def test_us_city_insufficient(self):
        """2 models for US city, most models failed."""
        result = validate_model_count(2, city_has_regional_models=True)
        assert result.valid is False
        assert result.code == "INSUFFICIENT_MODELS"
        assert "2" in result.error
        assert "4" in result.error

    def test_international_sufficient(self):
        """3 models for international city, acceptable minimum."""
        result = validate_model_count(3, city_has_regional_models=False)
        assert result.valid is True

    def test_international_insufficient(self):
        """2 models for international city, below minimum."""
        result = validate_model_count(2, city_has_regional_models=False)
        assert result.valid is False
        assert result.code == "INSUFFICIENT_MODELS"

    def test_zero_models(self):
        """No models at all, should always fail."""
        result = validate_model_count(0, city_has_regional_models=True)
        assert result.valid is False
        assert result.code == "INSUFFICIENT_MODELS"

    def test_zero_models_international(self):
        """No models for international city."""
        result = validate_model_count(0, city_has_regional_models=False)
        assert result.valid is False
        assert result.code == "INSUFFICIENT_MODELS"

    def test_exactly_at_us_minimum(self):
        """4 models for US city, exactly at the threshold."""
        result = validate_model_count(4, city_has_regional_models=True)
        assert result.valid is True

    def test_custom_minimum(self):
        """Override minimum model count."""
        result = validate_model_count(5, city_has_regional_models=True, min_models=6)
        assert result.valid is False
        assert result.code == "INSUFFICIENT_MODELS"


# ---------------------------------------------------------------------------
# validate_spread_uses_asks
# Spread = YES_ask + NO_ask. If we use midpoints, we see phantom profit.
# The ask is what we actually pay to buy; midpoint is what the book shows.
# ---------------------------------------------------------------------------

class TestValidateSpreadUsesAsks:

    def test_both_ask(self):
        """Correct: spread calculated from real ask prices."""
        result = validate_spread_uses_asks("ask", "ask")
        assert result.valid is True

    def test_yes_midpoint(self):
        """YES price from midpoint, phantom profit risk."""
        result = validate_spread_uses_asks("mid", "ask")
        assert result.valid is False
        assert result.code == "SPREAD_MIDPOINT_RISK"
        assert "YES" in result.error

    def test_both_midpoint(self):
        """Both from midpoints, completely unreliable spread calc."""
        result = validate_spread_uses_asks("mid", "mid")
        assert result.valid is False
        assert result.code == "SPREAD_MIDPOINT_RISK"
        assert "YES" in result.error
        assert "NO" in result.error

    def test_no_midpoint(self):
        """NO price from midpoint."""
        result = validate_spread_uses_asks("ask", "mid")
        assert result.valid is False
        assert result.code == "SPREAD_MIDPOINT_RISK"
        assert "NO" in result.error


# ---------------------------------------------------------------------------
# ContractResult dataclass
# ---------------------------------------------------------------------------

class TestContractResult:

    def test_valid_result_defaults(self):
        """Valid result has empty error and code by default."""
        r = ContractResult(valid=True)
        assert r.valid is True
        assert r.error == ""
        assert r.code == ""

    def test_invalid_result(self):
        """Invalid result carries error message and code."""
        r = ContractResult(valid=False, error="broken", code="BROKEN")
        assert r.valid is False
        assert r.error == "broken"
        assert r.code == "BROKEN"

    def test_frozen(self):
        """ContractResult is immutable, can't accidentally mutate."""
        r = ContractResult(valid=True)
        with pytest.raises(AttributeError):
            r.valid = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Taker fee calculations (fees.py)
# Dynamic formula: fee = 0.0125 * 4 * P * (1-P) * trade_value
# Peak at 50%, negligible at extremes. Makers pay $0.
# ---------------------------------------------------------------------------

class TestCalculateTakerFee:

    def test_fee_at_50_pct(self):
        """At 50% probability: fee = 0.0125 * 4 * 0.5 * 0.5 * 100 = $1.25."""
        fee = calculate_taker_fee(0.50, 100.0)
        assert fee == pytest.approx(1.25, abs=0.001)

    def test_fee_at_95_pct(self):
        """At 95% probability: fee ~0.24% of trade value."""
        fee = calculate_taker_fee(0.95, 100.0)
        # 0.0125 * 4 * 0.95 * 0.05 * 100 = 0.2375
        assert fee == pytest.approx(0.2375, abs=0.001)

    def test_fee_at_5_pct(self):
        """At 5% probability: fee ~0.24% (symmetric with 95%)."""
        fee = calculate_taker_fee(0.05, 100.0)
        assert fee == pytest.approx(0.2375, abs=0.001)

    def test_fee_at_99_pct(self):
        """At 99% probability: fee ~0.05% (negligible)."""
        fee = calculate_taker_fee(0.99, 100.0)
        # 0.0125 * 4 * 0.99 * 0.01 * 100 = 0.0495
        assert fee == pytest.approx(0.0495, abs=0.001)

    def test_fee_at_1_pct(self):
        """At 1% probability: symmetric with 99%."""
        fee = calculate_taker_fee(0.01, 100.0)
        assert fee == pytest.approx(0.0495, abs=0.001)

    def test_fee_scales_with_size(self):
        """Fee scales linearly with trade size."""
        fee_100 = calculate_taker_fee(0.50, 100.0)
        fee_200 = calculate_taker_fee(0.50, 200.0)
        assert fee_200 == pytest.approx(fee_100 * 2, abs=0.001)

    def test_fee_zero_size(self):
        """Zero trade size = zero fee."""
        assert calculate_taker_fee(0.50, 0.0) == 0.0

    def test_fee_symmetric(self):
        """Fee is symmetric around 50%: f(P) == f(1-P)."""
        assert calculate_taker_fee(0.30, 100.0) == pytest.approx(
            calculate_taker_fee(0.70, 100.0), abs=0.0001
        )


class TestMakerRebate:

    def test_25_pct_rebate(self):
        """25% of $1.25 taker fee = $0.3125 rebate."""
        rebate = calculate_maker_rebate(1.25)
        assert rebate == pytest.approx(0.3125, abs=0.001)

    def test_zero_fee_zero_rebate(self):
        """No taker fee = no rebate."""
        assert calculate_maker_rebate(0.0) == 0.0


class TestNetCostAfterFees:

    def test_maker_pays_zero_fees(self):
        """Makers (limit orders that don't cross spread) pay $0 in fees."""
        cost = net_cost_after_fees(0.50, 100.0, is_maker=True)
        assert cost == 100.0  # Exactly the trade value, no fee

    def test_taker_pays_fee(self):
        """Taker at 50%: $100 + $1.25 fee = $101.25."""
        cost = net_cost_after_fees(0.50, 100.0, is_maker=False)
        assert cost == pytest.approx(101.25, abs=0.01)


class TestFeeEatsAlpha:

    def test_large_edge_small_fee(self):
        """10% edge at 95% price: fee is tiny vs alpha. Trade is good."""
        assert fee_eats_alpha(0.10, 0.95, 50.0) is False

    def test_small_edge_high_fee(self):
        """1% edge at 50% price: fee is 1.25% > 40% of 1%. Skip."""
        assert fee_eats_alpha(0.01, 0.50, 100.0) is True

    def test_zero_edge(self):
        """Zero edge = no alpha to protect. Always skip."""
        assert fee_eats_alpha(0.0, 0.50, 100.0) is True

    def test_negative_edge(self):
        """Negative edge = losing trade. Always skip."""
        assert fee_eats_alpha(-0.05, 0.50, 100.0) is True


# ---------------------------------------------------------------------------
# validate_fee_alpha_ratio (contract #8)
# Ensures taker fee doesn't eat >40% of projected alpha before trade.
# ---------------------------------------------------------------------------

class TestValidateFeeAlphaRatio:

    def test_healthy_edge(self):
        """10% edge at 90% price: fee is small vs alpha."""
        result = validate_fee_alpha_ratio(0.10, 0.90, 50.0)
        assert result.valid is True

    def test_fee_kills_trade(self):
        """1% edge at 50% price: 1.25% fee >> 40% of 1% alpha."""
        result = validate_fee_alpha_ratio(0.01, 0.50, 100.0)
        assert result.valid is False
        assert result.code == "FEE_EATS_ALPHA"
        assert "fee" in result.error.lower() or "Fee" in result.error

    def test_penny_price_low_fee(self):
        """5% edge at 3% price: fee ~$0.006 on $50, alpha ~$2.50. Fine."""
        result = validate_fee_alpha_ratio(0.05, 0.03, 50.0)
        assert result.valid is True

    def test_zero_edge_invalid(self):
        """No edge = no alpha. Always invalid."""
        result = validate_fee_alpha_ratio(0.0, 0.50, 100.0)
        assert result.valid is False
        assert result.code == "FEE_EATS_ALPHA"

    def test_zero_size_invalid(self):
        """Zero size = nothing to trade."""
        result = validate_fee_alpha_ratio(0.05, 0.50, 0.0)
        assert result.valid is False
        assert result.code == "FEE_EATS_ALPHA"

    def test_custom_max_fee_pct(self):
        """With stricter 20% threshold, more trades get blocked."""
        # At 50% price, fee rate is 1.25%. With 5% edge:
        # fee = $1.25, alpha = $5.00, ratio = 25%
        # At 50% threshold: valid. At 20% threshold: invalid.
        result_50 = validate_fee_alpha_ratio(0.05, 0.50, 100.0, max_fee_pct=0.50)
        result_20 = validate_fee_alpha_ratio(0.05, 0.50, 100.0, max_fee_pct=0.20)
        assert result_50.valid is True
        assert result_20.valid is False


# ---------------------------------------------------------------------------
# validate_leverage_cap
# The Dallas incident: NO at market_prob=0.9885, effective 1.15¢, $72 → 6,261
# shares (87x leverage). Bad API quotes create absurd positions.
# ---------------------------------------------------------------------------

class TestValidateLeverageCap:

    def test_dallas_incident_caught(self):
        """The exact Dallas trade: NO at 0.9885 = 1.15¢ effective = 87x leverage."""
        result = validate_leverage_cap(0.9885, "NO", 72.0, is_penny=False)
        assert result.valid is False
        assert result.code == "LEVERAGE_EXCEEDED"
        assert "87x" in result.error

    def test_normal_yes_trade_ok(self):
        """YES at 50¢ = 2x leverage, well within limits."""
        result = validate_leverage_cap(0.50, "YES", 100.0)
        assert result.valid is True

    def test_normal_no_trade_ok(self):
        """NO at 0.30 = effective 70¢ = 1.4x leverage."""
        result = validate_leverage_cap(0.30, "NO", 100.0)
        assert result.valid is True

    def test_core_boundary_5pct(self):
        """Core trade at exactly 5¢ effective = 20x leverage, right at limit."""
        result = validate_leverage_cap(0.05, "YES", 50.0, is_penny=False)
        assert result.valid is True

    def test_core_below_5pct_rejected(self):
        """Core trade at 4¢ effective = 25x > 20x limit."""
        result = validate_leverage_cap(0.04, "YES", 50.0, is_penny=False)
        assert result.valid is False
        assert result.code == "LEVERAGE_EXCEEDED"

    def test_penny_at_3pct_ok(self):
        """Penny trade at 3¢ effective = 33x, within 50x penny limit."""
        result = validate_leverage_cap(0.03, "YES", 15.0, is_penny=True)
        assert result.valid is True

    def test_penny_at_1pct_rejected(self):
        """Penny trade at 1¢ effective = 100x > 50x penny limit."""
        result = validate_leverage_cap(0.01, "YES", 15.0, is_penny=True)
        assert result.valid is False
        assert result.code == "LEVERAGE_EXCEEDED"

    def test_no_side_high_yes_price(self):
        """NO at YES_price=0.97 → effective 3¢ = 33x. Core rejected, penny ok."""
        core = validate_leverage_cap(0.97, "NO", 50.0, is_penny=False)
        penny = validate_leverage_cap(0.97, "NO", 15.0, is_penny=True)
        assert core.valid is False
        assert penny.valid is True

    def test_zero_effective_price(self):
        """Edge case: YES at 0.0 or NO at 1.0, effective price is zero."""
        result = validate_leverage_cap(0.0, "YES", 50.0)
        assert result.valid is False
        assert result.code == "LEVERAGE_EXCEEDED"

    def test_penny_boundary_2pct(self):
        """Penny trade at exactly 2¢ = 50x, right at the limit."""
        result = validate_leverage_cap(0.02, "YES", 15.0, is_penny=True)
        assert result.valid is True
