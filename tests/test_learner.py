"""Tests for the learner skill-loss surrogate (#11).

The forecast-based score is NOT a probabilistic Brier score, it's a skill-loss
surrogate. These tests pin its ordering behaviour and the back-compat alias.
"""
from __future__ import annotations

from weather_edge.analysis.learner import (
    compute_brier_score,
    compute_model_brier_from_forecasts,
    compute_model_skill_loss,
)


def _snap(forecast, actual):
    return {"forecast_value": forecast, "actual_value": actual}


class TestSkillLoss:

    def test_perfect_forecast_zero_loss(self):
        assert compute_model_skill_loss([_snap(20.0, 20.0)]) == 0.0

    def test_better_model_has_lower_loss(self):
        good = compute_model_skill_loss([_snap(20.0, 20.2), _snap(15.0, 15.1)])
        bad = compute_model_skill_loss([_snap(20.0, 23.0), _snap(15.0, 18.0)])
        assert good < bad

    def test_none_when_no_usable_data(self):
        assert compute_model_skill_loss([]) is None
        assert compute_model_skill_loss([_snap(None, 20.0)]) is None

    def test_backcompat_alias_is_same_function(self):
        assert compute_model_brier_from_forecasts is compute_model_skill_loss


class TestRealBrierScore:

    def test_real_brier_is_squared_error(self):
        # This one IS a genuine Brier score: (p - outcome)^2
        assert compute_brier_score(1.0, True) == 0.0
        assert compute_brier_score(0.0, True) == 1.0
        assert abs(compute_brier_score(0.5, True) - 0.25) < 1e-9
