
from weather_edge.analysis.consensus import ConsensusResult
from weather_edge.analysis.pattern_detector import PatternAlert, get_pattern_adjustment
from weather_edge.fetchers.polymarket import MarketInfo
from weather_edge.models.enums import City
from weather_edge.scheduler import compute_model_prob_for_market


def test_celsius_bucket_width_fix():
    """Verify that Celsius buckets are treated as 1-degree wide, not 2."""
    model_values = {"m1": 25.0, "m2": 28.8} # average = 26.9
    consensus = ConsensusResult(
        city_id="hkg", target_date="2026-04-06", variable="temp_max_c",
        model_count=10, mean_value=26.9, median_value=26.9,
        std_dev=1.5, min_value=24.0, max_value=30.0,
        weighted_mean=26.9, model_values=model_values, model_weights={},
        confidence=0.8, threshold_probs={}
    )
    market = MarketInfo(
        market_id="hkg_25c", city_id=City.HKG, threshold_unit="celsius",
        threshold_dir="range", threshold_low_c=25.0, threshold_high_c=26.0,
        yes_price=0.05
    )
    prob = compute_model_prob_for_market(market, consensus)
    # Correct P(25 <= X < 26) = ~0.17
    assert 0.15 < prob < 0.20, f"Prob {prob} should be ~0.17 for 1-degree Celsius bucket"

def test_pattern_bias_adjustment():
    """Verify that PRD_HAZE_SUPPRESSION generates a negative bias."""
    alert = PatternAlert(
        city_id=City.HKG,
        pattern_name="prd_haze_suppression",
        description="test",
        affected_models=[],
        bias_direction="warm", # models too warm
        estimated_magnitude_c=3.0,
        confidence=0.6,
        trading_implication="fade"
    )
    conf_mult, bias = get_pattern_adjustment(City.HKG, [alert])
    assert bias == -3.0
    assert conf_mult > 1.0

def test_haze_shift_effect():
    """Verify how a -3C shift affects the 25C bucket probability."""
    # Base mean 26.9, shifted mean 23.9
    model_values = {"m1": 22.0, "m2": 25.8} # average = 23.9
    consensus = ConsensusResult(
        city_id="hkg", target_date="2026-04-06", variable="temp_max_c",
        model_count=10, mean_value=23.9, median_value=23.9,
        std_dev=1.5, min_value=21.0, max_value=27.0,
        weighted_mean=23.9, model_values=model_values, model_weights={},
        confidence=0.8, threshold_probs={}
    )
    market = MarketInfo(
        market_id="hkg_25c", city_id=City.HKG, threshold_unit="celsius",
        threshold_dir="range", threshold_low_c=25.0, threshold_high_c=26.0,
        yes_price=0.05
    )
    prob = compute_model_prob_for_market(market, consensus)
    # Z(25) = (25-23.9)/1.5 = 0.73 -> P(Z>=0.73) = 0.232
    # Z(26) = (26-23.9)/1.5 = 1.4 -> P(Z>=1.4) = 0.08
    # P = 0.232 - 0.08 = 0.152
    assert 0.13 < prob < 0.18, f"Shifted prob {prob} should be ~0.15"
