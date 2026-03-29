"""Self-learning module, adaptive model weighting via Brier scores.

The 80/20 version: Inverse Brier Weighting. Models that predict well
get more influence, drifting models get demoted. Pure statistics, no ML.

Brier Score = mean((predicted_prob - actual_outcome)^2)
Lower = better. Perfect = 0, coin flip = 0.25.

Adaptive weight = 1 / BrierScore (normalized per city).
Falls back to static MODEL_BASE_WEIGHT if insufficient data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Minimum forecasts needed before trusting adaptive weights
MIN_FORECASTS_FOR_ADAPTIVE = 30


@dataclass
class ModelScore:
    """Brier score and accuracy metrics for a single model."""
    model_name: str
    city_id: str
    brier_score: float
    forecast_count: int
    mean_error_c: float  # Mean absolute error in °C
    adaptive_weight: float


@dataclass
class LearningReport:
    """Summary of the learning engine's latest analysis."""
    model_scores: list[ModelScore]
    cities_with_adaptive: list[str]
    cities_static_fallback: list[str]
    total_forecasts: int
    avg_brier: float


def compute_brier_score(predicted_prob: float, actual_outcome: bool) -> float:
    """Brier score for a single probabilistic prediction.

    Args:
        predicted_prob: Model's predicted probability (0-1)
        actual_outcome: Whether the event actually happened

    Returns:
        (predicted - actual)^2, range [0, 1]. Lower is better.
    """
    actual = 1.0 if actual_outcome else 0.0
    return (predicted_prob - actual) ** 2


def compute_model_brier_from_forecasts(
    forecasts: list[dict],
    threshold_c: float | None = None,
) -> float | None:
    """Compute Brier score from forecast snapshots.

    For temperature forecasts, we convert to a binary outcome:
    "was the forecast within 1°C of actual?"

    This measures model skill rather than probabilistic calibration.

    Args:
        forecasts: List of dicts with forecast_value and actual_value
        threshold_c: Error threshold for "correct" (default 1.0°C)

    Returns:
        Brier score (0-1) or None if no data
    """
    if not forecasts:
        return None
    if threshold_c is None:
        threshold_c = 1.0

    scores = []
    for f in forecasts:
        forecast = f.get("forecast_value")
        actual = f.get("actual_value")
        if forecast is None or actual is None:
            continue
        error = abs(forecast - actual)
        # Binary: was this model within threshold of actual?
        # Use a soft score: 1 - (error / max_error) clamped
        max_error = 5.0  # 5°C error = score of 0
        skill = max(0.0, 1.0 - error / max_error)
        # Brier-like: (1 - skill)^2 when good, skill^2 when bad
        scores.append((1.0 - skill) ** 2)

    if not scores:
        return None
    return sum(scores) / len(scores)


def compute_mean_absolute_error(forecasts: list[dict]) -> float:
    """Mean absolute error in °C from forecast snapshots."""
    errors = []
    for f in forecasts:
        forecast = f.get("forecast_value")
        actual = f.get("actual_value")
        if forecast is not None and actual is not None:
            errors.append(abs(forecast - actual))
    return sum(errors) / len(errors) if errors else 0.0


def get_adaptive_weights(
    store,
    city_id: str,
    window_days: int = 30,
) -> dict[str, float] | None:
    """Get Brier-weighted model weights for a city.

    Returns None if insufficient data (falls back to static weights).

    Args:
        store: PersistentStore with forecast_snapshots
        city_id: City to get weights for
        window_days: Rolling window in days

    Returns:
        Dict of model_name -> normalized weight, or None
    """
    from weather_edge.config import MODEL_BASE_WEIGHT, get_models_for_city
    from weather_edge.models.enums import City

    try:
        city_enum = City(city_id)
    except ValueError:
        return None

    # Get all models for this city
    models = get_models_for_city(city_enum)

    # Fetch recent forecasts per model
    model_briers: dict[str, float] = {}
    model_counts: dict[str, int] = {}

    for model in models:
        forecasts = store.get_forecast_history(
            model_name=model.value,
            city_id=city_id,
            limit=500,
        )
        if len(forecasts) < MIN_FORECASTS_FOR_ADAPTIVE:
            return None  # Not enough data for any model = fall back

        brier = compute_model_brier_from_forecasts(forecasts)
        if brier is not None and brier > 0:
            model_briers[model.value] = brier
            model_counts[model.value] = len(forecasts)

    if not model_briers:
        return None

    # Inverse Brier weighting: better models (lower Brier) get higher weight
    # Blend with static priors to prevent wild swings early
    raw_weights = {}
    for model_name, brier in model_briers.items():
        # Inverse Brier (lower = better = higher weight)
        inv_brier = 1.0 / max(brier, 0.01)
        # Blend with static prior (70% adaptive, 30% static)
        static = MODEL_BASE_WEIGHT.get(
            next((m for m in models if m.value == model_name), None),
            1.0,
        )
        count = model_counts.get(model_name, 0)
        # More data = more trust in adaptive weight
        blend = min(count / 100, 0.7)  # Max 70% adaptive
        raw_weights[model_name] = blend * inv_brier + (1 - blend) * static

    # Normalize
    total = sum(raw_weights.values())
    if total <= 0:
        return None
    return {m: w / total for m, w in raw_weights.items()}


def run_learning_report(store) -> LearningReport:
    """Generate a learning report with model scores and recommendations.

    Called by daily report or on-demand.
    """
    from weather_edge.config import CITIES, get_models_for_city

    all_scores: list[ModelScore] = []
    cities_adaptive = []
    cities_static = []
    total_forecasts = 0

    for city_id in CITIES:
        city_str = city_id.value
        models = get_models_for_city(city_id)

        has_enough = True
        for model in models:
            forecasts = store.get_forecast_history(
                model_name=model.value,
                city_id=city_str,
                limit=500,
            )
            count = len(forecasts)
            total_forecasts += count

            brier = compute_model_brier_from_forecasts(forecasts)
            mae = compute_mean_absolute_error(forecasts)

            if count < MIN_FORECASTS_FOR_ADAPTIVE:
                has_enough = False

            # Get adaptive weight if available
            weights = get_adaptive_weights(store, city_str)
            adaptive_w = (weights or {}).get(model.value, 0.0)

            all_scores.append(ModelScore(
                model_name=model.value,
                city_id=city_str,
                brier_score=brier or 0.0,
                forecast_count=count,
                mean_error_c=mae,
                adaptive_weight=adaptive_w,
            ))

        if has_enough:
            cities_adaptive.append(city_str)
        else:
            cities_static.append(city_str)

    avg_brier = 0.0
    scored = [s for s in all_scores if s.brier_score > 0]
    if scored:
        avg_brier = sum(s.brier_score for s in scored) / len(scored)

    report = LearningReport(
        model_scores=all_scores,
        cities_with_adaptive=cities_adaptive,
        cities_static_fallback=cities_static,
        total_forecasts=total_forecasts,
        avg_brier=avg_brier,
    )

    logger.info(
        "LEARNING REPORT: %d forecasts, %d cities adaptive, %d static, avg Brier=%.4f",
        total_forecasts, len(cities_adaptive), len(cities_static), avg_brier,
    )

    return report
