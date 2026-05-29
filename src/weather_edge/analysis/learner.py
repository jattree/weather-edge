"""Self-learning module, adaptive model weighting from recent skill.

Models that have predicted well recently get more influence; drifting models
get demoted. Pure statistics, no ML.

STATISTICAL CAVEATS (read before trusting these weights)
--------------------------------------------------------
1. The per-model score here is NOT a probabilistic Brier score. A real Brier
   score needs a predicted PROBABILITY and a binary OUTCOME:
   mean((p - outcome)^2). We only have point temperature forecasts, so we use a
   *skill-loss* surrogate: mean((1 - skill)^2) where skill = 1 - |err|/5°C. It
   ranks models by accuracy but is not calibrated and is not comparable to a
   true Brier score. ``compute_brier_score`` (single probabilistic prediction)
   IS a real Brier score; the forecast-based one is not.

2. Weights are fit IN-SAMPLE. There is no train/test split and no time-forward
   holdout, the same snapshots are used to score and to weight. Treat adaptive
   weighting as a heuristic, not a validated edge.

3. MIN_FORECASTS_FOR_ADAPTIVE = 30 is roughly one month of daily forecasts per
   model, far too few to distinguish genuine skill from noise. The weights are
   suggestive at best until you have a few hundred resolved forecasts per model.

Falls back to static MODEL_BASE_WEIGHT when data is insufficient.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Minimum forecasts needed before trusting adaptive weights.
# NOTE: 30 ~= one month of daily forecasts per model. This is statistically
# thin (see module docstring, caveat 3), it gates obviously-insufficient data,
# not a level that makes the weights trustworthy.
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


def compute_model_skill_loss(
    forecasts: list[dict],
    max_error_c: float = 5.0,
) -> float | None:
    """Skill-loss surrogate from forecast snapshots (NOT a Brier score).

    For each snapshot we map the temperature error to a skill in [0, 1]
    (skill = 1 - |err|/max_error_c, clamped) and return the mean of
    (1 - skill)^2. Lower = better. This ranks models by recent accuracy but is
    not a calibrated/probabilistic Brier score, see the module docstring.

    Args:
        forecasts: List of dicts with forecast_value and actual_value
        max_error_c: Error (°C) that maps to zero skill (default 5.0)

    Returns:
        Mean skill-loss in [0, 1], or None if no usable data.
    """
    if not forecasts:
        return None

    scores = []
    for f in forecasts:
        forecast = f.get("forecast_value")
        actual = f.get("actual_value")
        if forecast is None or actual is None:
            continue
        error = abs(forecast - actual)
        skill = max(0.0, 1.0 - error / max_error_c)
        scores.append((1.0 - skill) ** 2)

    if not scores:
        return None
    return sum(scores) / len(scores)


# Backwards-compatible alias for the old (misleadingly named) function.
compute_model_brier_from_forecasts = compute_model_skill_loss


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

        skill_loss = compute_model_skill_loss(forecasts)
        # Keep a perfect (0.0) model too, the inverse below floors at 0.01, so a
        # 0.0 loss becomes the maximum weight rather than being silently dropped.
        if skill_loss is not None:
            model_briers[model.value] = skill_loss
            model_counts[model.value] = len(forecasts)

    if not model_briers:
        return None

    # Inverse skill-loss: better models (lower loss) get higher weight.
    # We blend the ADAPTIVE distribution with the STATIC-prior distribution. Both
    # are normalised to sum to 1 first, so "blend" is a true mixture weight. The
    # previous code mixed a raw 1/loss term (which ranges up to ~100) against a
    # static prior of ~1, so the adaptive term dominated and the "30% static
    # prior" was effectively a few percent, not the intended safeguard.
    inv = {m: 1.0 / max(loss, 0.01) for m, loss in model_briers.items()}
    inv_total = sum(inv.values())
    static_raw = {
        m: MODEL_BASE_WEIGHT.get(
            next((mm for mm in models if mm.value == m), None), 1.0,
        )
        for m in model_briers
    }
    static_total = sum(static_raw.values())
    if inv_total <= 0 or static_total <= 0:
        return None

    blended: dict[str, float] = {}
    for model_name in model_briers:
        a_rel = inv[model_name] / inv_total            # adaptive distribution
        s_rel = static_raw[model_name] / static_total  # static-prior distribution
        count = model_counts.get(model_name, 0)
        blend = min(count / 100, 0.7)                  # more data -> more adaptive
        blended[model_name] = blend * a_rel + (1 - blend) * s_rel

    total = sum(blended.values())
    if total <= 0:
        return None
    return {m: w / total for m, w in blended.items()}


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

            brier = compute_model_skill_loss(forecasts)
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
