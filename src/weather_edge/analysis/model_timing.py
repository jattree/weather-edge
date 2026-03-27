"""Model run timing detection for front-running stale Polymarket prices.

Weather models update on fixed schedules. Right after a new model run drops,
there's a 5-15 minute window where Polymarket prices haven't adjusted.
This is the "Golden Window" for placing trades with maximum edge.

Model update schedules (approximate UTC times for data availability on Open-Meteo):
- HRRR: Every hour (H+2h, i.e., data from 12z run available ~14:00 UTC)
- GFS: 4x/day, 00z (~04:00), 06z (~10:00), 12z (~16:00), 18z (~22:00)
- ECMWF: 2x/day, 00z (~07:00), 12z (~19:00)
- NAM: 4x/day, 00z (~03:30), 06z (~09:30), 12z (~15:30), 18z (~21:30)
- ICON: 4x/day, 00z (~04:00), 06z (~10:00), 12z (~16:00), 18z (~22:00)

Note: ECMWF and GFS times corrected upward from earlier estimates.
ECMWF requires ~7h (model run + computation + Open-Meteo ingest).
GFS requires ~4h. HRRR requires ~2h.

Source: Gemini CLI analysis + Open-Meteo generationtime_ms observations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from weather_edge.models.enums import WeatherModel

logger = logging.getLogger(__name__)


@dataclass
class ModelRun:
    """A scheduled model run."""
    model: WeatherModel
    run_hour_utc: int  # The initialization hour (00, 06, 12, 18)
    available_offset_minutes: int  # Minutes after run_hour when data is available


# Model run schedules: (run_hour_utc, minutes_after_when_available)
MODEL_SCHEDULES: dict[WeatherModel, list[ModelRun]] = {
    WeatherModel.HRRR: [
        ModelRun(WeatherModel.HRRR, h, 120) for h in range(24)  # Every hour, T+2h
    ],
    WeatherModel.GFS: [
        ModelRun(WeatherModel.GFS, 0, 240),   # 00z available ~04:00 UTC (T+4h)
        ModelRun(WeatherModel.GFS, 6, 240),   # 06z available ~10:00 UTC (T+4h)
        ModelRun(WeatherModel.GFS, 12, 240),  # 12z available ~16:00 UTC (T+4h)
        ModelRun(WeatherModel.GFS, 18, 240),  # 18z available ~22:00 UTC (T+4h)
    ],
    WeatherModel.ECMWF: [
        ModelRun(WeatherModel.ECMWF, 0, 420),   # 00z available ~07:00 UTC (T+7h)
        ModelRun(WeatherModel.ECMWF, 12, 420),  # 12z available ~19:00 UTC (T+7h)
    ],
    WeatherModel.NAM: [
        ModelRun(WeatherModel.NAM, 0, 210),
        ModelRun(WeatherModel.NAM, 6, 210),
        ModelRun(WeatherModel.NAM, 12, 210),
        ModelRun(WeatherModel.NAM, 18, 210),
    ],
    WeatherModel.ICON: [
        ModelRun(WeatherModel.ICON, 0, 240),
        ModelRun(WeatherModel.ICON, 6, 240),
        ModelRun(WeatherModel.ICON, 12, 240),
        ModelRun(WeatherModel.ICON, 18, 240),
    ],
    WeatherModel.UKV: [
        ModelRun(WeatherModel.UKV, 0, 240),
        ModelRun(WeatherModel.UKV, 12, 240),
    ],
    WeatherModel.KMA: [
        ModelRun(WeatherModel.KMA, 0, 300),
        ModelRun(WeatherModel.KMA, 12, 300),
    ],
}


def get_recent_model_updates(
    now: datetime | None = None,
    window_minutes: int = 15,
) -> list[tuple[WeatherModel, int]]:
    """Find models that have released new data within the last `window_minutes`.

    Returns list of (model, minutes_since_release) for models in the golden window.
    These are the models whose data is freshest, and whose impact on prices
    hasn't been fully absorbed yet.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    recent: list[tuple[WeatherModel, int]] = []

    for model, runs in MODEL_SCHEDULES.items():
        for run in runs:
            # Calculate when this run's data became available today
            avail_hour = run.run_hour_utc + (run.available_offset_minutes // 60)
            avail_minute = run.available_offset_minutes % 60

            avail_time = now.replace(
                hour=avail_hour % 24,
                minute=avail_minute,
                second=0,
                microsecond=0,
            )

            # If availability time is in the future today, check yesterday's run
            if avail_time > now:
                continue

            minutes_ago = (now - avail_time).total_seconds() / 60

            if 0 <= minutes_ago <= window_minutes:
                recent.append((model, int(minutes_ago)))
                logger.info(
                    "GOLDEN WINDOW: %s data dropped %d min ago (run %02dz)",
                    model.value, int(minutes_ago), run.run_hour_utc,
                )

    return recent


def is_golden_window(now: datetime | None = None, window_minutes: int = 15) -> bool:
    """Check if we're currently in a golden window (any model just updated)."""
    return len(get_recent_model_updates(now, window_minutes)) > 0


def get_confidence_boost(now: datetime | None = None) -> float:
    """Get a confidence multiplier based on model freshness.

    During golden windows, our forecast is likely more accurate than the market
    because we've incorporated the latest model run. Boost confidence slightly.

    Returns 1.0 (no boost) to 1.15 (max boost when ECMWF just dropped).
    """
    recent = get_recent_model_updates(now, window_minutes=15)
    if not recent:
        return 1.0

    # ECMWF updates are most impactful
    max_boost = 1.0
    for model, minutes_ago in recent:
        freshness = max(0, 1 - minutes_ago / 15)  # 1.0 at 0 min, 0.0 at 15 min
        if model == WeatherModel.ECMWF:
            max_boost = max(max_boost, 1.0 + 0.15 * freshness)
        elif model == WeatherModel.GFS:
            max_boost = max(max_boost, 1.0 + 0.10 * freshness)
        elif model == WeatherModel.HRRR:
            max_boost = max(max_boost, 1.0 + 0.08 * freshness)
        else:
            max_boost = max(max_boost, 1.0 + 0.05 * freshness)

    return min(max_boost, 1.15)
