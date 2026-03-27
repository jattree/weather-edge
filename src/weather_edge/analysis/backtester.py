"""Simple backtester using Open-Meteo historical forecast archive vs actuals."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, timedelta

import httpx

from weather_edge.config import CITIES, get_model_weights
from weather_edge.models.enums import City, WeatherModel

logger = logging.getLogger(__name__)

# Open-Meteo historical forecast archive endpoint
def _archive_url() -> str:
    from weather_edge.config import settings
    if settings.openmeteo_api_key:
        return "https://customer-historical-forecast-api.open-meteo.com/v1/forecast"
    return "https://historical-forecast-api.open-meteo.com/v1/forecast"

def _observation_url() -> str:
    from weather_edge.config import settings
    if settings.openmeteo_api_key:
        return "https://customer-archive-api.open-meteo.com/v1/archive"
    return "https://archive-api.open-meteo.com/v1/archive"

# Semaphore for rate-limiting
_SEM = asyncio.Semaphore(4)

# Common Polymarket-style temperature buckets (Fahrenheit)
_TEMP_BUCKETS_F = [
    (None, 32), (32, 40), (40, 50), (50, 60), (60, 70),
    (70, 80), (80, 90), (90, 100), (100, None),
]

# Celsius equivalents for non-US cities
_TEMP_BUCKETS_C = [
    (None, 0), (0, 5), (5, 10), (10, 15), (15, 20),
    (20, 25), (25, 30), (30, 35), (35, None),
]


def _c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def _bucket_label(lo: float | None, hi: float | None, unit: str) -> str:
    if lo is None:
        return f"<{hi}{unit}"
    if hi is None:
        return f">={lo}{unit}"
    return f"{lo}-{hi}{unit}"


def _find_bucket(value: float, buckets: list[tuple[float | None, float | None]]) -> int:
    for i, (lo, hi) in enumerate(buckets):
        if lo is None and value < hi:
            return i
        if hi is None and value >= lo:
            return i
        if lo is not None and hi is not None and lo <= value < hi:
            return i
    return len(buckets) - 1


@dataclass
class BacktestRow:
    """One backtest result row."""
    date: str
    city_id: str
    city_name: str
    predicted_temp_c: float | None
    actual_temp_c: float | None
    predicted_bucket: str
    actual_bucket: str
    would_have_won: bool
    theoretical_pnl: float  # +0.90 for win (typical Polymarket payout), -1.00 for loss

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "city_id": self.city_id,
            "city_name": self.city_name,
            "predicted_temp_c": round(self.predicted_temp_c, 1) if self.predicted_temp_c else None,
            "actual_temp_c": round(self.actual_temp_c, 1) if self.actual_temp_c else None,
            "predicted_bucket": self.predicted_bucket,
            "actual_bucket": self.actual_bucket,
            "would_have_won": self.would_have_won,
            "theoretical_pnl": round(self.theoretical_pnl, 2),
        }


async def _fetch_historical_forecasts(
    city_id: City, start_date: date, end_date: date
) -> dict[str, list[float]]:
    """Fetch what models predicted for a date range from the historical forecast archive.

    Returns {date_str: [temp_max values from each model]}.
    """
    config = CITIES[city_id]
    # Use global models available in Open-Meteo archive
    models = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless"]

    predictions: dict[str, list[float]] = {}

    async with _SEM:
        for model_id in models:
            try:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    from weather_edge.config import settings as _s
                    _p = {
                            "latitude": config.latitude,
                            "longitude": config.longitude,
                            "start_date": start_date.isoformat(),
                            "end_date": end_date.isoformat(),
                            "daily": "temperature_2m_max",
                            "models": model_id,
                            "timezone": "auto",
                        }
                    if _s.openmeteo_api_key:
                        _p["apikey"] = _s.openmeteo_api_key
                    resp = await client.get(
                        _archive_url(),
                        params=_p,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                daily = data.get("daily", {})
                dates = daily.get("time", [])
                temps = daily.get("temperature_2m_max", [])

                for i, d in enumerate(dates):
                    if i < len(temps) and temps[i] is not None:
                        predictions.setdefault(d, []).append(temps[i])

            except Exception:
                logger.warning("Historical forecast fetch failed for %s model %s", config.name, model_id)

    return predictions


async def _fetch_observations(
    city_id: City, start_date: date, end_date: date
) -> dict[str, float | None]:
    """Fetch actual observed temperatures for a date range.

    Returns {date_str: actual_temp_max_c}.
    """
    config = CITIES[city_id]
    observations: dict[str, float | None] = {}

    async with _SEM:
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                from weather_edge.config import settings as _s2
                _p2 = {
                        "latitude": config.latitude,
                        "longitude": config.longitude,
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "daily": "temperature_2m_max",
                        "timezone": "auto",
                    }
                if _s2.openmeteo_api_key:
                    _p2["apikey"] = _s2.openmeteo_api_key
                resp = await client.get(
                    _observation_url(),
                    params=_p2,
                )
                resp.raise_for_status()
                data = resp.json()

            daily = data.get("daily", {})
            dates = daily.get("time", [])
            temps = daily.get("temperature_2m_max", [])

            for i, d in enumerate(dates):
                observations[d] = temps[i] if i < len(temps) else None

        except Exception:
            logger.warning("Observation fetch failed for %s", config.name, exc_info=True)

    return observations


async def _backtest_city(city_id: City, start_date: date, end_date: date) -> list[BacktestRow]:
    """Run backtest for a single city over a date range."""
    config = CITIES[city_id]
    use_fahrenheit = config.temp_unit == "fahrenheit"
    buckets = _TEMP_BUCKETS_F if use_fahrenheit else _TEMP_BUCKETS_C
    unit = "F" if use_fahrenheit else "C"

    predictions, observations = await asyncio.gather(
        _fetch_historical_forecasts(city_id, start_date, end_date),
        _fetch_observations(city_id, start_date, end_date),
    )

    rows: list[BacktestRow] = []
    all_dates = sorted(set(list(predictions.keys()) + list(observations.keys())))

    for d in all_dates:
        pred_values = predictions.get(d, [])
        actual_c = observations.get(d)

        if not pred_values or actual_c is None:
            continue

        # Weighted mean of model predictions (simple average for backtest)
        predicted_c = sum(pred_values) / len(pred_values)

        # Convert to display unit for bucket assignment
        if use_fahrenheit:
            pred_display = _c_to_f(predicted_c)
            actual_display = _c_to_f(actual_c)
        else:
            pred_display = predicted_c
            actual_display = actual_c

        pred_bucket_idx = _find_bucket(pred_display, buckets)
        actual_bucket_idx = _find_bucket(actual_display, buckets)

        pred_label = _bucket_label(*buckets[pred_bucket_idx], unit)
        actual_label = _bucket_label(*buckets[actual_bucket_idx], unit)

        won = pred_bucket_idx == actual_bucket_idx
        # Typical Polymarket: risk $1 to win $0.90 (implied odds ~52.6%)
        pnl = 0.90 if won else -1.00

        rows.append(BacktestRow(
            date=d,
            city_id=city_id.value,
            city_name=config.name,
            predicted_temp_c=predicted_c,
            actual_temp_c=actual_c,
            predicted_bucket=pred_label,
            actual_bucket=actual_label,
            would_have_won=won,
            theoretical_pnl=pnl,
        ))

    return rows


async def run_backtest(days: int = 7, cities: list[str] | None = None) -> dict:
    """Run a backtest over the past N days for specified cities (or all).

    Returns a summary dict with rows, stats, and per-city performance.
    """
    end_date = date.today() - timedelta(days=1)  # yesterday (most recent complete day)
    start_date = end_date - timedelta(days=days - 1)

    # Filter cities if specified
    city_ids = []
    if cities:
        for c in cities:
            try:
                city_ids.append(City(c.lower()))
            except ValueError:
                logger.warning("Unknown city in backtest request: %s", c)
    else:
        city_ids = list(City)

    # Run all cities concurrently
    tasks = [_backtest_city(cid, start_date, end_date) for cid in city_ids]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_rows: list[BacktestRow] = []
    for result in results:
        if isinstance(result, Exception):
            logger.warning("Backtest city failed: %s", result)
            continue
        all_rows.extend(result)

    # Compute summary stats
    total_trades = len(all_rows)
    wins = sum(1 for r in all_rows if r.would_have_won)
    losses = total_trades - wins
    total_pnl = sum(r.theoretical_pnl for r in all_rows)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

    # Per-city breakdown
    city_stats: dict[str, dict] = {}
    for row in all_rows:
        cs = city_stats.setdefault(row.city_id, {
            "city_id": row.city_id,
            "city_name": row.city_name,
            "trades": 0,
            "wins": 0,
            "pnl": 0.0,
        })
        cs["trades"] += 1
        if row.would_have_won:
            cs["wins"] += 1
        cs["pnl"] += row.theoretical_pnl

    for cs in city_stats.values():
        cs["win_rate"] = round(cs["wins"] / cs["trades"] * 100, 1) if cs["trades"] > 0 else 0.0
        cs["pnl"] = round(cs["pnl"], 2)

    return {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "days": days,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "rows": [r.to_dict() for r in all_rows],
        "city_stats": list(city_stats.values()),
    }
