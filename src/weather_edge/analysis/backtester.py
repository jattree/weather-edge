"""Backtester for the Open-Meteo historical forecast archive vs actuals.

HONESTY NOTE (this is the whole point of the rewrite)
------------------------------------------------------
The original backtester reported a flat ``+0.90`` per bucket hit and ``-1.00``
per miss, with NO fees, NO spread, NO fill probability, and a plain average of
three models. That produced a large fictional P&L that the strategy was then
tuned against. The project lost real money partly because of it. See
OPEN_SOURCE_ARCHIVE.md.

Two hard truths this file now respects:

1. **There is no historical Polymarket order-book data here.** Real P&L depends
   on the price you actually paid versus the true probability. Without historical
   prices we CANNOT produce a real track record. So this backtester reports two
   different things and never conflates them:
     * **Forecast skill**, bucket hit rate, MAE, and a climatology baseline.
       These are real and need no price data.
     * **Illustrative P&L under EXPLICIT assumed costs**, computed from a stated
       ``BacktestCosts`` (spread, fee, fill probability, assumed market price).
       It is clearly labelled as illustrative, and the assumptions are echoed in
       the output. It is NOT a track record.

2. **Lead time matters and is not controlled here.** The Open-Meteo
   historical-forecast archive returns the most recently available forecast for
   each date (short lead), but the live bot traded multi-day-out markets. So even
   the skill numbers FLATTER the live setup. This caveat is surfaced in output.

The backtester also no longer claims to run the live model: it uses a 3-model
Open-Meteo subset with optional per-city weights, NOT the full live
bias-corrected, EMOS-calibrated consensus. That gap is stated in ``caveats``.
"""
from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import date, timedelta

import httpx

from weather_edge.config import CITIES, get_model_weights
from weather_edge.models.enums import City

logger = logging.getLogger(__name__)

# Open-Meteo historical endpoints, always free tier (customer archive needs Professional plan)
_ARCHIVE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
_OBSERVATION_URL = "https://archive-api.open-meteo.com/v1/archive"

# Semaphore for rate-limiting
_SEM = asyncio.Semaphore(4)

# 3-model subset available in the Open-Meteo historical-forecast archive.
_ARCHIVE_MODELS = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless"]

# Floor on ensemble spread. Three models badly understate true forecast
# uncertainty, so a too-tight std would make bucket probabilities overconfident.
_STD_FLOOR_C = 1.5

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


@dataclass
class BacktestCosts:
    """Explicit, stated cost model for illustrative P&L.

    These are assumptions, not measurements, there is no historical order book.
    Defaults are deliberately conservative for thin Polymarket weather markets.
    """
    spread: float = 0.04          # per-share cost of crossing the spread on entry
    fee_rate: float = 0.01        # taker fee as a fraction of entry notional
    fill_prob: float = 1.0        # probability a desired order actually fills
    payout: float = 1.0           # a winning share redeems for $1
    # Assumed price paid for the traded bucket. If None, assume the market prices
    # the bucket fairly at the model's own probability, making the bot a pure
    # price-taker whose expected edge is zero BEFORE costs (so costs alone make
    # it lose, which is exactly the lesson). Set a lower value to model buying a
    # bucket the crowd underpriced (a real edge).
    assumed_market_price: float | None = None

    def to_dict(self) -> dict:
        return {
            "spread": self.spread,
            "fee_rate": self.fee_rate,
            "fill_prob": self.fill_prob,
            "payout": self.payout,
            "assumed_market_price": self.assumed_market_price,
        }


def _c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def _bucket_probability(
    mu: float, sigma: float, lo: float | None, hi: float | None,
) -> float:
    """P(lo <= X < hi) under Normal(mu, sigma), with open-ended edges."""
    p_lo = _normal_cdf(lo, mu, sigma) if lo is not None else 0.0
    p_hi = _normal_cdf(hi, mu, sigma) if hi is not None else 1.0
    return max(0.0, p_hi - p_lo)


def _pnl_under_costs(won: bool, entry_price: float, costs: BacktestCosts) -> float:
    """Illustrative per-$1-stake P&L for one trade under explicit costs.

    entry_cost = price + spread + fee. A win redeems at ``payout``; a loss
    redeems at $0. Scaled by fill probability (a fractional-fill model).
    """
    entry_cost = entry_price + costs.spread + costs.fee_rate * entry_price
    realized = (costs.payout - entry_cost) if won else (-entry_cost)
    return costs.fill_prob * realized


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


def _weighted_mean(city_id: City, values: list[float]) -> float:
    """Weighted mean of the archive-model predictions using per-city weights.

    Falls back to a simple average when weights for the archive subset are not
    available. NOTE: this is still NOT the live bias-corrected EMOS consensus.
    """
    if not values:
        return 0.0
    try:
        weights = get_model_weights(city_id)
        wmap = {m.value: w for m, w in weights.items()}
        ws = [wmap.get(mid, 0.0) for mid in _ARCHIVE_MODELS[: len(values)]]
        if sum(ws) > 0:
            return sum(v * w for v, w in zip(values, ws)) / sum(ws)
    except Exception:
        pass
    return sum(values) / len(values)


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
    model_prob: float           # model probability assigned to the traded bucket
    would_have_won: bool
    pnl_under_costs: float      # illustrative P&L per $1 stake under BacktestCosts

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "city_id": self.city_id,
            "city_name": self.city_name,
            "predicted_temp_c": round(self.predicted_temp_c, 1) if self.predicted_temp_c else None,
            "actual_temp_c": round(self.actual_temp_c, 1) if self.actual_temp_c else None,
            "predicted_bucket": self.predicted_bucket,
            "actual_bucket": self.actual_bucket,
            "model_prob": round(self.model_prob, 3),
            "would_have_won": self.would_have_won,
            # Kept under the original key for dashboard compatibility, but the
            # value is now illustrative P&L under explicit costs, not a flat
            # +0.90/-1.00 fiction.
            "theoretical_pnl": round(self.pnl_under_costs, 3),
        }


async def _fetch_historical_forecasts(
    city_id: City, start_date: date, end_date: date
) -> dict[str, list[float]]:
    """Fetch what models predicted for a date range from the historical forecast archive.

    Returns {date_str: [temp_max values from each model]}.

    Caveat: this archive returns the latest available forecast per date (short
    lead time). It does NOT reproduce the multi-day-ahead forecast the live bot
    actually traded on, so skill here is optimistic relative to live conditions.
    """
    config = CITIES[city_id]

    predictions: dict[str, list[float]] = {}

    async with _SEM:
        for model_id in _ARCHIVE_MODELS:
            try:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    resp = await client.get(
                        _ARCHIVE_URL,
                        params={
                            "latitude": config.latitude,
                            "longitude": config.longitude,
                            "start_date": start_date.isoformat(),
                            "end_date": end_date.isoformat(),
                            "daily": "temperature_2m_max",
                            "models": model_id,
                            "timezone": "auto",
                        },
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

    Caveat: this uses the Open-Meteo reanalysis archive, NOT the METAR oracle
    Polymarket resolves against. For a resolution-faithful backtest, swap this
    for weather_edge.fetchers.metar.fetch_station_tmax_range.
    """
    config = CITIES[city_id]
    observations: dict[str, float | None] = {}

    async with _SEM:
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(
                    _OBSERVATION_URL,
                    params={
                        "latitude": config.latitude,
                        "longitude": config.longitude,
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "daily": "temperature_2m_max",
                        "timezone": "auto",
                    },
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


async def _backtest_city(
    city_id: City, start_date: date, end_date: date, costs: BacktestCosts,
) -> list[BacktestRow]:
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

        predicted_c = _weighted_mean(city_id, pred_values)
        # Ensemble spread -> probability. Three models understate uncertainty,
        # so floor the std.
        if len(pred_values) > 1:
            mean = sum(pred_values) / len(pred_values)
            var = sum((v - mean) ** 2 for v in pred_values) / (len(pred_values) - 1)
            std_c = max(_STD_FLOOR_C, math.sqrt(var))
        else:
            std_c = _STD_FLOOR_C

        if use_fahrenheit:
            mu = _c_to_f(predicted_c)
            sigma = std_c * 9.0 / 5.0
            actual_display = _c_to_f(actual_c)
        else:
            mu = predicted_c
            sigma = std_c
            actual_display = actual_c

        pred_bucket_idx = _find_bucket(mu, buckets)
        actual_bucket_idx = _find_bucket(actual_display, buckets)
        lo, hi = buckets[pred_bucket_idx]

        model_prob = _bucket_probability(mu, sigma, lo, hi)
        won = pred_bucket_idx == actual_bucket_idx

        # Illustrative P&L under explicit costs. Entry price defaults to the
        # model's own probability (a fair price -> zero edge before costs).
        entry_price = (
            costs.assumed_market_price
            if costs.assumed_market_price is not None
            else model_prob
        )
        pnl = _pnl_under_costs(won, entry_price, costs)

        rows.append(BacktestRow(
            date=d,
            city_id=city_id.value,
            city_name=config.name,
            predicted_temp_c=predicted_c,
            actual_temp_c=actual_c,
            predicted_bucket=_bucket_label(lo, hi, unit),
            actual_bucket=_bucket_label(*buckets[actual_bucket_idx], unit),
            model_prob=model_prob,
            would_have_won=won,
            pnl_under_costs=pnl,
        ))

    return rows


def _climatology_baseline_hits(rows: list[BacktestRow]) -> int:
    """How many days the single most-frequent actual bucket would have hit.

    A no-skill baseline: if you always bet the modal bucket per city, how often
    are you right? Model skill must beat this to be meaningful.
    """
    from collections import Counter, defaultdict
    by_city: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_city[r.city_id].append(r.actual_bucket)
    hits = 0
    for actuals in by_city.values():
        if not actuals:
            continue
        modal, _count = Counter(actuals).most_common(1)[0]
        hits += sum(1 for a in actuals if a == modal)
    return hits


async def run_backtest(
    days: int = 7,
    cities: list[str] | None = None,
    costs: BacktestCosts | None = None,
) -> dict:
    """Run a backtest over the past N days for specified cities (or all).

    Returns a summary dict with rows, forecast-skill stats, illustrative P&L
    under explicit costs, the cost assumptions used, and a list of caveats.
    """
    costs = costs or BacktestCosts()
    end_date = date.today() - timedelta(days=1)  # yesterday (most recent complete day)
    start_date = end_date - timedelta(days=days - 1)

    city_ids = []
    if cities:
        for c in cities:
            try:
                city_ids.append(City(c.lower()))
            except ValueError:
                logger.warning("Unknown city in backtest request: %s", c)
    else:
        city_ids = list(City)

    tasks = [_backtest_city(cid, start_date, end_date, costs) for cid in city_ids]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_rows: list[BacktestRow] = []
    for result in results:
        if isinstance(result, Exception):
            logger.warning("Backtest city failed: %s", result)
            continue
        all_rows.extend(result)

    total_trades = len(all_rows)
    wins = sum(1 for r in all_rows if r.would_have_won)
    losses = total_trades - wins
    total_pnl = sum(r.pnl_under_costs for r in all_rows)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    mae_c = (
        sum(abs(r.predicted_temp_c - r.actual_temp_c) for r in all_rows
            if r.predicted_temp_c is not None and r.actual_temp_c is not None)
        / total_trades
    ) if total_trades > 0 else 0.0
    baseline_hits = _climatology_baseline_hits(all_rows)
    baseline_rate = (baseline_hits / total_trades * 100) if total_trades > 0 else 0.0

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
        cs["pnl"] += row.pnl_under_costs

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
        "baseline_win_rate": round(baseline_rate, 1),
        "mae_c": round(mae_c, 2),
        "total_pnl": round(total_pnl, 2),
        "pnl_is_illustrative": True,
        "cost_assumptions": costs.to_dict(),
        "caveats": [
            "P&L is ILLUSTRATIVE under the stated cost_assumptions, not a real "
            "track record, there is no historical Polymarket order-book data.",
            "With assumed_market_price=None the bot is a price-taker at a fair "
            "price, so expected P&L before costs is ZERO; any loss shown is the "
            "cost of spread+fees. This is the core lesson.",
            "Forecasts come from the Open-Meteo historical-forecast archive "
            "(short lead time), not the multi-day-ahead forecasts the live bot "
            "traded, so skill here flatters live conditions.",
            "Actuals come from Open-Meteo reanalysis, NOT the METAR oracle "
            "Polymarket resolves against; swap in fetch_station_tmax_range for a "
            "resolution-faithful run.",
            "This uses a 3-model Open-Meteo subset, NOT the live bias-corrected "
            "EMOS consensus, so it does not fully reproduce the live model.",
            "Compare win_rate against baseline_win_rate (always-bet-modal-bucket) "
            "before concluding the model has skill.",
        ],
        "rows": [r.to_dict() for r in all_rows],
        "city_stats": list(city_stats.values()),
    }
