"""Compare forecast skill on a UTC-day vs station-local-day basis.

Polymarket resolves each market on the station's local civil day, but the
proving run fetched forecasts bucketed by UTC day. This measures how much
error that alone adds, against METAR (HKO for Hong Kong) observations taken
on the local day, using the same forecasts fetched both ways.

With --lead 0 (default) forecasts come from Open-Meteo's historical-forecast
API, which stitches the first hours of each model run: short lead times that
understate a 36h+ forecast. --lead 1 / --lead 2 use the previous-runs API
instead: hourly values from the run issued 1 / 2 days earlier, maxed over each
day, which matches the horizon the bot trades at.

Usage:
    python scripts/compare_day_basis.py --days 90 --end 2026-09-22
    python scripts/compare_day_basis.py --city nyc --out results.json
    python scripts/compare_day_basis.py --lead 2 --days 90
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from statistics import mean

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hindcast import HIST_FORECAST_URL, fetch_batch_metar_observations  # noqa: E402

from weather_edge.config import CITIES, get_models_for_city  # noqa: E402
from weather_edge.fetchers.openmeteo import OPENMETEO_MODEL_IDS  # noqa: E402
from weather_edge.models.enums import City  # noqa: E402

IEM_PAUSE_S = 6.0  # IEM throttles bursts ("Too many requests")
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
MIN_HOURS = 20  # hourly values needed to call a day's max
TRAILING_DAYS = 30
MIN_HISTORY = 10


def fetch_model_highs(city, start: date, end: date, tz: str) -> dict[str, dict[str, float]]:
    """{model_id: {date: daily max C}} for every model of ``city``, one request."""
    model_ids = [OPENMETEO_MODEL_IDS[m] for m in get_models_for_city(city.city_id)]
    resp = httpx.get(
        HIST_FORECAST_URL,
        params={
            "latitude": city.latitude, "longitude": city.longitude,
            "daily": "temperature_2m_max", "timezone": tz,
            "start_date": str(start), "end_date": str(end),
            "models": ",".join(model_ids),
        },
        timeout=60,
    )
    resp.raise_for_status()
    daily = resp.json().get("daily", {})
    dates = daily.get("time", [])
    out: dict[str, dict[str, float]] = {}
    for mid in model_ids:
        vals = daily.get(f"temperature_2m_max_{mid}") or (
            daily.get("temperature_2m_max") if len(model_ids) == 1 else None
        )
        if vals:
            out[mid] = {d: v for d, v in zip(dates, vals, strict=False) if v is not None}
    return out


def fetch_model_highs_lead(
    city, start: date, end: date, tz: str, lead: int,
) -> dict[str, dict[str, float]]:
    """{model_id: {date: max C}} from the run issued ``lead`` days earlier."""
    model_ids = [OPENMETEO_MODEL_IDS[m] for m in get_models_for_city(city.city_id)]
    var = f"temperature_2m_previous_day{lead}"
    resp = httpx.get(
        PREVIOUS_RUNS_URL,
        params={
            "latitude": city.latitude, "longitude": city.longitude,
            "hourly": var, "timezone": tz,
            "start_date": str(start), "end_date": str(end),
            "models": ",".join(model_ids),
        },
        timeout=90,
    )
    resp.raise_for_status()
    hourly = resp.json().get("hourly", {})
    times = hourly.get("time", [])
    out: dict[str, dict[str, float]] = {}
    for mid in model_ids:
        vals = hourly.get(f"{var}_{mid}")
        if not vals:
            continue
        by_day: dict[str, list[float]] = {}
        for t, v in zip(times, vals, strict=False):
            if v is not None:
                by_day.setdefault(t[:10], []).append(v)
        series = {d: max(v) for d, v in by_day.items() if len(v) >= MIN_HOURS}
        if series:
            out[mid] = series
    return out


def consensus(highs: dict[str, dict[str, float]], day: str) -> float | None:
    vals = [series[day] for series in highs.values() if day in series]
    return mean(vals) if vals else None


def bucket(temp_c: float, city) -> int:
    """The whole-degree label the market settles on (resolver rules)."""
    if city.city_id == City.HKG:
        return math.floor(temp_c)  # HKO 0.1 C value, label = integer part
    if city.temp_unit == "fahrenheit":
        return math.floor(temp_c * 9 / 5 + 32 + 0.5)
    return math.floor(temp_c + 0.5)


def score(city, highs, obs: dict[str, float]) -> dict:
    errs, hits = [], 0
    per_model: dict[str, list[float]] = {m: [] for m in highs}
    for day, actual in obs.items():
        c = consensus(highs, day)
        if c is None:
            continue
        errs.append(c - actual)
        hits += bucket(c, city) == bucket(actual, city)
        for m, series in highs.items():
            if day in series:
                per_model[m].append(abs(series[day] - actual))
    if not errs:
        return {"n": 0}
    # Leave-one-out bias correction: each day is corrected with the mean
    # error of the OTHER days only, so this is an honest estimate of what a
    # per-station bias correction could achieve (not an in-sample fit).
    total, n = sum(errs), len(errs)
    loo = [e - (total - e) / (n - 1) for e in errs] if n > 1 else errs
    days = [d for d in obs if consensus(highs, d) is not None]
    loo_hits = sum(
        bucket(obs[d] + e_corr, city) == bucket(obs[d], city)
        for d, e_corr in zip(days, loo, strict=True)
    )
    # Causal version: correct each day with the mean error of the previous
    # TRAILING_DAYS days only (what a live bot could actually know). The
    # first MIN_HISTORY days have too little history and are not scored.
    ordered = sorted(zip(days, errs, strict=True))
    trail_errs, trail_hits = [], 0
    for i, (d, e) in enumerate(ordered):
        past = [pe for _, pe in ordered[max(0, i - TRAILING_DAYS):i]]
        if len(past) < MIN_HISTORY:
            continue
        corrected = e - mean(past)
        trail_errs.append(corrected)
        trail_hits += bucket(obs[d] + corrected, city) == bucket(obs[d], city)
    return {
        "n": n,
        "n_trailing": len(trail_errs),
        "mae_trailing": mean(abs(e) for e in trail_errs) if trail_errs else None,
        "bucket_hit_trailing": trail_hits / len(trail_errs) if trail_errs else None,
        "mae": mean(abs(e) for e in errs),
        "bias": mean(errs),
        "bucket_hit": hits / n,
        "mae_debiased": mean(abs(e) for e in loo),
        "bucket_hit_debiased": loo_hits / n,
        "best_model_mae": min((mean(v) for v in per_model.values() if v), default=None),
    }


def run(city_ids: list[City], start: date, end: date, lead: int = 0) -> dict:
    results = {}
    for i, cid in enumerate(city_ids):
        city = CITIES[cid]
        if i:
            time.sleep(IEM_PAUSE_S)
        obs = fetch_batch_metar_observations(city.icao, start, end, city.timezone)
        if not obs:  # usually IEM throttling: wait and try once more
            time.sleep(IEM_PAUSE_S * 5)
            obs = fetch_batch_metar_observations(city.icao, start, end, city.timezone)
        if not obs:
            print(f"{cid.value}: no observations, skipped", file=sys.stderr)
            continue
        row = {}
        for basis, tz in (("utc", "UTC"), ("local", city.timezone)):
            try:
                highs = (fetch_model_highs_lead(city, start, end, tz, lead) if lead
                         else fetch_model_highs(city, start, end, tz))
                row[basis] = score(city, highs, obs)
            except httpx.HTTPError as e:
                print(f"{cid.value} {basis}: forecast fetch failed ({e})", file=sys.stderr)
        row["utc_offset_note"] = city.timezone
        results[cid.value] = row
        u, loc = row.get("utc", {}), row.get("local", {})
        if u.get("n") and loc.get("n"):
            print(f"{cid.value:4} {city.timezone:22} n={loc['n']:3}  "
                  f"MAE utc {u['mae']:.2f} -> local {loc['mae']:.2f} -> debiased "
                  f"{loc['mae_debiased']:.2f} | trailing {loc['mae_trailing']:.2f}  "
                  f"bucket hit {u['bucket_hit']:.0%} -> {loc['bucket_hit']:.0%} -> "
                  f"{loc['bucket_hit_debiased']:.0%} | trailing {loc['bucket_hit_trailing']:.0%}")
    return results


def summarise(results: dict) -> dict:
    both = [r for r in results.values() if r.get("utc", {}).get("n") and r.get("local", {}).get("n")]
    if not both:
        return {"cities": 0}

    def avg(basis, key):
        return mean(r[basis][key] for r in both)

    return {
        "cities": len(both),
        "mae_utc": avg("utc", "mae"), "mae_local": avg("local", "mae"),
        "bias_utc": avg("utc", "bias"), "bias_local": avg("local", "bias"),
        "bucket_hit_utc": avg("utc", "bucket_hit"), "bucket_hit_local": avg("local", "bucket_hit"),
        "mae_local_debiased": avg("local", "mae_debiased"),
        "bucket_hit_local_debiased": avg("local", "bucket_hit_debiased"),
        "mae_local_trailing": avg("local", "mae_trailing"),
        "bucket_hit_local_trailing": avg("local", "bucket_hit_trailing"),
        "utc_mae_trailing": avg("utc", "mae_trailing"),
        "utc_bucket_hit_trailing": avg("utc", "bucket_hit_trailing"),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--end", type=date.fromisoformat,
                    default=datetime.now(UTC).date() - timedelta(days=3))
    ap.add_argument("--city", type=str, default=None)
    ap.add_argument("--lead", type=int, choices=range(0, 8), default=0,
                    help="0 = historical API (short lead); N = run issued N days earlier")
    ap.add_argument("--out", type=Path, default=None, help="Write JSON results here")
    args = ap.parse_args(argv)

    start = args.end - timedelta(days=args.days - 1)
    city_ids = [City(args.city)] if args.city else list(City)
    print(f"Window {start} .. {args.end}, {len(city_ids)} cities, lead {args.lead}")
    results = run(city_ids, start, args.end, args.lead)
    summary = summarise(results)
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps({"window": [str(start), str(args.end)],
                                        "lead_days": args.lead,
                                        "summary": summary, "cities": results}, indent=2))


if __name__ == "__main__":
    main()
