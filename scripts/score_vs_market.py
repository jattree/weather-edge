"""Score day-ahead forecast probabilities against Polymarket's own prices.

For every resolved daily-high event in prices.db (see `weather_edge
backfill-prices`), take the market's prices at a decision time (default 36 h
before the end of the station's local day, the bot's entry horizon) and a
model distribution built only from information available then:

- consensus of the day-N-ahead model runs (Open-Meteo previous-runs API,
  lead 2 for horizons over 24 h so no run is newer than the decision time),
- minus that city's mean error over the previous TRAILING_DAYS days,
- spread by those days' RMSE, as a Normal over each bucket's resolution band.

Reports log-loss / Brier on the winning bucket for model vs market, each
distribution's implied-mean bias against observations, and a simple trading
simulation at the market price plus an assumed half-spread and the real
taker fee. No trading, no keys.

    python scripts/score_vs_market.py
    python scripts/score_vs_market.py --horizon 24 --edge 0.08 --half-spread 0.01
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
import zoneinfo
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_day_basis import (  # noqa: E402
    IEM_PAUSE_S,
    consensus,
    fetch_batch_metar_observations,
    fetch_model_highs_lead,
)

from weather_edge.config import CITIES  # noqa: E402
from weather_edge.models.enums import City  # noqa: E402
from weather_edge.trading.fees import calculate_taker_fee  # noqa: E402

TRAILING_DAYS = 30
MIN_HISTORY = 10
MIN_SIGMA_C = 0.4
PROB_FLOOR = 1e-4
TRAIN_LOOKBACK_DAYS = TRAILING_DAYS + 10


# --------------------------------------------------------------------------- data


def load_events(conn: sqlite3.Connection) -> dict[tuple[str, str], list[sqlite3.Row]]:
    """Resolved events that have price history: {(city, date): bucket rows}."""
    rows = conn.execute("""
        SELECT m.id, m.city_id, m.target_date, m.threshold_dir, m.bucket_low_int,
               m.bucket_high_int, m.threshold_unit, m.resolved_yes
        FROM markets m
        WHERE m.resolved_yes IS NOT NULL
          AND EXISTS (SELECT 1 FROM price_history h WHERE h.mkt = m.id)
    """).fetchall()
    events: dict[tuple[str, str], list[sqlite3.Row]] = {}
    for r in rows:
        events.setdefault((r["city_id"], r["target_date"]), []).append(r)
    # keep only complete events with exactly one winning bucket
    return {k: v for k, v in events.items() if sum(r["resolved_yes"] for r in v) == 1}


def price_at(conn: sqlite3.Connection, mkt: int, t: int) -> float | None:
    row = conn.execute(
        "SELECT price FROM price_history WHERE mkt = ? AND ts <= ? ORDER BY ts DESC LIMIT 1",
        (mkt, t),
    ).fetchone()
    return row[0] if row else None


def decision_time(city_id: str, target: str, horizon_h: float) -> int:
    """Unix time ``horizon_h`` hours before the end of the local target day."""
    tz = zoneinfo.ZoneInfo(CITIES[City(city_id)].timezone)
    end = datetime.combine(date.fromisoformat(target) + timedelta(days=1),
                           datetime.min.time(), tzinfo=tz)
    return int((end - timedelta(hours=horizon_h)).timestamp())


def load_city_data(city_ids, start: date, end: date, lead: int, cache: Path) -> dict:
    """{city: {"highs": ..., "obs": ...}}, cached as JSON between runs."""
    key = f"{start}_{end}_lead{lead}"
    data = json.loads(cache.read_text()) if cache.exists() else {}
    if data.get("key") != key:
        data = {"key": key, "cities": {}}
    for i, cid in enumerate(sorted(city_ids)):
        if cid in data["cities"]:
            continue
        city = CITIES[City(cid)]
        if i:
            time.sleep(IEM_PAUSE_S)
        obs = fetch_batch_metar_observations(city.icao, start, end, city.timezone)
        highs = fetch_model_highs_lead(city, start, end, city.timezone, lead)
        data["cities"][cid] = {"obs": obs, "highs": highs}
        cache.write_text(json.dumps(data))
        print(f"  fetched {cid}: {len(obs)} obs days, {len(highs)} models", file=sys.stderr)
    return data["cities"]


# --------------------------------------------------------------------------- model


def model_params(highs: dict, obs: dict, target: str) -> tuple[float, float] | None:
    """(mu, sigma) in C for ``target`` from prior days only, or None."""
    mu_raw = consensus(highs, target)
    if mu_raw is None:
        return None
    day = date.fromisoformat(target)
    errs = []
    for back in range(1, TRAILING_DAYS + 1):
        d = str(day - timedelta(days=back))
        c = consensus(highs, d)
        if c is not None and d in obs:
            errs.append(c - obs[d])
    if len(errs) < MIN_HISTORY:
        return None
    bias = mean(errs)
    sigma = math.sqrt(mean((e - bias) ** 2 for e in errs))
    return mu_raw - bias, max(sigma, MIN_SIGMA_C)


def _to_c(value: float, unit: str) -> float:
    return (value - 32) * 5 / 9 if unit == "fahrenheit" else value


def band_c(row: sqlite3.Row, is_hkg: bool) -> tuple[float, float]:
    """Continuous C interval that resolves this bucket YES (resolver rules)."""
    lo, hi, unit = row["bucket_low_int"], row["bucket_high_int"], row["threshold_unit"]
    lo_off, hi_off = (0.0, 1.0) if is_hkg else (-0.5, 0.5)
    low = -math.inf if lo is None else _to_c(lo + lo_off, unit)
    high = math.inf if hi is None else _to_c(hi + hi_off, unit)
    return low, high


def _cdf(x: float, mu: float, sigma: float) -> float:
    if math.isinf(x):
        return 0.0 if x < 0 else 1.0
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))


def model_probs(rows, mu: float, sigma: float, is_hkg: bool) -> list[float]:
    raw = []
    for r in rows:
        low, high = band_c(r, is_hkg)
        raw.append(max(_cdf(high, mu, sigma) - _cdf(low, mu, sigma), 0.0))
    total = sum(raw) or 1.0
    return [p / total for p in raw]


def implied_mean(rows, probs: list[float], is_hkg: bool) -> float:
    """Expected temperature (C) of a bucket distribution; tails 1 C past the edge."""
    exp = 0.0
    for r, p in zip(rows, probs, strict=True):
        low, high = band_c(r, is_hkg)
        mid = (high - 1.0 if math.isinf(low) else low + 1.0 if math.isinf(high)
               else (low + high) / 2)
        exp += p * mid
    return exp


# --------------------------------------------------------------------------- scoring


def score_event(conn, key, rows, city_data, horizon_h) -> dict | None:
    cid, target = key
    city = city_data.get(cid)
    if not city:
        return None
    params = model_params(city["highs"], city["obs"], target)
    t = decision_time(cid, target, horizon_h)
    prices = [price_at(conn, r["id"], t) for r in rows]
    if params is None or any(p is None for p in prices) or sum(prices) <= 0:
        return None
    is_hkg = cid == City.HKG.value
    p_model = model_probs(rows, *params, is_hkg)
    total = sum(prices)
    p_mkt = [p / total for p in prices]
    win = next(i for i, r in enumerate(rows) if r["resolved_yes"])
    obs = city["obs"].get(target)
    return {
        "city": cid, "date": target, "rows": rows, "raw_prices": prices,
        "p_model": p_model, "p_mkt": p_mkt, "win": win, "obs": obs,
        "mean_model": implied_mean(rows, p_model, is_hkg),
        "mean_mkt": implied_mean(rows, p_mkt, is_hkg),
    }


def _logloss(p: float) -> float:
    return -math.log(max(p, PROB_FLOOR))


def _brier(probs: list[float], win: int) -> float:
    return sum((p - (i == win)) ** 2 for i, p in enumerate(probs))


def _se(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


def forecast_metrics(scored: list[dict]) -> dict:
    ll_model = [_logloss(s["p_model"][s["win"]]) for s in scored]
    ll_mkt = [_logloss(s["p_mkt"][s["win"]]) for s in scored]
    diff = [a - b for a, b in zip(ll_model, ll_mkt, strict=True)]
    with_obs = [s for s in scored if s["obs"] is not None]
    return {
        "events": len(scored),
        "logloss_model": mean(ll_model), "logloss_market": mean(ll_mkt),
        "logloss_model_minus_market": mean(diff), "logloss_diff_se": _se(diff),
        "brier_model": mean(_brier(s["p_model"], s["win"]) for s in scored),
        "brier_market": mean(_brier(s["p_mkt"], s["win"]) for s in scored),
        "p_winner_model": mean(s["p_model"][s["win"]] for s in scored),
        "p_winner_market": mean(s["p_mkt"][s["win"]] for s in scored),
        "market_mean_bias_c": mean(s["mean_mkt"] - s["obs"] for s in with_obs),
        "model_mean_bias_c": mean(s["mean_model"] - s["obs"] for s in with_obs),
        "market_mean_mae_c": mean(abs(s["mean_mkt"] - s["obs"]) for s in with_obs),
        "model_mean_mae_c": mean(abs(s["mean_model"] - s["obs"]) for s in with_obs),
    }


def blend_metrics(scored: list[dict]) -> dict:
    """Log-loss of w*market + (1-w)*model: does the model add anything?"""
    out = {}
    for w in (1.0, 0.9, 0.8, 0.7, 0.5):
        out[f"w_market_{w:g}"] = mean(
            _logloss(w * s["p_mkt"][s["win"]] + (1 - w) * s["p_model"][s["win"]])
            for s in scored
        )
    return out


def _trade(p_model: float, price: float, won: bool, edge: float, half_spread: float):
    """P&L per $1 staked on the side the model prefers, or None if no trade."""
    for side_price, side_prob, side_won in ((price, p_model, won),
                                            (1 - price, 1 - p_model, not won)):
        cost = side_price + half_spread
        if 0.02 < cost < 0.98 and side_prob - cost >= edge:
            shares = 1.0 / cost
            fee = calculate_taker_fee(cost, 1.0)
            return shares * side_won - 1.0 - fee
    return None


def trading_metrics(scored: list[dict], edge: float, half_spread: float) -> dict:
    pnl = []
    for s in scored:
        for i, r in enumerate(s["rows"]):
            res = _trade(s["p_model"][i], s["raw_prices"][i], bool(r["resolved_yes"]),
                         edge, half_spread)
            if res is not None:
                pnl.append(res)
    return {
        "trades": len(pnl), "edge_threshold": edge, "half_spread": half_spread,
        "total_pnl_per_$1": sum(pnl), "mean_return": mean(pnl) if pnl else None,
        "mean_return_se": _se(pnl) if pnl else None,
        "win_rate": mean(p > 0 for p in pnl) if pnl else None,
    }


# --------------------------------------------------------------------------- main


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=Path(__file__).resolve().parent.parent / "prices.db")
    ap.add_argument("--horizon", type=float, default=36.0,
                    help="Decision time, hours before the end of the local target day")
    ap.add_argument("--edge", type=float, default=0.05, help="Min model-minus-cost to trade")
    ap.add_argument("--half-spread", type=float, default=0.01,
                    help="Assumed half-spread paid on entry (history has no bid/ask)")
    ap.add_argument("--cache", type=Path, default=Path("score_vs_market_cache.json"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    lead = 1 if args.horizon <= 24 else 2
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    events = load_events(conn)
    dates = sorted({date.fromisoformat(d) for _, d in events})
    start, end = dates[0] - timedelta(days=TRAIN_LOOKBACK_DAYS), dates[-1]
    print(f"{len(events)} events {dates[0]}..{dates[-1]}, horizon {args.horizon:g}h, "
          f"lead {lead}; fetching {start}..{end}", file=sys.stderr)
    city_data = load_city_data({c for c, _ in events}, start, end, lead, args.cache)

    scored = [s for k, rows in sorted(events.items())
              if (s := score_event(conn, k, rows, city_data, args.horizon))]
    result = {
        "horizon_h": args.horizon, "lead_days": lead,
        "forecast": forecast_metrics(scored),
        "blend_logloss": blend_metrics(scored),
        "trading": trading_metrics(scored, args.edge, args.half_spread),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
