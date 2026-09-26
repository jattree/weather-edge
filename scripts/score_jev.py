"""Can Jev (TypeSafe AI's decision model) forecast temperature buckets better
than the market?

Same events, decision time and information as score_vs_market.py, but the
bucket probabilities come from Jev instead of a Normal distribution. Jev is
market-blind and sees numbers only: no city, station, date or prices, so it
cannot recall outcomes it may have seen elsewhere. It gets the day-ahead model
highs, the city's recent bias/error, the last few days of forecast vs
observed, and the event's buckets as anonymous options.

Scored exactly like the model: log-loss / Brier on the winning bucket against
the market's normalised prices, plus a market/Jev blend.

Needs TYPESAFE_API_KEY in the environment or .env. Answers are cached in a
JSON file so a rerun never pays twice. Jev costs $0.042 per 1M input tokens.

    python scripts/score_jev.py --dry-run            # show one request, no calls
    python scripts/score_jev.py --limit 50           # pilot
    python scripts/score_jev.py --max-usd 0.50       # full run with a spend cap
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_day_basis import consensus  # noqa: E402
from score_vs_market import (  # noqa: E402
    PROB_FLOOR,
    TRAILING_DAYS,
    _brier,
    _logloss,
    _se,
    band_c,
    decision_time,
    load_city_data,
    load_events,
    model_params,
    model_probs,
    price_at,
)

from weather_edge.models.enums import City  # noqa: E402

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"  # pinned, as in the nexted pilot and beebots
USD_PER_MTOK = 0.042
CONTEXT_DAYS = 10
CONCURRENCY = 4
MAX_CONSECUTIVE_FAILURES = 8

INSTRUCTIONS = (
    "Forecast which bucket the observed daily maximum temperature at an airport "
    "weather station will fall in. The state holds numerical weather prediction "
    "model forecasts of the daily high (issued about two days ahead, gridded, so "
    "they can be biased relative to the station), the station's recent mean "
    "forecast error and its spread, and the last few days of consensus forecast "
    "versus observed. Assign realistic probabilities to every bucket."
)


# --------------------------------------------------------------------------- request


def _band_text(row, is_hkg: bool) -> str:
    low, high = band_c(row, is_hkg)
    if math.isinf(low):
        return f"below {high:.1f} C"
    if math.isinf(high):
        return f"{low:.1f} C or above"
    return f"from {low:.1f} C up to (not including) {high:.1f} C"


def _recent(highs: dict, obs: dict, target: str) -> list[dict]:
    day = date.fromisoformat(target)
    out = []
    for back in range(CONTEXT_DAYS, 0, -1):
        d = str(day - timedelta(days=back))
        c = consensus(highs, d)
        if c is not None and d in obs:
            out.append({"days_ago": back, "consensus_c": round(c, 1), "observed_c": round(obs[d], 1)})
    return out


def build_request(city: dict, target: str, rows, params: tuple[float, float], is_hkg: bool) -> dict:
    highs, obs = city["highs"], city["obs"]
    raw = consensus(highs, target)
    mu, sigma = params
    state = {
        "model_forecasts_c": {m: round(s[target], 1) for m, s in sorted(highs.items()) if target in s},
        "consensus_c": round(raw, 2),
        f"recent_{TRAILING_DAYS}d_mean_error_c": round(raw - mu, 2),
        f"recent_{TRAILING_DAYS}d_error_std_c": round(sigma, 2),
        "recent_days": _recent(highs, obs, target),
    }
    criteria = {f"b{i}": _band_text(r, is_hkg) for i, r in enumerate(rows)}
    return {
        "model": MODEL,
        "state": state,
        "questions": {"bucket": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}},
    }


def jev_probs(answer: dict, n: int) -> list[float] | None:
    """Normalised probabilities for b0..b{n-1}, or None if unusable."""
    probs = (answer.get("answers", {}).get("bucket") or {}).get("probabilities")
    if not isinstance(probs, dict):
        return None
    vals = [max(float(probs.get(f"b{i}", 0.0) or 0.0), PROB_FLOOR) for i in range(n)]
    total = sum(vals)
    return [v / total for v in vals] if total > 0 else None


# --------------------------------------------------------------------------- calls


class Budget:
    def __init__(self, max_usd: float):
        self.max_usd, self.spent, self.failures, self.stopped = max_usd, 0.0, 0, False

    def charge(self, answer: dict) -> None:
        tokens = (answer.get("usage") or {}).get("input_tokens", 0) or 0
        self.spent += tokens * USD_PER_MTOK / 1e6
        self.failures = 0
        self.stopped = self.stopped or self.spent >= self.max_usd

    def fail(self) -> None:
        self.failures += 1
        self.stopped = self.stopped or self.failures >= MAX_CONSECUTIVE_FAILURES


async def ask_jev(client: httpx.AsyncClient, key: str, body: dict) -> dict | None:
    for attempt in range(4):
        resp = await client.post(ENDPOINT, json=body, timeout=30.0,
                                 headers={"Authorization": f"Bearer {key}"})
        if resp.status_code in (429, 500, 502, 503, 529):
            await asyncio.sleep(2 ** attempt)
            continue
        if resp.status_code in (401, 403):
            raise SystemExit(f"Jev rejected the key (HTTP {resp.status_code}).")
        if resp.status_code != 200:
            return None
        return resp.json()
    return None


async def collect(jobs: list[tuple[str, dict]], key: str, cache: dict, cache_path: Path,
                  budget: Budget) -> None:
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient() as client:
        async def one(event_id: str, body: dict) -> None:
            async with sem:
                if budget.stopped or event_id in cache:
                    return
                try:
                    answer = await ask_jev(client, key, body)
                except httpx.HTTPError:
                    answer = None
            if answer is None:
                budget.fail()
                return
            budget.charge(answer)
            cache[event_id] = answer
            cache_path.write_text(json.dumps(cache))

        await asyncio.gather(*(one(e, b) for e, b in jobs))


# --------------------------------------------------------------------------- scoring


def prepare(conn, events, city_data, horizon_h):
    """[(event_id, rows, p_model, p_mkt, win, request)] for scorable events."""
    out = []
    for (cid, target), rows in sorted(events.items()):
        city = city_data.get(cid)
        params = city and model_params(city["highs"], city["obs"], target)
        if not params:
            continue
        t = decision_time(cid, target, horizon_h)
        prices = [price_at(conn, r["id"], t) for r in rows]
        if any(p is None for p in prices) or sum(prices) <= 0:
            continue
        is_hkg = cid == City.HKG.value
        ordered = sorted(rows, key=lambda r: band_c(r, is_hkg)[0])
        prices = [price_at(conn, r["id"], t) for r in ordered]
        p_mkt = [p / sum(prices) for p in prices]
        win = next(i for i, r in enumerate(ordered) if r["resolved_yes"])
        req = build_request(city, target, ordered, params, is_hkg)
        out.append((f"{cid}|{target}", ordered, model_probs(ordered, *params, is_hkg), p_mkt,
                    win, req))
    return out


def summarise(prepared, cache) -> dict:
    ll = {"jev": [], "model": [], "market": []}
    brier = {"jev": [], "market": []}
    blend = {w: [] for w in (0.9, 0.8, 0.7, 0.5)}
    for event_id, rows, p_model, p_mkt, win, _ in prepared:
        p_jev = jev_probs(cache.get(event_id, {}), len(rows))
        if p_jev is None:
            continue
        ll["jev"].append(_logloss(p_jev[win]))
        ll["model"].append(_logloss(p_model[win]))
        ll["market"].append(_logloss(p_mkt[win]))
        brier["jev"].append(_brier(p_jev, win))
        brier["market"].append(_brier(p_mkt, win))
        for w, scores in blend.items():
            scores.append(_logloss(w * p_mkt[win] + (1 - w) * p_jev[win]))
    n = len(ll["jev"])
    if not n:
        return {"events": 0}
    diff = [a - b for a, b in zip(ll["jev"], ll["market"], strict=True)]
    return {
        "events": n,
        "logloss": {k: mean(v) for k, v in ll.items()},
        "logloss_jev_minus_market": mean(diff), "logloss_diff_se": _se(diff),
        "brier": {k: mean(v) for k, v in brier.items()},
        "blend_logloss_w_market": {f"{w:g}": mean(v) for w, v in blend.items()},
    }


# --------------------------------------------------------------------------- main


def _api_key() -> str | None:
    key = os.environ.get("TYPESAFE_API_KEY")
    env = Path(__file__).resolve().parent.parent / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("TYPESAFE_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key or None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=Path(__file__).resolve().parent.parent / "prices.db")
    ap.add_argument("--horizon", type=float, default=36.0)
    ap.add_argument("--data-cache", type=Path, default=Path("score_vs_market_cache.json"))
    ap.add_argument("--jev-cache", type=Path, default=Path("score_jev_cache.json"))
    ap.add_argument("--limit", type=int, default=None, help="Only the first N events")
    ap.add_argument("--max-usd", type=float, default=0.50, help="Stop calling Jev past this spend")
    ap.add_argument("--dry-run", action="store_true", help="Print one request and exit")
    args = ap.parse_args(argv)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    events = load_events(conn)
    dates = sorted({date.fromisoformat(d) for _, d in events})
    lead = 1 if args.horizon <= 24 else 2
    city_data = load_city_data({c for c, _ in events}, dates[0] - timedelta(days=TRAILING_DAYS + 10),
                               dates[-1], lead, args.data_cache)
    prepared = prepare(conn, events, city_data, args.horizon)[: args.limit]
    print(f"{len(prepared)} scorable events", file=sys.stderr)
    if args.dry_run:
        print(json.dumps(prepared[0][5], indent=2))
        return

    key = _api_key()
    if not key:
        raise SystemExit("Set TYPESAFE_API_KEY (environment or .env) to run.")
    cache = json.loads(args.jev_cache.read_text()) if args.jev_cache.exists() else {}
    budget = Budget(args.max_usd)
    asyncio.run(collect([(p[0], p[5]) for p in prepared], key, cache, args.jev_cache, budget))
    result = summarise(prepared, cache)
    result["jev_spend_usd_this_run"] = round(budget.spent, 4)
    result["stopped_early"] = budget.stopped
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
