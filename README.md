# Weather Edge

An automated trading bot for Polymarket's daily-high-temperature markets. It
bias-corrects a multi-model weather forecast consensus against the same METAR
station data Polymarket resolves on, and bets where its probability disagrees
with the crowd's price.

> ### ⚠️ Status: sunset. This bot lost money. Read before forking.
>
> Over a live proving run it went **$210 → $51.61 (−75.4%)** and was retired.
> It is published as **(1) a cautionary tale** about how a clever architecture
> can hide a non-existent edge, and **(2) a clean reference implementation** of
> the moving parts (multi-model consensus, METAR-faithful resolution, dual-AI
> review, gasless redemption, a paper-trading + backtest harness).
>
> **Do not point real money at it.** The forecast edge was never proven. See
> [`OPEN_SOURCE_ARCHIVE.md`](OPEN_SOURCE_ARCHIVE.md) for the full post-mortem.

---

## Why it failed (the short version)

Three audit-preventable mistakes, then a deeper one:

1. **Wrong data source.** Backtests resolved against Open-Meteo gridded
   reanalysis; Polymarket resolves against Weather Underground's displayed METAR
   value (~0.9 °C MAE apart). The glowing paper P&L was fiction.
2. **Wrong stations.** Three of 24 cities resolved against the wrong airport
   (Denver, Houston, Hong Kong), a guaranteed loss on every trade there.
3. **Wrong execution structure.** Buying several adjacent YES buckets per city
   means most legs lose by construction.
4. **The deeper failure.** Even after fixing all of the above, there was no
   durable edge: forecast MAE (~1.4 °C) is *wider than the buckets* (~1.1 °C), and
   sub-48h markets belong to bots with direct NWS feeds. A clever pipeline cannot
   save a thesis with no alpha.

Full numbers and lessons: [`OPEN_SOURCE_ARCHIVE.md`](OPEN_SOURCE_ARCHIVE.md).

## Post-mortem cleanup (what the open-source version fixes)

After sunsetting, a fresh audit found the project was *still* wrong in ways the
original post-mortem missed, so the resolution layer never actually matched the
oracle, and the "no edge" verdict was measured through a distorted lens. The
public history is fixed so you fork from a correct base, not a broken one. The
exact as-it-died state is preserved at the git tag **`v1.0-as-it-died`**.

Twelve fixes (commit `5e07054`), each with regression tests:

| Area | Fix |
|------|-----|
| Resolution | Subzero buckets parse (signed regex); accept Fahrenheit-only METAR instead of silently using reanalysis; capture SPECI + the METAR `T`-group and 6-hr max group; round **half-up** (Wunderground) not banker's; bucket by the station's **local civil day**; hindcast shares the live parsing path |
| Signal | Fahrenheit range buckets integrate the correct round-half-up °C band (the old code added 1.0 °C to a converted bound, inflating YES probability) |
| Honesty | Backtester models spread/fees/fills and reports skill vs a climatology baseline instead of a flat fictional P&L; paper exits can realise losses; the "Brier" score is honestly renamed and statistically caveated |
| Docs | Station table corrected and **all 24 stations verified** against live market resolution URLs |

## How it works

1. Fetch forecasts from 6–8 weather models per city via the Open-Meteo API.
2. Apply **METAR-calibrated bias corrections** from hindcast snapshots.
3. Apply EMOS calibration (spread inflation, bias shrinkage, variance floor).
4. Compute a **Brier-weighted consensus** (better-scoring models get more weight).
5. Detect bust-causing weather patterns (Chinook, Foehn, marine layer, …).
6. Discover active Polymarket weather markets via the Gamma API.
7. **Claude (Meteorologist)**, physical plausibility, market-blind.
8. **Gemini (Risk Quant)**, execution cost / order-book risk, weather-blind.
   Both AI calls run in parallel.
9. Compute edge against market prices; apply risk controls (circuit breaker,
   correlation limits, model-agreement gate).
10. Resolve trades against **IEM METAR observations** (the Wunderground mirror).
11. Auto-redeem winners via the **Polymarket Relayer** (gasless).
12. Persist everything to SQL, trades, forecasts, AI decisions, fills.

## Resolution source (the most important detail)

Polymarket resolves daily-high temperature markets against the
**Weather Underground displayed value** at a specific airport METAR station. This
bot mirrors that exact source via **IEM ASOS** (Iowa Environmental Mesonet),
which serves the same raw METAR data.

- Celsius markets: `round_half_up(max(daily readings))`, whole degrees.
- Fahrenheit markets: `round_half_up(max(daily readings))`, whole degrees.
- Daily max is taken over the station's **local civil day**, includes SPECI
  reports and the METAR 6-hour max-temp group.
- **Hong Kong exception**: resolves from the HK Observatory "Absolute Daily Max"
  (`data.weather.gov.hk`), not an airport METAR.

All 24 station codes below were verified (2026-05) against each market's live
Wunderground/HKO resolution URL. Note Polymarket's *precipitation* markets for the
same cities use different sources (NYC→Central Park, London→Heathrow, Seoul→KMA),
do not reuse these codes for precip.

## Repository layout

```
src/weather_edge/
  fetchers/      Open-Meteo, Polymarket (Gamma), METAR/IEM, gribstream
  analysis/      consensus, bias_correction, edge, resolver, backtester,
                 pattern_detector, claude/gemini reasoning, learner, risk_controls
  trading/       paper trading, executor, fills, fees, portfolio sync
  dashboard/     FastAPI status dashboard
  scheduler.py   the main cycle (fetch → consensus → edge → trade → resolve)
  config.py      24 cities: ICAO station, timezone, unit, model weights
scripts/         hindcast/bias-table builders, ops helpers
tests/           pytest suite (164 tests)
OPEN_SOURCE_ARCHIVE.md   the post-mortem (read this)
```

## Quickstart

Requires Python ≥ 3.11.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # add ,dashboard,execution as needed
cp .env.example .env               # fill in keys; .env is gitignored

# Run the test suite (no network/keys needed)
python -m pytest tests/ -q

# Inspect what it would do, read-only, paper only
python -m weather_edge forecast    # consensus forecasts per city
python -m weather_edge markets     # discovered Polymarket weather markets
python -m weather_edge run --days 2  # one paper cycle
```

`AI reasoning` is degraded without `ANTHROPIC_API_KEY` / `GEMINI_API_KEY`, but the
core pipeline, tests, and backtester run without them. Trading is **paper by
default**; live execution requires `[execution]` deps and wallet credentials and
is **not recommended** (see the status banner).

### The backtester is honest now, read its caveats

There is **no historical Polymarket order-book data** in this repo, so the
backtester cannot produce a real track record. It reports two separate things:
**forecast skill** (hit rate, MAE, vs a climatology baseline, real) and
**illustrative P&L under explicit, stated costs** (spread, fee, fill probability).
With a fair entry price and zero costs, expected P&L is exactly zero, so any loss
it shows is the cost of the spread. That is the entire lesson. The result dict
echoes its `cost_assumptions` and a list of `caveats`; do not read its P&L as a
track record.

## Configuration (bring your own keys)

All configuration is environment variables, loaded from a gitignored `.env` (copy
[`.env.example`](.env.example) and fill it in). `.env.example` documents every
variable; this is the short version of what to get and where.

**Nothing is required** to run the tests or the read-only `forecast` / `markets`
commands. Add keys to unlock more:

| Variable | Needed for | Where to get it |
|----------|-----------|-----------------|
| `ANTHROPIC_API_KEY` | Claude "Meteorologist" review | <https://console.anthropic.com/> → API Keys |
| `GEMINI_API_KEY` | Gemini "Risk Quant" review | <https://aistudio.google.com/app/apikey> |
| `OPENMETEO_API_KEY` + `OPENMETEO_PAID_TIER=true` | Faster, parallel model fetches (free tier works without) | <https://open-meteo.com/en/pricing> |
| `GRIBSTREAM_API_KEY` | Extra AI models (GraphCast/AIFS) | <https://gribstream.com> |
| `DATABASE_URL`, `REDIS_*` | Live scheduler + dashboard persistence | your own Postgres/Redis |

**Live trading (not recommended).** To place real orders you also need Polymarket
credentials: `POLYMARKET_PRIVATE_KEY` (wallet key, the one true secret here,
controls real funds), `POLYMARKET_WALLET`, `POLYMARKET_SIGNATURE_TYPE`, the CLOB
`POLYMARKET_API_KEY`/`_SECRET`/`_PASSPHRASE` (create at
<https://polymarket.com/settings>), and `POLYMARKET_RELAYER_API_KEY` for gasless
redemption.

> **Security:** `.env` is gitignored, keep it that way; never commit real keys.
> Use a dedicated wallet with limited funds for any live experiment, and rotate
> any key you suspect has leaked. There are **no secrets in this repository's
> history**, it was scanned before publication; keep it that way.

## Cities (24)

ICAO/station codes are the ones the **temperature-high** markets resolve against
(verified against live Wunderground/HKO URLs).

| City | Station | City | Station | City | Station |
|------|---------|------|---------|------|---------|
| New York | KLGA | London | EGLC | Seoul | RKSI |
| Chicago | KORD | Madrid | LEMD | Tokyo | RJTT |
| Dallas | KDAL | Munich | EDDM | Hong Kong | HKO (`45005`) |
| Houston | KHOU | Warsaw | EPWA | Shanghai | ZSPD |
| Atlanta | KATL | | | Shenzhen | ZGSZ |
| Miami | KMIA | | | Buenos Aires | SAEZ |
| Denver | KBKF | | | Wellington | NZWN |
| Seattle | KSEA | | | Lucknow | VILK |
| Los Angeles | KLAX | Austin | KAUS | Toronto | CYYZ |
| San Francisco | KSFO | | | | |

> Houston is **KHOU** (Hobby), Denver is **KBKF** (Buckley), Hong Kong is the
> **HK Observatory**, the three stations the original run got wrong.

## Lessons (for the next person)

1. **Verify the resolution source end-to-end before depositing a dollar**, the
   exact station, exact rounding, exact URL, for *every* market, not one.
2. **Backtests without fees, spread, slippage, fill probability and market impact
   are fiction.** You will tune to beat the liar, not the market.
3. **A discrepancy is a stop-everything event.** Day-one P&L mismatched reality by
   $440 and the run continued. Layers of error compound.
4. **Don't spread across adjacent buckets.** Most legs lose by construction.
5. **Speed beats cleverness on short-horizon markets.** Without direct exchange/NWS
   feeds you can't win sub-48h.
6. **A clever architecture won't save a flawed thesis.** Prove the edge first;
   build the infrastructure second.

## License

Provided as-is for educational purposes. See [`OPEN_SOURCE_ARCHIVE.md`](OPEN_SOURCE_ARCHIVE.md).
