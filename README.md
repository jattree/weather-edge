# Weather Edge

An automated trading bot for Polymarket's daily-high-temperature markets. It
bias-corrects a multi-model weather forecast consensus against the same METAR
station data Polymarket resolves on, and bets where its probability disagrees
with the crowd's price.

> ### ⚠️ Status: sunset. This bot lost money. Read before forking.
>
> Over a live proving run it went **$210 → $51.61 (-75.4%)** and was retired.
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

## Cleanup since retirement

The public code is not the code that lost the money. Three review passes
after the bot was retired fixed the resolution layer, the signal math, the AI
review gate, live-trading safety and the paper-trading accounting, each with
regression tests. The biggest finding was that the proving run never
actually tested the thesis: every day of it traded through at least one
broken layer. So the honest verdict is "edge **unproven**", not "edge
disproven". What was fixed and when: [`docs/CLEANUP_HISTORY.md`](docs/CLEANUP_HISTORY.md).
The exact as-it-died state is preserved at the git tag **`v1.0-as-it-died`**.

## How it works

1. Fetch forecasts from 6-8 weather models per city via the Open-Meteo API.
2. Apply **METAR-calibrated bias corrections** from hindcast snapshots.
3. Apply EMOS calibration (spread inflation, bias shrinkage, variance floor).
4. Compute a **Brier-weighted consensus** (better-scoring models get more weight).
5. Detect bust-causing weather patterns (Chinook, Foehn, marine layer, …).
6. Discover active Polymarket weather markets via the Gamma API.
7. **Claude (Meteorologist)**: physical plausibility, market-blind.
8. **Gemini (Risk Quant)**: execution cost / order-book risk, weather-blind.
   Both AI calls run in parallel.
9. Compute edge against market prices; apply risk controls (circuit breaker,
   correlation limits, model-agreement gate).
10. Resolve trades against **IEM METAR observations** (the Wunderground mirror).
11. Auto-redeem winners via the **Polymarket Relayer** (gasless).
12. Persist everything to a local SQLite file: trades, forecasts, AI decisions, fills.

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

## Quickstart

Requires Python ≥ 3.11. Nothing needs a key to run the tests or the read-only
commands.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,dashboard]"   # add ,execution for the trading code
cp .env.example .env                # fill in keys; .env is gitignored

python -m pytest -q                 # no network or keys needed
python -m weather_edge forecast     # consensus forecasts per city
python -m weather_edge markets      # discovered Polymarket weather markets
python -m weather_edge run          # one paper cycle
python -m weather_edge dashboard    # http://127.0.0.1:8000
python -m weather_edge log-prices --interval 15   # record prices to prices.db, never trades
```

- **Paper by default.** Live execution needs the `[execution]` extra, wallet
  credentials and `LIVE_MODE=true`, and is **not recommended**.
- **AI review gates trading.** Without `ANTHROPIC_API_KEY`, and with
  `REQUIRE_AI_REVIEW=true` (the default), the pipeline runs but places no trades.
- **The dashboard has no authentication.** It binds to `127.0.0.1`; don't expose
  it on a public host.
- **The backtester is not a track record.** There is no historical order-book
  data here, so it reports forecast skill (against a climatology baseline) and
  illustrative P&L under stated costs. Read its `caveats` field.

## Configuration

Everything is an environment variable, loaded from `.env`.
[`.env.example`](.env.example) documents every one. The main keys:

| Variable | Needed for |
|----------|-----------|
| `ANTHROPIC_API_KEY` | Claude "Meteorologist" review |
| `GEMINI_API_KEY` | Gemini "Risk Quant" review |
| `OPENMETEO_API_KEY` | Faster, parallel model fetches (the free tier works without one) |
| `GRIBSTREAM_API_KEY` | Extra AI weather models (GraphCast/AIFS) |
| `REDIS_*` | Live scheduler cache, kill switch, heartbeats |
| `POLYMARKET_*` | Live trading only. `POLYMARKET_PRIVATE_KEY` controls real funds |

Never commit `.env`. Use a dedicated wallet with limited funds for any live
experiment. Report security issues privately via [`SECURITY.md`](SECURITY.md).

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

1. **Verify the resolution source end-to-end before depositing a dollar**: the
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

MIT. See [`LICENSE`](LICENSE) and [`OPEN_SOURCE_ARCHIVE.md`](OPEN_SOURCE_ARCHIVE.md).

> **Disclaimer:** This software places orders that move real funds on
> Polymarket. It is provided for educational and research purposes only, with
> no warranty of any kind, and is **not financial advice**. You are solely
> responsible for any trades it makes. Use a dedicated wallet with limited
> funds, and run in paper mode until you understand the behavior. Use at your
> own risk.
