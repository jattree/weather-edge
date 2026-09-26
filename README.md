# Weather Edge

An automated trading bot for Polymarket's daily-high-temperature markets. It
bias-corrects a multi-model weather forecast consensus against the same METAR
station data Polymarket resolves on, and bets where its probability disagrees
with the market's price.

> ### ⚠️ Status: retired. The market beats it. Do not trade it.
>
> A live proving run went **$210 → $51.61 (-75.4%)**. After every bug was
> fixed, the forecasts were scored against a month of real Polymarket prices:
> **the market's probabilities were better on every measure, and trading on
> the model lost money.** An AI decision model (Jev) given the same inputs did
> worse still.
>
> It is published as a measured negative result and a reference
> implementation of the moving parts: METAR-faithful resolution, multi-model
> consensus, a price logger and scoring harness, dual-AI review, and paper
> trading. The as-it-died code is at the git tag **`v1.0-as-it-died`**.

---

## What we found

Measured in September 2026, after the fixes in
[`docs/CLEANUP_HISTORY.md`](docs/CLEANUP_HISTORY.md). Full numbers and method:
[`docs/FINDINGS_2026-09.md`](docs/FINDINGS_2026-09.md).

**1. Forecast error is about one bucket wide at the horizon that matters.**
Polymarket lists these markets only about two days ahead. Two days out, the
raw model consensus misses the station's high by 1.33 °C on average. Raw
models run about 0.8 °C colder than airport sensors, and correcting that
with the previous 30 days' errors brings it to 0.98 °C, still roughly the
width of a bucket.

**2. The market already knows about that bias, and forecasts better.**
Over 697 events scored 36 hours before close, using only information
available at that moment:

| | Our model | Jev | Market |
|---|---|---|---|
| Log-loss (lower is better) | 1.59 | 1.95 | **1.31** |
| Probability on the winning bucket | 25.7% | 26.0% | **32.8%** |
| Best blend with the market | 0% model | 0% Jev | 100% market |

The market's implied temperature is off by only about 0.2 °C, against 0.8 °C
for raw models. Simulated trades at market prices plus a 1¢ half-spread and
the taker fee lost **13.9% ± 5.3% per trade** (1,905 trades).

**3. Jev, TypeSafe AI's decision model, does worse than plain statistics.**
Given the same numbers (and no city, date or prices, so it could not recall
outcomes), `jev-1.13.0` picked the winning bucket 30% of the time against the
market's 46%, and any weight on it made a market blend worse. It is built to
choose well among valid actions, not to turn weather-model output into
calibrated probabilities. The full run cost three cents.

**Verdict.** The original post-mortem's conclusion, no edge, was right. Its
reasoning was partly wrong: the proving-run losses were mostly software bugs
(see below), and "error wider than a bucket" was never the real argument.
The real one is that the market prices these buckets better than anything
built from the same public weather data.

## Why the proving run lost

Most of the -75% came from bugs that would lose money with any forecast:

- Live exits compared outcome sides with the wrong casing, so the bot sold
  winners and held losers; NO tokens were sold at the YES price.
- Failed order cancels were ignored, so replacements doubled exposure.
- Claude's SKIP never blocked a trade, and a failed AI review counted as an
  approval.
- Forecasts used the UTC day while markets settle on the local day, three
  stations were wrong, and backtests resolved against gridded reanalysis
  instead of the METAR value markets pay out on.
- For much of the run, the "hail-mary" configuration had no risk rails.

The original post-mortem is preserved in
[`OPEN_SOURCE_ARCHIVE.md`](OPEN_SOURCE_ARCHIVE.md).

## Lessons (for the next person)

1. **Score against the market's prices before building anything else.**
   Logging prices for a few weeks and computing log-loss would have answered
   this project's question before the first trade. Forecast accuracy on its
   own says nothing about edge.
2. **Verify the resolution source end to end** for every market: the exact
   station, the rounding rule, the local day, the URL.
3. **Backtests without fees, spread and fills are fiction.**
4. **A discrepancy is a stop-everything event.** Day-one P&L was off by $440
   and the run continued.
5. **An AI decision layer is not a forecaster.** An LLM veto or a decision
   model cannot add information the inputs don't contain, and it fails
   silently unless it fails closed.
6. **A clever architecture won't save a thesis with no edge.** Prove the edge
   first; build the infrastructure second.

## How it works

1. Fetch forecasts from 6 to 8 weather models per city (Open-Meteo), on the
   station's local day.
2. Apply METAR-calibrated bias corrections behind a significance gate, then
   EMOS calibration and a skill-weighted consensus.
3. Detect bust-prone patterns (Chinook, Foehn, marine layer).
4. Discover active markets (Polymarket Gamma API) and compute bucket
   probabilities with the resolver's own bucket rules.
5. Review each signal with Claude (weather, market-blind) and Gemini
   (execution risk, weather-blind). Vetoes block; failed reviews fail closed.
6. Apply risk controls (horizon filter, agreement gate, exposure caps,
   circuit breaker, kill switch), then trade on paper, or live if explicitly
   enabled.
7. Resolve against IEM METAR observations and redeem winners through the
   Polymarket Relayer.

## Resolution source (the most important detail)

Polymarket resolves daily-high markets against the **Weather Underground
displayed value** at a specific airport station. The bot mirrors it through
**IEM ASOS**, which serves the same METAR data.

- The daily max is taken over the station's **local civil day**, including
  SPECI reports and the METAR 6-hour max group, rounded half up to whole
  degrees in the market's unit.
- **Hong Kong** resolves from the HK Observatory "Absolute Daily Max"
  (`data.weather.gov.hk`); a label L covers [L, L+1) on its 0.1 °C value.
- Precipitation markets for the same cities use different stations; do not
  reuse these codes for them.

## Quickstart

Requires Python ≥ 3.11. The tests and read-only commands need no keys.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,dashboard]"   # add ,execution for the trading code
cp .env.example .env                # fill in keys; .env is gitignored
git config core.hooksPath .githooks # optional: the pre-commit checks

python -m pytest -q                 # no network or keys needed
python -m weather_edge forecast     # consensus forecasts per city
python -m weather_edge markets      # active Polymarket weather markets
python -m weather_edge run          # one paper cycle
python -m weather_edge dashboard    # http://127.0.0.1:8000
```

- **Paper by default.** Live execution needs the `[execution]` extra, wallet
  credentials and `LIVE_MODE=true`. Given the findings above, don't.
- **AI review gates trading.** Without `ANTHROPIC_API_KEY`, and with
  `REQUIRE_AI_REVIEW=true` (the default), the pipeline runs but places no
  trades.
- **The dashboard has no authentication.** It binds to `127.0.0.1`; don't
  expose it publicly.

### Reproducing the findings

```bash
python -m weather_edge backfill-prices --days 35   # price history + outcomes -> prices.db
python -m weather_edge log-prices --interval 15    # keep recording prices and spreads (never trades)
python scripts/compare_day_basis.py --lead 2       # forecast error: UTC vs local day, bias correction
python scripts/score_vs_market.py --horizon 36     # model vs market: log-loss, blend, trade simulation
python scripts/score_jev.py --max-usd 0.10         # Jev vs market (needs TYPESAFE_API_KEY)
```

The exchange keeps price history for resolved markets for only about a month,
so a backfill can only reach back that far; run the logger to keep more.

## Configuration

Everything is an environment variable, loaded from `.env`.
[`.env.example`](.env.example) documents every one. The main keys:

| Variable | Needed for |
|----------|-----------|
| `ANTHROPIC_API_KEY` | Claude "Meteorologist" review |
| `GEMINI_API_KEY` | Gemini "Risk Quant" review |
| `OPENMETEO_API_KEY` | Faster model fetches (the free tier works without one) |
| `GRIBSTREAM_API_KEY` | Extra AI weather models (GraphCast/AIFS) |
| `TYPESAFE_API_KEY` | `scripts/score_jev.py` only |
| `REDIS_*` | Live scheduler cache, kill switch, heartbeats |
| `POLYMARKET_*` | Live trading only. `POLYMARKET_PRIVATE_KEY` controls real funds |

Never commit `.env`. Report security issues privately via
[`SECURITY.md`](SECURITY.md).

## Cities (24)

Station codes are the ones the **daily-high** markets resolve against,
verified against each market's live Wunderground/HKO resolution URL.

| City | Station | City | Station | City | Station |
|------|---------|------|---------|------|---------|
| New York | KLGA | London | EGLC | Seoul | RKSI |
| Chicago | KORD | Madrid | LEMD | Tokyo | RJTT |
| Dallas | KDAL | Munich | EDDM | Hong Kong | HKO (`45005`) |
| Houston | KHOU | Warsaw | EPWA | Shanghai | ZSPD |
| Atlanta | KATL | Toronto | CYYZ | Shenzhen | ZGSZ |
| Miami | KMIA | Austin | KAUS | Buenos Aires | SAEZ |
| Denver | KBKF | Los Angeles | KLAX | Wellington | NZWN |
| Seattle | KSEA | San Francisco | KSFO | Lucknow | VILK |

Houston is **KHOU** (Hobby), Denver is **KBKF** (Buckley) and Hong Kong is the
**HK Observatory**: the three stations the original run got wrong.

## License

MIT. See [`LICENSE`](LICENSE).

> **Disclaimer:** This software can place orders that move real funds on
> Polymarket. It is provided for educational and research purposes only, with
> no warranty of any kind, and is **not financial advice**. You are solely
> responsible for any trades it makes.
