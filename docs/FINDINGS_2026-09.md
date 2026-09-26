# Findings, 2026-09: does the forecast beat the market?

The original post-mortem concluded there was no edge, but the proving run it
rested on traded through several broken layers, so it could not show that.
After the code was fixed (see [`CLEANUP_HISTORY.md`](CLEANUP_HISTORY.md)),
three measurements settle it. Each script below reproduces its numbers.

## 1. Forecast error by day basis, lead time and bias correction

`scripts/compare_day_basis.py`, 90 days (2026-06-25 to 2026-09-22), 24 cities,
against the METAR (HKO for Hong Kong) observations the markets resolve on.
"Corrected" subtracts each city's mean error over the previous 30 days only,
so it uses no future information.

| Forecast issued | Raw error | Raw right bucket | Corrected error | Corrected right bucket |
|---|---|---|---|---|
| Same day (short lead) | 1.02 °C | 24% | 0.62 °C | 40% |
| 1 day ahead | 1.20 °C | 21% | 0.85 °C | 32% |
| 2 days ahead | 1.33 °C | 20% | 0.98 °C | 29% |

- **Local vs UTC day matters little.** Bucketing forecasts by UTC day (a bug
  in the proving run) adds about 6% error overall. It is large only for
  Seattle.
- **Station bias is real and large.** Raw model consensus runs about 0.8 °C
  colder than the airport sensors at every lead time. Correcting it cuts
  error by 26% to 39%, but some cities (New York, Dallas, Warsaw, Lucknow)
  do not improve.
- **At the traded horizon, error is about one bucket wide even after
  correction.** Polymarket lists these markets only about two days ahead, so
  the 2-day row is the one that matters. The post-mortem's ~1.4 °C matched the
  raw 2-day error.

## 2. The model against the market's prices

`weather_edge backfill-prices` stored hourly price history and outcomes for
every resolved market from 2026-08-25 to 2026-09-25 (743 city-days, 9,086
markets; the exchange keeps about a month). `scripts/score_vs_market.py`
scores each event at a decision time, using only information available then:
day-ahead model runs (lead 2 for 36 h, lead 1 for 24 h), the trailing 30-day
bias and error, and a Normal distribution integrated over each bucket's
resolution band. Market probabilities are the last traded prices at that
time, normalised across the event.

| | 36 h before close | 24 h before close |
|---|---|---|
| Probability on the winning bucket, model / market | 25.7% / **32.8%** | 29.5% / **35.4%** |
| Log-loss, model minus market (positive: market better) | **+0.28 ± 0.03** | **+0.22 ± 0.03** |
| Implied-mean error, model / market | 0.97 / **0.81 °C** | 0.81 / **0.76 °C** |
| Implied-mean bias, model / market | -0.09 / -0.23 °C | -0.06 / -0.21 °C |
| Best model/market blend | 100% market | 100% market |
| Simulated trades (5% edge, 1¢ half-spread, taker fee) | 1,905, **-13.9% ± 5.3%** each | 1,791, **-6.4% ± 5.5%** each |

- **The market already prices the station bias.** Its implied mean is off by
  only about 0.2 °C, against 0.8 °C for raw models.
- **The market is better on every measure,** including the point estimate,
  and any weight on the model makes a blend worse. The gap is about ten
  standard errors.
- **Trading on the model loses,** even with a 1¢ half-spread; live snapshots
  show 0 to 5¢.

## 3. Jev (TypeSafe AI's decision model) against the market

`scripts/score_jev.py` asks `jev-1.13.0` which bucket will resolve YES for the
same 697 events and decision time (36 h), from the same numbers: the day-2
model highs, the city's recent bias and error, and the last 10 days of
forecast vs observed. Jev never sees the city, date or prices, so it cannot
recall outcomes. The run cost $0.03.

| | Jev | Model (Normal) | Market |
|---|---|---|---|
| Log-loss | 1.95 | 1.59 | **1.31** |
| Log-loss, best smoothing toward uniform | 1.88 | | |
| Probability on the winning bucket | 26.0% | 25.7% | **32.8%** |
| Top pick was the winner | 30% | | **46%** |
| Average miss, in buckets | 1.05 | | **0.67** |
| Winner given under 1% | 5% of events | 3% | 0% |

- **Jev does worse than the plain statistical model** on the same inputs,
  and smoothing its probabilities barely helps, so it is not just
  overconfident: it extracts less from the numbers.
- **Any weight on Jev makes a market blend worse.** A 20-event pilot looked
  close to the market; the full run showed that was chance.
- Jev is a general decision model that picks well from a menu of valid
  actions. Turning weather-model output into calibrated probabilities is a
  statistics problem, and the market solves it better than either.

## Verdict

The post-mortem's conclusion stands, now measured: there is no edge in
public weather-model data for these markets, whether it is processed
statistically or by an AI decision model. Its reasoning was partly wrong.
The proving-run losses were mostly software bugs, and "error wider than a
bucket" was never the real argument. The real one is that the market's
probabilities are better calibrated than anything built from the same
public inputs.

### Limits

- One month of prices, late summer only.
- A simple model: a Normal around a bias-corrected consensus. Serious
  station post-processing (for example, machine-learned corrections on
  ensembles) might narrow a 0.28-nat gap, but that is a large gap.
- One Jev prompt design. The calibration check suggests the gap is not a
  formatting problem.
- Historical prices are last trades, not bid/ask; the simulation assumes a
  half-spread. `weather_edge log-prices` records real spreads going forward.
