# Weather Edge, Open Source Archive

**Frozen state: 2026-04-07 19:00 UTC**
**Final result: $210 deposited → $51.61 portfolio = -75.4% drawdown**
**Status: Failed proving run. Sunsetting to open source as a cautionary tale.**

This document pins the project state at the moment we accepted the strategy
had not proven its edge and pivoted to "let it ride to zero, then publish."
Anyone reading this repo later, that is the context.

---

## What this project was

An automated weather prediction market trading bot for Polymarket. The
hypothesis was that multi-model weather forecast consensus, calibrated against
the Wunderground/METAR resolution oracle, would produce a persistent edge over
crowd-implied prices on 24 cities' daily-high temperature markets.

It used:

- 6-8 weather models per city via Open-Meteo customer API
- METAR-calibrated bias correction from IEM ASOS hindcast snapshots
- EMOS calibration (1.3x spread inflation, 0.9 bias shrinkage, 0.7°C variance floor)
- Adaptive Brier-weighted consensus
- 13 bust-pattern detectors (Chinook, Foehn, marine layer, etc.)
- Claude Sonnet 4 as a market-blind "Meteorologist"
- Gemini 2.5 Flash as a weather-blind "Risk Quant" red team
- A swing-bot execution layer (48h+ horizon, taker/maker hybrid, $5 min size)
- Three-tier exit system (2-min emergency, 30-min AI-reviewed, stale model detection)
- Auto-resolution against IEM METAR observations
- Auto-redemption via Polymarket Relayer V2 (gasless)

By the numbers it was an ambitious build. By the P&L it didn't matter.

---

## Final scoreboard

| Metric | Value |
|---|---|
| Deposited | $210.00 |
| Cash on hand | $21.15 |
| Position market value | $30.46 |
| Total portfolio | **$51.61** |
| Total drawdown | **-$158.39 (-75.4%)** |
| Live winners (still open) | 4/19 (London No, BA No, Wellington No, Miami 76-77 Yes) |
| Live losers (still open or already lost) | 15/19 (Houston ×4, Seattle ×2, HKG ×2, Miami ×2, Denver, Shenzhen, Munich, NYC, Toronto No, Houston tail) |
| First live order | 2026-03-31 05:13 UTC |
| Decision made to sunset | 2026-04-06 |

---

## Why it failed

The bot found "edges" that weren't real. We can identify three layered
mistakes, each one of which would have been caught by a 30-minute audit
before depositing money:

### 1. Wrong data source (caught Apr 2)
Backtests and paper trading resolved trades against Open-Meteo's gridded
reanalysis archive. Polymarket resolves against Wunderground, which is the
displayed METAR sensor reading at a specific airport. The two sources diverge
by ~0.9°C MAE, and 67% of trades resolved to a different whole-degree bucket.
The +$8,471 paper P&L the bot accumulated before going live was almost
entirely fiction. We tuned the strategy for weeks against a liar.

### 2. Wrong stations (caught Apr 3)
After fixing the data source, three of 24 cities still resolved against the
wrong ICAO station entirely:
- Denver: KDEN (the international airport) when Polymarket actually uses KBKF (Buckley SFB), 15 miles east at different elevation
- Houston: KIAH (Bush Intercontinental, north Houston) when Polymarket uses KHOU (Hobby, south Houston), with a single-day gap of 1.7°C, guaranteed loss on every Houston trade
- Hong Kong: VHHH (Chek Lap Kok airport) when Polymarket uses HK Observatory (45005), a different climate zone entirely. Required a custom scraper against weather.gov.hk because the station isn't on standard METAR networks.

### 3. Wrong execution structure (caught Apr 4)
Even with the correct data and stations, the bot structurally bled by buying
multiple adjacent temperature buckets per city as YES bets. Buying Yes on
3-4 buckets means 2-3 guaranteed losses per win, and the losers outpace the
winners. We pivoted to "one bet per city, tail-No priority", but by then
the bankroll was already cut in half.

### 4. The deeper failure
Even after all three fixes, the proving run failed. The model continued to
lose. The probable explanations:
- The model's edge is real on consensus (modal-bucket No bets won 4/4) but
  fictional on tails (penny-Yes bets lost 12/15)
- Polymarket weather markets are dominated by faster bots with direct NWS
  feeds. Open-Meteo's 1-2 hour data lag is fatal at sub-48h horizons
- We never rigorously backtested with real fees, real slippage, real spreads,
  real fill probabilities, or real market impact. Backtests without those
  things are fiction
- A few confident, high-conviction trades cannot recover 75% drawdown, the
  math doesn't work

---

## Lessons (for whoever reads this later)

These are the rules I'd impose on my own future self before depositing a
single dollar into another prediction-market bot:

1. **Verify the resolution source against the actual oracle, end-to-end, before depositing money.** Not just "we use METAR." Verify the *exact station*, the *exact rounding*, the *exact source URL*, and that the values you see match the values the market resolves on. Do this for *every single market* you intend to trade, not just one.

2. **Backtests without fees, slippage, spread cost, fill probability, and market impact are fiction.** You will tune your strategy to win against the liar, not against reality. This was the single biggest mistake.

3. **Discovering a discrepancy is a stop-everything event.** When the day-one P&L mismatched reality by $440 ($380 paper vs -$60 live), we patched one layer and kept trading. The right move was to halt all trading and audit comprehensively. Layers of error compound.

4. **Multi-bucket spreading on the same market is a structural loser.** If you're buying Yes on 3 adjacent temperature buckets, 2 of them will lose every single time. The losers outpace the winners. Pick one bet per market or take the high-probability No tail.

5. **Speed matters more than cleverness on short-horizon markets.** If you don't have direct exchange data feeds, you cannot win against bots that do. Sub-48h weather markets belong to the bots with NWS PRIVATE_KEY direct feeds, not to Open-Meteo polling on a 30-minute cycle.

6. **A clever architecture will not save a flawed thesis.** This codebase has every bell and whistle: dual-AI red teaming, Brier-weighted consensus, pattern detectors, calibrated EMOS, three-tier exits, gasless redemption, 128 unit tests. None of it mattered because the underlying alpha didn't exist. Build the strategy proof first; build the infrastructure second.

7. **"Almost working" is not a fix.** Auto-redeem went through six iterations of "it's working now" before any position was actually claimed. Don't declare a fix done until you've watched real money move.

---

## What happens next

After the hail-mary attempt described in the next section, this codebase is
being open-sourced as a portfolio piece and a public cautionary tale. The
long-form blog post and the GitHub release will both link to this archive
document so that the next person to attempt this kind of bot starts with the
complete failure log, not just the architecture diagrams.

The code itself is decent and could be useful to someone, the data fetchers,
the consensus engine, the dual-AI pattern, the dashboard, the swing-bot
execution layer, the Polymarket Relayer integration. Take what's useful, leave
what isn't, and please learn from the mistakes documented above.

---

## Final hail-mary attempt (2026-04-07)

Before sunsetting, the remaining ~$51 is being deployed as **penny lottery
tickets** with all safety rails removed. The reasoning:

- The "medium-conviction" strategy that got us here is a known loser
- Tail-Yes bets at 1-5¢ pay 20-100x if hit, a basket of 30+ at $1 each
  costs the same as 6 medium positions but gives many more chances at a
  multi-x outcome
- Empirically, the model has been right about the modal forecast (No-tail
  bets won 4/4) but wrong about tails, but tails by definition cost
  almost nothing, so wrong-but-cheap is acceptable
- Worst case: we lose the remaining $51 a week earlier than we would have
  by bleeding it through more medium-sized bets
- Best case: one lottery ticket hits, the portfolio doubles or triples,
  and the "decision to sunset" gets reversed

The rails removed for the hail mary:

- Horizon filter (was 36h+, now 0h, take any market)
- Model agreement gate (was std<2.0, now disabled)
- One-bet-per-city dedupe (was on, now off, let the bot stack the cheapest buckets)
- LOW-tier signal filter (was filtered out, now allowed)
- Max trades per cycle (was 6, now 999)
- Claude SKIP veto (was active, now ignored)
- Gemini DISSENT downsizing (was active, now ignored)
- USDC floor (was $20, now $0)
- Max positions cap (was 50, now 999)
- Per-trade min edge (was 2%, now 0.1%)
- Yes-exposure cap (was active, now bypassed)
- Correlation limit (was active, now bypassed)
- Gross exposure cap (was active, now bypassed)
- Position size override: every approved signal becomes a $1.00 fixed-size order

If this works, we'll know within 48 hours. If it doesn't, the run-to-zero
becomes the closing chapter and the open-source release goes out as planned.

---

*Frozen 2026-04-07. Do not edit this file. It is the closing snapshot.*
