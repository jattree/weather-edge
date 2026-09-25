# Cleanup history

What was fixed after the bot was retired, in three waves. Each wave found
problems the previous one missed. The exact as-it-died state is preserved at
the git tag `v1.0-as-it-died`; the post-mortem is
[`OPEN_SOURCE_ARCHIVE.md`](../OPEN_SOURCE_ARCHIVE.md).

## First cleanup wave (2026-05)

After sunsetting, a fresh audit found the project was *still* wrong in ways the
original post-mortem missed, so the resolution layer never actually matched the
oracle, and the "no edge" verdict was measured through a distorted lens. The
public history is fixed so you fork from a correct base, not a broken one. The
exact as-it-died state is preserved at the git tag **`v1.0-as-it-died`**.

Twelve fixes (commit `9262e5e`), each with regression tests:

| Area | Fix |
|------|-----|
| Resolution | Subzero buckets parse (signed regex); accept Fahrenheit-only METAR instead of silently using reanalysis; capture SPECI + the METAR `T`-group and 6-hr max group; round **half-up** (Wunderground) not banker's; bucket by the station's **local civil day**; hindcast shares the live parsing path |
| Signal | Fahrenheit range buckets integrate the correct round-half-up °C band (the old code added 1.0 °C to a converted bound, inflating YES probability) |
| Honesty | Backtester models spread/fees/fills and reports skill vs a climatology baseline instead of a flat fictional P&L; paper exits can realise losses; the "Brier" score is honestly renamed and statistically caveated |
| Docs | Station table corrected and **all 24 stations verified** against live market resolution URLs |

## Second cleanup wave (2026-06)

A later independent review ([`docs/PROVING_RUN_REVIEW.md`](PROVING_RUN_REVIEW.md))
concluded that the proving run never actually tested the thesis (every day of
it traded through at least one broken layer), so the honest verdict is
"edge **unproven**", not "edge disproven". It also found the published repo
still contradicted its own lessons. Fixed, by area:

| Area | Fix |
|------|-----|
| Safety | The rail-less **hail-mary configuration was the shipped default** (no horizon filter, no agreement gate, no dedupe, AI vetoes ignored, no exposure caps, exit monitor dead-gated). All rails restored; the hail-mary is preserved verbatim behind `HAIL_MARY_MODE` (default off, never enable with real money) |
| Paper honesty | Paper fills crossed nothing and paid nothing (the lesson-2 fiction, again). Paper entries now cross the spread and pay the dynamic taker fee; early exits pay a taker fee on proceeds |
| Signal | The dynamic bias correction now applies only when the measured bias clears a 2-standard-error significance gate (a uniform correction damages already-good cities: the 2026-04-01 validation of the Layer-2 station offsets made London's MAE 302% worse); Kelly sizes at the effective fill price (mid + half-spread) instead of the frictionless midpoint; the HIGH-tier spread gate is real instead of a hardcoded `True` |

## Third cleanup wave (2026-09)

A full code review found that several second-wave fixes existed in the
commit history but not in the behaviour, and that more layers were broken
than either earlier review knew. This strengthens the "unproven" verdict:
the proving run's numbers came from a pipeline with more wrong layers than
the post-mortem lists. Fixed, by area, each with regression tests:

| Area | Fix |
|------|-----|
| AI review | A Claude SKIP never blocked a trade: the vetoed signal stayed in the execution list. Unparseable or failed reviews counted as approval, a Gemini "skip" (size 0) was bumped up to the $5 live minimum, and sniper and refresh cycles skipped review entirely. Vetoes now remove the signal, reviews fail closed, vetoes are remembered for the day, and with `REQUIRE_AI_REVIEW=true` (default) nothing trades without an approval |
| Resolution | Forecasts used the UTC day while markets settle on the station's local day. Buckets worded "or higher" / "or lower" were never parsed. Hong Kong's resolver and probability model used different bucket boundaries. A failed METAR fetch quietly settled trades from reanalysis. Year rollover mis-dated December markets in January. All fixed; the backtester now scores with the resolver's own bucket rule |
| Signal | The 2-SE bias gate was defeated by duplicate snapshots (one per 30-minute cycle counted as independent samples). The KDE/empirical probability blend was never reached, so every probability was a plain Normal. The model-agreement gate was applied to the inflated spread. The spread estimate was always ~0 (it came from complementary prices); it now uses Gamma's best bid/ask |
| Live safety | The kill switch failed open on a Redis error. Live YES/NO sides were compared with the wrong casing, inverting exit logic and disabling the YES exposure cap. NO tokens were sold at the YES price with a slippage guard that could not fire. Refresh traded while trading was STOPPED. The circuit breaker never tripped the kill switch. One-off scripts traded live with no confirmation; they now dry-run unless given `--execute` with `LIVE_MODE` on |
| Paper honesty | Fees were lost on restart, losses never reduced deployable capital, and the spread-merge simulation overstated P&L |
| Hygiene | The quickstart `run` command crashed. The Gemini key leaked into logs through URL error text. Undeclared dependencies, a dead Postgres ORM layer, and undocumented settings were fixed; `run_cycle` (cyclomatic complexity 193) was split into stages, and a ruff plus complexity ratchet guards against regressions (see CONTRIBUTING.md) |
