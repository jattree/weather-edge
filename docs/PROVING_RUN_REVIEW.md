# Proving-Run Review (2026-06-11)

An independent post-sunset review of what the live proving run did and did not
prove, and the second cleanup wave it motivated. The original post-mortem is
[`OPEN_SOURCE_ARCHIVE.md`](../OPEN_SOURCE_ARCHIVE.md) (frozen 2026-04-07); the
first cleanup wave is commit `9262e5e` (the 12 correctness fixes). This
document records the conclusions of a fresh review of both, plus the fixes
that followed.

## Headline conclusion

The archive's verdict was *"the underlying alpha didn't exist; none of it
mattered."* The evidence does not support that strong form. What the evidence
supports is: **the alpha was never measured.** Every day of the proving run
traded through at least one (usually several) broken layers:

| Window (2026) | Broken layer live at the time |
|---|---|
| 03-31 → 04-02 | Resolution against Open-Meteo reanalysis instead of the Wunderground/METAR oracle |
| 03-31 → 04-03 | Wrong stations for Denver (KDEN→KBKF), Houston (KIAH→KHOU), Hong Kong (VHHH→HKO 45005) |
| 03-31 → 04-04 | Multi-bucket YES spreading (2-3 guaranteed losing legs per win) |
| 03-31 → 04-07 | All 12 bugs later fixed in `9262e5e`, including bug #3: Fahrenheit range buckets integrated ~0.8 °F too wide on the top edge, **systematically inflating YES probability on the most-traded market type** |
| 04-07 → end | Hail-mary mode: every safety rail deliberately removed, penny-YES only |

Bug #3 deserves emphasis: it manufactures the exact failure the post-mortem
attributed to the model ("edge fictional on tails, penny-Yes lost 12/15").
The bot was not necessarily wrong about the weather; it was integrating the
wrong bucket. It was found seven weeks *after* the sunset decision.

There is not a single day of the run where the thesis was tested under
correct plumbing. Nineteen trades under those conditions cannot establish
"no alpha." The only clean-ish signal, modal/consensus No bets, went 4/4
the *other* way (tiny sample; not proof either).

## What still stands from the original post-mortem

- **Stopping live trading was correct.** Risk discipline at -75% with an
  unproven edge is not a close call.
- **All seven lessons in the archive remain sound**, especially lesson 1
  (verify the oracle end-to-end), which, note, this team only fully achieved
  in `9262e5e`, after the money was gone.
- **Penny-YES tail buying is structurally bad** independent of bug #3:
  favorite-longshot bias overprices longshots, the dynamic taker fee eats
  thin edges, and the calibration treats 1-5% tail buckets identically to
  modal buckets.
- **The latency argument stands.** Open-Meteo's 1-2 h lag on a 30-minute poll
  cannot compete sub-48h against direct-feed bots. Any surviving edge lives
  at 48h+ horizons, No-side, maker-filled, which is where the 4/4 happened.

## Corrected conclusion

"The forecast edge was never proven" (the README banner) is exactly right.
"The alpha didn't exist" is not established. If the question ever matters
again, it is answerable for $0 of trading capital: a 4-8 week
resolution-faithful, cost-faithful **paper** run on the post-cleanup code,
restricted to 48h+ No-side strategies. The expected value of that experiment
is settling the question cheaply, not a likely path back to live trading,
because the structural headwinds (latency, fee regime) have not moved.

## The second cleanup wave (2026-06-11)

The review found the published repo still contradicted its own lessons in
four ways. Fixed, each with regression tests:

1. **The hail-mary configuration was the shipped default.** The repo told
   forkers "do not point real money at it" while shipping the deliberately
   rail-less 2026-04-07 lottery configuration as the only behavior: no
   horizon filter, no model-agreement gate, no dedupe, AI vetoes ignored, no
   exposure caps, exit monitor dead-gated with `if False`. All 13 rail sites
   now branch on `settings.hail_mary_mode` (default **off**); the hail-mary
   behavior is preserved verbatim behind the flag for the historical record.

2. **Paper trading still embodied lesson 2.** Paper fills executed at the
   Gamma midpoint with zero fees (the precise fiction that made the
   +$8,470 pre-launch paper P&L meaningless) while the real fee model
   (`fees.py`) and the per-signal spread sat unwired. Paper entries now
   cross the spread and pay the dynamic taker fee; early exits pay a taker
   fee on proceeds; `summary()` reports `total_fees`.

3. **The bias correction applied below the noise floor.** The Layer-1
   dynamic correction applied whenever 14 samples existed, even when the
   measured bias was statistically indistinguishable from zero. This is the
   uniform-correction failure the 2026-04-01 validation quantified (helped
   HKG +49%, damaged London -302%). Corrections now require
   |mean bias| > 2 standard errors.

4. **Kelly sizing ignored execution costs, and the spread gate was a
   placeholder.** Kelly now prices the entry at mid + half-spread (it was
   over-sizing every bet), and the HIGH-tier `spread_ok` flag, hardcoded
   `True` since launch, now applies the same 40%-of-alpha convention as
   the fee gate.

## Provenance

Review conducted with Claude (Fable 5) against the post-`9262e5e` codebase,
the git history, `OPEN_SOURCE_ARCHIVE.md`, and the surviving 2026-04-01
backtest output (`docs/backtest_results_2026-04-01.txt`). The as-it-died
state remains at tag `v1.0-as-it-died`.
