# Weather Edge

Automated weather prediction market trading system for Polymarket. Exploits the gap between multi-model weather forecast consensus and crowd-implied market prices.

## Current Strategy (2026-04-02)

**Card counter approach**, the model picks the right whole-degree bucket ~46% of the time (vs ~10% baseline). Over enough bets, the math works. Not every hand wins, but the edge compounds.

### Resolution Source: Weather Underground (METAR)

Polymarket resolves weather markets against **Wunderground displayed values** from specific airport METAR stations. Our bias correction and trade resolution are calibrated against the same source via **IEM ASOS** (Iowa Environmental Mesonet), which serves the identical raw METAR data.

- **Celsius markets**: `round(max(hourly_tmpc_readings))`, whole degrees
- **Fahrenheit markets**: `round(max(hourly_tmpf_readings))`, whole degrees
- **Hong Kong exception**: resolves from HK Observatory (0.1C precision), NOT Wunderground

### Calibration Settings

| Parameter | Value | Why |
|-----------|-------|-----|
| SPREAD_INFLATION_FACTOR | 1.3 | Models are correlated but 2.0x was killing real edges |
| BIAS_SHRINKAGE | 0.9 | Trust METAR-calibrated hindcast (was 0.5 with wrong data) |
| EMOS_VARIANCE_FLOOR_C | 0.7 | Old floor (1.2) was wider than the bucket itself |
| MAX_BUCKET_PROBABILITY | 0.70 | Cap single-bucket probability |

### City Accuracy Tiers (MAE vs METAR)

| Tier | Cities | MAE | Strategy |
|------|--------|-----|----------|
| **Gold** | London (0.63), Miami (0.85), Madrid (0.91) | < 1.0C | Full position, highest confidence |
| **Silver** | Seattle, Munich, Dallas, Houston, Tokyo | 1.0-1.2C | Standard position |
| **Bronze** | Wellington, Lucknow, Warsaw, Chicago, Shenzhen | 1.2-1.5C | Trade when mispricing is large |
| **Caution** | NYC, Atlanta, LA, Buenos Aires, Hong Kong | 1.4-1.5C | Only with strong model agreement |
| **Avoid** | Denver (1.83), Toronto (1.92), Shanghai (1.99), Seoul (2.08), SFO (2.13) | > 1.8C | Model error > bucket width, essentially random |

### Bankroll-Dependent Sizing

| Bankroll | MIN_SIZE | MAX_POSITIONS | Taker Threshold |
|----------|----------|---------------|-----------------|
| < $200 (current) | $5 | 20 | 8% edge |
| $200-500 | $10 | 20 | 10% edge |
| $500-630 | $10 | 15 | 10% edge |
| $630+ (friends' capital) | $15 | 12 | 12% edge |

### Model Agreement Gate (Gemini recommendation)

| Model Spread (raw std) | Action |
|------------------------|--------|
| < 1.5C | Full bet, models agree, high confidence |
| 1.5-2.0C | Spread across top 2 adjacent buckets |
| > 2.0C | Skip, models disagree, edge is noise |

### Key Lessons Learned

1. **Data source must match oracle.** Open-Meteo archive (gridded reanalysis) ≠ Wunderground (airport METAR sensors). 0.9C MAE gap caused 67% rounding mismatches. Paper P&L of +$8,471 was fiction.
2. **Verify ground truth before paper trading.** Paper results are worthless if calibrated against the wrong target.
3. **Execution matters as much as prediction.** 17% fill rate on maker orders, adverse selection on fills, data lag on <24h markets. Swing bot (48h+) addresses all three.
4. **Position sizing > prediction accuracy.** Card counter doesn't bet big on every hand.

## How It Works

1. Fetches forecasts from 6-8 weather models per city via Open-Meteo customer API
2. Applies **METAR-calibrated bias corrections** from hindcast snapshots (model vs actual station reading)
3. Applies EMOS calibration (1.3x spread inflation, 0.9 bias shrinkage, 0.7C variance floor)
4. Computes **adaptive Brier-weighted consensus**, models that predict well get more influence
5. Detects bust-causing weather patterns (13 patterns: Chinook, Foehn, marine layer, etc.)
6. Discovers active Polymarket weather markets via Gamma API
7. **Claude Sonnet 4 (Meteorologist)**, physical plausibility, market-blind
8. **Gemini 2.5 Flash (Risk Quant)**, execution cost, order book depth, weather-blind
9. Both AI calls run **in parallel** via `asyncio.gather`
10. Calculates edge against market prices
11. **Risk controls**: circuit breaker, correlation limits, model agreement gate
12. **Three-tier exit system**: 2-min emergency (edge < -15%), 30-min AI-reviewed, stale model detection
13. Auto-resolves trades against **IEM METAR observations** (same source as Wunderground)
14. Auto-redeems winning positions via **Polymarket Relayer API** (gasless)
15. Persists everything to SQLite, trades, forecasts, AI decisions, fills

## Trading Modes

Two independent settings in `.env`:

```
PAPER_MODE=true    # Run paper trading (simulated)
LIVE_MODE=true     # Run live trading (real money on Polymarket)
```

## Live Execution

### Swing Bot (48h+ Horizon)

The bot only trades markets resolving 48+ hours out, where:
- Our bias correction edge is strongest (no ground truth yet for faster bots to exploit)
- Data lag (Open-Meteo 1-2h behind NWS) is less damaging
- Less competition from faster bots with direct NWS feeds

### Entry Rules

| Condition | Action |
|-----------|--------|
| Edge >= 8% | Taker entry (cross spread, +3c above midpoint) |
| Edge 5-8% | Maker entry (post-only, 15 min timeout) |
| Edge < 5% | Skip |
| USDC < $20 | Block all entries (vulture mode) |
| Model std > 2.0C | Skip (models disagree) |

### Exit Architecture (Three Tiers)

| Tier | Interval | Trigger | AI Review | Order Type |
|------|----------|---------|-----------|------------|
| Emergency | 2 min | Edge < -15% | None (auto-kill) | Taker |
| Standard | 30 min | Edge < -7% | Claude + Gemini in parallel | Taker if urgent |
| Stale model | 30 min | Data >4h old | Full AI review | Based on urgency |

### Auto-Redeem

Winning positions are automatically redeemed via the **Polymarket Relayer API**. The relayer executes `CTF.redeemPositions` from the proxy wallet (gasless, Polymarket pays gas). No MATIC needed.

### Source of Truth

- **Orders** = intentions (`live_trades` table)
- **Fills** = reality (`fills` table, synced from exchange)
- **Positions** = aggregated fills (rebuilt every cycle)
- **Observations** = IEM METAR primary, Open-Meteo archive fallback
- **Resolution** = Polymarket on-chain (CTF `payoutDenominator`)

## Cities (24)

### Americas
| City | ICAO | MAE | Notes |
|------|------|-----|-------|
| New York | KLGA | 1.41C | Urban heat island, sea breeze |
| Chicago | KORD | 1.31C | Lake Michigan breeze |
| Dallas | KDAL | 1.12C | GFS dry-soil warm bias |
| Houston | KIAH | 1.12C | Gulf moisture |
| Atlanta | KATL | 1.44C | Convective timing |
| Miami | KMIA | 0.85C | Low variance, warm baseline |
| Denver | KDEN | 1.83C | Chinook swings, high MAE |
| Seattle | KSEA | 0.97C | Marine layer |
| Los Angeles | KLAX | 1.45C | June Gloom, Santa Ana |
| San Francisco | KSFO | 2.13C | Fog, highest MAE |
| Austin | KAUS | 1.40C | GFS dry-soil warm bias |
| Toronto | CYYZ | 1.92C | High variance |

### Europe
| City | ICAO | MAE | Notes |
|------|------|-----|-------|
| London | EGLC | 0.63C | Best accuracy |
| Madrid | LEMD | 0.91C | Saharan dust |
| Munich | EDDM | 1.02C | Alpine Foehn |
| Warsaw | EPWA | 1.20C | Winter inversions |

### Asia-Pacific
| City | ICAO | MAE | Notes |
|------|------|-----|-------|
| Seoul | RKSI | 2.08C | High MAE, caution |
| Tokyo | RJTT | 1.12C | Sea breeze timing |
| Hong Kong | VHHH | 1.49C | **Resolves from HK Observatory, not VHHH** |
| Shanghai | ZSPD | 1.99C | Coastal UHI, high MAE |
| Shenzhen | ZGSZ | 1.31C | Pearl River Delta |

### Other
| City | ICAO | MAE | Notes |
|------|------|-----|-------|
| Buenos Aires | SAEZ | 1.46C | Southern hemisphere autumn |
| Wellington | NZWN | 1.16C | Maritime, windy |
| Lucknow | VILK | 1.18C | Pre-monsoon transition |

## Production Deployment

**Server**: hf-toybox-001 (Rocky Linux 9.7, 10.30.20.200)

```bash
# Deploy
git push origin main
ssh root@10.30.20.200 "cd /home/weather/weather-edge && sudo -u weather git pull && systemctl restart weather-edge"

# Start/stop trading
curl -s -X POST http://10.30.20.200:8000/api/start
curl -s -X POST http://10.30.20.200:8000/api/stop

# Logs
journalctl -u weather-edge -f
```

## Testing

```bash
.venv/bin/python -m pytest tests/ -v    # 126 tests
```

## License

Private. Not for redistribution.
