# Weather Edge

Automated weather prediction market trading system for Polymarket. Exploits the gap between multi-model weather forecast consensus and crowd-implied market prices.

## How It Works

1. Fetches forecasts from 6-8 weather models per city in **one batched API call** via Open-Meteo customer API (1M calls/month)
2. Applies **dynamic bias corrections** computed from 60K+ hindcast snapshots (365-day rolling, per-model per-city)
3. Applies EMOS calibration (2x spread inflation, 50% bias shrinkage, 70% bucket cap, 1.2C variance floor)
4. Computes **adaptive Brier-weighted consensus**, models that predict well get more influence automatically
5. Detects bust-causing weather patterns (13 patterns: Chinook, Foehn, marine layer, PRD haze, Asian cold front, etc.)
6. Discovers active Polymarket weather markets via Gamma API (parallel 5-page fetch)
7. **Claude Sonnet 4 (Meteorologist)**, assesses physical plausibility of forecasts, not market prices
8. **Gemini 2.5 Flash (Risk Quant)**, execution cost analysis, order book depth, slippage assessment, portfolio impact. Weather-blind.
9. Both AI calls run **in parallel** via `asyncio.gather` (~15s total, not 60s serial)
10. Calculates edge with **penny-first strategy**: 35% today / 25% tomorrow / 40% penny pool
11. **Risk controls**: circuit breaker (drawdown kill-switch), correlation limits per weather system, gross exposure cap
12. **Three-tier exit system**: 2-min emergency auto-kill (edge < -15%, no AI), 30-min AI-reviewed exits (Claude physics + Gemini cost), stale model detection (>4h old data)
13. Snipes model drops, 3-minute hash-based probes detect ECMWF/GFS/HRRR updates
14. Auto-resolves trades against observations (Fahrenheit + Celsius bucket parsing)
15. **Self-learning engine**: forecast ledger (60K snapshots), AI decision persistence, inverse Brier weighting
16. Persists everything to SQLite, trades, forecasts, AI decisions, settings (survives restarts)

## Trading Modes

Two independent settings in `.env`:

```
PAPER_MODE=true    # Run paper trading (simulated)
LIVE_MODE=true     # Run live trading (real money on Polymarket)
```

Either can be enabled/disabled independently. When `PAPER_MODE=false`, all paper P&L and trades are excluded from the dashboard, only live data is shown.

## Live Execution

The system trades real money on Polymarket via the CLOB API.

### Architecture: Decoupled Paper + Live

```
                    Shared Layer
    Weather Models -> Signals -> AI Reasoning (Claude + Gemini)
                         |
              +----------+----------+
              |                     |
         Paper System          Live System
         (if enabled)          (if enabled)
              |                     |
    PaperTrader.place_trade()  TradeExecutor.place_limit_order()
                               TradeExecutor.place_sell_order()
              |                     |
         paper_trades          fills -> positions
         (SQLite)              (SQLite, synced from exchange)
```

### Source of Truth: Exchange, Not Local State

- **Orders** = intentions (what we tried to buy). Tracked in `live_trades` table.
- **Fills** = reality (what actually executed). Immutable ledger in `fills` table, synced from `get_trades()`.
- **Positions** = aggregated fills. Rebuilt from fills every cycle.
- **Balance** = checked from exchange before every order loop. Tracked in-loop to prevent overspending.
- Portfolio sync runs every 2 minutes (fast exit loop) + every 30-min cycle.

### Exit Architecture (Three Tiers)

| Tier | Interval | Trigger | AI Review | Order Type |
|------|----------|---------|-----------|------------|
| Emergency | 2 min | Edge < -15% | None (auto-kill) | Taker (pay ~2% fee) |
| Standard | 30 min | Edge < -7% | Claude (physics) + Gemini (cost) in parallel | Taker if high urgency, maker otherwise |
| Stale model | 30 min | Forecast data >4h old | Full AI review | Based on urgency |

### AI Exit Roles

| | Claude Sonnet 4 | Gemini 2.5 Flash |
|---|---|---|
| **Role** | Meteorologist | Risk Quant |
| **Focus** | Is the weather thesis physically dead? | Is exiting now the best risk/reward? |
| **Data** | Model forecasts, consensus, patterns | Shares, cost basis, order book depth, spread, taker fees |
| **Decision** | "The cold front isn't coming" | "Bid depth is 50 shares, we hold 112, exit will move price 5c" |

### Live Capabilities

| Feature | Status |
|---------|--------|
| Buy orders (maker, post_only, $0 fee) | Done |
| Sell orders (maker or taker based on urgency) | Done |
| Emergency exit loop (2-min, no AI) | Done |
| Slippage guard (max 2% drift from mark) | Done |
| Exchange balance check before orders | Done |
| Cancel-and-replace on price drift >1c | Done |
| Sell order persistence to SQLite | Done |
| Portfolio sync from exchange every 2 min | Done |
| Kill switch (Redis-backed, cancels all) | Done |
| Position-aware duplicate prevention | Done |
| Gas cost tracking (est $0.002/fill) | Done |
| Stale model detection (>4h = high urgency exit) | Done |
| Pool performance segregation (Core vs Penny) | Done |
| Parallel Open-Meteo fetching (semaphore 10) | Done |
| Parallel Gamma API discovery (5 pages concurrent) | Done |
| Portfolio value = market value (not cost basis) | Done |
| Polymarket Data API as SOA (positions, P&L, activity) | Done |
| Order reconciliation against CLOB every 2 min | Done |
| Z-score guard (reject core bets >2 std devs from mean) | Done |
| Adaptive sniper (30s during model drops, 180s otherwise) | Done |
| Dashboard: city sparklines, P&L bars, data freshness | Done |
| Dashboard: spread-vs-edge warnings on signals | Done |
| Paper mode toggle (PAPER_MODE setting) | Done |

### VPN (Required -- UK Geo-blocked)

WireGuard split tunnel to ProtonVPN Canada:
- Only Polymarket IPs routed through VPN (`104.18.0.0/16`, `172.64.0.0/16`)
- Config: `/etc/wireguard/protonvpn.conf`
- Enabled on boot: `systemctl enable wg-quick@protonvpn`

### Wallet Configuration

- Polymarket Magic Link proxy wallet
- `POLYMARKET_PRIVATE_KEY` = wallet private key (0x...)
- `POLYMARKET_WALLET` = EOA address derived from key (NOT the proxy address)
- `POLYMARKET_SIGNATURE_TYPE=1` (proxy wallet)
- API creds auto-derived via `create_or_derive_api_creds`

### Tax Compliance (UK CGT)

- `fills` table = immutable audit trail for HMRC
- Section 104 pooling: weighted average cost per asset
- Settlement: market resolves -> winning token = $1, losing = $0
- Maker trades = $0 fee, taker exits = ~2% fee (tracked in fills)

## Production Deployment

The system runs on **hf-toybox-001** (Rocky Linux 9.7):

- **Dashboard**: http://10.30.20.200:8000
- **Service**: `weather-edge.service` (systemd, auto-restart, survives reboots)
- **User**: `weather` (restricted service account)
- **Logs**: `journalctl -u weather-edge -f`

### Deploying Updates

```bash
git push origin main
ssh root@10.30.20.200 "cd /home/weather/weather-edge && sudo -u weather git pull && systemctl restart weather-edge"
```

### Service Management

```bash
systemctl status weather-edge
systemctl restart weather-edge
journalctl -u weather-edge -f
journalctl -u weather-edge --since '1 hour ago'
```

## Local Development

```bash
git clone git@gitlab.hulofuse.com:trading/weather-edge.git
cd weather-edge
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,dashboard]"
cp .env.example .env
# Edit .env -- add API keys
uvicorn weather_edge.dashboard.app:app --port 8000
```

API keys needed (set in `.env`):
- `ANTHROPIC_API_KEY` -- Claude reasoning layer
- `GEMINI_API_KEY` -- Gemini risk quant layer
- `OPENMETEO_API_KEY` -- Open-Meteo customer tier (1M calls/month)
- System works without keys but loses AI reasoning and uses free tier (rate-limited).

## Architecture

```
Open-Meteo (6-8 physics models/city) + GribStream (GraphCast AI model)
        |
  Dynamic Bias Correction (60K hindcast snapshots, per-model per-city)
        |
  Pattern Detector (13 patterns: Chinook, Foehn, marine layer, PRD haze, etc.)
        |
  EMOS Calibration (spread inflation, bias shrinkage, variance floor)
        |
  Adaptive Brier-Weighted Consensus (models scored by hindcast accuracy)
        |                              Polymarket Gamma API
        v                                      |
  Edge Detection + Kelly Sizing  <---  Market Prices + Order Book
        |                    |
        |    [parallel]  Claude (Meteorologist -- physics only)
        |                Gemini (Risk Quant -- cost/liquidity only)
        |                    |
        +-- Risk Controls: circuit breaker, correlation limits, exposure cap
        +-- Balance check from exchange before orders
        +-- Fee gate (skip if taker fee > 40% of edge)
        |
        v
  Live Executor (maker buys, taker/maker exits)
        |
  Fast Exit Loop (2 min) -- emergency sells, portfolio sync, balance refresh
        |
  Full Cycle (30 min) -- AI-reviewed exits, new signals, model refresh
        |
  Web Dashboard (FastAPI + WebSocket)
```

## Cities (22)

### Americas
| City | ICAO | Notes |
|------|------|-------|
| New York | LGA | Urban heat island, sea breeze timing |
| Chicago | KORD | Lake Michigan breeze -- GFS 4-8F warm bias |
| Dallas | KDAL | GFS dry-soil warm bias |
| Houston | KIAH | Gulf moisture, GFS warm bias 3-5F |
| Atlanta | KATL | Convective timing |
| Miami | KMIA | Convective quench, UHI |
| Denver | KDEN | Chinook winds -- models cold-biased 5-10F |
| Seattle | KSEA | Marine layer |
| Los Angeles | KLAX | June Gloom, Santa Ana winds |
| San Francisco | KSFO | Fog -- globals warm-biased 5-10F |
| Austin | KAUS | GFS dry-soil warm bias |
| Toronto | CYYZ | HRDPS regional model |

### Europe
| City | ICAO | Notes |
|------|------|-------|
| London | EGLC | UKV regional model |
| Madrid | LEMD | Saharan dust (Calima) |
| Munich | EDDM | Alpine Foehn -- models cold-biased 5-10C |
| Warsaw | EPWA | Winter inversions -- GFS warm-biased 5-15C |

### Asia-Pacific
| City | ICAO | Notes |
|------|------|-------|
| Seoul | RKSI | KMA regional |
| Tokyo | RJTT | Sea breeze timing, monsoon |
| Hong Kong | VHHH | Humidity plateau, typhoon subsidence |
| Shanghai | ZSPD | Coastal UHI, sea breeze |
| Shenzhen | ZGSZ | Pearl River Delta heat plume |

### South America
| City | ICAO | Notes |
|------|------|-------|
| Buenos Aires | SAEZ | Humidity-driven temp plateaus |

## Pattern Detection (The Edge)

| Pattern | Cities | Bias | Magnitude |
|---------|--------|------|-----------|
| Chinook/downslope | Denver | Models too cold | 5-10F |
| Alpine Foehn | Munich | Models too cold | 5-10C |
| Marine layer | SF, LA | Globals too warm | 5-10F |
| Santa Ana | LA | Models too cold | 5-8F |
| Lake breeze | Chicago | GFS too warm | 4-8F |
| Cold pool/inversion | Warsaw | GFS too warm | 5-15C |
| GFS dry bias | Houston, Dallas | GFS too warm | 3-5F |
| Sea breeze timing | Tokyo, Seoul, NYC | Variable | 2-3C |
| PRD haze suppression | Shenzhen, HK | Models too warm | 2-5C |
| Shanghai boundary layer | Shanghai | UHI vs haze | 3-6C |
| Asian cold front | SHA, SZN, HKG, SEL, TYO | Timing error | 8-12C |
| Return of Nantian | Shenzhen, HK, Shanghai | Models too warm | 2-4C |

## Configuration

Key `.env` settings:

```
# Trading modes
PAPER_MODE=false             # Disable paper trading
LIVE_MODE=true               # Enable live trading

# Bankroll
BANKROLL=210.0               # Starting capital (live)

# Sizing
KELLY_FRACTION=0.25          # Quarter-Kelly (conservative)
MAX_POSITION_PCT=0.03        # Max 3% of bankroll per trade

# Pools
POOL_TODAY_PCT=0.60
POOL_TOMORROW_PCT=0.30
POOL_PENNY_PCT=0.10

# Cycle timing
FETCH_INTERVAL_MINUTES=30    # Main cycle (fast exit loop runs every 2 min)

# Risk
MAX_SLIPPAGE_PCT=0.02        # Max 2% price drift for exit orders

# API keys
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
OPENMETEO_API_KEY=...
OPENMETEO_PAID_TIER=true
```

## Testing

```bash
# All tests (110 tests)
.venv/bin/python -m pytest tests/ -v

# Contract tests only (fast, run by pre-commit hook)
.venv/bin/python -m pytest tests/test_contracts.py -v

# Pre-commit runs: ruff lint + contract tests
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Dashboard HTML |
| GET | `/api/state` | Full system state |
| POST | `/api/start` | Start automated trading |
| POST | `/api/stop` | Stop trading (keep positions) |
| POST | `/api/close-all` | Close all positions (taker mode) |
| POST | `/api/new-session` | Reset to fresh bankroll |
| POST | `/api/refresh` | Trigger immediate cycle |
| POST | `/api/settings` | Update risk profile |
| POST | `/api/kill-switch` | Emergency stop + cancel all exchange orders |
| WS | `/ws` | Real-time state updates |

## TODO

- **Mobile responsive layout**, single-column view for phone with portfolio value, P&L, active alerts, and urgent positions. Currently unusable on mobile.
- **AI Decisions tab layout**, risk factors/rationale stacking in one row instead of per-decision. CSS/rendering bug.
- **P&L chart**, portfolio value plotted over time from fast exit loop data (every 2 min). Single line chart.
- **Blotter description column**, position title not mapped to description field in blotter rows.

## License

Private. Not for redistribution.
