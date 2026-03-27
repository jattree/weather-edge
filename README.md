# Weather Edge

Automated weather prediction market trading system for Polymarket. Exploits the gap between multi-model weather forecast consensus and crowd-implied market prices.

Inspired by [@ColdMath's Claude Trader](https://polymarket.com/profile/@ColdMath) ($77K+ positions, 4,300+ predictions, 73% win rate, ~$37K monthly P&L).

## How It Works

1. Fetches forecasts from 6-8 weather models per city in **one batched API call** via Open-Meteo customer API (1M calls/month)
2. Applies EMOS calibration (2x spread inflation, 50% bias shrinkage, 70% bucket cap, 1.2C variance floor) to prevent overconfident edges
3. Applies data-driven NWS station bias corrections (30-day rolling calibration)
4. Computes weighted consensus with KDE distribution fitting
5. Detects bust-causing weather patterns (Chinook, Foehn, marine layer, etc.) and adjusts confidence
6. Discovers active Polymarket weather markets via Gamma API (130+ temperature markets daily)
7. **Claude API** analyzes top 3 signals per cycle, approves or skips with rationale
8. **Gemini 2.5 Flash** red-teams Claude-approved trades, finds counter-arguments and dissent. High dissent (>=0.7) halves position size
9. Fetches real order book ask prices via CLOB `/book` endpoint for spread capture detection
10. Calculates edge (model probability vs market price) with quarter-Kelly sizing
11. Runs three-pool strategy: 60% today / 30% tomorrow / 10% penny (Gemini-validated allocation)
12. Generates spread capture hedge orders (buy opposite side for merge profit)
13. **Early exit monitor**, scans open positions each cycle for edge inversion (>7%), profit cap (88%+), pattern bust. Claude + Gemini review before closing. Penny bets never exit.
14. Snipes model drops, 3-minute hash-based probes detect ECMWF/GFS/HRRR updates, trades before market adjusts
15. Auto-resolves trades against Polymarket outcomes and Open-Meteo archive observations (free tier)
16. Persists all trades to SQLite (survives restarts)
17. **Contract-first runtime validation**, 7 pure validation functions catch silent failures before capital deploys

## Production Deployment

The system runs on **hf-toybox-001** (Rocky Linux 9.7):

- **Dashboard**: http://10.30.20.200:8000
- **Service**: `weather-edge.service` (systemd, auto-restart, survives reboots)
- **User**: `weather` (restricted service account)
- **Logs**: `journalctl -u weather-edge -f`
- **Deploy**: `/home/weather/deploy.sh` (git pull + pip install + restart)

### Deploying Updates

From your dev machine, push to GitLab then deploy:

```bash
git push origin main
ssh root@10.30.20.200 "cd /home/weather/weather-edge && sudo -u weather git pull && systemctl restart weather-edge"
```

Or SSH directly:

```bash
ssh root@10.30.20.200
su - weather
cd weather-edge
./deploy.sh
```

### Service Management

```bash
# On hf-toybox-001
systemctl status weather-edge    # Check status
systemctl restart weather-edge   # Restart
systemctl stop weather-edge      # Stop
journalctl -u weather-edge -f    # Tail logs
journalctl -u weather-edge --since '1 hour ago'  # Recent logs
```

### Server Details

| Component | Detail |
|-----------|--------|
| Host | hf-toybox-001 (10.30.20.200) |
| OS | Rocky Linux 9.7 (Blue Onyx) |
| Python | 3.11 |
| Timezone | UTC |
| App user | `weather` (restricted) |
| Repo | `/home/weather/weather-edge` |
| Branch | `main` |
| DB | `/home/weather/weather-edge/weather_edge.db` (SQLite) |
| Config | `/home/weather/weather-edge/.env` |
| Service | `/etc/systemd/system/weather-edge.service` |
| Log rotation | 14 days file, 30 days journald |
| Sysctl | Optimized for network-heavy workload |

## Local Development

```bash
git clone git@gitlab.hulofuse.com:trading/weather-edge.git
cd weather-edge
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,dashboard]"
pip install websockets
cp .env.example .env
# Edit .env, add ANTHROPIC_API_KEY for Claude reasoning
uvicorn weather_edge.dashboard.app:app --port 8000
```

API keys needed for full functionality (set in `.env`):
- `ANTHROPIC_API_KEY`, Claude reasoning layer
- `GEMINI_API_KEY`, Gemini red team dissent layer
- `OPENMETEO_API_KEY`, Open-Meteo customer tier (1M calls/month)
- System works without keys but loses AI reasoning and uses free tier (rate-limited).

## GitLab Repository

- **URL**: https://gitlab.hulofuse.com:9443/trading/weather-edge
- **Group**: trading
- **Branch**: `main` (production)

## Architecture

```
Open-Meteo (6-8 physics models/city) + GribStream (GraphCast AI model)
        |
        v
  Bias Correction (30-day calibrated, ENSO regime-aware shrinkage)
        |
        v
  Pattern Detector (Chinook, Foehn, marine layer, lake breeze, etc.)
        |
        v
  EMOS Calibration (spread inflation, bias shrinkage, variance floor)
        |
  KDE Consensus Engine (weighted by model skill)
        |                              Polymarket Gamma API
        v                                      |
  Edge Detection + Kelly Sizing  <---  Market Prices + Order Book Asks + AI/Physics Divergence
        |                    |
        |              Claude API (top 3 signals, bull case)
        |                    |
        |              Gemini 2.5 Flash (red team, bear case)
        |                    |
        +-- Today pool (60% bankroll, quarter-Kelly)
        +-- Tomorrow pool (30% bankroll, conviction bets)
        +-- Penny pool (10% bankroll, ColdMath-style sniping)
        +-- Spread capture (hedge orders on opposite side, 3% fee buffer)
        +-- Fee gate (skip if taker fee > 40% of edge)
        |
        v
  Paper Trader / Live Executor (post_only maker)  --->  SQLite (cold) + Redis (hot)
        |                                    |
  Auto-Resolver (settles trades       Trade History
   against NWS observations)          (survives restarts)
        |
        v
  Web Dashboard (FastAPI + WebSocket + 10 Bloomberg tabs)
        |
  Model-Drop Sniper (probes every 3min, hash-based dedup)
        |
  Early Exit Monitor (edge inversion, profit cap, AI-reviewed)
        |
  Contract Validation (7 runtime checks, EMOS, budgets, model count)
```

## Cities (22)

### Americas (HRRR + NAM regional models)
| City | ICAO | Notes |
|------|------|-------|
| New York | LGA | Urban heat island, sea breeze timing |
| Chicago | KORD | Lake Michigan breeze, GFS 4-8F warm bias |
| Dallas | KDAL | GFS dry-soil warm bias |
| Houston | KIAH | Gulf moisture, GFS warm bias 3-5F |
| Atlanta | KATL | Convective timing |
| Miami | KMIA | Convective quench, UHI |
| Denver | KDEN | Chinook winds, models cold-biased 5-10F |
| Seattle | KSEA | Marine layer |
| Los Angeles | KLAX | June Gloom, Santa Ana winds |
| San Francisco | KSFO | Fog, globals warm-biased 5-10F |
| Austin | KAUS | GFS dry-soil warm bias (same as Dallas) |
| Toronto | CYYZ | HRDPS regional model |

### Europe (global models only)
| City | ICAO | Notes |
|------|------|-------|
| London | EGLC | UKV regional model |
| Madrid | LEMD | Saharan dust (Calima) |
| Munich | EDDM | Alpine Foehn, models cold-biased 5-10C |
| Warsaw | EPWA | Winter inversions, GFS warm-biased 5-15C |

### Asia-Pacific (global models only)
| City | ICAO | Notes |
|------|------|-------|
| Seoul | RKSI | KMA regional, GFS over-predicts by 4.65C |
| Tokyo | RJTT | Sea breeze timing, monsoon |
| Hong Kong | VHHH | Humidity plateau, typhoon subsidence |
| Shanghai | ZSPD | Coastal UHI, sea breeze |
| Shenzhen | ZGSZ | Pearl River Delta heat plume |

### South America
| City | ICAO | Notes |
|------|------|-------|
| Buenos Aires | SAEZ | Humidity-driven temp plateaus |

Global models on all cities: ECMWF, GFS, ICON, GEM, JMA, MeteoFrance.

## Pattern Detection (The Edge)

The system detects weather patterns that cause systematic model failures:

| Pattern | Cities | Bias | Magnitude | Detection |
|---------|--------|------|-----------|-----------|
| Chinook/downslope | Denver | Models too cold | 5-10F | HRRR >> GFS spread |
| Alpine Foehn | Munich | Models too cold | 5-10C | Large model spread |
| Marine layer | SF, LA | Globals too warm | 5-10F | ECMWF >> HRRR |
| Santa Ana | LA | Models too cold | 5-8F | HRRR >> ECMWF |
| Lake breeze | Chicago | GFS too warm | 4-8F | GFS >> HRRR |
| Cold pool/inversion | Warsaw | GFS too warm | 5-15C | GFS >> ECMWF |
| GFS dry bias | Houston, Dallas | GFS too warm | 3-5F | GFS >> ECMWF |
| Sea breeze timing | Tokyo, Seoul, NYC | Variable | 2-3C | Large model spread |
| PRD haze suppression | Shenzhen, HK | Models too warm | 2-5C | Pollution blocks solar radiation |
| Shanghai boundary layer | Shanghai | UHI vs haze conflict | 3-6C | ECMWF vs GFS divergence |
| Asian cold front | SHA, SZN, HKG, SEL, TYO | Timing error | 8-12C | Large model spread (>5C) |
| Return of Nantian | Shenzhen, HK, Shanghai | Models too warm | 2-4C | Tight consensus + 18-26C range |

When a pattern is detected, trading confidence increases on the correctly-predicted side.

13 patterns across 22 cities. Asian markets (Shenzhen $111K volume on 3 markets) have outsized volume-per-market ratios driven by gambling-motivated participants, wide edges.

## Three-Pool Strategy (Gemini-Validated)

- **Today pool (60%)**, Same-day markets. Capital recycles nightly when trades resolve. Quarter-Kelly sizing.
- **Tomorrow pool (30%)**, Next-day conviction bets. Locked until resolution.
- **Penny pool (10%)**, ColdMath-style penny sniping. Buy YES at $0.01-$0.06 where model says 3x+ higher probability. Most lose, but 50:1 payoffs compound.

## Spread Capture (ColdMath Strategy)

Detects when buying both YES and NO on a bucket costs less than $1.00:
- Uses real order book ask prices from CLOB `/book` endpoint (not midpoints)
- 1.5% safety buffer: only flags profitable when `YES_ask + NO_ask < 0.985`
- Generates hedge orders on opposite side of directional trades
- On resolution, simulates merge: both sides redeem for $1/share = guaranteed profit
- Requires live execution for actual CTF contract merges (paper trading simulates P&L)

## Dual-AI Reasoning Pipeline

Two AI models argue about every trade before capital is deployed:

### Claude (Bull Case)
- Analyzes top 3 signals per cycle
- Returns: should_trade, confidence_adjustment (0.5-1.5x), rationale, risk_factors
- If Claude skips, the trade is dropped entirely
- Cost: ~$0.72/day

### Gemini 2.5 Flash (Red Team / Bear Case)
- Only runs on Claude-approved trades
- Prompted to find the strongest case AGAINST the trade
- Returns: dissent_strength (0-1), counter_arguments, risk_the_bull_missed
- High dissent (>=0.7) halves position size automatically
- Cost: ~$0.10/day

### Decision Flow
1. Model math finds edge → 2. Claude approves/skips → 3. Gemini red-teams approved trades → 4. Size adjusted by both AI confidence levels → 5. Contract validation → 6. Trade placed

AI reasoning only runs on main 30-min cycles (not sniper-triggered) to save API costs (~$0.80/day total).

All decisions logged to the **AI Decisions** dashboard tab with source (Claude/Gemini), rationale, and risk factors.

Set `ANTHROPIC_API_KEY` and `GEMINI_API_KEY` in `.env` to enable. System works without them but loses the AI reasoning layers.

## Early Exit Monitor

Scans open positions each cycle for AI-reviewed early exits:

| Trigger | Condition | Action |
|---------|-----------|--------|
| Edge inversion | Model prob flipped >7% against us | Claude confirms → Gemini argues hold → EXIT if both agree |
| Profit cap | Market price >88% and model <94% | Lock in ~80% of max payout |
| Pattern bust | Detected pattern invalidated | Aggressive exit |
| Penny bets | Entry <= $0.06 | **NEVER exit**, hold to 0 or 1 |

10% reserve pot ($200 on $2K bankroll) is kept uncommitted for HIGH-tier signals only.

## Auto-Resolution

Trades auto-settle each cycle:

1. Checks Polymarket Gamma API for resolved markets
2. Falls back to Open-Meteo historical observations if Polymarket hasn't resolved
3. Parses bucket boundaries from trade descriptions (e.g., "between 82-83F")
4. Compares actual observed temperature to determine YES/NO outcome
5. Marks trades as WON/LOST, calculates P&L, frees capital for new trades
6. Dashboard shows countdown timer per trade: "3h 42m" or "OVERDUE"

Resolution timing (after target date midnight local time + 2h NWS buffer):
- Asian cities: resolve ~afternoon UTC
- European cities: resolve ~overnight UTC
- US East: resolve ~6-7 AM UTC next day
- US West: resolve ~9-10 AM UTC next day

## Dashboard (Bloomberg-Grade)

Header bar: Portfolio Value | Free Cash | Invested | P&L | Return % | Win Rate | Positions | Cycles

11 tabs, all functional (no stubs):

| Tab | Feature |
|-----|---------|
| Trade Log | Live trades with pool tags, resolution countdown, session P&L sidebar, market calendar |
| Blotter | Full trade history, sortable columns, filters (All/Open/Won/Lost/Core/Tail), CSV export |
| Risk / P&L | Canvas equity curve (1H/6H/1D/1W/ALL), Sharpe ratio, max drawdown, win rate by city |
| Heat Map | 21-city edge color grid + market volume by city + calendar with per-pool resolution countdown |
| Alerts | System activity feed (trades, sniper events, pattern detections, cycle completions) |
| Weather | Live NWS alerts (US) + synthetic alerts from forecast cache (international, zero API calls) |
| Backtest | 7-day historical backtest via Open-Meteo archive, on-demand |
| Correlation | 21x21 city forecast correlation matrix from model data |
| Execution | Position sizing stats, edge distribution histogram, capital utilization |
| AI Decisions | Claude TRADE/SKIP + Gemini AGREE/DISSENT decisions with rationale and risk factors |
| System Status | Health monitoring for all 10 external services with status dots and key metrics |

### Dashboard Controls

- **START**, Begin automated trading (30-min cycles + 3-min sniper probes)
- **STOP**, Pause trading, keep positions open
- **CLOSE ALL**, Sell profitable positions, hold losers to resolution (confirmation required)
- **NEW SESSION**, Archive and reset (confirmation required)
- **REFRESH NOW**, Trigger immediate cycle (runs in background, shows "Refreshing..." feedback)

### Keyboard Shortcuts

Press `?` on the dashboard to see all shortcuts. Key ones:
- `S` Start, `X` Stop, `C` Close All, `N` New Session, `R` Refresh
- `1-9`, `0` Switch tabs (0 = AI Decisions)
- `Escape` Close modals
- Arrow keys to navigate city list

## Competitor Tracking

Tracks public Polymarket profiles of known weather traders each cycle:

- **@ColdMath**, $77K+ positions, 4,300+ predictions, 73% win rate, ~$37K monthly P&L
- Snapshots positions value and prediction count every 30 minutes
- Calculates growth rate per day for comparison against our paper P&L
- Displayed in dashboard sidebar: "WeatherEdge +$X vs coldmath +$Y"

## Key Features

- **22 cities** across 3 continents with 130+ daily temperature markets
- **KDE consensus**, Handles multimodal distributions when models disagree
- **Real bias corrections**, 30-day rolling calibration via Open-Meteo historical APIs
- **Pattern detection**, Chinook, Foehn, marine layer, lake breeze, cold pool, Santa Ana
- **Model-drop sniper**, Detects ECMWF/GFS/HRRR updates within 2 min, trades before market adjusts
- **Auto-resolver**, Settles trades against NWS observations with countdown timers
- **Bucket parity arbitrage**, Flags when bucket YES prices sum to >1.05
- **Dual-AI reasoning**, Claude approves/skips, Gemini red-teams with meteorological bust vectors (CIN, marine layer, soil moisture, bimodal ensemble, diurnal timing)
- **GraphCast AI model**, Google DeepMind forecasts via GribStream cross-check physics models. Divergence >3°C reduces confidence 30%
- **ENSO regime awareness**, Bias corrections shrink during La Nina → Neutral transition for sensitive cities (Seattle, Houston, SF)
- **Fee-aware execution**, Dynamic taker fee calculation, maker-only orders (post_only=True), fee-alpha gating (skip if fee >40% of edge)
- **Spread capture**, Detects profitable YES+NO spreads using real order book asks, 3% buffer for post-March-30 fees
- **Wet bulb temperature**, Humidity bias factor for tropical/humid cities (Houston, Miami, Hong Kong)
- **Competitor tracking**, Monitors @ColdMath's public stats + on-chain wallet activity
- **Quarter-Kelly sizing**, Conservative position sizing (ColdMath-validated)
- **Smart close**, Only sells winners; holds losers to resolution (capped downside)
- **SQLite persistence**, Sessions and trades survive restarts
- **Dual core/tail strategy**, Steady returns + asymmetric penny bets
- **NWS weather alerts**, Live alerts for US cities, synthetic alerts for international
- **Backtesting**, 7-day historical backtest against actual observations
- **Correlation matrix**, Pairwise city forecast correlation for diversification
- **Execution analytics**, Position sizing, edge distribution, capital utilization metrics
- **Bloomberg-grade dashboard**, 10 functional tabs, draggable splitter, keyboard shortcuts, CSV export
- **Draggable splitter**, Resize top/bottom panes, position persists in localStorage
- **Early exit monitor**, AI-reviewed exits on edge inversion, profit cap, pattern bust
- **Contract validation**, 8 runtime checks prevent silent failures (EMOS, budget, fees, model count, etc.)
- **Market volume**, 24h volume per city on Heat Map tab for liquidity assessment
- **Race condition protection**, asyncio.Lock prevents concurrent cycles from corrupting state
- **System status monitoring**, Health checks for all 10 external services with status indicators
- **Cross-market registry**, Dormant correlations (ERCOT energy, SFO aviation, AQI) ready to activate
- **Regret/adherence tracking**, Measures AI reasoning value: Claude accuracy, Gemini dissent accuracy, opportunity cost
- **Session heartbeat**, Polymarket keepalive prevents open order cancellation during live execution

## Testing

Contract-first methodology: "If this broke silently in production, would we know, and would we care?"

```bash
# Run all tests (57 tests)
.venv/bin/python -m pytest tests/ -v

# Contract tests only (fast, run by pre-commit hook)
.venv/bin/python -m pytest tests/test_contracts.py -v

# Property-based tests (requires hypothesis)
.venv/bin/pip install hypothesis
.venv/bin/python -m pytest tests/test_edge_math.py -v

# Lint
.venv/bin/ruff check src/weather_edge/
```

### Runtime Contracts (7)

| Contract | What it catches | Wired into |
|----------|----------------|------------|
| `validate_emos_active` | EMOS calibration disabled (fake 86% edges) | Scheduler cycle start |
| `validate_pool_budget` | Capital at risk exceeds bankroll | PaperTrader.should_trade() |
| `validate_reserve_pot` | Reserve pot (10%) depleted for non-HIGH signals | PaperTrader.should_trade() |
| `validate_penny_no_exit` | Penny bets flagged for early exit | ExitMonitor.scan_for_exits() |
| `validate_ai_keys_present` | AI keys empty (silent reasoning skip) | Module load (warning) |
| `validate_model_count` | Consensus from <4 models (unreliable) | Scheduler after forecast fetch |
| `validate_spread_uses_asks` | Spread profit from midpoints not asks | Spread capture detection |

### Pre-commit Hook

Runs automatically on every `git commit`:
1. `ruff check`, catches syntax errors and undefined names
2. `pytest tests/test_contracts.py`, validates all 7 business rule contracts

## Configuration

Edit `.env`:

```
BANKROLL=2000.0              # Starting capital
KELLY_FRACTION=0.25          # Quarter-Kelly (conservative)
MAX_POSITION_PCT=0.03        # Max 3% of bankroll per trade
POOL_TODAY_PCT=0.60          # 60% today pool
POOL_TOMORROW_PCT=0.30       # 30% tomorrow pool
POOL_PENNY_PCT=0.10          # 10% penny pool
PENNY_MIN_POSITION=10.0      # Min $10 per penny bet
PENNY_MAX_POSITION=20.0      # Max $20 per penny bet
FETCH_INTERVAL_MINUTES=30    # Main cycle interval (sniper probes every 3min)
ANTHROPIC_API_KEY=sk-ant-... # Claude reasoning (optional)
GEMINI_API_KEY=AIza...       # Gemini red team (optional)
OPENMETEO_API_KEY=...        # Customer tier (optional, uses free tier without)
OPENMETEO_PAID_TIER=true     # Auto-detected from API key
GRIBSTREAM_API_KEY=...       # GraphCast AI model (optional, free tier 1200 credits/day)
```

## Project Structure

```
src/weather_edge/
  config.py              # 21 cities, model weights, settings
  scheduler.py           # Main orchestration loop
  persistence.py         # SQLite session/trade storage
  cli.py                 # CLI commands
  reporting.py           # Rich terminal tables
  db.py                  # SQLAlchemy async engine (for PG migration)
  models/
    enums.py             # City, WeatherModel, MarketType
    orm.py               # SQLAlchemy ORM (for PG migration)
  live_state.py          # Redis hot-path cache (books, heartbeats, dashboard state)
  fetchers/
    openmeteo.py         # Multi-model forecast fetcher (batch, customer API)
    polymarket.py        # Market discovery + price fetcher + book asks
    gribstream.py        # GraphCast AI model via GribStream API
  analysis/
    consensus.py         # Weighted KDE consensus engine
    edge.py              # Edge detection + dual Kelly sizing
    market_mapper.py     # Maps markets to forecast variables
    bias_correction.py   # Data-driven station bias (30-day calibrated)
    pattern_detector.py  # Chinook/Foehn/marine layer/etc detection
    arbitrage.py         # Bucket parity checks
    model_timing.py      # Golden window detection
    sniper.py            # Model-drop triggered trading
    resolver.py          # Auto-resolves trades against NWS observations
    claude_reasoning.py  # Claude API trade analysis (bull case)
    gemini_reasoning.py  # Gemini red team dissent (bear case)
    competitor_tracker.py # @ColdMath and whale tracking
    whale_tracker.py     # On-chain ERC-1155 CTF token tracking
    weather_alerts.py    # Live NWS + synthetic weather alerts
    backtester.py        # Historical backtest engine
    correlation_matrix.py # City forecast correlation
    execution_analytics.py # Trading metrics and distributions
    exit_monitor.py      # AI-reviewed early exit (edge inversion, profit cap)
    contracts.py         # 8 pure validation functions (runtime contract checks)
    enso_regime.py       # ENSO state from NOAA CPC, regime-aware bias shrinkage
    wet_bulb.py          # Wet bulb temp for humidity-sensitive cities
    cross_market.py      # Dormant cross-market correlation registry
    regret_tracker.py    # AI adherence/regret analysis (Claude vs Gemini accuracy)
    service_health.py    # External service health tracking
  trading/
    paper.py             # Paper trader with three-pool split + spread trades + smart close
    executor.py          # Real Polymarket CLOB execution (post-only maker, heartbeat)
    market_maker.py      # Spread capture: hedge orders + merge simulation
    fees.py              # Polymarket 2026 fee calculation (taker/maker/rebate)
  dashboard/
    app.py               # FastAPI + WebSocket backend (20+ API endpoints)
    templates/
      index.html         # Bloomberg-grade dashboard (3,300+ lines)
    static/              # Static assets
tests/
  test_contracts.py      # 40 exhaustive tests for 7 validation contracts
  test_edge_math.py      # Property-based tests (hypothesis) for Kelly, EMOS, budgets
scripts/
  build_bias_table.py    # Re-run monthly to refresh bias corrections
sql/
  schema.sql             # Reference DDL for PostgreSQL migration
```

## Refreshing Bias Corrections

```bash
# On hf-toybox-001
su - weather
cd weather-edge
source .venv/bin/activate
python scripts/build_bias_table.py
# Automatically updates bias_correction.py with latest 30-day data
# Then deploy: ./deploy.sh
```

## Migrating to PostgreSQL

```bash
dnf install postgresql-server
postgresql-setup --initdb
systemctl enable --now postgresql
sudo -u postgres createdb weather_edge
sqlite3 weather_edge.db .dump | psql weather_edge
# Update DATABASE_URL in .env
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Dashboard HTML |
| GET | `/api/state` | Full system state (WebSocket also available at `/ws`) |
| POST | `/api/start` | Start automated trading |
| POST | `/api/stop` | Stop trading (keep positions) |
| POST | `/api/close-all` | Close all positions at market price |
| POST | `/api/new-session` | Reset to fresh bankroll |
| POST | `/api/refresh` | Trigger immediate cycle |
| GET | `/api/weather-alerts` | Current weather alerts |
| POST | `/api/backtest` | Run historical backtest (params: days, cities) |
| GET | `/api/correlation-matrix` | City correlation data |
| GET | `/api/execution-analytics` | Trading metrics |
| WS | `/ws` | Real-time state updates |

## License

Private. Not for redistribution.
