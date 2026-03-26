# Weather Edge

Automated weather prediction market trading system for Polymarket. Exploits the gap between multi-model weather forecast consensus and crowd-implied market prices.

Inspired by [@ColdMath's Claude Trader](https://polymarket.com/profile/@ColdMath) ($76K positions, 4,307 predictions, 73% win rate).

## How It Works

1. Fetches forecasts from 6-8 weather models per city via Open-Meteo API
2. Applies data-driven NWS station bias corrections (30-day rolling calibration)
3. Computes weighted consensus with KDE distribution fitting
4. Detects bust-causing weather patterns (Chinook, Foehn, marine layer, etc.) and adjusts confidence
5. Discovers active Polymarket weather markets via Gamma API (130+ temperature markets daily)
6. Calculates edge (model probability vs market price) with quarter-Kelly sizing
7. Runs dual strategy: 70% core bets + 30% ColdMath-style penny tail bets
8. Snipes model drops, triggers immediate trades when ECMWF/GFS release fresh data before the market adjusts
9. Persists all trades to SQLite (survives restarts)

## Quick Start

```bash
cd /Volumes/2TB_HD/weather
source .venv/bin/activate
uvicorn weather_edge.dashboard.app:app --port 8000
```

Open `http://localhost:8000` and click **START**.

## Setup From Scratch

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,dashboard]"
pip install websockets
cp .env.example .env
```

No API keys needed for paper trading. Open-Meteo and Polymarket read endpoints are free/public.

Optional: set `ANTHROPIC_API_KEY` in `.env` for Claude reasoning layer (~$0.72/day).

## Architecture

```
Open-Meteo (6-8 models/city)
        |
        v
  Bias Correction (30-day calibrated)
        |
        v
  Pattern Detector (Chinook, Foehn, marine layer, lake breeze, etc.)
        |
        v
  KDE Consensus Engine (weighted by model skill)
        |                              Polymarket Gamma API
        v                                      |
  Edge Detection + Kelly Sizing  <---  Market Prices
        |
        +-- Core bets (70% bankroll, quarter-Kelly)
        +-- Tail bets (30% bankroll, penny sniping)
        |
        v
  Paper Trader / Live Executor  --->  SQLite Persistence
        |
        v
  Web Dashboard (FastAPI + WebSocket)
        |
  Model-Drop Sniper (probes every 2 min)
```

## Cities (21)

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

When a pattern is detected, trading confidence increases on the correctly-predicted side.

## Dual Strategy

- **Core bets (70% of bankroll)**, Medium-probability buckets where models strongly disagree with the market. Quarter-Kelly sizing. Higher win rate, steady returns.
- **Tail bets (30% of bankroll)**, ColdMath-style penny sniping. Buy YES on buckets priced at $0.01-$0.05 where our model says 3x+ higher probability. Most lose, but 50:1 payoffs on the winners compound.

## Dashboard Controls

- **START**, Begin automated trading (30-min cycles + 2-min sniper)
- **STOP**, Pause trading, keep positions open
- **CLOSE ALL**, Sell profitable positions, hold losers to resolution
- **NEW SESSION**, Reset to fresh $1,000 bankroll
- **REFRESH NOW**, Trigger immediate data cycle

## Claude Reasoning Layer

Claude API is wired into the trading loop as an intelligent filter on the top 3 signals each cycle:

- **Regular cycles**, Claude analyzes only when a new signal appears or edge shifts >5%
- **Sniper triggers**, Claude ALWAYS runs (model just dropped, timing-critical)
- Claude receives: individual model forecasts, consensus stats, edge calculation, market description
- Claude returns: should_trade (yes/no), confidence_adjustment (0.5-1.5x), rationale, risk factors
- If Claude says skip, the signal is dropped. If Claude adjusts confidence, position size scales accordingly
- Cost: ~$0.72/day (3 calls/cycle, ~48 cycles/day)

Set `ANTHROPIC_API_KEY` in `.env` to enable. System works without it but loses the qualitative reasoning layer.

## Competitor Tracking

Tracks public Polymarket profiles of known weather traders each cycle:

- **@ColdMath**, $77K+ positions, 4,300+ predictions, 73% win rate, ~$37K monthly P&L
- Snapshots positions value and prediction count every 30 minutes
- Calculates growth rate per day for comparison against our paper P&L
- Available via `/api/state` → `competitors` field

## Key Features

- **21 cities** across 3 continents with 130+ daily temperature markets
- **KDE consensus**, Handles multimodal distributions when models disagree
- **Real bias corrections**, 30-day rolling calibration via Open-Meteo historical APIs
- **Pattern detection**, Chinook, Foehn, marine layer, lake breeze, cold pool, Santa Ana
- **Model-drop sniper**, Detects ECMWF/GFS/HRRR updates within 2 min, trades before market adjusts
- **Bucket parity arbitrage**, Flags when bucket YES prices sum to >1.05
- **Claude reasoning layer**, LLM analysis of top signals with confidence adjustment per trade
- **Competitor tracking**, Monitors @ColdMath's public stats for performance comparison
- **Quarter-Kelly sizing**, Conservative position sizing (ColdMath-validated)
- **Smart close**, Only sells winners; holds losers to resolution (capped downside)
- **SQLite persistence**, Sessions and trades survive restarts
- **Dual core/tail strategy**, Steady returns + asymmetric penny bets

## Configuration

Edit `.env`:

```
BANKROLL=1000.0          # Starting capital
MIN_EDGE=0.05            # Minimum edge to trade (5%)
MIN_CONFIDENCE=0.6       # Minimum model confidence
KELLY_FRACTION=0.25      # Quarter-Kelly (conservative)
MAX_POSITION_PCT=0.05    # Max 5% of bankroll per trade
FETCH_INTERVAL_MINUTES=30
```

## Project Structure

```
src/weather_edge/
  config.py              # 21 cities, model weights, settings
  scheduler.py           # Main orchestration loop
  persistence.py         # SQLite session/trade storage
  cli.py                 # CLI commands
  reporting.py           # Rich terminal tables
  models/
    enums.py             # City, WeatherModel, MarketType
    orm.py               # SQLAlchemy ORM (for PG migration)
  fetchers/
    openmeteo.py         # Multi-model forecast fetcher
    polymarket.py        # Market discovery + price fetcher
  analysis/
    consensus.py         # Weighted KDE consensus engine
    edge.py              # Edge detection + dual Kelly sizing
    market_mapper.py     # Maps markets to forecast variables
    bias_correction.py   # Data-driven station bias (30-day calibrated)
    pattern_detector.py  # Chinook/Foehn/marine layer/etc detection
    arbitrage.py         # Bucket parity checks
    model_timing.py      # Golden window detection
    sniper.py            # Model-drop triggered trading
    claude_reasoning.py  # Claude API trade analysis (wired into loop)
    competitor_tracker.py # @ColdMath and whale tracking
  trading/
    paper.py             # Paper trader with core/tail split + smart close
    executor.py          # Real Polymarket CLOB execution (limit orders)
  dashboard/
    app.py               # FastAPI + WebSocket backend
    templates/
      index.html         # Dark terminal-aesthetic dashboard
scripts/
  build_bias_table.py    # Re-run monthly to refresh bias corrections
```

## Refreshing Bias Corrections

```bash
source .venv/bin/activate
python scripts/build_bias_table.py
# Automatically updates bias_correction.py with latest 30-day data
```

## Migrating to PostgreSQL

```bash
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=weather_edge postgres:16
sqlite3 weather_edge.db .dump | psql weather_edge
# Update DATABASE_URL in .env
```

## License

Private. Not for redistribution.
