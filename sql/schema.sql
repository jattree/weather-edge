-- Weather Edge: Reference DDL
-- Managed by Alembic migrations, this file is for human reference only.

CREATE TABLE cities (
    city_id     VARCHAR(10) PRIMARY KEY,
    name        VARCHAR(50) NOT NULL,
    icao        VARCHAR(10) NOT NULL,
    latitude    DOUBLE PRECISION NOT NULL,
    longitude   DOUBLE PRECISION NOT NULL,
    timezone    VARCHAR(50) NOT NULL
);

CREATE TABLE forecasts (
    forecast_id SERIAL PRIMARY KEY,
    city_id     VARCHAR(10) NOT NULL,
    model_name  VARCHAR(50) NOT NULL,
    fetched_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    target_date DATE NOT NULL,
    temperature_2m  JSONB,
    precipitation   JSONB,
    snowfall        JSONB,
    wind_speed_10m  JSONB,
    temp_max_c      DOUBLE PRECISION,
    temp_min_c      DOUBLE PRECISION,
    precip_sum_mm   DOUBLE PRECISION,
    snow_sum_cm     DOUBLE PRECISION,
    wind_max_kmh    DOUBLE PRECISION,
    raw_response    JSONB
);
CREATE INDEX idx_forecasts_city_date ON forecasts(city_id, target_date);
CREATE INDEX idx_forecasts_fetched ON forecasts(fetched_at);

CREATE TABLE consensus (
    consensus_id SERIAL PRIMARY KEY,
    city_id      VARCHAR(10) NOT NULL,
    target_date  DATE NOT NULL,
    computed_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    variable     VARCHAR(30) NOT NULL,
    model_count  INTEGER NOT NULL,
    mean_value   DOUBLE PRECISION NOT NULL,
    median_value DOUBLE PRECISION NOT NULL,
    std_dev      DOUBLE PRECISION NOT NULL,
    min_value    DOUBLE PRECISION NOT NULL,
    max_value    DOUBLE PRECISION NOT NULL,
    model_values  JSONB NOT NULL,
    model_weights JSONB,
    threshold_probs JSONB
);
CREATE INDEX idx_consensus_lookup ON consensus(city_id, target_date, variable);

CREATE TABLE markets (
    market_id       VARCHAR(100) PRIMARY KEY,
    token_id_yes    VARCHAR(200),
    token_id_no     VARCHAR(200),
    city_id         VARCHAR(10),
    market_type     VARCHAR(20) NOT NULL,
    description     TEXT NOT NULL,
    target_date     DATE NOT NULL,
    threshold_value DOUBLE PRECISION NOT NULL,
    threshold_dir   VARCHAR(10) NOT NULL,
    threshold_unit  VARCHAR(20) NOT NULL,
    resolution_source VARCHAR(20) DEFAULT 'nws',
    resolved        BOOLEAN DEFAULT FALSE,
    outcome         BOOLEAN,
    created_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_markets_city_date ON markets(city_id, target_date);

CREATE TABLE market_prices (
    price_id    SERIAL PRIMARY KEY,
    market_id   VARCHAR(100) NOT NULL,
    fetched_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    bid         DOUBLE PRECISION,
    ask         DOUBLE PRECISION,
    midpoint    DOUBLE PRECISION,
    spread      DOUBLE PRECISION,
    volume_24h  DOUBLE PRECISION,
    liquidity   DOUBLE PRECISION
);
CREATE INDEX idx_prices_market ON market_prices(market_id, fetched_at);

CREATE TABLE signals (
    signal_id       SERIAL PRIMARY KEY,
    market_id       VARCHAR(100) NOT NULL,
    consensus_id    INTEGER,
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_prob      DOUBLE PRECISION NOT NULL,
    model_confidence DOUBLE PRECISION NOT NULL,
    market_prob     DOUBLE PRECISION NOT NULL,
    edge            DOUBLE PRECISION NOT NULL,
    edge_pct        DOUBLE PRECISION NOT NULL,
    kelly_fraction  DOUBLE PRECISION NOT NULL,
    half_kelly      DOUBLE PRECISION NOT NULL,
    recommended_side VARCHAR(5) NOT NULL,
    recommended_size DOUBLE PRECISION,
    confidence_tier  VARCHAR(10) NOT NULL
);
CREATE INDEX idx_signals_market ON signals(market_id, computed_at);

CREATE TABLE paper_trades (
    trade_id    SERIAL PRIMARY KEY,
    signal_id   INTEGER NOT NULL,
    market_id   VARCHAR(100) NOT NULL,
    city_id     VARCHAR(10) NOT NULL,
    side        VARCHAR(5) NOT NULL,
    size_usd    DOUBLE PRECISION NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    placed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    description TEXT,
    exit_price  DOUBLE PRECISION,
    resolved_at TIMESTAMPTZ,
    pnl         DOUBLE PRECISION,
    status      VARCHAR(10) DEFAULT 'open'
);

CREATE TABLE trades (
    trade_id            SERIAL PRIMARY KEY,
    signal_id           INTEGER,
    market_id           VARCHAR(100) NOT NULL,
    polymarket_order_id VARCHAR(200),
    side                VARCHAR(5) NOT NULL,
    size_usd            DOUBLE PRECISION NOT NULL,
    entry_price         DOUBLE PRECISION NOT NULL,
    placed_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    status              VARCHAR(20) DEFAULT 'pending',
    exit_price          DOUBLE PRECISION,
    resolved_at         TIMESTAMPTZ,
    pnl                 DOUBLE PRECISION
);
