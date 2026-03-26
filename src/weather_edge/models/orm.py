"""SQLAlchemy ORM table definitions."""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class CityRow(Base):
    __tablename__ = "cities"

    city_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    icao: Mapped[str] = mapped_column(String(10), nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    timezone: Mapped[str] = mapped_column(String(50), nullable=False)


class ForecastRow(Base):
    __tablename__ = "forecasts"

    forecast_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    city_id: Mapped[str] = mapped_column(String(10), nullable=False)
    model_name: Mapped[str] = mapped_column(String(50), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    target_date: Mapped[date] = mapped_column(Date, nullable=False)

    # Hourly values as JSON arrays
    temperature_2m: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    precipitation: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    snowfall: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    wind_speed_10m: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Daily aggregates
    temp_max_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    temp_min_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    precip_sum_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    snow_sum_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_max_kmh: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Full API response for auditing
    raw_response: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        Index("idx_forecasts_city_date", "city_id", "target_date"),
        Index("idx_forecasts_fetched", "fetched_at"),
    )


class ConsensusRow(Base):
    __tablename__ = "consensus"

    consensus_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    city_id: Mapped[str] = mapped_column(String(10), nullable=False)
    target_date: Mapped[date] = mapped_column(Date, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    variable: Mapped[str] = mapped_column(String(30), nullable=False)
    model_count: Mapped[int] = mapped_column(Integer, nullable=False)
    mean_value: Mapped[float] = mapped_column(Float, nullable=False)
    median_value: Mapped[float] = mapped_column(Float, nullable=False)
    std_dev: Mapped[float] = mapped_column(Float, nullable=False)
    min_value: Mapped[float] = mapped_column(Float, nullable=False)
    max_value: Mapped[float] = mapped_column(Float, nullable=False)
    model_values: Mapped[dict] = mapped_column(JSONB, nullable=False)
    model_weights: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    threshold_probs: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        Index("idx_consensus_lookup", "city_id", "target_date", "variable"),
    )


class MarketRow(Base):
    __tablename__ = "markets"

    market_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    token_id_yes: Mapped[str | None] = mapped_column(String(200), nullable=True)
    token_id_no: Mapped[str | None] = mapped_column(String(200), nullable=True)
    city_id: Mapped[str | None] = mapped_column(String(10), nullable=True)
    market_type: Mapped[str] = mapped_column(String(20), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    target_date: Mapped[date] = mapped_column(Date, nullable=False)
    threshold_value: Mapped[float] = mapped_column(Float, nullable=False)
    threshold_dir: Mapped[str] = mapped_column(String(10), nullable=False)  # gte, lte, range
    threshold_unit: Mapped[str] = mapped_column(String(20), nullable=False)
    resolution_source: Mapped[str] = mapped_column(String(20), default="nws")
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    outcome: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_markets_city_date", "city_id", "target_date"),
    )


class MarketPriceRow(Base):
    __tablename__ = "market_prices"

    price_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[str] = mapped_column(String(100), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    bid: Mapped[float | None] = mapped_column(Float, nullable=True)
    ask: Mapped[float | None] = mapped_column(Float, nullable=True)
    midpoint: Mapped[float | None] = mapped_column(Float, nullable=True)
    spread: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume_24h: Mapped[float | None] = mapped_column(Float, nullable=True)
    liquidity: Mapped[float | None] = mapped_column(Float, nullable=True)

    __table_args__ = (
        Index("idx_prices_market", "market_id", "fetched_at"),
    )


class SignalRow(Base):
    __tablename__ = "signals"

    signal_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[str] = mapped_column(String(100), nullable=False)
    consensus_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    model_prob: Mapped[float] = mapped_column(Float, nullable=False)
    model_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    market_prob: Mapped[float] = mapped_column(Float, nullable=False)

    edge: Mapped[float] = mapped_column(Float, nullable=False)
    edge_pct: Mapped[float] = mapped_column(Float, nullable=False)

    kelly_fraction: Mapped[float] = mapped_column(Float, nullable=False)
    half_kelly: Mapped[float] = mapped_column(Float, nullable=False)
    recommended_side: Mapped[str] = mapped_column(String(5), nullable=False)
    recommended_size: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_tier: Mapped[str] = mapped_column(String(10), nullable=False)

    __table_args__ = (
        Index("idx_signals_market", "market_id", "computed_at"),
    )


class PaperTradeRow(Base):
    __tablename__ = "paper_trades"

    trade_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_id: Mapped[int] = mapped_column(Integer, nullable=False)
    market_id: Mapped[str] = mapped_column(String(100), nullable=False)
    city_id: Mapped[str] = mapped_column(String(10), nullable=False)
    side: Mapped[str] = mapped_column(String(5), nullable=False)
    size_usd: Mapped[float] = mapped_column(Float, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    placed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(10), default="open")
