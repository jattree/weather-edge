"""Configuration for the weather edge system."""
from __future__ import annotations

from dataclasses import dataclass

from pydantic_settings import BaseSettings

from weather_edge.models.enums import City, WeatherModel


@dataclass(frozen=True)
class CityConfig:
    city_id: City
    name: str
    icao: str
    latitude: float
    longitude: float
    timezone: str
    regional_models: list[WeatherModel]
    temp_unit: str = "fahrenheit"  # or "celsius"


# All cities matching @ColdMath's dashboard + Polymarket weather markets
CITIES: dict[City, CityConfig] = {
    City.LON: CityConfig(
        city_id=City.LON,
        name="London",
        icao="EGLC",
        latitude=51.5053,
        longitude=0.0553,
        timezone="Europe/London",
        regional_models=[WeatherModel.UKV],
        temp_unit="celsius",
    ),
    City.NYC: CityConfig(
        city_id=City.NYC,
        name="New York",
        icao="LGA",
        latitude=40.7743,
        longitude=-73.8726,
        timezone="America/New_York",
        regional_models=[WeatherModel.HRRR, WeatherModel.NAM],
    ),
    City.DAL: CityConfig(
        city_id=City.DAL,
        name="Dallas",
        icao="KDAL",
        latitude=32.8471,
        longitude=-96.8518,
        timezone="America/Chicago",
        regional_models=[WeatherModel.HRRR, WeatherModel.NAM],
    ),
    City.SEA: CityConfig(
        city_id=City.SEA,
        name="Seattle",
        icao="KSEA",
        latitude=47.4502,
        longitude=-122.3088,
        timezone="America/Los_Angeles",
        regional_models=[WeatherModel.HRRR, WeatherModel.NAM],
    ),
    City.ATL: CityConfig(
        city_id=City.ATL,
        name="Atlanta",
        icao="KATL",
        latitude=33.6407,
        longitude=-84.4277,
        timezone="America/New_York",
        regional_models=[WeatherModel.HRRR, WeatherModel.NAM],
    ),
    City.TOR: CityConfig(
        city_id=City.TOR,
        name="Toronto",
        icao="CYYZ",
        latitude=43.6777,
        longitude=-79.6248,
        timezone="America/Toronto",
        regional_models=[WeatherModel.HRDPS, WeatherModel.NAM],
    ),
    City.BUE: CityConfig(
        city_id=City.BUE,
        name="Buenos Aires",
        icao="SAEZ",
        latitude=-34.8222,
        longitude=-58.5358,
        timezone="America/Argentina/Buenos_Aires",
        regional_models=[],
        temp_unit="celsius",
    ),
    City.SEL: CityConfig(
        city_id=City.SEL,
        name="Seoul",
        icao="RKSI",
        latitude=37.4691,
        longitude=126.4505,
        timezone="Asia/Seoul",
        regional_models=[WeatherModel.KMA],
        temp_unit="celsius",
    ),
    City.CHI: CityConfig(
        city_id=City.CHI,
        name="Chicago",
        icao="KORD",
        latitude=41.9742,
        longitude=-87.9073,
        timezone="America/Chicago",
        regional_models=[WeatherModel.HRRR, WeatherModel.NAM],
    ),
    City.MIA: CityConfig(
        city_id=City.MIA,
        name="Miami",
        icao="KMIA",
        latitude=25.7959,
        longitude=-80.2870,
        timezone="America/New_York",
        regional_models=[WeatherModel.HRRR, WeatherModel.NAM],
    ),
    City.LAX: CityConfig(
        city_id=City.LAX,
        name="Los Angeles",
        icao="KLAX",
        latitude=33.9425,
        longitude=-118.4081,
        timezone="America/Los_Angeles",
        regional_models=[WeatherModel.HRRR, WeatherModel.NAM],
    ),
    City.HOU: CityConfig(
        city_id=City.HOU,
        name="Houston",
        icao="KIAH",
        latitude=29.9844,
        longitude=-95.3414,
        timezone="America/Chicago",
        regional_models=[WeatherModel.HRRR, WeatherModel.NAM],
    ),
    City.DEN: CityConfig(
        city_id=City.DEN,
        name="Denver",
        icao="KDEN",
        latitude=39.8561,
        longitude=-104.6737,
        timezone="America/Denver",
        regional_models=[WeatherModel.HRRR, WeatherModel.NAM],
    ),
    City.SFO: CityConfig(
        city_id=City.SFO,
        name="San Francisco",
        icao="KSFO",
        latitude=37.6213,
        longitude=-122.3790,
        timezone="America/Los_Angeles",
        regional_models=[WeatherModel.HRRR, WeatherModel.NAM],
    ),
    City.SHA: CityConfig(
        city_id=City.SHA,
        name="Shanghai",
        icao="ZSPD",
        latitude=31.1443,
        longitude=121.8083,
        timezone="Asia/Shanghai",
        regional_models=[],
        temp_unit="celsius",
    ),
    City.MAD: CityConfig(
        city_id=City.MAD,
        name="Madrid",
        icao="LEMD",
        latitude=40.4936,
        longitude=-3.5668,
        timezone="Europe/Madrid",
        regional_models=[],
        temp_unit="celsius",
    ),
    City.TYO: CityConfig(
        city_id=City.TYO,
        name="Tokyo",
        icao="RJTT",
        latitude=35.5523,
        longitude=139.7798,
        timezone="Asia/Tokyo",
        regional_models=[],
        temp_unit="celsius",
    ),
    City.HKG: CityConfig(
        city_id=City.HKG,
        name="Hong Kong",
        icao="VHHH",
        latitude=22.3080,
        longitude=113.9185,
        timezone="Asia/Hong_Kong",
        regional_models=[],
        temp_unit="celsius",
    ),
    City.MUC: CityConfig(
        city_id=City.MUC,
        name="Munich",
        icao="EDDM",
        latitude=48.3537,
        longitude=11.7750,
        timezone="Europe/Berlin",
        regional_models=[],
        temp_unit="celsius",
    ),
    City.WAR: CityConfig(
        city_id=City.WAR,
        name="Warsaw",
        icao="EPWA",
        latitude=52.1657,
        longitude=20.9671,
        timezone="Europe/Warsaw",
        regional_models=[],
        temp_unit="celsius",
    ),
    City.SZN: CityConfig(
        city_id=City.SZN,
        name="Shenzhen",
        icao="ZGSZ",
        latitude=22.6393,
        longitude=113.8107,
        timezone="Asia/Shanghai",
        regional_models=[],
        temp_unit="celsius",
    ),
    City.AUS: CityConfig(
        city_id=City.AUS,
        name="Austin",
        icao="KAUS",
        latitude=30.1945,
        longitude=-97.6699,
        timezone="America/Chicago",
        regional_models=[WeatherModel.HRRR, WeatherModel.NAM],
    ),
    City.WLG: CityConfig(
        city_id=City.WLG,
        name="Wellington",
        icao="NZWN",
        latitude=-41.3272,
        longitude=174.8050,
        timezone="Pacific/Auckland",
        regional_models=[],
        temp_unit="celsius",
    ),
    City.LKO: CityConfig(
        city_id=City.LKO,
        name="Lucknow",
        icao="VILK",
        latitude=26.8467,
        longitude=80.9462,
        timezone="Asia/Kolkata",
        regional_models=[],
        temp_unit="celsius",
    ),
}

# Global models applied to every city
GLOBAL_MODELS: list[WeatherModel] = [
    WeatherModel.ECMWF,
    WeatherModel.GFS,
    WeatherModel.ICON,
    WeatherModel.GEM,
    WeatherModel.JMA,
    WeatherModel.METEOFRANCE,
]

# Model skill weights, regional models get a boost in their coverage area
# These are relative weights, will be normalized per city
MODEL_BASE_WEIGHT: dict[WeatherModel, float] = {
    WeatherModel.ECMWF: 1.3,  # Generally most skillful global model
    WeatherModel.GFS: 1.0,
    WeatherModel.ICON: 1.0,
    WeatherModel.GEM: 0.9,
    WeatherModel.JMA: 0.9,
    WeatherModel.METEOFRANCE: 1.0,
    WeatherModel.HRRR: 1.5,  # Best US short-range
    WeatherModel.NAM: 1.2,   # Good North America
    WeatherModel.UKV: 1.5,   # Best UK
    WeatherModel.HRDPS: 1.4,  # Best Canada
    WeatherModel.KMA: 1.4,   # Best Korea
}


def get_models_for_city(city_id: City) -> list[WeatherModel]:
    """Return all applicable models for a city (global + regional)."""
    city = CITIES[city_id]
    return GLOBAL_MODELS + city.regional_models


def get_model_weights(city_id: City) -> dict[WeatherModel, float]:
    """Return normalized model weights for a city.

    Uses adaptive Brier-weighted scores when available (30+ forecasts
    per model). Falls back to static MODEL_BASE_WEIGHT otherwise.
    """
    # Try adaptive weights from self-learning module
    try:
        from weather_edge.analysis.learner import get_adaptive_weights
        from weather_edge.persistence import PersistentStore
        store = PersistentStore()
        adaptive = get_adaptive_weights(store, city_id.value)
        store.close()
        if adaptive:
            models = get_models_for_city(city_id)
            return {
                m: adaptive.get(m.value, 1.0 / len(models))
                for m in models
            }
    except Exception:
        pass

    # Fall back to static weights
    models = get_models_for_city(city_id)
    raw = {m: MODEL_BASE_WEIGHT.get(m, 1.0) for m in models}
    total = sum(raw.values())
    return {m: w / total for m, w in raw.items()}


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/weather_edge"
    bankroll: float = 2000.0
    min_edge: float = 0.05
    min_confidence: float = 0.6
    kelly_fraction: float = 0.25
    max_position_pct: float = 0.03
    fetch_interval_minutes: int = 30

    # Pool allocation (hybrid: penny-first targeting, sustainable at $2K)
    # Post-Monday fee cliff: penny bets are fee-immune, core trades get hit 1.25%
    pool_today_pct: float = 0.35    # Same-day core trades (reduced, fee-penalised)
    pool_tomorrow_pct: float = 0.25  # Tomorrow conviction bets
    pool_penny_pct: float = 0.40     # Penny sniping (ColdMath's profit engine, fee-immune)

    # Penny sweep constraints (ColdMath targets 0.1-3c, $15-25 per position)
    penny_min_edge_multiplier: float = 3.0  # Model must say 3x market price
    penny_min_position: float = 10.0        # Min $10 per penny bet
    penny_max_position: float = 50.0        # Max $50 per penny bet (up from $20)
    penny_max_entry_price: float = 0.05     # Only enter at 5c or below (was 6c)

    # Order splitting for penny bets (avoid moving thin markets)
    penny_order_split_max_shares: int = 5   # Max shares per order (ColdMath uses 1-5)

    # API keys
    anthropic_api_key: str = ""  # Claude reasoning layer
    gemini_api_key: str = ""  # Gemini red team / dissent layer
    gribstream_api_key: str = ""  # GribStream AI models (GraphCast, AIFS)
    openmeteo_base_url: str = "https://api.open-meteo.com/v1/forecast"
    openmeteo_api_key: str = ""  # Customer API key for paid tier
    polymarket_gamma_url: str = "https://gamma-api.polymarket.com"
    polymarket_clob_url: str = "https://clob.polymarket.com"
    polymarket_api_key: str = ""  # CLOB API key for live execution
    polymarket_private_key: str = ""  # Polygon wallet private key
    polymarket_wallet: str = ""  # Polygon wallet address

    # Open-Meteo tier, auto-detected from api key presence
    # Paid tier: 1M req/month, dedicated servers, no rate-limit ban risk
    openmeteo_paid_tier: bool = False

    @property
    def effective_openmeteo_url(self) -> str:
        """Use customer endpoint when API key is set."""
        if self.openmeteo_api_key:
            return "https://customer-api.open-meteo.com/v1/forecast"
        return self.openmeteo_base_url


settings = Settings()
