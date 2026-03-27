"""Shared enumerations for the weather edge system."""
from enum import Enum


class City(str, Enum):
    NYC = "nyc"
    LON = "lon"
    DAL = "dal"
    SEA = "sea"
    ATL = "atl"
    TOR = "tor"
    BUE = "bue"
    SEL = "sel"
    CHI = "chi"
    MIA = "mia"
    LAX = "lax"
    HOU = "hou"
    DEN = "den"
    SFO = "sfo"
    SHA = "sha"
    MAD = "mad"
    TYO = "tyo"
    HKG = "hkg"
    MUC = "muc"
    WAR = "war"
    SZN = "szn"
    AUS = "aus"  # Austin, ColdMath trades this actively
    WLG = "wlg"  # Wellington, ColdMath's $7K+ wins, thin market
    LKO = "lko"  # Lucknow, ColdMath's $6.8K win at 0.1c, thin market


class WeatherModel(str, Enum):
    # Global models
    ECMWF = "ecmwf_ifs025"
    GFS = "gfs_seamless"
    ICON = "icon_seamless"
    GEM = "gem_seamless"
    JMA = "jma_seamless"
    METEOFRANCE = "meteofrance_seamless"
    # Regional models
    HRRR = "ncep_hrrr_conus"
    NAM = "ncep_nam_conus"
    UKV = "ukmo_seamless"
    HRDPS = "gem_hrdps_continental"
    KMA = "kma_seamless"


class MarketType(str, Enum):
    TEMP_HIGH = "temp_high"
    TEMP_LOW = "temp_low"
    PRECIP = "precip"
    SNOW = "snow"


class SignalTier(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TradeSide(str, Enum):
    YES = "YES"
    NO = "NO"


class TradeStatus(str, Enum):
    OPEN = "open"
    WON = "won"
    LOST = "lost"
