"""Regression tests for the signal / resolution review fixes.

One class per finding. No network: every HTTP client is replaced by a fake.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import sqlite3
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from weather_edge.analysis import consensus as consensus_mod
from weather_edge.analysis.bias_correction import compute_bias_from_store
from weather_edge.analysis.consensus import (
    _blend_threshold_probs,
    compute_consensus,
    get_probability_for_threshold,
)
from weather_edge.analysis.edge import calculate_edge
from weather_edge.analysis.resolver import (
    BucketInfo,
    _extract_target_date_from_trade,
    actual_falls_in_bucket,
    parse_bucket_from_description,
)
from weather_edge.fetchers.metar import _daily_max_from_rows, _point_temp_c_from_metar
from weather_edge.fetchers.openmeteo import ForecastResult, f_to_c
from weather_edge.fetchers.polymarket import (
    MarketInfo,
    parse_market_question,
    parse_target_date,
)
from weather_edge.models.enums import City, SignalTier
from weather_edge.persistence import PersistentStore
from weather_edge.scheduler import (
    MODEL_AGREEMENT_MAX_STD_C,
    _bucket_celsius_band,
    _model_agreement_std,
    compute_model_prob_for_market,
)
from weather_edge.trading.paper import PaperTrade

ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakeResponse:
    def __init__(self, payload, status_code: int = 200, text: str = ""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """Records GET params and replays canned payloads."""

    calls: list[tuple[str, dict]] = []
    payloads: list = []

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, params=None, **kw):
        type(self).calls.append((url, dict(params or {})))
        payload = type(self).payloads.pop(0) if type(self).payloads else {}
        return _FakeResponse(payload)


@pytest.fixture
def fake_client():
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.payloads = []
    return _FakeAsyncClient


# ---------------------------------------------------------------------------
# 1. Bias significance gate: one error per target date
# ---------------------------------------------------------------------------

class TestBiasOnePerDate:

    def _store(self):
        return PersistentStore(":memory:")

    def test_duplicate_snapshots_do_not_inflate_n(self):
        """45 snapshots/day for 3 days is 3 samples, not 90 -> insufficient."""
        store = self._store()
        for day in range(1, 4):
            for _ in range(45):
                store.save_forecast_snapshot("lon", f"2026-03-0{day}", {"ecmwf_ifs025": 12.5})
            store.backfill_actual("lon", f"2026-03-0{day}", 12.0)
        corr = compute_bias_from_store(store, "ecmwf_ifs025", City.LON)
        assert corr.temp_max_offset == 0.0
        assert corr.notes == "insufficient data"

    def test_noise_bias_gated_even_with_many_snapshots(self):
        """Per-day errors +1/-0.8 alternating: mean +0.1, noise ~0.9.

        Counted per-date (n=20) the bias is well inside 2 SE -> gated. The old
        LIMIT-90-raw-rows query saw ~2 dates x 45 identical snapshots and
        passed the gate.
        """
        store = self._store()
        for i in range(20):
            d = f"2026-03-{i + 1:02d}"
            err = 1.0 if i % 2 == 0 else -0.8
            for _ in range(45):
                store.save_forecast_snapshot("lon", d, {"gfs_seamless": 10.0 + err})
            store.backfill_actual("lon", d, 10.0)
        corr = compute_bias_from_store(store, "gfs_seamless", City.LON)
        assert corr.temp_max_offset == 0.0
        assert "gated" in corr.notes and "n=20" in corr.notes

    def test_daily_history_one_row_per_date(self):
        store = self._store()
        for _ in range(5):
            store.save_forecast_snapshot("nyc", "2026-03-01", {"m": 20.0})
        store.save_forecast_snapshot("nyc", "2026-03-01", {"m": 26.0})
        store.backfill_actual("nyc", "2026-03-01", 21.0)
        rows = store.get_daily_forecast_history("m", "nyc")
        assert len(rows) == 1
        assert rows[0]["n_snapshots"] == 6
        assert rows[0]["forecast_value"] == pytest.approx(21.0)

    def test_hindcast_rerun_is_idempotent(self):
        hindcast = _load_script("hindcast")
        conn = sqlite3.connect(":memory:")
        conn.execute(
            """CREATE TABLE forecast_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT, trade_id INTEGER,
                city_id TEXT NOT NULL, target_date TEXT NOT NULL,
                model_name TEXT NOT NULL, forecast_value REAL,
                actual_value REAL, created_at TEXT NOT NULL)"""
        )
        # A live scheduler snapshot for the same key must survive.
        conn.execute(
            "INSERT INTO forecast_snapshots (trade_id, city_id, target_date, model_name,"
            " forecast_value, actual_value, created_at) VALUES"
            " (NULL, 'nyc', '2026-03-01', 'm', 19.0, NULL, '2026-02-28T12:00:00.123456+00:00')"
        )
        batch = [(None, "nyc", "2026-03-01", "m", 20.0, 21.0, "2026-03-10T00:00:00+00:00")]
        hindcast.upsert_hindcast_rows(conn, batch)
        batch2 = [(None, "nyc", "2026-03-01", "m", 20.5, 21.0, "2026-03-11T00:00:00+00:00")]
        hindcast.upsert_hindcast_rows(conn, batch2)
        rows = conn.execute(
            "SELECT forecast_value FROM forecast_snapshots ORDER BY id"
        ).fetchall()
        assert [r[0] for r in rows] == [19.0, 20.5]


# ---------------------------------------------------------------------------
# 2. Forecasts on the station's local civil day
# ---------------------------------------------------------------------------

class TestLocalDayForecasts:

    def test_city_forecasts_request_local_timezone(self, monkeypatch, fake_client):
        from weather_edge.analysis import service_health
        from weather_edge.fetchers import openmeteo
        monkeypatch.setattr(openmeteo.httpx, "AsyncClient", fake_client)
        # Health recording tries Redis (localhost); keep the test offline.
        monkeypatch.setattr(service_health, "record_service_call", lambda *a, **k: None)
        asyncio.run(openmeteo.fetch_city_forecasts(City.WLG, date(2026, 3, 27)))
        assert fake_client.calls[0][1]["timezone"] == "Pacific/Auckland"

    def test_single_model_forecast_requests_local_timezone(self, monkeypatch, fake_client):
        from weather_edge.fetchers import openmeteo
        from weather_edge.models.enums import WeatherModel

        async def run():
            async with fake_client() as client:
                await openmeteo.fetch_model_forecast(
                    client, City.NYC, WeatherModel.GFS, date(2026, 3, 27), "http://x",
                )
        asyncio.run(run())
        assert fake_client.calls[0][1]["timezone"] == "America/New_York"

    def test_hindcast_forecasts_request_local_timezone(self, monkeypatch):
        hindcast = _load_script("hindcast")
        seen = []

        def fake_get(url, params=None, **kw):
            seen.append(params)
            return _FakeResponse({"daily": {"time": [], "temperature_2m_max": []}})
        monkeypatch.setattr(hindcast.httpx, "get", fake_get)
        hindcast.fetch_batch_forecasts(0, 0, date(2026, 3, 1), date(2026, 3, 2), "m",
                                       station_tz="Asia/Seoul")
        hindcast.fetch_historical_forecast(0, 0, date(2026, 3, 1), "m",
                                           station_tz="Asia/Seoul")
        assert all(p["timezone"] == "Asia/Seoul" for p in seen)

    def test_hindcast_no_reanalysis_by_default(self, monkeypatch):
        hindcast = _load_script("hindcast")
        monkeypatch.setattr(hindcast, "fetch_batch_metar_observations", lambda *a, **k: {})

        def boom(*a, **k):
            raise AssertionError("reanalysis must not be fetched by default")
        monkeypatch.setattr(hindcast.httpx, "get", boom)
        assert hindcast.fetch_batch_observations(
            0, 0, date(2026, 3, 1), date(2026, 3, 2), icao="KLGA",
        ) == {}


# ---------------------------------------------------------------------------
# 3. Hong Kong: resolver and probability band agree on [L, L+1)
# ---------------------------------------------------------------------------

def _hkg_market(lo, hi, direction):
    return MarketInfo(
        market_id="h", city_id=City.HKG, threshold_unit="celsius",
        threshold_dir=direction, bucket_low_int=lo, bucket_high_int=hi,
    )


def _in_band(x, band):
    lo, hi = band
    return (lo is None or x >= lo) and (hi is None or x < hi)


class TestHongKongConsistency:

    @pytest.mark.parametrize("reading", [27.9, 28.0, 28.4, 28.5, 28.9, 29.0, 29.3])
    def test_resolver_matches_band(self, reading):
        cases = [
            (BucketInfo(28.0, 29.0, "celsius", exclusive_upper=True),
             _hkg_market(28, 28, "range")),
            (BucketInfo(None, 28.0, "celsius"), _hkg_market(None, 28, "lte")),
            (BucketInfo(28.0, None, "celsius"), _hkg_market(28, None, "gte")),
        ]
        for bucket, market in cases:
            resolved = actual_falls_in_bucket(reading, bucket, is_hkg=True)
            assert resolved == _in_band(reading, _bucket_celsius_band(market)), (
                reading, bucket,
            )

    def test_or_below_tail_includes_decimal_part(self):
        # 28.5 has whole-degree label 28 -> inside "28°C or below".
        assert actual_falls_in_bucket(28.5, BucketInfo(None, 28.0, "celsius"), is_hkg=True)
        assert not actual_falls_in_bucket(29.0, BucketInfo(None, 28.0, "celsius"), is_hkg=True)

    def test_non_hkg_celsius_still_half_up(self):
        band = _bucket_celsius_band(MarketInfo(
            market_id="s", city_id=City.SEL, threshold_unit="celsius",
            threshold_dir="range", bucket_low_int=28, bucket_high_int=28,
        ))
        assert band == (27.5, 28.5)


# ---------------------------------------------------------------------------
# 4. "or higher" / "or lower" tail wording
# ---------------------------------------------------------------------------

class TestTailWording:

    def test_or_higher_market_question(self):
        q = "Will the highest temperature in Shanghai be 21°C or higher on March 27?"
        m = parse_market_question(q, "Highest temperature in Shanghai on March 27?", "c1")
        assert m is not None
        assert m.threshold_dir == "gte" and m.bucket_low_int == 21 and m.city_id == City.SHA

    def test_or_lower_market_question(self):
        q = "Will the highest temperature in Denver be 49°F or lower on March 27?"
        m = parse_market_question(q, "Highest temperature in Denver on March 27?", "c2")
        assert m is not None
        assert m.threshold_dir == "lte" and m.bucket_high_int == 49

    @pytest.mark.parametrize("desc,low,high", [
        ("Will the highest temperature in Shanghai be 21°C or higher on March 27?", 21.0, None),
        ("be 60°F or more on March 27", 60.0, None),
        ("be -2°C or lower on March 27", None, -2.0),
        ("be 40°F or less on March 27", None, 40.0),
    ])
    def test_resolver_parses_variants(self, desc, low, high):
        b = parse_bucket_from_description(desc)
        assert b is not None
        assert (b.low, b.high) == (low, high)


# ---------------------------------------------------------------------------
# 5. Year rollover
# ---------------------------------------------------------------------------

class TestYearRollover:

    def test_december_market_checked_in_january(self):
        d = parse_target_date("Highest temperature in NYC on December 31?",
                              reference=date(2027, 1, 2))
        assert d == date(2026, 12, 31)

    def test_january_market_listed_in_december(self):
        d = parse_target_date("Highest temperature in NYC on January 1?",
                              reference=date(2026, 12, 30))
        assert d == date(2027, 1, 1)

    def test_end_date_used_as_reference(self):
        d = parse_target_date("Highest temperature in NYC on December 31?",
                              fallback_end_date="2026-12-31T12:00:00Z")
        assert d == date(2026, 12, 31)

    def test_trade_uses_placed_at(self):
        trade = PaperTrade(
            market_id="m", description="Will it be 30°F or below on December 31?",
        )
        trade.placed_at = datetime(2026, 12, 30, 15, tzinfo=UTC)
        assert _extract_target_date_from_trade(trade) == date(2026, 12, 31)


# ---------------------------------------------------------------------------
# 6 / 15. No reanalysis settlement; loud METAR failure
# ---------------------------------------------------------------------------

class TestNoReanalysisSettlement:

    def _no_http(self, monkeypatch):
        from weather_edge.analysis import resolver

        class Boom:
            def __init__(self, *a, **k):
                raise AssertionError("resolver must not fall back to Open-Meteo")
        monkeypatch.setattr(resolver.httpx, "AsyncClient", Boom)

    def test_metar_exception_leaves_unresolved_and_warns(self, monkeypatch, caplog):
        from weather_edge.analysis import resolver
        from weather_edge.fetchers import metar

        async def raising(*a, **k):
            raise RuntimeError("IEM down")
        monkeypatch.setattr(metar, "fetch_station_tmax_both", raising)
        self._no_http(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=resolver.__name__):
            out = asyncio.run(resolver.check_nws_observations("nyc", date(2026, 3, 1)))
        assert out is None
        assert any("METAR fetch failed" in r.message for r in caplog.records)

    def test_missing_metar_leaves_unresolved(self, monkeypatch):
        from weather_edge.analysis import resolver
        from weather_edge.fetchers import metar

        async def empty(*a, **k):
            return None, None
        monkeypatch.setattr(metar, "fetch_station_tmax_both", empty)
        self._no_http(monkeypatch)
        assert asyncio.run(resolver.check_nws_observations("nyc", date(2026, 3, 1))) is None


# ---------------------------------------------------------------------------
# 7. Backtester uses resolver semantics
# ---------------------------------------------------------------------------

class TestBacktesterResolverSemantics:

    def test_rounded_value_crosses_bucket_edge(self):
        from weather_edge.analysis.backtester import _TEMP_BUCKETS_F, _find_bucket_resolved
        # 79.6°F displays as 80 -> 80-90 bucket, not 70-80.
        idx = _find_bucket_resolved(f_to_c(79.6), _TEMP_BUCKETS_F, "F")
        assert _TEMP_BUCKETS_F[idx] == (80, 90)
        idx = _find_bucket_resolved(f_to_c(79.4), _TEMP_BUCKETS_F, "F")
        assert _TEMP_BUCKETS_F[idx] == (70, 80)

    def test_celsius_and_open_tails(self):
        from weather_edge.analysis.backtester import _TEMP_BUCKETS_C, _find_bucket_resolved
        assert _TEMP_BUCKETS_C[_find_bucket_resolved(-0.6, _TEMP_BUCKETS_C, "C")] == (None, 0)
        assert _TEMP_BUCKETS_C[_find_bucket_resolved(-0.5, _TEMP_BUCKETS_C, "C")] == (0, 5)
        assert _TEMP_BUCKETS_C[_find_bucket_resolved(34.5, _TEMP_BUCKETS_C, "C")] == (35, None)
        # HKO floors: 34.9 stays in 30-35.
        assert _TEMP_BUCKETS_C[
            _find_bucket_resolved(34.9, _TEMP_BUCKETS_C, "C", is_hkg=True)
        ] == (30, 35)

    def test_probability_band_matches_rounding(self):
        from weather_edge.analysis.backtester import _display_band
        assert _display_band(70, 80) == (69.5, 79.5)
        assert _display_band(None, 32) == (None, 31.5)
        assert _display_band(30, 35, is_hkg=True) == (30, 35)


# ---------------------------------------------------------------------------
# 8. Calibrated blend is actually used for band edges
# ---------------------------------------------------------------------------

def _forecasts(values):
    now = datetime.now(UTC)
    return [
        ForecastResult(
            city_id="nyc", model_name=name, target_date=date(2026, 3, 27),
            fetched_at=now, temperature_2m_hourly=[], precipitation_hourly=[],
            snowfall_hourly=[], wind_speed_10m_hourly=[], temp_max_c=v,
            temp_min_c=None, precip_sum_mm=None, snow_sum_cm=None, wind_max_kmh=None,
        )
        for name, v in values.items()
    ]


@pytest.fixture
def no_bias(monkeypatch):
    monkeypatch.setattr(consensus_mod, "apply_bias_correction", lambda v, *a: v)


VALUES = {
    "ecmwf_ifs025": 20.0, "gfs_seamless": 21.5, "icon_seamless": 20.4,
    "gem_seamless": 22.8, "jma_seamless": 19.6,
}


class TestBlendUsed:

    def test_band_edge_uses_blend_not_pure_normal(self, no_bias, monkeypatch):
        c = compute_consensus(City.NYC, "2026-03-27", "temp_max_c", _forecasts(VALUES), [20.0])
        edge = f_to_c(70.5)  # a Fahrenheit band edge, not a pre-computed key
        assert f">={edge}" not in c.threshold_probs

        calls = []
        real = consensus_mod._blend_threshold_probs

        def spy(*a, **k):
            calls.append(a[-1])
            return real(*a, **k)
        monkeypatch.setattr(consensus_mod, "_blend_threshold_probs", spy)
        p = get_probability_for_threshold(c, edge, "gte")
        assert calls == [[edge]]

        names = list(c.model_values)
        expected = _blend_threshold_probs(
            [c.model_values[n] for n in names],
            [c.model_weights[n] for n in names],
            c.weighted_mean, c.std_dev, [edge],
        )[f">={edge}"]
        assert p == pytest.approx(expected)

    def test_precomputed_band_edges_are_hit(self, no_bias, monkeypatch):
        market = MarketInfo(
            market_id="f", city_id=City.NYC, threshold_unit="fahrenheit",
            threshold_dir="range", bucket_low_int=70, bucket_high_int=71,
        )
        band = _bucket_celsius_band(market)
        c = compute_consensus(City.NYC, "2026-03-27", "temp_max_c", _forecasts(VALUES),
                              sorted(band))
        monkeypatch.setattr(
            consensus_mod, "_blend_threshold_probs",
            lambda *a, **k: pytest.fail("pre-computed key should be used"),
        )
        prob = compute_model_prob_for_market(market, c)
        expected = c.threshold_probs[f">={band[0]}"] - c.threshold_probs[f">={band[1]}"]
        assert prob == pytest.approx(expected)

    def test_pattern_shift_moves_blend(self, no_bias):
        c = compute_consensus(City.NYC, "2026-03-27", "temp_max_c", _forecasts(VALUES), [21.0])
        before = get_probability_for_threshold(c, 21.0, "gte")
        c.weighted_mean -= 2.0  # pattern-bias shift applied by the scheduler
        after = get_probability_for_threshold(c, 21.0, "gte")
        assert after < before - 0.2


# ---------------------------------------------------------------------------
# 9. Agreement gate on raw spread
# ---------------------------------------------------------------------------

class TestAgreementGate:

    def test_gate_uses_raw_spread(self, no_bias):
        # Raw sample std ~1.8°C (< 2.0) but inflated std ~2.3°C (> 2.0).
        vals = {"a": 18.0, "b": 20.0, "c": 22.0, "d": 19.0, "e": 21.5}
        c = compute_consensus(City.NYC, "2026-03-27", "temp_max_c", _forecasts(vals))
        assert c.std_dev > MODEL_AGREEMENT_MAX_STD_C
        assert c.raw_std_dev < MODEL_AGREEMENT_MAX_STD_C
        assert _model_agreement_std(c) == c.raw_std_dev

    def test_fallback_uninflates(self):
        class Legacy:
            std_dev = 2.6
        assert _model_agreement_std(Legacy()) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 10. build_bias_table: no source rewrite, no double count
# ---------------------------------------------------------------------------

class TestBuildBiasTable:

    def test_does_not_rewrite_source(self, monkeypatch, capsys):
        bbt = _load_script("build_bias_table")
        target = ROOT / "src" / "weather_edge" / "analysis" / "bias_correction.py"
        before = target.read_text()

        async def fake_build(days, cities):
            return [bbt.BiasRow("nyc", "m", 20, 0.1, 0.0, "gated")]
        monkeypatch.setattr(bbt, "build", fake_build)
        assert bbt.main(["--city", "nyc"]) == 0
        assert target.read_text() == before
        assert "nyc" in capsys.readouterr().out
        assert not hasattr(bbt, "write_bias_correction")

    def test_single_gated_correction_per_model(self):
        bbt = _load_script("build_bias_table")
        obs = {f"2026-03-{i:02d}": 20.0 for i in range(1, 21)}
        fc = {d: 21.5 for d in obs}
        row = bbt.bias_row("hkg", "m", bbt.compute_errors(obs, fc))
        assert row.samples == 20
        assert row.bias_c == pytest.approx(1.5)
        # Zero-variance, clearly significant bias: the correction is -bias, once.
        assert row.correction_c == pytest.approx(-1.5)

    def test_writes_json_to_out(self, tmp_path, monkeypatch):
        bbt = _load_script("build_bias_table")

        async def fake_build(days, cities):
            return [bbt.BiasRow("nyc", "m", 20, 0.1, 0.0, "gated")]
        monkeypatch.setattr(bbt, "build", fake_build)
        out = tmp_path / "bias.json"
        assert bbt.main(["--out", str(out), "--city", "nyc"]) == 0
        assert json.loads(out.read_text())[0]["city"] == "nyc"

    def test_observations_use_shared_metar_path(self, monkeypatch, fake_client):
        bbt = _load_script("build_bias_table")
        seen = {}

        async def fake_range(icao, start, end, *, station_tz, **kw):
            seen["tz"] = station_tz
            return {}
        monkeypatch.setattr(bbt, "fetch_station_tmax_range", fake_range)
        monkeypatch.setattr(bbt.httpx, "AsyncClient", fake_client)
        asyncio.run(bbt.build(5, [City.TYO]))
        assert seen["tz"] == "Asia/Tokyo"


# ---------------------------------------------------------------------------
# 11. Only actually-resolved markets settle
# ---------------------------------------------------------------------------

class TestOnlyResolvedMarkets:

    def test_closed_unresolved_high_price_is_ignored(self, monkeypatch, fake_client):
        from weather_edge.analysis import resolver
        fake_client.payloads = [[{
            "markets": [
                {"conditionId": "closed_only", "closed": True,
                 "outcomePrices": "[\"0.97\", \"0.03\"]"},
                {"conditionId": "proposed", "closed": True,
                 "umaResolutionStatus": "proposed", "outcomePrices": ["0.99", "0.01"]},
                {"conditionId": "done", "closed": True,
                 "umaResolutionStatus": "resolved", "outcomePrices": ["0", "1"]},
            ],
        }]]
        monkeypatch.setattr(resolver.httpx, "AsyncClient", fake_client)
        out = asyncio.run(resolver.fetch_resolved_markets())
        assert out == {"done": False}


# ---------------------------------------------------------------------------
# 12. build_station_offsets: shared METAR path + round-half-up
# ---------------------------------------------------------------------------

class TestStationOffsets:

    def test_round_half_up_label(self):
        bso = _load_script("build_station_offsets")
        assert bso.display_label(72.5, False) == 73   # round() would give 72
        assert bso.display_label(28.9, True) == 28    # HKO floor

    def test_uses_shared_metar_local_day(self, monkeypatch):
        bso = _load_script("build_station_offsets")
        seen = {}

        async def fake_range(icao, start, end, *, station_tz, **kw):
            seen.update(icao=icao, tz=station_tz)
            return {"2026-03-01": 10.0}
        monkeypatch.setattr(bso, "fetch_station_tmax_range", fake_range)
        out = bso.fetch_station_daily_max("EGLC", date(2026, 3, 1), date(2026, 3, 2),
                                          "Europe/London")
        assert out == {"2026-03-01": 10.0}
        assert seen == {"icao": "EGLC", "tz": "Europe/London"}


# ---------------------------------------------------------------------------
# 13. Explicit zero edge thresholds are honoured
# ---------------------------------------------------------------------------

class TestExplicitZeroMinEdge:

    def _sig(self, min_edge):
        return calculate_edge(
            market_id="m", model_prob=0.54, market_prob=0.50, model_confidence=1.0,
            bankroll=2000.0, hours_to_resolution=2.0, spread=0.0,
            min_edge_yes=min_edge, min_edge_no=min_edge,
        )

    def test_zero_is_not_default(self):
        default = self._sig(None)
        zero = self._sig(0.0)
        assert default.confidence_tier == SignalTier.LOW   # 0.0275 < 0.6 * 0.05
        assert zero.confidence_tier == SignalTier.HIGH


# ---------------------------------------------------------------------------
# 14. METAR temperature-only T-group
# ---------------------------------------------------------------------------

class TestTemperatureOnlyTGroup:

    def test_t_group_without_dewpoint(self):
        assert _point_temp_c_from_metar("KSEA 011853Z 00000KT 10SM CLR 21/M "
                                        "A3001 RMK AO2 T0211") == [21.1]
        assert _point_temp_c_from_metar("X RMK AO2 T1005") == [-0.5]

    def test_full_t_group_still_parses(self):
        assert _point_temp_c_from_metar("X RMK AO2 T02110017") == [21.1]

    def test_temperature_only_group_raises_daily_max(self):
        rows = [{"valid": "2026-03-01 14:53", "tmpc": "21.00", "tmpf": "69.80",
                 "metar": "KSEA 012253Z RMK AO2 T0214"}]
        max_c, _f, _n = _daily_max_from_rows(rows)["2026-03-01"]
        assert max_c == pytest.approx(21.4)
