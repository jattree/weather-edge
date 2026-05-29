"""Regression tests for the bucket -> probability band conversion.

The pre-fix scheduler integrated Fahrenheit range buckets over a band that was
~0.8°F too wide on the top edge, because it added a flat 1.0 *Celsius* to a
°C-converted Fahrenheit bound (a unit mix). It also used a [X, X+1) convention
that was offset half a degree from the resolver's round-half-up rule.

_bucket_celsius_band now derives the band from native integer labels using
round-half-up semantics (label L covers [L-0.5, L+0.5)), so probability
integration matches resolution exactly.
"""
from __future__ import annotations

from weather_edge.fetchers.openmeteo import f_to_c
from weather_edge.fetchers.polymarket import MarketInfo
from weather_edge.scheduler import _bucket_celsius_band


def _mkt(**kw) -> MarketInfo:
    return MarketInfo(market_id="x", **kw)


class TestBucketCelsiusBand:

    def test_fahrenheit_range_width_is_exactly_two_degrees_f(self):
        """'50-51°F' covers displayed highs {50, 51} = continuous [49.5, 51.5)°F.
        The band width must equal 2°F worth of Celsius (1.111°C), NOT the buggy
        2.8°F (1.556°C) the +1.0°C unit-mix produced."""
        m = _mkt(threshold_dir="range", threshold_unit="fahrenheit",
                 bucket_low_int=50, bucket_high_int=51)
        lo_c, hi_c = _bucket_celsius_band(m)
        assert lo_c == f_to_c(49.5)
        assert hi_c == f_to_c(51.5)
        width = hi_c - lo_c
        assert abs(width - (2.0 * 5.0 / 9.0)) < 1e-9   # 2°F == 1.111°C
        assert abs(width - 1.5556) > 0.1               # NOT the old 2.8°F band

    def test_celsius_exact_band_is_one_degree_centered(self):
        """'8°C' covers displayed high 8 = [7.5, 8.5)°C, width 1.0°C."""
        m = _mkt(threshold_dir="range", threshold_unit="celsius",
                 bucket_low_int=8, bucket_high_int=8)
        lo_c, hi_c = _bucket_celsius_band(m)
        assert lo_c == 7.5
        assert hi_c == 8.5

    def test_lte_band_open_lower_edge(self):
        """'49°F or below' -> band upper edge at f_to_c(49.5), lower open."""
        m = _mkt(threshold_dir="lte", threshold_unit="fahrenheit",
                 bucket_low_int=None, bucket_high_int=49)
        lo_c, hi_c = _bucket_celsius_band(m)
        assert lo_c is None
        assert hi_c == f_to_c(49.5)

    def test_gte_band_open_upper_edge(self):
        """'60°F or above' -> band lower edge at f_to_c(59.5), upper open."""
        m = _mkt(threshold_dir="gte", threshold_unit="fahrenheit",
                 bucket_low_int=60, bucket_high_int=None)
        lo_c, hi_c = _bucket_celsius_band(m)
        assert lo_c == f_to_c(59.5)
        assert hi_c is None

    def test_no_native_labels_returns_none(self):
        """Snow / 'any' markets carry no temperature labels."""
        m = _mkt(threshold_dir="any", threshold_unit="cm")
        assert _bucket_celsius_band(m) is None
