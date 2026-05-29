"""Tests for resolver bucket parsing and rounding logic.

Polymarket resolves weather markets against Wunderground's displayed value,
which is rounded to whole degrees. Our METAR data gives raw readings.
These tests verify that the rounding step correctly matches Wunderground.
"""
from __future__ import annotations

from weather_edge.analysis.resolver import (
    BucketInfo,
    actual_falls_in_bucket,
    parse_bucket_from_description,
)


# ---------------------------------------------------------------------------
# parse_bucket_from_description
# ---------------------------------------------------------------------------

class TestParseBucket:

    def test_exact_celsius(self):
        """'be 28C on March 27' -> [28, 29)"""
        b = parse_bucket_from_description("Will the high temperature be 28°C on March 27")
        assert b is not None
        assert b.low == 28.0
        assert b.high == 29.0
        assert b.unit == "celsius"
        assert b.exclusive_upper is True

    def test_range_fahrenheit(self):
        """'76-77F' -> [76, 78) per Polymarket rules"""
        b = parse_bucket_from_description("Will it be 76-77°F on March 27")
        assert b is not None
        assert b.low == 76.0
        assert b.high == 78.0
        assert b.unit == "fahrenheit"
        assert b.exclusive_upper is True

    def test_below_fahrenheit(self):
        """'70F or below' -> (None, 70]"""
        b = parse_bucket_from_description("Will it be 70°F or below")
        assert b is not None
        assert b.low is None
        assert b.high == 70.0
        assert b.unit == "fahrenheit"

    def test_above_celsius(self):
        """'30C or above' -> [30, None)"""
        b = parse_bucket_from_description("Will it be 30°C or above")
        assert b is not None
        assert b.low == 30.0
        assert b.high is None
        assert b.unit == "celsius"

    def test_no_match(self):
        """Unrecognized description returns None."""
        assert parse_bucket_from_description("some random text") is None


# ---------------------------------------------------------------------------
# actual_falls_in_bucket: rounding logic
# Wunderground rounds to whole degrees (half-up) before display. Polymarket
# resolves against that displayed value. So 28.5C displays as 29, 28.6C as 29,
# 28.4C as 28. We use round-half-up, NOT Python's banker's rounding (which would
# wrongly give round(28.5) == 28).
# ---------------------------------------------------------------------------

class TestActualFallsInBucketRounding:

    def test_celsius_28_5_rounds_half_up_to_29(self):
        """28.5C -> round-half-up = 29 (Wunderground display). Falls in [29, 30),
        NOT [28, 29). Python's banker's round(28.5) == 28 would put it in the
        wrong bucket, this is exactly the bug round-half-up fixes."""
        bucket_28 = BucketInfo(28.0, 29.0, "celsius", exclusive_upper=True)
        bucket_29 = BucketInfo(29.0, 30.0, "celsius", exclusive_upper=True)
        assert actual_falls_in_bucket(28.5, bucket_28) is False
        assert actual_falls_in_bucket(28.5, bucket_29) is True

    def test_celsius_28_6_rounds_to_29(self):
        """28.6C -> round(28.6) = 29. Falls in [29, 30), NOT [28, 29)."""
        bucket_28 = BucketInfo(28.0, 29.0, "celsius", exclusive_upper=True)
        bucket_29 = BucketInfo(29.0, 30.0, "celsius", exclusive_upper=True)
        assert actual_falls_in_bucket(28.6, bucket_28) is False
        assert actual_falls_in_bucket(28.6, bucket_29) is True

    def test_celsius_exact_integer(self):
        """Exact 28.0C stays 28. Falls in [28, 29)."""
        bucket = BucketInfo(28.0, 29.0, "celsius", exclusive_upper=True)
        assert actual_falls_in_bucket(28.0, bucket) is True

    def test_celsius_28_4_rounds_to_28(self):
        """28.4C -> 28. Falls in [28, 29)."""
        bucket = BucketInfo(28.0, 29.0, "celsius", exclusive_upper=True)
        assert actual_falls_in_bucket(28.4, bucket) is True

    def test_fahrenheit_rounding(self):
        """METAR 28.5C = 83.3F -> round = 83. Falls in [82, 84)."""
        bucket = BucketInfo(82.0, 84.0, "fahrenheit", exclusive_upper=True)
        assert actual_falls_in_bucket(28.5, bucket) is True

    def test_fahrenheit_rounding_boundary(self):
        """METAR 28.06C = 82.5F -> round-half-up = 83. Falls in [82, 84)."""
        bucket = BucketInfo(82.0, 84.0, "fahrenheit", exclusive_upper=True)
        assert actual_falls_in_bucket(28.06, bucket) is True

    def test_below_bucket_with_rounding(self):
        """'70F or below': 21.1C = 69.98F -> round = 70. YES (70 <= 70)."""
        bucket = BucketInfo(None, 70.0, "fahrenheit")
        assert actual_falls_in_bucket(21.1, bucket) is True

    def test_above_bucket_with_rounding(self):
        """'30C or above': 29.5C -> round-half-up = 30. YES (30 >= 30)."""
        bucket = BucketInfo(30.0, None, "celsius")
        assert actual_falls_in_bucket(29.5, bucket) is True

    def test_above_bucket_below_threshold(self):
        """'30C or above': 29.4C -> round = 29. NO (29 < 30)."""
        bucket = BucketInfo(30.0, None, "celsius")
        assert actual_falls_in_bucket(29.4, bucket) is False


# ---------------------------------------------------------------------------
# Without rounding, these would give wrong answers
# ---------------------------------------------------------------------------

class TestRoundingPreventsWrongResolution:

    def test_without_rounding_28_6_would_be_in_wrong_bucket(self):
        """28.6C: without rounding it falls in [28, 29). With rounding (29),
        it correctly falls in [29, 30). This is the bug the rounding fixes."""
        bucket_29 = BucketInfo(29.0, 30.0, "celsius", exclusive_upper=True)
        # With rounding: round(28.6) = 29, so 29 is in [29, 30) -> True
        assert actual_falls_in_bucket(28.6, bucket_29) is True

    def test_without_rounding_fahrenheit_boundary(self):
        """METAR 20.0C = 68.0F exactly. Should be in [68, 70). Rounding
        doesn't change integer values."""
        bucket = BucketInfo(68.0, 70.0, "fahrenheit", exclusive_upper=True)
        assert actual_falls_in_bucket(20.0, bucket) is True


# ---------------------------------------------------------------------------
# Subzero buckets (regression: unsigned (\d+) regex dropped the minus sign,
# resolving "-2°C or below" as "2°C or below", a guaranteed mis-resolution
# on every freezing-weather market).
# ---------------------------------------------------------------------------

class TestSubzeroBuckets:

    def test_parse_below_negative_celsius(self):
        b = parse_bucket_from_description("Will the high be -2°C or below on March 27")
        assert b is not None
        assert b.high == -2.0
        assert b.low is None
        assert b.unit == "celsius"

    def test_parse_exact_negative_celsius(self):
        b = parse_bucket_from_description("Will the high temperature be -5°C on March 27")
        assert b is not None
        assert b.low == -5.0
        assert b.high == -4.0
        assert b.exclusive_upper is True

    def test_parse_negative_fahrenheit_range(self):
        b = parse_bucket_from_description("Will it be -5--3°F on January 10")
        assert b is not None
        assert b.low == -5.0
        assert b.high == -2.0  # high label -3, [low, high+1)

    def test_resolve_negative_below_bucket(self):
        """-3.2°C actual against '-2°C or below': round-half-up(-3.2) = -3,
        which is <= -2 -> YES. The pre-fix parser read this as '2 or below'
        and would have resolved -3°C (well below 2) as a spurious YES too,
        but a +1°C actual (round 1) would wrongly resolve YES under the buggy
        '2 or below' reading while being NO under the correct '-2 or below'."""
        bucket = parse_bucket_from_description("be -2°C or below")
        assert bucket is not None
        assert actual_falls_in_bucket(-3.2, bucket) is True
        assert actual_falls_in_bucket(1.0, bucket) is False  # would be YES under the bug
