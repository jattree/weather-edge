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
# Wunderground rounds to whole degrees before display. Polymarket resolves
# against that displayed value. So 28.5C rounds to 28 (Python banker's
# rounding), 28.6C rounds to 29, etc.
# ---------------------------------------------------------------------------

class TestActualFallsInBucketRounding:

    def test_celsius_28_5_rounds_to_28(self):
        """28.5C -> round(28.5) = 28 (banker's rounding). Falls in [28, 29)."""
        bucket = BucketInfo(28.0, 29.0, "celsius", exclusive_upper=True)
        assert actual_falls_in_bucket(28.5, bucket) is True

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
        """METAR 28.06C = 82.5F -> round = 82 (banker's). Falls in [82, 84)."""
        bucket = BucketInfo(82.0, 84.0, "fahrenheit", exclusive_upper=True)
        assert actual_falls_in_bucket(28.06, bucket) is True

    def test_below_bucket_with_rounding(self):
        """'70F or below': 21.1C = 69.98F -> round = 70. YES (70 <= 70)."""
        bucket = BucketInfo(None, 70.0, "fahrenheit")
        assert actual_falls_in_bucket(21.1, bucket) is True

    def test_above_bucket_with_rounding(self):
        """'30C or above': 29.5C -> round = 30 (banker's). YES (30 >= 30)."""
        bucket = BucketInfo(30.0, None, "celsius")
        # round(29.5) = 30 with banker's rounding (rounds to even)
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
