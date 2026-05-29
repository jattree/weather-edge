"""Tests for METAR daily-max extraction (the #4 fix).

The displayed daily high can exceed the max of hourly spot readings when the
true peak falls between reports. We recover it from the precise hourly
temperature group (Tsnnnsnnn) and the 6-hour maximum group (1snnn) in remarks.

Real example (Denver KDEN, 2026-03-26, local time): hourly tmpf peaked at 70°F,
but the 18:53-local remark carries `10217` = 21.7°C = 71°F. Taking max-of-hourly
would resolve the daily high one whole degree too low.
"""
from __future__ import annotations

from weather_edge.fetchers.metar import _daily_max_from_rows, _temps_c_from_metar


class TestTempsFromMetar:

    def test_t_group_and_six_hour_max(self):
        metar = ("KDEN 262353Z 06020KT 6SM HZ FEW030 SCT090 BKN140 BKN220 "
                 "15/02 A3011 RMK AO2 PK WND 05028/2310 SLP144 T01500022 "
                 "10217 20150 53037")
        temps = _temps_c_from_metar(metar)
        assert 15.0 in temps   # T-group precise hourly temp
        assert 21.7 in temps   # 6-hr max group (the true peak)

    def test_negative_temperatures(self):
        metar = "KORD 010053Z 00000KT 10SM CLR M03/M08 A3012 RMK AO2 T10031078 11031"
        temps = _temps_c_from_metar(metar)
        assert -0.3 in temps   # T-group, sign bit 1 = negative
        assert -3.1 in temps   # 6-hr max group, negative

    def test_no_remarks_no_six_hour_group(self):
        metar = "EGLC 261220Z 24008KT 9999 FEW035 11/05 Q1018"
        # No T-group, no RMK section -> nothing extracted
        assert _temps_c_from_metar(metar) == []

    def test_body_numbers_not_mistaken_for_temp(self):
        # "10SM" visibility in the body must not parse as a 6-hr max group.
        metar = "KLGA 261551Z 28010KT 10SM FEW250 18/02 A3001"
        assert _temps_c_from_metar(metar) == []


class TestDailyMaxFromRows:

    def test_six_hour_group_lifts_daily_max(self):
        rows = [
            {"valid": "2026-03-26 12:53", "tmpf": "70.00", "tmpc": "21.10", "metar": ""},
            {"valid": "2026-03-26 13:53", "tmpf": "70.00", "tmpc": "21.10", "metar": ""},
            {"valid": "2026-03-26 17:53", "tmpf": "59.00", "tmpc": "15.00",
             "metar": "KDEN 262353Z RMK AO2 SLP144 T01500022 10217 20150 53037"},
        ]
        out = _daily_max_from_rows(rows)
        max_c, max_f, readings = out["2026-03-26"]
        assert abs(max_c - 21.7) < 1e-9   # from the 6-hr max group, not the 21.1 hourly
        assert round(max_f) == 71         # 21.7°C = 71.06°F -> displays as 71
        assert readings == 3

    def test_after_midnight_six_hour_max_not_credited_to_new_day(self):
        """A 00:53-local report's 6-hr max covers the PREVIOUS evening and must
        not inflate the new day's high. Only the T-group (timestamped to this
        hour) counts for an after-midnight report."""
        rows = [
            {"valid": "2026-03-27 00:53", "tmpf": "45.00", "tmpc": "7.20",
             "metar": "KDEN 270653Z RMK AO2 SLP144 T00720011 10217 20060 53037"},
        ]
        out = _daily_max_from_rows(rows)
        max_c, _max_f, _readings = out["2026-03-27"]
        # 21.7 (the prior-evening 6-hr max) must NOT leak in; max is the 7.2 T-group.
        assert max_c < 10.0

    def test_afternoon_six_hour_max_is_credited(self):
        """An afternoon report's 6-hr max DOES count (it reflects today's peak)."""
        rows = [
            {"valid": "2026-03-26 16:53", "tmpf": "59.00", "tmpc": "15.00",
             "metar": "KDEN 262353Z RMK AO2 SLP144 T01500022 10217 20150 53037"},
        ]
        out = _daily_max_from_rows(rows)
        max_c, _max_f, _readings = out["2026-03-26"]
        assert abs(max_c - 21.7) < 1e-9

    def test_grouping_by_provided_local_date_string(self):
        rows = [
            {"valid": "2026-03-26 23:00", "tmpf": "40.00", "tmpc": "4.44", "metar": ""},
            {"valid": "2026-03-27 02:00", "tmpf": "38.00", "tmpc": "3.33", "metar": ""},
        ]
        out = _daily_max_from_rows(rows)
        assert set(out.keys()) == {"2026-03-26", "2026-03-27"}
