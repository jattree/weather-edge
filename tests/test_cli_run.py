"""`weather-edge run` / `watch` regression tests (no network)."""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, date
from pathlib import Path

from click.testing import CliRunner

from tests.test_ai_fixes import (  # noqa: F401  (fixtures)
    isolated_ai_state,
    offline_cycle,
    patch_reviews,
    reasoning,
)
from weather_edge import cli as cli_mod
from weather_edge import scheduler

ROOT = Path(__file__).resolve().parents[1]


def test_module_run_help():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    out = subprocess.run(
        [sys.executable, "-m", "weather_edge", "run", "--help"],
        capture_output=True, text=True, env=env, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert "--days" in out.stdout and "default: 4" in out.stdout


def test_run_end_to_end_paper_trades_approved_signal(monkeypatch, offline_cycle):  # noqa: F811
    patch_reviews(monkeypatch, claude=lambda s: reasoning(s, should_trade=s.city_id == "nyc"))
    placed = []
    from weather_edge.trading.paper import PaperTrader

    real = PaperTrader.place_trade
    monkeypatch.setattr(PaperTrader, "place_trade",
                        lambda self, s: placed.append(s.market_id) or real(self, s))

    result = CliRunner().invoke(cli_mod.cli, ["run"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert "Trading Signals" in result.output
    assert placed == ["nyc-d2"]


def test_run_explains_when_horizon_blocks_everything(monkeypatch):
    called = []

    async def fake_cycle(*a, **k):
        called.append(1)
        return [], {}, {}

    monkeypatch.setattr(scheduler, "run_cycle", fake_cycle)
    result = CliRunner().invoke(cli_mod.cli, ["run", "--days", "1"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "horizon" in result.output and "nothing to scan" in result.output
    assert called == []


def test_watch_passes_days_and_real_interval(monkeypatch):
    seen = {}

    async def fake_loop(trader, days=None, max_cycles=None):
        seen["days"] = days

    monkeypatch.setattr(scheduler, "run_loop", fake_loop)
    monkeypatch.setattr(scheduler.settings, "fetch_interval_minutes", 17)
    result = CliRunner().invoke(cli_mod.cli, ["watch", "--days", "3"], catch_exceptions=False)
    assert result.exit_code == 0
    assert seen["days"] == 3
    assert "every 17 minutes" in result.output


def test_default_days_always_scans_something():
    from datetime import datetime, timedelta

    for hour in range(24):
        now = datetime.combine(date.today(), datetime.min.time(), tzinfo=UTC)
        now += timedelta(hours=hour)
        kept, _, _ = scheduler.filter_dates_by_horizon(
            scheduler.default_target_dates(now.date(), cli_mod.DEFAULT_DAYS), now, 36,
        )
        assert kept
