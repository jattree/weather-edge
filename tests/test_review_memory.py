"""AI review memory must survive a live_state outage without losing vetoes."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from weather_edge import live_state
from weather_edge.analysis.claude_reasoning import _VETO_STATE_KEY, AIReviewMemory


class FlakyStore:
    """Stands in for get_value_persisted / set_json with a switchable outage."""

    def __init__(self, snapshot: dict | None = None):
        self.raw = json.dumps(snapshot) if snapshot is not None else None
        self.down = False
        self.writes = 0

    def get_value_persisted(self, key):
        assert key == _VETO_STATE_KEY
        if self.down:
            raise live_state.LiveStateUnavailableError("redis down")
        return self.raw

    def set_json(self, key, data, ttl=None):
        if self.down:
            raise ConnectionError("redis down")
        self.raw = json.dumps(data)
        self.writes += 1


def _signal(market_id: str, target_date: str = "2026-09-26"):
    return SimpleNamespace(
        market_id=market_id, target_date=target_date, recommended_side="YES",
    )


@pytest.fixture
def store(monkeypatch):
    s = FlakyStore({"m-old|2026-09-26": {"reason": "bust risk", "source": "claude",
                                         "recorded_at": 1.0}})
    monkeypatch.setattr(live_state, "get_value_persisted", s.get_value_persisted)
    monkeypatch.setattr(live_state, "set_json", s.set_json)
    return s


def test_outage_on_first_load_is_retried(store):
    mem = AIReviewMemory()
    store.down = True
    assert mem.veto_for(_signal("m-old")) is None  # unknown while down
    store.down = False
    assert mem.veto_for(_signal("m-old")) is not None  # recovered, veto restored


def test_save_during_outage_does_not_overwrite_snapshot(store):
    mem = AIReviewMemory()
    store.down = True
    mem.record_veto(_signal("m-new"), "outlier", "gemini")
    assert store.writes == 0
    assert "m-old|2026-09-26" in json.loads(store.raw)
    store.down = False
    mem.record_veto(_signal("m-new2"), "outlier", "claude")
    saved = json.loads(store.raw)
    assert {"m-old|2026-09-26", "m-new|2026-09-26", "m-new2|2026-09-26"} <= set(saved)


def test_first_save_merges_unseen_snapshot(store):
    mem = AIReviewMemory()
    mem._loaded = True  # e.g. loaded as empty before another process wrote
    mem.record_veto(_signal("m-new"), "outlier", "claude")
    assert "m-old|2026-09-26" in json.loads(store.raw)


def test_evicted_dates_are_not_resurrected(store):
    from datetime import date
    mem = AIReviewMemory()
    mem.evict_before(date(2026, 9, 27))
    mem.record_veto(_signal("m-new", "2026-09-28"), "outlier", "claude")
    assert set(json.loads(store.raw)) == {"m-new|2026-09-28"}



def test_persisted_reader_never_serves_the_fallback(monkeypatch):
    # Cold start with Redis never connected: the fallback may hold a value,
    # but a persisted read must refuse rather than return it.
    monkeypatch.setattr(live_state, "_get_redis", lambda: None)
    monkeypatch.setattr(live_state, "_redis_ever_connected", False)
    monkeypatch.setattr(live_state, "_fallback_cache", {_VETO_STATE_KEY: "{}"})
    with pytest.raises(live_state.LiveStateUnavailableError):
        live_state.get_value_persisted(_VETO_STATE_KEY)
    mem = AIReviewMemory()
    assert mem.veto_for(_signal("m-old")) is None
    assert mem._loaded is False  # retried on the next call
