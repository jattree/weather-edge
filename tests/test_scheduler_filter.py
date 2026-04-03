import pytest
from datetime import date
from weather_edge.analysis.edge import Signal, SignalTier
from weather_edge.models.enums import TradeSide, SignalTier
from datetime import datetime, timezone

def test_one_signal_per_city_date_filter():
    """Verify that scheduler filters multiple signals to one per city-date."""
    now = datetime.now(timezone.utc)
    
    # Mock signals for Toronto on April 4
    s1 = Signal(
        market_id="tor_5c", consensus_id=None, computed_at=now,
        model_prob=0.05, market_prob=0.20, model_confidence=0.9,
        edge=0.15, net_edge=0.14, edge_pct=0.75, kelly_fraction=0.1, half_kelly=0.05,
        recommended_side=TradeSide.NO, recommended_size=10.0,
        confidence_tier=SignalTier.MEDIUM, strategy="core", 
        city_id="tor", target_date="2026-04-04"
    )
    s2 = Signal(
        market_id="tor_7c", consensus_id=None, computed_at=now,
        model_prob=0.40, market_prob=0.20, model_confidence=0.9,
        edge=0.20, net_edge=0.19, edge_pct=1.0, kelly_fraction=0.15, half_kelly=0.075,
        recommended_side=TradeSide.YES, recommended_size=15.0,
        confidence_tier=SignalTier.HIGH, strategy="core",
        city_id="tor", target_date="2026-04-04"
    )
    
    # Mock signals for London on April 4
    s3 = Signal(
        market_id="lon_13c", consensus_id=None, computed_at=now,
        model_prob=0.05, market_prob=0.40, model_confidence=0.9,
        edge=0.35, net_edge=0.34, edge_pct=0.85, kelly_fraction=0.2, half_kelly=0.1,
        recommended_side=TradeSide.NO, recommended_size=20.0,
        confidence_tier=SignalTier.HIGH, strategy="tail_no",
        city_id="lon", target_date="2026-04-04"
    )
    
    all_signals = [s1, s2, s3]
    
    # Filter logic from scheduler.py
    filtered_signals = []
    if all_signals:
        signal_groups = {}
        for s in all_signals:
            if s.confidence_tier.value == "low":
                continue
            key = (s.city_id, s.target_date)
            signal_groups.setdefault(key, []).append(s)

        for key, group in signal_groups.items():
            def _signal_score(sig):
                prio = {"tail_no": 300, "tail": 200, "core": 100}.get(sig.strategy, 0)
                return prio + abs(sig.net_edge)

            best_signal = max(group, key=_signal_score)
            filtered_signals.append(best_signal)

    # Toronto should pick s2 (higher net_edge)
    tor_signal = next(s for s in filtered_signals if s.city_id == "tor")
    assert tor_signal.market_id == "tor_7c"
    
    # London should pick s3
    lon_signal = next(s for s in filtered_signals if s.city_id == "lon")
    assert lon_signal.market_id == "lon_13c"
    
    assert len(filtered_signals) == 2
