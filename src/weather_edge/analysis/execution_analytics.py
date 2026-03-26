"""Execution analytics computed from paper trading data."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Edge distribution buckets (percentage)
_EDGE_BUCKETS = [
    (0, 5, "0-5%"),
    (5, 10, "5-10%"),
    (10, 20, "10-20%"),
    (20, 50, "20-50%"),
    (50, 100, "50%+"),
]


def compute_execution_analytics(
    trades: list[Any],
    signals: list[dict],
    cycle_count: int,
    bankroll: float,
    capital_at_risk: float,
) -> dict:
    """Compute execution analytics from paper trading data.

    Args:
        trades: List of PaperTrade objects from the paper trader.
        signals: List of signal dicts from current state.
        cycle_count: Current cycle number.
        bankroll: Current bankroll.
        capital_at_risk: Current capital at risk.

    Returns:
        Dict with metrics, edge distribution, and per-cycle stats.
    """
    if not trades:
        return {
            "avg_position_size": 0.0,
            "avg_position_core": 0.0,
            "avg_position_tail": 0.0,
            "capital_utilization_pct": 0.0,
            "trades_per_cycle": 0.0,
            "total_trades": 0,
            "edge_distribution": [],
            "size_distribution": [],
            "win_rate_by_edge": [],
        }

    total_trades = len(trades)
    sizes = [t.size_usd for t in trades]
    avg_size = sum(sizes) / len(sizes) if sizes else 0.0

    # Separate core vs tail by size (core = larger positions, tail = smaller)
    # Use median as dividing line
    sorted_sizes = sorted(sizes)
    median_idx = len(sorted_sizes) // 2
    core_sizes = sorted_sizes[median_idx:]
    tail_sizes = sorted_sizes[:median_idx]

    avg_core = sum(core_sizes) / len(core_sizes) if core_sizes else 0.0
    avg_tail = sum(tail_sizes) / len(tail_sizes) if tail_sizes else 0.0

    # Capital utilization
    cap_util = (capital_at_risk / bankroll * 100) if bankroll > 0 else 0.0

    # Trades per cycle
    trades_per_cycle = total_trades / cycle_count if cycle_count > 0 else 0.0

    # Edge distribution from current signals
    edge_dist: list[dict] = []
    for lo, hi, label in _EDGE_BUCKETS:
        count = sum(
            1 for s in signals
            if s.get("edge_pct") is not None
            and lo <= abs(s["edge_pct"] * 100) < hi
        )
        edge_dist.append({"bucket": label, "count": count})

    # Size distribution (histogram of position sizes)
    size_buckets = [
        (0, 5, "$0-5"),
        (5, 10, "$5-10"),
        (10, 25, "$10-25"),
        (25, 50, "$25-50"),
        (50, 1000, "$50+"),
    ]
    size_dist: list[dict] = []
    for lo, hi, label in size_buckets:
        count = sum(1 for s in sizes if lo <= s < hi)
        size_dist.append({"bucket": label, "count": count})

    # Win rate by edge bucket (from closed trades with known outcomes)
    win_rate_by_edge: list[dict] = []
    closed = [t for t in trades if hasattr(t, 'status') and t.status.value in ('won', 'lost')]

    for lo, hi, label in _EDGE_BUCKETS:
        bucket_trades = [
            t for t in closed
            if hasattr(t, 'edge_pct') and t.edge_pct is not None
            and lo <= abs(t.edge_pct * 100) < hi
        ]
        bucket_wins = sum(1 for t in bucket_trades if t.status.value == 'won')
        bucket_total = len(bucket_trades)
        wr = round(bucket_wins / bucket_total * 100, 1) if bucket_total > 0 else 0.0
        win_rate_by_edge.append({
            "bucket": label,
            "trades": bucket_total,
            "wins": bucket_wins,
            "win_rate": wr,
        })

    return {
        "avg_position_size": round(avg_size, 2),
        "avg_position_core": round(avg_core, 2),
        "avg_position_tail": round(avg_tail, 2),
        "capital_utilization_pct": round(cap_util, 1),
        "trades_per_cycle": round(trades_per_cycle, 2),
        "total_trades": total_trades,
        "edge_distribution": edge_dist,
        "size_distribution": size_dist,
        "win_rate_by_edge": win_rate_by_edge,
    }
