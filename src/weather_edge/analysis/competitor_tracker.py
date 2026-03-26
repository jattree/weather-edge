"""Track competitor performance from public Polymarket profiles.

Snapshots public stats (positions value, predictions count) each cycle
so we can compare our paper P&L growth rate against ColdMath's.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

COMPETITORS = {
    "coldmath": "https://polymarket.com/@coldmath",
}


@dataclass
class CompetitorSnapshot:
    username: str
    timestamp: datetime
    positions_value: float | None = None
    predictions: int | None = None
    biggest_win: float | None = None


class CompetitorTracker:
    def __init__(self):
        self.history: dict[str, list[CompetitorSnapshot]] = {}

    async def fetch_profile(self, username: str) -> CompetitorSnapshot | None:
        """Fetch public stats from Polymarket profile page."""
        url = f"https://polymarket.com/@{username}"
        now = datetime.now(timezone.utc)

        try:
            async with httpx.AsyncClient() as client:
                # Use the Gamma API to get user stats
                resp = await client.get(
                    f"https://gamma-api.polymarket.com/users/by-username/{username}",
                    timeout=10.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return CompetitorSnapshot(
                        username=username,
                        timestamp=now,
                        positions_value=data.get("portfolioValue"),
                        predictions=data.get("totalTrades"),
                    )

                # Fallback: try profile page scraping for basic info
                resp = await client.get(
                    f"https://gamma-api.polymarket.com/profiles/{username}",
                    timeout=10.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return CompetitorSnapshot(
                        username=username,
                        timestamp=now,
                        positions_value=data.get("portfolioValue"),
                        predictions=data.get("numTrades"),
                    )

        except Exception as e:
            logger.debug("Failed to fetch %s profile: %s", username, e)

        return CompetitorSnapshot(username=username, timestamp=now)

    async def update_all(self) -> dict[str, CompetitorSnapshot]:
        """Fetch latest stats for all tracked competitors."""
        results = {}
        for username in COMPETITORS:
            snap = await self.fetch_profile(username)
            if snap:
                self.history.setdefault(username, []).append(snap)
                # Keep last 100 snapshots
                if len(self.history[username]) > 100:
                    self.history[username] = self.history[username][-100:]
                results[username] = snap
                if snap.positions_value:
                    logger.info(
                        "COMPETITOR %s: $%.0f positions, %s predictions",
                        username, snap.positions_value, snap.predictions or "?",
                    )
        return results

    def get_growth_rate(self, username: str) -> float | None:
        """Calculate positions growth rate per day for a competitor."""
        snaps = self.history.get(username, [])
        if len(snaps) < 2:
            return None

        first = snaps[0]
        last = snaps[-1]
        if not first.positions_value or not last.positions_value:
            return None

        days = max(1, (last.timestamp - first.timestamp).total_seconds() / 86400)
        return (last.positions_value - first.positions_value) / days

    def comparison_summary(self, our_pnl: float, our_trades: int) -> dict:
        """Compare our performance against tracked competitors."""
        result = {"our_pnl": our_pnl, "our_trades": our_trades, "competitors": {}}

        for username, snaps in self.history.items():
            if not snaps:
                continue
            latest = snaps[-1]
            growth = self.get_growth_rate(username)
            result["competitors"][username] = {
                "positions_value": latest.positions_value,
                "predictions": latest.predictions,
                "growth_per_day": round(growth, 2) if growth else None,
                "last_checked": latest.timestamp.isoformat(),
            }

        return result
