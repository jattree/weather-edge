"""Regression tests for paper-trading P&L, capital and persistence fixes."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from weather_edge.analysis import risk_controls
from weather_edge.analysis.edge import Signal
from weather_edge.models.enums import SignalTier, TradeSide, TradeStatus
from weather_edge.persistence import PersistentPaperTrader, PersistentStore
from weather_edge.trading.fees import calculate_taker_fee
from weather_edge.trading.market_maker import SpreadOrder
from weather_edge.trading.paper import PaperTrade, PaperTrader


@pytest.fixture(autouse=True)
def _fresh_risk_state(monkeypatch):
    monkeypatch.setattr(risk_controls, "_circuit_breaker", risk_controls.CircuitBreakerState())
    monkeypatch.setattr(risk_controls, "_active_profile_name", "balanced")


def make_signal(side="YES", market_prob=0.40, size=20.0, **kw) -> Signal:
    base = dict(
        market_id="m1", consensus_id=None, computed_at=datetime.now(timezone.utc),
        model_prob=0.6, model_confidence=0.9, market_prob=market_prob,
        edge=0.2, net_edge=0.2, edge_pct=0.5, kelly_fraction=0.1, half_kelly=0.05,
        recommended_side=TradeSide(side), recommended_size=size,
        confidence_tier=SignalTier.HIGH, city_id="nyc", description="NYC 70-71F",
        hours_to_resolution=10,
    )
    base.update(kw)
    return Signal(**base)


def make_hedge(side="NO", price=0.55, cost=20.0) -> SpreadOrder:
    return SpreadOrder(
        market_id="m1", city_id="nyc", token_id="", side=side, limit_price=price,
        shares=cost / price, cost=cost, guaranteed_profit=1.0, paired_with="m1",
        description=f"HEDGE {side}",
    )


# ---------------------------------------------------------------------------
# 5. fee_usd survives a restart
# ---------------------------------------------------------------------------

class TestFeePersistence:
    def test_fee_round_trip(self, tmp_path):
        db = tmp_path / "p.db"
        pt = PersistentPaperTrader(bankroll=1000.0, db_path=db)
        trade = pt.place_trade(make_signal())
        assert trade is not None and trade.fee_usd > 0
        fee = trade.fee_usd
        pt.store.close()

        resumed = PersistentPaperTrader(bankroll=1000.0, db_path=db)
        assert resumed.trades[0].fee_usd == pytest.approx(fee)
        # Loss after restart must still include the entry fee
        resumed.resolve_trade(resumed.trades[0], outcome_yes=False)
        assert resumed.trades[0].pnl == pytest.approx(-(trade.size_usd + fee))
        resumed.store.close()

    def test_migration_adds_fee_column(self, tmp_path):
        db = tmp_path / "old.db"
        conn = sqlite3.connect(db)
        conn.execute("""CREATE TABLE paper_trades (
            trade_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER NOT NULL,
            market_id TEXT NOT NULL, city_id TEXT NOT NULL, side TEXT NOT NULL,
            size_usd REAL NOT NULL, entry_price REAL NOT NULL, placed_at TEXT NOT NULL,
            description TEXT, strategy TEXT DEFAULT 'core', exit_price REAL,
            resolved_at TEXT, pnl REAL, status TEXT DEFAULT 'open')""")
        conn.execute(
            "INSERT INTO paper_trades (session_id, market_id, city_id, side, size_usd, "
            "entry_price, placed_at) VALUES (1, 'm', 'nyc', 'YES', 10, 0.4, ?)",
            (datetime.now(timezone.utc).isoformat(),),
        )
        conn.commit()
        conn.close()

        store = PersistentStore(db)
        cols = {r[1] for r in store.conn.execute("PRAGMA table_info(paper_trades)")}
        assert "fee_usd" in cols
        assert store.load_trades(1)[0].fee_usd == 0.0
        store.close()


# ---------------------------------------------------------------------------
# 9. Spread legs: gated, fee'd, persisted, resolved honestly
# ---------------------------------------------------------------------------

class TestSpreadTrades:
    def test_spread_leg_uses_yes_convention_and_fee(self):
        pt = PaperTrader(bankroll=1000.0)
        leg = pt.place_spread_trade(make_signal(), make_hedge("NO", 0.55, 20.0))
        assert leg.side == "NO"
        assert leg.entry_price == pytest.approx(0.45)  # YES-equivalent of NO@0.55
        assert leg.fee_usd == pytest.approx(round(calculate_taker_fee(0.55, 20.0), 4))
        assert leg.strategy == "spread"

    def test_spread_leg_goes_through_risk_gates(self):
        pt = PaperTrader(bankroll=1000.0)
        risk_controls._circuit_breaker.is_killed = True
        assert pt.place_spread_trade(make_signal(), make_hedge()) is None

    def test_merge_does_not_overwrite_directional_pnl(self):
        pt = PaperTrader(bankroll=1000.0)
        d = pt.place_trade(make_signal("YES", 0.40, 20.0))
        s = pt.place_spread_trade(make_signal(), make_hedge("NO", 0.55, 11.0))
        assert d and s

        pt.resolve_trade(s, outcome_yes=True)  # NO leg loses
        assert s.pnl == pytest.approx(-(s.size_usd + s.fee_usd))
        assert d.pnl is None and d.status == TradeStatus.OPEN  # untouched

        pt.resolve_trade(d, outcome_yes=True)
        d_shares = d.size_usd / d.entry_price
        assert d.pnl == pytest.approx(d_shares - d.size_usd - d.fee_usd)
        # Combined = matched pairs pay $1 + unmatched YES shares win, minus all
        # cost and fees (no fictitious "guaranteed" split)
        total_cost = d.size_usd + s.size_usd + d.fee_usd + s.fee_usd
        assert d.pnl + s.pnl == pytest.approx(d_shares - total_cost)

    def test_spread_leg_is_persisted(self, tmp_path):
        pt = PersistentPaperTrader(bankroll=1000.0, db_path=tmp_path / "s.db")
        pt.place_trade(make_signal())
        leg = pt.place_spread_trade(make_signal(), make_hedge())
        assert leg is not None
        rows = pt.store.load_trades(pt.session_id)
        assert any(r.description.startswith("[SPREAD]") for r in rows)
        pt.store.close()


class TestNoRowCollision:
    def test_unmapped_trade_never_overwrites_other_row(self, tmp_path):
        db = tmp_path / "c.db"
        pt = PersistentPaperTrader(bankroll=1000.0, db_path=db)
        # A row from ANOTHER session occupies db trade_id 1
        other = PaperTrade(market_id="other", city_id="lon", side="YES",
                           size_usd=5, entry_price=0.5)
        other_id = pt.store.save_trade(pt.session_id + 99, other)

        # In-memory trade whose trade_id collides with that row but has no mapping
        stray = PaperTrade(trade_id=other_id, market_id="m1", city_id="nyc",
                           side="YES", size_usd=10, entry_price=0.4)
        pt.trades.append(stray)
        pt.resolve_trade(stray, outcome_yes=True)

        row = pt.store.conn.execute(
            "SELECT market_id, pnl, status FROM paper_trades WHERE trade_id = ?", (other_id,),
        ).fetchone()
        assert row["market_id"] == "other" and row["pnl"] is None and row["status"] == "open"
        mine = pt.store.load_trades(pt.session_id)
        assert [t.market_id for t in mine] == ["m1"] and mine[0].status == TradeStatus.WON
        pt.store.close()


# ---------------------------------------------------------------------------
# 13. Available capital reflects losses
# ---------------------------------------------------------------------------

class TestAvailableCapital:
    def _closed(self, pnl):
        return PaperTrade(market_id="x", city_id="nyc", side="YES", size_usd=10,
                          entry_price=0.5, pnl=pnl, status=TradeStatus.LOST)

    def test_losses_reduce_capital_and_budgets(self):
        pt = PaperTrader(bankroll=1000.0)
        pt.trades.append(self._closed(-200.0))
        assert pt.deployment_base == pytest.approx(800.0)
        assert pt.available_capital == pytest.approx(800.0)
        assert pt.today_budget == pytest.approx(800.0 * pt._pool_today_pct)

    def test_profits_compound_by_profile(self):
        pt = PaperTrader(bankroll=1000.0)
        pt.trades.append(self._closed(200.0))
        # balanced compound_factor = 0.5
        assert pt.deployment_base == pytest.approx(1100.0)

    def test_open_exposure_and_fees_subtracted(self):
        pt = PaperTrader(bankroll=1000.0)
        pt.trades.append(self._closed(-100.0))
        pt.trades.append(PaperTrade(market_id="o", city_id="nyc", side="YES",
                                    size_usd=50, entry_price=0.5, fee_usd=1.0))
        assert pt.available_capital == pytest.approx(900.0 - 50.0 - 1.0)
