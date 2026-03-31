"""SQLite persistence for paper trades and session state.

Survives server restarts. No Docker needed, just a file on disk.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from weather_edge.trading.paper import PaperTrade, PaperTrader, TradeStatus

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "weather_edge.db"


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            bankroll REAL NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            final_pnl REAL,
            final_win_rate REAL,
            status TEXT DEFAULT 'active'
        );

        CREATE TABLE IF NOT EXISTS paper_trades (
            trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            market_id TEXT NOT NULL,
            city_id TEXT NOT NULL,
            side TEXT NOT NULL,
            size_usd REAL NOT NULL,
            entry_price REAL NOT NULL,
            placed_at TEXT NOT NULL,
            description TEXT,
            exit_price REAL,
            resolved_at TEXT,
            pnl REAL,
            status TEXT DEFAULT 'open',
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );

        CREATE TABLE IF NOT EXISTS state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS forecast_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER,
            city_id TEXT NOT NULL,
            target_date TEXT NOT NULL,
            model_name TEXT NOT NULL,
            forecast_value REAL,
            actual_value REAL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (trade_id) REFERENCES paper_trades(trade_id)
        );

        CREATE TABLE IF NOT EXISTS ai_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER,
            market_id TEXT,
            city_id TEXT,
            source TEXT NOT NULL,
            decision TEXT NOT NULL,
            rationale TEXT,
            confidence_adj REAL,
            dissent_strength REAL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (trade_id) REFERENCES paper_trades(trade_id)
        );

        CREATE TABLE IF NOT EXISTS live_trades (
            trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT UNIQUE,
            market_id TEXT NOT NULL,
            token_id TEXT NOT NULL,
            city_id TEXT NOT NULL,
            side TEXT NOT NULL,
            status TEXT DEFAULT 'open',
            limit_price REAL NOT NULL,
            avg_fill_price REAL,
            size_shares REAL NOT NULL,
            filled_shares REAL DEFAULT 0,
            size_usd REAL NOT NULL,
            is_maker BOOLEAN DEFAULT 1,
            fee_usd REAL DEFAULT 0,
            cost_basis REAL,
            proceeds REAL,
            pnl REAL,
            placed_at TEXT NOT NULL,
            filled_at TEXT,
            resolved_at TEXT,
            description TEXT,
            strategy TEXT DEFAULT 'core'
        );

        CREATE INDEX IF NOT EXISTS idx_live_trades_status
            ON live_trades(status);
        CREATE INDEX IF NOT EXISTS idx_live_trades_city
            ON live_trades(city_id);
        CREATE INDEX IF NOT EXISTS idx_live_trades_placed
            ON live_trades(placed_at);

        -- Immutable ledger: every fill from get_trades() (source of truth)
        CREATE TABLE IF NOT EXISTS fills (
            id TEXT PRIMARY KEY,
            order_id TEXT,
            asset_id TEXT NOT NULL,
            condition_id TEXT NOT NULL,
            city_id TEXT,
            side TEXT NOT NULL,
            size REAL NOT NULL,
            price REAL NOT NULL,
            fee_rate_bps INTEGER DEFAULT 0,
            is_maker INTEGER DEFAULT 1,
            filled_at TEXT NOT NULL,
            tx_hash TEXT,
            is_settled INTEGER DEFAULT 0,
            outcome TEXT,
            description TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_fills_asset
            ON fills(asset_id);
        CREATE INDEX IF NOT EXISTS idx_fills_condition
            ON fills(condition_id);
        CREATE INDEX IF NOT EXISTS idx_fills_settled
            ON fills(is_settled);

        -- Aggregated view: current net holdings per token
        CREATE TABLE IF NOT EXISTS positions (
            asset_id TEXT PRIMARY KEY,
            condition_id TEXT NOT NULL,
            city_id TEXT,
            side TEXT,
            outcome TEXT,
            total_shares REAL DEFAULT 0,
            avg_price REAL DEFAULT 0,
            cost_basis REAL DEFAULT 0,
            current_price REAL,
            current_value REAL,
            unrealized_pnl REAL,
            description TEXT,
            last_updated TEXT
        );

        -- Map asset_id to city/market for dashboard display
        CREATE TABLE IF NOT EXISTS market_map (
            asset_id TEXT PRIMARY KEY,
            condition_id TEXT NOT NULL,
            city_id TEXT,
            outcome TEXT,
            description TEXT,
            token_side TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_forecast_city_model
            ON forecast_snapshots(city_id, model_name);
        CREATE INDEX IF NOT EXISTS idx_forecast_date
            ON forecast_snapshots(target_date);
        CREATE INDEX IF NOT EXISTS idx_ai_decisions_trade
            ON ai_decisions(trade_id);
    """)


class PersistentStore:
    """SQLite-backed persistence for the paper trading system."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        _ensure_tables(self.conn)
        logger.info("Persistence: %s", self.db_path)

    def close(self) -> None:
        self.conn.close()

    # --- Sessions ---

    def get_active_session(self) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE status = 'active' ORDER BY session_id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def create_session(self, bankroll: float) -> int:
        cur = self.conn.execute(
            "INSERT INTO sessions (bankroll, started_at, status) VALUES (?, ?, 'active')",
            (bankroll, datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()
        session_id = cur.lastrowid
        logger.info("Created session %d with $%.0f bankroll", session_id, bankroll)
        return session_id

    def end_session(self, session_id: int, final_pnl: float, win_rate: float) -> None:
        self.conn.execute(
            "UPDATE sessions SET ended_at = ?, final_pnl = ?, final_win_rate = ?, status = 'ended' WHERE session_id = ?",
            (datetime.now(timezone.utc).isoformat(), final_pnl, win_rate, session_id),
        )
        self.conn.commit()

    # --- Trades ---

    def save_trade(self, session_id: int, trade: PaperTrade) -> int:
        cur = self.conn.execute(
            """INSERT INTO paper_trades
               (session_id, market_id, city_id, side, size_usd, entry_price, placed_at, description, exit_price, resolved_at, pnl, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                trade.market_id,
                trade.city_id,
                trade.side,
                trade.size_usd,
                trade.entry_price,
                trade.placed_at.isoformat(),
                trade.description,
                trade.exit_price,
                trade.resolved_at.isoformat() if trade.resolved_at else None,
                trade.pnl,
                trade.status.value,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def update_trade(self, trade_id: int, trade: PaperTrade) -> None:
        self.conn.execute(
            """UPDATE paper_trades SET exit_price = ?, resolved_at = ?, pnl = ?, status = ?
               WHERE trade_id = ?""",
            (
                trade.exit_price,
                trade.resolved_at.isoformat() if trade.resolved_at else None,
                trade.pnl,
                trade.status.value,
                trade_id,
            ),
        )
        self.conn.commit()

    def load_trades(self, session_id: int) -> list[PaperTrade]:
        rows = self.conn.execute(
            "SELECT * FROM paper_trades WHERE session_id = ? ORDER BY placed_at",
            (session_id,),
        ).fetchall()

        trades = []
        for r in rows:
            t = PaperTrade(
                trade_id=r["trade_id"],
                market_id=r["market_id"],
                city_id=r["city_id"],
                side=r["side"],
                size_usd=r["size_usd"],
                entry_price=r["entry_price"],
                placed_at=datetime.fromisoformat(r["placed_at"]),
                description=r["description"] or "",
                exit_price=r["exit_price"],
                resolved_at=datetime.fromisoformat(r["resolved_at"]) if r["resolved_at"] else None,
                pnl=r["pnl"],
                status=TradeStatus(r["status"]),
            )
            trades.append(t)
        return trades

    # --- Key-value state ---

    def get_state(self, key: str, default: str = "") -> str:
        row = self.conn.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_state(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
            (key, value),
        )
        self.conn.commit()

    # --- Live Trades (real money, tax-compliant) ---

    def save_live_trade(
        self,
        order_id: str,
        market_id: str,
        token_id: str,
        city_id: str,
        side: str,
        limit_price: float,
        size_shares: float,
        size_usd: float,
        description: str = "",
        strategy: str = "core",
        is_maker: bool = True,
    ) -> int:
        cur = self.conn.execute(
            """INSERT OR IGNORE INTO live_trades
               (order_id, market_id, token_id, city_id, side, limit_price,
                size_shares, size_usd, is_maker, placed_at, description, strategy)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                order_id, market_id, token_id, city_id, side, limit_price,
                size_shares, size_usd, is_maker,
                datetime.now(timezone.utc).isoformat(),
                description, strategy,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def update_live_trade_fill(
        self, order_id: str, avg_fill_price: float, filled_shares: float,
        fee_usd: float = 0.0, status: str = "filled",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        cost_basis = avg_fill_price * filled_shares + fee_usd
        self.conn.execute(
            """UPDATE live_trades
               SET avg_fill_price = ?, filled_shares = ?, fee_usd = ?,
                   cost_basis = ?, filled_at = ?, status = ?
               WHERE order_id = ?""",
            (avg_fill_price, filled_shares, fee_usd, cost_basis, now, status, order_id),
        )
        self.conn.commit()

    def resolve_live_trade(
        self, order_id: str, proceeds: float, pnl: float, status: str = "won",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """UPDATE live_trades
               SET proceeds = ?, pnl = ?, resolved_at = ?, status = ?
               WHERE order_id = ?""",
            (proceeds, pnl, now, status, order_id),
        )
        self.conn.commit()

    def cancel_live_trade(self, order_id: str) -> None:
        self.conn.execute(
            "UPDATE live_trades SET status = 'cancelled' WHERE order_id = ?",
            (order_id,),
        )
        self.conn.commit()

    def get_live_trades(self, limit: int = 100) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM live_trades ORDER BY placed_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_open_live_trades(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM live_trades WHERE status IN ('open', 'partial') ORDER BY placed_at DESC",
        ).fetchall()
        return [dict(r) for r in rows]

    def get_open_order_for_market(self, market_id: str) -> dict | None:
        """Check if there's already an open/partial order on this market."""
        row = self.conn.execute(
            "SELECT * FROM live_trades WHERE market_id = ? AND status IN ('open', 'partial') LIMIT 1",
            (market_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_live_stats(self) -> dict:
        """Summary stats for live trading dashboard."""
        row = self.conn.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END) as open_trades,
                SUM(CASE WHEN status = 'won' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN status = 'lost' THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN status IN ('open','partial') THEN size_usd ELSE 0 END) as capital_at_risk,
                COALESCE(SUM(pnl), 0) as total_pnl,
                COALESCE(SUM(fee_usd), 0) as total_fees,
                MAX(pnl) as best_trade,
                MIN(pnl) as worst_trade
            FROM live_trades
        """).fetchone()
        r = dict(row)
        total_resolved = (r["wins"] or 0) + (r["losses"] or 0)
        r["win_rate"] = (r["wins"] or 0) / total_resolved * 100 if total_resolved > 0 else 0
        return r

    # --- Fills & Positions (exchange truth) ---

    def upsert_fill(
        self, fill_id: str, order_id: str, asset_id: str, condition_id: str,
        city_id: str, side: str, size: float, price: float,
        filled_at: str, tx_hash: str = "", is_maker: int = 1,
        fee_rate_bps: int = 0, outcome: str = "", description: str = "",
    ) -> None:
        self.conn.execute(
            """INSERT OR IGNORE INTO fills
               (id, order_id, asset_id, condition_id, city_id, side, size, price,
                filled_at, tx_hash, is_maker, fee_rate_bps, outcome, description)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (fill_id, order_id, asset_id, condition_id, city_id, side, size, price,
             filled_at, tx_hash, is_maker, fee_rate_bps, outcome, description),
        )

    def upsert_market_map(
        self, asset_id: str, condition_id: str, city_id: str = "",
        outcome: str = "", description: str = "", token_side: str = "",
    ) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO market_map
               (asset_id, condition_id, city_id, outcome, description, token_side)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (asset_id, condition_id, city_id, outcome, description, token_side),
        )

    def rebuild_positions(self) -> None:
        """Rebuild positions table from fills (the truth)."""
        self.conn.execute("DELETE FROM positions")
        self.conn.execute("""
            INSERT INTO positions (asset_id, condition_id, city_id, side, outcome,
                                   total_shares, avg_price, cost_basis, last_updated, description)
            SELECT
                f.asset_id,
                MAX(f.condition_id),
                MAX(COALESCE(m.city_id, f.city_id, '')),
                MAX(f.side),
                MAX(COALESCE(m.outcome, f.outcome, '')),
                SUM(CASE WHEN f.side = 'BUY' THEN f.size ELSE -f.size END),
                CASE WHEN SUM(CASE WHEN f.side = 'BUY' THEN f.size ELSE 0 END) > 0
                     THEN SUM(CASE WHEN f.side = 'BUY' THEN f.size * f.price ELSE 0 END)
                          / SUM(CASE WHEN f.side = 'BUY' THEN f.size ELSE 0 END)
                     ELSE 0 END,
                SUM(CASE WHEN f.side = 'BUY' THEN f.size * f.price ELSE 0 END),
                datetime('now'),
                MAX(COALESCE(m.description, f.description, ''))
            FROM fills f
            LEFT JOIN market_map m ON f.asset_id = m.asset_id
            WHERE f.is_settled = 0
            GROUP BY f.asset_id
            HAVING SUM(CASE WHEN f.side = 'BUY' THEN f.size ELSE -f.size END) > 0
        """)
        # Backfill city_id from market_map for any positions missing it
        self.conn.execute("""
            UPDATE positions SET
                city_id = (SELECT m.city_id FROM market_map m WHERE m.asset_id = positions.asset_id),
                description = (SELECT m.description FROM market_map m WHERE m.asset_id = positions.asset_id),
                outcome = (SELECT m.outcome FROM market_map m WHERE m.asset_id = positions.asset_id)
            WHERE city_id = '' OR city_id IS NULL
        """)
        self.conn.commit()

    def get_positions(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM positions WHERE total_shares > 0 ORDER BY cost_basis DESC",
        ).fetchall()
        return [dict(r) for r in rows]

    def get_position_for_market(self, condition_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM positions WHERE condition_id = ? AND total_shares > 0 LIMIT 1",
            (condition_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_portfolio_summary(self) -> dict:
        row = self.conn.execute("""
            SELECT
                COUNT(*) as position_count,
                COALESCE(SUM(cost_basis), 0) as total_deployed,
                COALESCE(SUM(total_shares), 0) as total_shares
            FROM positions
            WHERE total_shares > 0
        """).fetchone()
        return dict(row)

    def get_fills(self, limit: int = 100) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM fills ORDER BY filled_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_fills_settled(self, condition_id: str) -> None:
        self.conn.execute(
            "UPDATE fills SET is_settled = 1 WHERE condition_id = ?",
            (condition_id,),
        )
        self.conn.commit()

    def commit(self) -> None:
        self.conn.commit()

    # --- Forecast Snapshots (for self-learning) ---

    def save_forecast_snapshot(
        self, city_id: str, target_date: str,
        model_values: dict[str, float], trade_id: int | None = None,
    ) -> None:
        """Save per-model forecast values for later Brier scoring."""
        now = datetime.now(timezone.utc).isoformat()
        for model_name, value in model_values.items():
            self.conn.execute(
                """INSERT INTO forecast_snapshots
                   (trade_id, city_id, target_date, model_name,
                    forecast_value, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (trade_id, city_id, target_date, model_name, value, now),
            )
        self.conn.commit()

    def backfill_actual(self, city_id: str, target_date: str, actual: float) -> int:
        """Fill in actual observed value for forecast snapshots."""
        cur = self.conn.execute(
            """UPDATE forecast_snapshots SET actual_value = ?
               WHERE city_id = ? AND target_date = ? AND actual_value IS NULL""",
            (actual, city_id, target_date),
        )
        self.conn.commit()
        return cur.rowcount

    def get_forecast_history(
        self, model_name: str | None = None, city_id: str | None = None,
        limit: int = 500,
    ) -> list[dict]:
        """Get resolved forecast snapshots for Brier scoring."""
        query = """SELECT * FROM forecast_snapshots
                   WHERE actual_value IS NOT NULL"""
        params: list = []
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        if city_id:
            query += " AND city_id = ?"
            params.append(city_id)
        query += " ORDER BY target_date DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self.conn.execute(query, params).fetchall()]

    # --- AI Decisions (for self-learning) ---

    def save_ai_decision(
        self, source: str, decision: str, city_id: str = "",
        market_id: str = "", trade_id: int | None = None,
        rationale: str = "", confidence_adj: float | None = None,
        dissent_strength: float | None = None,
    ) -> None:
        """Save an AI decision for later accuracy analysis."""
        self.conn.execute(
            """INSERT INTO ai_decisions
               (trade_id, market_id, city_id, source, decision,
                rationale, confidence_adj, dissent_strength, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (trade_id, market_id, city_id, source, decision,
             rationale, confidence_adj, dissent_strength,
             datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()

    def get_ai_decisions(self, source: str | None = None, limit: int = 200) -> list[dict]:
        """Get AI decisions for accuracy analysis."""
        query = "SELECT * FROM ai_decisions"
        params: list = []
        if source:
            query += " WHERE source = ?"
            params.append(source)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self.conn.execute(query, params).fetchall()]

    def get_cycle_count(self) -> int:
        return int(self.get_state("cycle_count", "0"))

    def increment_cycle(self) -> int:
        count = self.get_cycle_count() + 1
        self.set_state("cycle_count", str(count))
        return count


class PersistentPaperTrader(PaperTrader):
    """PaperTrader that persists trades to SQLite.

    Wraps the in-memory PaperTrader with automatic save/load.
    """

    def __init__(self, bankroll: float = 1000.0, db_path: Path | str = DEFAULT_DB_PATH):
        super().__init__(bankroll=bankroll)
        self.store = PersistentStore(db_path)
        self.session_id: int | None = None
        self._trade_id_map: dict[int, int] = {}  # in-memory trade_id -> db trade_id

        # Resume active session or create new one
        active = self.store.get_active_session()
        if active:
            self.session_id = active["session_id"]
            self.bankroll = active["bankroll"]
            self.trades = self.store.load_trades(self.session_id)
            self._next_id = max((t.trade_id or 0 for t in self.trades), default=0) + 1
            # Map loaded trade IDs
            for t in self.trades:
                if t.trade_id is not None:
                    self._trade_id_map[t.trade_id] = t.trade_id
            logger.info(
                "Resumed session %d: %d trades, $%.2f P&L, $%.0f bankroll",
                self.session_id, len(self.trades), self.total_pnl, self.bankroll,
            )
        else:
            self.session_id = self.store.create_session(bankroll)
            logger.info("Started new session %d", self.session_id)

    def place_trade(self, signal) -> PaperTrade | None:
        trade = super().place_trade(signal)
        if trade and self.session_id:
            db_id = self.store.save_trade(self.session_id, trade)
            if trade.trade_id is not None:
                self._trade_id_map[trade.trade_id] = db_id
        return trade

    def close_position(self, trade: PaperTrade, current_price: float) -> None:
        super().close_position(trade, current_price)
        if trade.trade_id is not None:
            db_id = self._trade_id_map.get(trade.trade_id, trade.trade_id)
            self.store.update_trade(db_id, trade)

    def resolve_trade(self, trade: PaperTrade, outcome_yes: bool) -> None:
        super().resolve_trade(trade, outcome_yes)
        if trade.trade_id is not None:
            db_id = self._trade_id_map.get(trade.trade_id, trade.trade_id)
            self.store.update_trade(db_id, trade)

    def close_all_positions(self, current_prices: dict[str, float] | None = None) -> float:
        result = super().close_all_positions(current_prices)
        return result

    def reset_session(self, bankroll: float | None = None) -> dict:
        final_stats = self.summary()
        # End current session in DB
        if self.session_id:
            self.store.end_session(self.session_id, self.total_pnl, self.win_rate)
        # Reset in-memory state
        self.trades = []
        self._next_id = 1
        self._trade_id_map = {}
        if bankroll is not None:
            self.bankroll = bankroll
        # Create new session
        self.session_id = self.store.create_session(self.bankroll)
        logger.info("New session %d started", self.session_id)
        return final_stats
