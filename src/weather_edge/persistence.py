"""SQLite persistence for paper trades and session state.

Survives server restarts. No Docker needed, just a file on disk.
"""
from __future__ import annotations

import json
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
