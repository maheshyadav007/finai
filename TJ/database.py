"""
database.py
-----------

This module encapsulates all interactions with the underlying SQLite
database used to persist trade data. Keeping database logic here makes
it easy to change the storage backend in the future (e.g. switching
to PostgreSQL) without affecting other parts of the application.

Functions and classes defined here provide CRUD operations for Trade
objects.
"""

import sqlite3
from datetime import datetime
from .models import Trade
from typing import List, Dict, Any, Iterable

# If Trade is declared in models.py:
# from .models import Trade

class TradeJournalDB:
    """SQLite-backed repository for trades + OHLC candles."""

    def __init__(self, db_path: str = "tradezella.db") -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    # ---------- schema ----------
    def _create_tables(self) -> None:
        """Create required tables (trades, ohlc) and indexes."""
        with self.conn:
            # Trades table
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    instrument TEXT NOT NULL,
                    date_time TEXT NOT NULL, -- ISO8601
                    position TEXT NOT NULL CHECK (position IN ('long','short')),
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    notes TEXT
                )
                """
            )
            # OHLC candles table
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ohlc (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    ts INTEGER NOT NULL,   -- epoch ms
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL DEFAULT 0,
                    UNIQUE(symbol, ts)
                )
                """
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ohlc_symbol_ts ON ohlc(symbol, ts)"
            )

    # ---------- trades ----------
    def add_trade(self, trade: "Trade") -> None:
        """Insert a new trade."""
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO trades
                    (instrument, date_time, position, entry_price, exit_price, quantity, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.instrument,
                    trade.date_time.isoformat(),
                    trade.position.lower(),
                    trade.entry_price,
                    trade.exit_price,
                    trade.quantity,
                    trade.notes,
                ),
            )

    def list_trades(self) -> List["Trade"]:
        """Return all trades ordered by date_time."""
        cur = self.conn.execute("SELECT * FROM trades ORDER BY date_time")
        rows = cur.fetchall()
        return [self._row_to_trade(r) for r in rows]

    def trades_between(self, start_date: datetime, end_date: datetime) -> List["Trade"]:
        """Return trades within [start_date, end_date]."""
        cur = self.conn.execute(
            """
            SELECT * FROM trades
            WHERE date_time >= ? AND date_time <= ?
            ORDER BY date_time
            """,
            (start_date.isoformat(), end_date.isoformat()),
        )
        rows = cur.fetchall()
        return [self._row_to_trade(r) for r in rows]

    def _row_to_trade(self, row: sqlite3.Row) -> "Trade":
        """Convert DB row -> Trade."""
        return Trade(
            id=row["id"],
            instrument=row["instrument"],
            date_time=datetime.fromisoformat(row["date_time"]),
            position=row["position"],
            entry_price=row["entry_price"],
            exit_price=row["exit_price"],
            quantity=row["quantity"],
            notes=row["notes"] or "",
        )

    # ---------- ohlc ----------
    def _normalize_epoch(self, ts: int) -> int:
        """Accept seconds or milliseconds; store as milliseconds."""
        return ts * 1000 if ts < 10_000_000_000 else ts

    def upsert_ohlc_rows(self, rows: Iterable[Dict[str, Any]]) -> int:
        """
        Upsert OHLC rows.
        Each row must include: symbol, ts (epoch sec or ms), open, high, low, close
        Optional: volume
        """
        inserted = 0
        with self.conn:
            for r in rows:
                symbol = str(r["symbol"]).strip().upper()
                ts = self._normalize_epoch(int(r["ts"]))
                o, h, l, c = float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])
                v = float(r.get("volume", 0.0))
                self.conn.execute(
                    """
                    INSERT INTO ohlc(symbol, ts, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(symbol, ts) DO UPDATE SET
                        open=excluded.open,
                        high=excluded.high,
                        low=excluded.low,
                        close=excluded.close,
                        volume=excluded.volume
                    """,
                    (symbol, ts, o, h, l, c, v),
                )
                inserted += 1
        return inserted

    def get_symbols(self) -> List[str]:
        """List distinct symbols available in OHLC."""
        cur = self.conn.execute("SELECT DISTINCT symbol FROM ohlc ORDER BY symbol")
        return [row[0] for row in cur.fetchall()]

    def fetch_ohlc(self, symbol: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Fetch ascending OHLC for a symbol.
        Returns dicts {time(sec), open, high, low, close, volume} for Lightweight Charts.
        """
        cur = self.conn.execute(
            """
            SELECT ts, open, high, low, close, volume
            FROM ohlc
            WHERE symbol = ?
            ORDER BY ts ASC
            LIMIT ?
            """,
            (symbol.upper(), limit),
        )
        out: List[Dict[str, Any]] = []
        for ts, o, h, l, c, v in cur.fetchall():
            out.append({
                "time": int(ts // 1000),  # ms -> sec
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": float(v or 0.0),
            })
        return out

    # ---------- housekeeping ----------
    def close(self) -> None:
        self.conn.close()
