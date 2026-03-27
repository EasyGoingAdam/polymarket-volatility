"""SQLite storage for price snapshots and trading signals."""
from __future__ import annotations

import sqlite3
from typing import List, Dict
from datetime import datetime, timezone
from config import DB_PATH

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS price_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    unix_ts REAL NOT NULL,
    yes_price REAL NOT NULL,
    best_bid REAL,
    best_ask REAL,
    spread REAL,
    imbalance REAL,
    volume_24hr REAL,
    liquidity REAL,
    is_backfill INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    unix_ts REAL NOT NULL,
    signal_type TEXT NOT NULL,
    price_at_signal REAL NOT NULL,
    bollinger_upper REAL,
    bollinger_lower REAL,
    bollinger_mean REAL,
    z_score REAL,
    strength REAL,
    imbalance REAL
);

CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON price_snapshots(unix_ts);
CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(unix_ts);
"""


def _conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    with _conn() as conn:
        conn.executescript(_CREATE_TABLES)


def store_snapshot(data: dict):
    now = datetime.now(timezone.utc)
    with _conn() as conn:
        conn.execute(
            """INSERT INTO price_snapshots
               (timestamp, unix_ts, yes_price, best_bid, best_ask, spread, imbalance, volume_24hr, liquidity, is_backfill)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data.get("timestamp", now.isoformat()),
                data.get("unix_ts", now.timestamp()),
                data["yes_price"],
                data.get("best_bid"),
                data.get("best_ask"),
                data.get("spread"),
                data.get("imbalance"),
                data.get("volume_24hr"),
                data.get("liquidity"),
                data.get("is_backfill", 0),
            ),
        )


def get_recent_snapshots(n: int = 200) -> List[Dict]:
    with _conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM price_snapshots ORDER BY unix_ts DESC LIMIT ?", (n,)
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def store_signal(data: dict):
    now = datetime.now(timezone.utc)
    with _conn() as conn:
        conn.execute(
            """INSERT INTO signals
               (timestamp, unix_ts, signal_type, price_at_signal,
                bollinger_upper, bollinger_lower, bollinger_mean, z_score, strength, imbalance)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data.get("timestamp", now.isoformat()),
                data.get("unix_ts", now.timestamp()),
                data["signal_type"],
                data["price_at_signal"],
                data.get("bollinger_upper"),
                data.get("bollinger_lower"),
                data.get("bollinger_mean"),
                data.get("z_score"),
                data.get("strength"),
                data.get("imbalance"),
            ),
        )


def get_recent_signals(n: int = 50) -> List[Dict]:
    with _conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM signals ORDER BY unix_ts DESC LIMIT ?", (n,)
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def get_snapshot_count() -> int:
    with _conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM price_snapshots").fetchone()[0]
