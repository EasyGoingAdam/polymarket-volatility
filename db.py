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

CREATE TABLE IF NOT EXISTS indicator_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    unix_ts REAL NOT NULL,
    rsi_14 REAL,
    macd_line REAL,
    macd_signal REAL,
    macd_histogram REAL,
    stoch_k REAL,
    stoch_d REAL,
    roc_12 REAL,
    williams_r REAL,
    atr_14 REAL,
    keltner_upper REAL,
    keltner_lower REAL,
    keltner_mid REAL,
    historical_vol REAL,
    vol_of_vol REAL,
    obv REAL,
    volume_roc REAL,
    liquidity_score REAL,
    cci_20 REAL,
    bollinger_upper REAL,
    bollinger_lower REAL,
    bollinger_sma REAL,
    bollinger_z REAL,
    adx REAL,
    hurst_exponent REAL,
    regime TEXT
);

CREATE TABLE IF NOT EXISTS composite_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    unix_ts REAL NOT NULL,
    signal_type TEXT NOT NULL,
    composite_score REAL,
    confidence REAL,
    mean_reversion_score REAL,
    momentum_score REAL,
    volatility_score REAL,
    orderflow_score REAL,
    volume_score REAL,
    kelly_fraction REAL,
    suggested_position_pct REAL,
    current_sharpe REAL,
    current_sortino REAL,
    max_drawdown REAL,
    var_95 REAL,
    regime TEXT,
    adx REAL,
    reasoning TEXT
);

CREATE TABLE IF NOT EXISTS risk_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    unix_ts REAL NOT NULL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    max_drawdown REAL,
    var_95 REAL,
    win_rate REAL,
    avg_profit REAL,
    kelly_fraction REAL,
    total_signals INTEGER,
    profitable_signals INTEGER
);

CREATE INDEX IF NOT EXISTS idx_indicators_ts ON indicator_snapshots(unix_ts);
CREATE INDEX IF NOT EXISTS idx_composite_ts ON composite_signals(unix_ts);
CREATE INDEX IF NOT EXISTS idx_risk_ts ON risk_metrics(unix_ts);
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


def store_indicator_snapshot(data: Dict):
    now = datetime.now(timezone.utc)
    with _conn() as conn:
        conn.execute(
            """INSERT INTO indicator_snapshots
               (timestamp, unix_ts, rsi_14, macd_line, macd_signal, macd_histogram,
                stoch_k, stoch_d, roc_12, williams_r, atr_14, keltner_upper, keltner_lower,
                keltner_mid, historical_vol, vol_of_vol, obv, volume_roc, liquidity_score,
                cci_20, bollinger_upper, bollinger_lower, bollinger_sma, bollinger_z,
                adx, hurst_exponent, regime)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                data.get("timestamp", now.isoformat()),
                data.get("unix_ts", now.timestamp()),
                data.get("rsi_14"), data.get("macd_line"), data.get("macd_signal"),
                data.get("macd_histogram"), data.get("stoch_k"), data.get("stoch_d"),
                data.get("roc_12"), data.get("williams_r"), data.get("atr_14"),
                data.get("keltner_upper"), data.get("keltner_lower"), data.get("keltner_mid"),
                data.get("historical_vol"), data.get("vol_of_vol"), data.get("obv"),
                data.get("volume_roc"), data.get("liquidity_score"), data.get("cci_20"),
                data.get("bollinger_upper"), data.get("bollinger_lower"),
                data.get("bollinger_sma"), data.get("bollinger_z"),
                data.get("adx"), data.get("hurst_exponent"), data.get("regime"),
            ),
        )


def get_recent_indicator_snapshots(n: int = 200) -> List[Dict]:
    with _conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM indicator_snapshots ORDER BY unix_ts DESC LIMIT ?", (n,)
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def store_composite_signal(data: Dict):
    now = datetime.now(timezone.utc)
    with _conn() as conn:
        conn.execute(
            """INSERT INTO composite_signals
               (timestamp, unix_ts, signal_type, composite_score, confidence,
                mean_reversion_score, momentum_score, volatility_score,
                orderflow_score, volume_score, kelly_fraction, suggested_position_pct,
                current_sharpe, current_sortino, max_drawdown, var_95,
                regime, adx, reasoning)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                data.get("timestamp", now.isoformat()),
                data.get("unix_ts", now.timestamp()),
                data["signal_type"], data.get("composite_score"), data.get("confidence"),
                data.get("mean_reversion_score"), data.get("momentum_score"),
                data.get("volatility_score"), data.get("orderflow_score"),
                data.get("volume_score"), data.get("kelly_fraction"),
                data.get("suggested_position_pct"), data.get("current_sharpe"),
                data.get("current_sortino"), data.get("max_drawdown"),
                data.get("var_95"), data.get("regime"), data.get("adx"),
                data.get("reasoning"),
            ),
        )


def get_recent_composite_signals(n: int = 50) -> List[Dict]:
    with _conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM composite_signals ORDER BY unix_ts DESC LIMIT ?", (n,)
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def store_risk_metrics(data: Dict):
    now = datetime.now(timezone.utc)
    with _conn() as conn:
        conn.execute(
            """INSERT INTO risk_metrics
               (timestamp, unix_ts, sharpe_ratio, sortino_ratio, max_drawdown,
                var_95, win_rate, avg_profit, kelly_fraction,
                total_signals, profitable_signals)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                data.get("timestamp", now.isoformat()),
                data.get("unix_ts", now.timestamp()),
                data.get("sharpe_ratio"), data.get("sortino_ratio"),
                data.get("max_drawdown"), data.get("var_95"),
                data.get("win_rate"), data.get("avg_profit"),
                data.get("kelly_fraction"), data.get("total_signals"),
                data.get("profitable_signals"),
            ),
        )


def get_latest_risk_metrics() -> Dict:
    with _conn() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM risk_metrics ORDER BY unix_ts DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else {}


def get_risk_metrics_history(n: int = 100) -> List[Dict]:
    with _conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM risk_metrics ORDER BY unix_ts DESC LIMIT ?", (n,)
        ).fetchall()
    return [dict(r) for r in reversed(rows)]
