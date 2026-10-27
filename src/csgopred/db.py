"""SQLite 存储层。

选择 SQLite 而非 DuckDB 的原因：Python 3.14 暂无 duckdb wheel，
且本项目数据量（~5 万场）用 SQLite 完全够用，核心管线保持零第三方依赖
（HLTV 抓取是可选依赖 curl_cffi，不影响回测与预测）。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    match_id   TEXT PRIMARY KEY,
    source     TEXT NOT NULL,
    start_date TEXT NOT NULL,
    event      TEXT,
    format     TEXT NOT NULL,          -- bo1 / bo3 / bo5
    map_name   TEXT,                   -- bo1 时已知的地图
    team_a     TEXT NOT NULL,
    team_b     TEXT NOT NULL,
    score_a    INTEGER NOT NULL,
    score_b    INTEGER NOT NULL,
    winner     TEXT NOT NULL,
    raw        TEXT,                   -- 原始行 JSON，便于追溯
    ingested_at TEXT NOT NULL,
    slug       TEXT,                   -- HLTV match slug（用于抓详情）
    stars      INTEGER DEFAULT 0,      -- HLTV 赛事星级（0-5）
    rank_a     INTEGER,                -- 比赛当时的 HLTV 排名（Kaggle 数据集）
    rank_b     INTEGER
);

CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(start_date);
CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(team_a, team_b);

CREATE TABLE IF NOT EXISTS maps (
    match_id  TEXT NOT NULL,
    map_order INTEGER NOT NULL,
    map_name  TEXT NOT NULL,
    score_a   INTEGER NOT NULL,
    score_b   INTEGER NOT NULL,
    winner    TEXT NOT NULL,           -- a / b
    PRIMARY KEY (match_id, map_order)
);

CREATE TABLE IF NOT EXISTS veto (
    match_id TEXT NOT NULL,
    ord      INTEGER NOT NULL,
    team     TEXT,
    action   TEXT NOT NULL,            -- removed / picked / left over
    map_name TEXT NOT NULL,
    PRIMARY KEY (match_id, ord)
);

CREATE TABLE IF NOT EXISTS team_match_stats (
    match_id   TEXT NOT NULL,
    team       TEXT NOT NULL,
    avg_rating REAL,
    top_rating REAL,
    n_players  INTEGER,
    PRIMARY KEY (match_id, team)
);

CREATE TABLE IF NOT EXISTS teams (
    team_id    TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    first_seen TEXT,
    last_seen  TEXT,
    matches    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ingest_manifests (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    source     TEXT NOT NULL,
    path       TEXT NOT NULL,
    sha256     TEXT NOT NULL,
    bytes      INTEGER NOT NULL,
    rows_read  INTEGER NOT NULL,
    rows_kept  INTEGER NOT NULL,
    rows_skip  INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    note       TEXT
);
"""


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    db_path = Path(path) if path else DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """给已存在的旧库补列（幂等）。"""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(matches)")}
    for name, ddl in (("slug", "TEXT"), ("stars", "INTEGER DEFAULT 0"),
                      ("rank_a", "INTEGER"), ("rank_b", "INTEGER")):
        if name not in cols:
            conn.execute(f"ALTER TABLE matches ADD COLUMN {name} {ddl}")
    conn.commit()


def reset(conn: sqlite3.Connection) -> None:
    """清空业务表（保留 schema），仅用于本地重建。"""
    for t in ("matches", "maps", "veto", "team_match_stats", "teams", "ingest_manifests"):
        conn.execute(f"DELETE FROM {t}")
    conn.commit()
