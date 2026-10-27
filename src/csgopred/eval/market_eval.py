"""M4：把赔率与模型对齐，输出市场对比指标。

流程：
1. 训练集 = 赔率窗口之前的比赛（避免用未来训练）；
2. 用 M2 模型预测赔率窗口内的比赛；
3. 与收盘赔率（去水后）比较 log loss / Brier；
4. 计算 CLV 分布与凯利 ROI。
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta

from ..features.pipeline import FEATURES, build
from ..models.market import Bet, clv, devig_proportional, log_loss, brier, roi
from ..models.ml import Model, labels, matrix, time_weights

SCHEMA = """
CREATE TABLE IF NOT EXISTS odds (
    match_key  TEXT PRIMARY KEY,   -- date|team_a|team_b（排序后）
    start_date TEXT NOT NULL,
    team_a     TEXT NOT NULL,
    team_b     TEXT NOT NULL,
    odds_a     REAL NOT NULL,
    odds_b     REAL NOT NULL,
    book       TEXT,
    captured_at TEXT,
    is_closing INTEGER DEFAULT 1,
    source     TEXT
);
"""


def _key(d: str, a: str, b: str) -> str:
    x, y = sorted((a, b))
    return f"{d}|{x}|{y}"


def load_odds_csv(path: str, source: str = "csv") -> list[dict]:
    import csv

    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append(
                    {
                        "start_date": r["date"][:10],
                        "team_a": r["team_a"].strip(),
                        "team_b": r["team_b"].strip(),
                        "odds_a": float(r["odds_a"]),
                        "odds_b": float(r["odds_b"]),
                        "book": r.get("book", ""),
                        "captured_at": r.get("captured_at", ""),
                        "is_closing": int(r.get("is_closing", 1) or 1),
                        "source": source,
                    }
                )
            except (KeyError, ValueError):
                continue
    return rows


def store_odds(conn: sqlite3.Connection, rows: list[dict]) -> int:
    conn.executescript(SCHEMA)
    n = 0
    for r in rows:
        cur = conn.execute(
            """INSERT OR REPLACE INTO odds
               (match_key, start_date, team_a, team_b, odds_a, odds_b, book,
                captured_at, is_closing, source)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (_key(r["start_date"], r["team_a"], r["team_b"]), r["start_date"], r["team_a"],
             r["team_b"], r["odds_a"], r["odds_b"], r["book"], r["captured_at"],
             r["is_closing"], r["source"]),
        )
        n += cur.rowcount
    conn.commit()
    return n


def evaluate(conn: sqlite3.Connection, odds_window_start: str | None = None) -> dict:
    from ..eval.backtest import load_matches

    conn.executescript(SCHEMA)
    odds = [dict(r) for r in conn.execute("SELECT * FROM odds")]
    if not odds:
        return {"error": "odds 表为空，先导入收盘赔率"}

    start = odds_window_start or min(o["start_date"] for o in odds)
    matches = load_matches(conn)
    train = [m for m in matches if m["start_date"] < start]
    if len(train) < 5000:
        return {"error": f"训练样本不足：{len(train)}"}

    rows = build(matches)
    by_key = {}
    for r in rows:
        k = _key(r["date"], r["team_a"], r["team_b"])
        by_key.setdefault(k, r)

    train_rows = []
    for m in train:
        r = by_key.get(_key(m["start_date"], m["team_a"], m["team_b"]))
        if r is not None:
            train_rows.append(r)

    model = Model("lr")
    model.feats = list(FEATURES)
    model.fit(matrix(train_rows, model.feats), labels(train_rows), time_weights(train_rows))

    ps, ys, qs, closes, clvs, bets = [], [], [], [], [], []
    skipped = 0
    for o in odds:
        k = _key(o["start_date"], o["team_a"], o["team_b"])
        r = by_key.get(k)
        if r is None:
            skipped += 1
            continue
        p = float(model.predict(matrix([r], model.feats))[0])
        q, qb = devig_proportional(o["odds_a"], o["odds_b"])
        ps.append(p)
        ys.append(r["y"])
        qs.append(q)
        closes.append(o["odds_a"])
        clvs.append(clv(p, o["odds_a"]))
        bets.append(Bet(p_model=p, odds=o["odds_a"], won=bool(r["y"] == 1)))

    if not ps:
        return {"error": "没有能对齐的赔率记录", "skipped": skipped}

    return {
        "n": len(ps),
        "skipped": skipped,
        "window": (min(o["start_date"] for o in odds), max(o["start_date"] for o in odds)),
        "model": {"log_loss": round(log_loss(ps, ys), 4), "brier": round(brier(ps, ys), 4)},
        "market": {"log_loss": round(log_loss(qs, ys), 4), "brier": round(brier(qs, ys), 4)},
        "clv_mean": round(sum(clvs) / len(clvs), 4),
        "clv_pos_rate": round(sum(1 for c in clvs if c > 0) / len(clvs), 4),
        "roi_threshold_0": roi(bets, 0.0),
        "roi_threshold_3pct": roi(bets, 0.03),
        "roi_threshold_5pct": roi(bets, 0.05),
    }
