"""Walk-forward backtest.

Protocol (must not be changed):
1. Sort by time; the first split_ratio is the training period and the rest is the test period;
2. Training period: advance Elo over time, collect (elo_diff, y), and fit the probability mapping;
3. Test period: keep advancing Elo (using only earlier matches); predict each match before updating;
4. Metrics are computed on the test period only.

This measures the real performance of "using the model match by match after launch", not inflated
numbers from training on future data.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import date

from ..features.elo import EloModel, expected_score
from ..models.rating import fit_logistic, predict_from_elo
from .metrics import summarize


SOURCE_PRIORITY = {"hltv": 2, "csv": 1, "hojlund": 1}


def load_matches(conn: sqlite3.Connection, date_from: str | None = None,
                 date_to: str | None = None, dedupe: bool = True) -> list[dict]:
    sql = "SELECT * FROM matches"
    where, params = [], []
    if date_from:
        where.append("start_date >= ?")
        params.append(date_from)
    if date_to:
        where.append("start_date <= ?")
        params.append(date_to)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY start_date, match_id"
    rows = [dict(r) for r in conn.execute(sql, params)]
    if not dedupe:
        return rows

    # The same match may come from both CSV and HLTV: dedupe by "date + both team names", preferring HLTV
    # (HLTV has maps and veto, so its information is more complete).
    best: dict[tuple, tuple[int, dict]] = {}
    for r in rows:
        key = (r["start_date"], tuple(sorted((r["team_a"], r["team_b"]))))
        prio = SOURCE_PRIORITY.get(r["source"], 0)
        if key not in best or prio > best[key][0]:
            best[key] = (prio, r)
    return sorted((v[1] for v in best.values()),
                  key=lambda r: (r["start_date"], r["match_id"]))


def walk_forward(
    matches: list[dict],
    split_ratio: float = 0.7,
    min_train: int = 1000,
) -> dict:
    if len(matches) < min_train + 100:
        raise ValueError(f"样本不足：{len(matches)} 场，至少需要 {min_train + 100} 场")

    split = max(min_train, int(len(matches) * split_ratio))
    train, test = matches[:split], matches[split:]

    model = EloModel()
    xs: list[float] = []
    ys: list[int] = []
    for m in train:
        d = model.step(m)
        xs.append((d["rating_a"] - d["rating_b"]) / 400.0)
        ys.append(int(d["y"]))

    a, b = fit_logistic(xs, ys)

    ps, raw_ps, ys_test, meta = [], [], [], []
    for m in test:
        d = model.step(m)  # Compute the pre-match ratings first, then update state
        x = (d["rating_a"] - d["rating_b"]) / 400.0
        p = predict_from_elo(a, b, d["rating_a"] - d["rating_b"])
        ps.append(p)
        raw_ps.append(expected_score(d["rating_a"], d["rating_b"]))
        ys_test.append(int(d["y"]))
        meta.append(
            {
                "date": m["start_date"],
                "team_a": m["team_a"],
                "team_b": m["team_b"],
                "format": m["format"],
                "event": m["event"],
                "p": p,
                "y": int(d["y"]),
                "elo_a": round(d["rating_a"], 1),
                "elo_b": round(d["rating_b"], 1),
            }
        )

    seg = defaultdict(lambda: {"ps": [], "ys": []})
    for row in meta:
        for key in (row["format"], row["date"][:4]):
            seg[key]["ps"].append(row["p"])
            seg[key]["ys"].append(row["y"])

    segments = {
        str(k): summarize(v["ps"], v["ys"], name=str(k)) for k, v in sorted(seg.items())
    }

    return {
        "split_index": split,
        "train_n": len(train),
        "test_n": len(test),
        "train_range": (train[0]["start_date"], train[-1]["start_date"]),
        "test_range": (test[0]["start_date"], test[-1]["start_date"]),
        "logistic": {"a": round(a, 5), "b": round(b, 5)},
        "model": summarize(ps, ys_test, "elo+logistic"),
        "elo_raw": summarize(raw_ps, ys_test, "elo raw (scale=400)"),
        "coin": summarize([0.5] * len(ys_test), ys_test, "0.5 baseline"),
        "segments": segments,
        "predictions": meta,
        "final_state": model,
    }


def train_full(matches: list[dict], asof: str | None = None) -> tuple[EloModel, float, float]:
    """Replay all matches (up to asof) and return (model, a, b)."""
    model = EloModel()
    xs, ys = [], []
    cutoff = date.fromisoformat(asof) if asof else None
    for m in matches:
        if cutoff and date.fromisoformat(m["start_date"]) >= cutoff:
            break
        d = model.step(m)
        xs.append((d["rating_a"] - d["rating_b"]) / 400.0)
        ys.append(int(d["y"]))
    a, b = fit_logistic(xs, ys)
    return model, a, b
