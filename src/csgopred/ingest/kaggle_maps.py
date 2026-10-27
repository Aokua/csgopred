"""导入 Kaggle「Counter-Strike Pro Matches」的地图级数据（2023+）。

数据集：ektarr/counter-strike-pro-matches
每行是一张地图（含比分）或一场比赛汇总（map_name 为空）。
附带选手级统计，后续可用于阵容强度特征。
"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date, timedelta

import re

from ..db import connect
from ..normalize import canonical_team

_SUFFIX = re.compile(r"\b(esports?|gaming|team|club|esport|e-?sports|cs2?|gg)\b", re.I)


def norm(name: str) -> str:
    """宽松归一：去后缀、去标点、小写。用于跨数据源匹配队名。"""
    s = _SUFFIX.sub(" ", name or "")
    s = re.sub(r"[^a-z0-9]", "", s.lower())
    return s or re.sub(r"[^a-z0-9]", "", (name or "").lower())

PATH = "data/raw/kaggle_pro/cs2_all_tiers_games.csv"


def load_rows(path: str = PATH) -> dict[str, dict]:
    """按 match_id 聚合，只保留有地图的行。"""
    by_match: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("map_name"):
                continue
            mid = row["match_id"]
            rec = by_match.setdefault(
                mid,
                {
                    "date": row["datetime"][:10],
                    "team1": row["team1"],
                    "team2": row["team2"],
                    "bestof": row["bestOf"],
                    "tournament": row["tournament"],
                    "maps": [],
                },
            )
            try:
                sa = int(float(row["score1_game"]))
                sb = int(float(row["score2_game"]))
            except (TypeError, ValueError):
                continue
            rec["maps"].append({"map": row["map_name"], "sa": sa, "sb": sb})
    return by_match


def ingest(conn, path: str = PATH, dry: bool = False) -> dict:
    data = load_rows(path)
    # 建立我们的比赛索引
    idx: dict[tuple[str, str, str], str] = {}
    rows = conn.execute(
        "SELECT match_id, start_date, team_a, team_b FROM matches WHERE start_date >= '2023-01-01'"
    ).fetchall()
    for r in rows:
        a, b = norm(canonical_team(r["team_a"])), norm(canonical_team(r["team_b"]))
        idx[(r["start_date"], *sorted((a, b)))] = r["match_id"]

    matched = maps_n = 0
    unmatched = []
    for mid, rec in data.items():
        a, b = norm(canonical_team(rec["team1"])), norm(canonical_team(rec["team2"]))
        key = tuple(sorted((a, b)))
        found = None
        d0 = date.fromisoformat(rec["date"])
        for delta in (0, -1, 1):
            k = ((d0 + timedelta(days=delta)).isoformat(), *key)
            if k in idx:
                found = idx[k]
                break
        if not found:
            unmatched.append((rec["date"], rec["team1"], rec["team2"]))
            continue
        if not dry:
            for i, mp in enumerate(rec["maps"], 1):
                conn.execute(
                    """INSERT OR REPLACE INTO maps
                       (match_id, map_order, map_name, score_a, score_b, winner)
                       VALUES (?,?,?,?,?,?)""",
                    (found, i, mp["map"], mp["sa"], mp["sb"], "a" if mp["sa"] > mp["sb"] else "b"),
                )
                maps_n += 1
        matched += 1
    if not dry:
        conn.commit()
    return {"dataset_matches": len(data), "matched": matched, "maps": maps_n,
            "unmatched": len(unmatched), "samples": unmatched[:5]}

def build_lineups(conn, path: str = PATH) -> dict[str, dict]:
    """our_match_id -> {"a": [...], "b": [...], "date": date}

    用与地图导入相同的「日期 ±1 + 宽松队名」匹配。
    """
    from ..features.player_features import load_players

    players = load_players(path)
    idx: dict[tuple[str, str, str], tuple[str, str, str]] = {}
    for r in conn.execute(
        "SELECT match_id, start_date, team_a, team_b FROM matches WHERE start_date >= '2023-01-01'"
    ):
        a, b = norm(canonical_team(r["team_a"])), norm(canonical_team(r["team_b"]))
        idx[(r["start_date"], *sorted((a, b)))] = (r["match_id"], a, b)

    out: dict[str, dict] = {}
    for mid, rec in players.items():
        a, b = norm(canonical_team(rec["team1"])), norm(canonical_team(rec["team2"]))
        key = tuple(sorted((a, b)))
        d0 = date.fromisoformat(rec["date"])
        hit = None
        for delta in (0, -1, 1):
            k = ((d0 + timedelta(days=delta)).isoformat(), *key)
            if k in idx:
                hit = idx[k]
                break
        if not hit:
            continue
        our_id, na, nb = hit
        # 判断 dataset 的 team1 对应我们的哪一边
        if norm(canonical_team(rec["team1"])) == na:
            first, second = rec["p1"], rec["p2"]
        else:
            first, second = rec["p2"], rec["p1"]
        out[our_id] = {"date": d0, "a": first, "b": second}
    return out
