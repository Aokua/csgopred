"""Kaggle「CS:GO Professional Matches」导入（results / picks / players）。

数据集：mateusdmachado/csgo-professional-matches（需 Kaggle token）
覆盖 2015-10 ~ 2020-03，价值在于：
- 逐图比分（results.csv）：地图模型的基础；
- veto 顺序（picks.csv）：唯一现成的 veto 数据源；
- 选手逐场 rating（players.csv）：阵容强度的代理特征；
- 当时的 HLTV 排名（rank_1/rank_2）：避免「用今天的排名回填历史」。

去重策略：优先把地图/选手数据挂到库里已有的同一天同两队比赛上（HLTV 记录），
否则新建 `kaggle:<id>` 记录。这样训练时用 HLTV 的赛果，又能吃到 Kaggle 的地图信息。
"""

from __future__ import annotations

import csv
import json
import sqlite3
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from ..normalize import canonical_team

MAP_NAME_FIX = {"Dust2": "Dust2", "Dust 2": "Dust2", "Overpass": "Overpass"}


def _norm_map(name: str) -> str:
    n = (name or "").strip()
    return MAP_NAME_FIX.get(n, n)


def _f(x: str | None) -> float | None:
    try:
        v = float(x)  # type: ignore[arg-type]
        return v
    except (TypeError, ValueError):
        return None


def _i(x: str | None) -> int | None:
    v = _f(x)
    return int(v) if v is not None else None


def build_match_index(conn: sqlite3.Connection) -> dict[tuple, str]:
    """(日期, 队伍集合) -> match_id，用于把 Kaggle 地图数据挂到已有比赛上。"""
    idx: dict[tuple, str] = {}
    for r in conn.execute("SELECT match_id, start_date, team_a, team_b FROM matches"):
        key = (r["start_date"], frozenset((r["team_a"], r["team_b"])))
        idx.setdefault(key, r["match_id"])
    return idx


def _find(index: dict[tuple, str], d: str, ta: str, tb: str) -> str | None:
    day = date.fromisoformat(d)
    for delta in (0, 1, -1):
        key = ((day + timedelta(days=delta)).isoformat(), frozenset((ta, tb)))
        if key in index:
            return index[key]
    return None


def ingest_kaggle(conn: sqlite3.Connection, base_dir: str | Path, source: str = "kaggle") -> dict:
    base = Path(base_dir)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    index = build_match_index(conn)

    stats = {"matches_new": 0, "matches_attached": 0, "maps": 0, "veto": 0, "team_stats": 0}

    # ---------------- results：逐图 ----------------
    by_match: dict[str, list[dict]] = defaultdict(list)
    with open(base / "results.csv", encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("match_id"):
                by_match[row["match_id"]].append(row)

    match_rows = []
    map_rows = []
    for kid, rows in by_match.items():
        r0 = rows[0]
        d = (r0.get("date") or "")[:10]
        if not d:
            continue
        ta, tb = canonical_team(r0["team_1"]), canonical_team(r0["team_2"])
        if ta == tb:
            continue
        # 先确定实际打了几张图：单图场次里 map_wins_* 存的是回合比分，不是系列赛比分
        distinct = {_norm_map(mr.get("_map", "")) for mr in rows if mr.get("_map")}
        distinct.discard("")
        n_maps = len(distinct)
        if n_maps == 0:
            continue
        if n_maps == 1:
            s1, s2 = _i(r0.get("result_1")), _i(r0.get("result_2"))
            if s1 is None or s2 is None or s1 == s2:
                continue
            fmt = "bo1"
            wins1, wins2 = (1, 0) if s1 > s2 else (0, 1)
        else:
            wins1 = _i(r0.get("map_wins_1")) or 0
            wins2 = _i(r0.get("map_wins_2")) or 0
            if wins1 == wins2:
                # 列缺失时用地图胜负回退
                wins1 = sum(1 for mr in rows if (_i(mr.get("map_winner")) or 0) == 1)
                wins2 = sum(1 for mr in rows if (_i(mr.get("map_winner")) or 0) == 2)
            if wins1 == wins2:
                continue
            fmt = "bo5" if n_maps >= 4 or max(wins1, wins2) >= 3 else "bo3"
        winner = ta if wins1 > wins2 else tb
        existing = _find(index, d, ta, tb)
        if existing:
            mid = existing
            stats["matches_attached"] += 1
        else:
            mid = f"kaggle:{kid}"
            stats["matches_new"] += 1
            match_rows.append(
                (
                    mid, source, d, "", fmt, None, ta, tb, wins1, wins2, winner,
                    json.dumps({k: r0.get(k) for k in ("match_id", "event_id", "rank_1", "rank_2")},
                               ensure_ascii=False),
                    now, None, 0,
                    _i(r0.get("rank_1")), _i(r0.get("rank_2")),
                )
            )
            index[(d, frozenset((ta, tb)))] = mid
        seen = set()
        for i, mr in enumerate(rows):
            mname = _norm_map(mr.get("_map", ""))
            if not mname or mname in seen:
                continue
            seen.add(mname)
            s1, s2 = _i(mr.get("result_1")), _i(mr.get("result_2"))
            if s1 is None or s2 is None:
                continue
            map_rows.append((mid, i, mname, s1, s2, "a" if s1 > s2 else "b"))

    conn.executemany(
        """INSERT OR IGNORE INTO matches
           (match_id, source, start_date, event, format, map_name, team_a, team_b,
            score_a, score_b, winner, raw, ingested_at, slug, stars, rank_a, rank_b)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        match_rows,
    )
    conn.executemany(
        """INSERT OR REPLACE INTO maps (match_id, map_order, map_name, score_a, score_b, winner)
           VALUES (?,?,?,?,?,?)""",
        map_rows,
    )
    stats["maps"] += len(map_rows)
    conn.commit()

    # ---------------- picks：veto ----------------
    veto_rows = []
    with open(base / "picks.csv", encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            kid = row.get("match_id")
            d = (row.get("date") or "")[:10]
            if not kid or not d:
                continue
            ta, tb = canonical_team(row["team_1"]), canonical_team(row["team_2"])
            mid = _find(index, d, ta, tb) or f"kaggle:{kid}"
            order = 0
            for side, team in (("t1", ta), ("t2", tb)):
                for k in (1, 2, 3):
                    mp = _norm_map(row.get(f"{side}_removed_{k}", ""))
                    if mp and mp != "0.0" and mp != "0":
                        order += 1
                        veto_rows.append((mid, order, team, "removed", mp))
            for side, team in (("t1", ta), ("t2", tb)):
                mp = _norm_map(row.get(f"{side}_picked_1", ""))
                if mp and mp not in ("0.0", "0"):
                    order += 1
                    veto_rows.append((mid, order, team, "picked", mp))
            left = _norm_map(row.get("left_over", ""))
            if left and left not in ("0.0", "0"):
                order += 1
                veto_rows.append((mid, order, None, "left over", left))

    conn.executemany(
        "INSERT OR REPLACE INTO veto (match_id, ord, team, action, map_name) VALUES (?,?,?,?,?)",
        veto_rows,
    )
    stats["veto"] += len(veto_rows)
    conn.commit()

    # ---------------- players：队伍强度 ----------------
    ratings: dict[tuple, list[float]] = defaultdict(list)
    events: dict[str, str] = {}
    with open(base / "players.csv", encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            kid, d = row.get("match_id"), (row.get("date") or "")[:10]
            if kid and row.get("event_name") and kid not in events:
                events[kid] = row["event_name"]
            rating = _f(row.get("rating"))
            if not kid or not d or rating is None:
                continue
            team = canonical_team(row.get("team", ""))
            mid = _find(index, d, team, canonical_team(row.get("opponent", ""))) or f"kaggle:{kid}"
            ratings[(mid, team)].append(rating)

    team_rows = []
    for (mid, team), vals in ratings.items():
        if not vals:
            continue
        vals.sort(reverse=True)
        team_rows.append((mid, team, round(sum(vals) / len(vals), 4), round(vals[0], 4), len(vals)))

    conn.executemany(
        """INSERT OR REPLACE INTO team_match_stats
           (match_id, team, avg_rating, top_rating, n_players)
           VALUES (?,?,?,?,?)""",
        team_rows,
    )
    stats["team_stats"] += len(team_rows)
    conn.commit()

    for kid, ev in events.items():
        conn.execute(
            "UPDATE matches SET event=? WHERE match_id=? AND (event IS NULL OR event='')",
            (ev, f"kaggle:{kid}"),
        )
    conn.commit()

    from .csv_source import _refresh_teams

    _refresh_teams(conn)
    stats["matches_total"] = conn.execute("SELECT count(*) FROM matches").fetchone()[0]
    return stats
