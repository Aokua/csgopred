"""通用 CSV 赛果导入。

当前已适配的数据集（开源、无需鉴权）：
- hojlund123/csgo-dataset: date, team1, team2, team1score, team2score, event, bostyle, ...

列映射通过 COLUMN_ALIASES 自动识别，换数据集时只要列名在别名表里就能直接读。
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ..config import MANIFESTS
from ..normalize import canonical_team, parse_format, team_id

COLUMN_ALIASES = {
    "date": ("date", "start_date", "datetime", "match_date"),
    "team_a": ("team1", "team_a", "team_a_name", "winner_team"),
    "team_b": ("team2", "team_b", "team_b_name"),
    "score_a": ("team1score", "score_a", "team1_score", "score1"),
    "score_b": ("team2score", "score_b", "team2_score", "score2"),
    "event": ("event", "tournament", "event_name"),
    "format": ("bostyle", "format", "bo", "best_of"),
    "map": ("map", "map_name"),
}


def _pick(row: dict, field: str) -> str | None:
    for key in COLUMN_ALIASES[field]:
        if key in row and row[key] not in (None, ""):
            return str(row[key]).strip()
    return None


def _match_id(source: str, parts: list[str]) -> str:
    h = hashlib.sha1("|".join([source, *parts]).encode("utf-8")).hexdigest()
    return h[:16]


def ingest_csv(
    conn: sqlite3.Connection,
    path: str | Path,
    source: str = "csv",
    dry_run: bool = False,
) -> dict:
    path = Path(path)
    raw_bytes = path.read_bytes()
    sha = hashlib.sha256(raw_bytes).hexdigest()

    text = raw_bytes.decode("utf-8-sig")
    reader = csv.DictReader(text.splitlines())

    kept, skipped, ties = 0, 0, 0
    skip_reasons: dict[str, int] = {}
    seen_dates: list[str] = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for row in reader:
        try:
            date = _pick(row, "date")
            a_raw, b_raw = _pick(row, "team_a"), _pick(row, "team_b")
            sa_raw, sb_raw = _pick(row, "score_a"), _pick(row, "score_b")
            if not (date and a_raw and b_raw and sa_raw is not None and sb_raw is not None):
                raise ValueError("missing_field")
            date = date[:10]
            datetime.fromisoformat(date)
            sa, sb = int(sa_raw), int(sb_raw)
            if sa == sb:
                ties += 1
                skipped += 1
                skip_reasons["tie"] = skip_reasons.get("tie", 0) + 1
                continue

            team_a, team_b = canonical_team(a_raw), canonical_team(b_raw)
            if team_id(team_a) == team_id(team_b):
                skipped += 1
                skip_reasons["same_team"] = skip_reasons.get("same_team", 0) + 1
                continue

            fmt, map_name = parse_format(_pick(row, "format") or "")
            if map_name is None:
                map_name = _pick(row, "map")

            event = _pick(row, "event") or ""
            mid = _match_id(source, [date, team_a, team_b, str(sa), str(sb), event])
            winner = team_a if sa > sb else team_b

            conn.execute(
                """INSERT OR IGNORE INTO matches
                   (match_id, source, start_date, event, format, map_name,
                    team_a, team_b, score_a, score_b, winner, raw, ingested_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (mid, source, date, event, fmt, map_name, team_a, team_b,
                 sa, sb, winner, json.dumps(row, ensure_ascii=False), now),
            )
            kept += 1
            seen_dates.append(date)
        except (ValueError, KeyError, TypeError) as exc:
            skipped += 1
            reason = str(exc) or "invalid"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    if not dry_run:
        if seen_dates:
            _refresh_teams(conn)
        conn.execute(
            """INSERT INTO ingest_manifests
               (source, path, sha256, bytes, rows_read, rows_kept, rows_skip, created_at, note)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (source, str(path), sha, len(raw_bytes), kept + skipped, kept, skipped,
             now, json.dumps({"skip_reasons": skip_reasons}, ensure_ascii=False)),
        )
        conn.commit()

    manifest = {
        "source": source,
        "path": str(path),
        "sha256": sha,
        "bytes": len(raw_bytes),
        "rows_kept": kept,
        "rows_skipped": skipped,
        "ties": ties,
        "skip_reasons": skip_reasons,
        "date_min": min(seen_dates) if seen_dates else None,
        "date_max": max(seen_dates) if seen_dates else None,
    }
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    (MANIFESTS / f"{source}_{stamp}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def merge_team_variants(conn: sqlite3.Connection) -> int:
    """合并仅大小写/空白不同的队名变体（Falcons vs FALCONS）。

    取出场次数最多的写法作为规范名，其余变体改写为它。返回合并的变体数。
    """
    rows = conn.execute(
        """SELECT team, count(*) c FROM (
               SELECT team_a AS team FROM matches
               UNION ALL SELECT team_b FROM matches
           ) GROUP BY team"""
    ).fetchall()
    best: dict[str, tuple[int, str]] = {}
    for r in rows:
        key = (r["team"] or "").strip().casefold()
        if key not in best or r["c"] > best[key][0]:
            best[key] = (r["c"], r["team"])

    merged = 0
    for r in rows:
        key = (r["team"] or "").strip().casefold()
        target = best[key][1]
        if r["team"] == target:
            continue
        conn.execute("UPDATE matches SET team_a=? WHERE team_a=?", (target, r["team"]))
        conn.execute("UPDATE matches SET team_b=? WHERE team_b=?", (target, r["team"]))
        conn.execute("UPDATE OR IGNORE veto SET team=? WHERE team=?", (target, r["team"]))
        conn.execute("UPDATE OR IGNORE team_match_stats SET team=? WHERE team=?", (target, r["team"]))
        merged += 1
    conn.commit()
    return merged


def _refresh_teams(conn: sqlite3.Connection) -> None:
    merge_team_variants(conn)
    conn.execute("DELETE FROM teams")
    conn.execute(
        """
        INSERT INTO teams (team_id, name, first_seen, last_seen, matches)
        SELECT lower(trim(t.team)) AS team_id, max(t.team) AS name,
               min(t.d) AS first_seen, max(t.d) AS last_seen, count(*) AS matches
        FROM (
            SELECT team_a AS team, start_date AS d FROM matches
            UNION ALL
            SELECT team_b AS team, start_date AS d FROM matches
        ) t
        GROUP BY lower(trim(t.team))
        """
    )
    conn.commit()
