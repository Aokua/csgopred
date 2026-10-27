"""Scrape HLTV match details (map scores + veto) with a CDP browser.

HLTV serves a Cloudflare challenge to direct HTTP requests, but a browser that executes JS gets through.
Throughput is about 3-4 seconds per match, suitable for slow background backfills.
"""

from __future__ import annotations

import time

from ..db import connect
from .browser import Browser
from .hltv import parse_match

BASE = "https://www.hltv.org/matches/"


def _todo(conn, limit: int, since: str | None, min_stars: int) -> list:
    sql = """
        SELECT m.match_id, m.slug, m.start_date FROM matches m
        WHERE m.source='hltv' AND m.slug IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM maps x WHERE x.match_id = m.match_id)
    """
    params: list = []
    if since:
        sql += " AND m.start_date >= ?"
        params.append(since)
    if min_stars:
        sql += " AND m.stars >= ?"
        params.append(min_stars)
    sql += " ORDER BY m.start_date DESC LIMIT ?"
    params.append(limit)
    return conn.execute(sql, params).fetchall()


def crawl(conn, limit: int = 50, since: str | None = "2023-01-01",
          min_stars: int = 0, pace: float = 2.5, out=print) -> dict:
    rows = _todo(conn, limit, since, min_stars)
    if not rows:
        return {"matches": 0, "maps": 0, "veto": 0, "pending": 0, "blocked": 0}
    out(f"待抓 {len(rows)} 场，最早 {rows[-1]['start_date']}")

    ok = maps_n = veto_n = blocked = skipped = 0
    with Browser() as b:
        b.new_page(BASE + rows[0]["slug"])
        for i, r in enumerate(rows, 1):
            url = BASE + r["slug"]
            try:
                html = b.goto(url, wait=2.5, ready_check="mapholder", max_wait=20)
            except Exception as exc:
                out(f"  [{i}] 打开失败 {type(exc).__name__}")
                skipped += 1
                continue
            if "mapholder" not in html and "veto-box" not in html:
                # May be a challenge page; wait a bit and retry once
                time.sleep(60)
                try:
                    html = b.goto(url, wait=3, ready_check="mapholder", max_wait=25)
                except Exception:
                    html = ""
                if "mapholder" not in html and "veto-box" not in html:
                    blocked += 1
                    out(f"  [{i}] 被挑战/无内容，停止本轮（已完成 {ok}）")
                    break
            detail = parse_match(html)
            for mp in detail["maps"]:
                conn.execute(
                    """INSERT OR REPLACE INTO maps
                       (match_id, map_order, map_name, score_a, score_b, winner)
                       VALUES (?,?,?,?,?,?)""",
                    (r["match_id"], mp["order"], mp["map"], mp["score_a"], mp["score_b"],
                     "a" if mp["a_won"] else "b"),
                )
                maps_n += 1
            for v in detail["veto"]:
                conn.execute(
                    """INSERT OR REPLACE INTO veto (match_id, ord, team, action, map_name)
                       VALUES (?,?,?,?,?)""",
                    (r["match_id"], v["order"], v["team"], v["action"], v["map"]),
                )
                veto_n += 1
            conn.commit()
            ok += 1
            if i % 20 == 0:
                out(f"  [{i}/{len(rows)}] maps={maps_n} veto={veto_n} 最新 {r['start_date']}")
            time.sleep(pace)
    return {"matches": ok, "maps": maps_n, "veto": veto_n,
            "pending": len(rows), "blocked": blocked, "skipped": skipped}
