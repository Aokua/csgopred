"""HLTV scraping (result list + match details: map scores, veto).

Why this is needed: the open CSV datasets only cover 2019-2023 and have no maps/veto,
while CS2 starts in 2023-09, so new data must be collected. HLTV is behind Cloudflare and
plain requests gets a 403; this module uses curl_cffi's browser TLS fingerprint (safari17_0)
+ session warm-up + rate limiting + disk cache to scrape reliably. It still follows:
- 2.5s per request by default, with jitter;
- exponential backoff and session rebuild when a challenge page is hit, at most 4 attempts;
- all raw HTML is cached under data/raw/hltv/cache, so reruns do not re-request.

Note: this is low-frequency scraping for personal research; do not run it concurrently and do not bypass captchas.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup
from curl_cffi import requests as creq

from ..config import RAW
from ..normalize import canonical_team, parse_format

BASE = "https://www.hltv.org"
CACHE = RAW / "hltv" / "cache"


class HltvBlocked(RuntimeError):
    """Still blocked by Cloudflare after several retries."""


class HltvClient:
    def __init__(self, rate: float = 2.5, impersonate: str = "safari17_0", attempts: int = 4):
        self.rate = rate
        self.impersonate = impersonate
        self.attempts = attempts
        self._session = creq.Session(impersonate=impersonate)
        self._requests = 0
        CACHE.mkdir(parents=True, exist_ok=True)

    def _sleep(self) -> None:
        if self._requests:
            time.sleep(self.rate * (0.8 + 0.4 * random.random()))

    def get(self, path: str, use_cache: bool = True, warm: bool = True) -> str:
        key = hashlib.sha1(path.encode()).hexdigest()[:24]
        f = CACHE / f"{key}.html"
        if use_cache and f.exists() and f.stat().st_size > 2000:
            cached = f.read_text(encoding="utf-8", errors="ignore")
            if "Just a moment" not in cached[:3000]:
                return cached

        last_err = ""
        for attempt in range(self.attempts):
            self._sleep()
            if warm and self._requests == 0:
                self._session.get(f"{BASE}/results", timeout=(10, 30))
                self._requests += 1
                time.sleep(self.rate)
            try:
                r = self._session.get(
                    BASE + path,
                    timeout=(10, 30),
                    headers={"Referer": f"{BASE}/results"},
                )
                self._requests += 1
                body = r.text
                if r.status_code == 200 and "Just a moment" not in body[:3000]:
                    f.write_text(body, encoding="utf-8")
                    return body
                last_err = f"status={r.status_code}"
            except Exception as exc:  # Network jitter
                last_err = f"{type(exc).__name__}: {exc}"
            # Blocked: back off + rebuild the session
            time.sleep(8 * (attempt + 1))
            self._session = creq.Session(impersonate=self.impersonate)
        raise HltvBlocked(f"{path} 连续失败：{last_err}")

    # ------------------------------------------------------------ List pages
    def results(self, offset: int = 0) -> list[dict]:
        html = self.get(f"/results?offset={offset}")
        return parse_results(html)

    # ------------------------------------------------------------ Detail pages
    def match(self, match_id: str, slug: str) -> dict:
        html = self.get(f"/matches/{match_id}/{slug}")
        return parse_match(html)


def _to_date(ms: str | None) -> str | None:
    if not ms:
        return None
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).date().isoformat()


def parse_results(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    out = []
    for rc in soup.select("div.result-con"):
        a = rc.select_one("a[href^='/matches/']")
        if not a:
            continue
        m = re.search(r"/matches/(\d+)/([^/?#]+)", a["href"])
        if not m:
            continue
        teams = [d.get_text(strip=True) for d in rc.select("div.team")]
        scores = [int(x.get_text(strip=True)) for x in rc.select(".result-score span")]
        if len(teams) < 2 or len(scores) < 2:
            continue
        event_el = rc.select_one(".event-name")
        fmt_el = rc.select_one(".map-text") or rc.select_one(".map")
        date = _to_date(rc.get("data-zonedgrouping-entry-unix"))
        out.append(
            {
                "match_id": m.group(1),
                "slug": m.group(2),
                "date": date,
                "team_a": teams[0],
                "team_b": teams[1],
                "score_a": scores[0],
                "score_b": scores[1],
                "event": event_el.get_text(strip=True) if event_el else "",
                "format": (fmt_el.get_text(strip=True) if fmt_el else "bo3"),
                "stars": len(rc.select(".star")),
            }
        )
    return out


_VETO_RE = re.compile(r"^(\d+)\.\s+(.+?)\s+(removed|picked)\s+(.+)$")
_LEFTOVER_RE = re.compile(r"^(\d+)\.\s+(.+?)\s+was left over$")


def parse_match(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    names = [d.get_text(strip=True) for d in soup.select(".teamName")][:2]

    veto = []
    for box in soup.select(".veto-box"):
        padding = box.select_one(".padding")
        if padding is None:
            continue
        lines = [d.get_text(" ", strip=True) for d in padding.find_all("div", recursive=False)]
        if not lines:
            continue
        first = lines[0]
        if not re.match(r"^\d+\.", first):
            continue
        for line in lines:
            mv = _VETO_RE.match(line)
            if mv:
                veto.append(
                    {
                        "order": int(mv.group(1)),
                        "team": mv.group(2),
                        "action": mv.group(3),
                        "map": mv.group(4),
                    }
                )
                continue
            ml = _LEFTOVER_RE.match(line)
            if ml:
                veto.append(
                    {
                        "order": int(ml.group(1)),
                        "team": None,
                        "action": "left over",
                        "map": ml.group(2),
                    }
                )

    maps = []
    for i, mh in enumerate(soup.select(".mapholder")):
        name_el = mh.select_one(".mapname")
        left = mh.select_one(".results-left")
        right = mh.select_one(".results-right")
        if not (name_el and left and right):
            continue
        la = left.select_one(".results-team-score")
        rb = right.select_one(".results-team-score")
        if not (la and rb):
            continue
        name = name_el.get_text(strip=True)
        if name in ("Default", "TBD"):
            continue  # Placeholder rows for forfeits / undecided matches
        try:
            sa = int(la.get_text(strip=True))
            sb = int(rb.get_text(strip=True))
        except ValueError:
            continue  # Maps that were not played or were forfeited show "-"; skip them
        if sa == 0 and sb == 0:
            continue
        maps.append(
            {
                "order": i,
                "map": name,
                "score_a": sa,
                "score_b": sb,
                "a_won": "won" in (left.get("class") or []),
            }
        )

    return {"teams": names, "veto": veto, "maps": maps}


def ingest_results(
    conn,
    client: HltvClient,
    max_pages: int = 1,
    since: str | None = None,
    start_offset: int = 0,
) -> dict:
    """Scrape the result list and write it to the database. Returns statistics."""
    inserted = 0
    pages = 0
    oldest = None
    for page in range(max_pages):
        offset = start_offset + page * 100
        rows = client.results(offset=offset)
        pages += 1
        for r in rows:
            if not r["date"]:
                continue
            oldest = r["date"] if oldest is None else min(oldest, r["date"])
            if (r["format"] or "").strip().lower() in ("def", "default", "-", ""):
                continue  # Forfeit / default win-loss, not a real match, so it is excluded from the model
            fmt, _ = parse_format(r["format"])
            ta, tb = canonical_team(r["team_a"]), canonical_team(r["team_b"])
            if ta == tb:
                continue
            winner = ta if r["score_a"] > r["score_b"] else tb
            cur = conn.execute(
                """INSERT OR IGNORE INTO matches
                   (match_id, source, start_date, event, format, map_name,
                    team_a, team_b, score_a, score_b, winner, raw, ingested_at, slug, stars)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    f"hltv:{r['match_id']}", "hltv", r["date"], r["event"], fmt, None,
                    ta, tb, r["score_a"], r["score_b"], winner,
                    json.dumps(r, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    r["slug"], r["stars"],
                ),
            )
            inserted += cur.rowcount
        conn.commit()
        if since and oldest and oldest < since:
            break
    return {"pages": pages, "inserted": inserted, "oldest": oldest,
            "next_offset": start_offset + pages * 100}


def ingest_details(conn, client: HltvClient, limit: int = 10, since: str | None = None,
                   min_stars: int = 0) -> dict:
    """Scrape match details (map scores + veto), only for hltv matches that have no map data yet."""
    sql = """
        SELECT m.match_id, m.slug, m.start_date FROM matches m
        WHERE m.source='hltv' AND m.slug IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM maps x WHERE x.match_id = m.match_id)
    """
    params = []
    if since:
        sql += " AND m.start_date >= ?"
        params.append(since)
    if min_stars:
        sql += " AND m.stars >= ?"
        params.append(min_stars)
    sql += " ORDER BY m.start_date DESC LIMIT ?"
    params.append(limit)
    todo = conn.execute(sql, params).fetchall()

    done = 0
    maps_written = veto_written = 0
    for row in todo:
        mid, slug = row["match_id"], row["slug"]
        hltv_id = mid.split(":")[1]
        detail = client.match(hltv_id, slug)
        for mp in detail["maps"]:
            conn.execute(
                """INSERT OR REPLACE INTO maps
                   (match_id, map_order, map_name, score_a, score_b, winner)
                   VALUES (?,?,?,?,?,?)""",
                (mid, mp["order"], mp["map"], mp["score_a"], mp["score_b"],
                 "a" if mp["a_won"] else "b"),
            )
            maps_written += 1
        for v in detail["veto"]:
            conn.execute(
                """INSERT OR REPLACE INTO veto (match_id, ord, team, action, map_name)
                   VALUES (?,?,?,?,?)""",
                (mid, v["order"], v["team"], v["action"], v["map"]),
            )
            veto_written += 1
        conn.commit()
        done += 1
    return {"matches": done, "maps": maps_written, "veto": veto_written, "pending": len(todo)}
