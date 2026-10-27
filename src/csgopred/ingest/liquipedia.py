"""Liquipedia map data: parse scores from match popups on event pages.

Liquipedia's API is not behind Cloudflare, and every match popup on a page carries the map name
and total score, making it the only stable 2023+ map source available right now.

Limitations:
- Only covers events documented on Liquipedia (mainly tier 1/2);
- Parsing relies on the "total score half1 half2 map name ..." structure in the popup text;
  overtime maps contain extra numbers and are skipped (better to miss data than to be wrong).
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone

from bs4 import BeautifulSoup
from curl_cffi import requests as creq

API = "https://liquipedia.net/counterstrike/api.php"
UA = {"User-Agent": "csgopred-research/1.0 (local research; contact: local)"}

MAPS = ["Dust II", "Dust2", "Mirage", "Inferno", "Nuke", "Ancient", "Anubis",
        "Train", "Vertigo", "Overpass", "Cache", "Cobblestone", "Tuscan", "Season"]
_MAP_RE = re.compile(
    r"(\d{1,2}) (\d{1,2}) (\d{1,2}) (" + "|".join(MAPS) + r") (\d{1,2}) (\d{1,2}) (\d{1,2})"
)


def _session():
    return creq.Session(impersonate="chrome124")


def _get(params: dict, timeout=(10, 60)):
    """Request with 429 backoff. Liquipedia allows about 1 request per 2 seconds; exceeding it returns 429."""
    import time

    s = _session()
    last = None
    for attempt in range(4):
        try:
            r = s.get(API, params=params, headers=UA, timeout=timeout)
            if r.status_code == 429:
                time.sleep(45 * (attempt + 1))
                last = "429"
                continue
            return r
        except Exception as exc:
            last = exc
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"liquipedia request failed: {last}")


def search_titles(query: str, limit: int = 5) -> list[str]:
    """Try prefix search first (the event page is usually the query name), then fall back to full-text search."""
    import time

    s = _session()
    out: list[str] = []
    for lst in ("prefixsearch", "search"):
        time.sleep(2.0)  # Liquipedia rate limit: about 1 request / 2 seconds
        try:
            r = _get({"action": "query", "format": "json", "list": lst,
                      ("pssearch" if lst == "prefixsearch" else "srsearch"): query,
                      ("pslimit" if lst == "prefixsearch" else "srlimit"): limit}, timeout=(10, 30))
            items = r.json()["query"][lst if lst == "prefixsearch" else "search"]
        except Exception:
            continue
        for x in items:
            t = x.get("title")
            if t and t not in out:
                out.append(t)
        if out:
            break
    return out


def fetch_page(title: str) -> str:
    import time

    r = _get({"action": "parse", "format": "json", "page": title, "prop": "text"})
    d = r.json()
    if "error" in d:
        raise RuntimeError(d["error"].get("info", "parse error"))
    return d["parse"]["text"]["*"]


def parse_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    out = []
    for pop in soup.select(".brkts-popup"):
        ts = pop.select_one(".timer-object[data-timestamp]")
        if ts is None:
            continue
        try:
            d = datetime.fromtimestamp(int(ts["data-timestamp"]), tz=timezone.utc).date()
        except (ValueError, OSError):
            continue
        ops = pop.select(".match-info-header-opponent")
        t1 = t2 = None
        if len(ops) >= 2:
            a1 = ops[0].select_one("a[title]")
            a2 = ops[1].select_one("a[title]")
            t1 = a1["title"] if a1 else None
            t2 = a2["title"] if a2 else None
        if not (t1 and t2):
            continue
        text = re.sub(r"\s+", " ", pop.get_text(" ", strip=True))
        maps = []
        for m in _MAP_RE.finditer(text):
            maps.append({"map": m.group(4), "sa": int(m.group(1)), "sb": int(m.group(5))})
        if maps:
            out.append({"date": d, "team1": t1, "team2": t2, "maps": maps})
    return out


def store(conn, rows: list[dict], event_hint: str = "") -> dict:
    """Match parsed results against the matches table (date +/-1 day + normalized team names) and write them to maps."""
    from ..normalize import canonical_team

    written = matched = skipped = 0
    for r in rows:
        a, b = canonical_team(r["team1"]), canonical_team(r["team2"])
        lo, hi = r["date"].toordinal() - 1, r["date"].toordinal() + 1
        cand = conn.execute(
            """SELECT match_id, start_date, team_a, team_b FROM matches
               WHERE ((team_a=? AND team_b=?) OR (team_a=? AND team_b=?))
                 AND start_date >= date(?, '-2 day') AND start_date <= date(?, '+2 day')""",
            (a, b, b, a, r["date"].isoformat(), r["date"].isoformat()),
        ).fetchall()
        if not cand:
            skipped += 1
            continue
        mid = cand[0]["match_id"]
        for i, mp in enumerate(r["maps"]):
            conn.execute(
                """INSERT OR REPLACE INTO maps (match_id, map_order, map_name, score_a, score_b, winner)
                   VALUES (?,?,?,?,?,?)""",
                (mid, i, mp["map"].replace("Dust2", "Dust II"), mp["sa"], mp["sb"],
                 "a" if mp["sa"] > mp["sb"] else "b"),
            )
            written += 1
        matched += 1
    conn.commit()
    return {"parsed": len(rows), "matched": matched, "maps": written, "unmatched": skipped}
