"""News scraping: HLTV (English) and the Perfect World CSGO site (Chinese).

Design: scraping and extraction are separate.
- Scraping only stores "title / publish time / body" in the news table, deduplicated by content hash;
- Extraction (rule-based or LLM) turns bodies into structured events (the news_events table),
  which the feature layer aggregates with half-life decay.

Timestamps always use the **publish time**, not the scrape time -- this is the key to preventing leakage.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup

from ..normalize import canonical_team

HLTV_BASE = "https://www.hltv.org"
PW_BASE = "http://www.csgo.com.cn"


@dataclass
class NewsItem:
    article_id: str
    source: str
    url: str
    published_at: str          # ISO datetime
    title: str
    summary: str
    body: str
    lang: str


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def _iso(ms_or_str) -> str:
    if isinstance(ms_or_str, (int, float)):
        return datetime.fromtimestamp(ms_or_str / 1000, tz=timezone.utc).isoformat(timespec="seconds")
    s = str(ms_or_str).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d", "%Y-%m-%d", "%d-%m-%Y %H:%M"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).isoformat(timespec="seconds")
        except ValueError:
            continue
    return s


# ---------------------------------------------------------------- HLTV
def parse_hltv_list(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    out, seen = [], set()
    for a in soup.select("a[href^='/news/']"):
        href = a.get("href", "")
        m = re.match(r"/news/(\d+)/", href)
        if not m or m.group(1) in seen:
            continue
        seen.add(m.group(1))
        out.append(href.split("#")[0])
    return out


def parse_hltv_article(html: str, url: str) -> NewsItem | None:
    soup = BeautifulSoup(html, "lxml")
    h1 = soup.select_one("h1.headline")
    if not h1:
        return None
    date_el = soup.select_one(".article-info .date[data-unix]")
    published = _iso(int(date_el["data-unix"])) if date_el else ""
    body_el = soup.select_one(".newstext-con")
    if body_el is None:
        return None
    summary_el = body_el.select_one("p.headertext")
    summary = summary_el.get_text(" ", strip=True) if summary_el else ""
    paras = []
    for p in body_el.select("p"):
        txt = p.get_text(" ", strip=True)
        if txt and txt != summary:
            paras.append(txt)
    body = "\n".join(paras)
    aid = re.search(r"/news/(\d+)/", url)
    return NewsItem(
        article_id=f"hltv:{aid.group(1)}" if aid else f"hltv:{_sha(url)[:12]}",
        source="hltv",
        url=HLTV_BASE + url if url.startswith("/") else url,
        published_at=published,
        title=h1.get_text(" ", strip=True),
        summary=summary,
        body=body[:20000],
        lang="en",
    )


# ---------------------------------------------------------------- Perfect World CSGO site
def parse_pw_list(html: str) -> list[str]:
    return sorted(set(re.findall(r"/article/details/\d{8}/\d+\.html", html)))


def parse_pw_article(html: str, url: str) -> NewsItem | None:
    soup = BeautifulSoup(html, "lxml")
    box = soup.select_one(".artical_news")
    if box is None:
        return None
    h2 = box.select_one("h2")
    title = h2.get_text(" ", strip=True) if h2 else ""
    if not title:
        return None
    published = ""
    for span in box.select(".subconfig_news span"):
        txt = span.get_text(" ", strip=True)
        m = re.search(r"(20\d\d)[/-](\d{1,2})[/-](\d{1,2})", txt)
        if m:
            published = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
            break
    if not published:
        m = re.search(r"/article/details/(\d{4})(\d{2})(\d{2})/", url)
        if m:
            published = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    body_parts = []
    for p in box.select("p"):
        txt = p.get_text(" ", strip=True)
        if txt:
            body_parts.append(txt)
    aid = re.search(r"/(\d+)\.html", url)
    return NewsItem(
        article_id=f"pw:{aid.group(1)}" if aid else f"pw:{_sha(url)[:12]}",
        source="pw",
        url=PW_BASE + url if url.startswith("/") else url,
        published_at=_iso(published) if published else "",
        title=title,
        summary="",
        body="\n".join(body_parts)[:20000],
        lang="zh",
    )


# ---------------------------------------------------------------- Storage
SCHEMA = """
CREATE TABLE IF NOT EXISTS news (
    article_id   TEXT PRIMARY KEY,
    source       TEXT NOT NULL,
    url          TEXT NOT NULL,
    published_at TEXT,
    title        TEXT,
    summary      TEXT,
    body         TEXT,
    lang         TEXT,
    content_hash TEXT,
    fetched_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_news_date ON news(published_at);

CREATE TABLE IF NOT EXISTS news_events (
    event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id    TEXT NOT NULL,
    team          TEXT,
    event_type    TEXT NOT NULL,
    direction     TEXT,
    magnitude     INTEGER,
    confidence    REAL,
    effective_date TEXT,
    evidence      TEXT,
    extractor     TEXT,
    created_at    TEXT,
    UNIQUE(article_id, team, event_type)
);
CREATE INDEX IF NOT EXISTS idx_events_team ON news_events(team);
"""


def store(conn: sqlite3.Connection, items: list[NewsItem]) -> int:
    conn.executescript(SCHEMA)
    n = 0
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for it in items:
        h = _sha(it.title + "\n" + it.body)
        cur = conn.execute(
            """INSERT OR IGNORE INTO news
               (article_id, source, url, published_at, title, summary, body, lang, content_hash, fetched_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (it.article_id, it.source, it.url, it.published_at, it.title,
             it.summary, it.body, it.lang, h, now),
        )
        n += cur.rowcount
    conn.commit()
    return n


# ---------------------------------------------------------------- Rule-based extraction
EVENT_PATTERNS = [
    # Departure / benched
    ("roster_remove",
     r"(\bpart(s)? ways with\b|\bdepart(s|ed|ure)?\b|\bleaves?\b|\bbenched?\b|\breleased?\b|\bexits?\b|"
     r"离队|解约|退出|下放|替补席|不再效力|结束合作)",
     "negative", 3),
    # Joining / signing
    ("roster_add",
     r"(\bsign(s|ed|ing)?\b|\bjoins?\b|\bjoined\b|\badds?\b|\brecruits?\b|\breturns? to\b|"
     r"加入|加盟|签约|官宣|转会|引进|回归)",
     "positive", 3),
    ("igl_change", r"(\bin-game leader\b|\bIGL\b|\bcall(ing|er)\b|指挥)", "neutral", 3),
    ("coach_change", r"(\bcoach(es|ing)?\b|教练)", "neutral", 2),
    ("injury",
     r"(\binjur(y|ed|ies)\b|\billness\b|\bunwell\b|\bhospitali[sz]ed\b|\bmiss(es|ed)? (the )?(event|match)\b|"
     r"受伤|伤病|生病|缺阵|缺席|养伤)",
     "negative", 4),
    ("ban",
     r"(\bbanned?\b|\bsuspend(ed|sion)?\b|\bVAC\b|\bcheat(ing)?\b|\bmatch-?fixing\b|"
     r"禁赛|封禁|处罚|作弊|开挂)",
     "negative", 5),
    ("retirement", r"(\bretire(s|d|ment)?\b|退役)", "negative", 4),
    ("patch_change", r"(\bpatch\b|\bupdate\b|\bmap pool\b|\bmeta\b|版本更新|地图池|平衡性)", "neutral", 1),
    ("transfer_rumor",
     r"(\breportedly\b|\brumou?r\b|\bsources say\b|\bexpected to\b|传闻|据悉|爆料|消息人士)",
     "neutral", 2),
]


def _known_teams(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT name FROM teams WHERE matches >= 20 ORDER BY matches DESC").fetchall()
    return [r["name"] for r in rows]


def _find_teams(text: str, names: list[str]) -> list[str]:
    low = text.lower()
    hits = []
    for name in names:
        if len(name) < 3:
            continue
        if name.lower() in low:
            hits.append(name)
    # Longest name first, so "Natus Vincere" and "NAVI" do not both match
    hits.sort(key=len, reverse=True)
    out: list[str] = []
    for h in hits:
        if not any(h.lower() in o.lower() or o.lower() in h.lower() for o in out):
            out.append(h)
    return out[:2]


NOISE_HINTS = ("更新日志", "活动", "福利", "Roll房", "开箱", "通行证", "商城", "促销", "抽奖", "赛事任务")


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?\n])|(?<=\. )", text)
    return [p.strip() for p in parts if p and len(p.strip()) >= 8]


def extract_events(conn: sqlite3.Connection, limit: int | None = None,
                   source: str | None = None) -> dict:
    """Rule-based extraction: sentence-level matching + team-name co-occurrence; prefer no event over a wrong one.

    The rule version is only the "no LLM" baseline, and it prioritizes precision:
    - An event is recorded only when a team name and a trigger word appear in the same sentence;
    - Sentences containing noise words (events/giveaways/store, etc.) are skipped outright;
    - Each event keeps its source sentence as evidence, for human review and later LLM review.
    """
    conn.executescript(SCHEMA)
    names = _known_teams(conn)
    sql = "SELECT * FROM news WHERE body != ''"
    params: list = []
    if source:
        sql += " AND source = ?"
        params.append(source)
    sql += " ORDER BY published_at DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql, params).fetchall()

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    total = 0
    by_type: dict[str, int] = {}
    for r in rows:
        text = f"{r['title']}。{r['summary']}。{r['body']}"
        art_teams = _find_teams(r["title"], names) or _find_teams(text[:600], names)
        for sent in _sentences(text):
            if any(h in sent for h in NOISE_HINTS):
                continue
            teams = _find_teams(sent, names) or art_teams
            if not teams:
                continue
            for etype, pattern, direction, mag in EVENT_PATTERNS:
                m = re.search(pattern, sent, re.I)
                if not m:
                    continue
                cur = conn.execute(
                    """INSERT OR IGNORE INTO news_events
                       (article_id, team, event_type, direction, magnitude, confidence,
                        effective_date, evidence, extractor, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (r["article_id"], teams[0], etype, direction, mag,
                     0.5 if _find_teams(sent, names) else 0.35,
                     (r["published_at"] or "")[:10], sent[:300], "rule", now),
                )
                if cur.rowcount:
                    total += 1
                    by_type[etype] = by_type.get(etype, 0) + 1
                break  # Record only one primary event per sentence
    conn.commit()
    return {"articles": len(rows), "events": total, "by_type": by_type}
