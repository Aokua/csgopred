"""LLM event extraction (OpenAI-compatible API).

Without an API key only a dry run is performed: the prompt and article that would be sent are printed,
so the template can be checked first.
Configuration (environment variables):
  OPENAI_API_KEY   required
  OPENAI_BASE_URL  default https://api.openai.com/v1
  OPENAI_MODEL     default gpt-4o-mini
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timezone

import requests

from .news import SCHEMA, _known_teams, _find_teams

SYSTEM = """你是 CS2 电竞赛事的信息抽取器。输入一篇新闻，输出严格符合 schema 的 JSON，不输出任何解释。

规则：
1. 只抽取文中明确陈述的事实；推测、传闻、记者观点用 transfer_rumor，其余一律不抽。
2. event_type 只能取给定枚举值。
3. evidence_quote 必须是原文逐字片段（<=200 字符）；找不到证据就不输出该事件。
4. 与竞技实力无关的内容（商业合作、周边、活动、更新日志的 UI 改动）返回 {"events": []}。
5. team 必须是文章明确提到的战队名；不确定填 null。
6. effective_date 是事件生效日，不确定填 null。
"""

SCHEMA_HINT = {
    "events": [
        {
            "team": "string|null",
            "player": "string|null",
            "event_type": "roster_add|roster_remove|igl_change|coach_change|injury|illness|"
                          "ban|retirement|transfer_rumor|patch_change|other",
            "direction": "positive|negative|neutral|null",
            "magnitude": "1-5",
            "confidence": "0-1",
            "effective_date": "YYYY-MM-DD|null",
            "evidence_quote": "string",
        }
    ]
}


def build_prompt(title: str, body: str) -> str:
    return (
        "schema: " + json.dumps(SCHEMA_HINT, ensure_ascii=False) + "\n\n"
        "文章标题：" + title + "\n\n文章正文：\n" + body[:6000]
    )


def _post(base: str, key: str, model: str, prompt: str, timeout: int = 60) -> dict:
    r = requests.post(
        base.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt}],
        },
        timeout=timeout,
    )
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    return json.loads(content)


SIGNAL_TITLE = re.compile(
    "(sign|join|part|leave|bench|injur|illness|coach|ban|suspen|retir|transfer|roster|"
    "stand-?in|replace|depart|kick|miss|out of|\u8f6c\u4f1a|\u52a0\u76df|\u79bb\u961f|\u7b7e\u7ea6|"
    "\u4e0b\u653e|\u4f24|\u7981\u8d5b|\u9000\u5f79|\u6559\u7ec3|\u6307\u6325|\u9635\u5bb9|"
    "\u590d\u51fa|\u79bb\u5f00|\u52a0\u5165)",
    re.I,
)


def extract_with_llm(conn: sqlite3.Connection, limit: int = 10, dry_run: bool = False,
                     source: str | None = None, signal_only: bool = False,
                     titles_only: bool = False, require_team: bool = False) -> dict:
    conn.executescript(SCHEMA)
    key = os.environ.get("OPENAI_API_KEY", "")
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    sql = """SELECT * FROM news n WHERE body != ''
             AND NOT EXISTS (SELECT 1 FROM news_events e
                             WHERE e.article_id = n.article_id AND e.extractor = 'llm')"""
    params: list = []
    if source:
        sql += " AND n.source = ?"
        params.append(source)
    sql += " ORDER BY published_at DESC"
    rows = conn.execute(sql, params).fetchall()
    if signal_only:
        rows = [r for r in rows if SIGNAL_TITLE.search(r["title"] or "")]
    if require_team:
        names = [n.lower() for n in _known_teams(conn) if len(n) >= 3]
        rows = [r for r in rows if any(n in (r["title"] or "").lower() for n in names)]
    rows = rows[:limit]

    if dry_run:
        out = []
        for r in rows[:3]:
            out.append({"title": r["title"], "prompt_chars": len(build_prompt(r["title"], r["body"]))})
        return {"dry_run": True, "pending": len(rows), "samples": out,
                "configured": bool(key)}

    if not key:
        return {"error": "未配置 OPENAI_API_KEY，可用 --dry-run 先检查 prompt"}

    names = _known_teams(conn)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    total = 0
    by_type: dict[str, int] = {}
    errors = 0
    for r in rows:
        try:
            data = _post(base, key, model, build_prompt(r["title"], r["body"]))
        except Exception:
            errors += 1
            continue
        for ev in data.get("events", []) or []:
            quote = (ev.get("evidence_quote") or "").strip()
            if not quote or quote not in (r["body"] or "") and quote not in (r["title"] or ""):
                continue  # The evidence must be findable in the source text
            team = ev.get("team")
            if not team:
                found = _find_teams(quote, names)
                team = found[0] if found else None
            etype = ev.get("event_type") or "other"
            cur = conn.execute(
                """INSERT OR IGNORE INTO news_events
                   (article_id, team, event_type, direction, magnitude, confidence,
                    effective_date, evidence, extractor, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (r["article_id"], team, etype, ev.get("direction"),
                 int(ev.get("magnitude") or 2), float(ev.get("confidence") or 0.6),
                 ev.get("effective_date"), quote[:300], "llm", now),
            )
            if cur.rowcount:
                total += 1
                by_type[etype] = by_type.get(etype, 0) + 1
        conn.commit()
    return {"articles": len(rows), "events": total, "by_type": by_type, "errors": errors}
