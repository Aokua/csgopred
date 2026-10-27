"""5EPlay 抓取：列表走官方 JSON 接口，正文走文章页 DOM。

接口（浏览器内 fetch，天然带 WAF cookie）：
    GET /api/article?page=N&type_id=0&time=0&order_by=0
返回 data.list[]：title / dateline(unix) / jump_link / hits / comments。

正文没有详情接口，只能打开文章页解析 .article-main。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from .browser import Browser
from .browser_news import wy_article

API = "/api/article?page={page}&type_id=0&time=0&order_by=0"
LIST_PAGE = "https://csgo.5eplay.com/article"


def list_articles(browser: Browser, pages: int = 3, since_ts: int = 0,
                  start_page: int = 1) -> list[dict]:
    """按页拉列表，直到超过页数或早于 since_ts。"""
    import re

    import time as _t

    out: list[dict] = []
    empty_streak = 0
    for page in range(start_page, start_page + pages):
        raw = None
        for attempt in range(3):
            raw = browser.eval(
                f"fetch('{API.format(page=page)}').then(r=>r.text())", timeout=30
            )
            if raw:
                break
            _t.sleep(2.0 * (attempt + 1))
        if not raw:
            break
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            break
        if not data.get("success"):
            break
        items = data.get("data", {}).get("list", [])
        if not items:
            # 限流：等一分钟，重试同一页（不跳过，否则会漏数据）
            for _ in range(3):
                _t.sleep(60.0)
                raw2 = browser.eval(
                    f"fetch('{API.format(page=page)}').then(r=>r.text())", timeout=30
                )
                try:
                    d2 = json.loads(raw2 or "{}")
                except json.JSONDecodeError:
                    continue
                items = (d2.get("data") or {}).get("list") or []
                if items:
                    break
            if not items:
                break
        empty_streak = 0
        _t.sleep(1.8)  # 接口约 40 请求/分钟，2 秒一页
        oldest = None
        for it in items:
            ts = int(it.get("dateline") or 0)
            oldest = ts if oldest is None else min(oldest, ts)
            if since_ts and ts < since_ts:
                continue
            url = it.get("jump_link") or ""
            m = re.search(r"/article/article-(\d+)-(\d+)", url)
            if not m:
                continue
            out.append(
                {
                    "url": url,
                    "path": f"/article/article-{m.group(1)}-{m.group(2)}",
                    "title": it.get("title", ""),
                    "date": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")
                    if ts else "",
                }
            )
        if since_ts and oldest and oldest < since_ts:
            break
    return out


def fetch_body(browser: Browser, item: dict):
    """打开文章页并复用 browser_news.wy_article 的解析。"""
    if not item.get("url"):
        return None
    return wy_article(browser, item)
