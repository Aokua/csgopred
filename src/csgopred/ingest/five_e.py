"""5EPlay scraping: the list comes from the official JSON API, article bodies come from the article page DOM.

API (fetched inside the browser, so it carries the WAF cookies automatically):
    GET /api/article?page=N&type_id=0&time=0&order_by=0
Returns data.list[]: title / dateline(unix) / jump_link / hits / comments.

There is no detail API for article bodies, so the article page must be opened and .article-main parsed.
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
    """Fetch the list page by page until the page limit is exceeded or results are older than since_ts."""
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
            # Rate limited: wait a minute and retry the same page (do not skip, or data would be missed)
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
        _t.sleep(1.8)  # The API allows roughly 40 requests/minute, so one page every 2 seconds
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
    """Open an article page and reuse the parser from browser_news.wy_article."""
    if not item.get("url"):
        return None
    return wy_article(browser, item)
