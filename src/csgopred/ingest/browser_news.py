"""用 CDP 浏览器抓新闻（处理 JS 挑战：HLTV 的 Cloudflare、5EPlay 的阿里云 WAF）。

浏览器负责「拿到渲染后的 DOM」，解析仍用 news.py 里的纯函数，
这样离线测试和缓存复用都不依赖浏览器。
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from bs4 import BeautifulSoup

from .browser import Browser
from .news import NewsItem, parse_hltv_article

HLTV_ARCHIVE = "https://www.hltv.org/news/archive"
WY_LIST = "https://csgo.5eplay.com/article"


def hltv_items(html: str) -> list[dict]:
    """HLTV 归档页：链接 + 标题 + 日期（不用逐篇打开）。"""
    soup = BeautifulSoup(html, "lxml")
    out = []
    for a in soup.select("a.newsline.article[href^='/news/']"):
        href = a.get("href", "")
        title_el = a.select_one(".newstext")
        date_el = a.select_one(".newsrecent")
        if not (href and title_el and date_el):
            continue
        out.append(
            {
                "url": "https://www.hltv.org" + href,
                "path": href,
                "title": title_el.get_text(" ", strip=True),
                "date": date_el.get_text(strip=True)[:10],
            }
        )
    return out


def hltv_article(browser: Browser, item: dict) -> NewsItem | None:
    html = browser.goto(item["url"], wait=3.5, ready_check="h1", max_wait=25)
    art = parse_hltv_article(html, item["path"])
    if art and not art.published_at:
        art.published_at = item["date"] + "T00:00:00+00:00"
    return art


WY_TIME_RE = re.compile(r"(20\d\d-\d\d-\d\d \d\d:\d\d:\d\d)")


def wy_items(html: str) -> list[dict]:
    out, seen = [], set()
    for m in re.finditer(r'href="(?:https?://csgo\.5eplay\.com)?(/article/article-(\d+)-(\d+))"', html):
        path = m.group(1)
        if path in seen:
            continue
        seen.add(path)
        ts = int(m.group(2))
        # 5EPlay 的文章 ID 前缀是秒级时间戳
        try:
            d = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")
        except (ValueError, OSError):
            d = ""
        out.append({"url": "https://csgo.5eplay.com" + path, "path": path, "date": d})
    return out


def wy_article(browser: Browser, item: dict) -> NewsItem | None:
    html = browser.goto(item["url"], wait=4, ready_check="article-main", max_wait=30)
    soup = BeautifulSoup(html, "lxml")
    main = soup.select_one(".article-main")
    if main is None:
        return None
    title = ""
    for h in soup.find_all("h1"):
        t = h.get_text(" ", strip=True)
        if t:
            title = t
            break
    if not title and soup.title:
        title = soup.title.get_text(" ", strip=True)
    title = re.sub(r"_5EPlay.*$", "", title or "").strip()

    published = ""
    info = soup.select_one(".article-info")
    if info:
        m = WY_TIME_RE.search(info.get_text(" ", strip=True))
        if m:
            published = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc
            ).isoformat(timespec="seconds")
    if not published:
        published = item.get("date", "")

    paras = []
    for p in main.select("p"):
        txt = p.get_text(" ", strip=True)
        if len(txt) >= 10:
            paras.append(txt)
    body = "\n".join(paras)
    aid = re.search(r"/article/article-\d+-(\d+)", item["path"])
    return NewsItem(
        article_id=f"5e:{aid.group(1)}" if aid else f"5e:{abs(hash(item['path'])) % 10**10}",
        source="5e",
        url=item["url"],
        published_at=published,
        title=title,
        summary="",
        body=body[:20000],
        lang="zh",
    )


SIGNAL_RE = re.compile(
    r"(sign|join|part|leave|bench|injur|coach|ban|suspen|retir|transfer|roster|stand-?in|"
    r"replace|depart|kick|miss|out of|break|"
    r"转会|加盟|离队|签约|下放|伤|禁赛|退役|教练|指挥|阵容|复出|教练组)",
    re.I,
)
