"""新闻事件 → 特征。

思路：事件是点过程，按半衰期衰减成「截至某日的信号强度」。
只进模型的特征，不直接改概率；权重由模型学。

特征：
- news_net_a / news_net_b：带方向的衰减和（正=利好，负=利空）
- news_cnt_a / news_cnt_b：事件条数（衰减后）
- news_net_diff：A 减 B
"""

from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from datetime import date

SIGN = {
    "roster_add": 1.0,
    "roster_remove": -1.0,
    "igl_change": -0.3,
    "coach_change": 0.0,
    "injury": -1.5,
    "illness": -1.0,
    "ban": -2.5,
    "retirement": -1.0,
    "transfer_rumor": 0.2,
    "patch_change": 0.0,
    "other": 0.0,
}

HALF_LIFE_DAYS = 30.0
WINDOW_DAYS = 120


def load_events(conn: sqlite3.Connection, extractor: str = "llm") -> dict[str, list[tuple[date, float]]]:
    """team -> [(date, signed_value)]，按日期排序。"""
    rows = conn.execute(
        """SELECT team, event_type, magnitude, confidence, published_at
           FROM news_events e JOIN news n ON n.article_id = e.article_id
           WHERE e.extractor = ? AND e.team IS NOT NULL AND n.published_at != ''""",
        (extractor,),
    ).fetchall()
    out: dict[str, list[tuple[date, float]]] = defaultdict(list)
    for r in rows:
        try:
            d = date.fromisoformat((r["published_at"] or "")[:10])
        except ValueError:
            continue
        sign = SIGN.get(r["event_type"], 0.0)
        if sign == 0:
            continue
        val = sign * float(r["magnitude"] or 2) * float(r["confidence"] or 0.6)
        out[r["team"]].append((d, val))
    for k in out:
        out[k].sort()
    return out


def features_for(team: str, asof: date, events: dict[str, list[tuple[date, float]]]) -> tuple[float, float]:
    """返回 (净信号, 强度计数)，只统计 asof 之前 WINDOW_DAYS 天内的事件。"""
    items = events.get(team)
    if not items:
        return 0.0, 0.0
    net = cnt = 0.0
    for d, val in reversed(items):
        days = (asof - d).days
        if days < 0:
            continue
        if days > WINDOW_DAYS:
            break
        w = 0.5 ** (days / HALF_LIFE_DAYS)
        net += val * w
        cnt += abs(val) * w
    return round(net, 4), round(cnt, 4)
