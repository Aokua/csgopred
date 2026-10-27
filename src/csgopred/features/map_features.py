"""地图级特征：队伍在各张图上的近期胜率。

数据来自 maps 表（2015–2020 HLTV + 2023+ Kaggle）。
只使用比赛日之前的地图结果，按半衰期衰减。
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date

HALF_LIFE = 120.0
WINDOW = 365


class MapIndex:
    def __init__(self) -> None:
        self.rec: dict[str, list[tuple[date, int]]] = defaultdict(list)

    def wr(self, team: str, d: date) -> tuple[float, int]:
        """返回 (胜率-0.5, 样本数)，窗口 365 天，半衰期 120 天。"""
        items = self.rec.get(team)
        if not items:
            return 0.0, 0
        sw = swin = 0.0
        n = 0
        for dd, won in reversed(items):
            days = (d - dd).days
            if days < 0:
                continue
            if days > WINDOW:
                break
            w = 0.5 ** (days / HALF_LIFE)
            sw += w
            swin += w * won
            n += 1
        if sw < 1e-9:
            return 0.0, 0
        return swin / sw - 0.5, n

    def add(self, d: date, team_a: str, team_b: str, maps: list[tuple[str, str]]) -> None:
        for _name, side in maps:
            self.rec[team_a].append((d, 1 if side == "a" else 0))
            self.rec[team_b].append((d, 0 if side == "a" else 1))
