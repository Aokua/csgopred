"""Map-level features: each team's recent win rate on each map.

Data comes from the maps table (2015-2020 HLTV + 2023+ Kaggle).
Only map results from before the match date are used, decayed by half-life.
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
        """Return (win rate - 0.5, sample count) over a 365-day window with a 120-day half-life."""
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
