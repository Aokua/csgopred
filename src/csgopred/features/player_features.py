"""Player-level features: lineup strength.

Data comes from Kaggle's cs2_all_tiers_games.csv (per-map, per-player ADR / KAST / KD differentials).
"Pre-match" rolling means represent player form; averaging over the team's five players gives lineup strength.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date

PATH = "data/raw/kaggle_pro/cs2_all_tiers_games.csv"
HALF_LIFE = 120.0
WINDOW = 365
MIN_MAPS = 3


def load_players(path: str = PATH) -> dict[str, dict]:
    """match_id -> {date, team1, team2, p1:[(name,adr,kast,kd)], p2:[...]}"""
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mid = row["match_id"]
            rec = out.get(mid)
            if rec is None:
                rec = {"date": row["datetime"][:10], "team1": row["team1"],
                       "team2": row["team2"], "p1": [], "p2": []}
                out[mid] = rec
            for side, key in (("team1", "p1"), ("team2", "p2")):
                players = []
                for i in range(1, 6):
                    name = row.get(f"{side}_player{i}") or ""
                    if not name:
                        continue
                    try:
                        adr = float(row.get(f"{side}_player{i}_adr") or 0)
                        kast = float(row.get(f"{side}_player{i}_kast") or 0)
                        kd = float(row.get(f"{side}_player{i}_kddiff") or 0)
                    except ValueError:
                        continue
                    players.append((name, adr, kast, kd))
                if players:
                    rec[key] = players
    return out


class PlayerIndex:
    def __init__(self) -> None:
        self.rec: dict[str, list[tuple[date, float, float, float]]] = defaultdict(list)

    def rating(self, name: str, d: date) -> tuple[float, float, float, int]:
        items = self.rec.get(name)
        if not items:
            return 0.0, 0.0, 0.0, 0
        sw = sadr = skast = skd = 0.0
        n = 0
        for dd, adr, kast, kd in reversed(items):
            days = (d - dd).days
            if days < 0:
                continue
            if days > WINDOW:
                break
            w = 0.5 ** (days / HALF_LIFE)
            sw += w
            sadr += w * adr
            skast += w * kast
            skd += w * kd
            n += 1
        if sw < 1e-9:
            return 0.0, 0.0, 0.0, 0
        return sadr / sw, skast / sw, skd / sw, n

    def lineup(self, players: list[tuple[str, float, float, float]], d: date
               ) -> tuple[float, float, float, int]:
        """Return the pre-match lineup averages (adr, kast, kd), using only players with >= MIN_MAPS records."""
        vals = [self.rating(p[0], d) for p in players]
        good = [v for v in vals if v[3] >= MIN_MAPS]
        if not good:
            return 0.0, 0.0, 0.0, 0
        return (sum(v[0] for v in good) / len(good),
                sum(v[1] for v in good) / len(good),
                sum(v[2] for v in good) / len(good),
                len(good))

    def add(self, d: date, players: list[tuple[str, float, float, float]]) -> None:
        for name, adr, kast, kd in players:
            self.rec[name].append((d, adr, kast, kd))
