"""Map-level Elo + veto simulation + BO series probabilities.

Approach:
1. Each team keeps a separate rating on each map;
2. Simulate ban/pick from relative strength (ban weak maps first, pick strong maps first);
3. Compute the series win probability from the selected map set using per-map win rates.

Features use only map results from before the match date, to avoid leakage.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date

K_MAP = 24.0
K_SERIES = 32.0
PRIOR_GAMES = 5.0

POOL = ["Ancient", "Anubis", "Dust2", "Inferno", "Mirage", "Nuke", "Train"]


def _expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


class MapModel:
    def __init__(self) -> None:
        self.rating: dict[tuple[str, str], float] = {}
        self.games: dict[tuple[str, str], int] = defaultdict(int)
        self.team_map: dict[str, list[tuple[str, float, int]]] = defaultdict(list)

    # ---------- Per-map ratings ----------
    def map_rating(self, team: str, mp: str, team_elo: float) -> float:
        key = (team, mp)
        r = self.rating.get(key)
        n = self.games.get(key, 0)
        if r is None or n < PRIOR_GAMES:
            w = n / PRIOR_GAMES
            return w * (r if r is not None else team_elo) + (1 - w) * team_elo
        return r

    def _set(self, team: str, mp: str, r: float) -> None:
        key = (team, mp)
        self.rating[key] = r
        self.games[key] += 1
        self.team_map[team].append((mp, r, self.games[key]))

    def update_map(self, a: str, b: str, mp: str, a_won: bool,
                   elo_a: float, elo_b: float) -> None:
        ra = self.map_rating(a, mp, elo_a)
        rb = self.map_rating(b, mp, elo_b)
        e = _expected(ra, rb)
        s = 1.0 if a_won else 0.0
        self._set(a, mp, ra + K_MAP * (s - e))
        self._set(b, mp, rb + K_MAP * ((1 - s) - (1 - e)))

    # ---------- veto ----------
    def _pref(self, team: str, elo: float) -> dict[str, float]:
        vals = {mp: self.map_rating(team, mp, elo) for mp in POOL}
        avg = sum(vals.values()) / len(vals)
        return {mp: v - avg for mp, v in vals.items()}

    def veto(self, a: str, b: str, fmt: str, elo_a: float, elo_b: float) -> list[str]:
        """Simulate ban/pick by relative strength and return the maps that will actually be played (in order)."""
        pa, pb = self._pref(a, elo_a), self._pref(b, elo_b)
        remaining = list(POOL)
        pick_order: list[str] = []
        n_maps = {"bo1": 1, "bo3": 3, "bo5": 5}.get(fmt, 3)

        def ban(t: dict[str, float]) -> None:
            if len(remaining) <= n_maps:
                return
            worst = min(remaining, key=lambda m: t[m])
            remaining.remove(worst)

        def pick(t: dict[str, float]) -> None:
            if len(remaining) <= n_maps:
                return
            best = max(remaining, key=lambda m: t[m])
            remaining.remove(best)
            pick_order.append(best)

        if n_maps == 1:
            for i in range(6 - 1):
                ban(pa if i % 2 == 0 else pb)
            return remaining[:1]

        ban(pa); ban(pb); pick(pa); pick(pb); ban(pb); ban(pa)
        # Fill in the remaining maps in pick order
        return (pick_order + remaining)[:n_maps]

    # ---------- Series ----------
    def series_prob(self, a: str, b: str, fmt: str, elo_a: float, elo_b: float,
                    prob_fn=None) -> float:
        maps = self.veto(a, b, fmt, elo_a, elo_b)
        ps = []
        for mp in maps:
            ra = self.map_rating(a, mp, elo_a)
            rb = self.map_rating(b, mp, elo_b)
            raw = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
            ps.append(prob_fn((ra - rb) / 400.0, (elo_a - elo_b) / 400.0, raw)
                      if prob_fn else raw)
        need = {"bo1": 1, "bo3": 2, "bo5": 3}.get(fmt, 2)
        # dp[(wa, wb)] = probability
        dp = {(0, 0): 1.0}
        for p in ps:
            nd: dict[tuple[int, int], float] = defaultdict(float)
            for (wa, wb), pr in dp.items():
                if wa >= need or wb >= need:
                    nd[(wa, wb)] += pr
                    continue
                nd[(wa + 1, wb)] += pr * p
                nd[(wa, wb + 1)] += pr * (1 - p)
            dp = nd
        return sum(pr for (wa, wb), pr in dp.items() if wa >= need)

    # ---------- Advance ----------
    def update_match(self, a: str, b: str, d: date,
                     maps: list[tuple[str, str]], elo_a: float, elo_b: float) -> None:
        for mp, side in maps:
            if mp in POOL:
                self.update_map(a, b, mp, side == "a", elo_a, elo_b)
