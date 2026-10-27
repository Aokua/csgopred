"""特征工程：按时间顺序推进，任何时点只用「此前」的信息。

设计成 FeatureBuilder 而不是一次性 pandas 变换，是为了让训练和线上预测
走完全相同的代码路径——回测里怎么算，predict 就怎么算。
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from datetime import date

from .elo import EloModel
from .news_features import features_for as _news_features
from .map_features import MapIndex
from .map_elo import MapModel
from .map_model import MapLevelModel
from .player_features import PlayerIndex

MAJOR_KEYWORDS = ("major",)
TIER1_KEYWORDS = (
    "iem", "blast", "esl pro league", "esl one", "pgl", "katowice", "cologne",
    "dreamhack masters", "esl", "blast premier", "iem cologne", "iem katowice",
)
LAN_KEYWORDS = TIER1_KEYWORDS + ("lan", "arena", "major")

FEATURES = [
    "elo_diff",
    "elo_a",
    "elo_b",
    "log_games_a",
    "log_games_b",
    "form10_a",
    "form10_b",
    "form30_a",
    "form30_b",
    "rest_a",
    "rest_b",
    "density_a",
    "density_b",
    "h2h",
    "is_bo1",
    "is_bo5",
    "tier_major",
    "tier_top",
    "lan",
    "year",
]

NEWS_FEATURES = ["news_net_a", "news_net_b", "news_net_diff", "news_cnt_a", "news_cnt_b"]
EXTRA_FEATURES = ["unc_a", "unc_b", "h2h_n", "elo_bo3"]
MAP_FEATURES = ["map_wr_a", "map_wr_b", "map_wr_diff", "map_n_a", "map_n_b"]
MAPSERIES_FEATURES = ["map_series_a", "map_series_diff"]
PLAYER_FEATURES = ["lineup_adr_a", "lineup_adr_b", "lineup_kast_a", "lineup_kast_b",
                   "lineup_kd_a", "lineup_kd_b", "lineup_adr_diff", "lineup_kd_diff"]
ALL_FEATURES = FEATURES + NEWS_FEATURES


def _tier(event: str) -> tuple[int, int]:
    e = (event or "").lower()
    major = 1 if any(k in e for k in MAJOR_KEYWORDS) else 0
    top = 1 if (major or any(k in e for k in TIER1_KEYWORDS)) else 0
    return major, top


def _is_lan(event: str) -> int:
    e = (event or "").lower()
    return 1 if any(k in e for k in LAN_KEYWORDS) else 0


class FeatureBuilder:
    def __init__(self, news: dict | None = None) -> None:
        self.news = news or {}
        self.elo = EloModel()
        self.recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=30))
        self.last: dict[str, date] = {}
        self.h2h: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])

    # ---------------- 状态查询 ----------------
    def _form(self, team: str, n: int) -> float:
        items = list(self.recent.get(team, ()))[-n:]
        if len(items) < 3:
            return 0.0
        return sum(y - e for _, y, e in items) / len(items)

    def _rest(self, team: str, d: date) -> float:
        last = self.last.get(team)
        if last is None:
            return 1.0
        return min((d - last).days, 60) / 60.0

    def _density(self, team: str, d: date) -> float:
        items = self.recent.get(team, ())
        n = sum(1 for dd, _, _ in items if (d - dd).days <= 14)
        return n / 10.0

    def _h2h(self, a: str, b: str) -> float:
        key = tuple(sorted((a, b)))
        w0, w1 = self.h2h.get(key, (0, 0))
        wins_a = w0 if key[0] == a else w1
        n = w0 + w1
        return (wins_a + 2.0) / (n + 4.0) - 0.5

    # ---------------- 单场特征 ----------------
    def features(self, team_a: str, team_b: str, d: date, fmt: str, event: str) -> dict:
        ra = self.elo.rating(team_a, d)
        rb = self.elo.rating(team_b, d)
        ga = self.elo.n_games(team_a)
        gb = self.elo.n_games(team_b)
        major, top = _tier(event)
        net_a, cnt_a = _news_features(team_a, d, self.news)
        net_b, cnt_b = _news_features(team_b, d, self.news)
        key = tuple(sorted((team_a, team_b)))
        h2h_total = sum(self.h2h.get(key, (0, 0)))
        return {
            "elo_diff": (ra - rb) / 400.0,
            "elo_a": (ra - 1500.0) / 400.0,
            "elo_b": (rb - 1500.0) / 400.0,
            "log_games_a": math.log1p(ga) / 5.0,
            "log_games_b": math.log1p(gb) / 5.0,
            "form10_a": self._form(team_a, 10),
            "form10_b": self._form(team_b, 10),
            "form30_a": self._form(team_a, 30),
            "form30_b": self._form(team_b, 30),
            "rest_a": self._rest(team_a, d),
            "rest_b": self._rest(team_b, d),
            "density_a": self._density(team_a, d),
            "density_b": self._density(team_b, d),
            "h2h": self._h2h(team_a, team_b),
            "is_bo1": 1.0 if fmt == "bo1" else 0.0,
            "is_bo5": 1.0 if fmt == "bo5" else 0.0,
            "tier_major": float(major),
            "tier_top": float(top),
            "lan": float(_is_lan(event)),
            "year": (d.year - 2020) / 6.0,
            "news_net_a": net_a,
            "news_net_b": net_b,
            "news_net_diff": net_a - net_b,
            "news_cnt_a": cnt_a,
            "news_cnt_b": cnt_b,
            "unc_a": 1.0 / math.sqrt(1.0 + ga / 5.0),
            "unc_b": 1.0 / math.sqrt(1.0 + gb / 5.0),
            "h2h_n": math.log1p(h2h_total) / 3.0,
            "elo_bo3": (ra - rb) / 400.0 * (1.0 if fmt == "bo3" else 0.0),
        }

    # ---------------- 状态更新 ----------------
    def update(self, match: dict, y: float, exp: float) -> None:
        a, b = match["team_a"], match["team_b"]
        d = date.fromisoformat(match["start_date"])
        self.recent[a].append((d, y, exp))
        self.recent[b].append((d, 1.0 - y, 1.0 - exp))
        self.last[a] = d
        self.last[b] = d
        key = tuple(sorted((a, b)))
        if key[0] == a:
            self.h2h[key][0 if y == 1.0 else 1] += 1
        else:
            self.h2h[key][1 if y == 1.0 else 0] += 1
        self.elo.step(match)


def replay(matches: list[dict], news: dict | None = None) -> FeatureBuilder:
    """只推进状态、不算特征（给 predict 用）。"""
    fb = FeatureBuilder(news)
    for m in matches:
        d = date.fromisoformat(m["start_date"])
        ra = fb.elo.rating(m["team_a"], d)
        rb = fb.elo.rating(m["team_b"], d)
        exp = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        y = 1.0 if m["score_a"] > m["score_b"] else 0.0
        fb.update(m, y, exp)
    return fb


def build(matches: list[dict], news: dict | None = None,
          maps: dict[str, list[tuple[str, str]]] | None = None,
          lineups: dict[str, dict] | None = None) -> list[dict]:
    """把赛果序列变成特征行（每行都只含赛前信息）。

    maps: match_id -> [(map_name, winner_side)]，用于地图特征。
    """
    fb = FeatureBuilder(news)
    mi = MapIndex()
    mm = MapModel()
    mlm = MapLevelModel()
    pi = PlayerIndex()
    rows = []
    for m in matches:
        d = date.fromisoformat(m["start_date"])
        feat = fb.features(m["team_a"], m["team_b"], d, m.get("format", "bo3"), m.get("event", ""))
        if maps is not None:
            wr_a, n_a = mi.wr(m["team_a"], d)
            wr_b, n_b = mi.wr(m["team_b"], d)
            feat["map_wr_a"] = wr_a
            feat["map_wr_b"] = wr_b
            feat["map_wr_diff"] = wr_a - wr_b
            feat["map_n_a"] = min(n_a, 200) / 200.0
            feat["map_n_b"] = min(n_b, 200) / 200.0
            mi.add(d, m["team_a"], m["team_b"], maps.get(m["match_id"], []))
            ea = fb.elo.rating(m["team_a"], d)
            eb = fb.elo.rating(m["team_b"], d)
            sp = mm.series_prob(m["team_a"], m["team_b"], m.get("format", "bo3"), ea, eb,
                                prob_fn=mlm.prob)
            feat["map_series_a"] = sp
            feat["map_series_diff"] = sp - 0.5
            for mp, side in maps.get(m["match_id"], []):
                if mp in ("Ancient", "Anubis", "Dust2", "Inferno", "Mirage", "Nuke", "Train"):
                    ra = mm.map_rating(m["team_a"], mp, ea)
                    rb = mm.map_rating(m["team_b"], mp, eb)
                    mlm.add((ra - rb) / 400.0, (ea - eb) / 400.0, 1 if side == "a" else 0)
            mm.update_match(m["team_a"], m["team_b"], d, maps.get(m["match_id"], []), ea, eb)
        if lineups is not None:
            lu = lineups.get(m["match_id"])
            if lu and lu["a"] and lu["b"]:
                adr_a, kast_a, kd_a, n_a2 = pi.lineup(lu["a"], d)
                adr_b, kast_b, kd_b, n_b2 = pi.lineup(lu["b"], d)
                feat["lineup_adr_a"] = adr_a / 100.0
                feat["lineup_adr_b"] = adr_b / 100.0
                feat["lineup_kast_a"] = kast_a / 100.0
                feat["lineup_kast_b"] = kast_b / 100.0
                feat["lineup_kd_a"] = kd_a / 10.0
                feat["lineup_kd_b"] = kd_b / 10.0
                feat["lineup_adr_diff"] = (adr_a - adr_b) / 100.0
                feat["lineup_kd_diff"] = (kd_a - kd_b) / 10.0
            for k in PLAYER_FEATURES:
                feat.setdefault(k, 0.0)
        exp = fb.elo.rating(m["team_a"], d)
        y = 1.0 if m["score_a"] > m["score_b"] else 0.0
        rows.append(
            {
                "date": m["start_date"],
                "match_id": m["match_id"],
                "team_a": m["team_a"],
                "team_b": m["team_b"],
                "format": m.get("format", "bo3"),
                "event": m.get("event", ""),
                "y": int(y),
                "features": feat,
            }
        )
        if lineups is not None:
            lu = lineups.get(m["match_id"])
            if lu and lu["a"]:
                pi.add(d, lu["a"])
            if lu and lu["b"]:
                pi.add(d, lu["b"])
        e = 1.0 / (1.0 + 10.0 ** ((fb.elo.rating(m["team_b"], d) - fb.elo.rating(m["team_a"], d)) / 400.0))
        fb.update(m, y, e)
    return rows


def map_features_asof(matches: list[dict], maps: dict[str, list[tuple[str, str]]],
                      when, team_a: str, team_b: str, fmt: str, elo) -> dict:
    """给单场预测补地图特征：只用 when 之前的地图结果。"""
    mi = MapIndex()
    mm = MapModel()
    for m in matches:
        d = date.fromisoformat(m["start_date"])
        if d >= when:
            break
        ea = elo.rating(m["team_a"], d)
        eb = elo.rating(m["team_b"], d)
        mm.update_match(m["team_a"], m["team_b"], d, maps.get(m["match_id"], []), ea, eb)
        mi.add(d, m["team_a"], m["team_b"], maps.get(m["match_id"], []))

    wr_a, n_a = mi.wr(team_a, when)
    wr_b, n_b = mi.wr(team_b, when)
    ea = elo.rating(team_a, when)
    eb = elo.rating(team_b, when)
    sp = mm.series_prob(team_a, team_b, fmt, ea, eb)
    return {
        "map_wr_a": wr_a,
        "map_wr_b": wr_b,
        "map_wr_diff": wr_a - wr_b,
        "map_n_a": min(n_a, 200) / 200.0,
        "map_n_b": min(n_b, 200) / 200.0,
        "map_series_a": sp,
        "map_series_diff": sp - 0.5,
    }


def map_breakdown_asof(matches: list[dict], maps: dict[str, list[tuple[str, str]]],
                       when, team_a: str, team_b: str, fmt: str, elo) -> list[dict]:
    """返回 veto 模拟选出的地图及每张图的胜率（只用 when 之前的数据）。"""
    mm = MapModel()
    for m in matches:
        d = date.fromisoformat(m["start_date"])
        if d >= when:
            break
        ea = elo.rating(m["team_a"], d)
        eb = elo.rating(m["team_b"], d)
        mm.update_match(m["team_a"], m["team_b"], d, maps.get(m["match_id"], []), ea, eb)
    ea = elo.rating(team_a, when)
    eb = elo.rating(team_b, when)
    chosen = mm.veto(team_a, team_b, fmt, ea, eb)
    out = []
    for mp in chosen:
        ra = mm.map_rating(team_a, mp, ea)
        rb = mm.map_rating(team_b, mp, eb)
        out.append({"map": mp, "p_a": round(1.0 / (1.0 + 10 ** ((rb - ra) / 400.0)), 4)})
    return out
