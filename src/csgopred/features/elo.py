"""地图级 / 队伍级 Elo。

设计要点：
1. 不活跃衰减在「读取时」计算，并写回基准值，避免衰减被下一次更新覆盖掉。
2. K 值按赛事级别启发式分档；净胜图数给小幅加成。
3. 所有状态都在内存中按时间顺序推进——回测和线上预测走同一套代码，
   这是保证「回测数字可信」的前提。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ..config import (
    ELO_BASE,
    ELO_K_DEFAULT,
    ELO_K_MID,
    ELO_K_TOP,
    ELO_SCALE,
    INACTIVE_GRACE_DAYS,
    INACTIVE_HALF_LIFE_DAYS,
    TIER1_KEYWORDS,
)


# 超参数（2026-10-27 在 2015–2026 全量数据上网格搜索，两个时间切分一致更优）
ELO_K_SCALE = 1.6       # K 值整体放大：CS 强弱变化比棋类快
ELO_MARGIN_SCALE = 3.0  # 净胜分加成放大：2-0 比 2-1 的信息量更大


def expected_score(rating_a: float, rating_b: float, scale: float = ELO_SCALE) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / scale))


def decay_toward_mean(
    rating: float,
    days_inactive: int,
    grace: int = INACTIVE_GRACE_DAYS,
    half_life: float = INACTIVE_HALF_LIFE_DAYS,
    mean: float = ELO_BASE,
) -> float:
    if days_inactive <= grace:
        return rating
    years = (days_inactive - grace) / half_life
    factor = 0.5 ** years
    return mean + (rating - mean) * factor


def k_factor(event: str, fmt: str) -> float:
    e = (event or "").lower()
    if "major" in e:
        base = ELO_K_TOP
    elif any(k in e for k in TIER1_KEYWORDS):
        base = ELO_K_MID
    elif fmt == "bo1":
        base = ELO_K_DEFAULT * 0.75  # bo1 噪声大，更新更保守
    else:
        base = ELO_K_DEFAULT
    return base * ELO_K_SCALE


def margin_multiplier(score_a: int, score_b: int, fmt: str) -> float:
    diff = abs(score_a - score_b)
    if fmt in ("bo3", "bo5"):
        base = min(1.0 + 0.15 * diff, 1.4)
    else:
        base = min(1.0 + 0.15 * (diff / 8.0), 1.4)  # bo1 用回合差比例
    return 1.0 + ELO_MARGIN_SCALE * (base - 1.0)


@dataclass
class EloModel:
    scale: float = ELO_SCALE
    mean: float = ELO_BASE
    ratings: dict[str, float] = field(default_factory=dict)
    last_played: dict[str, date] = field(default_factory=dict)
    games: dict[str, int] = field(default_factory=dict)

    # ---- 读取 ----
    def rating(self, team: str, asof: date | None = None) -> float:
        base = self.ratings.get(team, self.mean)
        last = self.last_played.get(team)
        if asof is None or last is None:
            return base
        return decay_toward_mean(base, (asof - last).days, mean=self.mean)

    def n_games(self, team: str) -> int:
        return self.games.get(team, 0)

    # ---- 单场推进 ----
    def step(self, match: dict, asof: date | None = None) -> dict:
        a, b = match["team_a"], match["team_b"]
        when = asof or date.fromisoformat(match["start_date"])

        ra = self.rating(a, when)
        rb = self.rating(b, when)
        ea = expected_score(ra, rb, self.scale)

        sa = 1.0 if match["score_a"] > match["score_b"] else 0.0
        k = k_factor(match.get("event", ""), match.get("format", "bo3"))
        mov = margin_multiplier(match["score_a"], match["score_b"], match.get("format", "bo3"))

        # 衰减后的值作为新的基准，再叠加更新
        self.ratings[a] = ra + k * mov * (sa - ea)
        self.ratings[b] = rb + k * mov * ((1.0 - sa) - (1.0 - ea))
        self.last_played[a] = when
        self.last_played[b] = when
        self.games[a] = self.games.get(a, 0) + 1
        self.games[b] = self.games.get(b, 0) + 1

        return {
            "team_a": a,
            "team_b": b,
            "rating_a": ra,
            "rating_b": rb,
            "expected_a": ea,
            "k": k,
            "mov": mov,
            "y": sa,
        }

    # ---- 序列化 ----
    def snapshot(self, asof: date | None = None) -> list[dict]:
        rows = []
        for team, r in self.ratings.items():
            rows.append(
                {
                    "team": team,
                    "elo": round(self.rating(team, asof), 2),
                    "raw_elo": round(r, 2),
                    "games": self.games.get(team, 0),
                    "last_played": self.last_played.get(team).isoformat()
                    if self.last_played.get(team)
                    else None,
                }
            )
        rows.sort(key=lambda x: -x["elo"])
        return rows
