"""M4：赔率与市场评估。

包含：
- 去水（proportional / shin）
- CLV（相对收盘线的赔率优势）
- 凯利下注与 ROI 模拟
- 模型 vs 市场的 log loss / Brier 对比

赔率数据来源由调用方提供（CSV 或数据库），本模块不绑定抓取方式。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def implied(odds: float) -> float:
    return 1.0 / odds if odds and odds > 1.0 else 0.0


def devig_proportional(odds_a: float, odds_b: float) -> tuple[float, float]:
    """按比例去水：q_i = (1/o_i) / Σ(1/o_j)。"""
    ia, ib = implied(odds_a), implied(odds_b)
    s = ia + ib
    if s <= 0:
        return 0.5, 0.5
    return ia / s, ib / s


def devig_shin(odds_a: float, odds_b: float, iters: int = 50) -> tuple[float, float]:
    """Shin 去水（两路盘），考虑内幕交易比例 z。"""
    ia, ib = implied(odds_a), implied(odds_b)
    total = ia + ib
    if total <= 1.0 or ia <= 0 or ib <= 0:
        return devig_proportional(odds_a, odds_b)
    z = 0.0
    for _ in range(iters):
        pa = (math.sqrt(z * z + 4 * (1 - z) * ia * ia / total) - z) / (2 * (1 - z))
        pb = (math.sqrt(z * z + 4 * (1 - z) * ib * ib / total) - z) / (2 * (1 - z))
        f = pa + pb - 1.0
        if abs(f) < 1e-10:
            break
        # 简单二分更新 z
        z = min(max(z + 0.01 * (1 if f > 0 else -1), 0.0), 0.5)
    pa = (math.sqrt(z * z + 4 * (1 - z) * ia * ia / total) - z) / (2 * (1 - z))
    pb = (math.sqrt(z * z + 4 * (1 - z) * ib * ib / total) - z) / (2 * (1 - z))
    s = pa + pb
    return pa / s, pb / s


@dataclass
class Bet:
    p_model: float
    odds: float
    won: bool
    stake: float = 1.0

    @property
    def edge(self) -> float:
        return self.p_model * self.odds - 1.0

    @property
    def pnl(self) -> float:
        return (self.odds - 1.0) * self.stake if self.won else -self.stake


def kelly(p: float, odds: float, fraction: float = 0.25) -> float:
    b = odds - 1.0
    if b <= 0:
        return 0.0
    f = (p * odds - 1.0) / b
    return max(0.0, f * fraction)


def clv(model_p: float, closing_odds: float) -> float:
    """CLV 的一种定义：log(模型公平赔率 / 收盘赔率)。正值表示拿到比收盘更好的价格。"""
    fair = 1.0 / max(min(model_p, 1 - 1e-6), 1e-6)
    if closing_odds <= 1.0:
        return 0.0
    return math.log(fair / closing_odds)


def roi(bets: list[Bet], threshold: float = 0.0, kelly_fraction: float = 0.25) -> dict:
    selected = []
    for b in bets:
        if b.edge <= threshold:
            continue
        stake = kelly(b.p_model, b.odds, kelly_fraction)
        if stake <= 0:
            continue
        selected.append((stake, b))
    if not selected:
        return {"n": 0, "staked": 0.0, "pnl": 0.0, "roi": 0.0}
    staked = sum(s for s, _ in selected)
    pnl = sum((b.odds - 1.0) * s if b.won else -s for s, b in selected)
    return {
        "n": len(selected),
        "staked": round(staked, 4),
        "pnl": round(pnl, 4),
        "roi": round(pnl / staked, 4) if staked else 0.0,
        "avg_edge": round(sum(b.edge for _, b in selected) / len(selected), 4),
    }


def log_loss(ps: list[float], ys: list[int], eps: float = 1e-6) -> float:
    return -sum(
        y * math.log(max(min(p, 1 - eps), eps)) + (1 - y) * math.log(max(min(1 - p, 1 - eps), eps))
        for p, y in zip(ps, ys)
    ) / len(ps)


def brier(ps: list[float], ys: list[int]) -> float:
    return sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ps)
