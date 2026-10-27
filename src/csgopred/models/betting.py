"""下注建议：凯利公式、期望增长、回撤模拟。

原则：
- 只用「模型概率 + 用户实际能拿到的赔率」算，不用我们自己的报价倒推；
- 默认给 1/4 凯利（模型有误差，满凯利波动过大）；
- 用 95% 区间下界做「保守凯利」，避免高估优势；
- 没有正期望时明确返回 0，不粉饰。
"""

from __future__ import annotations

import math
import random

MAX_FRACTION = 0.10  # 单场最多压 10% 本金，防止模型极端输出


def kelly(p: float, odds: float) -> float:
    """满凯利比例（占本金）。无正期望返回 0。"""
    if odds <= 1.0 or not (0 < p < 1):
        return 0.0
    b = odds - 1.0
    f = (p * odds - 1.0) / b
    return max(0.0, min(f, 1.0))


def expected_growth(p: float, odds: float, f: float) -> float:
    """单次下注的期望对数增长率（每单位本金）。"""
    if f <= 0:
        return 0.0
    win = 1.0 + f * (odds - 1.0)
    lose = 1.0 - f
    if win <= 0 or lose <= 0:
        return float("-inf")
    return p * math.log(win) + (1 - p) * math.log(lose)


def simulate(p: float, odds: float, f: float, bets: int = 100,
             paths: int = 2000, seed: int = 7) -> dict:
    """蒙特卡洛：固定比例下注 N 场，返回回撤与破产概率。"""
    if f <= 0:
        return {"median_max_dd": 0.0, "p95_max_dd": 0.0, "ruin_prob": 0.0, "median_final": 1.0}
    rng = random.Random(seed)
    dds, finals, ruined = [], [], 0
    for _ in range(paths):
        eq = 1.0
        peak = 1.0
        max_dd = 0.0
        ruin = False
        for _ in range(bets):
            if rng.random() < p:
                eq *= 1.0 + f * (odds - 1.0)
            else:
                eq *= 1.0 - f
            peak = max(peak, eq)
            max_dd = max(max_dd, 1.0 - eq / peak)
            if eq < 0.5:
                ruin = True
        dds.append(max_dd)
        finals.append(eq)
        if ruin:
            ruined += 1
    dds.sort()
    finals.sort()
    return {
        "median_max_dd": round(dds[len(dds) // 2], 4),
        "p95_max_dd": round(dds[int(len(dds) * 0.95) - 1], 4),
        "ruin_prob": round(ruined / paths, 4),
        "median_final": round(finals[len(finals) // 2], 4),
    }


def advice(p: float, odds: float, bankroll: float, fraction: float = 0.25,
           p_low: float | None = None, bets_horizon: int = 100) -> dict:
    """给出一场比赛的下注建议。

    p        模型胜率
    odds     用户实际能拿到的赔率（十进制）
    bankroll 全部本金
    fraction 凯利折扣（0.25 = 1/4 凯利）
    p_low    95% 区间下界（用于保守凯利）
    """
    full = kelly(p, odds)
    edge = p * odds - 1.0
    fair = 1.0 / p if p > 0 else float("inf")
    min_odds = fair

    if edge <= 0 or full <= 0:
        return {
            "verdict": "不下注",
            "reason": f"模型胜率 {p:.1%} 对应的公平赔率是 {fair:.3f}，当前赔率 {odds:.2f} 没有正期望（EV {edge:+.2%}）",
            "edge": round(edge, 4),
            "kelly_full": 0.0,
            "kelly_frac": 0.0,
            "stake": 0.0,
            "stake_pct": 0.0,
            "fair_odds": round(fair, 3),
            "min_odds": round(min_odds, 3),
            "ev": 0.0,
        }

    frac_bet = min(full * fraction, MAX_FRACTION)
    conservative = None
    if p_low is not None and p_low > 0:
        conservative = min(kelly(p_low, odds) * fraction, MAX_FRACTION)

    stake = round(bankroll * frac_bet, 2)
    sim = simulate(p, odds, frac_bet, bets=bets_horizon)
    growth = expected_growth(p, odds, frac_bet)

    return {
        "verdict": "可以下注" if frac_bet >= 0.005 else "优势太小，建议跳过",
        "edge": round(edge, 4),
        "ev": round(bankroll * edge * frac_bet, 2),
        "fair_odds": round(fair, 3),
        "min_odds": round(min_odds, 3),
        "kelly_full": round(full, 4),
        "kelly_full_stake": round(bankroll * min(full, MAX_FRACTION), 2),
        "kelly_frac": round(frac_bet, 4),
        "kelly_frac_label": f"{fraction:g} 凯利",
        "stake": stake,
        "stake_pct": round(frac_bet * 100, 2),
        "conservative_stake": round(bankroll * conservative, 2) if conservative is not None else None,
        "growth_per_bet": round(growth * 100, 3),
        "expected_profit": round(bankroll * frac_bet * edge, 2),
        "risk": sim,
        "notes": [
            f"公平赔率 {fair:.3f}，低于这个价没有优势",
            *(["⚠ 优势超过 15%，先确认赔率是否过期、是否同一盘口"] if edge > 0.15 else []),
            f"满凯利会压 {min(full, MAX_FRACTION) * 100:.1f}% 本金，波动很大",
            "模型有误差，建议用 1/4 凯利并参考保守金额",
        ],
    }
