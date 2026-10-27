"""赔率换算：公平赔率、加水、分数凯利。"""

from __future__ import annotations


def fair_odds(p: float) -> float:
    return 1.0 / p


def offer_odds(p: float, margin: float = 0.06) -> tuple[float, float]:
    """按比例法加水。

    q_i = p_i * (1 + margin)，两路盘 q_a + q_b = 1 + margin，赔率 = 1/q。
    注意：比例法会让热门和冷门承担相同水位，真实盘口通常对冷门抽更多水，
    需要更精细时再换 Shin 方法（见 docs/技术规划.md §7.2）。
    """
    qa = p * (1.0 + margin)
    qb = (1.0 - p) * (1.0 + margin)
    return round(1.0 / qa, 3), round(1.0 / qb, 3)


def implied_prob(odds: float) -> float:
    return 1.0 / odds


def overround(odds_a: float, odds_b: float) -> float:
    return implied_prob(odds_a) + implied_prob(odds_b) - 1.0


def kelly_fraction(p: float, odds: float, fraction: float = 0.25) -> float:
    """分数凯利；返回建议下注比例（可为 0）。"""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    f = (p * odds - 1.0) / b
    return max(0.0, f * fraction)
