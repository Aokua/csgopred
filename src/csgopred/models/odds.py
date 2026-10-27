"""Odds conversion: fair odds, margin addition, fractional Kelly."""

from __future__ import annotations


def fair_odds(p: float) -> float:
    return 1.0 / p


def offer_odds(p: float, margin: float = 0.06) -> tuple[float, float]:
    """Add margin by the proportional method.

    q_i = p_i * (1 + margin); for a two-way market q_a + q_b = 1 + margin, and odds = 1/q.
    Note: the proportional method makes favourites and underdogs carry the same margin, while real
    markets usually take more margin on underdogs; switch to the Shin method when finer precision is
    needed (see docs/技术规划.md, section 7.2).
    """
    qa = p * (1.0 + margin)
    qb = (1.0 - p) * (1.0 + margin)
    return round(1.0 / qa, 3), round(1.0 / qb, 3)


def implied_prob(odds: float) -> float:
    return 1.0 / odds


def overround(odds_a: float, odds_b: float) -> float:
    return implied_prob(odds_a) + implied_prob(odds_b) - 1.0


def kelly_fraction(p: float, odds: float, fraction: float = 0.25) -> float:
    """Fractional Kelly; returns the suggested stake fraction (may be 0)."""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    f = (p * odds - 1.0) / b
    return max(0.0, f * fraction)
