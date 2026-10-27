"""Probability mapping from the Elo difference to win probability (logistic regression, two parameters, fitted with Newton's method).

Form: p = sigmoid(a + b * (elo_a - elo_b) / 400)

Fitting b separately is necessary: Elo's 400-point scale was only calibrated for chess, and
CS's samples, formats, and map differences make the true slope deviate clearly from 1.
"""

from __future__ import annotations

import math


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def fit_logistic(
    xs: list[float],
    ys: list[int],
    l2: float = 1e-4,
    iters: int = 25,
) -> tuple[float, float]:
    """Return (a, b). xs is the normalized Elo difference and ys is in {0, 1}."""
    a, b = 0.0, 1.0
    n = len(xs)
    if n == 0:
        return a, b
    for _ in range(iters):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for x, y in zip(xs, ys):
            p = _sigmoid(a + b * x)
            w = p * (1.0 - p)
            r = y - p
            g0 += r
            g1 += r * x
            h00 += w
            h01 += w * x
            h11 += w * x * x
        g0 -= l2 * a
        g1 -= l2 * b
        h00 += l2
        h11 += l2
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            break
        da = (h11 * g0 - h01 * g1) / det
        db = (h00 * g1 - h01 * g0) / det
        # Step-size guard, so early data does not send the parameters flying
        step = max(abs(da), abs(db), 1.0)
        a += da / step
        b += db / step
        if abs(da) < 1e-8 and abs(db) < 1e-8:
            break
    return a, b


def predict_from_elo(a: float, b: float, elo_diff: float, scale: float = 400.0) -> float:
    return _sigmoid(a + b * (elo_diff / scale))
