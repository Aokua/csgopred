"""Elo 差 → 胜率的概率映射（逻辑回归，双参数，Newton 法拟合）。

形式：p = sigmoid(a + b * (elo_a - elo_b) / 400)

单独拟合 b 是必要的：Elo 的 400 分基准只在「棋类」标定过，
CS 的样本、赛制、地图差异会让真实斜率明显偏离 1。
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
    """返回 (a, b)。xs 为标准化后的 Elo 差，ys ∈ {0,1}。"""
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
        # 步长保护，避免早期数据把参数推飞
        step = max(abs(da), abs(db), 1.0)
        a += da / step
        b += db / step
        if abs(da) < 1e-8 and abs(db) < 1e-8:
            break
    return a, b


def predict_from_elo(a: float, b: float, elo_diff: float, scale: float = 400.0) -> float:
    return _sigmoid(a + b * (elo_diff / scale))
