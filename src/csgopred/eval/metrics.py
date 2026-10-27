"""Evaluation metrics: log loss / Brier / accuracy / calibration."""

from __future__ import annotations

import math

EPS = 1e-6


def clip(p: float) -> float:
    return min(max(p, EPS), 1.0 - EPS)


def log_loss(ps: list[float], ys: list[int]) -> float:
    return -sum(
        y * math.log(clip(p)) + (1 - y) * math.log(1 - clip(p)) for p, y in zip(ps, ys)
    ) / len(ps)


def brier(ps: list[float], ys: list[int]) -> float:
    return sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ps)


def accuracy(ps: list[float], ys: list[int]) -> float:
    return sum((1 if p >= 0.5 else 0) == y for p, y in zip(ps, ys)) / len(ps)


def reliability_table(ps: list[float], ys: list[int], bins: int = 10) -> list[dict]:
    rows = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        idx = [j for j, p in enumerate(ps) if (lo <= p < hi) or (i == bins - 1 and p == 1.0)]
        if not idx:
            continue
        conf = sum(ps[j] for j in idx) / len(idx)
        acc = sum(ys[j] for j in idx) / len(idx)
        rows.append(
            {
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": len(idx),
                "avg_p": round(conf, 4),
                "win_rate": round(acc, 4),
                "gap": round(acc - conf, 4),
            }
        )
    return rows


def ece(ps: list[float], ys: list[int], bins: int = 10) -> float:
    rows = reliability_table(ps, ys, bins)
    n = len(ps)
    return sum(r["n"] / n * abs(r["gap"]) for r in rows)


def max_calibration_gap(ps: list[float], ys: list[int], bins: int = 10) -> float:
    rows = reliability_table(ps, ys, bins)
    return max((abs(r["gap"]) for r in rows), default=0.0)


def summarize(ps: list[float], ys: list[int], name: str = "model") -> dict:
    return {
        "name": name,
        "n": len(ps),
        "log_loss": round(log_loss(ps, ys), 5),
        "brier": round(brier(ps, ys), 5),
        "accuracy": round(accuracy(ps, ys), 4),
        "ece": round(ece(ps, ys), 4),
        "max_gap": round(max_calibration_gap(ps, ys), 4),
    }


def bootstrap_ci(
    ps: list[float],
    ys: list[int],
    metric: str = "log_loss",
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    import random

    fn = {"log_loss": log_loss, "brier": brier, "accuracy": accuracy}[metric]
    rng = random.Random(seed)
    n = len(ps)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(fn([ps[i] for i in idx], [ys[i] for i in idx]))
    vals.sort()
    lo = vals[int(alpha / 2 * n_boot)]
    hi = vals[int((1 - alpha / 2) * n_boot) - 1]
    return round(lo, 5), round(hi, 5)
