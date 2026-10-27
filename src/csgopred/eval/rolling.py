"""M2 evaluation: model comparison across time-based splits.

Three tiers are compared:
- elo:     one-dimensional logistic regression on the Elo difference only (M1 baseline)
- lr:      all features + L2 logistic regression
- hgb:     all features + HistGradientBoosting (+ Platt calibration)

The test set always follows the training set, and training samples are weighted by time decay
(540-day half-life).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..features.pipeline import FEATURES
from ..models.ml import (
    Model, apply_elo_baseline, fit_elo_baseline, labels, matrix, time_weights,
)
from .metrics import summarize


def split_eval(rows: list[dict], test_from: str, kind: str = "lr",
               half_life: float = 1500.0, calib_frac: float = 0.15,
               feats: list[str] | None = None) -> dict:
    train = [r for r in rows if r["date"] < test_from]
    test = [r for r in rows if r["date"] >= test_from]
    if len(train) < 2000 or len(test) < 500:
        raise ValueError(f"样本不足：train={len(train)} test={len(test)}")

    w = time_weights(train, half_life)
    model = None
    if kind == "elo":
        a, b = fit_elo_baseline(train, w)
        p = apply_elo_baseline(test, a, b)
    else:
        model = Model(kind)
        model.feats = list(feats or FEATURES)
        model.fit(matrix(train, model.feats), labels(train), w, calib_frac=calib_frac)
        p = model.predict(matrix(test, model.feats))

    y = labels(test)
    seg = defaultdict(lambda: {"p": [], "y": []})
    for i, r in enumerate(test):
        for key in (r["format"], r["date"][:4]):
            seg[str(key)]["p"].append(float(p[i]))
            seg[str(key)]["y"].append(int(y[i]))

    return {
        "kind": kind,
        "test_from": test_from,
        "train_n": len(train),
        "test_n": len(test),
        "metrics": summarize([float(x) for x in p], [int(x) for x in y], kind),
        "segments": {k: summarize(v["p"], v["y"], k) for k, v in sorted(seg.items())},
        "p": p,
        "y": y,
        "rows": test,
        "model": model,
    }


def ablation(rows: list[dict], test_from: str, feats: list[str] | None = None) -> dict[str, dict]:
    return {k: split_eval(rows, test_from, k, feats=feats) for k in ("elo", "lr", "hgb")}


def feature_importance(model: Model) -> list[tuple[str, float]]:
    if model.kind != "lr":
        return []
    coef = model.est.coef_[0]
    return sorted(zip(FEATURES, (float(c) for c in coef)), key=lambda x: -abs(x[1]))
