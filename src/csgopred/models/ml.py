"""M2 model layer: logistic regression / gradient boosting + Platt calibration.

Depends on numpy + scikit-learn (bundled with Anaconda; no lightgbm, HistGradientBoosting is equivalent).
"""

from __future__ import annotations

import pickle
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from ..features.pipeline import FEATURES


def matrix(rows: list[dict], feats: list[str] | None = None) -> np.ndarray:
    feats = feats or FEATURES
    return np.array([[r["features"][k] for k in feats] for r in rows], dtype=np.float64)


def labels(rows: list[dict]) -> np.ndarray:
    return np.array([r["y"] for r in rows], dtype=np.float64)


def time_weights(rows: list[dict], half_life_days: float = 1500.0) -> np.ndarray:
    if not rows:
        return np.array([])
    end = max(date.fromisoformat(r["date"]) for r in rows)
    w = []
    for r in rows:
        days = (end - date.fromisoformat(r["date"])).days
        w.append(0.5 ** (days / half_life_days))
    return np.array(w, dtype=np.float64)


class Platt:
    """One-dimensional logistic calibration: p' = sigmoid(a * logit(p) + b)."""

    def __init__(self) -> None:
        self.a = 1.0
        self.b = 0.0

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Platt":
        z = self._logit(p)
        a, b = 1.0, 0.0
        for _ in range(20):
            q = 1.0 / (1.0 + np.exp(-(a * z + b)))
            r = y - q
            w = q * (1 - q)
            g = np.array([np.sum(r * z), np.sum(r)])
            h = np.array([[np.sum(w * z * z), np.sum(w * z)],
                          [np.sum(w * z), np.sum(w)]])
            try:
                step = np.linalg.solve(h, g)
            except np.linalg.LinAlgError:
                break
            a += step[0]
            b += step[1]
            if np.max(np.abs(step)) < 1e-8:
                break
        self.a, self.b = float(a), float(b)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.a * self._logit(p) + self.b)))


class Model:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.est = None
        self.platt: Platt | None = None
        self.feats: list[str] = list(FEATURES)

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, calib_frac: float = 0.0) -> "Model":
        if self.kind == "lr":
            self.est = LogisticRegression(C=0.5, max_iter=3000, solver="lbfgs")
            self.est.fit(X, y, sample_weight=w)
        elif self.kind == "hgb":
            n = len(X)
            cut = int(n * (1 - calib_frac)) if calib_frac > 0 else n
            self.est = HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.05, max_depth=6,
                l2_regularization=1.0, early_stopping=False, random_state=0,
            )
            self.est.fit(X[:cut], y[:cut], sample_weight=w[:cut])
            if calib_frac > 0 and n - cut > 200:
                raw = self.est.predict_proba(X[cut:])[:, 1]
                self.platt = Platt().fit(raw, y[cut:])
        else:
            raise ValueError(f"unknown kind {self.kind}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.est.predict_proba(X)[:, 1]
        if self.platt is not None:
            p = self.platt.predict(p)
        return np.clip(p, 1e-6, 1 - 1e-6)


def save(model: Model, path: str | Path, meta: dict) -> None:
    payload = {"kind": model.kind, "est": model.est, "platt": model.platt,
               "features": model.feats, "meta": meta}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load(path: str | Path) -> tuple[Model, dict]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    m = Model(payload["kind"])
    m.est = payload["est"]
    m.platt = payload["platt"]
    m.feats = payload.get("features", list(FEATURES))
    return m, payload.get("meta", {})


def fit_elo_baseline(train_rows: list[dict], w: np.ndarray) -> tuple[float, float]:
    """M1 baseline: one-dimensional logistic regression on elo_diff only."""
    X = np.array([[r["features"]["elo_diff"]] for r in train_rows])
    y = labels(train_rows)
    lr = LogisticRegression(C=1.0, max_iter=2000)
    lr.fit(X, y, sample_weight=w)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def apply_elo_baseline(rows: list[dict], a: float, b: float) -> np.ndarray:
    x = np.array([r["features"]["elo_diff"] for r in rows])
    return 1.0 / (1.0 + np.exp(-(a * x + b)))
