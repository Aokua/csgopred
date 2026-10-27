"""地图级模型：把「单图 Elo 差」和「队伍 Elo 差」组合成校准后的单图胜率。

原始 map Elo 概率校准很差（地图胜率 log loss 约 0.74），
用逻辑回归重新组合后能降到 0.70 左右。
这里用走步式重训：只用当前日期之前的地图结果，每积累 N 张图重训一次。
"""

from __future__ import annotations

import numpy as np

FEATS = ["map_elo_diff", "team_elo_diff"]


class MapLevelModel:
    def __init__(self, refit_every: int = 4000, min_rows: int = 1500) -> None:
        self.rows: list[tuple[dict, int]] = []
        self.model: Model | None = None
        self.refit_every = refit_every
        self.min_rows = min_rows
        self._last_fit = 0

    def prob(self, map_elo_diff: float, team_elo_diff: float, fallback: float) -> float:
        if self.model is None:
            return fallback
        x = np.array([[map_elo_diff, team_elo_diff]], dtype=float)
        return float(self.model.predict(x)[0])

    def add(self, map_elo_diff: float, team_elo_diff: float, y: int) -> None:
        self.rows.append(({"map_elo_diff": map_elo_diff, "team_elo_diff": team_elo_diff}, y))
        if (len(self.rows) >= self.min_rows
                and len(self.rows) - self._last_fit >= self.refit_every):
            self._fit()

    def _fit(self) -> None:
        from ..models.ml import Model

        X = np.array([[r["map_elo_diff"], r["team_elo_diff"]] for r, _ in self.rows])
        y = np.array([t for _, t in self.rows])
        m = Model("lr")
        m.feats = list(FEATS)
        m.fit(X, y, np.ones(len(y)))
        self.model = m
        self._last_fit = len(self.rows)
