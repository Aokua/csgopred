"""验证「去掉赛制赛事组」的稳健性，并给出推荐特征集。"""

import json
import pathlib
import sys
import time

sys.path.insert(0, "src")

from csgopred.db import connect
from csgopred.eval.backtest import load_matches
from csgopred.eval.rolling import split_eval
from csgopred.features.pipeline import (
    FEATURES, MAP_FEATURES, MAPSERIES_FEATURES, PLAYER_FEATURES, EXTRA_FEATURES, build,
)
from csgopred.ingest.kaggle_maps import build_lineups

S = ["elo_diff", "elo_a", "elo_b", "log_games_a", "log_games_b"]
FORM = ["form10_a", "form10_b", "form30_a", "form30_b"]
SCHED = ["rest_a", "rest_b", "density_a", "density_b"]
H2H = ["h2h"]
FMT = ["is_bo1", "is_bo5", "tier_major", "tier_top", "lan", "year"]
MAP = MAP_FEATURES + MAPSERIES_FEATURES
PL = PLAYER_FEATURES

CONFIGS = {
    "base20": FEATURES,
    "no_format": S + FORM + SCHED + H2H + EXTRA_FEATURES + MAP + PL,
    "core": S + H2H + MAP,
    "core+form": S + FORM + H2H + MAP,
    "recommended": S + FORM + SCHED + H2H + EXTRA_FEATURES + MAP,
}

SPLITS = ["2023-01-01", "2025-08-01", "2026-09-01"]


def main() -> None:
    t0 = time.time()
    conn = connect()
    matches = load_matches(conn)
    maps = {}
    for r in conn.execute("SELECT match_id, map_name, winner FROM maps ORDER BY match_id, map_order"):
        maps.setdefault(r["match_id"], []).append((r["map_name"], r["winner"]))
    rows = build(matches, maps=maps, lineups=build_lineups(conn))
    print("built", round(time.time() - t0), "s")

    out = {"configs": {}, "splits": SPLITS}
    for name, feats in CONFIGS.items():
        entry = {"k": len(feats), "splits": {}}
        for sp in SPLITS:
            entry["splits"][sp] = split_eval(rows, sp, "lr", feats=feats)["metrics"]
        out["configs"][name] = entry
        vals = "/".join(f"{entry['splits'][sp]['log_loss']:.5f}" for sp in SPLITS)
        print(f"{name:<12} k={len(feats):2d}  {vals}")

    pathlib.Path("paper/data").mkdir(parents=True, exist_ok=True)
    pathlib.Path("paper/data/feature_sets.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("total", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
