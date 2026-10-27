"""Information-source contribution decomposition experiment.

Outputs:
- Nested incremental (adding one group at a time) log-loss curve
- Leave-one-out (dropping one group from the full model) marginal contributions
Used for Section 5.5 of the paper and Figure 3.
"""

import json
import pathlib
import sys
import time

sys.path.insert(0, "src")

from csgopred.db import connect
from csgopred.eval.backtest import load_matches
from csgopred.eval.rolling import split_eval
from csgopred.features.pipeline import (
    FEATURES, MAP_FEATURES, MAPSERIES_FEATURES, PLAYER_FEATURES, NEWS_FEATURES,
    EXTRA_FEATURES, build,
)
from csgopred.ingest.kaggle_maps import build_lineups

GROUPS = {
    "队伍强度": ["elo_diff", "elo_a", "elo_b", "log_games_a", "log_games_b"],
    "近期状态": ["form10_a", "form10_b", "form30_a", "form30_b"],
    "赛程休息": ["rest_a", "rest_b", "density_a", "density_b"],
    "交手记录": ["h2h"],
    "赛制赛事": ["is_bo1", "is_bo5", "tier_major", "tier_top", "lan", "year"],
    "不确定度": EXTRA_FEATURES,
    "地图信息": MAP_FEATURES + MAPSERIES_FEATURES,
    "选手阵容": PLAYER_FEATURES,
}

SPLITS = ["2025-08-01", "2026-09-01"]


def main() -> None:
    t0 = time.time()
    conn = connect()
    matches = load_matches(conn)
    maps = {}
    for r in conn.execute("SELECT match_id, map_name, winner FROM maps ORDER BY match_id, map_order"):
        maps.setdefault(r["match_id"], []).append((r["map_name"], r["winner"]))
    lineups = build_lineups(conn)
    print(f"matches={len(matches)} maps={len(maps)} lineups={len(lineups)}")
    rows = build(matches, maps=maps, lineups=lineups)
    print(f"built in {time.time()-t0:.0f}s")

    order = ["队伍强度", "近期状态", "赛程休息", "交手记录", "赛制赛事", "不确定度", "地图信息", "选手阵容"]
    full = [f for g in order for f in GROUPS[g]]

    result = {"nested": [], "leave_one_out": [], "splits": SPLITS, "elo": {}, "full": {}}

    # Baseline: Elo mapping
    for sp in SPLITS:
        result["elo"][sp] = split_eval(rows, sp, "elo")["metrics"]

    # Nested incremental
    current: list[str] = []
    for g in order:
        current = current + GROUPS[g]
        entry = {"group": g, "k": len(current), "splits": {}}
        for sp in SPLITS:
            m = split_eval(rows, sp, "lr", feats=current)["metrics"]
            entry["splits"][sp] = m
        result["nested"].append(entry)
        print(f"nested +{g:<6} k={len(current):2d} ll={entry['splits'][SPLITS[0]]['log_loss']:.5f}/"
              f"{entry['splits'][SPLITS[1]]['log_loss']:.5f}  [{time.time()-t0:.0f}s]")

    # Full model and leave-one-out
    for sp in SPLITS:
        result["full"][sp] = split_eval(rows, sp, "lr", feats=full)["metrics"]
    for g in order:
        feats = [f for f in full if f not in GROUPS[g]]
        entry = {"group": g, "k": len(feats), "splits": {}}
        for sp in SPLITS:
            entry["splits"][sp] = split_eval(rows, sp, "lr", feats=feats)["metrics"]
        result["leave_one_out"].append(entry)
        print(f"without {g:<6} ll={entry['splits'][SPLITS[0]]['log_loss']:.5f}/"
              f"{entry['splits'][SPLITS[1]]['log_loss']:.5f}  [{time.time()-t0:.0f}s]")

    out = pathlib.Path("paper/data")
    out.mkdir(parents=True, exist_ok=True)
    (out / "decomp.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", out / "decomp.json", f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
