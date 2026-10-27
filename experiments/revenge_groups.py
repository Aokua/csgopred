"""Grouped tests of the revenge effect: by gap and by whether the previous winner was the favourite."""

import math
import sys
from collections import defaultdict
from datetime import date

sys.path.insert(0, "src")

from csgopred.db import connect
from csgopred.eval.backtest import load_matches
from csgopred.features.pipeline import build
from csgopred.ingest.kaggle_maps import build_lineups


def main() -> None:
    conn = connect()
    maps = {}
    for r in conn.execute("SELECT match_id, map_name, winner FROM maps ORDER BY match_id, map_order"):
        maps.setdefault(r["match_id"], []).append((r["map_name"], r["winner"]))
    rows = build(load_matches(conn), maps=maps, lineups=build_lineups(conn))

    history: dict[tuple, dict] = {}
    recs = []
    for r in rows:
        key = tuple(sorted((r["team_a"], r["team_b"])))
        d = date.fromisoformat(r["date"])
        prev = history.get(key)
        if prev:
            gap = (d - prev["date"]).days
            p_a = 1.0 / (1.0 + 10 ** (-r["features"]["elo_diff"]))
            prev_winner_is_a = prev["winner"] == r["team_a"]
            p_prev = p_a if prev_winner_is_a else 1 - p_a
            won = (r["y"] == 1) if prev_winner_is_a else (r["y"] == 0)
            recs.append({"gap": gap, "p_prev": p_prev, "won": 1 if won else 0,
                         "fav": p_prev >= 0.55, "dog": p_prev <= 0.45})
        history[key] = {"date": d, "winner": r["team_a"] if r["y"] == 1 else r["team_b"]}

    def agg(sub, label):
        if not sub:
            return
        n = len(sub)
        obs = sum(x["won"] for x in sub) / n
        exp = sum(x["p_prev"] for x in sub) / n
        se = math.sqrt(max(obs * (1 - obs), 1e-9) / n)
        print(f"{label:<22} n={n:6d}  实际={obs:.4f}  期望={exp:.4f}  "
              f"差={obs-exp:+.4f}  CI[{obs-1.96*se:+.4f},{obs+1.96*se:+.4f}]")

    print("== 全部再相遇 ==")
    agg(recs, "all")

    print("\n== 按间隔 ==")
    for lo, hi in ((0, 14), (15, 30), (31, 90), (91, 180), (181, 720)):
        agg([x for x in recs if lo <= x["gap"] <= hi], f"{lo}-{hi} 天")

    print("\n== 按上次赢家身份 ==")
    agg([x for x in recs if x["fav"]], "上次赢家是热门")
    agg([x for x in recs if x["dog"]], "上次赢家是冷门")
    agg([x for x in recs if not x["fav"] and not x["dog"]], "接近五五开")

    print("\n== 间隔 + 身份 ==")
    for lo, hi in ((0, 30), (31, 90)):
        agg([x for x in recs if lo <= x["gap"] <= hi and x["fav"]], f"{lo}-{hi} 天 / 热门赢")
        agg([x for x in recs if lo <= x["gap"] <= hi and x["dog"]], f"{lo}-{hi} 天 / 冷门赢")


if __name__ == "__main__":
    main()
