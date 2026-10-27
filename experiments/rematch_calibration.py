"""再相遇的校准检验：Elo 在再相遇时是否过度自信？"""

import math
import sys
from datetime import date

sys.path.insert(0, "src")

from csgopred.db import connect
from csgopred.eval.backtest import load_matches
from csgopred.eval.rolling import split_eval
from csgopred.features.pipeline import (
    FEATURES, MAP_FEATURES, MAPSERIES_FEATURES, build,
)
from csgopred.ingest.kaggle_maps import build_lineups

SPLITS = ["2025-08-01", "2026-09-01"]


def main() -> None:
    conn = connect()
    maps = {}
    for r in conn.execute("SELECT match_id, map_name, winner FROM maps ORDER BY match_id, map_order"):
        maps.setdefault(r["match_id"], []).append((r["map_name"], r["winner"]))
    rows = build(load_matches(conn), maps=maps, lineups=build_lineups(conn))

    last = {}
    for r in rows:
        f = r["features"]
        key = tuple(sorted((r["team_a"], r["team_b"])))
        d = date.fromisoformat(r["date"])
        prev = last.get(key)
        if prev and (d - prev).days <= 365:
            f["rematch"] = 1.0
        else:
            f["rematch"] = 0.0
        f["rematch_x_elo"] = f["rematch"] * f["elo_diff"]
        last[key] = d

    # 描述：全样本 vs 再相遇，按热门/冷门看实际胜率
    for label, cond in (("全部比赛", lambda r: True),
                        ("再相遇(<=365天)", lambda r: r["features"]["rematch"] > 0.5)):
        sub = [r for r in rows if cond(r)]
        for tag, pick in (("热门", lambda p: p >= 0.55), ("冷门", lambda p: p <= 0.45)):
            ps = [1.0 / (1.0 + 10 ** (-r["features"]["elo_diff"])) for r in sub]
            ys = []
            for r, p in zip(sub, ps):
                fav_is_a = p >= 0.5
                ys.append(r["y"] if fav_is_a else 1 - r["y"])
            idx = [i for i, p in enumerate(ps) if pick(p)]
            if not idx:
                continue
            obs = sum(ys[i] for i in idx) / len(idx)
            exp = sum(ps[i] if ps[i] >= 0.5 else 1 - ps[i] for i in idx) / len(idx)
            se = math.sqrt(max(obs * (1 - obs), 1e-9) / len(idx))
            print(f"{label:<14}{tag} n={len(idx):6d} 实际={obs:.4f} 期望={exp:.4f} "
                  f"差={obs-exp:+.4f} CI[{obs-1.96*se:+.4f},{obs+1.96*se:+.4f}]")

    mapall = MAP_FEATURES + MAPSERIES_FEATURES
    configs = {
        "base+map": FEATURES + mapall,
        "+rematch": FEATURES + mapall + ["rematch"],
        "+rematch_x_elo": FEATURES + mapall + ["rematch", "rematch_x_elo"],
    }
    for name, feats in configs.items():
        vals = []
        for sp in SPLITS:
            m = split_eval(rows, sp, "lr", feats=feats)["metrics"]
            vals.append(m["log_loss"])
        print(f"{name:<16} " + "  ".join(f"{v:.5f}" for v in vals))


if __name__ == "__main__":
    main()
