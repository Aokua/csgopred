"""Validation experiment for two changes:

1) Model map counter-picks separately by series length (BO1/BO3/BO5) instead of only the current format;
2) Analyst effect (does a team's win rate improve when it meets the same opponent again soon after losing?).

Output: descriptive statistics + feature-group ablation.
"""

import json
import math
import pathlib
import sys
from collections import defaultdict
from datetime import date

sys.path.insert(0, "src")

from csgopred.db import connect
from csgopred.eval.backtest import load_matches
from csgopred.eval.metrics import log_loss
from csgopred.eval.rolling import split_eval
from csgopred.features.map_elo import MapModel
from csgopred.features.pipeline import (
    FEATURES, MAP_FEATURES, MAPSERIES_FEATURES, EXTRA_FEATURES, build,
)
from csgopred.ingest.kaggle_maps import build_lineups

SPLITS = ["2025-08-01", "2026-09-01"]


def add_format_features(rows, maps):
    """Compute series probabilities under BO1/BO3/BO5 for every match."""
    mm = MapModel()
    for r in rows:
        f = r["features"]
        ra = 1500.0 + 400.0 * f["elo_a"]
        rb = 1500.0 + 400.0 * f["elo_b"]
        d = date.fromisoformat(r["date"])
        for fmt in ("bo1", "bo3", "bo5"):
            p = mm.series_prob(r["team_a"], r["team_b"], fmt, ra, rb)
            f[f"ms_{fmt}"] = p
            f[f"ms_{fmt}_diff"] = p - 0.5
        mm.update_match(r["team_a"], r["team_b"], d, maps.get(r["match_id"], []), ra, rb)


def add_revenge_features(rows):
    """Most recent meeting between the same pair of teams: gap, previous result, win streak."""
    last: dict[tuple, dict] = {}
    for r in rows:
        f = r["features"]
        key = tuple(sorted((r["team_a"], r["team_b"])))
        d = date.fromisoformat(r["date"])
        prev = last.get(key)
        if prev is None:
            f["rev_gap"] = 1.0
            f["rev_lost_a"] = 0.0
            f["rev_lost_b"] = 0.0
            f["rev_lost_a_recent"] = 0.0
            f["rev_streak_a"] = 0.0
        else:
            gap = (d - prev["date"]).days
            f["rev_gap"] = min(gap, 720) / 720.0
            a_lost = 1.0 if prev["winner"] == r["team_b"] else 0.0
            b_lost = 1.0 if prev["winner"] == r["team_a"] else 0.0
            f["rev_lost_a"] = a_lost
            f["rev_lost_b"] = b_lost
            f["rev_lost_a_recent"] = a_lost if gap <= 90 else 0.0
            # Win streak: consecutive wins by the same side (positive = A is streaking)
            prev_streak = prev.get("streak", 0)
            won = r["y"] == 1
            if prev["winner"] == r["team_a"]:
                streak = prev_streak + 1 if prev_streak > 0 else 1
            else:
                streak = prev_streak - 1 if prev_streak < 0 else -1
            f["rev_streak_a"] = max(min(streak, 3), -3) / 3.0
        winner = r["team_a"] if r["y"] == 1 else r["team_b"]
        last[key] = {"date": d, "winner": winner,
                     "streak": f.get("rev_streak_a", 0.0) * 3}


def rematch_stats(rows):
    """On a rematch, is the previous winner more likely to win again?"""
    total = repeat = 0
    exp_sum = 0.0
    gaps = []
    for r in rows:
        if not r["features"].get("rev_gap", 1.0) < 1.0:
            continue
        # Recover the previous winner from rev_lost_*
        if r["features"]["rev_lost_a"] > 0.5:
            prev_winner_is_a = False
        elif r["features"]["rev_lost_b"] > 0.5:
            prev_winner_is_a = True
        else:
            continue
        p_a = 1.0 / (1.0 + 10 ** (-r["features"]["elo_diff"]))
        p_prev = p_a if prev_winner_is_a else 1.0 - p_a
        won = r["y"] == 1 if prev_winner_is_a else r["y"] == 0
        total += 1
        repeat += 1 if won else 0
        exp_sum += p_prev
        gaps.append(r["features"]["rev_gap"] * 720)
    if not total:
        return {}
    obs = repeat / total
    exp = exp_sum / total
    se = math.sqrt(obs * (1 - obs) / total)
    return {
        "n": total,
        "observed_repeat": round(obs, 4),
        "expected_repeat": round(exp, 4),
        "diff": round(obs - exp, 4),
        "ci95": [round(obs - 1.96 * se, 4), round(obs + 1.96 * se, 4)],
        "median_gap_days": sorted(gaps)[len(gaps) // 2],
    }


def main() -> None:
    conn = connect()
    matches = load_matches(conn)
    maps = {}
    for r in conn.execute("SELECT match_id, map_name, winner FROM maps ORDER BY match_id, map_order"):
        maps.setdefault(r["match_id"], []).append((r["map_name"], r["winner"]))
    rows = build(matches, maps=maps, lineups=build_lineups(conn))
    print("rows", len(rows))

    add_format_features(rows, maps)
    add_revenge_features(rows)
    print("rematch stats:", json.dumps(rematch_stats(rows), ensure_ascii=False))

    base = FEATURES
    mapall = MAP_FEATURES + MAPSERIES_FEATURES
    fmt_feats = [f"ms_{x}" for x in ("bo1", "bo3", "bo5")] + \
                [f"ms_{x}_diff" for x in ("bo1", "bo3", "bo5")]
    rev_feats = ["rev_gap", "rev_lost_a", "rev_lost_b", "rev_lost_a_recent"]

    configs = {
        "base20": base,
        "+map": base + mapall,
        "+map+fmt": base + mapall + fmt_feats,
        "+fmt_only": base + fmt_feats,
        "+map+revenge": base + mapall + rev_feats,
        "+map+fmt+rev": base + mapall + fmt_feats + rev_feats,
    }
    out = {}
    for name, feats in configs.items():
        line = {"k": len(feats), "splits": {}}
        for sp in SPLITS:
            m = split_eval(rows, sp, "lr", feats=feats)["metrics"]
            line["splits"][sp] = m
        out[name] = line
        print(f"{name:<14} k={len(feats):2d}  " +
              "  ".join(f"{line['splits'][sp]['log_loss']:.5f}" for sp in SPLITS))

    # Value of map features by series format
    for fmt_key in ("is_bo1", "is_bo5"):
        sub = [r for r in rows if r["features"].get(fmt_key, 0) > 0.5]
        print(f"\n{fmt_key} 子集 n={len(sub)}")
        for name in ("base20", "+map"):
            feats = configs[name]
            for sp in SPLITS:
                train = [r for r in sub if r["date"] < sp]
                test = [r for r in sub if r["date"] >= sp]
                if not train or not test:
                    continue
                from csgopred.models.ml import Model, labels, matrix, time_weights
                mdl = Model("lr")
                mdl.feats = list(feats)
                mdl.fit(matrix(train, feats), labels(train), time_weights(train))
                p = list(mdl.predict(matrix(test, feats)))
                print(f"  {name:<8} {sp} n={len(test)} ll={log_loss(p, labels(test)):.5f}")

    pathlib.Path("paper/data").mkdir(parents=True, exist_ok=True)
    pathlib.Path("paper/data/format_revenge.json").write_text(
        json.dumps({"configs": out, "rematch": rematch_stats(rows)}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("saved")


if __name__ == "__main__":
    main()
