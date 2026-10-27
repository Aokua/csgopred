"""Generate English-labelled figures for the JSDSS submission."""

import json
import pathlib
import sys

sys.path.insert(0, "src")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from csgopred.db import connect
from csgopred.eval.backtest import load_matches
from csgopred.eval.metrics import reliability_table
from csgopred.eval.rolling import split_eval
from csgopred.features.pipeline import (
    FEATURES, MAP_FEATURES, MAPSERIES_FEATURES, EXTRA_FEATURES, build,
)
from csgopred.ingest.kaggle_maps import build_lineups

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

OUT = pathlib.Path("paper/images_en")
OUT.mkdir(parents=True, exist_ok=True)

C1, C2, C3 = "#2f6f9f", "#d9822b", "#4a9e6f"

GROUPS_EN = {
    "队伍强度": "Team strength",
    "近期状态": "Recent form",
    "赛程休息": "Schedule / rest",
    "交手记录": "Head-to-head",
    "赛制赛事": "Format & tier",
    "不确定度": "Uncertainty",
    "地图信息": "Map information",
    "选手阵容": "Lineup",
}


def fig_architecture():
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.75)
    ax.axis("off")
    layers = [
        ("Data layer", ["Multi-source collection\nHLTV / Kaggle",
                        "Cleaning & normalization\nteam names / formats / dedup",
                        "Tables\nmatches / maps / news"], 3.55),
        ("Feature layer", ["Ratings\nElo / map Elo",
                           "Form & schedule\nform / rest",
                           "Events\nnews decay"], 2.3),
        ("Model layer", ["Logistic regression\nL2 + time weights",
                         "Calibration & evaluation\nlog loss / ECE",
                         "Map veto\nDP combination"], 1.05),
    ]
    for name, boxes, y in layers:
        ax.text(0.42, y + 0.35, name, fontsize=9, color="#333", rotation=90,
                va="center", ha="center")
        for i, b in enumerate(boxes):
            x = 1.0 + i * 3.0
            ax.add_patch(FancyBboxPatch((x, y), 2.6, 0.75,
                                        boxstyle="round,pad=0.04,rounding_size=0.08",
                                        linewidth=0.8, edgecolor="#8aa4bd",
                                        facecolor="#eef4fa"))
            ax.text(x + 1.3, y + 0.375, b, ha="center", va="center", fontsize=7.6)
    for y in (3.35, 2.1):
        ax.add_patch(FancyArrowPatch((5.0, y), (5.0, y - 0.22), arrowstyle="-|>",
                                     mutation_scale=9, color="#6b7f92", linewidth=0.8))
    ax.text(9.7, 0.45, "Service layer: CLI / HTTP API / dashboard / daily job",
            ha="right", va="center", fontsize=7.6, color="#555")
    fig.tight_layout()
    fig.savefig(OUT / "fig_arch.png", bbox_inches="tight")
    plt.close(fig)


def fig_coverage(conn):
    rows = conn.execute("""
        SELECT substr(m.start_date,1,7) ym, count(*) n,
               sum(CASE WHEN EXISTS (SELECT 1 FROM maps x WHERE x.match_id=m.match_id) THEN 1 ELSE 0 END) m
        FROM matches m GROUP BY ym ORDER BY ym
    """).fetchall()
    ym = [r["ym"] for r in rows]
    n = np.array([r["n"] for r in rows])
    m = np.array([r["m"] for r in rows])
    x = np.arange(len(ym))
    fig, ax1 = plt.subplots(figsize=(6.6, 2.8))
    ax1.fill_between(x, n, color="#bcd4e6", alpha=0.8, label="Matches")
    ax1.plot(x, n, color=C1, linewidth=1.0)
    ax1.set_ylabel("Matches per month", fontsize=9)
    ax2 = ax1.twinx()
    share = np.divide(m, n, out=np.zeros_like(m, dtype=float), where=n > 0) * 100
    ax2.plot(x, share, color=C2, linewidth=1.2, label="Map coverage")
    ax2.set_ylabel("Map coverage (%)", fontsize=9)
    ax2.grid(False)
    ticks = [i for i, v in enumerate(ym) if v.endswith(("-01", "-07"))]
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([ym[i] for i in ticks], rotation=45, ha="right", fontsize=7)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_coverage.png", bbox_inches="tight")
    plt.close(fig)


def fig_contribution(decomp):
    splits = decomp["splits"]
    groups = [GROUPS_EN.get(e["group"], e["group"]) for e in decomp["leave_one_out"]]
    full = {sp: decomp["full"][sp]["log_loss"] for sp in splits}
    deltas = {sp: [e["splits"][sp]["log_loss"] - full[sp] for e in decomp["leave_one_out"]]
              for sp in splits}
    y = np.arange(len(groups))
    h = 0.36
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    ax.barh(y + h / 2, deltas[splits[0]], height=h, color=C1,
            label=f"Test from {splits[0]}", alpha=0.9)
    ax.barh(y - h / 2, deltas[splits[1]], height=h, color=C2,
            label=f"Test from {splits[1]}", alpha=0.9)
    ax.axvline(0, color="#555", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(groups, fontsize=8.5)
    ax.set_xlabel("Change in log loss after removing the group (positive = helpful)", fontsize=8.5)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "fig_contribution.png", bbox_inches="tight")
    plt.close(fig)


def fig_nested(decomp):
    splits = decomp["splits"]
    ks, v1, v2 = [], [], []
    for e in decomp["nested"]:
        ks.append(e["k"])
        v1.append(e["splits"][splits[0]]["log_loss"])
        v2.append(e["splits"][splits[1]]["log_loss"])
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    ax.plot(ks, v1, marker="o", markersize=3.5, color=C1, label=f"Test from {splits[0]}")
    ax.plot(ks, v2, marker="s", markersize=3.5, color=C2, label=f"Test from {splits[1]}")
    ax.axhline(decomp["elo"][splits[0]]["log_loss"], color=C1, linestyle=":", linewidth=0.9, alpha=0.6)
    ax.axhline(decomp["elo"][splits[1]]["log_loss"], color=C2, linestyle=":", linewidth=0.9, alpha=0.6)
    ax.set_xlabel("Cumulative number of features", fontsize=9)
    ax.set_ylabel("Log loss", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_nested.png", bbox_inches="tight")
    plt.close(fig)


def fig_calibration(rows, feats, split):
    res = split_eval(rows, split, "lr", feats=feats)
    p, y = np.array(res["p"]), np.array(res["y"])
    table = reliability_table(p, y, bins=10)
    xs = [t["avg_p"] for t in table if t["n"] > 0]
    ys = [t["win_rate"] for t in table if t["n"] > 0]
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999", linewidth=0.9, label="Perfect calibration")
    ax.plot(xs, ys, marker="o", markersize=4, color=C3, label="This model")
    ax.set_xlabel("Predicted probability", fontsize=9)
    ax.set_ylabel("Observed frequency", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title(f"Test from {split}", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_calibration.png", bbox_inches="tight")
    plt.close(fig)


def fig_map_model():
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    names = ["Raw map Elo", "Calibrated model"]
    vals = [0.74329, 0.69911]
    bars = ax.bar(names, vals, width=0.5, color=[C1, C3])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v - 0.02, f"{v:.4f}", ha="center", va="top",
                fontsize=8, color="white")
    ax.set_ylim(0.6, 0.78)
    ax.set_ylabel("Map-level log loss", fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_map_model.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    decomp = json.loads(pathlib.Path("paper/data/decomp.json").read_text(encoding="utf-8"))
    fig_architecture()
    fig_contribution(decomp)
    fig_nested(decomp)

    conn = connect()
    fig_coverage(conn)
    fig_map_model()

    maps = {}
    for r in conn.execute("SELECT match_id, map_name, winner FROM maps ORDER BY match_id, map_order"):
        maps.setdefault(r["match_id"], []).append((r["map_name"], r["winner"]))
    rows = build(load_matches(conn), maps=maps, lineups=build_lineups(conn))

    S = ["elo_diff", "elo_a", "elo_b", "log_games_a", "log_games_b"]
    FORM = ["form10_a", "form10_b", "form30_a", "form30_b"]
    SCHED = ["rest_a", "rest_b", "density_a", "density_b"]
    recommended = S + FORM + SCHED + ["h2h"] + EXTRA_FEATURES + MAP_FEATURES + MAPSERIES_FEATURES
    fig_calibration(rows, recommended, "2026-09-01")
    print("figures:", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
