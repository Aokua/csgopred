"""Command-line entry point.

Usage:
  python -m csgopred ingest --path data/raw/.../csgoresults.csv --source hojlund
  python -m csgopred stats
  python -m csgopred backtest --split 0.7 --report reports/M1_baseline.md
  python -m csgopred rank --top 20
  python -m csgopred predict -a "Team Vitality" -b "Natus Vincere" --format bo3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

from ..config import DATA, DB_PATH, REPORTS, ensure_dirs
from ..db import connect
from ..eval.backtest import load_matches, train_full, walk_forward
from ..eval.metrics import bootstrap_ci, reliability_table
from ..models.odds import fair_odds, kelly_fraction, offer_odds
from ..models.rating import predict_from_elo
from ..normalize import canonical_team, resolve_team


def _conn(args):
    return connect(getattr(args, "db", None))


def _out(s: str = "") -> None:
    print(s)


def _table(headers: list[str], rows: list[list], aligns: list[str] | None = None) -> str:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))
    aligns = aligns or ["l"] * len(headers)

    def fmt(cells):
        out = []
        for i, c in enumerate(cells):
            s = str(c)
            out.append(s.rjust(widths[i]) if aligns[i] == "r" else s.ljust(widths[i]))
        return "| " + " | ".join(out) + " |"

    lines = [fmt(headers), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    lines += [fmt(r) for r in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------- ingest
def cmd_ingest(args) -> int:
    from ..ingest.csv_source import ingest_csv

    conn = _conn(args)
    m = ingest_csv(conn, args.path, source=args.source, dry_run=args.dry_run)
    _out(json.dumps(m, ensure_ascii=False, indent=2))
    total = conn.execute("SELECT count(*) FROM matches").fetchone()[0]
    _out(f"\n库内总计：{total} 场")
    return 0


def cmd_stats(args) -> int:
    conn = _conn(args)
    n = conn.execute("SELECT count(*) FROM matches").fetchone()[0]
    if n == 0:
        _out("数据库为空，先运行 ingest。")
        return 1
    rng = conn.execute("SELECT min(start_date), max(start_date) FROM matches").fetchone()
    teams = conn.execute("SELECT count(*) FROM teams").fetchone()[0]
    fmts = conn.execute(
        "SELECT format, count(*) c FROM matches GROUP BY format ORDER BY c DESC"
    ).fetchall()

    _out(f"比赛数     : {n}")
    _out(f"日期范围   : {rng[0]} ~ {rng[1]}")
    _out(f"战队数     : {teams}")
    _out("赛制分布   :")
    for f in fmts:
        _out(f"  {f['format']:<6} {f['c']}")
    top = conn.execute(
        "SELECT * FROM teams ORDER BY matches DESC LIMIT 10"
    ).fetchall()
    _out("\n出场最多：")
    for t in top:
        _out(f"  {t['name']:<32} {t['matches']}")
    return 0


# ---------------------------------------------------------------- backtest
def cmd_backtest(args) -> int:
    conn = _conn(args)
    matches = load_matches(conn, date_from=args.date_from, date_to=args.date_to)
    res = walk_forward(matches, split_ratio=args.split, min_train=args.min_train)

    _out(f"训练期 : {res['train_range'][0]} ~ {res['train_range'][1]}  ({res['train_n']} 场)")
    _out(f"测试期 : {res['test_range'][0]} ~ {res['test_range'][1]}  ({res['test_n']} 场)")
    _out(f"拟合参数: a={res['logistic']['a']}  b={res['logistic']['b']}")

    rows = []
    for key in ("model", "elo_raw", "coin"):
        s = res[key]
        rows.append([s["name"], s["n"], s["log_loss"], s["brier"], s["accuracy"], s["ece"]])
    _out("\n" + _table(["model", "n", "log_loss", "brier", "acc", "ece"], rows,
                       ["l", "r", "r", "r", "r", "r"]))

    ps = [p["p"] for p in res["predictions"]]
    ys = [p["y"] for p in res["predictions"]]
    lo, hi = bootstrap_ci(ps, ys, "log_loss")
    _out(f"\nlog loss 95% CI: [{lo}, {hi}]")

    _out("\n分段指标：")
    seg_rows = [
        [k, v["n"], v["log_loss"], v["brier"], v["accuracy"], v["ece"]]
        for k, v in res["segments"].items()
    ]
    _out(_table(["segment", "n", "log_loss", "brier", "acc", "ece"], seg_rows,
                ["l", "r", "r", "r", "r", "r"]))

    cal = reliability_table(ps, ys)
    _out("\n校准表：")
    _out(_table(["bin", "n", "avg_p", "win_rate", "gap"],
                [[r["bin"], r["n"], r["avg_p"], r["win_rate"], r["gap"]] for r in cal],
                ["l", "r", "r", "r", "r"]))

    if args.report:
        REPORTS.mkdir(parents=True, exist_ok=True)
        path = Path(args.report)
        if not path.is_absolute():
            path = REPORTS / path
        _write_report(path, res, cal, (lo, hi))
        _out(f"\n报告已写入：{path}")
    return 0


def _write_report(path, res, cal, ci) -> None:
    rows = []
    for key in ("model", "elo_raw", "coin"):
        s = res[key]
        rows.append([s["name"], s["n"], s["log_loss"], s["brier"], s["accuracy"], s["ece"]])
    seg_rows = [
        [k, v["n"], v["log_loss"], v["brier"], v["accuracy"], v["ece"]]
        for k, v in res["segments"].items()
    ]
    md = [
        "# M1 基线：Elo + 逻辑回归",
        "",
        f"> 生成时间：{datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"> 训练期：{res['train_range'][0]} ~ {res['train_range'][1]}（{res['train_n']} 场）",
        f"> 测试期：{res['test_range'][0]} ~ {res['test_range'][1]}（{res['test_n']} 场）",
        "",
        "## 主指标",
        "",
        "| model | n | log_loss | brier | acc | ece |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    md += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |" for r in rows]
    md += [
        "",
        f"log loss 95% bootstrap CI：**[{ci[0]}, {ci[1]}]**",
        "",
        "## 分段",
        "",
        "| segment | n | log_loss | brier | acc | ece |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    md += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |" for r in seg_rows]
    md += [
        "",
        "## 校准",
        "",
        "| bin | n | avg_p | win_rate | gap |",
        "|---|---:|---:|---:|---:|",
    ]
    md += [f"| {r['bin']} | {r['n']} | {r['avg_p']} | {r['win_rate']} | {r['gap']} |" for r in cal]
    md += [
        "",
        "## 说明与局限",
        "",
        "- 本基线只用了赛果（谁赢、比分），没有地图 veto、阵容、选手数据。",
        "- 训练期/测试期按时间切分，测试期每场只用此前信息，无未来数据泄漏。",
        "- Elo 的 K 值按赛事名启发式分档，属于工程近似，不是最优参数。",
        "- 概率映射在训练期拟合，测试期固定；后续 M2 引入特征后再做滚动重训。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------- rank
def cmd_rank(args) -> int:
    conn = _conn(args)
    matches = load_matches(conn)
    model, a, b = train_full(matches)
    latest = date.fromisoformat(max(m["start_date"] for m in matches))
    snap = model.snapshot(latest)
    _out(_table(["#", "team", "elo", "games", "last_played"],
                [[i + 1, r["team"], r["elo"], r["games"], r["last_played"]]
                 for i, r in enumerate(snap[: args.top])],
                ["r", "l", "r", "r", "l"]))
    return 0


# ---------------------------------------------------------------- predict
def cmd_predict(args) -> int:
    conn = _conn(args)
    team_a = canonical_team(args.a)
    team_b = canonical_team(args.b)
    asof = date.fromisoformat(args.date) if args.date else None
    matches = load_matches(conn)
    before = [m for m in matches if date.fromisoformat(m["start_date"]) < asof] if asof else matches

    elo_model, ca, cb = train_full(before)
    known = list(elo_model.ratings.keys())
    ra_name, how_a = resolve_team(team_a, known)
    rb_name, how_b = resolve_team(team_b, known)
    notes = []
    if ra_name:
        team_a = ra_name
    else:
        notes.append(f"未找到「{team_a}」，按初始分 1500 处理")
    if rb_name:
        team_b = rb_name
    else:
        notes.append(f"未找到「{team_b}」，按初始分 1500 处理")

    when = asof or date.fromisoformat(max(m["start_date"] for m in before))
    ra = elo_model.rating(team_a, when)
    rb = elo_model.rating(team_b, when)
    na = elo_model.n_games(team_a)
    nb = elo_model.n_games(team_b)
    p = predict_from_elo(ca, cb, ra - rb)
    model_used = "elo"

    from pathlib import Path as _P
    import numpy as _np
    from ..features.pipeline import FEATURES as _FEATS
    from ..features.pipeline import replay as _replay
    from ..models import ml as _ml

    model_path = _P(args.model) if args.model else (DATA / "model.pkl")
    drivers = []
    if model_path.exists():
        try:
            m2, meta = _ml.load(model_path)
            news_ev = {}
            try:
                from ..features.news_features import load_events as _le
                news_ev = _le(conn)
            except Exception:
                news_ev = {}
            fb = _replay(before, news_ev)
            feat = fb.features(team_a, team_b, when, args.format, args.event)
            feats = m2.feats or _FEATS
            if any(k.startswith("map_") for k in feats):
                from ..features.pipeline import map_features_asof
                maps = {}
                for r in conn.execute(
                        "SELECT match_id, map_name, winner FROM maps ORDER BY match_id, map_order"):
                    maps.setdefault(r["match_id"], []).append((r["map_name"], r["winner"]))
                feat.update(map_features_asof(before, maps, when, team_a, team_b,
                                              args.format, fb.elo))
            X = _np.array([[feat[k] for k in feats]])
            p = float(m2.predict(X)[0])
            model_used = meta.get("kind", m2.kind)
            if m2.kind == "lr":
                coef = m2.est.coef_[0]
                contrib = sorted(
                    ((feats[i], float(coef[i] * X[0][i])) for i in range(len(feats))),
                    key=lambda t: -abs(t[1]),
                )[:4]
                drivers = [{"type": k, "text": f"{k} {v:+.3f}"} for k, v in contrib]
        except Exception as exc:
            notes.append(f"模型加载失败（{type(exc).__name__}: {exc}），回退 Elo")

    # Uncertainty: recompute after perturbing the Elo features to approximate a 95% interval
    rng = random.Random(42)
    sa = 110.0 / (1.0 + na / 5) ** 0.5 / 400.0
    sb = 110.0 / (1.0 + nb / 5) ** 0.5 / 400.0
    delta = (ra - rb) / 400.0
    sims = []
    if model_path.exists() and model_used != "elo":
        try:
            fb = _replay(before)
            base = fb.features(team_a, team_b, when, args.format, args.event)
            for _ in range(300):
                f = dict(base)
                d = delta + rng.gauss(0, (sa ** 2 + sb ** 2) ** 0.5)
                f["elo_diff"] = d
                f["elo_a"] = (ra - 1500) / 400 + rng.gauss(0, sa)
                f["elo_b"] = (rb - 1500) / 400 + rng.gauss(0, sb)
                sims.append(float(m2.predict(_np.array([[f[k] for k in feats]]))[0]))
        except Exception:
            sims = []
    if not sims:
        sims = sorted(
            predict_from_elo(ca, cb, (ra + rng.gauss(0, sa * 400)) - (rb + rng.gauss(0, sb * 400)))
            for _ in range(600)
        )
    sims.sort()
    p_lo, p_hi = sims[int(0.025 * len(sims))], sims[int(0.975 * len(sims)) - 1]

    fo_a, fo_b = fair_odds(p), fair_odds(1 - p)
    o_a, o_b = offer_odds(p, args.margin)
    k_a = kelly_fraction(p, o_a)
    k_b = kelly_fraction(1 - p, o_b)

    payload = {
        "match": {"team_a": team_a, "team_b": team_b, "format": args.format,
                  "asof": asof.isoformat() if asof else "latest"},
        "model": model_used,
        "p_a": round(p, 4),
        "p_a_ci95": [round(p_lo, 4), round(p_hi, 4)],
        "fair_odds": {"a": round(fo_a, 3), "b": round(fo_b, 3)},
        "offer_odds": {"a": o_a, "b": o_b, "margin": args.margin},
        "kelly_fraction": {"a": round(k_a, 5), "b": round(k_b, 5)},
        "elo": {"a": round(ra, 1), "b": round(rb, 1), "games_a": na, "games_b": nb},
        "drivers": drivers or [
            {"type": "rating", "text": f"Elo 差 {ra - rb:+.1f}"},
            {"type": "sample", "text": f"场次 A={na} B={nb}"},
        ],
        "data_quality": {"low_sample": na < 5 or nb < 5},
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if args.json:
        _out(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    _out(f"{team_a}  vs  {team_b}   [{args.format}]   model={model_used}")
    _out(f"Elo      : {ra:.1f}  vs  {rb:.1f}   (场次 {na} / {nb})")
    _out(f"胜率 A   : {p:.1%}   (95% CI {p_lo:.1%} ~ {p_hi:.1%})")
    _out(f"公平赔率 : {fo_a:.2f} / {fo_b:.2f}")
    _out(f"报价(水{args.margin:.0%}): {o_a:.2f} / {o_b:.2f}")
    if drivers:
        _out("主要驱动 : " + "；".join(d["text"] for d in drivers))
    for n in notes:
        _out(f"⚠ {n}")
    if payload["data_quality"]["low_sample"] and not notes:
        _out("⚠ 样本不足 5 场，区间仅供参考")
    for name, how in ((team_a, how_a), (team_b, how_b)):
        if how in ("stripped", "prefixed", "fuzzy"):
            _out(f"  · 「{name}」按 {how} 匹配")
    return 0


def cmd_train(args) -> int:
    """M2: build features, compare models across time splits, save the online model."""
    import numpy as _np
    from ..features.pipeline import FEATURES as _FEATS
    from ..features.pipeline import build as _build
    from ..models import ml as _ml
    from ..eval.rolling import ablation, feature_importance, split_eval
    from ..eval.metrics import reliability_table

    conn = _conn(args)
    matches = load_matches(conn)
    _out(f"比赛 {len(matches)} 场，构建特征…")
    rows = _build(matches)
    _out(f"特征行 {len(rows)}，维度 {len(_FEATS)}")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    all_res = {}
    for sp in splits:
        res = ablation(rows, sp)
        all_res[sp] = res
        _out(f"\n== 测试起点 {sp} ==")
        _out(_table(["model", "n", "log_loss", "brier", "acc", "ece"],
                    [[k, res[k]["test_n"], res[k]["metrics"]["log_loss"],
                      res[k]["metrics"]["brier"], res[k]["metrics"]["accuracy"],
                      res[k]["metrics"]["ece"]] for k in ("elo", "lr", "hgb")],
                    ["l", "r", "r", "r", "r", "r"]))

    # Pick the best model type (by mean log loss across splits)
    mean_ll = {
        k: sum(all_res[sp][k]["metrics"]["log_loss"] for sp in splits) / len(splits)
        for k in ("elo", "lr", "hgb")
    }
    best_kind = min((k for k in mean_ll if k != "elo"), key=lambda k: mean_ll[k])
    _out(f"\n平均 log loss: " + ", ".join(f"{k}={v:.4f}" for k, v in mean_ll.items()))
    _out(f"选用模型: {best_kind}")

    # Retrain on the full data for predict to use
    X = _ml.matrix(rows)
    y = _ml.labels(rows)
    w = _ml.time_weights(rows)
    final = _ml.Model(best_kind).fit(X, y, w, calib_frac=0.15 if best_kind == "hgb" else 0.0)
    meta = {"kind": best_kind, "trained_to": rows[-1]["date"], "rows": len(rows),
            "mean_log_loss": {k: round(v, 5) for k, v in mean_ll.items()},
            "features": _FEATS}
    _ml.save(final, args.model_out or (DATA / "model.pkl"), meta)
    _out(f"模型已保存: {args.model_out or (DATA / 'model.pkl')}")

    if args.report:
        path = Path(args.report)
        if not path.is_absolute():
            path = REPORTS / path
        _write_m2_report(path, all_res, splits, best_kind, mean_ll,
                         feature_importance(all_res[splits[0]]["lr"]["model"]),
                         reliability_table([float(x) for x in all_res[splits[0]][best_kind]["p"]],
                                           [int(x) for x in all_res[splits[0]][best_kind]["y"]]))
        _out(f"报告已写入: {path}")
    return 0


def _write_m2_report(path, all_res, splits, best_kind, mean_ll, importance, cal) -> None:
    md = ["# M2：特征模型对比", "",
          f"> 生成时间：{datetime.now(timezone.utc).isoformat(timespec='seconds')}", ""]
    for sp in splits:
        res = all_res[sp]
        md += [f"## 测试起点 {sp}", "",
               f"训练 {res['lr']['train_n']} 场 / 测试 {res['lr']['test_n']} 场", "",
               "| model | log_loss | Brier | 准确率 | ECE |", "|---|---:|---:|---:|---:|"]
        for k in ("elo", "lr", "hgb"):
            m = res[k]["metrics"]
            md.append(f"| {k} | {m['log_loss']} | {m['brier']} | {m['accuracy']} | {m['ece']} |")
        md += ["", "分段（" + best_kind + "）：", "",
               "| segment | n | log_loss | ECE |", "|---|---:|---:|---:|"]
        for seg, m in res[best_kind]["segments"].items():
            md.append(f"| {seg} | {m['n']} | {m['log_loss']} | {m['ece']} |")
        md.append("")
    md += ["## 特征权重（逻辑回归，全量训练）", "", "| feature | coef |", "|---|---:|"]
    md += [f"| {k} | {v:+.4f} |" for k, v in importance]
    md += ["", "## 校准（第一个切分，最优模型）", "", "| bin | n | avg_p | win_rate | gap |",
           "|---|---:|---:|---:|---:|"]
    md += [f"| {r['bin']} | {r['n']} | {r['avg_p']} | {r['win_rate']} | {r['gap']} |" for r in cal]
    md += ["", "## 说明", "",
           "- 所有特征在比赛前计算（Elo、近况、休息天数、密度、H2H、赛制、赛事级别、LAN）。",
           "- 训练样本按时间衰减加权，半衰期 540 天。",
           "- HGB 用训练期最后 15% 做 Platt 校准。", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(md), encoding="utf-8")


def cmd_hltv(args) -> int:
    """Scrape HLTV: result list + optional match details (maps/veto)."""
    from ..ingest.csv_source import _refresh_teams
    from ..ingest.hltv import HltvClient, ingest_details, ingest_results

    conn = _conn(args)
    client = HltvClient(rate=args.rate)
    try:
        res = ingest_results(conn, client, max_pages=args.max_pages,
                             since=args.since, start_offset=args.offset)
        _out(f"结果列表: {res['pages']} 页, 新增 {res['inserted']} 场, 最旧 {res['oldest']}")
        _out(f"下次续抓: --offset {res['next_offset']}")
        if args.details > 0:
            det = ingest_details(conn, client, limit=args.details, since=args.since)
            _out(f"详情页  : {det['matches']} 场, 地图 {det['maps']} 条, veto {det['veto']} 条")
        _refresh_teams(conn)
    except Exception as exc:
        _out(f"抓取中断: {type(exc).__name__}: {exc}")
        _out(f"下次续抓: --offset {args.offset + args.max_pages * 100}（或降低频率、间隔一段时间再跑）")
        return 2
    total = conn.execute("SELECT count(*) FROM matches").fetchone()[0]
    _out(f"库内总计: {total} 场")
    return 0


def cmd_hltv_backfill(args) -> int:
    """Unattended slow backfill: scrape in batches, and cool down and retry when blocked, until --since is reached."""
    import time
    from ..ingest.csv_source import _refresh_teams
    from ..ingest.hltv import HltvBlocked, HltvClient, ingest_results

    conn = _conn(args)
    offset = args.offset
    fails = 0
    while True:
        client = HltvClient(rate=args.rate, attempts=2)
        try:
            res = ingest_results(conn, client, max_pages=args.pages,
                                 since=args.since, start_offset=offset)
            fails = 0
            _out(f"[{offset}] 新增 {res['inserted']} 场, 最旧 {res['oldest']}")
            offset = res["next_offset"]
            if res["oldest"] and args.since and res["oldest"] < args.since:
                _out("已到达目标日期，回填完成。")
                break
            time.sleep(args.pause)
        except HltvBlocked as exc:
            fails += 1
            wait = min(args.cooldown * (2 ** (fails - 1)), args.max_cooldown)
            _out(f"[{offset}] 被拦截（第 {fails} 次），冷却 {wait:.0f}s：{exc}")
            if fails > args.max_retries:
                _out("连续被拦，先停。冷却一段时间后用 --offset 继续。")
                break
            time.sleep(wait)
    _refresh_teams(conn)
    total = conn.execute("SELECT count(*) FROM matches").fetchone()[0]
    _out(f"库内总计: {total} 场（续抓 offset={offset}）")
    return 0


def cmd_hltv_details(args) -> int:
    """Scrape match details (per-map scores + veto) for the map model."""
    import time
    from ..ingest.hltv import HltvBlocked, HltvClient, ingest_details

    conn = _conn(args)
    total_maps = total_veto = total_matches = 0
    fails = 0
    while True:
        client = HltvClient(rate=args.rate, attempts=2)
        try:
            res = ingest_details(conn, client, limit=args.batch, since=args.since,
                                 min_stars=args.min_stars)
            fails = 0
            total_matches += res["matches"]
            total_maps += res["maps"]
            total_veto += res["veto"]
            _out(f"累计 {total_matches} 场 / 地图 {total_maps} / veto {total_veto}"
                 f"（本批 {res['matches']}）")
            if res["matches"] == 0 or total_matches >= args.limit:
                break
            time.sleep(args.pause)
        except HltvBlocked as exc:
            fails += 1
            wait = min(args.cooldown * (2 ** (fails - 1)), args.max_cooldown)
            _out(f"被拦截（第 {fails} 次），冷却 {wait:.0f}s：{exc}")
            if fails > args.max_retries:
                break
            time.sleep(wait)
    _out(f"完成：{total_matches} 场详情，地图 {total_maps} 条，veto {total_veto} 条")
    return 0


def cmd_ingest_kaggle(args) -> int:
    """Import the Kaggle dataset (results/picks/players)."""
    from ..ingest.kaggle_source import ingest_kaggle

    conn = _conn(args)
    s = ingest_kaggle(conn, args.dir)
    _out(json.dumps(s, ensure_ascii=False, indent=2))
    return 0


def cmd_merge_teams(args) -> int:
    """Merge team-name variants that differ only in case."""
    from ..ingest.csv_source import _refresh_teams, merge_team_variants

    conn = _conn(args)
    n = merge_team_variants(conn)
    _refresh_teams(conn)
    _out(f"合并变体 {n} 个")
    return 0


def cmd_news(args) -> int:
    """News scraping / extraction / overview."""
    import time as _time
    from ..ingest import news as _news

    conn = _conn(args)
    if args.action == "stats":
        conn.executescript(_news.SCHEMA)
        rows = conn.execute(
            "SELECT source, count(*) c, min(published_at) a, max(published_at) b FROM news GROUP BY source"
        ).fetchall()
        if not rows:
            _out("新闻库为空")
            return 0
        for r in rows:
            _out(f"{r['source']:<6} {r['c']:>6} 篇  {r['a']} ~ {r['b']}")
        ev = conn.execute(
            "SELECT event_type, count(*) c FROM news_events GROUP BY event_type ORDER BY c DESC"
        ).fetchall()
        _out("事件：" + ", ".join(f"{e['event_type']}={e['c']}" for e in ev) if ev else "事件：无")
        return 0

    if args.action == "extract":
        res = _news.extract_events(conn, limit=args.limit, source=args.source)
        _out(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    # ingest
    items = []
    if args.source == "hltv":
        from ..ingest.hltv import HltvClient

        client = HltvClient(rate=args.rate, attempts=2)
        for page in range(1, args.pages + 1):
            url = "/news/archive" if page == 1 else f"/news/archive?page={page}"
            try:
                html = client.get(url)
            except Exception as exc:
                _out(f"列表页失败（{url}）：{exc}")
                break
            links = _news.parse_hltv_list(html)
            if args.signal_only:
                import re as _re
                sig = _re.compile(r"(sign|join|part|leave|bench|injur|coach|ban|suspen|retir|transfer|roster|stand-?in|replace|depart|kick|miss|out-of|break)", _re.I)
                links = [l for l in links if sig.search(l)]
            _out(f"第 {page} 页：{len(links)} 条")
            for link in links[: args.limit]:
                try:
                    it = _news.parse_hltv_article(client.get(link), link)
                    if it:
                        items.append(it)
                except Exception:
                    continue
    else:
        from curl_cffi import requests as creq

        sess = creq.Session(impersonate="chrome124")
        html = sess.get(_news.PW_BASE + "/", timeout=(10, 25)).content.decode("utf-8", "ignore")
        links = _news.parse_pw_list(html)
        _out(f"列表：{len(links)} 条")
        for link in links[: args.limit]:
            _time.sleep(args.rate)
            try:
                raw = sess.get(_news.PW_BASE + link, timeout=(10, 25)).content
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    text = raw.decode("gbk", "ignore")
                it = _news.parse_pw_article(text, link)
                if it:
                    items.append(it)
            except Exception as exc:
                _out(f"  skip {link}: {type(exc).__name__}")
    n = _news.store(conn, items)
    _out(f"入库 {n} 篇（抓取 {len(items)} 篇）")
    return 0


def cmd_news_llm(args) -> int:
    """LLM event extraction (with no key, --dry-run only prints the template)."""
    from ..ingest.llm_extract import extract_with_llm

    conn = _conn(args)
    res = extract_with_llm(conn, limit=args.limit, dry_run=args.dry_run,
                           source=args.source, signal_only=args.signal_only,
                           titles_only=args.titles_only, require_team=args.require_team)
    _out(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


_MONTHS = ["january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december"]


def _month_range(since: str):
    """since -> current month, returns ["2023-09", ...]."""
    from datetime import date as _d
    y, m = int(since[:4]), int(since[5:7])
    today = _d.today()
    out = []
    while (y, m) <= (today.year, today.month):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def _goto_retry(b, url: str, ready: str, tries: int = 3, wait: float = 45.0) -> str:
    """Cloudflare challenge-page detection + backoff retry."""
    import time as _t
    html = ""
    for attempt in range(tries):
        html = b.goto(url, wait=4, ready_check=ready, max_wait=25)
        if ready in html:
            return html
        _out(f"    challenge? 第 {attempt+1} 次未拿到内容，等待 {wait:.0f}s")
        _t.sleep(wait)
    return html


def cmd_news_browser(args) -> int:
    """Scrape news with a CDP browser (HLTV / 5EPlay), then store it."""
    from ..ingest import browser_news as bn
    from ..ingest import news as _news
    from ..ingest.browser import Browser

    conn = _conn(args)
    items = []
    with Browser() as b:
        if args.source == "hltv":
            b.new_page(bn.HLTV_ARCHIVE)
            months = _month_range(args.since or "2026-01-01")
            total = 0
            for ym in months:
                year, mon = ym.split("-")
                url = f"{bn.HLTV_ARCHIVE}/{year}/{_MONTHS[int(mon)-1]}"
                try:
                    html = _goto_retry(b, url, "newsline")
                except Exception as exc:
                    _out(f"{ym} 打开失败：{type(exc).__name__}")
                    continue
                rows = bn.hltv_items(html)
                if args.since:
                    rows = [r for r in rows if r["date"] >= args.since]
                if args.signal_only:
                    rows = [r for r in rows if bn.SIGNAL_RE.search(r["title"])]
                _out(f"{ym}: {len(rows)} 条候选")
                for r in rows:
                    if total >= args.limit:
                        break
                    try:
                        it = bn.hltv_article(b, r)
                        if it:
                            items.append(it)
                            total += 1
                    except Exception as exc:
                        _out(f"  skip {r['path']}: {type(exc).__name__}")
                if total >= args.limit:
                    break
        else:
            from ..ingest import five_e as fe

            b.new_page(fe.LIST_PAGE)
            b.goto(fe.LIST_PAGE, wait=6, ready_check="article-main", max_wait=30)
            since_ts = 0
            if args.since:
                from datetime import datetime as _dt, timezone as _tz
                since_ts = int(_dt.fromisoformat(args.since).replace(tzinfo=_tz.utc).timestamp())
            rows = fe.list_articles(b, pages=max(1, args.pages), since_ts=since_ts,
                                    start_page=max(1, args.start_page))
            if args.signal_only:
                rows = [r for r in rows if bn.SIGNAL_RE.search(r["title"])]
            if args.cs_only:
                import re as _re
                # Exclude non-CS titles (Valorant / LoL, etc.); unicode escapes avoid encoding issues
                exclude = _re.compile("Valorant|\u65e0\u754f\u5951\u7ea6|\u82f1\u96c4\u8054\u76df|Dota|\u5200\u5854", _re.I)
                rows = [r for r in rows if not exclude.search(r["title"])]
            _out(f"5E 列表：{len(rows)} 条候选")
            for r in rows[: args.limit]:
                try:
                    it = fe.fetch_body(b, r)
                    if it:
                        items.append(it)
                except Exception as exc:
                    _out(f"  skip {r['path']}: {type(exc).__name__}")
    n = _news.store(conn, items)
    _out(f"入库 {n} 篇（抓取 {len(items)} 篇）")
    return 0


def cmd_news_llm(args) -> int:
    """LLM event extraction (with no key, --dry-run only prints the template)."""
    from ..ingest.llm_extract import extract_with_llm

    conn = _conn(args)
    res = extract_with_llm(conn, limit=args.limit, dry_run=args.dry_run,
                           source=args.source, signal_only=args.signal_only,
                           titles_only=args.titles_only, require_team=args.require_team)
    _out(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


_MONTHS = ["january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december"]


def _month_range(since: str):
    """since -> current month, returns ["2023-09", ...]."""
    from datetime import date as _d
    y, m = int(since[:4]), int(since[5:7])
    today = _d.today()
    out = []
    while (y, m) <= (today.year, today.month):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def _goto_retry(b, url: str, ready: str, tries: int = 3, wait: float = 45.0) -> str:
    """Cloudflare challenge-page detection + backoff retry."""
    import time as _t
    html = ""
    for attempt in range(tries):
        html = b.goto(url, wait=4, ready_check=ready, max_wait=25)
        if ready in html:
            return html
        _out(f"    challenge? 第 {attempt+1} 次未拿到内容，等待 {wait:.0f}s")
        _t.sleep(wait)
    return html


def cmd_news_browser(args) -> int:
    """Scrape news with a CDP browser (HLTV / 5EPlay), then store it."""
    from ..ingest import browser_news as bn
    from ..ingest import news as _news
    from ..ingest.browser import Browser

    conn = _conn(args)
    items = []
    with Browser() as b:
        if args.source == "hltv":
            b.new_page(bn.HLTV_ARCHIVE)
            months = _month_range(args.since or "2026-01-01")
            total = 0
            for ym in months:
                year, mon = ym.split("-")
                url = f"{bn.HLTV_ARCHIVE}/{year}/{_MONTHS[int(mon)-1]}"
                try:
                    html = _goto_retry(b, url, "newsline")
                except Exception as exc:
                    _out(f"{ym} 打开失败：{type(exc).__name__}")
                    continue
                rows = bn.hltv_items(html)
                if args.since:
                    rows = [r for r in rows if r["date"] >= args.since]
                if args.signal_only:
                    rows = [r for r in rows if bn.SIGNAL_RE.search(r["title"])]
                _out(f"{ym}: {len(rows)} 条候选")
                for r in rows:
                    if total >= args.limit:
                        break
                    try:
                        it = bn.hltv_article(b, r)
                        if it:
                            items.append(it)
                            total += 1
                    except Exception as exc:
                        _out(f"  skip {r['path']}: {type(exc).__name__}")
                if total >= args.limit:
                    break
        else:
            b.new_page(bn.WY_LIST)
            url = bn.WY_LIST if args.pages == 1 else bn.WY_LIST
            html = b.goto(url, wait=6, ready_check="article-main", max_wait=30)
            rows = bn.wy_items(html)
            if args.signal_only:
                rows = [r for r in rows if bn.SIGNAL_RE.search(r["path"])]
            _out(f"5E 列表：{len(rows)} 条候选")
            for r in rows[: args.limit]:
                try:
                    it = bn.wy_article(b, r)
                    if it:
                        items.append(it)
                except Exception as exc:
                    _out(f"  skip {r['path']}: {type(exc).__name__}")
    n = _news.store(conn, items)
    _out(f"入库 {n} 篇（抓取 {len(items)} 篇）")
    return 0


_EXCLUDE_5E = None


def cmd_news_browser(args) -> int:
    """Scrape news with a CDP browser: HLTV (monthly archive) / 5EPlay (official JSON API)."""
    import re as _re
    from ..ingest import browser_news as bn
    from ..ingest import five_e as fe
    from ..ingest import news as _news
    from ..ingest.browser import Browser

    global _EXCLUDE_5E
    if _EXCLUDE_5E is None:
        _EXCLUDE_5E = _re.compile(
            "Valorant|\u65e0\u754f\u5951\u7ea6|\u82f1\u96c4\u8054\u76df|Dota|\u5200\u5854", _re.I
        )

    conn = _conn(args)
    items: list = []
    with Browser() as b:
        if args.source == "hltv":
            b.new_page(bn.HLTV_ARCHIVE)
            total = 0
            for ym in _month_range(args.since or "2026-01-01"):
                y, m = ym.split("-")
                url = f"{bn.HLTV_ARCHIVE}/{y}/{_MONTHS[int(m) - 1]}"
                try:
                    html = _goto_retry(b, url, "newsline")
                except Exception as exc:
                    _out(f"{ym}: 打开失败 {type(exc).__name__}")
                    continue
                rows = bn.hltv_items(html)
                if args.since:
                    rows = [r for r in rows if r["date"] >= args.since]
                if args.signal_only:
                    rows = [r for r in rows if bn.SIGNAL_RE.search(r["title"])]
                _out(f"{ym}: {len(rows)} 条候选")
                for r in rows:
                    if total >= args.limit:
                        break
                    try:
                        it = bn.hltv_article(b, r)
                        if it:
                            items.append(it)
                            total += 1
                    except Exception:
                        continue
                if total >= args.limit:
                    break
        elif args.source == "5e":
            b.new_page(fe.LIST_PAGE)
            b.goto(fe.LIST_PAGE, wait=6, ready_check="article-main", max_wait=30)
            since_ts = 0
            if args.since:
                since_ts = int(
                    datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc).timestamp()
                )
            rows = fe.list_articles(b, pages=max(1, args.pages), since_ts=since_ts,
                                    start_page=max(1, args.start_page))
            if args.signal_only:
                rows = [r for r in rows if bn.SIGNAL_RE.search(r["title"])]
            if args.cs_only:
                rows = [r for r in rows if not _EXCLUDE_5E.search(r["title"])]
            _out(f"5E: {len(rows)} 条候选")
            if args.titles_only:
                from ..ingest.news import NewsItem as _NI
                import re as _re2
                for r in rows[: args.limit]:
                    m = _re2.search(r"/article/article-(\d+)-(\d+)", r["path"])
                    items.append(
                        _NI(
                            article_id=f"5e:{m.group(2)}" if m else f"5e:{len(items)}",
                            source="5e",
                            url=r["url"],
                            published_at=r["date"],
                            title=r["title"],
                            summary="",
                            body="",
                            lang="zh",
                        )
                    )
            else:
                for r in rows[: args.limit]:
                    try:
                        it = fe.fetch_body(b, r)
                        if it:
                            items.append(it)
                    except Exception as exc:
                        _out(f"  skip {r['path']}: {type(exc).__name__}")
        else:
            from curl_cffi import requests as creq

            sess = creq.Session(impersonate="chrome124")
            html = sess.get(_news.PW_BASE + "/", timeout=(10, 25)).content.decode("utf-8", "ignore")
            for link in _news.parse_pw_list(html)[: args.limit]:
                raw = sess.get(_news.PW_BASE + link, timeout=(10, 25)).content
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    text = raw.decode("gbk", "ignore")
                it = _news.parse_pw_article(text, link)
                if it:
                    items.append(it)

    n = _news.store(conn, items)
    _out(f"入库 {n} 篇（抓取 {len(items)} 篇）")
    return 0


def cmd_train(args) -> int:
    """M2/M3: Elo -> feature model -> news-feature ablation, evaluated across time splits."""
    import numpy as _np
    from ..features.pipeline import ALL_FEATURES, FEATURES as BASE_FEATURES
    from ..features.pipeline import MAPSERIES_FEATURES, MAP_FEATURES
    from ..features.pipeline import build as _build
    from ..features.news_features import load_events
    from ..models import ml as _ml
    from ..eval.rolling import split_eval, feature_importance
    from ..eval.metrics import reliability_table

    conn = _conn(args)
    matches = load_matches(conn)
    news = load_events(conn) if args.with_news else {}
    maps: dict[str, list[tuple[str, str]]] = {}
    for r in conn.execute("SELECT match_id, map_name, winner FROM maps ORDER BY match_id, map_order"):
        maps.setdefault(r["match_id"], []).append((r["map_name"], r["winner"]))
    _out(f"比赛 {len(matches)} 场，新闻队伍 {len(news)} 支，地图 {len(maps)} 场，构建特征…")
    rows = _build(matches, news, maps)
    MAP_FEATS = BASE_FEATURES + MAPSERIES_FEATURES + MAP_FEATURES
    covered = sum(1 for r in rows if r["features"].get("news_cnt_a", 0) + r["features"].get("news_cnt_b", 0) > 0)
    _out(f"特征行 {len(rows)}，其中带新闻特征 {covered} 行")

    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    results = {}
    for sp in splits:
        res = {
            "elo": split_eval(rows, sp, "elo"),
            "lr": split_eval(rows, sp, "lr", feats=BASE_FEATURES),
            "lr_maps": split_eval(rows, sp, "lr", feats=MAP_FEATS),
        }
        if news:
            res["lr_news"] = split_eval(rows, sp, "lr", feats=ALL_FEATURES)
            res["hgb_news"] = split_eval(rows, sp, "hgb", feats=ALL_FEATURES)
        else:
            res["hgb"] = split_eval(rows, sp, "hgb", feats=BASE_FEATURES)
        results[sp] = res
        _out(f"\n== 测试起点 {sp} ==")
        keys = list(res.keys())
        _out(_table(["model", "n", "log_loss", "brier", "acc", "ece"],
                    [[k, res[k]["test_n"], res[k]["metrics"]["log_loss"],
                      res[k]["metrics"]["brier"], res[k]["metrics"]["accuracy"],
                      res[k]["metrics"]["ece"]] for k in keys],
                    ["l", "r", "r", "r", "r", "r"]))

    # News-covered subset: global metrics get diluted by the many matches without news
    if news:
        from ..eval.metrics import summarize as _summ
        for sp in splits:
            res = results[sp]
            base = res["lr"]
            idx = [i for i, r in enumerate(base["rows"])
                   if r["features"].get("news_cnt_a", 0) + r["features"].get("news_cnt_b", 0) > 0]
            if not idx:
                _out(f"\n[{sp}] 新闻覆盖子集：0 场")
                continue
            line = [f"[{sp}] 新闻覆盖子集 n={len(idx)}"]
            for key in ("lr", "lr_news"):
                ps = [float(res[key]["p"][i]) for i in idx]
                ys = [int(res[key]["y"][i]) for i in idx]
                m = _summ(ps, ys, key)
                line.append(f"{key}: logloss={m['log_loss']:.4f} acc={m['accuracy']:.3f}")
            _out("\n" + "  ".join(line))

    # Model selection: among lr / lr_news / hgb, by mean log loss
    candidates = [k for k in ("lr", "lr_maps", "lr_news", "hgb", "hgb_news")
                  if all(k in results[s] for s in splits)]
    mean_ll = {k: sum(results[s][k]["metrics"]["log_loss"] for s in splits) / len(splits) for k in candidates}
    best = min(candidates, key=lambda k: mean_ll[k])
    if "maps" in best:
        feats = MAP_FEATS
    elif "news" in best:
        feats = ALL_FEATURES
    else:
        feats = BASE_FEATURES
    _out("\n平均 log loss: " + ", ".join(f"{k}={v:.4f}" for k, v in mean_ll.items()))
    _out(f"选用: {best}（{len(feats)} 维）")

    kind = "hgb" if best.startswith("hgb") else "lr"
    X = _ml.matrix(rows, feats)
    y = _ml.labels(rows)
    w = _ml.time_weights(rows)
    final = _ml.Model(kind)
    final.feats = list(feats)
    final.fit(X, y, w, calib_frac=0.15 if kind == "hgb" else 0.0)
    meta = {"kind": kind, "features": feats, "trained_to": rows[-1]["date"],
            "rows": len(rows), "news_rows": covered,
            "mean_log_loss": {k: round(v, 5) for k, v in mean_ll.items()}}
    path = Path(args.model_out) if args.model_out else (DATA / "model.pkl")
    _ml.save(final, path, meta)
    _out(f"模型已保存: {path}")

    if args.report:
        rp = Path(args.report)
        if not rp.is_absolute():
            rp = REPORTS / rp
        _write_m3_report(rp, results, splits, best, mean_ll, covered,
                         feature_importance(results[splits[0]]["lr"]["model"]))
        _out(f"报告已写入: {rp}")
    return 0


def _write_m3_report(path, results, splits, best, mean_ll, covered, importance) -> None:
    md = ["# M2/M3：特征与新闻消融", "",
          f"> 生成时间：{datetime.now(timezone.utc).isoformat(timespec='seconds')}",
          f"> 带新闻特征样本：{covered} 行", ""]
    names = {"elo": "Elo 基线", "lr": "LR（基础）", "lr_maps": "LR + 地图",
             "lr_news": "LR + 新闻", "hgb": "HGB", "hgb_news": "HGB + 新闻"}
    for sp in splits:
        res = results[sp]
        md += [f"## 测试起点 {sp}", "", "| 模型 | log_loss | Brier | 准确率 | ECE |",
               "|---|---:|---:|---:|---:|"]
        for k, r in res.items():
            m = r["metrics"]
            md.append(f"| {names.get(k, k)} | {m['log_loss']} | {m['brier']} | {m['accuracy']} | {m['ece']} |")
        md.append("")
    md += ["## 平均", "", "| 模型 | mean log_loss |", "|---|---:|"]
    md += [f"| {names.get(k, k)} | {v:.5f} |" for k, v in mean_ll.items()]
    md += ["", "## 特征权重（LR，无新闻）", "", "| feature | coef |", "|---|---:|"]
    md += [f"| {k} | {v:+.4f} |" for k, v in importance]
    md += ["", "## 说明", "",
           "- 新闻特征 = 事件按 30 天半衰期衰减的净信号，仅用发布时间早于比赛的文章。",
           "- 当前新闻覆盖窗口很短，消融只在少数比赛上生效，结论仅作方向参考。", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(md), encoding="utf-8")


def cmd_news_bodies(args) -> int:
    """Backfill bodies for news items that only have titles (5E)."""
    from ..ingest import browser_news as bn
    from ..ingest import five_e as fe
    from ..ingest.browser import Browser

    conn = _conn(args)
    rows = conn.execute(
        """SELECT article_id, url, published_at FROM news
           WHERE source='5e' AND (body IS NULL OR body='')
           ORDER BY published_at DESC LIMIT ?""",
        (args.limit,),
    ).fetchall()
    if not rows:
        _out("没有待补正文的新闻")
        return 0
    _out(f"待补 {len(rows)} 篇")
    done = 0
    with Browser() as b:
        b.new_page(rows[0]["url"])
        for r in rows:
            item = {"url": r["url"], "path": r["url"].split("5eplay.com")[-1],
                    "date": r["published_at"]}
            try:
                art = fe.fetch_body(b, item)
            except Exception:
                continue
            if not art or not art.body:
                continue
            conn.execute(
                "UPDATE news SET body=?, title=COALESCE(NULLIF(?,''), title) WHERE article_id=?",
                (art.body, art.title, r["article_id"]),
            )
            done += 1
            if done % 25 == 0:
                conn.commit()
                _out(f"  已补 {done}/{len(rows)}")
    conn.commit()
    _out(f"完成：{done} 篇")
    return 0


def cmd_market(args) -> int:
    """M4: import odds and run the market comparison (CLV / ROI / calibration)."""
    from ..eval.market_eval import evaluate, load_odds_csv, store_odds

    conn = _conn(args)
    if args.odds:
        rows = load_odds_csv(args.odds)
        n = store_odds(conn, rows)
        _out(f"导入赔率 {n} 条（{len(rows)} 行）")
    if args.eval:
        res = evaluate(conn, args.start)
        _out(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def cmd_serve(args) -> int:
    """Start the web interface."""
    from .api import run

    run(port=args.port, host=args.host)
    return 0


def cmd_daily(args) -> int:
    """Daily update: scrape new results -> (optional) retrain -> write the daily report."""
    import time as _t
    from ..features.pipeline import FEATURES, build
    from ..models import ml
    from ..models.ml import Model, labels, matrix, time_weights

    conn = _conn(args)
    last = conn.execute("SELECT max(start_date) FROM matches").fetchone()[0]
    _out(f"当前最新比赛日期: {last}")

    new = 0
    if not args.skip_ingest:
        from ..ingest.hltv import HltvClient, ingest_results

        client = HltvClient(rate=args.rate, attempts=2)
        try:
            res = ingest_results(conn, client, max_pages=args.pages, since=last)
            new = res["inserted"]
            _out(f"抓取: {res['pages']} 页, 新增 {new} 场, 最旧 {res['oldest']}")
        except Exception as exc:
            _out(f"抓取失败（继续用现有数据）: {type(exc).__name__}: {exc}")

    trained = False
    if new >= args.min_new or args.force_train:
        matches = load_matches(conn)
        _out(f"重训模型（{len(matches)} 场）…")
        rows = build(matches)
        model = Model("lr")
        model.feats = list(FEATURES)
        model.fit(matrix(rows, model.feats), labels(rows), time_weights(rows))
        meta = {"kind": "lr", "features": list(FEATURES), "trained_to": rows[-1]["date"],
                "rows": len(rows), "daily": True}
        ml.save(model, DATA / "model.pkl", meta)
        trained = True
        _out("模型已更新")

    stamp = datetime.now().strftime("%Y-%m-%d")
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"daily_{stamp}.md"
    stats = conn.execute("SELECT count(*), max(start_date) FROM matches").fetchone()
    path.write_text(
        f"# 每日更新 {stamp}\n\n"
        f"- 新增比赛：{new}\n"
        f"- 重训：{'是' if trained else '否'}\n"
        f"- 库内总计：{stats[0]} 场，最新 {stats[1]}\n",
        encoding="utf-8",
    )
    _out(f"日报: {path}")
    return 0


def cmd_hltv_details_browser(args) -> int:
    """Scrape HLTV map scores/veto with a browser (to get past Cloudflare)."""
    from ..ingest.hltv_browser_details import crawl

    conn = _conn(args)
    res = crawl(conn, limit=args.limit, since=args.since, min_stars=args.min_stars,
                pace=args.pace, out=_out)
    _out(json.dumps(res, ensure_ascii=False))
    return 0


def cmd_liquipedia(args) -> int:
    """Backfill map scores from Liquipedia (2023+)."""
    import time as _t
    from ..ingest import liquipedia as lq

    conn = _conn(args)
    if args.page:
        html = lq.fetch_page(args.page)
        rows = lq.parse_page(html)
        _out(json.dumps(lq.store(conn, rows, args.page), ensure_ascii=False))
        return 0

    events = conn.execute(
        """SELECT event, count(*) c FROM matches
           WHERE start_date >= ? AND stars >= ? AND event != ''
           GROUP BY event ORDER BY c DESC LIMIT ?""",
        (args.since, args.min_stars, args.limit),
    ).fetchall()
    _out(f"候选赛事 {len(events)} 个")
    total = {"parsed": 0, "matched": 0, "maps": 0, "unmatched": 0, "events": 0, "failed": 0}
    for i, e in enumerate(events, 1):
        name = e["event"]
        _t.sleep(2.0)
        try:
            titles = lq.search_titles(name, limit=3)
        except Exception as exc:
            _out(f"[{i}] 搜索失败 {name}: {type(exc).__name__}")
            total["failed"] += 1
            continue
        if not titles:
            _out(f"[{i}] 未找到页面: {name}")
            continue
        _t.sleep(2.0)
        title = titles[0]
        try:
            html = lq.fetch_page(title)
        except Exception as exc:
            _out(f"[{i}] 拉取失败 {title}: {type(exc).__name__}")
            total["failed"] += 1
            continue
        rows = lq.parse_page(html)
        res = lq.store(conn, rows, title)
        total["parsed"] += res["parsed"]
        total["matched"] += res["matched"]
        total["maps"] += res["maps"]
        total["unmatched"] += res["unmatched"]
        total["events"] += 1
        _out(f"[{i}/{len(events)}] {name} -> {title[:40]} | 场次 {res['parsed']} 命中 {res['matched']} 地图 {res['maps']}")
        _t.sleep(2.5)
    _out(json.dumps(total, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    ensure_dirs()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(prog="csgopred", description="CS2 胜率与赔率模型")
    ap.add_argument("--db", default=str(DB_PATH), help="SQLite 路径")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="导入 CSV 赛果")
    p.add_argument("--path", required=True)
    p.add_argument("--source", default="csv")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("stats", help="数据概览")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("backtest", help="walk-forward 回测")
    p.add_argument("--split", type=float, default=0.7)
    p.add_argument("--min-train", type=int, default=1000)
    p.add_argument("--report", default="M1_baseline.md")
    p.add_argument("--date-from", default=None, help="只用该日期之后的比赛")
    p.add_argument("--date-to", default=None, help="只用该日期之前的比赛")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("rank", help="当前 Elo 排名")
    p.add_argument("--top", type=int, default=20)
    p.set_defaults(func=cmd_rank)

    p = sub.add_parser("predict", help="预测单场比赛")
    p.add_argument("-a", "--a", required=True)
    p.add_argument("-b", "--b", required=True)
    p.add_argument("--format", default="bo3", choices=["bo1", "bo3", "bo5"])
    p.add_argument("--date", default=None, help="as-of 日期 YYYY-MM-DD")
    p.add_argument("--margin", type=float, default=0.06)
    p.add_argument("--event", default="", help="赛事名（用于赛事级别/LAN 特征）")
    p.add_argument("--model", default=None, help="模型路径，默认 data/model.pkl")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("hltv", help="从 HLTV 抓取（需 curl_cffi）")
    p.add_argument("--max-pages", type=int, default=1, help="结果列表页数（每页 100 场）")
    p.add_argument("--since", default=None, help="抓到该日期为止 YYYY-MM-DD")
    p.add_argument("--details", type=int, default=0, help="额外抓取多少场比赛详情")
    p.add_argument("--offset", type=int, default=0, help="从第几条结果开始（续抓用）")
    p.add_argument("--rate", type=float, default=2.5, help="请求间隔秒")
    p.set_defaults(func=cmd_hltv)

    p = sub.add_parser("hltv-backfill", help="慢速无人值守回填（被拦自动冷却重试）")
    p.add_argument("--since", default="2023-09-01", help="回填到该日期为止")
    p.add_argument("--pages", type=int, default=20, help="每批页数")
    p.add_argument("--rate", type=float, default=8.0, help="请求间隔秒")
    p.add_argument("--pause", type=float, default=120.0, help="批间暂停秒")
    p.add_argument("--cooldown", type=float, default=900.0, help="被拦冷却秒")
    p.add_argument("--max-cooldown", type=float, default=3600.0, help="冷却上限秒")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--offset", type=int, default=0)
    p.set_defaults(func=cmd_hltv_backfill)

    p = sub.add_parser("hltv-details", help="抓比赛详情（地图/veto），可无人值守")
    p.add_argument("--limit", type=int, default=500, help="总场次上限")
    p.add_argument("--batch", type=int, default=25, help="每批场次")
    p.add_argument("--min-stars", type=int, default=3, help="只要该星级以上（0-5）")
    p.add_argument("--since", default="2024-01-01")
    p.add_argument("--rate", type=float, default=6.0)
    p.add_argument("--pause", type=float, default=30.0)
    p.add_argument("--cooldown", type=float, default=600.0)
    p.add_argument("--max-cooldown", type=float, default=3600.0)
    p.add_argument("--max-retries", type=int, default=20)
    p.set_defaults(func=cmd_hltv_details)

    p = sub.add_parser("ingest-kaggle", help="导入 Kaggle 数据集（地图/veto/选手）")
    p.add_argument("--dir", default="data/raw/kaggle")
    p.set_defaults(func=cmd_ingest_kaggle)

    p = sub.add_parser("train", help="M2：特征模型训练与对比")
    p.add_argument("--splits", default="2023-01-01,2025-08-01", help="逗号分隔的测试起点")
    p.add_argument("--model-out", default=None)
    p.add_argument("--report", default="M3_news.md")
    p.add_argument("--with-news", action="store_true", help="加入新闻事件特征")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("merge-teams", help="合并大小写不同的队名变体")
    p.set_defaults(func=cmd_merge_teams)

    p = sub.add_parser("news", help="新闻：ingest / extract / stats")
    p.add_argument("action", choices=["ingest", "extract", "stats"])
    p.add_argument("--source", default="pw", choices=["pw", "hltv"])
    p.add_argument("--pages", type=int, default=1)
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--rate", type=float, default=1.5)
    p.add_argument("--signal-only", action="store_true", help="HLTV：只抓标题含转会/伤病等关键词的新闻")
    p.set_defaults(func=cmd_news)

    p = sub.add_parser("news-llm", help="用 LLM 抽取新闻事件（OpenAI 兼容）")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--source", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--signal-only", action="store_true")
    p.add_argument("--titles-only", action="store_true")
    p.add_argument("--require-team", action="store_true")
    p.set_defaults(func=cmd_news_llm)

    p = sub.add_parser("news-bodies", help="给 5E 标题-only 新闻补正文")
    p.add_argument("--limit", type=int, default=100)
    p.set_defaults(func=cmd_news_bodies)

    p = sub.add_parser("news-browser", help="用 CDP 浏览器抓新闻（HLTV/5EPlay）")
    p.add_argument("--source", default="hltv", choices=["hltv", "5e"])
    p.add_argument("--pages", type=int, default=1)
    p.add_argument("--limit", type=int, default=15)
    p.add_argument("--since", default=None, help="YYYY-MM-DD，只要该日期之后的")
    p.add_argument("--signal-only", action="store_true")
    p.add_argument("--cs-only", action="store_true", help="5E：只保留 CS 相关标题")
    p.add_argument("--titles-only", action="store_true", help="5E：只存标题（历史回填用）")
    p.add_argument("--start-page", type=int, default=1, help="5E：从第几页开始")
    p.set_defaults(func=cmd_news_browser)

    p = sub.add_parser("market", help="M4：赔率导入与市场评估")
    p.add_argument("--odds", default=None, help="CSV: date,team_a,team_b,odds_a,odds_b[,book]")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--start", default=None, help="赔率窗口起点（默认取最早一条）")
    p.set_defaults(func=cmd_market)

    p = sub.add_parser("serve", help="启动 Web 界面")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="127.0.0.1")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("daily", help="每日更新（抓赛果 + 重训 + 日报）")
    p.add_argument("--pages", type=int, default=2)
    p.add_argument("--rate", type=float, default=4.0)
    p.add_argument("--min-new", type=int, default=1)
    p.add_argument("--force-train", action="store_true")
    p.add_argument("--skip-ingest", action="store_true")
    p.set_defaults(func=cmd_daily)

    p = sub.add_parser("hltv-details-browser", help="浏览器抓 HLTV 地图比分")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--since", default="2023-01-01")
    p.add_argument("--min-stars", type=int, default=0)
    p.add_argument("--pace", type=float, default=2.5)
    p.set_defaults(func=cmd_hltv_details_browser)

    p = sub.add_parser("liquipedia", help="从 Liquipedia 回填地图比分")
    p.add_argument("--page", default=None, help="单个 Liquipedia 页面标题")
    p.add_argument("--since", default="2023-01-01")
    p.add_argument("--min-stars", type=int, default=2)
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_liquipedia)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
