"""服务层：把模型能力包装成可复用的函数（CLI / API / 每日任务共用）。"""

from __future__ import annotations

import json
import random
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

from ..config import DATA
from ..eval.backtest import load_matches, train_full
from ..features.pipeline import FEATURES, map_breakdown_asof, replay
from ..features.news_features import load_events
from ..models import ml
from ..models.odds import fair_odds, kelly_fraction, offer_odds
from ..models.rating import predict_from_elo
from ..normalize import canonical_team, resolve_team

MODEL_PATH = DATA / "model.pkl"


def _model():
    if MODEL_PATH.exists():
        return ml.load(MODEL_PATH)
    return None, {}


def rankings(conn: sqlite3.Connection, top: int = 30) -> list[dict]:
    matches = load_matches(conn)
    if not matches:
        return []
    model, a, b = train_full(matches)
    latest = date.fromisoformat(max(m["start_date"] for m in matches))
    out = []
    for r in model.snapshot(latest)[:top]:
        out.append(
            {
                "team": r["team"],
                "elo": round(r["elo"], 1),
                "games": r["games"],
                "last_played": r["last_played"],
            }
        )
    return out


def predict(conn: sqlite3.Connection, team_a: str, team_b: str, fmt: str = "bo3",
            asof: str | None = None, event: str = "", margin: float = 0.06,
            bankroll: float | None = None, odds_a: float | None = None,
            odds_b: float | None = None) -> dict:
    asof_d = date.fromisoformat(asof) if asof else None
    matches = load_matches(conn)
    before = [m for m in matches if date.fromisoformat(m["start_date"]) < asof_d] if asof_d else matches
    elo_model, ca, cb = train_full(before)

    ta_in = team_a
    tb_in = team_b
    known = list(elo_model.ratings.keys())
    ra_name, how_a = resolve_team(ta_in, known)
    rb_name, how_b = resolve_team(tb_in, known)
    team_a = ra_name or canonical_team(ta_in)
    team_b = rb_name or canonical_team(tb_in)

    when = asof_d or date.fromisoformat(max(m["start_date"] for m in before))
    ra = elo_model.rating(team_a, when)
    rb = elo_model.rating(team_b, when)
    na = elo_model.n_games(team_a)
    nb = elo_model.n_games(team_b)
    p = predict_from_elo(ca, cb, ra - rb)
    used = "elo"

    m2, meta = _model()
    drivers = []
    map_detail: list[dict] = []
    if m2 is not None:
        try:
            news_ev = load_events(conn)
            fb = replay(before, news_ev)
            feat = fb.features(team_a, team_b, when, fmt, event)
            feats = m2.feats or FEATURES
            if any(k.startswith("map_") for k in feats):
                from ..features.pipeline import map_features_asof
                from ..db import connect as _conn
                maps = {}
                for r in conn.execute(
                        "SELECT match_id, map_name, winner FROM maps ORDER BY match_id, map_order"):
                    maps.setdefault(r["match_id"], []).append((r["map_name"], r["winner"]))
                elo_model2 = fb.elo
                feat.update(map_features_asof(before, maps, when, team_a, team_b, fmt, elo_model2))
                map_detail = map_breakdown_asof(before, maps, when, team_a, team_b, fmt, elo_model2)
            import numpy as np

            X = np.array([[feat[k] for k in feats]])
            p = float(m2.predict(X)[0])
            used = meta.get("kind", m2.kind)
            if m2.kind == "lr":
                coef = m2.est.coef_[0]
                contrib = sorted(
                    ((feats[i], float(coef[i] * X[0][i])) for i in range(len(feats))),
                    key=lambda t: -abs(t[1]),
                )[:4]
                drivers = [{"name": k, "value": round(v, 3)} for k, v in contrib]
        except Exception:
            pass

    rng = random.Random(42)
    sa = 110.0 / (1.0 + na / 5) ** 0.5
    sb = 110.0 / (1.0 + nb / 5) ** 0.5
    sims: list[float] = []
    if m2 is not None:
        try:
            import numpy as np

            news_ev = load_events(conn)
            base_feat = replay(before, news_ev).features(team_a, team_b, when, fmt, event)
            for _ in range(300):
                f = dict(base_feat)
                f["elo_diff"] = base_feat["elo_diff"] + rng.gauss(0, (sa**2 + sb**2) ** 0.5 / 400)
                f["elo_a"] = base_feat["elo_a"] + rng.gauss(0, sa / 400)
                f["elo_b"] = base_feat["elo_b"] + rng.gauss(0, sb / 400)
                sims.append(float(m2.predict(np.array([[f[k] for k in feats]]))[0]))
        except Exception:
            sims = []
    if not sims:
        sims = sorted(
            predict_from_elo(ca, cb, (ra + rng.gauss(0, sa)) - (rb + rng.gauss(0, sb)))
            for _ in range(400)
        )
    sims.sort()
    p_lo, p_hi = sims[int(0.025 * len(sims))], sims[int(0.975 * len(sims)) - 1]

    o_a, o_b = offer_odds(p, margin)


    bets = []
    if bankroll and bankroll > 0 and (odds_a or odds_b):
        from ..models.betting import advice

        if odds_a:
            bets.append({"side": team_a, "odds": odds_a,
                         **advice(p, odds_a, bankroll, p_low=p_lo)})
        if odds_b:
            bets.append({"side": team_b, "odds": odds_b,
                         **advice(1 - p, odds_b, bankroll, p_low=1 - p_hi)})

    return {
        "team_a": team_a,
        "team_b": team_b,
        "format": fmt,
        "asof": asof or "latest",
        "model": used,
        "p_a": round(p, 4),
        "p_a_ci95": [round(p_lo, 4), round(p_hi, 4)],
        "fair_odds": {"a": round(fair_odds(p), 3), "b": round(fair_odds(1 - p), 3)},
        "offer_odds": {"a": o_a, "b": o_b, "margin": margin},
        "kelly": {"a": round(kelly_fraction(p, o_a), 5), "b": round(kelly_fraction(1 - p, o_b), 5)},
        "elo": {"a": round(ra, 1), "b": round(rb, 1), "games_a": na, "games_b": nb},
        "drivers": drivers,
        "bets": bets,
        "maps": map_detail,
        "notes": [f"「{team_a}」按 {how_a} 匹配" if how_a not in ("exact", "unknown") else ""],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def dataset_stats(conn: sqlite3.Connection) -> dict:
    q = conn.execute("SELECT count(*), min(start_date), max(start_date) FROM matches").fetchone()
    teams = conn.execute("SELECT count(*) FROM teams").fetchone()[0]
    news = conn.execute("SELECT count(*) FROM news").fetchone()[0]
    events = conn.execute("SELECT count(*) FROM news_events").fetchone()[0]
    by_source = [dict(r) for r in conn.execute("SELECT source, count(*) c FROM matches GROUP BY source")]
    return {
        "matches": q[0],
        "first": q[1],
        "last": q[2],
        "teams": teams,
        "news": news,
        "events": events,
        "by_source": by_source,
    }


def recent_news(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    rows = conn.execute(
        """SELECT e.team, e.event_type, e.direction, e.magnitude, e.evidence,
                  n.published_at, n.title, n.url
           FROM news_events e JOIN news n ON n.article_id = e.article_id
           ORDER BY n.published_at DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def model_info() -> dict:
    _, meta = _model()
    return meta or {}
