"""复现论文主结果：训练、评估、保存模型。

用法：python repro/train.py
依赖：data/csgopred.db（见 repro/snapshot.json 的 sha256）
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from csgopred.db import connect  # noqa: E402
from csgopred.eval.backtest import load_matches, train_full  # noqa: E402
from csgopred.eval.rolling import split_eval  # noqa: E402
from csgopred.features.pipeline import (  # noqa: E402
    FEATURES, MAP_FEATURES, MAPSERIES_FEATURES, build,
)
from csgopred.ingest.kaggle_maps import build_lineups  # noqa: E402
from csgopred.models import ml  # noqa: E402

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    set_seed()
    cfg = json.loads((ROOT / "repro" / "config.json").read_text(encoding="utf-8"))
    conn = connect()
    matches = load_matches(conn)
    maps = {}
    for r in conn.execute("SELECT match_id, map_name, winner FROM maps ORDER BY match_id, map_order"):
        maps.setdefault(r["match_id"], []).append((r["map_name"], r["winner"]))
    rows = build(matches, maps=maps, lineups=build_lineups(conn))
    print(f"matches={len(rows)} maps={sum(len(v) for v in maps.values())}")

    base = FEATURES
    mapall = MAP_FEATURES + MAPSERIES_FEATURES
    results = {}
    for name, feats in (("base20", base), ("base+map", base + mapall)):
        results[name] = {}
        for sp in cfg["splits"]:
            m = split_eval(rows, sp, "lr", feats=feats)["metrics"]
            results[name][sp] = round(m["log_loss"], 5)
    print(json.dumps(results, indent=2))

    # 用推荐特征集在全量数据上训练并保存
    recommended = cfg["features"]["recommended25"]
    model, a, b = train_full(matches)
    m = ml.Model("lr")
    m.feats = list(recommended)
    from csgopred.models.ml import labels, matrix, time_weights
    m.fit(matrix(rows, recommended), labels(rows), time_weights(rows))
    ml.save(m, ROOT / "data" / "model_repro.pkl", {"kind": "lr", "trained_to": rows[-1]["date"],
                "rows": len(rows), "features": recommended, "seed": SEED})
    print("saved data/model_repro.pkl")

    out = ROOT / "repro" / "results.json"
    out.write_text(json.dumps({"seed": SEED, "results": results,
                               "features": {"base20": len(base),
                                            "base+map": len(base) + len(mapall),
                                            "recommended25": len(recommended)}},
                              indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
