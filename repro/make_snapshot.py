"""生成可复现快照：数据/模型哈希、行数、环境版本。

用法：python repro/make_snapshot.py
输出：repro/snapshot.json
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import sklearn  # noqa: E402

from csgopred.db import connect  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    db = ROOT / "data" / "csgopred.db"
    model = ROOT / "data" / "model.pkl"
    conn = connect(db)
    counts = {}
    for table in ("matches", "maps", "news", "news_events", "teams"):
        try:
            counts[table] = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        except Exception:
            counts[table] = None

    snap = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": 42,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "database": {
            "path": "data/csgopred.db",
            "bytes": db.stat().st_size if db.exists() else None,
            "sha256": sha256(db) if db.exists() else None,
        },
        "model": {
            "path": "data/model.pkl",
            "bytes": model.stat().st_size if model.exists() else None,
            "sha256": sha256(model) if model.exists() else None,
        },
        "counts": counts,
    }
    out = ROOT / "repro" / "snapshot.json"
    out.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(snap, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
