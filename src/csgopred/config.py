"""路径与全局常量。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
MANIFESTS = DATA / "manifests"
REPORTS = ROOT / "reports"
CONFIGS = ROOT / "configs"

DB_PATH = DATA / "csgopred.db"

# Elo 默认参数（可在 backtest/predict 里覆盖）
ELO_BASE = 1500.0
ELO_SCALE = 400.0
ELO_K_DEFAULT = 24.0
ELO_K_MID = 32.0
ELO_K_TOP = 40.0

# 不活跃衰减：60 天宽限，之后半衰期 180 天回归均值
INACTIVE_GRACE_DAYS = 60
INACTIVE_HALF_LIFE_DAYS = 180

# 赛事级别关键词（用于 K 值分档；数据源无 tier 字段时的启发式）
TIER1_KEYWORDS = (
    "major", "iem", "blast", "esl pro league", "esl one", "pgl",
    "katowice", "cologne", "dreamhack masters", "epl",
)


def ensure_dirs() -> None:
    for p in (RAW, INTERIM, PROCESSED, MANIFESTS, REPORTS, CONFIGS):
        p.mkdir(parents=True, exist_ok=True)
