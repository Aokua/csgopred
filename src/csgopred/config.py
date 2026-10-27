"""Paths and global constants."""

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

# Default Elo parameters (can be overridden in backtest/predict)
ELO_BASE = 1500.0
ELO_SCALE = 400.0
ELO_K_DEFAULT = 24.0
ELO_K_MID = 32.0
ELO_K_TOP = 40.0

# Inactivity decay: 60-day grace, then a 180-day half-life back toward the mean
INACTIVE_GRACE_DAYS = 60
INACTIVE_HALF_LIFE_DAYS = 180

# Event-tier keywords (used to bucket K; heuristic when the data source has no tier field)
TIER1_KEYWORDS = (
    "major", "iem", "blast", "esl pro league", "esl one", "pgl",
    "katowice", "cologne", "dreamhack masters", "epl",
)


def ensure_dirs() -> None:
    for p in (RAW, INTERIM, PROCESSED, MANIFESTS, REPORTS, CONFIGS):
        p.mkdir(parents=True, exist_ok=True)
