"""队名归一与赛制解析。

原则：只做确定性归一（大小写、空白、别名表），不做模糊匹配——
错误合并两支不同战队比保留重复更危险。别名表在 configs/team_aliases.json
里手工维护，冲突时以别名表为准。
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from typing import Iterable
from pathlib import Path

from .config import CONFIGS

# bo1 时数据源会把地图缩写写进 bostyle 字段
MAP_CODES = {
    "inf": "Inferno",
    "mrg": "Mirage",
    "nuke": "Nuke",
    "d2": "Dust2",
    "ovp": "Overpass",
    "trn": "Train",
    "vtg": "Vertigo",
    "anc": "Ancient",
    "anb": "Anubis",
    "cbl": "Cobblestone",
    "tcn": "Tuscan",
}

_WS = re.compile(r"\s+")


def _alias_path() -> Path:
    return CONFIGS / "team_aliases.json"


@lru_cache(maxsize=1)
def load_aliases() -> dict[str, str]:
    p = _alias_path()
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {canonical_key(k): v for k, v in data.items()}


def canonical_key(name: str) -> str:
    """归一为比较用的 key：NFKC、去首尾空白、压缩空格、casefold。"""
    s = unicodedata.normalize("NFKC", name or "")
    s = _WS.sub(" ", s).strip()
    return s.casefold()


def canonical_team(name: str) -> str:
    """返回展示用队名；命中别名表则返回别名表里的规范名。"""
    key = canonical_key(name)
    aliases = load_aliases()
    if key in aliases:
        return aliases[key]
    s = unicodedata.normalize("NFKC", name or "")
    return _WS.sub(" ", s).strip()


def team_id(name: str) -> str:
    """稳定 ID：规范名的 key（小写、去空白）。"""
    return canonical_key(canonical_team(name))


def resolve_team(name: str, known: Iterable[str]) -> tuple[str | None, str]:
    """把用户输入的队名解析到已有队伍。

    返回 (规范名 | None, 匹配方式)：exact / stripped / prefixed / fuzzy / unknown。
    只在唯一命中时才接受模糊匹配，避免把两支不同战队混为一谈。
    """
    known_map = {canonical_key(k): k for k in known}
    key = canonical_key(canonical_team(name))
    if key in known_map:
        return known_map[key], "exact"
    if key.startswith("team ") and key[5:] in known_map:
        return known_map[key[5:]], "stripped"
    if ("team " + key) in known_map:
        return known_map["team " + key], "prefixed"
    cands = [v for k, v in known_map.items() if key and (key in k or k in key)]
    if len(cands) == 1:
        return cands[0], "fuzzy"
    return None, "unknown"


def parse_format(bostyle: str) -> tuple[str, str | None]:
    """把 bostyle 解析成 (format, map_name)。

    'bo3' -> ('bo3', None)
    'inf' -> ('bo1', 'Inferno')
    'bo1' -> ('bo1', None)
    """
    raw = (bostyle or "").strip().lower()
    if raw in ("bo1", "bo3", "bo5"):
        return raw, None
    if raw in MAP_CODES:
        return "bo1", MAP_CODES[raw]
    # 兜底：未知值按 bo1 处理，但地图置空，避免污染地图统计
    return "bo1", None
