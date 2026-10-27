"""Team-name normalization and format parsing.

Principle: only deterministic normalization (case, whitespace, alias table); no fuzzy matching --
merging two different teams by mistake is more dangerous than keeping duplicates. The alias table is
maintained by hand in configs/team_aliases.json and wins any conflict.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from typing import Iterable
from pathlib import Path

from .config import CONFIGS

# For bo1 the data source puts the map abbreviation in the bostyle field
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
    """Normalize to a comparison key: NFKC, strip, collapse whitespace, casefold."""
    s = unicodedata.normalize("NFKC", name or "")
    s = _WS.sub(" ", s).strip()
    return s.casefold()


def canonical_team(name: str) -> str:
    """Return the display team name; if it hits the alias table, return the canonical name from the table."""
    key = canonical_key(name)
    aliases = load_aliases()
    if key in aliases:
        return aliases[key]
    s = unicodedata.normalize("NFKC", name or "")
    return _WS.sub(" ", s).strip()


def team_id(name: str) -> str:
    """Stable ID: the key of the canonical name (lowercase, whitespace removed)."""
    return canonical_key(canonical_team(name))


def resolve_team(name: str, known: Iterable[str]) -> tuple[str | None, str]:
    """Resolve a user-supplied team name to an existing team.

    Returns (canonical name | None, match method): exact / stripped / prefixed / fuzzy / unknown.
    Fuzzy matching is accepted only on a unique hit, to avoid conflating two different teams.
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
    """Parse bostyle into (format, map_name).

    'bo3' -> ('bo3', None)
    'inf' -> ('bo1', 'Inferno')
    'bo1' -> ('bo1', None)
    """
    raw = (bostyle or "").strip().lower()
    if raw in ("bo1", "bo3", "bo5"):
        return raw, None
    if raw in MAP_CODES:
        return "bo1", MAP_CODES[raw]
    # Fallback: treat unknown values as bo1 but leave the map empty, so map statistics are not polluted
    return "bo1", None
