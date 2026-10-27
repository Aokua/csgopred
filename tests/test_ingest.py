"""导入与赔率测试。"""

import csv

import pytest

from csgopred.db import connect
from csgopred.ingest.csv_source import ingest_csv
from csgopred.models.odds import fair_odds, kelly_fraction, offer_odds, overround


@pytest.fixture()
def sample_csv(tmp_path):
    p = tmp_path / "m.csv"
    rows = [
        ["date", "team1", "team2", "team1score", "team2score", "event", "bostyle"],
        ["2020-01-01", "A", "B", "2", "0", "IEM Katowice", "bo3"],
        ["2020-01-02", "C", "D", "14", "16", "ESEA", "mrg"],
        ["2020-01-03", "A", "B", "1", "1", "Tie Cup", "bo1"],   # 平局应跳过
        ["2020-01-04", "E", "E", "2", "1", "Self Cup", "bo3"],  # 同队应跳过
        ["2020-01-05", "F", "G", "", "1", "Bad Cup", "bo3"],    # 缺分应跳过
    ]
    with open(p, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    return p


def test_ingest_counts_and_skips(sample_csv, tmp_path):
    conn = connect(tmp_path / "t.db")
    m = ingest_csv(conn, sample_csv, source="test")
    assert m["rows_kept"] == 2
    assert m["ties"] == 1
    assert conn.execute("SELECT count(*) FROM matches").fetchone()[0] == 2


def test_ingest_is_idempotent(sample_csv, tmp_path):
    conn = connect(tmp_path / "t.db")
    ingest_csv(conn, sample_csv, source="test")
    ingest_csv(conn, sample_csv, source="test")
    assert conn.execute("SELECT count(*) FROM matches").fetchone()[0] == 2


def test_bo1_map_name_recorded(sample_csv, tmp_path):
    conn = connect(tmp_path / "t.db")
    ingest_csv(conn, sample_csv, source="test")
    row = conn.execute("SELECT format, map_name FROM matches WHERE team_a='C'").fetchone()
    assert row["format"] == "bo1"
    assert row["map_name"] == "Mirage"


def test_teams_table_populated(sample_csv, tmp_path):
    conn = connect(tmp_path / "t.db")
    ingest_csv(conn, sample_csv, source="test")
    assert conn.execute("SELECT count(*) FROM teams").fetchone()[0] == 4


def test_fair_and_offer_odds():
    assert fair_odds(0.5) == 2.0
    a, b = offer_odds(0.6, 0.06)
    assert overround(a, b) == pytest.approx(0.06, abs=1e-3)
    assert a < fair_odds(0.6)


def test_kelly_zero_when_no_edge():
    assert kelly_fraction(0.5, 1.9) == 0.0


def test_kelly_positive_with_edge():
    assert kelly_fraction(0.6, 2.0) > 0
