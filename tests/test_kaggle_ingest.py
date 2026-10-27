"""Kaggle 导入器的离线测试（最小 CSV 夹具）。"""

import csv

import pytest

from csgopred.db import connect
from csgopred.ingest.kaggle_source import ingest_kaggle


@pytest.fixture()
def kaggle_dir(tmp_path):
    d = tmp_path / "kaggle"
    d.mkdir()

    results = [
        ["date", "team_1", "team_2", "_map", "result_1", "result_2", "map_winner",
         "starting_ct", "ct_1", "t_2", "t_1", "ct_2", "event_id", "match_id",
         "rank_1", "rank_2", "map_wins_1", "map_wins_2", "match_winner"],
        ["2020-03-18", "Alpha", "Beta", "Dust2", "16", "10", "1", "2", "8", "6", "8", "10",
         "1", "1001", "5", "9", "2", "0", "1"],
        ["2020-03-18", "Alpha", "Beta", "Inferno", "16", "12", "1", "1", "9", "6", "7", "9",
         "1", "1001", "5", "9", "2", "0", "1"],
        ["2020-03-19", "Gamma", "Delta", "Mirage", "16", "13", "1", "2", "8", "5", "8", "8",
         "2", "1002", "3", "4", "16", "13", "1"],
    ]
    picks = [
        ["date", "team_1", "team_2", "inverted_teams", "match_id", "event_id", "best_of",
         "system", "t1_removed_1", "t1_removed_2", "t1_removed_3", "t2_removed_1",
         "t2_removed_2", "t2_removed_3", "t1_picked_1", "t2_picked_1", "left_over"],
        ["2020-03-18", "Alpha", "Beta", "0", "1001", "1", "3", "123412",
         "Nuke", "Train", "0.0", "Vertigo", "Overpass", "0.0", "Dust2", "Inferno", "Mirage"],
    ]
    players = [
        ["date", "player_name", "team", "opponent", "country", "player_id", "match_id",
         "event_id", "event_name", "best_of", "map_1", "map_2", "rating"],
        ["2020-03-18", "p1", "Alpha", "Beta", "SE", "1", "1001", "1", "Test Cup", "3",
         "Dust2", "Inferno", "1.20"],
        ["2020-03-18", "p2", "Alpha", "Beta", "SE", "2", "1001", "1", "Test Cup", "3",
         "Dust2", "Inferno", "0.80"],
    ]
    for name, rows in (("results.csv", results), ("picks.csv", picks), ("players.csv", players)):
        with open(d / name, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
    return d


def test_ingest_basic(kaggle_dir, tmp_path):
    conn = connect(tmp_path / "t.db")
    s = ingest_kaggle(conn, kaggle_dir)
    assert s["matches_new"] == 2
    assert conn.execute("select count(*) from maps").fetchone()[0] == 3
    assert conn.execute("select count(*) from veto").fetchone()[0] == 7
    assert s["team_stats"] == 1  # 两名球员同属 Alpha，聚合为一行


def test_bo1_uses_series_score(kaggle_dir, tmp_path):
    conn = connect(tmp_path / "t.db")
    ingest_kaggle(conn, kaggle_dir)
    row = conn.execute(
        "select format, score_a, score_b from matches where team_a='Gamma'"
    ).fetchone()
    assert row["format"] == "bo1"
    assert (row["score_a"], row["score_b"]) == (1, 0)


def test_bo3_score_and_ranks(kaggle_dir, tmp_path):
    conn = connect(tmp_path / "t.db")
    ingest_kaggle(conn, kaggle_dir)
    row = conn.execute(
        "select format, score_a, score_b, rank_a, rank_b from matches where team_a='Alpha'"
    ).fetchone()
    assert row["format"] == "bo3"
    assert (row["score_a"], row["score_b"]) == (2, 0)
    assert (row["rank_a"], row["rank_b"]) == (5, 9)


def test_event_name_filled_from_players(kaggle_dir, tmp_path):
    conn = connect(tmp_path / "t.db")
    ingest_kaggle(conn, kaggle_dir)
    ev = conn.execute("select event from matches where team_a='Alpha'").fetchone()["event"]
    assert ev == "Test Cup"


def test_team_stats_average(kaggle_dir, tmp_path):
    conn = connect(tmp_path / "t.db")
    ingest_kaggle(conn, kaggle_dir)
    row = conn.execute(
        "select avg_rating, n_players from team_match_stats where team='Alpha'"
    ).fetchone()
    assert row["n_players"] == 2
    assert abs(row["avg_rating"] - 1.0) < 1e-9
