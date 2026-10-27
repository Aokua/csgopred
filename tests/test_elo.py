"""Elo 与队名归一的单元测试。"""

from datetime import date

from csgopred.features.elo import EloModel, decay_toward_mean, expected_score
from csgopred.normalize import canonical_team, parse_format, team_id


def _m(a="A", b="B", sa=2, sb=0, fmt="bo3", event="", d="2020-01-01"):
    return {"team_a": a, "team_b": b, "score_a": sa, "score_b": sb,
            "format": fmt, "event": event, "start_date": d}


def test_equal_ratings_give_half():
    assert expected_score(1500, 1500) == 0.5


def test_expected_score_monotone():
    assert expected_score(1600, 1500) > 0.5
    assert expected_score(1500, 1600) < 0.5


def test_winner_gains_loser_loses():
    model = EloModel()
    model.step(_m())
    assert model.ratings["A"] > 1500
    assert model.ratings["B"] < 1500


def test_zero_sum_updates():
    """Elo 的核心不变量：每场总分守恒（双方得分之和不变）。"""
    model = EloModel()
    model.step(_m("A", "B", 2, 0))
    assert abs((model.ratings["A"] + model.ratings["B"]) - 3000) < 1e-6
    model.step(_m("B", "A", 2, 1, d="2020-01-02"))
    assert abs((model.ratings["A"] + model.ratings["B"]) - 3000) < 1e-6


def test_symmetric_round_trip_near_start():
    model = EloModel()
    model.step(_m("A", "B", 2, 0))
    model.step(_m("B", "A", 2, 0, d="2020-01-02"))
    # 往返后接近起点（不完全相等：第二场 B 是劣势方，赢下后收益略大）
    assert abs(model.ratings["A"] - 1500) < 12.0
    assert abs(model.ratings["B"] - 1500) < 12.0


def test_decay_grace_then_revert():
    assert decay_toward_mean(1700, 60) == 1700
    assert 1500 < decay_toward_mean(1700, 240) < 1700
    assert decay_toward_mean(1700, 100000) < 1501


def test_bo1_k_is_smaller_than_bo3():
    m1 = EloModel(); m1.step(_m(fmt="bo1", sa=16, sb=10))
    m3 = EloModel(); m3.step(_m(fmt="bo3", sa=2, sb=0))
    assert m1.ratings["A"] < m3.ratings["A"]


def test_parse_format_map_codes():
    assert parse_format("inf") == ("bo1", "Inferno")
    assert parse_format("BO3") == ("bo3", None)
    assert parse_format("") == ("bo1", None)


def test_team_normalization_case_and_space():
    assert team_id("  paiN  ") == team_id("PAIN")
    assert canonical_team("  paiN  ") == "paiN"


def test_alias_merges_navi():
    assert team_id("NAVI") == team_id("Natus Vincere")


def test_last_played_updates():
    model = EloModel()
    model.step(_m(d="2020-03-05"))
    assert model.last_played["A"] == date(2020, 3, 5)
