"""回测流程冒烟测试：验证无泄漏与结构完整。"""

import random

from csgopred.eval.backtest import walk_forward


def _synthetic(n=400, seed=7):
    rng = random.Random(seed)
    teams = [f"T{i}" for i in range(12)]
    strength = {t: rng.gauss(0, 1) for t in teams}
    rows = []
    day = 0
    for i in range(n):
        a, b = rng.sample(teams, 2)
        diff = strength[a] - strength[b]
        p = 1 / (1 + 10 ** (-diff))
        a_wins = rng.random() < p
        rows.append({
            "match_id": f"m{i:05d}",
            "start_date": f"2020-{1 + (day // 28) % 12:02d}-{1 + day % 28:02d}",
            "team_a": a, "team_b": b,
            "score_a": 2 if a_wins else rng.choice([0, 1]),
            "score_b": 0 if a_wins else 2,
            "format": "bo3", "event": "test", "map_name": None,
        })
        day += 1
    return rows


def test_walk_forward_structure():
    res = walk_forward(_synthetic(), split_ratio=0.7, min_train=100)
    assert res["train_n"] + res["test_n"] == 400
    assert 0 < res["model"]["log_loss"] < 1.0
    assert res["model"]["log_loss"] < 0.693  # 合成数据有真实信号，应优于瞎猜
    assert res["test_range"][0] >= res["train_range"][1]


def test_predictions_are_probabilities():
    res = walk_forward(_synthetic(600), split_ratio=0.7, min_train=100)
    for row in res["predictions"]:
        assert 0.0 < row["p"] < 1.0
        assert row["y"] in (0, 1)


def test_test_predictions_use_only_past():
    res = walk_forward(_synthetic(300), split_ratio=0.7, min_train=100)
    dates = [row["date"] for row in res["predictions"]]
    assert dates == sorted(dates)
