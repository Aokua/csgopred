"""M4 赔率数学测试（不依赖外部赔率数据）。"""

import math

import pytest

from csgopred.models.market import (
    Bet, brier, clv, devig_proportional, devig_shin, implied, kelly, log_loss, roi,
)


def test_implied():
    assert implied(2.0) == 0.5
    assert implied(0.5) == 0.0


def test_devig_proportional_sums_to_one():
    a, b = devig_proportional(1.5, 2.6)
    assert a + b == pytest.approx(1.0)
    assert a > b


def test_devig_shin_sums_to_one_and_favourite_higher():
    a, b = devig_shin(1.5, 2.6)
    assert a + b == pytest.approx(1.0, abs=1e-6)
    assert 0 < a < 1


def test_kelly_zero_without_edge():
    assert kelly(0.5, 1.9) == 0.0


def test_kelly_positive_with_edge():
    f = kelly(0.6, 2.0)
    assert f == pytest.approx(0.05, abs=1e-9)  # 0.25 * (0.6*2-1)/(2-1)


def test_clv_sign():
    # 模型公平赔率 1/0.6=1.67；收盘 1.5 比公平价更短 -> 市场向模型靠拢，正 CLV
    assert clv(0.6, 1.5) > 0
    # 收盘 2.0 比公平价更长 -> 市场不认同模型
    assert clv(0.6, 2.0) < 0


def test_roi_math():
    bets = [Bet(0.6, 2.0, True), Bet(0.6, 2.0, False)]
    res = roi(bets, threshold=0.0)
    assert res["n"] == 2
    assert res["pnl"] == pytest.approx(0.0, abs=1e-9)


def test_roi_threshold_filters():
    bets = [Bet(0.51, 2.0, True)]
    assert roi(bets, threshold=0.05)["n"] == 0


def test_log_loss_brier_bounds():
    assert log_loss([0.5, 0.5], [0, 1]) == pytest.approx(math.log(2), rel=1e-6)
    assert 0 <= brier([0.5, 0.5], [0, 1]) <= 1
