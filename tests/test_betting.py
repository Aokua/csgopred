"""下注建议数学测试。"""

import pytest

from csgopred.models.betting import (
    MAX_FRACTION, advice, expected_growth, kelly, simulate,
)


def test_kelly_zero_without_edge():
    assert kelly(0.5, 1.9) == 0.0


def test_kelly_matches_formula():
    # f = (p*o - 1) / (o - 1)
    assert kelly(0.6, 2.0) == pytest.approx(0.2, abs=1e-9)
    assert kelly(0.5, 2.0) == pytest.approx(0.0, abs=1e-9)


def test_kelly_capped_at_one():
    assert kelly(0.99, 10.0) == pytest.approx(0.99, abs=1e-2)


def test_expected_growth_positive_at_edge():
    assert expected_growth(0.6, 2.0, 0.1) > 0


def test_expected_growth_negative_when_overbetting():
    # 超过凯利比例后长期增长为负
    assert expected_growth(0.6, 2.0, 0.9) < 0


def test_advice_no_bet_when_odds_below_fair():
    r = advice(0.6, 1.5, 10000)
    assert r["verdict"] == "不下注"
    assert r["stake"] == 0.0
    assert "没有正期望" in r["reason"]


def test_advice_stake_positive_and_capped():
    r = advice(0.6, 2.2, 10000)
    assert r["stake"] > 0
    assert r["kelly_frac"] <= MAX_FRACTION
    assert r["stake"] == pytest.approx(10000 * r["kelly_frac"], abs=1.0)


def test_advice_conservative_uses_lower_bound():
    r = advice(0.6, 2.2, 10000, p_low=0.5)
    assert r["conservative_stake"] is not None
    assert r["conservative_stake"] < r["stake"]


def test_advice_warns_on_huge_edge():
    r = advice(0.8, 3.0, 10000)
    assert any("优势超过" in n for n in r["notes"])


def test_simulate_zero_fraction():
    r = simulate(0.6, 2.0, 0.0)
    assert r["ruin_prob"] == 0.0 and r["median_max_dd"] == 0.0
