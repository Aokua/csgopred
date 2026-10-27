"""指标层单元测试。"""

import math

from csgopred.eval.metrics import (
    accuracy,
    brier,
    ece,
    log_loss,
    reliability_table,
    summarize,
)


def test_log_loss_at_half_is_ln2():
    assert math.isclose(log_loss([0.5, 0.5], [0, 1]), math.log(2), rel_tol=1e-6)


def test_log_loss_perfect_is_near_zero():
    assert log_loss([0.999999, 0.000001], [1, 0]) < 0.01


def test_brier_perfect_is_zero():
    assert brier([1.0, 0.0], [1, 0]) == 0.0


def test_accuracy_threshold():
    assert accuracy([0.6, 0.4, 0.5], [1, 0, 1]) == 1.0


def test_ece_is_zero_when_perfectly_calibrated():
    ps = [0.2] * 100 + [0.8] * 100
    ys = [1] * 20 + [0] * 80 + [1] * 80 + [0] * 20
    assert ece(ps, ys) < 0.05


def test_reliability_table_covers_all_bins():
    rows = reliability_table([0.05, 0.15, 0.95], [0, 0, 1])
    assert sum(r["n"] for r in rows) == 3


def test_summarize_shape():
    s = summarize([0.5, 0.5], [0, 1])
    assert set(s) == {"name", "n", "log_loss", "brier", "accuracy", "ece", "max_gap"}
