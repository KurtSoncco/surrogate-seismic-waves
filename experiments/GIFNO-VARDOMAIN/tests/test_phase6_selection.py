"""The default checkpoint score stays the one that produced P128n128.

``--select-harness-sqrt`` is off unless the flag is passed. The loss default
stays ``log_l1``.
"""

from __future__ import annotations

import numpy as np
import pytest

from data import STAGE_UNIFIED
from evaluate import station_metrics
from train import build_parser, epoch_selection_score, selection_score


def _res() -> dict:
    return {"bins": {
        "iid_w500": {"log_l1": 0.10, "log_l1_overall_sqrt": 0.30},
        "iid_w1000": {"log_l1": 0.20, "log_l1_overall_sqrt": 0.40},
        "manifold_w500": {"log_l1": 0.20, "log_l1_overall_sqrt": 0.50},
        "manifold_w1000": {"log_l1": 0.40, "log_l1_overall_sqrt": 0.70},
        "manifold_near_w500": {"log_l1": 0.01, "log_l1_overall_sqrt": 0.01},
        "manifold_blind_w500": {"log_l1": 0.99, "log_l1_overall_sqrt": 0.99},
        "ood_dipping_w500": {"log_l1": 0.80, "log_l1_overall_sqrt": 0.80},
    }}


def test_flag_defaults_off_and_loss_stays_log_l1():
    args = build_parser().parse_args(["--cache", "unused"])
    assert args.loss == "log_l1"
    assert args.select_harness_sqrt is False
    assert args.protocol == "unified"
    turned = build_parser().parse_args(["--cache", "unused", "--select-harness-sqrt"])
    assert turned.select_harness_sqrt is True
    assert turned.loss == "log_l1"


def test_default_scores_ignore_the_sqrt_column_and_the_blind_bins():
    res = _res()
    unified = selection_score(res, STAGE_UNIFIED)
    assert unified == pytest.approx(0.30)
    assert epoch_selection_score(res, STAGE_UNIFIED, protocol="unified", harness_sqrt=False) == pytest.approx(unified)
    p128 = epoch_selection_score(res, STAGE_UNIFIED, protocol="p128", harness_sqrt=False)
    assert p128 == pytest.approx(0.5 * (0.15 + 0.30))
    stage0 = epoch_selection_score(res, 0, protocol="unified", harness_sqrt=False)
    assert stage0 == pytest.approx(selection_score(res, 0))
    assert stage0 == pytest.approx(0.15)


def test_opt_in_uses_the_sqrt_column_on_the_same_families():
    res = _res()
    unified = epoch_selection_score(res, STAGE_UNIFIED, protocol="unified", harness_sqrt=True)
    assert unified == pytest.approx(0.60)
    p128 = epoch_selection_score(res, STAGE_UNIFIED, protocol="p128", harness_sqrt=True)
    assert p128 == pytest.approx(0.5 * (0.35 + 0.60))
    stage0 = epoch_selection_score(res, 0, protocol="unified", harness_sqrt=True)
    assert stage0 == pytest.approx(0.35)


def test_sqrt_overall_matches_the_harness_and_is_absent_by_default():
    from analysis.score_harness import weighted_overall_g_log_l1

    freq = np.array([0.2, 1.0, 8.0])
    log_true = np.zeros((2, 3))
    log_pred = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    plain = station_metrics(log_pred, log_true, freq)
    assert "log_l1_overall_sqrt" not in plain
    weighted = station_metrics(log_pred, log_true, freq, sqrt_overall=True)
    expected = weighted_overall_g_log_l1(log_pred, log_true, freq)
    np.testing.assert_allclose(weighted["log_l1_overall_sqrt"], expected)
    unweighted = plain["log_l1"]
    assert weighted["log_l1_overall_sqrt"][0] > unweighted[0]
