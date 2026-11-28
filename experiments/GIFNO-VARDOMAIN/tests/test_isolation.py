"""Shipped Case 7 defaults and checkpoint paths stay untouched."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "case7a"))

from generate_deep import DEPTH_M, sample_deep  # noqa: E402
from train import BLOCKED_OUT, _assert_isolated  # noqa: E402


def test_section_generator_default_depth_is_still_150():
    import section_generator as sg

    assert sg.DEPTH_M == 150.0
    rows = sg.sample_rows(1, seed=0)
    assert rows[0]["depth_m"] == 150.0
    vs, _, _ = sg.section_field(rows[0])
    assert vs.shape[0] == 150


def test_csv_depth_m_builds_a_250_m_mesh():
    import section_generator as sg

    row = dict(sg.sample_rows(1, seed=1)[0])
    row["depth_m"] = 250.0
    row["soil_thickness_m"] = 200.0
    vs, _, _ = sg.section_field(row)
    assert vs.shape[0] == 250
    assert sg.DEPTH_M == 150.0


def test_generate_deep_restores_the_150_m_default():
    import section_generator as sg

    rows = sample_deep(1, seed=2)
    assert rows[0]["depth_m"] == DEPTH_M == 250.0
    assert 60.0 <= rows[0]["soil_thickness_m"] <= 220.0
    assert rows[0]["step_width_m"] == 40.0
    assert rows[0]["contact_blend_m"] == 25.0
    assert rows[0]["vs_smooth_m"] == 20.0
    assert rows[0]["max_vs_ratio"] == 3.0
    assert abs(rows[0]["feature_dh_m"]) <= 30.0
    assert sg.DEPTH_M == 150.0
    assert sg.RANGES["soil_thickness_m"] == (60.0, 120.0)
    assert sg.STEP_WIDTH_M == 10.0


def test_deep_row_caps_lateral_vs_jump():
    import section_generator as sg

    worst = 1.0
    for row in sample_deep(32, seed=7):
        vs, _, _ = sg.section_field(row)
        left, right = vs[:, :-1], vs[:, 1:]
        worst = max(worst, float(np.max(np.maximum(left, right) / np.maximum(np.minimum(left, right), 1.0))))
    assert worst <= 3.0 + 0.15


def test_default_section_field_keeps_the_10_m_step():
    import section_generator as sg

    row = sg.sample_rows(1, seed=3)[0]
    row["feature"] = 2
    row["feature_name"] = "step"
    a = sg.contact_depth(row, np.arange(0.5, 2000.0, 1.0))
    b = sg.contact_depth({**row, "step_width_m": 10.0}, np.arange(0.5, 2000.0, 1.0))
    np.testing.assert_allclose(a, b)


def test_train_refuses_shipped_checkpoint_dirs(tmp_path):
    for token in BLOCKED_OUT:
        with pytest.raises(SystemExit):
            _assert_isolated(tmp_path / token / "best")
    _assert_isolated(tmp_path / "checkpoints" / "vardomain" / "metric_s0")
