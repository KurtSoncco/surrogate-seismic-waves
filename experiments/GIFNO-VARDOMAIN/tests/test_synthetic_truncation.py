"""Synthetic 1D records through the label pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "case7a"))

from labels import tf_over_input  # noqa: E402
from synthetic_truncation import _ttf, slowest_columns, truncation_bias  # noqa: E402


@pytest.fixture(scope="module")
def ttf():
    try:
        return _ttf()
    except Exception as exc:  # seiskit TTF not on this machine
        pytest.skip(f"no seiskit TTF: {exc}")


@pytest.fixture(scope="module")
def column():
    vs = np.r_[np.linspace(150.0, 350.0, 60), np.full(20, 1200.0)]
    rock = np.r_[np.zeros(60, bool), np.ones(20, bool)]
    return vs, rock


def test_steady_label_matches_the_exact_tf_and_bias_dies_out(ttf, column):
    vs, rock = column
    res = truncation_bias(vs, rock, (0.03, 0.005), 0.8, 2.4, dz=1.0, t_grid=(5.0, 15.0, 40.0), ttf=ttf)
    assert bool(res["reference_steady"]) and res["tail_ratio"] < 1e-8
    assert np.abs(res["pipeline_minus_exact"]).max() < 0.02
    worst = np.abs(res["bias"]).max(axis=1)
    assert worst[0] > worst[1] > worst[2] and worst[2] < 1e-3


def test_fixed_window_rejects_a_longer_record(ttf):
    x = np.zeros(2000)
    with pytest.raises(ValueError):
        tf_over_input(x[:, None], x, 0.01, ttf, window_s=10.0)


def test_slowest_column_is_the_deepest_soil():
    vs = np.full((10, 5), 200.0)
    rock = np.zeros((10, 5), dtype=bool)
    rock[4:, :] = True
    rock[8:, 1] = True
    rock[4:8, 1] = False
    assert 1 in slowest_columns(vs, rock, 1.0)
