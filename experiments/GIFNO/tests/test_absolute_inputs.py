"""Physical domain-size channels so 250 m and 500 m windows do not alias."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from absolute_inputs import (
    CH_LOG_H,
    CH_LOG_LX,
    CH_MASK,
    CH_X,
    DEPTH_REF_M,
    N_ABS_CHANNELS,
    X_REF_M,
    encode_absolute,
    pad_lateral,
)


def _vs(nx: int, nz: int = 80, v: float = 200.0) -> np.ndarray:
    return np.full((nz, nx), v, dtype=np.float64)


def test_different_lx_do_not_share_an_x_channel():
    a = encode_absolute(_vs(250), np.full((80, 250), 0.02), 80, 160, dx=1.0, dz=1.0)
    b = encode_absolute(_vs(500), np.full((80, 500), 0.02), 80, 160, dx=1.0, dz=1.0)
    assert a.shape[0] == b.shape[0] == N_ABS_CHANNELS
    assert a[CH_X, 0, -1] < b[CH_X, 0, -1]
    assert abs(float(a[CH_LOG_LX, 0, 0]) - np.log(250 / X_REF_M)) < 1e-6
    assert abs(float(b[CH_LOG_LX, 0, 0])) < 1e-6


def test_different_h_sets_log_h():
    shallow = encode_absolute(_vs(100, nz=60), np.full((60, 100), 0.02), 60, 160)
    deep = encode_absolute(_vs(100, nz=150), np.full((150, 100), 0.02), 150, 160)
    assert shallow[CH_LOG_H, 0, 0] < deep[CH_LOG_H, 0, 0]
    assert abs(float(deep[CH_LOG_H, 0, 0]) - np.log(150 / DEPTH_REF_M)) < 1e-6


def test_pad_lateral_keeps_physics_and_zeros_the_mask():
    raw = encode_absolute(_vs(200), np.full((80, 200), 0.02), 80, 160)
    pad = pad_lateral(raw, 500)
    assert pad.shape == (N_ABS_CHANNELS, 160, 500)
    np.testing.assert_allclose(pad[..., :200], raw)
    assert np.all(pad[CH_MASK, :80, 200:] == 0)
    assert np.all(pad[CH_MASK, :80, :200] == 1)
