from __future__ import annotations

import numpy as np

from score_trunk_permutation import shuffle_trunk_channel, trunk_channel_names


def test_trunk_channel_names_serial():
    names = trunk_channel_names(serial=True, trunk_dim=5)
    assert names == ["f_star", "sin_f", "cos_f", "x_over_lambda", "log_tf1d"]


def test_trunk_channel_names_no_serial():
    names = trunk_channel_names(serial=False, trunk_dim=4)
    assert names == ["f_star", "sin_f", "cos_f", "x_over_lambda"]
    assert "log_tf1d" not in names


def test_shuffle_trunk_channel_keeps_other_dims():
    rng = np.random.default_rng(0)
    trunk = rng.normal(size=(8, 12, 5)).astype(np.float32)
    perm = rng.permutation(8)
    out = shuffle_trunk_channel(trunk, 2, perm)
    np.testing.assert_array_equal(out[:, :, 2], trunk[perm][:, :, 2])
    for ch in (0, 1, 3, 4):
        np.testing.assert_array_equal(out[:, :, ch], trunk[:, :, ch])
