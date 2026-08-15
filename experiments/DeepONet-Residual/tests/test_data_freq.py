from __future__ import annotations

import numpy as np

from data import build_trunk_queries, freq_screen_indices, trunk_feature_names


def test_freq_screen_unique_sorted():
    freq = np.logspace(-1, 1, 1000)
    idx = freq_screen_indices(freq, 50)
    assert len(idx) == 50
    assert np.all(np.diff(idx) > 0)
    full = freq_screen_indices(freq, 1000)
    assert len(full) == 1000


def test_append_serial_tf1d_adds_log_channel():
    from data import append_serial_tf1d

    trunk = np.ones((6, 4), dtype=np.float32)
    tf1d = np.full(6, np.e, dtype=np.float32)
    out = append_serial_tf1d(trunk, tf1d)
    assert out.shape == (6, 5)
    np.testing.assert_allclose(out[:, 4], 1.0, rtol=1e-5)


def test_build_trunk_full_shape():
    n_rec, n_f = 21, 50
    vs_col = np.full(n_rec, 200.0)
    rec = np.linspace(0, 499, n_rec)
    freq = np.logspace(-1, 1, n_f)
    sin_f = np.sin(np.linspace(0, 1, n_f))
    cos_f = np.cos(np.linspace(0, 1, n_f))
    names = trunk_feature_names("full")
    y = build_trunk_queries(
        vs_col=vs_col,
        H=40.0,
        recorder_x=rec,
        freq_s=freq,
        sin_f=sin_f,
        cos_f=cos_f,
        trunk_names=names,
    )
    assert y.shape == (n_rec * n_f, 4)
    assert np.isfinite(y).all()
