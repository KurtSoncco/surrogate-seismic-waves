"""Shared-POD collocation: 21 stations are the loss, not the output grid."""

from __future__ import annotations

import numpy as np
import torch

# Imports resolve after conftest puts this experiment ahead of GIFNO.
import config
from losses import _collocation_curves
from train import window_consistency_loss  # noqa: E402


def test_collocation_curves_mix_iid_and_dense():
    pred = torch.ones(2, 8, 3)
    target = torch.arange(8, dtype=torch.float32).view(1, 8, 1).expand(2, 8, 3).clone()
    mask = torch.zeros(2, 8)
    mask[0, [2, 4, 6]] = 1.0
    mask[1, :] = 1.0
    p, t = _collocation_curves(pred, target, mask)
    assert p.shape == (2, 8, 3)
    assert torch.all(p[0, [2, 4, 6]] == 1)
    assert torch.all(p[0, [0, 1, 3, 5, 7]] == 0)
    assert torch.all(t[0, [0, 1, 3, 5, 7]] == 0)
    assert torch.all(p[1] == 1)


def test_dense_window_consistency_uses_column_overlap():
    nx, f = 10, 2
    shift_stations = 2
    shift_cols = int(round(shift_stations * config.LATERAL_SPACING_M / config.DX))
    a = torch.arange(nx, dtype=torch.float32).view(1, nx, 1).expand(1, nx, f) + 1
    b = torch.zeros(1, nx, f)
    b[:, : nx - shift_cols] = a[:, shift_cols:]
    paired = torch.stack([a[0], b[0]], dim=0)
    loss = window_consistency_loss(
        paired, torch.arange(3), shift_stations, dense=True
    )
    assert float(loss) < 1e-6


def test_shifted_valid_follows_source_slot_not_destination():
    from data_loader import _scatter_tf_to_grid, _shift_recorders, _shift_valid

    tf = np.zeros((5, 4), dtype=np.float32)
    tf[0] = 1.0
    tf[2] = 2.0
    valid = np.array([True, False, True, False, False])
    rec = np.array([10, 20, 30, 40, 50])
    tf_s, _ = _shift_recorders(tf, 2)
    valid_s = _shift_valid(valid, 2)
    # Destination-AND would keep slot 0 labeled while filling it from slot 2
    # (ok) *and* keep slot 2 labeled while filling it from unlabeled slot 4.
    target, mask = _scatter_tf_to_grid(tf_s, rec, nx=80, n_freq=4, valid=valid_s)
    assert mask[10] == 1.0
    assert float(target[10, 0]) == 2.0
    assert mask[20] == 0.0
    assert mask[30] == 0.0
    assert mask[40] == 0.0


def test_align_collocation_pads_iid_to_section_slots():
    from build_outcrop_dataset import _align_collocation

    iid_x = np.array([100, 250, 400])
    dense_x = np.array([100, 150, 200, 250, 300, 400])
    f = 4
    parts = [
        {
            "src": type("S", (), {"__truediv__": lambda self, n: None})(),
            "tf": np.ones((2, 3, f), dtype=np.float32),
            "tf1d": np.ones((2, 3, f), dtype=np.float32),
        },
        {
            "src": type("S", (), {"__truediv__": lambda self, n: None})(),
            "tf": 2 * np.ones((1, 6, f), dtype=np.float32),
            "tf1d": 2 * np.ones((1, 6, f), dtype=np.float32),
        },
    ]

    def fake_source_recorder(src, n_rec, fallback):
        return iid_x if n_rec == 3 else dense_x

    import build_outcrop_dataset as bod

    orig = bod._source_recorder_x
    bod._source_recorder_x = fake_source_recorder
    try:
        union, mixed = _align_collocation(parts, iid_x)
    finally:
        bod._source_recorder_x = orig
    assert mixed
    assert np.array_equal(union, dense_x)
    assert parts[0]["tf"].shape == (2, 6, f)
    assert parts[0]["collocation_valid"][0].sum() == 3
    assert parts[1]["collocation_valid"][0].all()
