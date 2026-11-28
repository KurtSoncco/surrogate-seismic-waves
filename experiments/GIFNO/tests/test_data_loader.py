# tests/test_data_loader.py
"""Tests for GIFNO input normalization."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

GIFNO_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("GIFNO_DATA_ROOT", str(GIFNO_DIR / "dummy_data"))
if str(GIFNO_DIR) not in sys.path:
    sys.path.insert(0, str(GIFNO_DIR))

from data_loader import (  # noqa: E402
    GIFNODataset,
    _normalize_vs_by_surface,
    _normalize_zeta_by_max,
)


def test_normalize_vs_by_surface():
    vs = np.array(
        [
            [200.0, 400.0],
            [300.0, 600.0],
            [100.0, 200.0],
        ],
        dtype=np.float32,
    )
    out = _normalize_vs_by_surface(vs, eps=1e-6)
    assert np.allclose(out[0, :], 1.0)
    assert np.allclose(out[1, 0], 1.5)
    assert np.allclose(out[1, 1], 1.5)
    assert np.allclose(out[2, 0], 0.5)


def test_normalize_zeta_by_max():
    zeta = np.array(
        [
            [0.02, 0.04],
            [0.05, 0.10],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    out = _normalize_zeta_by_max(zeta, nz=2, eps=1e-12)
    assert np.allclose(out[0, 1], 0.4)
    assert np.allclose(out[1, 1], 1.0)
    assert out[2, 0] == 0.0


def test_raw_sample_caches_h5_fields():
    ds = GIFNODataset(
        manifest_rows=[{}],
        tf_array=np.zeros((1, 2, 4), dtype=np.float32),
        recorder_x_idx=np.array([0, 1]),
        cache_in_memory=True,
    )
    reads: list[int] = []

    def fake_read(self, idx):
        reads.append(idx)
        vs = np.ones((2, 8), dtype=np.float32)
        zeta = np.ones((2, 8), dtype=np.float32)
        return vs, zeta, 2, 1.0, 1.0, 1.0

    ds._read_h5_fields = fake_read.__get__(ds, GIFNODataset)
    a = ds.raw_sample(0)
    b = ds.raw_sample(0)
    assert reads == [0]
    assert a is b
