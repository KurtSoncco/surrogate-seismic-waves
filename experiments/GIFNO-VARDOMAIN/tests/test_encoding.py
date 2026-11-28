"""Metric vs reference encodings: same shape per family, different physics for W."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoding import (  # noqa: E402
    CH_LATERAL,
    CH_LOG_F1,
    CH_LOG_H,
    CH_LOG_LX,
    CH_MASK,
    CH_X,
    CONTEXT_M,
    N_CHANNELS,
    NX_METRIC,
    NX_REF,
    NZ_METRIC,
    NZ_REF,
    encode,
    station_columns,
)
from model import MaskedDepthCollapse, VarDomainLOGLO, fit_log_pod  # noqa: E402


def _field(nz: int, nx: int, vs: float = 200.0) -> tuple[np.ndarray, np.ndarray]:
    return np.full((nz, nx), vs), np.full((nz, nx), 0.02)


def test_metric_same_shape_different_lx_and_x():
    a = encode("metric", *_field(80, 500), 80)
    b = encode("metric", *_field(80, 1500), 80)
    assert a.shape == b.shape == (N_CHANNELS, NZ_METRIC, NX_METRIC)
    assert a[CH_X, 0, 100] == b[CH_X, 0, 100]
    assert a[CH_LOG_LX, 0, 0] < b[CH_LOG_LX, 0, 0]
    assert a[CH_LATERAL, 0, CONTEXT_M + 250] > 0.9
    assert a[CH_LATERAL, 0, CONTEXT_M + 750] < 0.1
    assert b[CH_LATERAL, 0, CONTEXT_M + 750] > 0.9
    assert a[CH_LOG_F1, 0, 0] == 0.0  # default f1 = 1 Hz, log(f1 / 1 Hz)


def test_reference_same_shape_and_unit_x():
    a = encode("reference", *_field(80, 500), 80)
    b = encode("reference", *_field(150, 1500), 150)
    assert a.shape == b.shape == (N_CHANNELS, NZ_REF, NX_REF)
    np.testing.assert_allclose(a[CH_X, 0], b[CH_X, 0], atol=1e-6)
    assert a[CH_LOG_LX, 0, 0] < b[CH_LOG_LX, 0, 0]
    assert a[CH_LOG_H, 0, 0] < b[CH_LOG_H, 0, 0]


def test_padded_row_does_not_enter_collapse():
    vs, zeta = _field(40, 500)
    vs[-1] = 9e3
    x = torch.from_numpy(encode("metric", vs, zeta, 40))
    collapse = MaskedDepthCollapse(4, NZ_METRIC)
    collapse.weight.data.fill_(1.0)
    mask = x[CH_MASK].unsqueeze(0)
    out = collapse(x[:4].unsqueeze(0), mask)
    edited = x.clone()
    edited[:, 200:, :] = 99.0
    out2 = collapse(edited[:4].unsqueeze(0), mask)
    torch.testing.assert_close(out, out2)


def test_head_reads_width_from_latent():
    modes, mean = fit_log_pod(np.random.default_rng(0).normal(size=(32, 8)).astype(np.float32), 4)
    model = VarDomainLOGLO("reference", modes, mean, n_modes_x=8, narrow=8, latent=16, n_layers=1, patch_w=20)
    x = torch.from_numpy(encode("reference", *_field(40, 700), 40)).unsqueeze(0)
    cols = torch.tensor([[0, 10, 20]])
    y = model(x, cols)
    assert y.shape == (1, 3, 8)
    assert torch.isfinite(y).all()


def test_metric_canvas_matches_the_padded_absolute_encoding():
    rng = np.random.default_rng(1)
    vs = rng.uniform(80.0, 2500.0, size=(90, 700))
    zeta = rng.uniform(0.005, 0.08, size=(90, 700))
    vs[10, 4] = np.nan
    got = encode("metric", vs, zeta, 80, f1_hz=1.7, f2_hz=5.1)
    left, right = CONTEXT_M, NX_METRIC - CONTEXT_M - vs.shape[1]
    vs_p = np.pad(np.asarray(vs, dtype=np.float64), ((0, 0), (left, right)), mode="edge")
    zeta_p = np.pad(np.asarray(zeta, dtype=np.float64), ((0, 0), (left, right)), mode="edge")
    from absolute_inputs import encode_absolute

    enc = encode_absolute(vs_p, zeta_p, 80, NZ_METRIC, lx=float(vs.shape[1]))
    from encoding import smoothed_lateral_mask

    lateral = smoothed_lateral_mask(vs.shape[1], NX_METRIC, origin=CONTEXT_M).astype(np.float32)
    freq = np.stack([
        np.full((NZ_METRIC, NX_METRIC), np.log(1.7), np.float32),
        np.full((NZ_METRIC, NX_METRIC), np.log(5.1), np.float32),
    ])
    ref = np.concatenate([enc, np.broadcast_to(lateral, (1, NZ_METRIC, NX_METRIC)), freq], axis=0)
    np.testing.assert_array_equal(got, ref)


def test_station_columns_metric_are_metres():
    cols = station_columns("metric", np.array([0.0, 249.0, 499.0]), width=500)
    np.testing.assert_array_equal(cols, CONTEXT_M + np.array([0, 249, 499]))
    cols = station_columns("reference", np.array([0.0, 250.0, 499.0]), width=500)
    assert cols[0] == 0 and cols[-1] == NX_REF - 1
