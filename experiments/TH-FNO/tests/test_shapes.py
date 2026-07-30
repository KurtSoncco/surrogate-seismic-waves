"""Live shape probes (AGENTS §8) — RV + capability; GIFNO when mounted."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

import config
from losses_th import masked_smooth_l1
from model import create_model

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass


def test_shapes_agreed_constants():
    assert config.N_LATERAL == config.SHAPE_N_LATERAL == 21
    assert config.N_FREQ == config.SHAPE_N_FREQ == 1000
    assert config.NX == config.SHAPE_NX_STRIP == 500
    assert config.NZ_MAX == config.SHAPE_NZ_MAX == 128
    rec = config.recorder_x_indices()
    assert len(rec) == 21


def test_create_model_forward():
    model = create_model(n_freq=64, latent_channels=16, deeponet_dim=8)
    x = torch.randn(1, 6, config.NZ_MAX, config.NX)
    h = torch.ones(1, config.NX, 64)
    phys = torch.tensor([[0.2, 50.0, 20.0]])
    y = model(x, h, torch.tensor([0.2]), torch.tensor([0.1]), physics=phys)
    assert y.shape == (1, config.NX, 64)
    assert torch.isfinite(y).all()


def test_smooth_l1_runs():
    pred = torch.rand(2, config.NX, 32) + 0.1
    target = torch.rand(2, config.NX, 32) + 0.1
    mask = torch.zeros(2, config.NX)
    mask[:, config.NX // 2] = 1.0
    loss = masked_smooth_l1(pred, target, mask)
    assert torch.isfinite(loss)


def test_central_strip_defaults_to_bc_window():
    sl = config.central_strip_slice()
    assert sl.start == config.BC_WIDTH == 500
    assert sl.stop == config.BC_WIDTH + config.NX == 1000
    # Wider crop still centered — strip-extent OOD helper
    wide = config.central_strip_slice(700)
    assert wide.start == (1500 - 700) // 2
    assert wide.stop - wide.start == 700


def test_rv_opensees_shapes():
    path = config.RV_H5_DIR / "run_26320.h5"
    if not path.is_file():
        pytest.skip("RV H5 not available")
    with h5py.File(path, "r") as f:
        vs = f["Vs_field"]
        assert vs.ndim == 2
        assert vs.shape[1] == config.SHAPE_NX_FULL
        af = f["transfer_function"]["AF"]
        assert af.shape == (config.SHAPE_N_FREQ,)
        freq = f["transfer_function"]["freq"]
        assert freq.shape == (config.SHAPE_N_FREQ,)


def test_capability_three_layer_shapes():
    root = config.CAPABILITY_ROOT / "three_layer" / "h5"
    cases = sorted(root.glob("case_*.h5")) if root.is_dir() else []
    if not cases:
        pytest.skip("three_layer H5 not available")
    with h5py.File(cases[0], "r") as f:
        vs = f["Vs_realization_2D"]
        assert vs.ndim == 2
        assert vs.shape[1] == config.SHAPE_NX_FULL


def test_gifno_cache_shapes_if_present():
    if not config.TF_PER_SAMPLE_PATH.is_file():
        pytest.skip("GIFNO TF cache not mounted")
    tf = np.load(config.TF_PER_SAMPLE_PATH, mmap_mode="r")
    assert tf.ndim == 3
    assert tf.shape[1] == config.SHAPE_N_LATERAL
    assert tf.shape[2] == config.SHAPE_N_FREQ
