"""Forward-shape tests for GIFNO-FDO-XT."""

from __future__ import annotations

import numpy as np
import torch

import config
from model import GIFNOXTModel, create_model


def test_create_model_output_shape():
    model = create_model(n_freq=64, deeponet_dim=32)
    x = torch.randn(2, config.IN_CHANNELS, config.NZ_MAX, config.NX)
    y = model(x)
    assert y.shape == (2, config.NX, 64)


def test_recorder_columns_nonzero():
    rec = config.recorder_x_indices()
    freq = np.logspace(-1, 1, 128)
    model = GIFNOXTModel(
        in_channels=4,
        latent_channels=32,
        n_freq=128,
        nx=config.NX,
        fno_modes=(16, 16),
        num_fno_layers=2,
        deeponet_dim=16,
        branch_mode="surface",
        x_coord_mode="normalized",
        recorder_x=rec,
        freq=freq,
    )
    x = torch.randn(1, 4, config.NZ_MAX, config.NX)
    y = model(x)
    off = np.setdiff1d(np.arange(config.NX), rec)
    assert torch.all(y[0, off, :] == 0)
    assert torch.any(y[0, rec[0], :] != 0)


def test_depth_branch_mode():
    rec = config.recorder_x_indices()
    freq = np.logspace(-1, 1, 64)
    model = GIFNOXTModel(
        in_channels=4,
        latent_channels=32,
        n_freq=64,
        nx=config.NX,
        fno_modes=(16, 16),
        num_fno_layers=2,
        deeponet_dim=16,
        branch_mode="depth",
        x_coord_mode="normalized",
        recorder_x=rec,
        freq=freq,
    )
    x = torch.randn(1, 4, config.NZ_MAX, config.NX)
    assert model(x).shape == (1, config.NX, 64)


def test_output_activation_none_allows_negative():
    """Baseline (no softplus) can produce negatives — used by legacy checkpoints."""
    rec = config.recorder_x_indices()
    freq = np.logspace(-1, 1, 32)
    model = GIFNOXTModel(
        in_channels=4,
        latent_channels=16,
        n_freq=32,
        nx=config.NX,
        fno_modes=(8, 8),
        num_fno_layers=1,
        deeponet_dim=8,
        branch_mode="surface",
        output_activation="none",
        recorder_x=rec,
        freq=freq,
    )
    # Force a forward; just check shape / finite.
    y = model(torch.randn(1, 4, 16, config.NX))
    assert torch.isfinite(y).all()


def test_x_norm_domain_based_and_symmetric():
    rec = config.recorder_x_indices()
    x_norm = config.recorder_x_trunk_coords(rec, mode="normalized")
    half_w = config.domain_half_width_m()
    assert x_norm.shape == rec.shape
    assert np.isclose(x_norm[0], -x_norm[-1], atol=1e-5)
    # Outermost recorders at ±150 m; domain half-width is 250 m → |x_norm| = 0.6
    assert np.isclose(abs(x_norm[0]), 150.0 / half_w)
    assert np.isclose(abs(x_norm[-1]), 150.0 / half_w)
    assert np.isclose(abs(x_norm).max(), 150.0 / half_w)
    # Domain edges (columns 0 and nx-1) would be exactly ±1
    center = config.NX // 2
    left_edge = ((0 - center) * config.DX) / half_w
    right_edge = ((config.NX - 1 - center) * config.DX) / half_w
    assert np.isclose(left_edge, -1.0)
    assert np.isclose(right_edge, 249.0 / half_w)


def test_x_meters_mode():
    rec = config.recorder_x_indices()
    x_m = config.recorder_x_trunk_coords(rec, mode="meters")
    center_idx = np.argmin(np.abs(rec - config.NX // 2))
    assert np.isclose(x_m[center_idx], 0.0)
