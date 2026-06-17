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


def test_x_norm_symmetric_and_unit():
    rec = config.recorder_x_indices()
    x_norm = config.recorder_x_trunk_coords(rec, mode="normalized")
    assert x_norm.shape == rec.shape
    assert np.isclose(np.max(np.abs(x_norm)), 1.0)
    assert np.isclose(x_norm[0], -x_norm[-1], atol=1e-5)


def test_x_meters_mode():
    rec = config.recorder_x_indices()
    x_m = config.recorder_x_trunk_coords(rec, mode="meters")
    center_idx = np.argmin(np.abs(rec - config.NX // 2))
    assert np.isclose(x_m[center_idx], 0.0)
