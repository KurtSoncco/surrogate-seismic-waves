# tests/test_model.py
"""Forward-shape tests for GIFNO-FDO."""

from __future__ import annotations

import numpy as np
import torch

import config
from model import GIFNOFDOModel, create_model


def test_create_model_output_shape():
    model = create_model(n_freq=64, deeponet_dim=32)
    x = torch.randn(2, config.IN_CHANNELS, config.NZ_MAX, config.NX)
    y = model(x)
    assert y.shape == (2, config.NX, 64)


def test_recorder_columns_nonzero():
    rec = config.recorder_x_indices()
    freq = np.logspace(-1, 1, 128)
    model = GIFNOFDOModel(
        in_channels=4,
        latent_channels=32,
        n_freq=128,
        nx=config.NX,
        fno_modes=(16, 16),
        num_fno_layers=2,
        deeponet_dim=16,
        branch_mode="surface",
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
    model = GIFNOFDOModel(
        in_channels=4,
        latent_channels=32,
        n_freq=64,
        nx=config.NX,
        fno_modes=(16, 16),
        num_fno_layers=2,
        deeponet_dim=16,
        branch_mode="depth",
        recorder_x=rec,
        freq=freq,
    )
    x = torch.randn(1, 4, config.NZ_MAX, config.NX)
    assert model(x).shape == (1, config.NX, 64)
