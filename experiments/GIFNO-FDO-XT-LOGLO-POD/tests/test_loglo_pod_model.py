"""Forward-shape tests for GIFNO-FDO-XT-LOGLO-POD."""

from __future__ import annotations

import numpy as np
import torch

import config


def test_create_model_output_shape():
    from model import create_model

    model = create_model(num_fno_layers=2)
    x = torch.randn(2, config.IN_CHANNELS, config.NZ_MAX, config.NX)
    y = model(x)
    assert y.shape == (2, config.NX, config.N_FREQ)


def test_recorder_columns_nonzero():
    from model import create_model

    rec = config.recorder_x_indices()
    model = create_model(num_fno_layers=2)
    x = torch.randn(1, 4, config.NZ_MAX, config.NX)
    y = model(x)
    off = np.setdiff1d(np.arange(config.NX), rec)
    assert torch.all(y[0, off, :] == 0)
    assert torch.any(y[0, rec[0], :] != 0)


def test_dual_path_encoder_returns_two_streams():
    from spectral_layers import DualPathLOGLOStack

    enc = DualPathLOGLOStack(
        n_modes=(8, 8),
        channels=16,
        n_layers=2,
        patch_size=(16, 20),
    )
    x = torch.randn(2, 16, config.NZ_MAX, config.NX)
    x_g, x_l = enc(x)
    assert x_g.shape == x_l.shape == (2, 16, config.NZ_MAX, config.NX)
