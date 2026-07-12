"""Forward-shape tests for GIFNO-FDO-XT-HFNO."""

from __future__ import annotations

import numpy as np
import torch

import config


def test_create_model_output_shape():
    from model import create_model

    model = create_model(n_freq=64, deeponet_dim=32, num_fno_layers=2)
    x = torch.randn(2, config.IN_CHANNELS, config.NZ_MAX, config.NX)
    y = model(x)
    assert y.shape == (2, config.NX, 64)


def test_recorder_columns_nonzero():
    from model import create_model

    rec = config.recorder_x_indices()
    model = create_model(n_freq=128, deeponet_dim=16, num_fno_layers=2)
    x = torch.randn(1, 4, config.NZ_MAX, config.NX)
    y = model(x)
    off = np.setdiff1d(np.arange(config.NX), rec)
    assert torch.all(y[0, off, :] == 0)
    assert torch.any(y[0, rec[0], :] != 0)
