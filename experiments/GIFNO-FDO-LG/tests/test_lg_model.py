"""Forward-shape tests for GIFNO-FDO-LG."""

from __future__ import annotations

import numpy as np
import torch

import config
from model import GIFNOLGModel, create_model


def test_create_model_output_shape():
    model = create_model(n_freq=64, deeponet_dim=32, latent_channels=32)
    x = torch.randn(2, config.IN_CHANNELS, config.NZ_MAX, config.NX)
    y = model(x)
    assert y.shape == (2, config.NX, 64)


def test_recorder_columns_nonzero():
    rec = config.recorder_x_indices()
    freq = np.logspace(-1, 1, 128)
    model = GIFNOLGModel(
        in_channels=4,
        latent_channels=32,
        n_freq=128,
        nx=config.NX,
        fno_modes=(16, 16),
        num_fno_layers=2,
        deeponet_dim=16,
        unet_base_channels=16,
        fusion_mode="concat",
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


def test_gated_fusion_shape():
    rec = config.recorder_x_indices()
    model = GIFNOLGModel(
        in_channels=4,
        latent_channels=32,
        n_freq=64,
        nx=config.NX,
        fno_modes=(16, 16),
        num_fno_layers=2,
        deeponet_dim=16,
        unet_base_channels=16,
        fusion_mode="gated",
        recorder_x=rec,
        freq=np.logspace(-1, 1, 64),
    )
    y = model(torch.randn(1, 4, config.NZ_MAX, config.NX))
    assert y.shape == (1, config.NX, 64)
