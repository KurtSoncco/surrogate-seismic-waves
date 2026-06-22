"""Forward-shape tests for GIFNO-FDO-XT-HFS."""

from __future__ import annotations

import torch

import config


def test_create_model_output_shape():
    from model import create_model

    model = create_model(n_freq=64, deeponet_dim=32, num_fno_layers=2)
    x = torch.randn(2, config.IN_CHANNELS, config.NZ_MAX, config.NX)
    y = model(x)
    assert y.shape == (2, config.NX, 64)


def test_hfs_after_fno_only():
    from model import create_model

    model = create_model(n_freq=64, num_fno_layers=2)
    assert hasattr(model, "hfs")
    assert hasattr(model.fno, "fno")
