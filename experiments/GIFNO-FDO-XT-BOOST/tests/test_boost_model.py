"""Tests for GIFNO-FDO-XT-BOOST."""

from __future__ import annotations

import torch

import config


def test_boost_forward_shape():
    from model import create_model

    model = create_model(n_freq=64, deeponet_dim=32, num_fno_layers=2)
    x = torch.randn(1, config.IN_CHANNELS, config.NZ_MAX, config.NX)
    y = model(x)
    assert y.shape == (1, config.NX, 64)


def test_load_xt_base_and_freeze(tmp_path):
    from model import create_model
    from transfer import freeze_base, load_xt_base

    model = create_model(n_freq=64, num_fno_layers=2, deeponet_dim=16)
    ckpt = tmp_path / "xt.pt"
    torch.save(model.base.state_dict(), ckpt)

    model.residual_head.pred_tower[0].weight.data.fill_(1.0)
    load_xt_base(model, ckpt)
    freeze_base(model)

    assert not any(p.requires_grad for p in model.base.parameters())
    assert any(p.requires_grad for p in model.residual_head.parameters())


def test_band_weighted_loss():
    from losses import BandWeightedMaskedLoss

    freq = torch.logspace(-1, 1, 64).numpy()
    loss_fn = BandWeightedMaskedLoss(p=1, h1_weight=0.25, freq=freq)
    mask = torch.ones(2, config.NX)
    loss = loss_fn(
        torch.zeros(2, config.NX, 64),
        torch.zeros(2, config.NX, 64),
        mask,
    )
    assert loss.ndim == 0
