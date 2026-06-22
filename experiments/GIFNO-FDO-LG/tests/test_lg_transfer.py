"""Transfer-learning tests for GIFNO-FDO-LG."""

from __future__ import annotations

import torch

from model import create_model
from transfer import apply_train_phase, l2sp_penalty, load_xt_pretrained


def test_apply_train_phase_freezes_fno_in_phase1():
    model = create_model(latent_channels=32, deeponet_dim=16, n_freq=64)
    apply_train_phase(model, phase=1)
    for name, p in model.named_parameters():
        if (
            name.startswith("lift.")
            or name.startswith("fno.")
            or name.startswith("head.")
        ):
            assert not p.requires_grad, name
        if name.startswith(("lift_unet.", "unet.", "fusion.")):
            assert p.requires_grad, name


def test_partial_xt_load_from_lg_model(tmp_path):
    """Simulate XT checkpoint by saving FNO+head keys from a fresh LG model."""
    model = create_model(latent_channels=32, deeponet_dim=16, n_freq=64)
    xt_keys = {
        k: v
        for k, v in model.state_dict().items()
        if k.startswith(("lift.", "fno.", "head."))
    }
    ckpt = tmp_path / "fake_xt.pt"
    torch.save(xt_keys, ckpt)

    model2 = create_model(latent_channels=32, deeponet_dim=16, n_freq=64)
    torch.nn.init.constant_(model2.unet.enc1.net[0].weight, 99.0)
    anchor = load_xt_pretrained(model2, ckpt)
    assert "lift.conv.0.weight" in anchor
    assert model2.lift.conv[0].weight.abs().mean() < 99.0


def test_l2sp_penalty_skips_frozen_complex_fno(tmp_path):
    """L2-SP must not sum complex FNO spectral weights when FNO is frozen."""
    model = create_model(latent_channels=32, deeponet_dim=16, n_freq=64)
    xt_keys = {
        k: v
        for k, v in model.state_dict().items()
        if k.startswith(("lift.", "fno.", "head."))
    }
    anchor = {k: v.detach().clone() for k, v in xt_keys.items()}
    apply_train_phase(model, phase=1)
    pen = l2sp_penalty(model, anchor, weight=1.0)
    assert not torch.is_complex(pen)
    assert float(pen) == 0.0


def test_l2sp_penalty_zero_at_anchor():
    model = create_model(latent_channels=32, deeponet_dim=16, n_freq=64)
    anchor = {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
        if k.startswith("lift.")
    }
    pen = l2sp_penalty(model, anchor, weight=1.0)
    assert float(pen) == 0.0
