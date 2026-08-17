from __future__ import annotations

import torch

from model import build_model


def _forward_ok(encoder: str, residual_fno: bool) -> None:
    n_rec, n_freq, trunk_dim = 21, 16, 5
    model = build_model(
        "single",
        field_channels=3,
        stoch_dim=20,
        trunk_dim=trunk_dim,
        latent_dim=16,
        field_hidden=8,
        branch_hidden=16,
        trunk_hidden=16,
        trunk_layers=2,
        field_encoder=encoder,  # type: ignore[arg-type]
        residual_fno=residual_fno,
        n_rec=n_rec,
        fno_width=8,
        fno_n_modes=(4, 4),
        fno_n_layers=2,
        n_gno_layers=2,
    )
    fields = torch.randn(2, 3, 32, n_rec)
    stoch = torch.randn(2, 20)
    trunk = torch.randn(2, n_rec * n_freq, trunk_dim)
    out = model(fields, stoch, trunk)
    assert out.shape == (2, n_rec * n_freq)
    out.sum().backward()


def test_resunet_forward():
    _forward_ok("resunet", False)


def test_gno_forward():
    _forward_ok("gno", False)


def test_fno_forward():
    _forward_ok("resunet", True)


def test_gino_forward():
    _forward_ok("gno", True)


def test_ufno_forward():
    n_rec, n_freq, trunk_dim = 21, 16, 5
    model = build_model(
        "single",
        field_channels=3,
        stoch_dim=20,
        trunk_dim=trunk_dim,
        latent_dim=16,
        field_hidden=8,
        branch_hidden=16,
        trunk_hidden=16,
        trunk_layers=2,
        field_encoder="gno",
        residual_fno=True,
        n_rec=n_rec,
        fno_width=8,
        fno_n_modes=(4, 4),
        fno_n_layers=2,
        n_gno_layers=2,
        fno_kind="ufno",
    )
    out = model(torch.randn(2, 3, 32, n_rec), torch.randn(2, 20), torch.randn(2, n_rec * n_freq, trunk_dim))
    assert out.shape == (2, n_rec * n_freq)
    out.sum().backward()


def test_ffno_forward():
    n_rec, n_freq, trunk_dim = 21, 16, 5
    model = build_model(
        "single",
        field_channels=3,
        stoch_dim=20,
        trunk_dim=trunk_dim,
        latent_dim=16,
        field_hidden=8,
        branch_hidden=16,
        trunk_hidden=16,
        trunk_layers=2,
        field_encoder="gno",
        residual_fno=True,
        n_rec=n_rec,
        fno_width=8,
        fno_n_modes=(4, 4),
        fno_n_layers=2,
        n_gno_layers=2,
        fno_kind="ffno",
    )
    out = model(torch.randn(2, 3, 32, n_rec), torch.randn(2, 20), torch.randn(2, n_rec * n_freq, trunk_dim))
    assert out.shape == (2, n_rec * n_freq)
    out.sum().backward()


def test_attn_gino_forward():
    _forward_ok("attn", True)


def test_gat_gino_forward():
    _forward_ok("gat", True)


def _fno_kind_ok(kind: str) -> None:
    n_rec, n_freq, trunk_dim = 21, 16, 5
    model = build_model(
        "single",
        field_channels=3,
        stoch_dim=20,
        trunk_dim=trunk_dim,
        latent_dim=16,
        field_hidden=8,
        branch_hidden=16,
        trunk_hidden=16,
        trunk_layers=2,
        field_encoder="gno",
        residual_fno=True,
        n_rec=n_rec,
        fno_width=8,
        fno_n_modes=(4, 4),
        fno_n_layers=2,
        n_gno_layers=2,
        fno_kind=kind,  # type: ignore[arg-type]
    )
    out = model(
        torch.randn(2, 3, 32, n_rec),
        torch.randn(2, 20),
        torch.randn(2, n_rec * n_freq, trunk_dim),
    )
    assert out.shape == (2, n_rec * n_freq)
    out.sum().backward()


def test_afno_forward():
    _fno_kind_ok("afno")


def test_wno_forward():
    _fno_kind_ok("wno")


def test_fno1d_forward():
    _fno_kind_ok("fno1d")
