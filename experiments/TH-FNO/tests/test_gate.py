"""Hard gate: dip=0 and σ²=0 ⇒ g=0 and pred == H_1D(trend) (AGENTS §1.1)."""

from __future__ import annotations

import numpy as np
import torch

import config
from context_features import residual_gate_scalar, residual_gate_torch
from model import GatedDeltaModel


def test_gate_exact_zero():
    assert residual_gate_scalar(0.0, 0.0) == 0.0
    g = residual_gate_torch(torch.zeros(3), torch.zeros(3))
    assert torch.all(g == 0)


def test_pred_equals_trend_when_gate_zero():
    freq = np.logspace(-1, 1, 32)
    model = GatedDeltaModel(
        in_channels=6,
        latent_channels=16,
        n_freq=32,
        nx=config.NX,
        fno_modes=(8, 8),
        num_fno_layers=1,
        deeponet_dim=8,
        use_fourier=True,
        n_fourier=4,
        physics_dim=3,
        residual_mode="additive",
        freq=freq,
    )
    B = 2
    x = torch.randn(B, 6, 32, config.NX)
    trend = torch.randn(B, config.NX, 32).abs() + 0.1
    cov = torch.zeros(B)
    dip = torch.zeros(B)
    physics = torch.randn(B, 3)
    with torch.no_grad():
        y = model(x, trend, cov, dip, physics=physics)
    assert torch.allclose(y, trend, atol=1e-6)


def test_log_mult_gate_zero_equals_trend():
    freq = np.logspace(-1, 1, 16)
    model = GatedDeltaModel(
        in_channels=6,
        latent_channels=8,
        n_freq=16,
        nx=config.NX,
        fno_modes=(4, 4),
        num_fno_layers=1,
        deeponet_dim=4,
        use_fourier=False,
        physics_dim=0,
        residual_mode="log_mult",
        freq=freq,
    )
    trend = torch.ones(1, config.NX, 16) * 2.0
    with torch.no_grad():
        y = model(torch.randn(1, 6, 16, config.NX), trend, torch.zeros(1), torch.zeros(1))
    assert torch.allclose(y, trend, atol=1e-6)
