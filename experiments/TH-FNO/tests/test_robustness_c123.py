"""Session N+1 robustness: soft clamp, zero-init head, loss-term norm."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

import config
from losses_th import RunningTermNorm, THFNOLoss
from model import GatedDeltaModel, create_model, soft_clamp_log_delta


def test_soft_clamp_bounds_correction_to_3x():
    c = math.log(3.0)
    huge = torch.tensor([100.0, -100.0, 0.0])
    out = soft_clamp_log_delta(huge, c)
    assert torch.allclose(out, torch.tensor([c, -c, 0.0]), atol=1e-5)
    # exp(Δ_eff) ∈ [1/3, 3]
    factor = torch.exp(out)
    assert float(factor[0]) <= 3.0 + 1e-5
    assert float(factor[1]) >= 1.0 / 3.0 - 1e-5


def test_log_mult_cannot_blow_past_3x_even_with_large_raw():
    freq = np.logspace(-1, 1, 16)
    model = GatedDeltaModel(
        in_channels=6,
        latent_channels=8,
        n_freq=16,
        nx=64,
        fno_modes=(4, 4),
        num_fno_layers=1,
        deeponet_dim=4,
        use_fourier=False,
        physics_dim=0,
        residual_mode="log_mult",
        log_delta_c=math.log(3.0),
        predict_mode="residual",
        freq=freq,
        recorder_x=np.array([0, 32, 63], dtype=np.int64),
    )
    # Force huge raw head by filling branch last layer
    with torch.no_grad():
        model.head.branch[-1].weight.fill_(10.0)
        model.head.branch[-1].bias.fill_(10.0)
    trend = torch.ones(1, 64, 16) * 2.0
    # Nonzero gate (cov, dip large)
    cov = torch.ones(1) * 1.0
    dip = torch.ones(1) * 1.0
    with torch.no_grad():
        y = model(torch.randn(1, 6, 16, 64), trend, cov, dip)
    ratio = (y / trend).clamp_min(1e-12)
    assert float(ratio.max()) <= 3.0 + 1e-4
    assert float(ratio.min()) >= 1.0 / 3.0 - 1e-4


def test_zero_init_residual_starts_at_trend():
    freq = np.logspace(-1, 1, 16)
    model = GatedDeltaModel(
        in_channels=6,
        latent_channels=8,
        n_freq=16,
        nx=64,
        fno_modes=(4, 4),
        num_fno_layers=1,
        deeponet_dim=4,
        use_fourier=False,
        physics_dim=0,
        residual_mode="log_mult",
        log_delta_c=math.log(3.0),
        predict_mode="residual",
        zero_init_residual_head=True,
        freq=freq,
        recorder_x=np.array([0, 32, 63], dtype=np.int64),
    )
    last = model.head.branch[-1]
    assert torch.count_nonzero(last.weight) == 0
    assert torch.count_nonzero(last.bias) == 0
    trend = torch.rand(2, 64, 16).abs() + 0.5
    cov = torch.ones(2) * 0.5
    dip = torch.ones(2) * 0.1
    with torch.no_grad():
        y = model(torch.randn(2, 6, 16, 64), trend, cov, dip)
    assert torch.allclose(y, trend, atol=1e-6)


def test_create_model_zero_inits_only_residual():
    prev = config.PREDICT_MODE
    try:
        config.PREDICT_MODE = "residual"
        m_res = create_model(n_freq=32, predict_mode="residual")
        assert torch.count_nonzero(m_res.head.branch[-1].weight) == 0
        config.PREDICT_MODE = "direct"
        m_dir = create_model(n_freq=32, predict_mode="direct")
        # Direct should NOT zero-init (would pin softplus(0)=ln2 everywhere)
        assert torch.count_nonzero(m_dir.head.branch[-1].weight) > 0
    finally:
        config.PREDICT_MODE = prev


def test_running_term_norm_equalizes_scales():
    norm_a = RunningTermNorm(momentum=0.0, eps=1e-8, enabled=True)  # momentum 0 → last
    norm_b = RunningTermNorm(momentum=0.0, eps=1e-8, enabled=True)
    # First call initializes; second call with momentum=0 uses current |term|
    t_small = torch.tensor(0.5)
    t_large = torch.tensor(80.0)
    _ = norm_a(t_small)
    _ = norm_b(t_large)
    n_small = norm_a(t_small)
    n_large = norm_b(t_large)
    assert abs(float(n_small) - 1.0) < 1e-5
    assert abs(float(n_large) - 1.0) < 1e-5


def test_loss_term_norm_makes_weighted_terms_oom_comparable():
    """After norm, raw magnitude disparity should not dominate λ."""
    prev = config.LOSS_TERM_NORM
    try:
        config.LOSS_TERM_NORM = True
        loss_fn = THFNOLoss(freq=torch.logspace(-1, 1, 64))
        B, nx, nf = 2, 64, 64
        # Mask only a few columns
        pred = torch.rand(B, nx, nf).abs() + 0.1
        target = pred + 0.05 * torch.randn(B, nx, nf)
        mask = torch.zeros(B, nx)
        mask[:, [0, 32, 63]] = 1.0
        # Warm up EMA a few steps
        for _ in range(5):
            _, parts = loss_fn(pred, target, mask)
        # Normalized reported terms should be O(λ), not O(80)
        assert parts["loss_spec"] < 5.0  # λ_spec=0.05 * ~1
        assert parts["loss_smooth_l1"] < 5.0
        # Raw still shows the old imbalance exists
        assert parts["loss_spec_raw"] > parts["loss_smooth_l1_raw"]
    finally:
        config.LOSS_TERM_NORM = prev


def test_gate_still_exact_with_soft_clamp():
    freq = np.logspace(-1, 1, 16)
    model = GatedDeltaModel(
        in_channels=6,
        latent_channels=8,
        n_freq=16,
        nx=32,
        fno_modes=(4, 4),
        num_fno_layers=1,
        deeponet_dim=4,
        use_fourier=False,
        physics_dim=0,
        residual_mode="log_mult",
        log_delta_c=math.log(3.0),
        predict_mode="residual",
        zero_init_residual_head=False,
        freq=freq,
        recorder_x=np.arange(0, 32, 8, dtype=np.int64),
    )
    with torch.no_grad():
        model.head.branch[-1].weight.normal_(0, 1)
        model.head.branch[-1].bias.normal_(0, 1)
    trend = torch.ones(1, 32, 16) * 1.7
    with torch.no_grad():
        y = model(
            torch.randn(1, 6, 16, 32),
            trend,
            torch.zeros(1),
            torch.zeros(1),
        )
    assert torch.allclose(y, trend, atol=1e-6)
