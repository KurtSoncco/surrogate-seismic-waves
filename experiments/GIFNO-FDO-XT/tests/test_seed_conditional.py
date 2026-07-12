"""Tests for seed-conditional XT: scale-split, dual-path, softplus, contrast loss."""

from __future__ import annotations

import numpy as np
import torch

import config
from data_loader import SeedGroupBatchSampler, _box_blur_2d, split_vs_macro_rf
from losses import (
    SeedAwareCompositeLoss,
    build_training_loss,
    seed_contrast_delta_loss,
    seed_sigma_ln_loss,
)
from model import GIFNOXTModel, create_model


def test_box_blur_preserves_shape_and_mean():
    vs = np.ones((32, 64), dtype=np.float32) * 3.0
    out = _box_blur_2d(vs, kernel=15)
    assert out.shape == vs.shape
    assert np.allclose(out, 3.0, atol=1e-5)


def test_split_vs_macro_rf_residual_sums():
    rng = np.random.default_rng(0)
    vs = rng.random((16, 32), dtype=np.float32) + 1.0
    macro, rf = split_vs_macro_rf(vs, kernel=7)
    assert np.allclose(macro + rf, vs, atol=1e-5)


def test_softplus_output_positive():
    rec = config.recorder_x_indices()
    freq = np.logspace(-1, 1, 32)
    model = GIFNOXTModel(
        in_channels=4,
        latent_channels=16,
        n_freq=32,
        nx=config.NX,
        fno_modes=(8, 8),
        num_fno_layers=1,
        deeponet_dim=8,
        branch_mode="depth",
        output_activation="softplus",
        recorder_x=rec,
        freq=freq,
        dual_path=False,
    )
    x = torch.randn(2, 4, 16, config.NX)
    y = model(x)
    assert y.shape == (2, config.NX, 32)
    assert torch.all(y[:, rec, :] >= 0)


def test_dual_path_forward_shape():
    rec = config.recorder_x_indices()
    freq = np.logspace(-1, 1, 32)
    model = GIFNOXTModel(
        in_channels=6,
        latent_channels=16,
        n_freq=32,
        nx=config.NX,
        fno_modes=(8, 8),
        num_fno_layers=1,
        deeponet_dim=8,
        branch_mode="depth",
        output_activation="softplus",
        recorder_x=rec,
        freq=freq,
        dual_path=True,
        fno_modes_rf=(8, 8),
        num_fno_layers_rf=1,
    )
    x = torch.randn(2, 6, 16, config.NX)
    y = model(x)
    assert y.shape == (2, config.NX, 32)
    assert torch.all(y[:, rec, :] >= 0)


def test_seed_contrast_zero_when_identical_pairs():
    b, nx, f = 4, config.NX, 16
    # Two pairs of identical sample_ids with matching pred/target deltas.
    target = torch.rand(b, nx, f).abs() + 0.1
    pred = target.clone()
    mask = torch.zeros(b, nx)
    rec = config.recorder_x_indices()
    mask[:, rec] = 1.0
    sids = torch.tensor([0, 0, 1, 1])
    loss = seed_contrast_delta_loss(pred, target, sids, mask)
    assert float(loss) < 1e-5


def test_seed_sigma_ln_positive_for_mismatched_spread():
    b, nx, f = 4, config.NX, 32
    rec = config.recorder_x_indices()
    mask = torch.zeros(b, nx)
    mask[:, rec] = 1.0
    # Truth varies across seeds; pred is constant -> sigma mismatch.
    base = torch.ones(1, nx, f)
    target = torch.cat([base * 1.0, base * 2.0, base * 1.0, base * 2.0], dim=0)
    pred = torch.ones(b, nx, f)
    sids = torch.tensor([0, 0, 1, 1])
    loss = seed_sigma_ln_loss(pred, target, sids, mask)
    assert float(loss) > 0


def test_seed_group_batch_sampler_pairs():
    # 3 sample_ids x 4 replicates
    sids = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    sampler = SeedGroupBatchSampler(sids, batch_size=4, seed=0, drop_last=True)
    batches = list(sampler)
    assert len(batches) >= 1
    for batch in batches:
        assert len(batch) == 4
        batch_sids = [sids[i] for i in batch]
        # Each sample_id should appear exactly twice in a well-formed pair batch.
        from collections import Counter

        counts = Counter(batch_sids)
        assert all(v == 2 for v in counts.values())
        assert len(counts) == 2


def test_seed_aware_loss_factory(monkeypatch):
    monkeypatch.setattr(config, "SEED_CONTRAST_WEIGHT", 0.1)
    monkeypatch.setattr(config, "SEED_SIGMA_LN_WEIGHT", 0.05)
    monkeypatch.setattr(config, "LOSS_H1_WEIGHT", 0.0)
    monkeypatch.setattr(config, "LOG_TF_LOSS", False)
    monkeypatch.setattr(config, "BAND_CURRICULUM", False)
    loss = build_training_loss()
    assert isinstance(loss, SeedAwareCompositeLoss)


def test_create_model_respects_dual_path_config(monkeypatch):
    monkeypatch.setattr(config, "SCALE_SPLIT_VS", True)
    monkeypatch.setattr(config, "DUAL_PATH_ENCODER", True)
    monkeypatch.setattr(config, "IN_CHANNELS", 6)
    monkeypatch.setattr(config, "OUTPUT_ACTIVATION", "softplus")
    monkeypatch.setattr(config, "BRANCH_MODE", "depth")
    monkeypatch.setattr(config, "LATENT_CHANNELS", 16)
    monkeypatch.setattr(config, "NUM_FNO_LAYERS", 1)
    monkeypatch.setattr(config, "NUM_FNO_LAYERS_RF", 1)
    monkeypatch.setattr(config, "FNO_MODES", (8, 8))
    monkeypatch.setattr(config, "FNO_MODES_RF", (8, 8))
    monkeypatch.setattr(config, "DEEPONET_LATENT_DIM", 8)
    model = create_model(n_freq=32)
    assert model.dual_path is True
    y = model(torch.randn(1, 6, 16, config.NX))
    assert y.shape == (1, config.NX, 32)
