"""Offline autoencoder: mask, rank-1 curves, and the frequency report."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import ColumnDepthAutoencoder  # noqa: E402
from pretrain_ae import (  # noqa: E402
    depth_masked_l1,
    fit_report,
    pod_holdout_mae,
    spectral_holdout_mae,
    train_spectral,
    wide_grid,
)


def test_mask_zero_row_does_not_change_the_depth_loss():
    torch.manual_seed(0)
    model = ColumnDepthAutoencoder(2)
    values = torch.randn(1, 2, 12, 3)
    mask = torch.ones(1, 12, 3)
    mask[:, 9:] = 0
    depth = torch.linspace(0, 1, 12).view(1, 1, 12, 1).expand(1, 1, 12, 3)
    _surface, recon = model(values, mask, depth)
    loss = depth_masked_l1(recon, values, mask)
    shifted = values.clone()
    shifted[:, :, 9:] += 4
    _surface2, recon2 = model(shifted, mask, depth)
    loss2 = depth_masked_l1(recon2, shifted, mask)
    assert torch.allclose(loss, loss2)
    assert torch.allclose(recon[:, :, :9], recon2[:, :, :9])


def test_rank1_curve_family_is_reconstructed_by_the_autoencoder_and_pod():
    freq = np.geomspace(0.1, 10.0, 24)
    mode = np.sin(np.log(freq))
    mode = mode / np.linalg.norm(mode)
    amplitude = np.linspace(-1.5, 1.5, 40)
    curves = amplitude[:, None] * mode[None, :]
    train, hold = curves[:32], curves[32:]
    pod = pod_holdout_mae(train, hold, 1)
    assert pod.shape == freq.shape
    assert float(pod.mean()) < 1e-4
    encoder, decoder = train_spectral(
        train, freq, 1, epochs=80, batch=16, lr=1e-3, hidden=64, seed=0, device=torch.device("cpu"),
    )
    mae = spectral_holdout_mae(encoder, decoder, hold, freq, torch.device("cpu"), 16)
    assert mae.shape == freq.shape
    assert float(mae.mean()) < 0.05


def test_report_scores_each_label_frequency_and_stores_an_unscored_wide_query(tmp_path):
    freq = np.geomspace(0.1, 10.0, 8)
    rng = np.random.default_rng(0)
    curves = rng.normal(size=(20, freq.size))
    columns = [
        (rng.uniform(100.0, 400.0, size=5), rng.uniform(0.01, 0.05, size=5))
        for _ in range(6)
    ]
    report = fit_report(
        curves, freq, columns[:5], columns[5:],
        ranks=(4,), epochs=1, batch=4, lr=1e-3, seed=0, device=torch.device("cpu"), out=tmp_path,
    )
    block = report["spectral"]["4"]
    assert len(block["holdout_mae"]) == freq.size
    assert len(block["pod_holdout_mae"]) == freq.size
    assert np.isfinite(block["holdout_mae"]).all()
    assert np.isfinite(block["pod_holdout_mae"]).all()
    wide = wide_grid()
    assert wide[0] == np.float64(0.001)
    assert wide[-1] == np.float64(50.0)
    assert report["wide_freq_hz"][0] == 0.001
    assert report["wide_freq_hz"][-1] == 50.0
    assert len(block["wide_pred_mean"]) == wide.size
    assert "wide_mae" not in block
    assert "wide_holdout_mae" not in block
    assert "wide_error" not in report
    assert "holdout_masked_l1" in report["depth"]
    assert np.isfinite(report["depth"]["holdout_masked_l1"])
