"""Unit tests for radial binned spectral loss."""

import torch

from radial_spectral_loss import RadialBinnedSpectralLoss


def test_radial_loss_scalar_on_matching_inputs():
    loss_fn = RadialBinnedSpectralLoss()
    pred = torch.randn(4, 21, 128)
    target = pred.clone()
    out = loss_fn(pred, target)
    assert out.ndim == 0
    assert out.item() == 0.0


def test_radial_loss_positive_on_mismatch():
    loss_fn = RadialBinnedSpectralLoss()
    pred = torch.zeros(2, 21, 128)
    target = torch.ones(2, 21, 128)
    out = loss_fn(pred, target)
    assert out.item() > 0.0
