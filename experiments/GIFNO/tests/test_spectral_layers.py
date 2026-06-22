"""Unit tests for spectral_layers."""

from __future__ import annotations

import torch

from spectral_layers import (
    DepthwiseLocalConv2d,
    FNOEncoderLoop,
    HFSModule,
    HybridFNOStack,
)


def test_depthwise_local_conv_shape():
    m = DepthwiseLocalConv2d(16)
    x = torch.randn(2, 16, 32, 64)
    assert m(x).shape == x.shape


def test_hybrid_fno_stack_shape():
    m = HybridFNOStack(n_modes=(8, 8), channels=16, n_layers=3)
    x = torch.randn(2, 16, 32, 64)
    assert m(x).shape == x.shape


def test_fno_encoder_loop_runs_all_layers():
    m = FNOEncoderLoop(n_modes=(8, 8), in_channels=16, out_channels=16, n_layers=3)
    x = torch.randn(2, 16, 32, 64)
    one = m.fno(x, index=0)
    full = m(x)
    assert not torch.allclose(one, full)


def test_hfs_module_shape_and_init():
    m = HFSModule(channels=8, patch_size=4)
    x = torch.randn(2, 8, 32, 64)
    y = m(x)
    assert y.shape == x.shape
    assert torch.allclose(m.lambda_dc, torch.ones_like(m.lambda_dc))
    assert torch.allclose(m.lambda_hfc, torch.ones_like(m.lambda_hfc))
