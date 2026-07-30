"""Log losses must use ln(max(|TF|, EPS)) — never raw ln|TF| (AGENTS §1.3)."""

from __future__ import annotations

import numpy as np
import torch

import config
from losses_th import amplitude_map, masked_smooth_l1, peak_smooth_l1


def test_amplitude_map_floor():
    t = torch.tensor([0.0, 1e-20, config.TF_LOG_EPS / 10.0, 1.0])
    a = amplitude_map(t)
    assert torch.all(torch.isfinite(a))
    floor = float(np.log(config.TF_LOG_EPS))
    assert float(a.min()) >= floor - 1e-5


def test_no_raw_log_zeros_in_loss():
    pred = torch.zeros(1, 8, 16)
    target = torch.zeros(1, 8, 16)
    mask = torch.ones(1, 8)
    loss = masked_smooth_l1(pred, target, mask)
    assert torch.isfinite(loss)


def test_peak_branch_uses_floor():
    pred = torch.zeros(1, 4, 32)
    target = torch.zeros(1, 4, 32)
    mask = torch.ones(1, 4)
    loss = peak_smooth_l1(pred, target, mask)
    assert torch.isfinite(loss)
