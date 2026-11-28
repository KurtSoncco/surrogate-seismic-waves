"""Neighbor mask, non-overlapping frequency windows, and the linear depth smear."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import ColumnDepthAutoencoder, SparseWindowAutoencoder, frequency_neighbor_mask  # noqa: E402
from screen_hifreq import (  # noqa: E402
    WINDOW,
    band_means,
    linear_broadcast,
    stack_windows,
    tile_windows,
)


def test_neighbor_mask_touches_only_nearby_bins():
    mask = frequency_neighbor_mask(50, knn=5, block=4, stride=1)
    assert mask.shape == (50, 4 + 49)
    center = 20
    # Blocks overlap when block > stride, so a bin's support is the union of its neighbors' blocks.
    owned = set(range(18, 26))
    row = mask[center]
    assert torch.equal(row[sorted(owned)], torch.ones(len(owned)))
    assert float(row[:18].sum()) == 0.0
    assert float(row[26:].sum()) == 0.0
    model = SparseWindowAutoencoder(50, 3, knn=5, block=4, stride=1)
    assert model.encoder[-1].bias is None
    assert model.mask.shape[1] == model.out_weight.shape[1]


def test_windows_do_not_overlap_and_stack_back():
    curves = np.arange(2 * 100, dtype=np.float64).reshape(2, 100)
    windows = tile_windows(curves, window=50)
    assert windows.shape == (2, 2, 50)
    assert np.array_equal(stack_windows(windows), curves)
    assert not np.array_equal(windows[:, 0], windows[:, 1])


def test_wider_code_keeps_the_soil_channel_count_and_ignores_a_masked_row():
    torch.manual_seed(0)
    wide = ColumnDepthAutoencoder(2, code_dim=32)
    values = torch.randn(1, 2, 12, 3)
    mask = torch.ones(1, 12, 3)
    mask[:, 9:] = 0
    depth = torch.linspace(0, 1, 12).view(1, 1, 12, 1).expand(1, 1, 12, 3)
    surface, recon = wide(values, mask, depth)
    assert surface.shape == (1, 32, 1, 3)
    assert recon.shape == values.shape
    shifted = values.clone()
    shifted[:, :, 9:] += 4
    surface2, recon2 = wide(shifted, mask, depth)
    assert torch.allclose(surface, surface2)
    assert torch.allclose(recon, recon2)
    narrow = ColumnDepthAutoencoder(2)
    assert narrow.code_dim == 2
    assert narrow.row[0].out_channels == 2


def test_equal_weight_collapse_smears_a_contact_across_rows():
    values = torch.zeros(1, 2, 4, 1)
    values[:, :, 2:] = 1.0
    mask = torch.ones(1, 4, 1)
    recon = linear_broadcast(values, mask)
    assert torch.allclose(recon, torch.full_like(values, 0.5))
    freq = np.array([0.1, 0.3, 0.5, 2.0, 4.0, 8.0, 10.0])
    table = band_means(freq, {"ae": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])})
    assert set(table) == {"0.1-0.4", "0.4-1", "1-2.5", "2.5-5", "5-10"}
    assert table["0.1-0.4"]["ae"] == 0.5
    assert WINDOW == 50
