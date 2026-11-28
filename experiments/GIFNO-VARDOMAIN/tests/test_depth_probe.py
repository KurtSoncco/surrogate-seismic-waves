"""Masked column PCA, travel-time f0, and the probe decision."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depth_probe import (  # noqa: E402
    NormalEquations,
    accumulate_gram,
    decide,
    masked_pca,
    physical_features,
    profile_l1,
    project_columns,
    quarter_wavelength_hz,
)


def test_quarter_wavelength_of_a_uniform_column_is_vs_over_four_h():
    vs = np.full((100, 3), 400.0)
    f0 = quarter_wavelength_hz(vs)
    assert np.allclose(f0, 1.0)
    features = physical_features(vs, np.full_like(vs, 0.02))
    assert features.shape == (3, 4)
    assert np.allclose(features[:, 0], 0.0)
    assert np.allclose(features[:, 1], np.log(100.0))


def test_masked_pca_ignores_rows_a_short_column_does_not_have():
    torch.manual_seed(0)
    tall = torch.zeros(1, 2, 4, 6)
    tall[0, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0])[:, None]
    short = torch.zeros(1, 2, 2, 6)
    short[0, 0] = torch.tensor([0.0, 1.0])[:, None]
    cross = np.zeros((8, 8))
    cross_n = np.zeros((8, 8))
    total = np.zeros(8)
    count = np.zeros(8)
    accumulate_gram(cross, cross_n, total, count, tall)
    accumulate_gram(cross, cross_n, total, count, short)
    pca = masked_pca(cross, cross_n, total, count, rank=2)
    scores = project_columns(short, pca)
    recon = scores @ pca["components"] + pca["mean"]
    observed = _paired(short)
    assert np.allclose(recon[:, : observed.shape[1]], observed, atol=1e-5)
    assert profile_l1(short, pca, scores) < 1e-5
    # The deep rows of the short columns were not part of their loss of information:
    # shifting them must not change the short-column projection's observed reconstruction.
    assert count[4] == 6  # vs at row 2 is present only on the six tall columns
    assert count[0] == 12


def _paired(values: torch.Tensor) -> np.ndarray:
    vs = values[0, 0].numpy()
    width = vs.shape[1]
    paired = np.empty((width, vs.shape[0] * 2))
    paired[:, 0::2] = vs.T
    paired[:, 1::2] = values[0, 1].numpy().T
    return paired


def test_linear_readout_recovers_a_scaled_target():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 3))
    beta = np.array([[2.0, -1.0], [0.0, 0.5], [1.0, 0.0]])
    y = x @ beta + np.array([0.3, -0.2])
    eq = NormalEquations(3, 2)
    eq.add(x, y)
    pred = eq.predict(x, eq.solve(alpha=1e-8))
    assert np.allclose(pred, y, atol=1e-5)


def test_decision_keeps_pca_on_a_tie_and_the_autoencoder_only_when_it_is_clearly_ahead():
    def scores(ae, pca, physical):
        return {
            "autoencoder": {"mae": ae},
            "pca": {"mae": pca},
            "physical": {"mae": physical},
        }

    assert decide(scores(0.20, 0.20, 0.40)) == "pca"
    assert decide(scores(0.21, 0.20, 0.40)) == "pca"
    assert decide(scores(0.16, 0.20, 0.40)) == "autoencoder"
    assert decide(scores(0.20, 0.18, 0.185)) == "physical"
