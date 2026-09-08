"""Wave 1 inductive helpers (no checkpoint)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from model import build_model
from response_variability.gino_inductive import (
    dummy_residual_mean_predictor,
    invariance_orbit_check,
    mean_r_on_train,
    prior_collapse_curve,
    represent_vs_use,
    scale_all_lengths,
    scale_all_velocities,
)


def test_cv_preserves_impedance_and_scales_f0():
    vs1 = np.array([150.0, 220.0])
    vs2 = np.array([900.0, 1100.0])
    H = np.array([30.0, 60.0])
    out = invariance_orbit_check(vs1, vs2, H, c_v=1.3, c_l=0.8)
    assert out["impedance_preserved_cv"] is True
    assert out["f0_scales_cv"] is True
    assert out["f0_scales_cl"] is True
    assert out["alpha_vs1_H_breaks_impedance"] is True


def test_length_scale_moves_every_length():
    scaled = scale_all_lengths(
        H=np.array([40.0]),
        rH=np.array([25.0]),
        recorder_x=np.array([0.0, 250.0, 500.0]),
        bedrock_thickness=10.0,
        domain_length=500.0,
        vs2_layer=np.array([12.0]),
        c_l=2.0,
    )
    assert scaled["H"][0] == pytest.approx(80.0)
    assert scaled["rH"][0] == pytest.approx(50.0)
    assert scaled["recorder_x"][-1] == pytest.approx(1000.0)
    assert scaled["bedrock_thickness"] == pytest.approx(20.0)
    assert scaled["vs2_layer"][0] == pytest.approx(24.0)


def test_velocity_scale_all_entries():
    field = np.array([[100.0, 200.0], [300.0, 400.0]])
    out = scale_all_velocities(field, np.array([100.0]), np.array([800.0]), 1.5)
    np.testing.assert_allclose(out["vs_field"], field * 1.5)
    assert out["vs2"][0] == pytest.approx(1200.0)


def test_prior_collapse_cliff_vs_smooth():
    d = np.linspace(0.0, 4.0, 40)
    cliff = np.where(d <= 0.2, 1.0, 0.1)
    smooth = np.exp(-d)
    c = prior_collapse_curve(d, cliff, hull=0.0)
    s = prior_collapse_curve(d, smooth, hull=0.0)
    assert c["shape"] == "cliff"
    assert s["shape"] == "smooth_or_flat"
    assert c["no_ground_truth"] is True


def test_mean_r_near_zero_flag():
    r = np.array([0.01, -0.02, 0.00, 0.01])
    stats = mean_r_on_train(r)
    assert stats["near_zero"] is True
    far = mean_r_on_train(np.array([1.0, 1.2, 0.8]))
    assert far["near_zero"] is False


def test_dummy_predictor_is_haskell():
    pred = dummy_residual_mean_predictor({"vs1": np.zeros(3), "n_freq": 8})
    assert pred.shape == (3, 8)
    assert np.allclose(pred, 0.0)


def test_probe_not_represented_on_noise():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(80, 12))
    y = rng.normal(size=80)
    err = rng.normal(size=80)
    out = represent_vs_use(z, y, err, seed=1)
    assert out["not_represented"] is True
    assert out["n_test"] >= 20


def test_probe_reads_linear_token():
    rng = np.random.default_rng(1)
    f0 = rng.uniform(0.4, 2.0, size=90)
    z = np.column_stack([f0, rng.normal(size=(90, 6))])
    err = 0.3 * f0 + 0.05 * rng.normal(size=90)
    out = represent_vs_use(z, f0, err, seed=2)
    assert out["r2_ridge"] > 0.7
    assert out["not_represented"] is False


def test_gno_exposes_last_branch_and_fno():
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
    )
    fields = torch.randn(2, 3, 32, n_rec)
    stoch = torch.randn(2, 20)
    trunk = torch.randn(2, n_rec * n_freq, trunk_dim)
    out = model(fields, stoch, trunk)
    assert out.shape == (2, n_rec * n_freq)
    core = model.base
    assert core.last_branch is not None
    assert core.last_trunk is not None
    assert core.last_nodes is not None
    assert model.last_fno is not None
    assert core.last_branch.shape[:2] == (2, n_rec)
    assert model.last_fno.shape[0] == 2
