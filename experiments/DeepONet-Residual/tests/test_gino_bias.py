"""Wave 0 GINO bias instruments (no checkpoint, synthetic leftover)."""

from __future__ import annotations

import numpy as np
import pytest

from response_variability.gino_bias import (
    analyze_pack,
    collapse_and_r2_scalar,
    dct_transfer_ratio,
    leftover,
    offline_rebaseline,
    ols_ab,
    phase3_protocol,
    rhat_on_r_slope,
    svd_targets,
    trough_safe_log_bias,
    wraparound_report,
)
from response_variability.plot_presentation import make_synthetic_pack


def _leftover_pack(*, b: float = 0.5, n: int = 24, n_rec: int = 7, n_freq: int = 64, seed: int = 0):
    rng = np.random.default_rng(seed)
    freq = np.logspace(-1, 1, n_freq)
    nom = 2.0 + 0.3 * rng.standard_normal((n, n_rec, n_freq))
    r = rng.standard_normal((n, n_rec, n_freq))
    ops = nom + r
    gino = nom + b * r
    vs1 = rng.uniform(120.0, 300.0, n)
    H = rng.uniform(20.0, 90.0, n)
    cov = rng.uniform(0.1, 0.3, n)
    vs2 = rng.uniform(800.0, 1400.0, n)
    # Two exact 4D replicates at the end
    vs1[-1], H[-1], cov[-1], vs2[-1] = vs1[-2], H[-2], cov[-2], vs2[-2]
    return {
        "tf_opensees": ops,
        "tf_gino": gino,
        "tf_haskell_nominal": nom,
        "tf_pretell": nom.mean(axis=1) * 0.9,
        "freq": freq,
        "vs1": vs1,
        "H": H,
        "cov": cov,
        "vs2": vs2,
    }


def test_ols_recovers_known_slope():
    x = np.linspace(-2, 2, 200)
    y = 0.25 + 0.4 * x
    fit = ols_ab(x, y)
    assert fit["b"] == pytest.approx(0.4, rel=1e-6)
    assert fit["a"] == pytest.approx(0.25, abs=1e-6)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-6)


def test_rhat_on_r_slope_under_correction():
    pack = _leftover_pack(b=0.5)
    out = rhat_on_r_slope(
        pack["tf_gino"],
        pack["tf_opensees"],
        pack["tf_haskell_nominal"],
        pack["freq"],
        n_boot=80,
        seed=0,
    )
    assert out["b"] == pytest.approx(0.5, abs=0.05)
    assert out["reading"] == "under_correction"
    assert out["b_hi"] < 0.9


def test_offline_rebaseline_identity_when_alt_is_nom():
    pack = _leftover_pack(b=0.7)
    rec = offline_rebaseline(
        pack["tf_gino"],
        pack["tf_opensees"],
        pack["tf_haskell_nominal"],
        pack["tf_haskell_nominal"],
    )
    assert rec["delta_rel_l2"] == pytest.approx(0.0, abs=1e-9)
    assert rec["rel_l2_alt_recon"] == pytest.approx(rec["rel_l2_gino"], abs=1e-9)


def test_wraparound_flags_edge_error():
    n, n_rec, n_f = 12, 9, 32
    ops = np.ones((n, n_rec, n_f))
    gino = np.ones_like(ops)
    gino[:, 0, :] = 3.0
    gino[:, -1, :] = 3.0
    rep = wraparound_report(gino, ops)
    assert rep["edge_minus_center"] > 0.2
    assert rep["rel_l2_edge"] > rep["rel_l2_center"]


def test_svd_low_rank_not_ceiling():
    rng = np.random.default_rng(1)
    base = rng.standard_normal((40, 8))
    tf = base @ rng.standard_normal((8, 64))
    nom = 0.1 * tf
    pack_tf = tf[:, None, :]
    pack_nom = nom[:, None, :]
    svd = svd_targets(pack_tf, pack_nom, latent_dim=128)
    assert svd["tf_rank_99"] < 20
    assert svd["tf_ceiling"] is False


def test_collapse_ratio_when_gino_is_cell_mean():
    n_freq = 32
    freq = np.logspace(-1, 1, n_freq)
    # 8 cells × 3 RF draws
    vs1 = np.repeat(np.linspace(150.0, 280.0, 8), 3)
    H = np.repeat(np.linspace(20.0, 80.0, 8), 3)
    cov = np.repeat(np.linspace(0.12, 0.28, 8), 3)
    vs2 = np.repeat(np.linspace(900.0, 1300.0, 8), 3)
    rng = np.random.default_rng(2)
    ops = np.empty((24, 1, n_freq))
    gino = np.empty_like(ops)
    for c in range(8):
        sl = slice(3 * c, 3 * c + 3)
        mean = 2.0 + 0.1 * c + 0.05 * np.sin(np.linspace(0, 6, n_freq))
        draws = mean + 0.4 * rng.standard_normal((3, n_freq))
        ops[sl, 0] = np.exp(draws)
        gino[sl, 0] = np.exp(mean)  # field-blind cell mean
    x4 = np.column_stack([vs1, H, cov, vs2])
    stats = collapse_and_r2_scalar(gino, ops, x4, freq)
    assert stats["n_exact_cells_ge2"] == 8
    assert stats["collapse_ratio"] < 0.2
    assert stats["r2_scalar_exact"] < 0.25


def test_trough_safe_bias_ignores_near_zero():
    ops = np.ones((4, 3, 20))
    ops[:, :, 0:3] = 1e-8
    gino = np.ones_like(ops) * 2.0
    gino[:, :, 0:3] = 1e-2  # huge ratio in the trough
    b = trough_safe_log_bias(gino, ops)
    assert np.nanmean(b) == pytest.approx(np.log(2.0), rel=1e-3)


def test_dct_ratio_one_when_equal():
    rng = np.random.default_rng(3)
    r = rng.standard_normal((10, 5, 48))
    out = dct_transfer_ratio(r, r, cutoff=16)
    assert out["ratio_mean_below"] == pytest.approx(1.0, abs=1e-6)


def test_analyze_pack_and_phase3(tmp_path):
    pack = _leftover_pack(b=0.4, n=32)
    blob = analyze_pack(pack, domain="iid", n_boot=40, seed=0)
    assert blob["slope"]["b"] == pytest.approx(0.4, abs=0.08)
    assert blob["n"] == 32
    assert "quartile_rel_l2" in blob
    summary = {"domains": {"iid": blob, "three_layer": blob}}
    proto = phase3_protocol(summary)
    assert proto["primary"]
    assert proto["no_op"]["required_if_trained"] is True
    assert "FNO mode count" in proto["cannot_smoke"]


def test_synthetic_presentation_pack_runs():
    pack = make_synthetic_pack(domain="iid", n=16, n_rec=7, n_freq=32, seed=4)
    blob = analyze_pack(pack, domain="iid", n_boot=30, seed=1)
    assert blob["n"] == 16
    assert np.isfinite(blob["slope"]["b"])
    assert leftover(pack["tf_gino"], pack["tf_haskell_nominal"]).shape[0] == 16
