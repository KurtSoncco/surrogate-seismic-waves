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
    assert "quartile_pearson" in blob
    assert "quartile_gof" in blob
    assert "pearson" in blob["per_sample"]
    assert "gof_af" in blob["per_sample"]
    assert "pearson_low" in blob["per_sample"]
    assert "gof_high" in blob["per_sample"]
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


def test_eval_bias_leftover_and_ranking_figures(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from response_variability.plot_eval_bias import (
        plot_bias_vs_covariates,
        plot_bias_vs_covariates_bands,
        plot_leftover_calibration,
        plot_leftover_vs_freq,
        plot_method_ranking_iid,
        plot_method_ranking_ood_compact,
        plot_pearson_boxes_corner,
        plot_pearson_boxes_heldout,
        plot_quartile_forest,
    )
    from response_variability.eval_iid import band_misfit_table, summarize_methods
    from response_variability.plot_presentation import DOMAIN_SPECS, make_synthetic_pack

    from response_variability.seiskit_arms import attach_pretell_p84

    packs = {
        domain: attach_pretell_p84(
            make_synthetic_pack(domain=domain, n=12, n_rec=7, n_freq=32, seed=i)
        )
        for i, domain in enumerate(DOMAIN_SPECS)
    }
    for pack in packs.values():
        pack["tf_dmult"] = pack["tf_pretell"] * 0.97
        pack["tf_toro"] = pack["tf_pretell"] * 0.93
        pack["tf_passeri"] = pack["tf_pretell"] * 1.04
        pack["tf_haskell_column"] = pack["tf_haskell_nominal"]
    p = plot_leftover_calibration(packs, tmp_path / "leftover_calibration.png")
    assert p.is_file() and p.stat().st_size > 0
    assert plot_leftover_vs_freq(packs, tmp_path / "leftover_vs_freq.png").is_file()
    assert plot_bias_vs_covariates(packs, tmp_path / "bias_vs_covariates.png").is_file()
    assert plot_bias_vs_covariates_bands(
        packs, tmp_path / "bias_vs_covariates_bands.png"
    ).is_file()
    assert plot_quartile_forest(packs, tmp_path / "bias_quartile_forest.png").is_file()
    summary, _ = summarize_methods(packs["iid"])
    misfit = band_misfit_table(packs["iid"])
    assert "gof_low" in misfit.columns
    assert "pearson_high" in misfit.columns
    methods = set(summary["method"])
    assert "Dmult" in methods
    assert "Pretell median" in methods
    assert "Pretell percentile" in methods
    assert "Pretell's approach" not in methods
    assert "1D column" not in methods
    q = plot_method_ranking_iid(summary, misfit, tmp_path / "method_ranking_iid.png")
    assert q.is_file()
    assert plot_method_ranking_ood_compact(
        packs, tmp_path / "method_ranking_ood_compact.png"
    ).is_file()
    dip_summary, _ = summarize_methods(packs["dipping"])
    held = plot_pearson_boxes_heldout(
        {"iid": summary, "dipping": dip_summary},
        tmp_path / "method_ranking_pearson_heldout.png",
    )
    assert held.is_file() and held.stat().st_size > 0
    corner = plot_pearson_boxes_corner(
        {"corner_train": summary, "corner_held": dip_summary},
        tmp_path / "method_ranking_pearson_corner.png",
    )
    assert corner.is_file() and corner.stat().st_size > 0


def test_bias_csv_includes_rh_ahv_and_campaign_extras():
    from response_variability.covariates import attach_h5_covariates, present_covariates
    from response_variability.plot_presentation import make_synthetic_pack

    iid = attach_h5_covariates(make_synthetic_pack(domain="iid", n=12, seed=0), domain="iid")
    dip = attach_h5_covariates(
        make_synthetic_pack(domain="dipping", n=12, seed=1), domain="dipping"
    )
    tl = attach_h5_covariates(
        make_synthetic_pack(domain="three_layer", n=12, seed=2), domain="three_layer"
    )
    assert "rH" in present_covariates(iid, "iid")
    assert "aHV" in present_covariates(iid, "iid")
    assert "dip_angle_deg" in present_covariates(dip, "dipping")
    assert "H1" in present_covariates(tl, "three_layer")
    assert "vs_contrast" in present_covariates(tl, "three_layer")
    blob = analyze_pack(tl, domain="three_layer", n_boot=20, seed=0)
    assert "pearson" in blob["per_sample"]
    assert "gof_all" in blob["per_sample"]


def test_6d_cell_ids_group_replicates():
    from response_variability.tail_a_vs_b import cell_ids, param_matrix

    n = 6
    pack = {
        "vs1": np.array([100.0, 100.0, 200.0, 100.0, 200.0, 300.0]),
        "H": np.array([20.0, 20.0, 40.0, 20.0, 40.0, 50.0]),
        "cov": np.array([0.2, 0.2, 0.3, 0.2, 0.3, 0.1]),
        "rH": np.array([30.0, 30.0, 80.0, 30.0, 80.0, 10.0]),
        "aHV": np.array([20.0, 20.0, 15.0, 20.0, 15.0, 12.0]),
        "vs2": np.array([900.0, 900.0, 1100.0, 900.0, 1100.0, 800.0]),
    }
    ids = cell_ids(param_matrix(pack))
    assert ids.shape == (n,)
    assert len(np.unique(ids)) == 3
    assert ids[0] == ids[1] == ids[3]
    assert ids[2] == ids[4]
    assert ids[5] != ids[0]


def test_ops_ops_pearson_identical_is_one():
    from response_variability.tail_a_vs_b import pairwise_ops_pearson

    rng = np.random.default_rng(0)
    freq = np.logspace(-1, 1, 64)
    tf = rng.standard_normal((3, 7, 64)) + 2.0
    tf[1] = tf[0]
    tf[2] = tf[0]
    out = pairwise_ops_pearson(tf, freq, np.array([0, 1, 2]))
    assert out["n_pairs"] == 3.0
    assert out["ops_ops_pearson_median_central"] == pytest.approx(1.0, abs=1e-9)
    assert out["ops_ops_pearson_median_array"] == pytest.approx(1.0, abs=1e-9)


def test_array_mean_pearson_differs_from_central_when_edges_differ():
    from response_variability.tail_a_vs_b import array_pearson, central_pearson

    rng = np.random.default_rng(1)
    n, n_rec, n_f = 4, 7, 48
    freq = np.logspace(-1, 1, n_f)
    ops = 2.0 + 0.3 * rng.standard_normal((n, n_rec, n_f))
    gino = ops.copy()
    gino[:, 0, :] = rng.standard_normal((n, n_f))
    gino[:, -1, :] = rng.standard_normal((n, n_f))
    c = central_pearson(gino, ops, freq)
    a = array_pearson(gino, ops, freq)
    assert np.all(c > 0.99)
    assert np.all(a < c)
    assert np.all(a < 0.95)

