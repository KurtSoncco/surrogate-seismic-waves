from __future__ import annotations

import numpy as np
import torch

from data import (
    build_stoch_vector,
    build_trunk_queries,
    f0_quarter_wavelength,
    infer_stoch_layout,
    stoch_dim,
    travel_time_s,
    trunk_feature_names,
)
from unified_metrics import rel_l1_rows, residual_tv_rows, score_leftover_batch


def test_rel_l1_formula():
    true = np.array([[1.0, 2.0, 3.0], [4.0, 0.0, 2.0]])
    pred = np.array([[1.0, 1.0, 3.0], [4.0, 1.0, 2.0]])
    got = rel_l1_rows(pred, true)
    expect = np.array([1.0 / 6.0, 1.0 / 6.0])
    np.testing.assert_allclose(got, expect)
    tv = residual_tv_rows(np.array([[0.0, 1.0, -1.0]]))
    np.testing.assert_allclose(tv, [3.0])


def test_rel_l1_in_leftover_batch():
    n_rec, n_f = 3, 8
    tf1d = np.ones((2, n_rec * n_f))
    r_true = np.zeros((2, n_rec * n_f))
    r_hat = np.zeros((2, n_rec * n_f))
    tf2d = tf1d.copy()
    out = score_leftover_batch(
        tf1d=tf1d, r_true=r_true, r_hat=r_hat, tf2d=tf2d, n_rec=n_rec, n_freq=n_f
    )
    assert out["rel_l1_tf_hat"] == 0.0
    assert out["rel_l1_tf_1d"] == 0.0
    assert out["residual_tv_hat"] == 0.0
    assert "peak_dlnA_vs_2d_mean" not in out


def test_leftover_peak_dlnA_uses_f0_window_not_global_argmax():
    from data import f0_quarter_wavelength

    h = np.array([10.0, 12.0])
    vs = np.array([180.0, 420.0])
    f0_calc = f0_quarter_wavelength(h, vs, vs_rock=900.0)
    freq = np.logspace(-1, 1, 800)
    n_rec, n_f = 3, freq.size
    f0_true = f0_calc * 1.10
    f_harm = 3.0 * f0_calc
    af_2d = 1.8 * np.exp(
        -0.5 * ((np.log(freq) - np.log(f0_true)) / 0.02) ** 2
    ) + 5.0 * np.exp(-0.5 * ((np.log(freq) - np.log(f_harm)) / 0.02) ** 2)
    af_hat = 1.8 * np.exp(-0.5 * ((np.log(freq) - np.log(f0_true)) / 0.02) ** 2)
    tf2d = np.broadcast_to(af_2d[None, None, :], (1, n_rec, n_f)).copy()
    tf_hat = np.broadcast_to(af_hat[None, None, :], tf2d.shape).copy()
    tf1d = np.ones_like(tf2d)
    out = score_leftover_batch(
        tf1d=tf1d.reshape(1, -1),
        r_true=(tf2d - tf1d).reshape(1, -1),
        r_hat=(tf_hat - tf1d).reshape(1, -1),
        tf2d=tf2d.reshape(1, -1),
        freq=freq,
        n_rec=n_rec,
        n_freq=n_f,
        f0_calc=np.array([f0_calc]),
        tf_hat=tf_hat.reshape(1, -1),
    )
    assert out["peak_dlnA_vs_2d_mean"] < 0.05
    assert out["peak_df0_hz_vs_2d_mean"] < 0.05 * f0_true


def test_fstar_tts_three_layer_toy():
    h_soil = np.array([10.0, 12.0])
    vs_soil = np.array([180.0, 420.0])
    vs_rock = 900.0
    # Bedrock thickness/Vs must not enter T even if concatenated on the stack.
    h = np.array([10.0, 12.0, 8.0])
    vs = np.array([180.0, 420.0, vs_rock])
    t = travel_time_s(h, vs, vs_rock=1105.0, vs_1d=np.append(vs_soil, vs_rock))
    np.testing.assert_allclose(t, np.sum(h_soil / vs_soil))
    f0 = f0_quarter_wavelength(h, vs, vs_rock=vs_rock)
    np.testing.assert_allclose(f0, 1.0 / (4.0 * t))
    freq = np.array([0.25, 1.0, 4.0])
    names = trunk_feature_names("full", x_coord="xH")
    y = build_trunk_queries(
        vs_col=np.full(5, 300.0),
        H=float(np.sum(h_soil)),
        recorder_x=np.arange(5, dtype=float),
        freq_s=freq,
        sin_f=np.zeros_like(freq),
        cos_f=np.ones_like(freq),
        trunk_names=names,
        fstar_kind="tts",
        layer_H=h,
        layer_Vs=vs,
        vs_rock=vs_rock,
    )
    assert "x_over_H" in names
    f_star = y[:, names.index("f_star")].reshape(5, 3)
    expect = 4.0 * freq * t
    np.testing.assert_allclose(f_star[0], expect, rtol=1e-6)
    np.testing.assert_allclose(f_star[0], freq / f0, rtol=1e-6)


def test_travel_time_ignores_bedrock_from_meta():
    from data import nom_layers_from_meta, soil_layers_excluding_bedrock, travel_time_s

    # 3+ layers: last Vs is bedrock even if Vs2 is wrong (three-layer misspec).
    meta = {
        "layer_H": np.array([np.array([10.0, 12.0, 8.0])], dtype=object),
        "layer_Vs": np.array([np.array([180.0, 420.0, 900.0])], dtype=object),
        "Vs2": np.array([1105.0]),
        "H": np.array([30.0]),
        "Vs1": np.array([180.0]),
    }
    h, vs = nom_layers_from_meta(meta, 0)
    np.testing.assert_allclose(h, [10.0, 12.0])
    np.testing.assert_allclose(vs, [180.0, 420.0])

    # Two soil layers + 1D column whose bottom is rock: keep both soil Vs.
    vs_1d = np.concatenate([np.full(20, 180.0), np.full(12, 420.0), np.full(8, 900.0)])
    h2, vs2 = soil_layers_excluding_bedrock(
        [10.0, 12.0],
        [180.0, 420.0],
        vs_rock=1105.0,
        vs_1d=vs_1d,
    )
    np.testing.assert_allclose(h2, [10.0, 12.0])
    np.testing.assert_allclose(vs2, [180.0, 420.0])
    np.testing.assert_allclose(travel_time_s(h2, vs2), 10.0 / 180.0 + 12.0 / 420.0)

    # Prefer Vs_1d[-1] over a wrong Vs2 on a stack that includes rock.
    h3, vs3 = soil_layers_excluding_bedrock(
        [10.0, 12.0, 8.0],
        [180.0, 420.0, 900.0],
        vs_rock=1105.0,
        vs_1d=vs_1d,
    )
    np.testing.assert_allclose(h3, [10.0, 12.0])
    np.testing.assert_allclose(vs3, [180.0, 420.0])


def test_cov_only_dim_and_vector():
    assert stoch_dim(layout="cov_only") == 1
    v = build_stoch_vector(
        xi_vals=np.arange(16, dtype=np.float32),
        rH=50.0,
        aHV=20.0,
        CoV=0.22,
        xi_damp=0.05,
        layout="cov_only",
    )
    assert v.shape == (1,)
    assert v[0] == np.float32(0.22)
    state = {"base.stoch_mlp.0.weight": torch.zeros(64, 1)}
    assert infer_stoch_layout(state) == "cov_only"


def test_default_score_domains_omit_three_layer():
    from scoring.score_mscale_distributions import DEFAULT_SCORE_DOMAINS, DOMAINS

    assert DEFAULT_SCORE_DOMAINS == ("iid", "ood_dipping")
    assert "ood_three_layer" in DOMAINS
    assert "ood_three_layer" not in DEFAULT_SCORE_DOMAINS


def test_t4b4_cpu_forward_xi_cov_and_cov_only():
    from model import build_model

    n_rec, n_freq, trunk_dim = 21, 16, 5
    common = dict(
        field_channels=3,
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
        n_mscale_trunk=4,
        n_mscale_branch=4,
    )
    fields = torch.randn(2, 3, 32, n_rec)
    trunk = torch.randn(2, n_rec * n_freq, trunk_dim)
    for sdim in (17, 1):
        model = build_model("single", stoch_dim=sdim, **common)
        out = model(fields, torch.randn(2, sdim), trunk)
        assert out.shape == (2, n_rec * n_freq)


def test_identity_mixer_forward():
    from model import build_model, gno_core

    n_rec = 21
    model = build_model(
        "single",
        field_channels=3,
        stoch_dim=17,
        trunk_dim=5,
        latent_dim=16,
        field_hidden=8,
        branch_hidden=16,
        trunk_hidden=16,
        trunk_layers=2,
        field_encoder="identity",
        residual_fno=True,
        n_rec=n_rec,
        fno_width=8,
        fno_n_modes=(4, 4),
        fno_n_layers=1,
        n_gno_layers=2,
        n_mscale_trunk=4,
        n_mscale_branch=4,
    )
    core = gno_core(model)
    assert isinstance(core.gno, torch.nn.Identity)
    out = model(
        torch.randn(1, 3, 32, n_rec),
        torch.randn(1, 17),
        torch.randn(1, n_rec * 8, 5),
    )
    assert out.shape == (1, n_rec * 8)


def test_practitioner_pack_no_seed():
    from practitioner import NominalLayers, pack_practitioner_inputs, sample_vs_ensemble

    layers = NominalLayers(
        H=np.array([20.0, 20.0]),
        Vs=np.array([180.0, 400.0]),
        zeta=np.array([0.03, 0.02]),
        vs_rock=900.0,
    )
    rec = np.linspace(0, 499, 21)
    freq = np.logspace(-1, 1, 32)
    vs = np.full((48, 500), 900.0)
    vs[:20] = 180.0
    vs[20:40] = 400.0
    pack = pack_practitioner_inputs(
        vs=vs, layers=layers, cov=0.2, recorder_x=rec, freq=freq, trunk_scales=1
    )
    assert pack["stoch"].shape == (1,)
    assert pack["fields"].shape[0] == 3
    assert pack["fields"].shape[-1] == 21
    assert pack["f0"] > 0
    ens = sample_vs_ensemble(
        layers, cov=0.2, rf_seed=7, rH=30.0, aHV=10.0, nx=64, warn_support=False
    )
    assert ens.shape[1] == 64
    assert ens.min() > 0


def test_spectral_kl_from_field_shape():
    from features import spectral_kl_from_field

    rng = np.random.default_rng(0)
    vs = 180.0 * np.exp(0.2 * rng.standard_normal((40, 64)))
    xi, names = spectral_kl_from_field(vs, rH=30.0, aHV=10.0, k=8)
    assert xi.shape == (16,)
    assert len(names) == 16
    assert np.isfinite(xi).all()
