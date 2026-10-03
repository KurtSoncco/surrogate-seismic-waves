"""Harm-rate metrics, gated GINO, physics tokens, learned-1D head."""

from __future__ import annotations

import numpy as np
import torch

from unified_metrics import (
    leftover_rel_l2,
    harm_mask,
    score_leftover_batch,
    tf_from_residual,
)
from data import geom_flags_from_name
from model import build_model, gno_core


def test_harm_rate_zero_when_residual_helps():
    tf1d = np.ones((4, 8))
    r = np.full((4, 8), 0.2)
    tf2d = tf1d + r
    r_hat = r.copy()
    assert harm_mask(tf1d, r_hat, tf2d).sum() == 0
    mets = score_leftover_batch(
        tf1d=tf1d, r_true=r, r_hat=r_hat, tf2d=tf2d, n_rec=2, n_freq=4
    )
    assert mets["harm_rate"] == 0.0


def test_harm_rate_one_when_residual_flips_sign():
    tf1d = np.ones((4, 8))
    r = np.full((4, 8), 0.2)
    tf2d = tf1d + r
    r_hat = -2.0 * r
    assert float(harm_mask(tf1d, r_hat, tf2d).mean()) == 1.0


def test_fail_soft_small_leftover_quantile():
    rng = np.random.default_rng(1)
    n_s, q = 20, 10
    tf1d = np.ones((n_s, q))
    r_true = np.zeros((n_s, q))
    r_true[:4] = 1.0
    r_hat = rng.normal(size=(n_s, q))
    r_hat[4:] = 0.0
    tf2d = tf1d + r_true
    mets = score_leftover_batch(
        tf1d=tf1d, r_true=r_true, r_hat=r_hat, tf2d=tf2d, n_rec=1, n_freq=q
    )
    assert mets["fail_soft_n"] >= 1
    assert mets["fail_soft_mean_abs_rhat"] < 0.05


def test_geom_flags_from_name():
    assert geom_flags_from_name("ood_dipping").tolist() == [1.0, 0.0]
    assert geom_flags_from_name("ood_three_layer").tolist() == [0.0, 1.0]
    assert geom_flags_from_name("iid").tolist() == [0.0, 0.0]


def _tiny_gino(**kw):
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
        **kw,
    )
    fields = torch.randn(2, 3, 32, n_rec)
    stoch = torch.randn(2, 20)
    trunk = torch.randn(2, n_rec * n_freq, trunk_dim)
    return model, fields, stoch, trunk, n_rec, n_freq


def test_gated_gino_forward_and_sparsity_signal():
    model, fields, stoch, trunk, n_rec, n_freq = _tiny_gino(gated=True)
    out = model(fields, stoch, trunk)
    assert out.shape == (2, n_rec * n_freq)
    assert model.last_gate is not None
    assert model.last_gate.shape[:2] == (2, n_rec)
    assert torch.all((model.last_gate >= 0) & (model.last_gate <= 1))
    out.sum().backward()


def test_physics_tokens_change_nodes():
    model, fields, stoch, trunk, n_rec, n_freq = _tiny_gino(
        physics_tokens=True, geom_flag_dim=2
    )
    flags = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    out = model(fields, stoch, trunk, geom_flags=flags)
    assert out.shape == (2, n_rec * n_freq)
    core = gno_core(model)
    assert core.last_nodes is not None
    out.sum().backward()


def test_learned_1d_head_matches_query_grid():
    model, fields, stoch, trunk, n_rec, n_freq = _tiny_gino(learned_1d=True)
    out = model(fields, stoch, trunk)
    core = gno_core(model)
    assert core.last_tf1d_hat is not None
    assert core.last_tf1d_hat.shape == out.shape
    pear = torch.corrcoef(
        torch.stack([core.last_tf1d_hat.ravel(), trunk[..., 0].ravel()])
    )[0, 1]
    assert pear == pear
    out.sum().backward()


def test_eval_harm_rate_synthetic():
    import scoring.eval_harm_rate as ehr

    rep = ehr.synthetic_report()
    assert rep["good_residual"]["harm_rate"] < rep["harmful_residual"]["harm_rate"]
    assert leftover_rel_l2(np.ones((2, 4)), np.ones((2, 4))).shape == (2,)
    assert "logspec_rel_l2_tf_hat" in rep["good_residual"]
    assert "harm_rate_small_leftover" in rep["good_residual"]
    assert "rel_l2_band_mid_hat" in rep["good_residual"]
    assert rep["good_log_residual"]["harm_rate"] <= rep["harmful_residual"]["harm_rate"]


def test_log_multiplicative_reconstruct():
    tf1d = np.ones((2, 8))
    r = np.full((2, 8), np.log(1.1))
    hat = tf_from_residual(tf1d, r, log_residual=True)
    np.testing.assert_allclose(hat, 1.1, rtol=1e-6)


def test_convert_targets_to_log_residual():
    from train import convert_targets_to_log_residual

    class _DS:
        def __init__(self):
            self._cache = [
                {
                    "tf1d": torch.full((4,), 2.0),
                    "tf2d": torch.full((4,), 2.2),
                    "target": torch.full((4,), 0.2),
                }
            ]

        def __len__(self):
            return 1

    ds = _DS()
    convert_targets_to_log_residual(ds)
    want = torch.log(torch.tensor(2.2)) - torch.log(torch.tensor(2.0))
    torch.testing.assert_close(ds._cache[0]["target"], torch.full((4,), float(want)))


def test_loglo_leftover_head_forward():
    model, fields, stoch, trunk, n_rec, n_freq = _tiny_gino(
        fno_kind="loglo", gated=True
    )
    out = model(fields, stoch, trunk)
    assert out.shape == (2, n_rec * n_freq)
    out.sum().backward()


def test_pod_readout_forward():
    n_rec, n_freq = 21, 16
    modes = np.zeros((n_rec, 4, 1000), dtype=np.float32)
    mean = np.zeros((n_rec, 1000), dtype=np.float32)
    model, fields, stoch, trunk, n_rec, n_freq = _tiny_gino(
        pod_readout=True, pod_modes=modes, pod_mean=mean, pod_n_modes=4
    )
    out = model(fields, stoch, trunk)
    assert out.shape == (2, n_rec * n_freq)
    out.sum().backward()


def test_frozen_boost_forward_and_shrink():
    from model import FrozenBoost

    frozen, fields, stoch, trunk, n_rec, n_freq = _tiny_gino()
    booster, _, _, _, _, _ = _tiny_gino(fno_kind="loglo")
    wrap = FrozenBoost(frozen, booster, shrink=0.5)
    out = wrap(fields, stoch, trunk)
    assert out.shape == (2, n_rec * n_freq)
    out.sum().backward()
    assert not any(p.requires_grad for p in wrap.frozen.parameters())
    wrap.select_shrink_from_val(0.0)
    assert float(wrap.shrink.detach()) == 0.0


def test_boost_checkpoint_roundtrip():
    from model import FrozenBoost, build_from_checkpoint_blob

    frozen, fields, stoch, trunk, n_rec, n_freq = _tiny_gino()
    booster, _, _, _, _, _ = _tiny_gino(fno_kind="loglo", gated=True)
    wrap = FrozenBoost(frozen, booster, shrink=0.25)
    wrap.log_residual = True  # type: ignore[attr-defined]
    arch = {
        "field_encoder": "gno",
        "residual_fno": True,
        "n_rec": n_rec,
        "fno_width": 8,
        "fno_n_modes": (4, 4),
        "fno_n_layers": 2,
        "n_gno_layers": 2,
        "log_residual": True,
    }
    blob = {
        "branch_mode": "single",
        "boost": True,
        "log_residual": True,
        "boost_shrink": 0.25,
        "frozen_arch": {
            **arch,
            "fno_kind": "vanilla",
            "gated": False,
            "pod_readout": False,
        },
        "booster_arch": {
            **arch,
            "fno_kind": "loglo",
            "gated": True,
            "pod_readout": False,
        },
        "model": wrap.state_dict(),
    }
    loaded = build_from_checkpoint_blob(
        blob,
        field_channels=3,
        stoch_dim=20,
        trunk_dim=5,
        latent_dim=16,
        field_hidden=8,
        branch_hidden=16,
        trunk_hidden=16,
        trunk_layers=2,
    )
    loaded.load_state_dict(blob["model"])
    wrap.eval()
    loaded.eval()
    with torch.no_grad():
        a = wrap(fields, stoch, trunk)
        b = loaded(fields, stoch, trunk)
    torch.testing.assert_close(a, b)
    assert bool(getattr(loaded, "log_residual", False))


def test_leftover_pod_fit():
    from leftover_pod import fit_residual_pod

    rng = np.random.default_rng(0)
    r = rng.normal(size=(12, 4, 16))
    modes, mean = fit_residual_pod(r, n_modes=3)
    assert modes.shape == (4, 3, 16)
    assert mean.shape == (4, 16)
