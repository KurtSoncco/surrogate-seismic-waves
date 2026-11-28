"""Unified-training pieces: equivariant FiLM, losses, retrieval, metrics, pilot checks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import KIND_COV0, STAGE_UNIFIED, make_cov0_window  # noqa: E402
from deltaphi import DeltaPhiHead, RetrievalIndex, station_descriptor  # noqa: E402
from encoding import CH_X, encode  # noqa: E402
from evaluate import station_metrics  # noqa: E402
from model import LOCAL_KERNELS_M, VarDomainLOGLO, fit_log_pod, masked_box_average  # noqa: E402
from sampling import STAGE_UNIFIED as SAMPLER_STAGE, QuotaBatchSampler  # noqa: E402
from spectral_loss import hybrid_loss, rel_l1  # noqa: E402
from vrba import VRBAWeights, grad_norm_balance, weighted_residual  # noqa: E402
from width_physics_check import band_dlog  # noqa: E402


def _tiny_model(**kwargs):
    modes, mean = fit_log_pod(np.random.default_rng(0).normal(size=(16, 8)).astype(np.float32), 4)
    return VarDomainLOGLO(
        "reference", modes, mean, n_modes_x=4, narrow=4, latent=8, n_layers=1, patch_w=20, **kwargs,
    )


def test_translation_equivariant_drops_absolute_x_and_shifts_with_the_field():
    model = _tiny_model(translation_equivariant=True)
    assert int(CH_X) not in set(int(i) for i in model.lift_idx.tolist())
    vs = np.full((30, 200), 200.0)
    zeta = np.full_like(vs, 0.02)
    vs[:, 80:100] = 120.0
    base = torch.from_numpy(encode("reference", vs, zeta, 30))
    shifted = torch.from_numpy(encode("reference", np.roll(vs, 20, axis=1), zeta, 30))
    left = model.local_context(base.unsqueeze(0))[0, 40]
    right = model.local_context(shifted.unsqueeze(0))[0, 40]
    assert left.shape[-1] == 4 * len(LOCAL_KERNELS_M) + 2
    assert not torch.allclose(left, right)
    y = model(base.unsqueeze(0), torch.tensor([[10.0, 20.0]]))
    assert y.shape == (1, 2, 8)
    assert torch.isfinite(y).all()


def test_masked_box_average_ignores_the_exterior():
    signal = torch.zeros(1, 1, 11)
    signal[0, 0, 5] = 10.0
    mask = torch.zeros(1, 1, 11)
    mask[0, 0, 4:7] = 1.0
    out = masked_box_average(signal, mask, 5)
    assert torch.isfinite(out).all()
    assert float(out[0, 0, 5]) > float(out[0, 0, 0])


def test_rel_l1_weights_the_peak_more_than_a_trough():
    true = torch.tensor([[[1.0, 10.0, 0.4]]])
    valid = torch.tensor([[True]])
    peak = true.clone()
    peak[0, 0, 1] = 8.0
    trough = true.clone()
    trough[0, 0, 2] = 0.2
    assert float(rel_l1(peak, true, valid)) > float(rel_l1(trough, true, valid))


def test_hybrid_and_vrba_are_finite():
    pred = torch.nn.Parameter(torch.zeros(1, 1, 16))
    target = torch.zeros(1, 1, 16)
    prior = torch.zeros(1, 1, 16)
    valid = torch.ones(1, 1, dtype=torch.bool)
    freq = torch.linspace(0.1, 10.0, 16)
    width = torch.tensor([500.0])

    def log_l1(p, t, v, f, w, fw):
        return (p - t).abs().mean()

    loss, parts = hybrid_loss(pred, target, prior, valid, freq, width, None, residual=True, log_l1_fn=log_l1)
    assert torch.isfinite(loss)
    weights = VRBAWeights(16, potential="quadratic")
    lam = weights(pred.detach() - target, valid, 0)
    assert torch.isfinite(lam).all()
    data = weighted_residual(pred - target, valid, lam)
    balanced = grad_norm_balance({"data": data, "spectral": parts["spectral"]}, [pred])
    assert torch.isfinite(balanced)


def test_cov0_residual_is_zero():
    class Cache:
        def campaign(self, index):
            vs = np.full((1, 12, 40), 180.0)
            zeta = np.full((1, 12, 40), 0.02)
            camp = type("C", (), {})()
            camp.vs, camp.zeta = vs, zeta
            camp.n_active = np.array([12])
            camp.rec_x = np.array([20])
            camp.log_tf1d = np.zeros((1, 1, 6), dtype=np.float32)
            camp.log_tf1d[0, 0, 2] = 1.5
            camp.f1 = np.array([0.8])
            camp.f2 = np.array([2.4])
            camp.group = np.array([3])
            camp.name = "iid"
            return camp, 0

    _x, _q, tf, prior, valid, info, _xa = make_cov0_window(Cache(), "metric", 0, width=500)
    assert int(info[0]) == KIND_COV0
    assert torch.equal(tf[valid], prior[valid])
    assert float(tf[valid][0, 2]) == 1.5


def test_unified_sampler_draws_every_family():
    assert SAMPLER_STAGE == STAGE_UNIFIED
    records = []
    for family, n in (("iid", 8), ("ood_dipping", 4), ("ood_three_layer", 4), ("cov0", 4)):
        for i in range(n):
            records.append({"kind": "run", "family": family, "near_1d": family == "cov0"})
    for width in (500, 1000, 1500, 2000):
        for i in range(6):
            records.append({
                "kind": "section", "family": "manifold", "lx": width, "stratum": "one_feature",
                "near_1d": i == 0,
            })
    sampler = QuotaBatchSampler(records, batch=8, stage=STAGE_UNIFIED, n_batches=12, seed=0, use_pairs=False, near_1d_share=0.5)
    seen = set()
    for batch in sampler:
        for i in batch:
            seen.add(records[i]["family"])
    assert {"iid", "manifold", "cov0"} <= seen
    assert "ood_dipping" not in seen
    assert "ood_three_layer" not in seen


def test_deltaphi_excludes_the_query_group_and_prefers_a_near_1d_neighbour():
    freq = 8
    flat = station_descriptor(np.full((20, 40), 200.0), 20.0, f1_hz=1.0, f2_hz=3.0, lateral_log_std=0.0)
    basin = station_descriptor(np.linspace(120.0, 400.0, 20)[:, None] * np.linspace(0.7, 1.3, 40)[None, :], 20.0, f1_hz=0.4, f2_hz=1.2, lateral_log_std=0.4)
    index = RetrievalIndex(
        np.stack([flat, basin, flat]),
        np.stack([np.zeros(freq), np.ones(freq), np.full(freq, 0.01)]),
        np.array([1, 2, 3]),
    )
    curve, _sim, top = index.retrieve(flat, group=1, k=2, train=False)
    assert 0 not in top
    assert float(np.max(np.abs(curve))) < 0.05
    head = DeltaPhiHead(4, freq)
    delta = head(torch.zeros(1, 4), torch.zeros(1, freq), torch.ones(1, 1))
    assert torch.allclose(delta, torch.zeros_like(delta))


def test_station_metrics_match_a_perfect_curve_and_a_shifted_peak():
    freq = np.geomspace(0.1, 10.0, 64)
    peak = np.exp(-0.5 * ((np.log(freq) - np.log(1.0)) / 0.15) ** 2) * 8 + 0.5
    second = np.exp(-0.5 * ((np.log(freq) - np.log(3.0)) / 0.12) ** 2) * 4
    true = peak + second
    perfect = station_metrics(np.log(true)[None], np.log(true)[None], freq)
    assert perfect["log_l1"][0] < 1e-8
    assert perfect["pearson"][0] > 0.999
    assert perfect["spearman"][0] > 0.999
    assert perfect["ccc_log"][0] > 0.999
    assert perfect["m1_freq_err"][0] == 0.0
    shifted = true * np.exp(-0.3 * (freq > 1.0))
    scored = station_metrics(np.log(shifted)[None], np.log(true)[None], freq)
    assert scored["rel_l1"][0] > 0.0
    assert np.isfinite(scored["anderson"][0])


def test_band_dlog_is_zero_on_a_copy_and_separates_a_low_band_change():
    freq = np.geomspace(0.1, 10.0, 40)
    base = np.ones((3, 40))
    same = band_dlog(base, base, freq)
    assert same["all"]["abs_dlog"] == 0.0
    other = base.copy()
    other[:, freq < 0.4] = 2.0
    changed = band_dlog(base, other, freq)
    assert changed["low"]["abs_dlog"] > changed["high"]["abs_dlog"]


def test_unified_selection_uses_manifold_validation_only():
    from train import accept_improvement, selection_score

    res = {"bins": {
        "iid_w500": {"log_l1": 0.10},
        "manifold_w1000": {"log_l1": 0.20},
        "manifold_near_w1000": {"log_l1": 0.01},
        "manifold_blind_w1000": {"log_l1": 0.50},
        "ood_dipping_w500": {"log_l1": 0.80},
        "ood_three_layer_w500": {"log_l1": 0.90},
    }}
    score = selection_score(res, STAGE_UNIFIED)
    assert abs(score - 0.20) < 1e-12
    assert accept_improvement(0.049, 0.050, delta=1e-3)
    assert not accept_improvement(0.0495, 0.050, delta=1e-3)


def test_broadband_checks_pass(tmp_path):
    from broadband_pilot import write_pilot

    report = write_pilot(tmp_path, n_geologies=8)
    assert report["n_jobs"] == 32
    assert report["mesh_ok"]
    assert report["wavelet_within_20db"]
    assert report["flat_abs_g_max"] < 0.01
    widths = {int(row["width_m"]) for row in __import__("csv").DictReader((tmp_path / "jobs.csv").open())}
    assert widths == {500, 1000, 1500, 2000}
