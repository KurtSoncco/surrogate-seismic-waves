"""Continuous-x readout: linear in metres, exact on the grid, order-independent."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import subprocess
import sys

from encoding import CH_LOG_LX, CH_MASK, CONTEXT_M, FILM_CHANNELS, encode  # noqa: E402
from model import (  # noqa: E402
    ColumnDepthAutoencoder,
    FrozenColumnPCA,
    MaskedDepthCollapse,
    VarDomainLOGLO,
    extend_pod_modes,
    fit_log_pod,
    model_from_config,
    sample_grid,
)


def _model():
    modes, mean = fit_log_pod(np.random.default_rng(0).normal(size=(40, 8)).astype(np.float32), 4)
    return VarDomainLOGLO("metric", modes, mean, n_modes_x=8, narrow=4, latent=8, n_layers=1, patch_w=20)


def test_frozen_pca_drops_rows_outside_the_soil_and_does_not_train():
    height = 4
    mean = np.zeros(2 * height, dtype=np.float32)
    components = np.zeros((2, 2 * height), dtype=np.float32)
    components[0, 2] = 1.0
    layer = FrozenColumnPCA(mean, components)
    assert list(layer.parameters()) == []
    x = torch.zeros(1, 12, height, 3)
    x[0, 0, 1, 1] = 0.5
    x[0, CH_MASK, :, 1] = 1.0
    code = layer(x)
    assert code.shape == (1, 2, 1, 3)
    assert abs(float(code[0, 0, 0, 1]) - 0.5) < 1e-6
    rock = x.clone()
    rock[0, 0, -1, 1] = 9.0
    rock[0, CH_MASK, -1, 1] = 0.0
    assert torch.allclose(code, layer(rock))
    modes, pod_mean = fit_log_pod(np.random.default_rng(0).normal(size=(40, 8)).astype(np.float32), 4)
    model = VarDomainLOGLO(
        "metric", modes, pod_mean, n_modes_x=8, narrow=4, latent=8, n_layers=1, patch_w=20,
        depth_pca=True, pca_mean=mean, pca_components=components,
    )
    assert not any(name.startswith("column_pca") for name, _param in model.named_parameters())


def test_log_width_is_not_a_film_input():
    assert CH_LOG_LX not in FILM_CHANNELS


def test_query_stays_on_the_metre_when_the_grid_is_bfloat16():
    field = torch.zeros(1, 1, 2048)
    field[0, 0, 1250] = 3.0
    field[0, 0, 1248] = 9.0
    field[0, 0, 1200] = 1.0
    field[0, 0, 1201] = 3.0
    grid = field.to(dtype=torch.bfloat16)
    query = torch.tensor([[1250.0, 1200.5]])
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        out = sample_grid(grid, query)
    assert float(out[0, 0]) == 3.0
    assert abs(float(out[0, 1]) - 2.0) < 1e-5


def test_local_cutoff_drops_the_four_metre_mode():
    modes, mean = fit_log_pod(np.random.default_rng(0).normal(size=(40, 8)).astype(np.float32), 4)
    model = VarDomainLOGLO(
        "metric", modes, mean, n_modes_x=8, narrow=4, latent=8, n_layers=1, local_min_wavelength=10.0,
    )
    mask = model.encoder.local_layers[0].mode_mask
    assert float(mask[2]) == 1.0
    assert float(mask[5]) == 0.0


def test_half_metre_is_the_midpoint_and_order_does_not_matter():
    torch.manual_seed(0)
    model = _model().eval()
    vs = np.full((30, 600), 200.0)
    vs[:, 300:] = 400.0
    x = torch.from_numpy(encode("metric", vs, np.full_like(vs, 0.02), 30)).unsqueeze(0)
    with torch.no_grad():
        y0 = model(x, torch.tensor([[1200.0]]))
        y1 = model(x, torch.tensor([[1201.0]]))
        ym = model(x, torch.tensor([[1200.5]]))
        order = model(x, torch.tensor([[1200.5, 1500.0, 1200.0]]))
        flipped = model(x, torch.tensor([[1200.0, 1500.0, 1200.5]]))
    torch.testing.assert_close(ym, 0.5 * (y0 + y1), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(order, flipped[:, [2, 1, 0]])


def test_extend_pod_leaves_the_old_span_unchanged():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(30, 6)).astype(np.float32)
    modes, mean = fit_log_pod(base, 3)
    extra = np.concatenate([base, rng.normal(size=(10, 6)).astype(np.float32)])
    extended = extend_pod_modes(modes, mean, extra, n_extra=2)
    assert extended.shape[0] >= modes.shape[0]
    np.testing.assert_allclose(extended[: modes.shape[0]], modes, atol=1e-5)
    gram = extended @ extended.T
    np.testing.assert_allclose(gram, np.eye(gram.shape[0]), atol=1e-4)


def test_n_keep_zeros_appended_modes_only():
    torch.manual_seed(0)
    model = _model().eval()
    n_old = int(model.pod_modes.shape[0])
    extra, _ = fit_log_pod(np.random.default_rng(1).normal(size=(20, 8)).astype(np.float32), 2)
    model.grow_modes(torch.as_tensor(extra))
    with torch.no_grad():
        model.branch[-1].weight[n_old:].normal_()
        model.branch[-1].bias[n_old:].normal_()
    vs = np.full((30, 600), 250.0)
    x = torch.from_numpy(encode("metric", vs, np.full_like(vs, 0.02), 30)).unsqueeze(0)
    query = torch.tensor([[1500.0, 1600.0]])
    with torch.no_grad():
        full = model(x, query)
        kept = model(x, query, n_keep=n_old)
        same = model(x, query, n_keep=int(model.pod_modes.shape[0]))
    assert not torch.allclose(full, kept)
    torch.testing.assert_close(full, same)


def _basis(n_freq: int = 8, n_modes: int = 4):
    curves = np.random.default_rng(0).normal(size=(40, n_freq)).astype(np.float32)
    return fit_log_pod(curves, n_modes)


def _nonlinear(encoding: str = "metric", **kwargs) -> VarDomainLOGLO:
    modes, mean = _basis()
    model = VarDomainLOGLO(
        encoding, modes, mean, n_modes_x=8, narrow=4, latent=8, n_layers=1, patch_w=20,
        readout="nonlinear", depth_ae=True, local_min_wavelength=0.0, hfp_kernel=4, hfp_stride=4,
        **kwargs,
    )
    model.set_frequencies(torch.linspace(0.1, 10.0, modes.shape[1]))
    return model


def _window(encoding: str, vs: np.ndarray):
    return torch.from_numpy(encode(encoding, vs, np.full_like(vs, 0.02), vs.shape[0])).unsqueeze(0)


def _inside(left: torch.Tensor, right: torch.Tensor, mid: torch.Tensor) -> None:
    slack = 0.25 * (left - right).abs() + 1e-3
    lo = torch.minimum(left, right) - slack
    hi = torch.maximum(left, right) + slack
    assert torch.all(mid >= lo - 1e-5)
    assert torch.all(mid <= hi + 1e-5)


def test_mask_zero_row_does_not_change_the_surface_code():
    torch.manual_seed(0)
    ae = ColumnDepthAutoencoder(4)
    h = torch.randn(2, 4, 16, 8)
    mask = torch.ones(2, 16, 8)
    mask[:, 12:] = 0
    depth = torch.linspace(0, 1, 16).view(1, 1, 16, 1).expand(2, 1, 16, 8)
    surface, recon = ae(h, mask, depth)
    h_rock = h.clone()
    h_rock[:, :, 12:] += 10
    surface_rock, recon_rock = ae(h_rock, mask, depth)
    torch.testing.assert_close(surface, surface_rock)
    torch.testing.assert_close(recon, recon_rock)
    weight = mask[:, None]
    loss = ((recon - h).abs() * weight).sum()
    loss_rock = ((recon_rock - h_rock).abs() * weight).sum()
    torch.testing.assert_close(loss, loss_rock)

    model = _nonlinear().eval()
    vs = np.full((30, 500), 220.0)
    x = _window("metric", vs)
    row = int(torch.nonzero(x[0, CH_MASK, :, 0] == 0)[0])
    x_rock = x.clone()
    x_rock[:, 0, row] += 3
    with torch.no_grad():
        _, latent, _ = model.encode(x)
        loss_clean = model.reconstruction_l1()
        _, latent_rock, _ = model.encode(x_rock)
        loss_rock_full = model.reconstruction_l1()
    torch.testing.assert_close(latent, latent_rock)
    torch.testing.assert_close(loss_clean, loss_rock_full)


def test_nonlinear_forward_on_both_grids():
    torch.manual_seed(0)
    for encoding, shape in (("metric", (30, 600)), ("reference", (40, 700))):
        model = _nonlinear(encoding).eval()
        x = _window(encoding, np.full(shape, 220.0))
        with torch.no_grad():
            y = model(x, torch.tensor([[10.0, 20.5]]))
        assert y.shape == (1, 2, 8)
        assert isinstance(model.collapse, ColumnDepthAutoencoder)


def test_missing_readout_key_keeps_the_linear_collapse():
    modes, mean = _basis()
    model = model_from_config(
        {"encoding": "metric", "narrow": 4, "latent": 8, "layers": 1, "modes_x": 8}, modes, mean,
    )
    assert isinstance(model.collapse, MaskedDepthCollapse)
    assert model.readout_kind == "pod"
    assert model.spectral is None


def test_local_patch_keeps_the_two_metre_mode():
    model = _nonlinear()
    mask = model.encoder.local_layers[0].mode_mask
    assert float(mask[10]) == 1.0
    assert model.hfp_stride == 4
    assert model.local_min_wavelength == 0.0


def test_nonlinear_half_metre_is_the_decoded_midpoint_latent():
    torch.manual_seed(0)
    model = _nonlinear().eval()
    with torch.no_grad():
        model.spectral.net[-1].weight.normal_()
        model.spectral.net[-1].bias.normal_()
    x = _window("metric", np.full((30, 600), 250.0))
    query = torch.tensor([[1200.5]])
    with torch.no_grad():
        y = model(x, query)
        _coeff, latent, _recon = model.encode(x)
        mid = sample_grid(latent, query)
        expect = model.spectral(mid, model.freq_hz).clamp(-model.log_clamp, model.log_clamp)
    torch.testing.assert_close(y, expect, atol=1e-5, rtol=1e-5)


def test_step_interface_envelope():
    torch.manual_seed(0)
    model = _nonlinear().eval()
    vs = np.full((30, 600), 200.0)
    vs[:, 300:] = 400.0
    x = _window("metric", vs)
    jump = CONTEXT_M + 300
    cols = [jump - 2, jump - 1, jump, jump + 1, jump + 2]
    halves = [jump - 1.5, jump - 0.5, jump + 0.5]
    with torch.no_grad():
        spectra = {c: model(x, torch.tensor([[float(c)]])) for c in cols}
        half = {c: model(x, torch.tensor([[c]])) for c in halves}
        _coeff, latent, _recon = model.encode(x)
        for c in halves:
            query = torch.tensor([[c]])
            lat = sample_grid(latent, query)
            lat_mean = 0.5 * (sample_grid(latent, torch.tensor([[c - 0.5]])) + sample_grid(latent, torch.tensor([[c + 0.5]])))
            torch.testing.assert_close(lat, lat_mean, atol=1e-5, rtol=1e-5)
            expect = model.spectral(lat, model.freq_hz).clamp(-model.log_clamp, model.log_clamp)
            torch.testing.assert_close(half[c], expect, atol=1e-5, rtol=1e-5)
            _inside(spectra[c - 0.5], spectra[c + 0.5], half[c])


def _write_smoke_cache(root) -> None:
    freq = np.geomspace(0.1, 10.0, 16).astype(np.float32)
    np.save(root / "freq.npy", freq)
    n, z, stations = 12, 8, 21
    rec_x = np.linspace(20, 460, stations).round().astype(np.int64)
    camp = root / "iid"
    camp.mkdir()
    np.save(camp / "vs.npy", np.full((n, z, 500), 250.0, np.float32))
    np.save(camp / "zeta.npy", np.full((n, z, 500), 0.02, np.float32))
    np.save(camp / "n_active.npy", np.full(n, z, np.int32))
    curves = np.random.default_rng(0).normal(size=(n, stations, freq.size)).astype(np.float32) * 0.05
    np.save(camp / "log_tf.npy", curves)
    np.save(camp / "log_tf1d.npy", np.zeros_like(curves))
    np.save(camp / "valid.npy", np.ones(n, dtype=bool))
    split = np.array(["train"] * 8 + ["val"] * 2 + ["test"] * 2)
    np.save(camp / "split.npy", split)
    np.save(camp / "rec_x.npy", rec_x)
    np.save(camp / "group.npy", np.arange(n, dtype=np.int64))


def test_nonlinear_smoke_records_the_lateral_settings(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    _write_smoke_cache(cache)
    out = tmp_path / "run"
    train = Path(__file__).resolve().parents[1] / "train.py"
    subprocess.run(
        [
            sys.executable, str(train), "--smoke", "--cache", str(cache), "--encoding", "metric",
            "--stage", "0", "--target", "residual", "--seed", "0", "--depth-ae", "--readout", "nonlinear",
            "--recon-weight", "0.1", "--local-min-wavelength", "0", "--hfp-kernel", "4", "--hfp-stride", "4",
            "--out", str(out), "--overwrite",
        ],
        check=True,
    )
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["readout"] == "nonlinear"
    assert cfg["depth_ae"] is True
    assert cfg["local_min_wavelength"] == 0.0
    assert cfg["hfp_stride"] == 4
    assert cfg["hfp_kernel"] == 4
    assert (out / "best_model.pt").is_file()
