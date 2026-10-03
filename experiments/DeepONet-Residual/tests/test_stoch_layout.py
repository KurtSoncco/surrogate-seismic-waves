from __future__ import annotations

import numpy as np
import torch

from data import (
    build_stoch_vector,
    infer_stoch_layout,
    remap_stoch_last_dim,
    stoch_dim,
    stoch_in_features_from_state,
    stoch_layout_from_dim,
    stoch_layout_parts,
)


def test_stoch_dim_xi_cov_is_17():
    assert stoch_dim() == 17
    assert stoch_dim(layout="xi_cov") == 17
    assert stoch_dim(layout="legacy20") == 20
    assert stoch_dim(layout="cov_only") == 1
    assert stoch_dim(layout="xi_field_acf") == 18


def test_build_stoch_drops_rh_ahv_damp():
    xi = np.arange(16, dtype=np.float32)
    v = build_stoch_vector(
        xi_vals=xi, rH=50.0, aHV=20.0, CoV=0.22, xi_damp=0.05, layout="xi_cov"
    )
    assert v.shape == (17,)
    np.testing.assert_allclose(v[:16], xi)
    assert v[16] == np.float32(0.22)
    legacy = build_stoch_vector(
        xi_vals=xi, rH=50.0, aHV=20.0, CoV=0.22, xi_damp=0.05, layout="legacy20"
    )
    assert legacy.shape == (20,)
    np.testing.assert_allclose(legacy[16:], [50.0, 20.0, 0.22, 0.05], atol=1e-6)


def test_infer_stoch_layout_from_linear_width():
    state17 = {"base.stoch_mlp.0.weight": torch.zeros(128, 17)}
    state20 = {"inner.stoch_mlp.0.weight": torch.zeros(128, 20)}
    assert stoch_in_features_from_state(state17) == 17
    assert infer_stoch_layout(state17) == "xi_cov"
    assert infer_stoch_layout(state20) == "legacy20"
    assert infer_stoch_layout(recorded="legacy20") == "legacy20"
    state1 = {"inner.stoch_mlp.0.weight": torch.zeros(32, 1)}
    assert infer_stoch_layout(state1) == "cov_only"
    state18 = {"base.stoch_mlp.0.weight": torch.zeros(128, 18)}
    assert infer_stoch_layout(state18) == "xi_field_acf"
    assert infer_stoch_layout(recorded="xi_field_acf") == "xi_field_acf"


def test_build_stoch_xi_field_acf_is_18():
    xi = np.arange(16, dtype=np.float32)
    v = build_stoch_vector(
        xi_vals=xi,
        rH=80.0,
        aHV=12.0,
        CoV=0.28,
        xi_damp=0.05,
        layout="xi_field_acf",
        acf_length=42.5,
    )
    assert v.shape == (18,)
    np.testing.assert_allclose(v[:16], xi)
    np.testing.assert_allclose(v[16:], [0.28, 42.5], atol=1e-6)


def test_empirical_acf_length_exponential_field():
    from features import empirical_acf_length

    rng = np.random.default_rng(7)
    nx, nz = 512, 24
    r = 40.0
    h = np.arange(nx)
    acf = np.exp(-np.minimum(h, nx - h) / r)
    spec = np.fft.fft(acf).real
    spec = np.maximum(spec, 0.0)
    z = rng.standard_normal(nx)
    field_1d = np.fft.ifft(np.sqrt(spec) * np.fft.fft(z)).real
    vs = np.exp(0.2 * field_1d)[None, :].repeat(nz, axis=0)
    length = empirical_acf_length(vs, dx=1.0)
    assert 20.0 < length < 80.0


def test_stoch_layout_from_dim_rejects_unknown_width():
    assert stoch_layout_from_dim(20) == "legacy20"
    assert stoch_layout_from_dim(18) == "xi_field_acf"
    assert stoch_layout_from_dim(17) == "xi_cov"
    try:
        stoch_layout_from_dim(15)
    except ValueError as exc:
        assert "15" in str(exc)
    else:
        raise AssertionError("expected ValueError for dim 15")


def test_remap_legacy20_keeps_xi_and_cov_not_prefix():
    # Distinct values so a naive prefix slice would pick rH/aHV instead of CoV.
    src = torch.arange(20, dtype=torch.float32)
    cov = remap_stoch_last_dim(src, 17, new_fill=0.0)
    assert tuple(cov.shape) == (17,)
    assert torch.equal(cov[:16], src[:16])
    assert cov[16].item() == src[18].item()  # CoV, not rH
    acf = remap_stoch_last_dim(src, 18, new_fill=-1.0)
    assert tuple(acf.shape) == (18,)
    assert torch.equal(acf[:16], src[:16])
    assert acf[16].item() == src[18].item()
    assert acf[17].item() == -1.0


def test_remap_xi_cov_to_acf_is_suffix_fill():
    src = torch.arange(17, dtype=torch.float32)
    out = remap_stoch_last_dim(src, 18, new_fill=0.0)
    assert torch.equal(out[:17], src)
    assert out[17].item() == 0.0


def test_pad_stoch_stats_legacy20_to_xi_field_acf():
    from train import _pad_stoch_stats_to

    stats = {
        "stoch_mean": torch.arange(20, dtype=torch.float32),
        "stoch_std": torch.full((20,), 2.0),
    }
    _pad_stoch_stats_to(stats, 18)
    assert tuple(stats["stoch_mean"].shape) == (18,)
    assert torch.equal(stats["stoch_mean"][:16], torch.arange(16, dtype=torch.float32))
    assert stats["stoch_mean"][16].item() == 18.0
    assert stats["stoch_mean"][17].item() == 0.0
    assert torch.equal(stats["stoch_std"][:16], torch.full((16,), 2.0))
    assert stats["stoch_std"][16].item() == 2.0
    assert stats["stoch_std"][17].item() == 1.0
    _pad_stoch_stats_to(stats, 17)
    assert tuple(stats["stoch_mean"].shape) == (17,)
    assert stats["stoch_mean"][16].item() == 18.0


def test_compatible_state_remaps_stoch_mlp_from_legacy20():
    from torch import nn

    from train import _compatible_state

    class _Branch(nn.Module):
        def __init__(self, din: int):
            super().__init__()
            self.stoch_mlp = nn.Linear(din, 4, bias=False)

    weight20 = torch.arange(4 * 20, dtype=torch.float32).reshape(4, 20)
    state = {"stoch_mlp.weight": weight20}
    model17 = _Branch(17)
    filtered, skipped = _compatible_state(model17, state)
    assert skipped == []
    got = filtered["stoch_mlp.weight"]
    assert tuple(got.shape) == (4, 17)
    assert torch.equal(got[:, :16], weight20[:, :16])
    assert torch.equal(got[:, 16], weight20[:, 18])
    model18 = _Branch(18)
    filtered18, skipped18 = _compatible_state(model18, state)
    assert skipped18 == []
    got18 = filtered18["stoch_mlp.weight"]
    assert tuple(got18.shape) == (4, 18)
    assert torch.equal(got18[:, :16], weight20[:, :16])
    assert torch.equal(got18[:, 16], weight20[:, 18])
    assert torch.equal(got18[:, 17], torch.zeros(4))


def test_stoch_layout_parts_cover_known_layouts():
    assert sum(w for _, w in stoch_layout_parts("legacy20")) == 20
    assert sum(w for _, w in stoch_layout_parts("xi_field_acf")) == 18
    assert sum(w for _, w in stoch_layout_parts("xi_cov")) == 17
    assert sum(w for _, w in stoch_layout_parts("cov_only")) == 1
