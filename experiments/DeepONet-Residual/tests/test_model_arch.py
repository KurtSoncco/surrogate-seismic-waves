from __future__ import annotations

import numpy as np
import torch

from model import build_model


def _forward_ok(encoder: str, residual_fno: bool) -> None:
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
        field_encoder=encoder,  # type: ignore[arg-type]
        residual_fno=residual_fno,
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
    out.sum().backward()


def test_resunet_forward():
    _forward_ok("resunet", False)


def test_gno_forward():
    _forward_ok("gno", False)


def test_fno_forward():
    _forward_ok("resunet", True)


def test_gino_forward():
    _forward_ok("gno", True)


def test_ufno_forward():
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
        fno_kind="ufno",
    )
    out = model(
        torch.randn(2, 3, 32, n_rec),
        torch.randn(2, 20),
        torch.randn(2, n_rec * n_freq, trunk_dim),
    )
    assert out.shape == (2, n_rec * n_freq)
    out.sum().backward()


def test_ffno_forward():
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
        fno_kind="ffno",
    )
    out = model(
        torch.randn(2, 3, 32, n_rec),
        torch.randn(2, 20),
        torch.randn(2, n_rec * n_freq, trunk_dim),
    )
    assert out.shape == (2, n_rec * n_freq)
    out.sum().backward()


def test_attn_gino_forward():
    _forward_ok("attn", True)


def test_gat_gino_forward():
    _forward_ok("gat", True)


def _fno_kind_ok(kind: str) -> None:
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
        fno_kind=kind,  # type: ignore[arg-type]
    )
    out = model(
        torch.randn(2, 3, 32, n_rec),
        torch.randn(2, 20),
        torch.randn(2, n_rec * n_freq, trunk_dim),
    )
    assert out.shape == (2, n_rec * n_freq)
    out.sum().backward()


def test_afno_forward():
    _fno_kind_ok("afno")


def test_wno_forward():
    _fno_kind_ok("wno")


def test_fno1d_forward():
    _fno_kind_ok("fno1d")


def test_loglo_fno_kind_forward():
    _fno_kind_ok("loglo")


def test_tf_fno_kind_forward():
    _fno_kind_ok("tf")


def test_band2_fno_kind_forward():
    _fno_kind_ok("band2")


def test_hz_band_masks_split_at_2hz():
    from model import hz_band_masks

    f = torch.tensor([0.2, 1.9, 2.0, 5.0, 10.0])
    lo, hi = hz_band_masks(f, split_hz=2.0)
    lo = lo.reshape(-1)
    hi = hi.reshape(-1)
    assert bool(lo[0]) and bool(lo[1]) and not bool(lo[2])
    assert not bool(hi[0]) and bool(hi[2]) and bool(hi[4])
    assert torch.equal(lo, ~hi)


def test_band2_hard_mask_uses_query_freq():
    n_rec, n_freq, trunk_dim = 21, 4, 5
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
        fno_n_modes=(4, 2),
        fno_n_layers=1,
        n_gno_layers=2,
        fno_kind="band2",
    )
    freq = np.array([0.2, 1.0, 4.0, 8.0], dtype=np.float32)
    model.set_query_freq(freq)
    out = model(
        torch.randn(2, 3, 32, n_rec),
        torch.randn(2, 20),
        torch.randn(2, n_rec * n_freq, trunk_dim),
    )
    assert out.shape == (2, n_rec * n_freq)
    assert model.fno is None
    assert model.fno_low is not None and model.fno_high is not None
    lo = model.last_band_low.reshape(-1)
    hi = model.last_band_high.reshape(-1)
    assert bool(lo[0]) and bool(lo[1]) and not bool(lo[2])
    assert bool(hi[2]) and bool(hi[3])
    out.sum().backward()


def test_mirror_vanilla_fno_into_band2():
    from train import _mirror_vanilla_fno_into_band2

    state = {
        "base.fuse.0.weight": torch.ones(2),
        "fno.convs.0.weight": torch.ones(3),
        "proj.weight": torch.ones(4),
        "lift.weight": torch.ones(5),
    }
    out = _mirror_vanilla_fno_into_band2(state)
    assert torch.equal(out["fno_low.convs.0.weight"], state["fno.convs.0.weight"])
    assert torch.equal(out["fno_high.convs.0.weight"], state["fno.convs.0.weight"])
    assert torch.equal(out["proj_high.weight"], state["proj.weight"])
    assert "base.fuse.0.weight" in out


def test_freeze_gno_encoder_grads():
    from model import freeze_gno_encoder, gno_core

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
        fno_kind="vanilla",
    )
    n_frozen = freeze_gno_encoder(model)
    assert n_frozen > 0
    out = model(
        torch.randn(2, 3, 32, n_rec),
        torch.randn(2, 20),
        torch.randn(2, n_rec * n_freq, trunk_dim),
    )
    out.sum().backward()
    core = gno_core(model)
    for p in list(core.col_enc.parameters()) + list(core.gno.parameters()):
        assert p.requires_grad is False
        assert p.grad is None
    assert model.lift.weight.grad is not None
    assert model.proj.weight.grad is not None
    assert model.base.fuse[0].weight.grad is not None


def test_freeze_fno_head_grads():
    from model import freeze_fno_head, gno_core

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
        fno_kind="vanilla",
    )
    n_frozen = freeze_fno_head(model)
    assert n_frozen > 0
    out = model(
        torch.randn(2, 3, 32, n_rec),
        torch.randn(2, 20),
        torch.randn(2, n_rec * n_freq, trunk_dim),
    )
    out.sum().backward()
    assert model.lift.weight.requires_grad is False
    assert model.lift.weight.grad is None
    core = gno_core(model)
    assert core.col_enc.proj.weight.grad is not None
    assert core.gno.layers[0][0].weight.grad is not None


def test_column_encoder_depth_tokens_widen_proj():
    n_rec, n_freq, trunk_dim = 21, 16, 5
    kw = dict(
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
        fno_n_layers=1,
        n_gno_layers=2,
    )
    m1 = build_model("single", col_enc_depth_tokens=1, **kw)
    m8 = build_model("single", col_enc_depth_tokens=8, **kw)
    assert m1.base.col_enc.proj.in_features * 8 == m8.base.col_enc.proj.in_features
    out = m8(
        torch.randn(2, 3, 32, n_rec),
        torch.randn(2, 20),
        torch.randn(2, n_rec * n_freq, trunk_dim),
    )
    assert out.shape == (2, n_rec * n_freq)
    out.sum().backward()
    assert m8.base.col_enc.proj.weight.grad is not None


def test_domain_item_indices_three_layer():
    from train import domain_item_indices

    class _DS:
        domain_names_per_item = ["iid", "iid_extra", "ood_dipping", "ood_three_layer"]

    ds = _DS()
    assert domain_item_indices(ds, "iid") == [0, 1]
    assert domain_item_indices(ds, "ood_dipping") == [2]
    assert domain_item_indices(ds, "three_layer") == [3]


def test_dataset_freq_unwraps_subset():
    from torch.utils.data import Subset

    from train import dataset_freq

    class _DS:
        freq_s = np.array([0.2, 1.0, 4.0])

    inner = _DS()
    wrapped = Subset(inner, [0])
    np.testing.assert_array_equal(dataset_freq(wrapped), inner.freq_s)
    assert dataset_freq(Subset(object(), [0])) is None


def test_compatible_state_skips_mismatched_fno_modes():
    from torch import nn

    from train import _compatible_state

    class _Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.fno_weight = nn.Parameter(torch.ones(2, 8, 16))
            self.lift = nn.Parameter(torch.ones(4))

    model = _Head()
    state = {
        "fno_weight": torch.zeros(2, 8, 32),
        "lift": torch.zeros(4),
        "missing_elsewhere": torch.ones(1),
    }
    filtered, skipped = _compatible_state(model, state)
    assert skipped == ["fno_weight"]
    assert "fno_weight" not in filtered
    assert tuple(filtered["lift"].shape) == (4,)


def test_compatible_state_strips_fno_base_prefix():
    from torch import nn

    from train import _compatible_state

    class _Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.col_enc = nn.Linear(3, 4)

    model = _Enc()
    state = {
        "base.col_enc.weight": torch.ones(4, 3),
        "base.col_enc.bias": torch.ones(4),
        "base.fno.weight": torch.zeros(2),
    }
    filtered, skipped = _compatible_state(model, state)
    assert "col_enc.weight" in filtered
    torch.testing.assert_close(filtered["col_enc.weight"], torch.ones(4, 3))
    assert skipped == []


def test_pad_trunk_stats_zero_mean_unit_std():
    from train import _pad_trunk_stats_to

    stats = {
        "trunk_mean": torch.arange(5, dtype=torch.float32),
        "trunk_std": torch.full((5,), 2.0),
    }
    _pad_trunk_stats_to(stats, 13)
    assert tuple(stats["trunk_mean"].shape) == (13,)
    assert tuple(stats["trunk_std"].shape) == (13,)
    assert torch.equal(stats["trunk_mean"][:5], torch.arange(5, dtype=torch.float32))
    assert torch.equal(stats["trunk_mean"][5:], torch.zeros(8))
    assert torch.equal(stats["trunk_std"][:5], torch.full((5,), 2.0))
    assert torch.equal(stats["trunk_std"][5:], torch.ones(8))
    _pad_trunk_stats_to(stats, 13)  # idempotent


def test_hfp_shape_on_leftover_grid():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "GIFNO" / "spectral_layers.py"
    spec = importlib.util.spec_from_file_location("_gifno_hfp_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    hfp = mod.HighFrequencyPropagation(kernel_size=4, stride=4)
    x = torch.randn(2, 4, 21, 64)
    y = hfp(x)
    assert y.shape == (2, 4, 21, 64)
    assert not torch.allclose(y, x)


def _mscale_kw(**extra):
    n_rec = 21
    kw = dict(
        field_channels=3,
        stoch_dim=20,
        trunk_dim=5,
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
    kw.update(extra)
    return kw, n_rec


def test_mscale_s1_plain_trunk_and_fuse():
    from model import TrunkMLP, gno_core

    kw, n_rec = _mscale_kw()
    model = build_model("single", n_mscale_trunk=1, n_mscale_branch=1, **kw)
    core = gno_core(model)
    assert isinstance(core.trunk, TrunkMLP)
    assert core.fuses is None
    assert any(k.startswith("trunk.net") for k in core.state_dict())
    out = model(
        torch.randn(2, 3, 32, n_rec),
        torch.randn(2, 20),
        torch.randn(2, n_rec * 16, 5),
    )
    assert out.shape == (2, n_rec * 16)
    out.sum().backward()


def test_mscale_trunk4_concat_and_backward():
    from model import MscaleTrunk, gno_core

    kw, n_rec = _mscale_kw()
    model = build_model("single", n_mscale_trunk=4, n_mscale_branch=1, **kw)
    core = gno_core(model)
    assert isinstance(core.trunk, MscaleTrunk)
    assert len(core.trunk.subnets) == 4
    bq = core.trunk(torch.randn(6, 5))
    assert bq.shape == (6, 16)
    out = model(
        torch.randn(2, 3, 32, n_rec),
        torch.randn(2, 20),
        torch.randn(2, n_rec * 16, 5),
    )
    assert out.shape == (2, n_rec * 16)
    out.sum().backward()


def test_mscale_branch4_shares_gno():
    from model import gno_core

    kw, n_rec = _mscale_kw()
    model = build_model("single", n_mscale_trunk=4, n_mscale_branch=4, **kw)
    core = gno_core(model)
    assert core.fuses is not None
    assert len(core.fuses) == 4
    assert not isinstance(core.col_enc, torch.nn.ModuleList)
    assert not isinstance(core.gno, torch.nn.ModuleList)
    n_col = sum(p.numel() for p in core.col_enc.parameters())
    n_gno = sum(p.numel() for p in core.gno.parameters())
    assert n_col > 0 and n_gno > 0
    out = model(
        torch.randn(2, 3, 32, n_rec),
        torch.randn(2, 20),
        torch.randn(2, n_rec * 16, 5),
    )
    assert out.shape == (2, n_rec * 16)
    out.sum().backward()


def test_mscale_t4_trunk_params_not_above_control():
    from model import gno_core

    kw, _ = _mscale_kw(
        latent_dim=128, trunk_hidden=256, trunk_layers=5, residual_fno=False
    )
    m1 = build_model("single", n_mscale_trunk=1, **kw)
    m4 = build_model("single", n_mscale_trunk=4, **kw)
    n1 = sum(p.numel() for p in gno_core(m1).trunk.parameters())
    n4 = sum(p.numel() for p in gno_core(m4).trunk.parameters())
    assert n4 <= int(n1 * 1.20)
    assert len(gno_core(m4).trunk.subnets) == 4


def test_mscale_resunet_forward():
    kw, n_rec = _mscale_kw(field_encoder="resunet", residual_fno=False)
    model = build_model("single", n_mscale_trunk=4, n_mscale_branch=2, **kw)
    out = model(
        torch.randn(2, 3, 32, n_rec),
        torch.randn(2, 20),
        torch.randn(2, n_rec * 16, 5),
    )
    assert out.shape == (2, n_rec * 16)
    out.sum().backward()


def _ablate_forward(model, n_rec: int, stoch_dim: int = 20):
    out = model(
        torch.randn(2, 3, 32, n_rec),
        torch.randn(2, stoch_dim),
        torch.randn(2, n_rec * 16, 5),
    )
    assert out.shape == (2, n_rec * 16)
    out.sum().backward()
    return out


def test_stoch_inject_concat_forward():
    from model import gno_core

    kw, n_rec = _mscale_kw()
    model = build_model(
        "single",
        n_mscale_trunk=4,
        n_mscale_branch=4,
        stoch_inject="concat",
        **kw,
    )
    core = gno_core(model)
    assert core.stoch_mlp is None
    assert core.stoch_inject == "concat"
    fuse0 = core.fuses[0]
    linear = next(m for m in fuse0.modules() if isinstance(m, torch.nn.Linear))
    assert linear.in_features == kw["latent_dim"] + kw["stoch_dim"]
    _ablate_forward(model, n_rec)


def test_fuse_add_forward_t4b4():
    from model import gno_core

    kw, n_rec = _mscale_kw()
    model = build_model(
        "single",
        n_mscale_trunk=4,
        n_mscale_branch=4,
        fuse_kind="add",
        **kw,
    )
    core = gno_core(model)
    assert core.fuse_kind == "add"
    assert core.fuses is None
    assert isinstance(core.fuse, torch.nn.Identity)
    assert core.n_mscale_branch == 4
    _ablate_forward(model, n_rec)


def test_fuse_add_requires_stoch_mlp():
    kw, _ = _mscale_kw()
    try:
        build_model(
            "single",
            n_mscale_trunk=4,
            n_mscale_branch=4,
            stoch_inject="concat",
            fuse_kind="add",
            **kw,
        )
    except ValueError as exc:
        assert "stoch-inject=mlp" in str(exc)
    else:
        raise AssertionError("concat + add should raise")


def test_col_enc_mlp_forward():
    from model import _ColumnMLPEncoder, gno_core

    kw, n_rec = _mscale_kw()
    model = build_model(
        "single", n_mscale_trunk=4, n_mscale_branch=4, col_enc="mlp", **kw
    )
    core = gno_core(model)
    assert isinstance(core.col_enc, _ColumnMLPEncoder)
    assert core.col_enc.proj.in_features == 3 * 16
    _ablate_forward(model, n_rec)


def test_col_enc_attn_forward():
    from model import _ColumnAttnEncoder, gno_core

    kw, n_rec = _mscale_kw()
    model = build_model(
        "single", n_mscale_trunk=4, n_mscale_branch=4, col_enc="attn", **kw
    )
    core = gno_core(model)
    assert isinstance(core.col_enc, _ColumnAttnEncoder)
    _ablate_forward(model, n_rec)


def test_ablation_blob_roundtrip():
    from model import build_from_checkpoint_blob, gno_core

    kw, n_rec = _mscale_kw(stoch_dim=17)
    model = build_model(
        "single",
        n_mscale_trunk=4,
        n_mscale_branch=4,
        stoch_inject="concat",
        col_enc="mlp",
        **kw,
    )
    blob = {
        "branch_mode": "single",
        "field_encoder": "gno",
        "residual_fno": True,
        "n_rec": n_rec,
        "fno_width": 8,
        "fno_n_modes": (4, 4),
        "fno_n_layers": 2,
        "n_gno_layers": 2,
        "n_mscale_trunk": 4,
        "n_mscale_branch": 4,
        "stoch_inject": "concat",
        "col_enc": "mlp",
        "fuse_kind": "mlp",
        "model": model.state_dict(),
    }
    loaded = build_from_checkpoint_blob(
        blob,
        field_channels=3,
        stoch_dim=17,
        trunk_dim=5,
        latent_dim=16,
        field_hidden=8,
        branch_hidden=16,
        trunk_hidden=16,
        trunk_layers=2,
    )
    loaded.load_state_dict(blob["model"])
    core = gno_core(loaded)
    assert core.stoch_mlp is None
    assert core.col_enc_kind == "mlp"
    out = loaded(
        torch.randn(2, 3, 32, n_rec),
        torch.randn(2, 17),
        torch.randn(2, n_rec * 16, 5),
    )
    assert out.shape == (2, n_rec * 16)


def test_gno_dilation_steps_clip():
    from model import gno_dilation_steps

    assert gno_dilation_steps(10.0) == 1
    assert gno_dilation_steps(25.0) == 1
    assert gno_dilation_steps(37.5) == 2
    assert gno_dilation_steps(87.5) == 4
    assert gno_dilation_steps(200.0) == 8
    assert gno_dilation_steps(float("nan")) == 1
    assert gno_dilation_steps(0.0) == 1


def test_rh_dilated_gno_forward_and_skip_params():
    from model import _ChainGNO, apply_gno_dilation, build_model

    n_rec, n_freq, trunk_dim = 21, 16, 5
    model = build_model(
        "single",
        field_channels=3,
        stoch_dim=17,
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
        gno_rh_dilate=True,
    )
    gno = None
    for m in model.modules():
        if isinstance(m, _ChainGNO):
            gno = m
            break
    assert gno is not None
    assert gno.rh_dilate
    assert gno.dilate_layers is not None
    ship = build_model(
        "single",
        field_channels=3,
        stoch_dim=17,
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
        gno_rh_dilate=False,
    )
    ship_keys = set(ship.state_dict())
    extra = [k for k in model.state_dict() if k not in ship_keys]
    assert any("dilate_layers" in k for k in extra)
    apply_gno_dilation(model, torch.tensor([83.56, 25.0]))
    out = model(
        torch.randn(2, 3, 32, n_rec),
        torch.randn(2, 17),
        torch.randn(2, n_rec * n_freq, trunk_dim),
    )
    assert out.shape == (2, n_rec * n_freq)
    out.sum().backward()


def test_kernel_forward_query_ne_support():
    n_s, n_q, n_freq, trunk_dim = 40, 7, 16, 5
    model = build_model(
        "single",
        field_channels=3,
        stoch_dim=17,
        trunk_dim=trunk_dim,
        latent_dim=16,
        field_hidden=8,
        branch_hidden=16,
        trunk_hidden=16,
        trunk_layers=2,
        field_encoder="kernel",
        residual_fno=True,
        n_rec=n_q,
        fno_width=8,
        fno_n_modes=(4, 4),
        fno_n_layers=2,
        n_gno_layers=2,
        kernel_k=2,
    )
    from model import DeepONetFNO, KernelGNO, gno_core

    assert not any(isinstance(m, DeepONetFNO) for m in model.modules())
    assert isinstance(gno_core(model).gno, KernelGNO)
    assert bool(getattr(model, "accepts_query_x", False))
    fields = torch.randn(2, 3, 32, n_s)
    stoch = torch.randn(2, 17)
    trunk = torch.randn(2, n_q * n_freq, trunk_dim)
    query_x = torch.linspace(0.0, 100.0, n_q)
    support_x = torch.linspace(0.0, 100.0, n_s)
    out = model(fields, stoch, trunk, query_x=query_x, support_x=support_x)
    assert out.shape == (2, n_q * n_freq)
    out.sum().backward()


def test_kernel_support_permutation_invariance():
    from model import KernelGNO

    torch.manual_seed(0)
    gno = KernelGNO(8, n_layers=2, k=2)
    nodes = torch.randn(2, 11, 8)
    support_x = torch.linspace(0.0, 50.0, 11)
    query_x = torch.tensor([5.0, 20.0, 41.0])
    p0 = gno(nodes, support_x, query_x)
    perm = torch.randperm(11)
    p1 = gno(nodes[:, perm], support_x[perm], query_x)
    torch.testing.assert_close(p0, p1, rtol=1e-5, atol=1e-5)


def test_kernel_query_on_support_k1():
    from model import KernelGNO

    torch.manual_seed(1)
    gno = KernelGNO(4, n_layers=1, k=1)
    nodes = torch.randn(1, 6, 4)
    support_x = torch.linspace(0.0, 25.0, 6)
    q = support_x[3:4]
    p = gno(nodes, support_x, q)
    dx = torch.zeros(1, 1, 1, 1)
    feat = torch.cat([nodes[:, 3:4].unsqueeze(2), dx, dx.abs()], dim=-1)
    expect = gno.mix(feat).squeeze(2)
    torch.testing.assert_close(p, expect, rtol=1e-5, atol=1e-5)


def test_latent_grid_fno_kernel():
    n_s, n_q, n_freq, trunk_dim = 16, 5, 8, 5
    model = build_model(
        "single",
        field_channels=3,
        stoch_dim=17,
        trunk_dim=trunk_dim,
        latent_dim=16,
        field_hidden=8,
        branch_hidden=16,
        trunk_hidden=16,
        trunk_layers=2,
        field_encoder="kernel",
        residual_fno=False,
        latent_fno=True,
        n_latent=8,
        n_rec=n_q,
        fno_width=8,
        fno_n_modes=(2, 4),
        fno_n_layers=1,
        n_gno_layers=1,
    )
    from model import LatentGridFNO

    assert any(isinstance(m, LatentGridFNO) for m in model.modules())
    out = model(
        torch.randn(2, 3, 16, n_s),
        torch.randn(2, 17),
        torch.randn(2, n_q * n_freq, trunk_dim),
        query_x=torch.linspace(10.0, 90.0, n_q),
        support_x=torch.linspace(0.0, 100.0, n_s),
    )
    assert out.shape == (2, n_q * n_freq)
    out.sum().backward()


def test_freeze_gno_leaves_kernel_trainable():
    from model import KernelGNO, freeze_gno_encoder, gno_core

    model = build_model(
        "single",
        field_channels=3,
        stoch_dim=17,
        trunk_dim=5,
        latent_dim=8,
        field_hidden=8,
        branch_hidden=8,
        trunk_hidden=8,
        trunk_layers=2,
        field_encoder="kernel",
        residual_fno=False,
        n_gno_layers=1,
    )
    n = freeze_gno_encoder(model)
    assert n > 0
    gno = gno_core(model).gno
    assert isinstance(gno, KernelGNO)
    assert all(p.requires_grad for p in gno.parameters())
