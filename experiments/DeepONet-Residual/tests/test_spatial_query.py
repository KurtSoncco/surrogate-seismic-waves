"""Kernel-query GNO: support stride, station hold-out, interpolate-p helpers."""

from __future__ import annotations

import numpy as np
import torch

import config
from data import (
    ResidualDeepONetDataset,
    column_x_m,
    query_station_indices,
    support_column_indices,
)
from scoring.eval_spatial_query import _metrics_from_arrays, write_markdown
from model import interp_along_x
from residual_signed import stack_field_columns


def test_support_stride_five_is_100_columns():
    cols = support_column_indices(5, nx=500)
    assert cols.shape == (100,)
    assert cols[0] == 0
    assert cols[-1] == 495
    x = column_x_m(cols)
    assert x.shape == (100,)
    np.testing.assert_allclose(x[0], 0.5)


def test_query_station_splits():
    even = query_station_indices(21, "even")
    odd = query_station_indices(21, "odd")
    interior = query_station_indices(21, "interior")
    edge = query_station_indices(21, "edge")
    assert even.tolist() == list(range(0, 21, 2))
    assert odd.tolist() == list(range(1, 21, 2))
    assert interior[0] == 1 and interior[-1] == 19
    assert edge.tolist() == [0, 20]
    assert len(even) + len(odd) == 21


def test_stack_field_columns_shape():
    vs = np.full((20, 500), 200.0, dtype=np.float32)
    vs[0] = 150.0
    zeta = np.full_like(vs, 0.05)
    cols = np.array([100, 115, 400])
    fld, vc = stack_field_columns(vs, zeta, cols, soil_nz=12, nz=20)
    assert fld.shape == (3, config.NZ_MAX, 3)
    assert vc.shape == (3,)
    assert np.isfinite(fld).all()


def test_interp_along_x_midpoint():
    values = torch.tensor([[[0.0, 10.0], [2.0, 20.0]]])
    x_src = torch.tensor([0.0, 2.0])
    x_dst = torch.tensor([1.0])
    out = interp_along_x(values, x_src, x_dst)
    torch.testing.assert_close(out, torch.tensor([[[1.0, 15.0]]]), rtol=1e-5, atol=1e-5)


def test_query_split_dataset_even(tmp_path):
    n, n_rec, n_freq = 2, 21, 20
    rec = np.arange(100, 401, 15, dtype=np.int64)
    assert rec.shape == (n_rec,)
    meta = {
        "h5_path": np.array(["missing.h5", "missing.h5"]),
        "nz": np.array([16, 16]),
        "H": np.array([40.0, 42.0]),
        "soil_nz": np.array([12, 12]),
        "CoV": np.array([0.2, 0.2]),
        "rH": np.array([40.0, 40.0]),
        "aHV": np.array([20.0, 20.0]),
        "rf_seed": np.array([1, 2]),
        "Vs1": np.array([200.0, 200.0]),
        "Vs2": np.array([1000.0, 1000.0]),
        "xi_damp": np.array([0.05, 0.05]),
    }
    np.savez(tmp_path / "meta.npz", **meta)
    np.save(tmp_path / "r_nom_signed.npy", np.ones((n, n_rec, n_freq), np.float32))
    np.save(tmp_path / "tf1d_nom.npy", np.ones((n, n_rec, n_freq), np.float32))
    np.save(tmp_path / "sample_indices.npy", np.arange(n, dtype=int))
    np.save(tmp_path / "fields.npy", np.ones((n, 3, 16, n_rec), np.float32))
    np.save(tmp_path / "vs_col.npy", np.full((n, n_rec), 200.0, np.float32))
    np.save(tmp_path / "freq.npy", np.logspace(-1, 1, n_freq))
    np.save(tmp_path / "recorder_x.npy", rec)
    ds = ResidualDeepONetDataset(
        tmp_path,
        [0, 1],
        target="R_nom",
        n_freq=8,
        serial_tf1d=True,
        query_split="even",
        support_stride=0,
    )
    assert ds.n_rec == 11
    item = ds[0]
    assert item["fields"].shape[-1] == n_rec
    assert item["query_x"].numel() == 11
    assert item["support_x"].numel() == n_rec
    assert item["target"].numel() == 11 * len(ds.f_idx)
    assert item["trunk_y"].shape[0] == 11 * len(ds.f_idx)


def test_spatial_query_markdown(tmp_path):
    report = {
        "hold_out": "odd",
        "train_split": "even",
        "domains": {
            "iid": {
                "kernel": {"pearson_R_freq": 0.6},
                "interpolate_p": {"pearson_R_freq": 0.4},
                "haskell": {"pearson_R_freq": 0.1},
                "beats_interpolate_p": True,
            }
        },
    }
    path = tmp_path / "spatial.md"
    write_markdown(report, path)
    text = path.read_text()
    assert "odd" in text
    assert "0.600" in text or "0.6" in text
    assert "yes" in text


def test_metrics_from_arrays_haskell_zero():
    rng = np.random.default_rng(0)
    r = rng.normal(size=(4, 3, 16))
    tf1d = np.exp(rng.normal(size=r.shape))
    tf2d = tf1d + r
    freq = np.logspace(-1, 1, 16)
    mets = _metrics_from_arrays(
        r, np.zeros_like(r), tf2d, tf1d, freq, n_rec=3, n_freq=16
    )
    assert mets["rel_l2_R"] > 0.5
    assert np.isfinite(mets["pearson_R_freq"])
    assert np.isfinite(mets["anderson_mean"])


def test_slice_support_from_parent(tmp_path, monkeypatch):
    import residual_signed as rs

    monkeypatch.setattr(rs.config, "CACHE_DIR", tmp_path)
    parent = tmp_path / "n7680_seed42"
    child = tmp_path / "n1000_seed42"
    parent.mkdir()
    child.mkdir()
    np.save(parent / "sample_indices.npy", np.arange(8, dtype=int))
    np.save(child / "sample_indices.npy", np.array([1, 4, 7], dtype=int))
    np.save(
        parent / "fields_support.npy",
        np.arange(8 * 3 * 2 * 4, dtype=np.float32).reshape(8, 3, 2, 4),
    )
    np.save(parent / "support_x.npy", np.linspace(0, 1, 4, dtype=np.float32))
    np.save(parent / "support_cols.npy", np.arange(4, dtype=np.int64))
    rs.slice_support_fields_from_parent("n1000_seed42", "n7680_seed42")
    got = np.load(child / "fields_support.npy")
    assert got.shape == (3, 3, 2, 4)
    np.testing.assert_array_equal(got[0], np.load(parent / "fields_support.npy")[1])
