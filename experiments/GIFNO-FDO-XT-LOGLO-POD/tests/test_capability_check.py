"""Tests for capability_check.py (input build, cache, comparison)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import config

config.setup_import_paths()

from capability_check import (  # noqa: E402
    _gt_cache_valid,
    build_input_from_h5,
    case_slug,
    compare_tfs,
    default_capability_h5_paths,
    load_or_compute_ground_truth,
)


def _write_minimal_h5(path: Path, *, nz: int = 25, nx: int = 1500) -> None:
    vs = np.linspace(150.0, 800.0, nz * nx, dtype=np.float32).reshape(nz, nx)
    zeta = np.full((nz, nx), 0.025, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("Vs_realization_2D", data=vs)
        f.create_dataset("Damping_zeta", data=zeta)
        grid = f.create_group("grid")
        grid.attrs["Lx"] = float(nx)
        grid.attrs["Lz"] = float(nz)
        grid.attrs["dx"] = 1.0
        grid.attrs["dz"] = 1.0
        grid.attrs["dt"] = 0.01
        params = f.create_group("params")
        params.attrs["f0_effective"] = 3.0
        accel = f.create_group("recorders").create_group("accel")
        n_time, n_ch = 100, 42
        accel.create_dataset("time", data=np.linspace(0, 1, n_time, dtype=np.float32))
        accel.create_dataset(
            "data",
            data=np.random.randn(n_time, n_ch).astype(np.float32),
        )


def test_case_slug():
    p = Path("/tmp/seiskit/neural-operator/experiments/three_layer/h5/case_0.h5")
    assert case_slug(p) == "three_layer_case_0"


def test_build_input_shape(tmp_path: Path):
    h5_path = tmp_path / "case_0.h5"
    _write_minimal_h5(h5_path)
    x = build_input_from_h5(h5_path)
    assert x.shape == (4, config.NZ_MAX, config.NX)
    assert x.dtype == np.float32
    assert np.allclose(x[0, 0, :5], 1.0)


def test_gt_cache_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    h5_path = tmp_path / "case_0.h5"
    _write_minimal_h5(h5_path)
    case_dir = tmp_path / "out" / "three_layer_case_0"

    def fake_gt(_h5: Path):
        freq = np.logspace(-1, 1, config.N_FREQ, dtype=np.float32)
        tf = np.ones((config.N_LATERAL, config.N_FREQ), dtype=np.float32)
        return tf, freq

    monkeypatch.setattr(
        "capability_check.compute_ground_truth_tf", fake_gt, raising=True
    )

    tf1, freq1 = load_or_compute_ground_truth(h5_path, case_dir)
    assert tf1.shape == (config.N_LATERAL, config.N_FREQ)
    assert _gt_cache_valid(h5_path, case_dir)

    calls = {"n": 0}

    def counting_gt(_h5: Path):
        calls["n"] += 1
        return fake_gt(_h5)

    monkeypatch.setattr(
        "capability_check.compute_ground_truth_tf", counting_gt, raising=True
    )
    tf2, freq2 = load_or_compute_ground_truth(h5_path, case_dir)
    assert calls["n"] == 0
    np.testing.assert_array_equal(tf1, tf2)
    np.testing.assert_array_equal(freq1, freq2)


def test_compare_tfs_writes_metrics(tmp_path: Path):
    n_rec = config.N_LATERAL
    n_freq = config.N_FREQ
    freq = np.logspace(-1, 1, n_freq, dtype=np.float32)
    tf_true = (
        np.exp(-((np.log10(freq) - 0.0) ** 2))[None, :]
        * np.linspace(1.0, 2.0, n_rec)[:, None]
    )
    tf_pred = tf_true * 1.05
    metrics = compare_tfs(
        tf_pred.astype(np.float32),
        tf_true.astype(np.float32),
        freq,
        case_name="test_case",
        out_dir=tmp_path / "plots",
    )
    assert "rel_l2_mean" in metrics
    assert "logspec_rel_l2_mean" in metrics
    assert metrics["tf_space"] == "linear_amplitude"
    assert metrics["rel_l2_mean"] < 0.2
    assert metrics["logspec_rel_l2_mean"] < 0.2
    assert (tmp_path / "plots" / "metrics.json").is_file()
    assert (tmp_path / "plots" / "comparison_central.png").is_file()


def test_default_paths_skip_missing_root(tmp_path: Path):
    assert default_capability_h5_paths(tmp_path / "nonexistent") == []
