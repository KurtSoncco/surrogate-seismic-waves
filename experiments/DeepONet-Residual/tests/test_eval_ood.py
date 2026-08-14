"""OOD Haskell floor on synthetic dipping / three-layer H5."""

from __future__ import annotations

from pathlib import Path

import eval_ood
import h5py
import numpy as np


def _write_h5(
    path: Path,
    *,
    nz: int = 32,
    nx: int = 1500,
    vs1: float = 200.0,
    vs2: float = 800.0,
    H: float = 20.0,
    dip: bool = False,
    three_layer: bool = False,
) -> None:
    vs = np.full((nz, nx), vs2, dtype=np.float32)
    zeta = np.full((nz, nx), 0.05, dtype=np.float32)
    soil_nz = int(H)
    if dip:
        for j in range(nx):
            local_h = int(10 + 20 * j / max(nx - 1, 1))
            vs[:local_h, j] = vs1
    elif three_layer:
        vs[:8, :] = vs1
        vs[8:16, :] = 0.5 * (vs1 + vs2)
        vs[16:, :] = vs2
    else:
        vs[:soil_nz, :] = vs1
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
        params.attrs["Vs1"] = vs1
        params.attrs["Vs2"] = vs2
        params.attrs["H"] = H
        params.attrs["H_discretized"] = H
        params.attrs["CoV"] = 0.2
        params.attrs["rf_seed"] = 0
        params.attrs["rH"] = 30.0
        params.attrs["aHV"] = 10.0


def test_crop_variability_full_and_narrow():
    vs_full = np.arange(128 * 1500, dtype=np.float32).reshape(128, 1500)
    zeta_full = np.ones((128, 1500), dtype=np.float32)
    vs_c, zeta_c = eval_ood.crop_variability(vs_full, zeta_full)
    assert vs_c.shape == (128, 500)
    assert np.array_equal(vs_c, vs_full[:, 500:1000])
    assert zeta_c.shape == (128, 500)

    vs_mid = np.ones((32, 700), dtype=np.float32)
    zeta_mid = np.zeros((32, 700), dtype=np.float32)
    vs_c, zeta_c = eval_ood.crop_variability(vs_mid, zeta_mid)
    assert vs_c.shape == (32, 500)


def test_discover_h5_gifno_and_flat(tmp_path: Path):
    root = tmp_path / "ood_dipping"
    _write_h5(root / "h5" / "run_0.h5")
    _write_h5(root / "h5" / "nested" / "case_1.h5")
    found = eval_ood.discover_h5(root)
    assert len(found) == 2


def test_haskell_floor_three_layer_nom_worse_than_col(tmp_path: Path):
    root = tmp_path / "ood_three_layer"
    h5_path = root / "h5" / "case_0.h5"
    _write_h5(h5_path, three_layer=True, vs1=180.0, vs2=900.0, H=16.0)
    rec = eval_ood._read_h5(h5_path)
    vs_crop, zeta_crop = eval_ood.crop_variability(rec["vs"], rec["zeta"])
    freq = np.logspace(-1, 1, 32)
    rec_x = np.array([50, 250, 450], dtype=int)
    tf_nom, tf_col, _ = eval_ood.haskell_floors(
        vs_crop, zeta_crop, rec["params"], freq, rec_x
    )
    # Use column Haskell as a stand-in for 2D truth: nom should be farther.
    assert eval_ood.rel_l2(tf_nom, tf_col) > 0.05


def test_evaluate_h5_uses_tf_true_cache(tmp_path: Path):
    root = tmp_path / "ood_dipping"
    h5_path = root / "h5" / "case_0.h5"
    _write_h5(h5_path, dip=True)
    rec = eval_ood._read_h5(h5_path)
    vs_crop, zeta_crop = eval_ood.crop_variability(rec["vs"], rec["zeta"])
    freq = np.logspace(-1, 1, 24)
    rec_x = eval_ood.recorder_x_indices(nx=vs_crop.shape[1])[:4]
    _tf_nom, tf_col, _ = eval_ood.haskell_floors(
        vs_crop, zeta_crop, rec["params"], freq, rec_x
    )
    # Cached "truth" = column Haskell so col rel L2 ~ 0
    np.save(h5_path.parent / "tf_true.npy", tf_col[:4])
    np.save(h5_path.parent / "freq.npy", freq)
    row = eval_ood.evaluate_h5(h5_path, root, n_freq=24)
    assert row["haskell_col"]["rel_l2"] < 1e-5
    assert row["haskell_nom"]["rel_l2"] >= 0.0
