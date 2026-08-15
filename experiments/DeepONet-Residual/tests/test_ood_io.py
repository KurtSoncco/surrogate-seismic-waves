from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from ood_io import (
    clamp_residual,
    discover_h5_files,
    nominal_layer_params,
    parse_cache_tag,
    soil_nz_from_params,
    tf_cache_dir,
)


def test_parse_cache_tag():
    assert parse_cache_tag("n2000_seed42") == (2000, 42)
    assert parse_cache_tag("n3000_seed7") == (3000, 7)


def test_nominal_iid_like_attrs():
    p = {"Vs1": 200.0, "Vs2": 800.0, "H_discretized": 40.0, "CoV": 0.2}
    nom = nominal_layer_params(p)
    assert nom["source"] == "attrs_Vs1_H_Vs2"
    assert nom["misspecified"] is False
    assert nom["H"] == 40.0
    assert nom["vs2"] == 800.0


def test_nominal_three_layer_is_misspecified():
    p = {
        "Vs1": 114.0,
        "H1": 11.0,
        "H1_requested": 10.7,
        "H2": 11.0,
        "H2_requested": 10.64,
        "Vs_mid": 498.0,
        "Vs_bedrock": 1105.0,
    }
    nom = nominal_layer_params(p)
    assert nom["misspecified"] is True
    assert nom["source"] == "three_layer_topVs1_totalH_bedrock"
    assert nom["H"] == 22.0
    assert nom["vs1"] == 114.0
    assert nom["vs2"] == 1105.0
    assert nom["true_layers"]["H"] == [11.0, 11.0]
    assert nom["true_layers"]["Vs"] == [114.0, 498.0]
    assert nom["true_layers"]["vs_rock"] == 1105.0
    assert soil_nz_from_params(p, vs_nz=32) == 22


def test_soil_nz_from_H_attr():
    assert soil_nz_from_params({"H": 54.0, "Vs1": 180.0, "Vs2": 800.0}, vs_nz=80) == 54


def test_clamp_residual():
    r = np.array([0.0, 10.0, -10.0], dtype=np.float32)
    z = clamp_residual(r, "zero")
    assert np.allclose(z, 0.0)
    t = clamp_residual(r, "tanh")
    ln3 = np.log(3.0)
    assert abs(t[1]) <= ln3 + 1e-6
    assert np.allclose(clamp_residual(r, "none"), r)


def test_discover_gifno_mini_corpus(tmp_path: Path):
    h5_dir = tmp_path / "ood_dipping" / "h5"
    h5_dir.mkdir(parents=True)
    for i in (0, 2, 10):
        p = h5_dir / f"run_{i}.h5"
        with h5py.File(p, "w") as f:
            f.create_dataset("x", data=[1])
    found = discover_h5_files(tmp_path / "ood_dipping")
    assert [p.name for p in found] == ["run_0.h5", "run_2.h5", "run_10.h5"]
    assert tf_cache_dir(tmp_path / "ood_dipping") is None


def test_discover_via_root_manifest(tmp_path: Path):
    root = tmp_path / "ood_dipping"
    h5_dir = root / "h5"
    h5_dir.mkdir(parents=True)
    for i in (0, 1):
        with h5py.File(h5_dir / f"run_{i}.h5", "w") as f:
            f.create_dataset("x", data=[1])
    (root / "manifest.csv").write_text("index,sobol_id\n0,0\n1,1\n")
    found = discover_h5_files(root)
    assert [p.name for p in found] == ["run_0.h5", "run_1.h5"]


def test_discover_seiskit_flat(tmp_path: Path):
    case = tmp_path / "dipping" / "h5"
    case.mkdir(parents=True)
    with h5py.File(case / "case_0.h5", "w") as f:
        f.create_dataset("x", data=[1])
    found = discover_h5_files(tmp_path / "dipping")
    assert found[0].name == "case_0.h5"
