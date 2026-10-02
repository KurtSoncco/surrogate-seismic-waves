"""TF atlas pagination / f0 sort / covering axes (no checkpoint, no H5)."""

from __future__ import annotations

import numpy as np
import pytest

from response_variability.names import (
    DMULT,
    GINO,
    HASKELL_NOMINAL,
    OPENSEES,
    PASSERI,
    PASSERI_DIP,
    PASSERI_FIXED,
    PRETELL,
    TORO,
    TORO_DIP,
    TORO_FIXED,
)
from response_variability.plot_iid import select_f0_quantile_indices
from response_variability.sobol_cover import (
    DIPPING_COVER_KEYS,
    IID_COVER_KEYS,
    cover_keys,
    unique_overlap,
)
from response_variability.tf_atlas import (
    ATLAS_METHODS,
    ATLAS_YLIM,
    PAGE_SIZE,
    atlas_methods_in,
    atlas_panel_title,
    concat_heldout,
    page_index_slices,
    sort_indices_by_extracted_f0,
)


def test_cover_keys_are_6d_and_7d():
    assert IID_COVER_KEYS == ("vs1", "H", "cov", "rH", "aHV", "vs2")
    assert len(IID_COVER_KEYS) == 6
    assert DIPPING_COVER_KEYS == IID_COVER_KEYS + ("dip_angle_deg",)
    assert len(DIPPING_COVER_KEYS) == 7
    assert cover_keys("iid") == IID_COVER_KEYS
    assert cover_keys("dipping") == DIPPING_COVER_KEYS
    with pytest.raises(ValueError):
        cover_keys("three_layer")


def test_page_index_slices_cover_every_index_once():
    for n in (0, 1, 15, 16, 17, 300, 288):
        pages = page_index_slices(n, PAGE_SIZE)
        if n == 0:
            assert pages == []
            continue
        cat = np.concatenate(pages)
        np.testing.assert_array_equal(cat, np.arange(n))
        assert all(1 <= len(p) <= PAGE_SIZE for p in pages)
        assert len(pages) == (n + PAGE_SIZE - 1) // PAGE_SIZE


def test_sort_indices_by_extracted_f0_is_monotone():
    f0 = np.array([1.2, 0.4, np.nan, 3.1, 0.4, 2.0])
    pack = {"f0": f0}
    idx = sort_indices_by_extracted_f0(pack)
    finite = idx[:-1]  # nan last
    vals = f0[finite]
    assert np.all(np.diff(vals[np.isfinite(vals)]) >= 0)
    assert not np.isfinite(f0[idx[-1]])
    assert len(idx) == len(f0)


def test_f0_quantile_highlight_has_16_unique_sorted():
    n = 80
    pack = {"f0": np.linspace(0.28, 4.3, n)}
    idx = select_f0_quantile_indices(pack, n=PAGE_SIZE)
    assert len(idx) == PAGE_SIZE
    assert len(set(idx.tolist())) == PAGE_SIZE
    assert np.all(np.diff(pack["f0"][idx]) >= 0)


def _mini_pack(n: int, *, split: str, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    freq = np.logspace(-1, 1, 8)
    ops = rng.random((n, 21, 8)) + 0.5
    return {
        "tf_opensees": ops,
        "tf_gino": ops * 0.95 + 0.02,
        "tf_haskell_nominal": ops * 0.7 + 0.2,
        "tf_pretell": ops.mean(axis=1) * 0.9,
        "tf_toro": ops.mean(axis=1) * 0.8,
        "tf_passeri": ops.mean(axis=1) * 0.78,
        "tf_dmult": ops.mean(axis=1) * 0.5,
        "freq": freq,
        "vs1": np.linspace(120.0, 300.0, n),
        "H": np.linspace(20.0, 90.0, n),
        "vs2": np.linspace(900.0, 1200.0, n),
        "cov": np.linspace(0.1, 0.3, n),
        "rH": np.linspace(20.0, 80.0, n),
        "aHV": np.linspace(12.0, 40.0, n),
        "f0": np.linspace(0.4, 3.0, n),
        "rf_seed": np.arange(n, dtype=int) + seed * 1000,
        "split": np.full(n, split),
        "domain": np.array("iid"),
        "local_idx": np.arange(n),
    }


def test_concat_heldout_tags_val_then_test():
    val = _mini_pack(4, split="val", seed=0)
    test = _mini_pack(3, split="test", seed=1)
    test["vs1"] = test["vs1"] + 10.0
    out = concat_heldout(val, test)
    assert out["tf_opensees"].shape[0] == 7
    assert out["tf_haskell_nominal"].shape[0] == 7
    assert list(out["split"]) == ["val"] * 4 + ["test"] * 3
    np.testing.assert_array_equal(out["vs1"][:4], val["vs1"])
    np.testing.assert_array_equal(out["vs1"][4:], test["vs1"])
    np.testing.assert_array_equal(out["freq"], val["freq"])


def test_atlas_methods_include_1d_and_not_p84():
    pack = _mini_pack(2, split="val", seed=2)
    methods = atlas_methods_in(pack)
    assert DMULT not in ATLAS_METHODS
    assert DMULT not in methods
    assert methods == [OPENSEES, GINO, HASKELL_NOMINAL, PRETELL, TORO, PASSERI]
    assert TORO_FIXED in ATLAS_METHODS
    assert TORO_DIP in ATLAS_METHODS
    assert PASSERI_FIXED in ATLAS_METHODS
    assert PASSERI_DIP in ATLAS_METHODS
    assert HASKELL_NOMINAL in methods
    assert OPENSEES in methods
    assert GINO in methods
    assert TORO in methods
    assert PASSERI in methods
    assert PRETELL in methods
    title = atlas_panel_title(pack, 0)
    assert "f_0" in title
    assert "val" in title
    assert r"V_{s2}" in title
    assert r"r_H" in title
    assert r"a_{HV}" in title
    assert "CoV" in title
    assert "seed=2000" in title.split("\n")[-1]
    assert "dip" not in title


def test_atlas_title_includes_dip_on_dipping():
    pack = _mini_pack(1, split="test", seed=3)
    pack["dip_angle_deg"] = np.array([-1.75])
    title = atlas_panel_title(pack, 0)
    assert "dip=" in title
    assert r"^\circ" in title
    assert "seed=3000" in title.split("\n")[-1]
    assert "test" in title


def test_ood_dipping_overlay_maps_sobol_and_drops_dmult(tmp_path):
    import h5py

    from response_variability.ood_dipping_box import apply_ood_dipping_toro_passeri

    root = tmp_path / "ood_dipping"
    (root / "toro_comparison").mkdir(parents=True)
    (root / "passeri_comparison").mkdir()
    with h5py.File(root / "toro_comparison" / "pearson.h5", "w") as handle:
        handle.create_dataset("index", data=np.array([0, 1, 2]))
        handle.create_dataset("sobol_id", data=np.array([1, 0, 1]))
    with h5py.File(root / "toro_comparison" / "ensembles.h5", "w") as handle:
        handle.create_dataset("sobol_id", data=np.array([0, 1], dtype=np.int32))
        handle.create_dataset(
            "toro_fixed_geomean",
            data=np.array([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.float32),
        )
        handle.create_dataset("toro_fixed_sigma_ln", data=np.zeros((2, 4), dtype=np.float32))
        handle.create_dataset(
            "toro_dip_geomean",
            data=np.array([[3, 3, 3, 3], [4, 4, 4, 4]], dtype=np.float32),
        )
        handle.create_dataset("toro_dip_sigma_ln", data=np.ones((2, 4), dtype=np.float32))
    with h5py.File(root / "passeri_comparison" / "ensembles.h5", "w") as handle:
        handle.create_dataset("sobol_id", data=np.array([0, 1], dtype=np.int32))
        handle.create_dataset(
            "passeri_fixed_geomean",
            data=np.array([[5, 5, 5, 5], [6, 6, 6, 6]], dtype=np.float32),
        )
        handle.create_dataset("passeri_fixed_sigma_ln", data=np.zeros((2, 4), dtype=np.float32))
        handle.create_dataset(
            "passeri_dip_geomean",
            data=np.array([[7, 7, 7, 7], [8, 8, 8, 8]], dtype=np.float32),
        )
        handle.create_dataset("passeri_dip_sigma_ln", data=np.ones((2, 4), dtype=np.float32))
    center = np.arange(12, dtype=np.float32).reshape(3, 4)
    with h5py.File(root / "toro_comparison" / "tf_2d_center.h5", "w") as handle:
        handle.create_dataset("tf_center", data=center)
    pack = {
        "sample_idx": np.array([2, 0, 1]),
        "tf_opensees": np.zeros((3, 5, 4)),
        "tf_toro": np.ones((3, 4)),
        "tf_passeri": np.ones((3, 4)),
        "tf_dmult": np.ones((3, 4)),
        "tf_dmult_p84": np.ones((3, 4)),
        "sigma_ln_dmult": np.ones((3, 4)),
    }
    out = apply_ood_dipping_toro_passeri(pack, box_root=root)
    assert "tf_dmult" not in out
    assert "tf_dmult_p84" not in out
    assert "tf_toro" not in out
    assert "tf_passeri" not in out
    np.testing.assert_array_equal(out["sobol_id"], [1, 1, 0])
    np.testing.assert_allclose(out["tf_toro_dip"][0], [4, 4, 4, 4])
    np.testing.assert_allclose(out["tf_toro_fixed"][2], [1, 1, 1, 1])
    np.testing.assert_allclose(out["tf_passeri_dip"][0], [8, 8, 8, 8])
    np.testing.assert_allclose(out["tf_opensees"][:, 2, :], center[[2, 0, 1]])


def test_atlas_ylim_is_fixed():
    assert ATLAS_YLIM == (1e-1, 5e1)


def test_unique_overlap_counts_held_only():
    train = np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]])
    held = np.array([[1.0, 2.0], [9.0, 8.0]])
    rec = unique_overlap(train, held)
    assert rec["n_train"] == 2
    assert rec["n_held"] == 2
    assert rec["n_overlap"] == 1
    assert rec["n_held_only"] == 1
