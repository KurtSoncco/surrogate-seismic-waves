from __future__ import annotations

import numpy as np

from mix_ladder import extra_local_indices, is_iid_only_mix


def test_extra_local_no_parent_leak():
    parent = np.array([10, 20, 30, 40])
    child = np.array([10, 99, 20, 88, 30, 77, 40, 66])
    extra = extra_local_indices(parent, child, n_extra=3, seed=42)
    assert len(extra) == 3
    got = {int(child[i]) for i in extra}
    assert got.isdisjoint({10, 20, 30, 40})
    assert got <= {99, 88, 77, 66}


def test_extra_local_all_outside():
    parent = np.arange(4)
    child = np.arange(10)
    extra = extra_local_indices(parent, child, n_extra=6, seed=0)
    assert len(extra) == 6
    got = {int(child[i]) for i in extra}
    assert got == {4, 5, 6, 7, 8, 9}


def test_extra_local_reproducible():
    parent = np.arange(5)
    child = np.arange(15)
    a = extra_local_indices(parent, child, 4, seed=42)
    b = extra_local_indices(parent, child, 4, seed=42)
    np.testing.assert_array_equal(a, b)


def test_is_iid_only_mix():
    assert is_iid_only_mix("IID2000")
    assert is_iid_only_mix("IID7680")
    assert not is_iid_only_mix("M1400")
    assert not is_iid_only_mix("M7680")


def test_iid2000_train_no_ood_no_test_leak(tmp_path, monkeypatch):
    import config
    from mix_ladder import mix_train_parts, mix_val_parts

    n1000 = tmp_path / "n1000_seed42"
    n2000 = tmp_path / "n2000_seed42"
    splits = tmp_path / "splits"
    n1000.mkdir()
    n2000.mkdir()
    splits.mkdir()
    parent = np.arange(10, dtype=int)
    child = np.arange(20, dtype=int)
    np.save(n1000 / "sample_indices.npy", parent)
    np.save(n2000 / "sample_indices.npy", child)
    np.savez(
        splits / "iid_n1000_seed42.npz",
        train=np.arange(7),
        val=np.array([7, 8]),
        test=np.array([9]),
    )
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)

    parts = mix_train_parts("IID2000")
    names = [n for n, _, _ in parts]
    assert "ood_dipping" not in names
    assert "ood_three_layer" not in names
    assert names == ["iid", "iid_extra"]
    extra_local = parts[1][2]
    extra_global = {int(child[i]) for i in extra_local}
    assert extra_global.isdisjoint(set(int(x) for x in parent))
    assert 9 not in extra_global
    assert extra_global == set(range(10, 20))

    val = mix_val_parts(iid_only=True)
    assert [n for n, _, _ in val] == ["iid"]
    np.testing.assert_array_equal(val[0][2], np.array([7, 8]))


def test_iid7680_train_no_ood(tmp_path, monkeypatch):
    import config
    from mix_ladder import mix_train_parts, mix_val_lookup, mix_val_parts

    n1000 = tmp_path / "n1000_seed42"
    n7680 = tmp_path / "n7680_seed42"
    splits = tmp_path / "splits"
    n1000.mkdir()
    n7680.mkdir()
    splits.mkdir()
    parent = np.arange(10, dtype=int)
    child = np.arange(30, dtype=int)
    np.save(n1000 / "sample_indices.npy", parent)
    np.save(n7680 / "sample_indices.npy", child)
    np.savez(
        splits / "iid_n1000_seed42.npz",
        train=np.arange(7),
        val=np.array([7, 8]),
        test=np.array([9]),
    )
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)

    parts = mix_train_parts("IID7680")
    names = [n for n, _, _ in parts]
    assert names == ["iid", "iid_extra"]
    extra_global = {int(child[i]) for i in parts[1][2]}
    assert extra_global == set(range(10, 30))
    assert extra_global.isdisjoint(set(int(x) for x in parent))
    val = mix_val_parts(iid_only=True)
    assert [n for n, _, _ in val] == ["iid"]
    lookup = mix_val_lookup(iid_only=True)
    assert set(lookup) == {"iid"}
    np.testing.assert_array_equal(lookup["iid"][1], np.array([7, 8]))


def test_materialize_signed_from_parent(tmp_path, monkeypatch):
    import config
    from ood_signed_cache import materialize_signed_from_parent

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    parent = tmp_path / "n7680_seed42"
    parent.mkdir()
    g = np.array([10, 20, 30, 40, 50], dtype=int)
    np.save(parent / "sample_indices.npy", g)
    rng = np.random.default_rng(0)
    r = rng.normal(size=(5, 4)).astype(np.float32)
    np.save(parent / "r_nom_signed.npy", r)
    np.save(parent / "tf1d_nom.npy", r + 1)
    np.save(parent / "fields.npy", rng.normal(size=(5, 3, 2, 2)).astype(np.float32))
    np.savez(parent / "meta.npz", sample_idx=g, CoV=np.arange(5.0))
    child = tmp_path / "n2000_seed42"
    child.mkdir()
    np.save(child / "sample_indices.npy", np.array([20, 50, 10], dtype=int))
    out = materialize_signed_from_parent("n2000_seed42", "n7680_seed42")
    assert out == child
    got = np.load(child / "r_nom_signed.npy")
    np.testing.assert_array_equal(got, r[[1, 4, 0]])
    np.testing.assert_array_equal(np.load(child / "sample_indices.npy"), [20, 50, 10])
