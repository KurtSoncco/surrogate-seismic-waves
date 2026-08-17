from __future__ import annotations

import numpy as np
from residual_signed import (
    _load_existing_indices,
    _nested_parent_tag,
    write_sample_indices,
)
from residual_target import stratified_sample_indices


def _fake_manifest(n: int = 80) -> list[dict[str, str]]:
    rng = np.random.default_rng(0)
    cov = rng.uniform(0.05, 0.4, size=n)
    H = rng.uniform(20.0, 90.0, size=n)
    return [
        {"CoV": str(c), "H_discretized": str(h), "h5_path": f"run_{i}.h5"}
        for i, (c, h) in enumerate(zip(cov, H))
    ]


def test_stratified_nested_same_seed():
    man = _fake_manifest(80)
    a = {int(i) for i in stratified_sample_indices(man, 20, seed=42)}
    b = {int(i) for i in stratified_sample_indices(man, 40, seed=42)}
    assert a.issubset(b)
    assert len(b) == 40


def test_nested_parent_tag():
    assert _nested_parent_tag("n2000_seed42") == "n1000_seed42"
    assert _nested_parent_tag("n3000_seed42") == "n2000_seed42"
    assert _nested_parent_tag("n7680_seed42") == "n1000_seed42"
    assert _nested_parent_tag("n1000_seed42") is None


def test_write_and_load_indices(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "RESIDUAL_CACHE_DIR", tmp_path / "no_residual")
    idx = np.array([1, 4, 9], dtype=int)
    write_sample_indices("n3_seed42", idx)
    loaded = _load_existing_indices("n3_seed42")
    assert loaded is not None
    np.testing.assert_array_equal(loaded, idx)
