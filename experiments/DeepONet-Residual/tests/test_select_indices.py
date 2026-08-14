"""Tests for nested stratified screen indices."""

from __future__ import annotations

import config
import numpy as np
import pytest
import select_indices as sel
from residual_target import stratified_sample_indices


def test_parse_cache_tag():
    assert sel.parse_cache_tag("n2000_seed42") == (2000, 42)
    with pytest.raises(ValueError):
        sel.parse_cache_tag("n2000")


def test_nested_prefix_from_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "RESIDUAL_CACHE_DIR", tmp_path / "residual_cache")
    config.CACHE_DIR.mkdir(parents=True)
    config.RESIDUAL_CACHE_DIR.mkdir(parents=True)

    manifest = [
        {"CoV": str(0.1 + (i % 3) * 0.1), "H_discretized": str(20 + (i % 5) * 10)}
        for i in range(80)
    ]
    parent = stratified_sample_indices(manifest, 30, seed=42)
    (config.CACHE_DIR / "n30_seed42").mkdir()
    np.save(config.CACHE_DIR / "n30_seed42" / "sample_indices.npy", parent)

    idx20 = sel.resolve_sample_indices("n20_seed42", n_max=30, write=True)
    idx10 = sel.resolve_sample_indices("n10_seed42", n_max=30, write=True)
    assert len(idx20) == 20
    assert len(idx10) == 10
    # n10 is nested in n20 when both come from the same parent prefix
    assert set(idx10).issubset(set(idx20))
    assert set(idx20).issubset(set(parent))


def test_residual_cache_preferred(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "RESIDUAL_CACHE_DIR", tmp_path / "residual_cache")
    config.CACHE_DIR.mkdir(parents=True)
    residual_dir = config.RESIDUAL_CACHE_DIR / "n5_seed42"
    residual_dir.mkdir(parents=True)
    residual_idx = np.array([1, 3, 5, 7, 9], dtype=int)
    np.save(residual_dir / "sample_indices.npy", residual_idx)

    idx = sel.resolve_sample_indices("n5_seed42", n_max=30, write=True)
    assert np.array_equal(idx, residual_idx)


def test_generates_parent_and_extends_residual(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "RESIDUAL_CACHE_DIR", tmp_path / "residual_cache")
    config.CACHE_DIR.mkdir(parents=True)
    config.RESIDUAL_CACHE_DIR.mkdir(parents=True)

    nested = np.array([2, 4, 6, 8, 10], dtype=int)
    (config.RESIDUAL_CACHE_DIR / "n5_seed7").mkdir()
    np.save(
        config.RESIDUAL_CACHE_DIR / "n5_seed7" / "sample_indices.npy", nested
    )

    manifest = [
        {"CoV": str(0.1 + (i % 3) * 0.1), "H_discretized": str(20 + (i % 5) * 10)}
        for i in range(80)
    ]
    monkeypatch.setattr(sel, "load_manifest", lambda: manifest)

    idx = sel.resolve_sample_indices("n12_seed7", n_max=30, write=True)
    assert len(idx) == 12
    assert np.array_equal(idx[:5], nested)
    assert (config.CACHE_DIR / "n30_seed7" / "sample_indices.npy").is_file()
    assert (config.CACHE_DIR / "n12_seed7" / "sample_indices.npy").is_file()
