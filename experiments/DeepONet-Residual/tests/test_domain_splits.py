from __future__ import annotations

from pathlib import Path

import numpy as np
from data import make_splits
from domain_splits import IID_N, write_iid_splits


def test_iid_split_sizes():
    splits = make_splits(IID_N, seed=42)
    assert len(splits.train) == 700
    assert len(splits.val) == 150
    assert len(splits.test) == 150
    assert len(set(splits.train) | set(splits.val) | set(splits.test)) == IID_N


def test_ood_split_sizes_formula():
    splits = make_splits(960, seed=42)
    assert len(splits.train) == 672
    assert len(splits.val) == 144
    assert len(splits.test) == 144


def test_write_iid_splits(tmp_path: Path, monkeypatch):
    import domain_splits
    import config

    monkeypatch.setattr(domain_splits, "SPLIT_DIR", tmp_path)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    path = write_iid_splits(seed=42, n=100)
    blob = np.load(path)
    assert len(blob["train"]) + len(blob["val"]) + len(blob["test"]) == 100


def test_combined_dataset_concat():
    import torch
    from data import CombinedResidualDataset

    class _Stub:
        def __init__(self, n: int, n_rec: int = 21):
            self._cache = [
                {"target": torch.zeros(n_rec), "trunk_y": torch.zeros(n_rec, 4)}
                for _ in range(n)
            ]
            self.n_rec = n_rec
            self.f_idx = np.arange(10)
            self.trunk_names = ["f_star"]
            self.serial_tf1d = False

    ds = CombinedResidualDataset([_Stub(3), _Stub(2)])
    assert len(ds) == 5
    assert ds.n_rec == 21
