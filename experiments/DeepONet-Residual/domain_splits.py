"""Fixed seed-42 train/val/test splits for IID n=1000 and Box OOD corpora."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import config
from data import make_splits
from ood_io import discover_h5_files, default_ood_roots

SPLIT_DIR = config.CACHE_DIR / "splits"
IID_N = 1000


def write_iid_splits(*, seed: int = config.SEED, n: int = IID_N) -> Path:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    splits = make_splits(n, seed=seed)
    path = SPLIT_DIR / f"iid_n{n}_seed{seed}.npz"
    np.savez(
        path,
        train=splits.train,
        val=splits.val,
        test=splits.test,
    )
    return path


def write_ood_splits(*, seed: int = config.SEED) -> dict[str, Path]:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for name, root in default_ood_roots().items():
        h5s = discover_h5_files(root)
        n = len(h5s)
        if n == 0:
            raise FileNotFoundError(f"no H5 files under {root}")
        splits = make_splits(n, seed=seed)
        path = SPLIT_DIR / f"{name}_seed{seed}.npz"
        np.savez(
            path,
            train=splits.train,
            val=splits.val,
            test=splits.test,
            n=np.array(n),
            names=np.array([p.name for p in h5s]),
        )
        out[name] = path
        print(
            f"[splits] {name} n={n} "
            f"train={len(splits.train)} val={len(splits.val)} "
            f"test={len(splits.test)} → {path}",
            flush=True,
        )
    return out


def load_split(path: Path) -> dict[str, np.ndarray]:
    blob = np.load(path, allow_pickle=True)
    return {k: blob[k] for k in blob.files}


def ensure_splits(*, seed: int = config.SEED) -> dict[str, Path]:
    paths = {"iid": write_iid_splits(seed=seed)}
    paths.update(write_ood_splits(seed=seed))
    return paths


if __name__ == "__main__":
    ensure_splits()
