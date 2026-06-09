# data_loader.py
"""Data loading for GIFNO: H5 Vs grids + precomputed spatial transfer functions."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

import h5py

import config


def _resolve_h5_path(stored_path: str) -> Path:
    """Map manifest H5 paths to GIFNO_H5_DIR when running on Savio/scratch."""
    p = Path(stored_path)
    h5_dir = os.environ.get("GIFNO_H5_DIR")
    if h5_dir:
        return Path(h5_dir) / p.name
    return p


def _pad_depth(arr: np.ndarray, nz_max: int) -> np.ndarray:
    """Pad or truncate depth dimension; surface stays at row 0."""
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    nz, nx = arr.shape
    if nz >= nz_max:
        return arr[:nz_max, :nx]
    out = np.zeros((nz_max, arr.shape[1]), dtype=np.float32)
    out[:nz, :] = arr
    return out


def _build_sample_coord_grids(
    nz: int,
    nx: int,
    nz_max: int,
    lz: float,
    lx: float,
    dz: float = 1.0,
    dx: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build normalized (x, z) coords from the actual discretized grid.

    Uses physical Lz/Lx from the H5 (tied to H_discretized + bedrock), not NZ_MAX.
    Padded depth rows (below nz) are left at zero — invalid for the FNO until GINO.
    """
    x_cols = (np.arange(nx, dtype=np.float32) * dx) / max(lx, 1e-6)
    z_rows = (np.arange(min(nz, nz_max), dtype=np.float32) * dz) / max(lz, 1e-6)

    x_coord = np.tile(x_cols[None, :], (nz_max, 1))
    z_coord = np.zeros((nz_max, nx), dtype=np.float32)
    z_coord[: min(nz, nz_max), :] = z_rows[:nz_max, None]
    return x_coord, z_coord


def _scatter_tf_to_grid(
    tf_lateral: np.ndarray,
    recorder_x_idx: np.ndarray,
    nx: int,
    n_freq: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scatter (n_lateral, n_freq) TF onto (nx, n_freq) grid; return mask."""
    target = np.zeros((nx, n_freq), dtype=np.float32)
    mask = np.zeros(nx, dtype=np.float32)
    for ch, x_idx in enumerate(recorder_x_idx):
        if 0 <= x_idx < nx:
            target[x_idx, :] = tf_lateral[ch]
            mask[x_idx] = 1.0
    return target, mask


class GIFNODataset(Dataset):
    """PyTorch dataset: H5 Vs/zeta grids -> spatial TF targets."""

    def __init__(
        self,
        manifest_rows: List[dict],
        tf_array: np.ndarray,
        recorder_x_idx: np.ndarray,
        nz_max: int = config.NZ_MAX,
        nx: int = config.NX,
        n_freq: int = config.N_FREQ,
    ):
        self.manifest_rows = manifest_rows
        self.tf_array = tf_array
        self.recorder_x_idx = recorder_x_idx
        self.nz_max = nz_max
        self.nx = nx
        self.n_freq = n_freq

    def __len__(self) -> int:
        return len(self.manifest_rows)

    def __getitem__(self, idx: int):
        row = self.manifest_rows[idx]
        h5_path = _resolve_h5_path(row["h5_path"])

        with h5py.File(h5_path, "r") as f:
            vs_raw = f["Vs_realization_2D"][:]
            zeta_raw = f["Damping_zeta"][:]
            nz = int(vs_raw.shape[0])
            lz = float(f["grid"].attrs["Lz"])
            dz = float(f["grid"].attrs.get("dz", config.DZ))
            dx = float(f["grid"].attrs.get("dx", config.DX))

            sl = slice(config.X_SLICE_START, config.X_SLICE_END)
            vs = _pad_depth(vs_raw, self.nz_max)[:, sl]
            zeta = _pad_depth(zeta_raw, self.nz_max)[:, sl]
            x_coord, z_coord = _build_sample_coord_grids(
                nz,
                self.nx,
                self.nz_max,
                lz,
                float(config.LX_VARIABILITY),
                dz=dz,
                dx=dx,
            )

        if vs[0].max() <= 0:
            raise ValueError(f"Surface row has zero Vs in {h5_path}")

        x = np.stack([vs, zeta, x_coord, z_coord], axis=0)
        tf_lateral = self.tf_array[idx]
        target, mask = _scatter_tf_to_grid(
            tf_lateral, self.recorder_x_idx, self.nx, self.n_freq
        )

        return (
            torch.from_numpy(x),
            torch.from_numpy(target),
            torch.from_numpy(mask),
        )


def load_manifest(manifest_path: Path) -> List[dict]:
    with open(manifest_path, newline="") as f:
        return list(csv.DictReader(f))


def get_data_loaders(
    tf_path: Path = config.TF_PER_SAMPLE_PATH,
    manifest_path: Path = config.MANIFEST_PATH,
    recorder_x_path: Path = config.TF_RESULTS_DIR / "recorder_x_idx.npy",
    train_split: float = config.TRAIN_SPLIT,
    val_split: float = config.VAL_SPLIT,
    test_split: float = config.TEST_SPLIT,
    batch_size: int = config.BATCH_SIZE,
    seed: int = config.SEED,
    num_workers: int = config.NUM_WORKERS,
    limit: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    """Build train/val/test loaders and return frequency axis."""
    manifest = load_manifest(manifest_path)
    if limit is not None:
        manifest = manifest[:limit]

    tf_array = np.load(tf_path, mmap_mode="r")
    recorder_x_idx = np.load(recorder_x_path)
    freq = np.load(config.TF_FREQ_PATH)

    dataset = GIFNODataset(manifest, tf_array, recorder_x_idx)
    n = len(dataset)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()

    n_train = int(n * train_split)
    n_val = int(n * val_split)

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader, freq
