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


def _normalize_vs_by_surface(vs: np.ndarray, eps: float) -> np.ndarray:
    """Divide each column by its surface Vs (row 0); padded rows scale the same way."""
    surface = np.maximum(vs[0:1, :], eps)
    return (vs / surface).astype(np.float32)


def _normalize_zeta_by_max(zeta: np.ndarray, nz: int, eps: float) -> np.ndarray:
    """Divide by peak damping over active depth rows (excludes zero-padded tail)."""
    if nz <= 0:
        return zeta
    zeta_max = float(np.max(zeta[:nz, :]))
    if zeta_max < eps:
        return zeta
    return (zeta / zeta_max).astype(np.float32)


def _box_blur_2d(arr: np.ndarray, kernel: int) -> np.ndarray:
    """Separable box blur with edge padding; kernel forced odd and >=1."""
    k = int(kernel)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    if k == 1:
        return arr.astype(np.float32, copy=True)
    pad = k // 2
    # Horizontal then vertical via cumsum (O(nz*nx)).
    x = np.pad(arr.astype(np.float64), ((0, 0), (pad, pad)), mode="edge")
    c = np.cumsum(x, axis=1)
    # sum of window [j, j+k): c[..., j+k] - c[..., j] with c padded by leading 0
    c = np.pad(c, ((0, 0), (1, 0)), mode="constant")
    horiz = (c[:, k:] - c[:, :-k]) / k
    y = np.pad(horiz, ((pad, pad), (0, 0)), mode="edge")
    c2 = np.cumsum(y, axis=0)
    c2 = np.pad(c2, ((1, 0), (0, 0)), mode="constant")
    out = (c2[k:, :] - c2[:-k, :]) / k
    return out.astype(np.float32)


def split_vs_macro_rf(
    vs: np.ndarray, kernel: int | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Vs_macro, Vs_rf) with Vs_rf = Vs - Vs_macro."""
    if kernel is None:
        kernel = int(getattr(config, "VS_MACRO_KERNEL", 15))
    macro = _box_blur_2d(vs, kernel)
    rf = (vs - macro).astype(np.float32)
    return macro, rf


def stack_model_input_channels(
    vs: np.ndarray,
    zeta: np.ndarray,
    x_coord: np.ndarray,
    z_coord: np.ndarray,
) -> np.ndarray:
    """Stack (C, Nz, Nx). C=6 when SCALE_SPLIT_VS else C=4."""
    if getattr(config, "SCALE_SPLIT_VS", False):
        vs_macro, vs_rf = split_vs_macro_rf(vs)
        return np.stack([vs, vs_macro, vs_rf, zeta, x_coord, z_coord], axis=0)
    return np.stack([vs, zeta, x_coord, z_coord], axis=0)


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
        cache_in_memory: bool = False,
    ):
        self.manifest_rows = manifest_rows
        self.tf_array = tf_array
        self.recorder_x_idx = recorder_x_idx
        self.nz_max = nz_max
        self.nx = nx
        self.n_freq = n_freq
        self.cache_in_memory = cache_in_memory
        self._cache: dict[int, tuple] = {}

    def __len__(self) -> int:
        return len(self.manifest_rows)

    def preload(self) -> None:
        """Read every sample once into RAM so per-epoch H5 reads are eliminated.

        Called in the main process before forking DataLoader workers, so the
        cache is shared copy-on-write across workers (no per-worker duplication).
        """
        from tqdm import tqdm

        for idx in tqdm(range(len(self)), desc="Preloading dataset", unit="sample"):
            self._cache[idx] = self._load_sample(idx)

    def __getitem__(self, idx: int):
        if self.cache_in_memory:
            cached = self._cache.get(idx)
            if cached is None:
                cached = self._load_sample(idx)
                self._cache[idx] = cached
            return cached
        return self._load_sample(idx)

    def _load_sample(self, idx: int):
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

        if config.NORMALIZE_VS_SURFACE:
            vs = _normalize_vs_by_surface(vs, config.VS_NORM_EPS)
        if config.NORMALIZE_ZETA:
            zeta = _normalize_zeta_by_max(zeta, nz, config.ZETA_NORM_EPS)

        x = stack_model_input_channels(vs, zeta, x_coord, z_coord)
        tf_lateral = self.tf_array[idx]
        target, mask = _scatter_tf_to_grid(
            tf_lateral, self.recorder_x_idx, self.nx, self.n_freq
        )

        sample = (
            torch.from_numpy(x),
            torch.from_numpy(target),
            torch.from_numpy(mask),
        )
        if getattr(config, "RETURN_SAMPLE_ID", False):
            sample_id = _sample_id_from_manifest_row(row)
            return (*sample, torch.tensor(sample_id, dtype=torch.long))
        return sample


def _sample_id_from_manifest_row(row: dict) -> int:
    """Sobol sample_id from GIFNO manifest row (run_index // seeds_per_sample)."""
    seeds = int(getattr(config, "SEEDS_PER_SAMPLE", 30))
    if "sample_id" in row and row["sample_id"] not in ("", None):
        return int(row["sample_id"])
    if "run_index" in row and row["run_index"] not in ("", None):
        return int(row["run_index"]) // max(seeds, 1)
    if "sample_idx" in row and row["sample_idx"] not in ("", None):
        return int(row["sample_idx"]) // max(seeds, 1)
    return -1


def load_manifest(manifest_path: Path) -> List[dict]:
    with open(manifest_path, newline="") as f:
        return list(csv.DictReader(f))


class SeedGroupBatchSampler:
    """Yield batches with paired RF replicates from the same Sobol sample_id.

    Each batch is built from ``batch_size // 2`` distinct sample_ids with two
    random replicates each. Requires the underlying dataset indices to map to
    sample_ids via ``sample_ids[i]`` (Subset-local indices 0..N-1).
    """

    def __init__(
        self,
        sample_ids: List[int],
        batch_size: int,
        *,
        seed: int = 42,
        drop_last: bool = True,
    ):
        if batch_size < 2:
            raise ValueError("SEED_GROUP_BATCH requires batch_size >= 2")
        self.batch_size = batch_size
        self.drop_last = drop_last
        self._n_pairs = batch_size // 2
        self._rng_seed = seed
        self._by_sid: dict[int, list[int]] = {}
        for idx, sid in enumerate(sample_ids):
            self._by_sid.setdefault(int(sid), []).append(idx)
        self._pairable = [s for s, ix in self._by_sid.items() if len(ix) >= 2]
        if not self._pairable:
            raise ValueError(
                "SEED_GROUP_BATCH requires at least one sample_id with >=2 replicates"
            )

    def __iter__(self):
        rng = np.random.default_rng(self._rng_seed)
        # Re-seed each epoch from a fresh draw so shuffle varies... use epoch counter
        # via np random; callers recreate sampler each epoch if needed. For
        # DataLoader reuse, shuffle with a random seed derived from os.urandom.
        rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        pairable = list(self._pairable)
        rng.shuffle(pairable)
        n_pairs = self._n_pairs
        i = 0
        while i + n_pairs <= len(pairable):
            batch: list[int] = []
            for sid in pairable[i : i + n_pairs]:
                choices = np.asarray(self._by_sid[sid], dtype=np.int64)
                picked = rng.choice(choices, size=2, replace=False)
                batch.extend(int(x) for x in picked)
            yield batch
            i += n_pairs
        if not self.drop_last and i < len(pairable):
            batch = []
            for sid in pairable[i:]:
                choices = np.asarray(self._by_sid[sid], dtype=np.int64)
                picked = rng.choice(choices, size=min(2, len(choices)), replace=False)
                batch.extend(int(x) for x in picked)
                if len(batch) >= self.batch_size:
                    break
            while len(batch) < self.batch_size:
                s = int(rng.choice(self._pairable))
                batch.append(int(rng.choice(self._by_sid[s])))
            yield batch[: self.batch_size]

    def __len__(self) -> int:
        return max(0, len(self._pairable) // max(self._n_pairs, 1))


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

    cache_in_memory = bool(getattr(config, "CACHE_DATASET", False))
    # Seed-contrast needs sample_id on each item.
    if (
        getattr(config, "SEED_CONTRAST_WEIGHT", 0.0) > 0
        or getattr(config, "SEED_SIGMA_LN_WEIGHT", 0.0) > 0
        or getattr(config, "SEED_GROUP_BATCH", False)
    ):
        # Force RETURN_SAMPLE_ID for this process.
        if not getattr(config, "RETURN_SAMPLE_ID", False):
            config.RETURN_SAMPLE_ID = True

    dataset = GIFNODataset(
        manifest, tf_array, recorder_x_idx, cache_in_memory=cache_in_memory
    )
    if cache_in_memory:
        dataset.preload()
    n = len(dataset)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()

    n_train = int(n * train_split)
    n_val = int(n * val_split)

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    use_seed_batch = bool(getattr(config, "SEED_GROUP_BATCH", False))
    if use_seed_batch:
        train_sids = [_sample_id_from_manifest_row(manifest[i]) for i in train_idx]
        train_subset = Subset(dataset, train_idx)
        sampler = SeedGroupBatchSampler(
            train_sids, batch_size=batch_size, seed=seed, drop_last=True
        )
        seed_loader_kwargs: dict = {
            "batch_sampler": sampler,
            "num_workers": num_workers,
            "pin_memory": True,
        }
        if num_workers > 0:
            seed_loader_kwargs["persistent_workers"] = True
            seed_loader_kwargs["prefetch_factor"] = 2
        train_loader = DataLoader(train_subset, **seed_loader_kwargs)
    else:
        train_loader = DataLoader(
            Subset(dataset, train_idx),
            shuffle=True,
            **loader_kwargs,
        )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, test_loader, freq
