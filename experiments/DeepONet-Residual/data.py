"""Dataset: material fields + stochastic branch, nondimensional trunk queries."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

import config
import h5py

_RES = config.RESIDUAL_DIR
if str(_RES) not in sys.path:
    sys.path.insert(0, str(_RES))

from features import fourier_freq_features, spectral_kl_coefficients  # noqa: E402

_EPS = 1e-12
TargetName = Literal["R_col", "R_nom"]
TrunkSet = Literal["fstar", "fstar_fourier", "xL", "full"]


def freq_screen_indices(freq: np.ndarray, n: int) -> np.ndarray:
    if n >= len(freq):
        return np.arange(len(freq))
    targets = np.logspace(np.log10(freq[0]), np.log10(freq[-1]), n)
    idx = np.unique([int(np.argmin(np.abs(freq - t))) for t in targets])
    if len(idx) < n:
        extra = [i for i in range(len(freq)) if i not in set(idx)]
        idx = np.concatenate([idx, extra[: n - len(idx)]])
    return np.sort(idx[:n])


def pad_depth(arr: np.ndarray, nz_max: int) -> np.ndarray:
    nz, nx = arr.shape
    if nz == nz_max:
        return arr.astype(np.float32, copy=False)
    out = np.zeros((nz_max, nx), dtype=np.float32)
    n = min(nz, nz_max)
    out[:n] = arr[:n]
    if nz < nz_max and nz > 0:
        out[nz:] = arr[-1]
    return out


def normalize_vs_surface(vs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    surface = np.maximum(vs[0:1, :], eps)
    return (vs / surface).astype(np.float32)


def normalize_zeta_max(zeta: np.ndarray, nz: int, eps: float = 1e-12) -> np.ndarray:
    n = max(1, min(int(nz), zeta.shape[0]))
    zmax = float(np.max(zeta[:n]))
    if zmax < eps:
        return zeta.astype(np.float32)
    return (zeta / zmax).astype(np.float32)


def stoch_dim(k_xi: int = config.K_XI) -> int:
    return 2 * k_xi + 4  # xi re/im + r_H, aHV, CoV, xi_damp


def append_serial_tf1d(trunk: np.ndarray, tf1d: np.ndarray) -> np.ndarray:
    """Concatenate log(TF_1D) onto trunk queries (serial / discrepancy operator)."""
    extra = np.log(np.maximum(np.asarray(tf1d).reshape(-1, 1), _EPS)).astype(np.float32)
    return np.concatenate([np.asarray(trunk, dtype=np.float32), extra], axis=-1)


def trunk_feature_names(trunk_set: TrunkSet) -> list[str]:
    if trunk_set == "fstar":
        return ["f_star"]
    if trunk_set == "fstar_fourier":
        return ["f_star", "sin_f", "cos_f"]
    if trunk_set == "xL":
        return ["f_star", "sin_f", "cos_f", "x_over_L"]
    return ["f_star", "sin_f", "cos_f", "x_over_lambda"]


@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def make_splits(
    n: int,
    *,
    seed: int = config.SEED,
    train_frac: float = config.TRAIN_FRAC,
    val_frac: float = config.VAL_FRAC,
) -> SplitIndices:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return SplitIndices(
        train=perm[:n_train],
        val=perm[n_train : n_train + n_val],
        test=perm[n_train + n_val :],
    )


def build_trunk_queries(
    *,
    vs_col: np.ndarray,
    H: float,
    recorder_x: np.ndarray,
    freq_s: np.ndarray,
    sin_f: np.ndarray,
    cos_f: np.ndarray,
    trunk_names: Sequence[str],
) -> np.ndarray:
    """Vectorized trunk features, shape (n_rec * n_freq, n_feat)."""
    vs_c = np.maximum(np.asarray(vs_col, dtype=np.float64).ravel(), _EPS)
    f = np.asarray(freq_s, dtype=np.float64).ravel()
    cols = np.asarray(recorder_x, dtype=np.float64).ravel()
    x_m = (cols + 0.5) * config.DX
    n_rec = vs_c.size
    n_f = f.size
    f_star = (f[None, :] * float(H) / vs_c[:, None]).astype(np.float32)
    lam = vs_c[:, None] / np.maximum(f[None, :], _EPS)
    x_over_lambda = (x_m[:, None] / np.maximum(lam, _EPS)).astype(np.float32)
    x_over_L = np.broadcast_to(
        (x_m / float(config.LX_VARIABILITY)).astype(np.float32)[:, None],
        (n_rec, n_f),
    )
    sin_b = np.broadcast_to(np.asarray(sin_f, dtype=np.float32)[None, :], (n_rec, n_f))
    cos_b = np.broadcast_to(np.asarray(cos_f, dtype=np.float32)[None, :], (n_rec, n_f))
    feat_map = {
        "f_star": f_star,
        "sin_f": sin_b,
        "cos_f": cos_b,
        "x_over_L": x_over_L,
        "x_over_lambda": x_over_lambda,
    }
    stacked = np.stack([feat_map[name] for name in trunk_names], axis=-1)
    return stacked.reshape(n_rec * n_f, -1).astype(np.float32)


class ResidualDeepONetDataset(Dataset):
    """One item = one realization; queries all recorders × selected freqs."""

    def __init__(
        self,
        cache_dir: Path,
        indices: Sequence[int],
        *,
        target: TargetName = "R_col",
        trunk_set: TrunkSet = "full",
        n_freq: int = config.N_FREQ_TRAIN,
        n_freq_train: int | None = None,
        serial_tf1d: bool = False,
    ):
        if n_freq_train is not None:
            n_freq = n_freq_train
        self.cache_dir = Path(cache_dir)
        self.indices = np.asarray(indices, dtype=int)
        self.target = target
        self.trunk_set = trunk_set
        self.n_freq_requested = int(n_freq)
        self.serial_tf1d = bool(serial_tf1d)

        self.meta = dict(np.load(self.cache_dir / "meta.npz", allow_pickle=True))
        key = "r_col_signed.npy" if target == "R_col" else "r_nom_signed.npy"
        self.r = np.load(self.cache_dir / key, mmap_mode="r")
        tf_key = "tf1d_col.npy" if target == "R_col" else "tf1d_nom.npy"
        self.tf1d = np.load(self.cache_dir / tf_key, mmap_mode="r")
        tf2d_path = self.cache_dir / "tf2d.npy"
        self.tf2d_local = (
            np.load(tf2d_path, mmap_mode="r") if tf2d_path.is_file() else None
        )
        self.tf_all = None
        if self.tf2d_local is None:
            self.tf_all = np.load(config.TF_PER_SAMPLE_PATH, mmap_mode="r")
        self.sample_indices = np.load(self.cache_dir / "sample_indices.npy")
        fields_path = self.cache_dir / "fields.npy"
        vs_col_path = self.cache_dir / "vs_col.npy"
        self._fields_all = (
            np.load(fields_path, mmap_mode="r") if fields_path.is_file() else None
        )
        self._vs_col_all = (
            np.load(vs_col_path, mmap_mode="r") if vs_col_path.is_file() else None
        )

        self.freq = np.load(
            self.cache_dir / "freq.npy"
            if (self.cache_dir / "freq.npy").is_file()
            else config.TF_FREQ_PATH
        )
        rec_path = self.cache_dir / "recorder_x.npy"
        if rec_path.is_file():
            self.recorder_x = np.load(rec_path)
        elif config.RECORDER_X_IDX_PATH.is_file():
            self.recorder_x = np.load(config.RECORDER_X_IDX_PATH)
        else:
            self.recorder_x = np.arange(self.r.shape[1])
        self.f_idx = freq_screen_indices(self.freq, n_freq)
        self.freq_s = self.freq[self.f_idx]
        self.sin_f, self.cos_f = fourier_freq_features(
            self.freq_s,
            f_min=config.FREQ_START_HZ,
            f_max=config.FREQ_END_HZ,
        )
        self.n_rec = len(self.recorder_x)
        self.n_q = self.n_rec * len(self.f_idx)
        self.trunk_names = trunk_feature_names(trunk_set)
        self.stoch_dim = stoch_dim()
        # Preload tensors for this split (n=100 ablation is H5-bound otherwise).
        self._cache: list[dict[str, torch.Tensor]] = []
        desc = f"dataset {target}/{trunk_set}/nf={len(self.f_idx)}"
        for local_i in tqdm(self.indices, desc=desc, leave=False):
            item = self._load_item(int(local_i))
            self._cache.append({k: torch.from_numpy(v) for k, v in item.items()})

    def __len__(self) -> int:
        return len(self.indices)

    def _stoch(self, local_i: int) -> np.ndarray:
        rf_seed = int(self.meta["rf_seed"][local_i])
        rH = float(self.meta["rH"][local_i])
        aHV = float(self.meta["aHV"][local_i])
        CoV = float(self.meta["CoV"][local_i])
        nz = int(self.meta["nz"][local_i])
        xi_damp = float(
            self.meta["xi_damp"][local_i]
            if "xi_damp" in self.meta
            else config.DEFAULT_XI_TREND
        )
        xi_vals, _ = spectral_kl_coefficients(
            rf_seed=rf_seed,
            rH=rH,
            aHV=aHV,
            nx=config.NX,
            nz=nz,
            dx=config.DX,
            dz=config.DZ,
            k=config.K_XI,
        )
        return np.concatenate(
            [xi_vals, np.array([rH, aHV, CoV, xi_damp], dtype=np.float32)]
        ).astype(np.float32)

    def _load_item(self, local_i: int) -> dict[str, np.ndarray]:
        h5_path = Path(str(self.meta["h5_path"][local_i]))
        nz = int(self.meta["nz"][local_i])
        H = float(self.meta["H"][local_i])
        soil_nz = int(self.meta["soil_nz"][local_i])
        if self._fields_all is not None and self._vs_col_all is not None:
            fields = np.asarray(self._fields_all[local_i], dtype=np.float32)
            vs_col = np.asarray(self._vs_col_all[local_i], dtype=np.float64)
        else:
            with h5py.File(h5_path, "r") as f:
                vs = np.asarray(f["Vs_realization_2D"][:], dtype=np.float64)
                zeta = np.asarray(f["Damping_zeta"][:], dtype=np.float64)
            vs = vs[:, config.X_SLICE_START : config.X_SLICE_END]
            zeta = zeta[:, config.X_SLICE_START : config.X_SLICE_END]
            vs_pad = pad_depth(vs, config.NZ_MAX)
            zeta_pad = pad_depth(zeta, config.NZ_MAX)
            vs_n = normalize_vs_surface(vs_pad)
            zeta_n = normalize_zeta_max(zeta_pad, nz)
            z_imp = (config.RHO * vs_pad).astype(np.float32)
            z_imp = z_imp / max(float(z_imp.max()), _EPS)
            cols = self.recorder_x.astype(int)
            fields = np.stack(
                [vs_n[:, cols], zeta_n[:, cols], z_imp[:, cols]], axis=0
            ).astype(np.float32)
            n = max(1, min(soil_nz, vs.shape[0]))
            vs_col = vs[:n, cols].mean(axis=0)

        r = np.asarray(self.r[local_i][:, self.f_idx], dtype=np.float32)
        tf1d = np.asarray(self.tf1d[local_i][:, self.f_idx], dtype=np.float32)
        if self.tf2d_local is not None:
            tf2d = np.asarray(self.tf2d_local[local_i][:, self.f_idx], dtype=np.float32)
        else:
            sidx = int(self.sample_indices[local_i])
            assert self.tf_all is not None
            tf2d = np.asarray(self.tf_all[sidx][:, self.f_idx], dtype=np.float32)
        trunk_y = build_trunk_queries(
            vs_col=vs_col,
            H=H,
            recorder_x=self.recorder_x,
            freq_s=self.freq_s,
            sin_f=self.sin_f,
            cos_f=self.cos_f,
            trunk_names=self.trunk_names,
        )
        if self.serial_tf1d:
            trunk_y = append_serial_tf1d(trunk_y, tf1d)
        return {
            "fields": fields,
            "stoch": self._stoch(local_i),
            "trunk_y": trunk_y,
            "target": r.reshape(-1),
            "tf1d": tf1d.reshape(-1),
            "tf2d": tf2d.reshape(-1),
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._cache[idx]


def make_loaders(
    cache_dir: Path,
    splits: SplitIndices,
    *,
    target: TargetName,
    trunk_set: TrunkSet = "full",
    batch_size: int = config.BATCH_SIZE,
    n_freq: int = config.N_FREQ_TRAIN,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    def _loader(idxs: np.ndarray, shuffle: bool) -> DataLoader:
        ds = ResidualDeepONetDataset(
            cache_dir, idxs, target=target, trunk_set=trunk_set, n_freq=n_freq
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=config.NUM_WORKERS,
            drop_last=False,
        )

    return (
        _loader(splits.train, True),
        _loader(splits.val, False),
        _loader(splits.test, False),
    )


class CombinedResidualDataset(Dataset):
    """Concat of preloaded ResidualDeepONetDataset caches (for multi-domain train)."""

    def __init__(
        self,
        datasets: Sequence[ResidualDeepONetDataset],
        domain_names: Sequence[str] | None = None,
    ):
        if not datasets:
            raise ValueError("need at least one dataset")
        self._parts = list(datasets)
        names = list(domain_names) if domain_names is not None else ["unk"] * len(self._parts)
        if len(names) != len(self._parts):
            raise ValueError("domain_names must match datasets")
        self._cache: list[dict[str, torch.Tensor]] = []
        self.domain_names_per_item: list[str] = []
        for ds, name in zip(self._parts, names):
            self._cache.extend(ds._cache)
            self.domain_names_per_item.extend([str(name)] * len(ds._cache))
        self.n_rec = self._parts[0].n_rec
        self.f_idx = self._parts[0].f_idx
        self.freq_s = getattr(self._parts[0], "freq_s", None)
        self.trunk_names = self._parts[0].trunk_names
        self.serial_tf1d = self._parts[0].serial_tf1d

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._cache[idx]


def iid_resample_sampler(
    ds: CombinedResidualDataset,
    iid_frac: float,
) -> WeightedRandomSampler:
    """Weighted sampler so an expected ``iid_frac`` of each epoch is IID."""
    is_iid = np.array(
        [name.startswith("iid") for name in ds.domain_names_per_item], dtype=bool
    )
    n_iid = int(is_iid.sum())
    n_ood = int((~is_iid).sum())
    if n_iid == 0 or n_ood == 0:
        raise ValueError("iid resampling needs both IID and OOD items")
    weights = np.zeros(len(ds), dtype=np.float64)
    weights[is_iid] = float(iid_frac) / n_iid
    weights[~is_iid] = (1.0 - float(iid_frac)) / n_ood
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(ds),
        replacement=True,
    )
