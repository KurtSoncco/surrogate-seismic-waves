"""GIFNO corpus dataset: H_1D(trend) baseline from manifest Vs1/H/Vs2."""

from __future__ import annotations

import csv
from pathlib import Path

import h5py

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import config
from context_features import (
    bedrock_interface_depth,
    dip_field_broadcast,
    impedance_gradient_field,
    interface_dip,
    stack_delta_input_channels,
)
from haskell_baseline import H_1D_trend, scatter_recorder_tf

_EPS = 1e-12


def _pad_depth(field: np.ndarray, nz_max: int) -> np.ndarray:
    nz, nx = field.shape
    if nz >= nz_max:
        return field[:nz_max].astype(np.float32)
    out = np.zeros((nz_max, nx), dtype=np.float32)
    out[:nz] = field
    if nz > 0:
        out[nz:] = field[-1]
    return out


def _norm_vs(vs: np.ndarray) -> np.ndarray:
    return (vs / np.maximum(vs[0:1], config.VS_NORM_EPS)).astype(np.float32)


def _norm_zeta(zeta: np.ndarray, nz: int) -> np.ndarray:
    m = float(np.max(zeta[:nz])) if nz > 0 else 0.0
    if m < config.ZETA_NORM_EPS:
        return zeta.astype(np.float32)
    return (zeta / m).astype(np.float32)


def _coords(nz: int, nx: int, nz_max: int, lz: float):
    x = (np.arange(nx, dtype=np.float32) - (nx - 1) / 2.0) * config.DX
    x = x / max(config.LX_VARIABILITY / 2.0, _EPS)
    z = np.arange(nz_max, dtype=np.float32) * config.DZ / max(lz, _EPS)
    return (
        np.broadcast_to(x, (nz_max, nx)).copy(),
        np.broadcast_to(z[:, None], (nz_max, nx)).copy(),
    )


def _resolve_h5(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_file():
        return p
    # relative to H5_DIR by basename
    cand = config.H5_DIR / p.name
    if cand.is_file():
        return cand
    raise FileNotFoundError(path_str)


class GIFNOTrendDeltaDataset(Dataset):
    """Supervised residual dataset with AGENTS trend baseline (not realization geomean)."""

    def __init__(self, indices: np.ndarray, manifest_rows: list[dict], tf_array: np.ndarray):
        self.indices = np.asarray(indices, dtype=int)
        self.manifest_rows = manifest_rows
        self.tf_array = tf_array
        self.recorder_x = config.recorder_x_indices()
        self.freq = (
            np.load(config.TF_FREQ_PATH)
            if config.TF_FREQ_PATH.is_file()
            else np.logspace(-1, 1, config.N_FREQ)
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        row = self.manifest_rows[idx]
        h5_path = _resolve_h5(row["h5_path"])
        with h5py.File(h5_path, "r") as f:
            vs_raw = f["Vs_realization_2D"][:]
            zeta_raw = f["Damping_zeta"][:]
            dz = float(f["grid"].attrs.get("dz", config.DZ))
            dx = float(f["grid"].attrs.get("dx", config.DX))
            lz = float(f["grid"].attrs.get("Lz", vs_raw.shape[0] * dz))
            pattrs = dict(f["params"].attrs) if "params" in f else {}

        sl = config.central_strip_slice()  # IID: [500:1000] on full nx=1500
        vs = vs_raw[:, sl].astype(np.float32)
        zeta = zeta_raw[:, sl].astype(np.float32)
        nz, nx = vs.shape

        # Prefer H5 params attrs (manifest often lacks Vs1/rH/aHV)
        def _p(key: str, *alts, default=None):
            for k in (key, *alts):
                if k in row and row[k] not in (None, ""):
                    try:
                        return float(row[k])
                    except (TypeError, ValueError):
                        pass
                if k in pattrs:
                    return float(pattrs[k])
            return default

        vs1 = _p("Vs1", "vs1", default=float(vs[0].mean()))
        H = _p("H", "H_discretized", "h", default=float(nz * dz * 0.85))
        vs2 = _p("Vs2", "vs2", default=float(np.median(vs[-5:])))
        cov = _p("CoV", "cov", default=0.0)
        rH = _p("rH", "rh", default=50.0)
        aHV = _p("aHV", "ahv", default=20.0)

        z_bed = bedrock_interface_depth(vs, vs_rock=vs2, dz=dz)
        dip = interface_dip(z_bed, dx=dx)
        dip_rms = float(np.sqrt(np.mean(dip**2)))
        # If CoV==0, force dip gate-compatible zero for no-field
        if cov <= 0:
            dip_rms = 0.0

        vs_pad = _pad_depth(vs, config.NZ_MAX)
        zeta_pad = _pad_depth(zeta, config.NZ_MAX)
        dip_pad = _pad_depth(dip_field_broadcast(dip, nz), config.NZ_MAX)
        imp_pad = _pad_depth(
            impedance_gradient_field(vs, rho=config.DEFAULT_RHO, dx=dx), config.NZ_MAX
        )
        x_c, z_c = _coords(nz, nx, config.NZ_MAX, lz)
        x_in = stack_delta_input_channels(
            _norm_vs(vs_pad),
            _norm_zeta(zeta_pad, nz),
            x_c,
            z_c,
            dip_pad,
            imp_pad,
        )

        trend_af = H_1D_trend(
            self.freq, vs1=vs1, H=H, vs2=vs2, xi=config.DEFAULT_XI_TREND
        ).astype(np.float32)
        # Broadcast trend to all recorders (horizontally layered limit)
        trend_rec = np.broadcast_to(
            trend_af, (len(self.recorder_x), len(trend_af))
        ).copy()
        haskell_grid = scatter_recorder_tf(trend_rec, self.recorder_x, config.NX)

        tf_lat = np.asarray(self.tf_array[idx], dtype=np.float32)
        target = scatter_recorder_tf(tf_lat, self.recorder_x, config.NX)
        mask = np.zeros(config.NX, dtype=np.float32)
        mask[self.recorder_x] = 1.0

        # Physics latents (interim without KL): CoV, rH_norm, aHV_norm
        physics = np.array(
            [cov, rH / 100.0, aHV / 50.0], dtype=np.float32
        )

        return (
            torch.from_numpy(x_in),
            torch.from_numpy(haskell_grid),
            torch.from_numpy(target),
            torch.from_numpy(mask),
            torch.tensor(cov, dtype=torch.float32),
            torch.tensor(dip_rms, dtype=torch.float32),
            torch.from_numpy(physics),
        )


def _load_manifest() -> list[dict]:
    rows = []
    with open(config.MANIFEST_PATH, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def get_gifno_loaders(limit: int | None = None, batch_size: int | None = None):
    if not config.TF_PER_SAMPLE_PATH.is_file() or not config.MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            "GIFNO TF cache missing. Set GIFNO_DATA_ROOT and ensure "
            f"{config.TF_PER_SAMPLE_PATH} and {config.MANIFEST_PATH}"
        )
    manifest = _load_manifest()
    tf = np.load(config.TF_PER_SAMPLE_PATH, mmap_mode="r")
    n = len(manifest)
    if limit is not None:
        n = min(n, limit)
    rng = np.random.RandomState(config.SEED)
    idx = rng.permutation(n)
    n_train = int(config.TRAIN_SPLIT * n)
    n_val = int(config.VAL_SPLIT * n)
    train_i, val_i, test_i = (
        idx[:n_train],
        idx[n_train : n_train + n_val],
        idx[n_train + n_val :],
    )
    bs = batch_size or config.BATCH_SIZE
    return (
        DataLoader(
            GIFNOTrendDeltaDataset(train_i, manifest, tf),
            batch_size=bs,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
        ),
        DataLoader(
            GIFNOTrendDeltaDataset(val_i, manifest, tf),
            batch_size=bs,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
        ),
        DataLoader(
            GIFNOTrendDeltaDataset(test_i, manifest, tf),
            batch_size=bs,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
        ),
    )
