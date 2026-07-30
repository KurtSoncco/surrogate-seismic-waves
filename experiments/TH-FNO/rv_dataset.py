"""Response_Variability OpenSees-2D dataset for gated-delta prototype training."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

import config
from context_features import (
    bedrock_interface_depth,
    dip_field_broadcast,
    impedance_gradient_field,
    interface_dip,
    stack_delta_input_channels,
)
from haskell_baseline import haskell_af_within

_EPS = 1e-12


def _normalize_vs(vs: np.ndarray) -> np.ndarray:
    surf = np.maximum(vs[0:1], config.VS_NORM_EPS)
    return (vs / surf).astype(np.float32)


def _normalize_zeta(zeta: np.ndarray, nz_active: int) -> np.ndarray:
    m = float(np.max(zeta[:nz_active])) if nz_active > 0 else 0.0
    if m < config.ZETA_NORM_EPS:
        return zeta.astype(np.float32)
    return (zeta / m).astype(np.float32)


def _pad_depth(field: np.ndarray, nz_max: int) -> np.ndarray:
    nz, nx = field.shape
    if nz >= nz_max:
        return field[:nz_max].astype(np.float32)
    out = np.zeros((nz_max, nx), dtype=np.float32)
    out[:nz] = field
    if nz > 0:
        out[nz:] = field[-1]
    return out


def _coord_grids(nz: int, nx: int, nz_max: int, lz: float) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(nx, dtype=np.float32) - (nx - 1) / 2.0) * config.DX
    x = (x / max(config.LX_VARIABILITY / 2.0, _EPS)).astype(np.float32)
    z = np.arange(nz_max, dtype=np.float32) * config.DZ / max(lz, _EPS)
    x_coord = np.broadcast_to(x, (nz_max, nx)).copy()
    z_coord = np.broadcast_to(z[:, None], (nz_max, nx)).copy()
    return x_coord, z_coord


def build_rv_index_lookup(rv_root: Path | None = None) -> dict[tuple[int, str, int], int]:
    rv_root = rv_root or config.RV_ROOT
    sys.path.insert(0, str(rv_root))
    os.environ.pop("RV_SMOKE", None)
    from manifest import (  # noqa: WPS433
        _hallal_block_size,
        active_sobol_count,
        index_to_params,
        total_combinations,
    )

    hallal0 = _hallal_block_size(active_sobol_count())
    lookup: dict[tuple[int, str, int], int] = {}
    for i in range(hallal0, total_combinations()):
        p = index_to_params(i)
        if p.method == "opensees_2d":
            lookup[(p.sobol_id, p.method, p.seed)] = i
    return lookup


class RVOpenSeesDeltaDataset(Dataset):
    """Paired OpenSees-2D center/lateral TFs with Haskell baseline + context."""

    def __init__(
        self,
        sobol_ids: Sequence[int],
        h5_dir: Path | None = None,
        max_seeds: int | None = None,
        use_all_recorders: bool = False,
    ):
        self.h5_dir = Path(h5_dir or config.RV_H5_DIR)
        self.use_all_recorders = use_all_recorders
        self.recorder_x = config.recorder_x_indices()  # on 500-wide strip
        self.lookup = build_rv_index_lookup()
        sys.path.insert(0, str(config.RV_ROOT))
        os.environ.pop("RV_SMOKE", None)
        from manifest import active_rf_seeds  # noqa: WPS433

        seeds = active_rf_seeds()
        if max_seeds is not None:
            seeds = seeds[:max_seeds]
        self.samples: list[tuple[int, int, int]] = []
        for sid in sobol_ids:
            for seed in seeds:
                idx = self.lookup.get((int(sid), "opensees_2d", int(seed)))
                if idx is not None and (self.h5_dir / f"run_{idx}.h5").is_file():
                    self.samples.append((int(sid), int(seed), int(idx)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        sid, seed, idx = self.samples[i]
        path = self.h5_dir / f"run_{idx}.h5"
        with h5py.File(path, "r") as f:
            vs_full = f["Vs_field"][:].astype(np.float32)
            zeta_full = f["Damping_zeta"][:].astype(np.float32)
            dz = float(f["grid"].attrs.get("dz", config.DZ))
            dx = float(f["grid"].attrs.get("dx", config.DX))
            vs2 = float(f["params"].attrs["Vs2"])
            H = float(f["params"].attrs["H"])
            cov = float(f["params"].attrs["CoV"])
            af = f["transfer_function"]["AF"][:].astype(np.float32)
            freq = f["transfer_function"]["freq"][:].astype(np.float64)

        # Variability strip
        i0, i1 = config.X_SLICE_START, config.X_SLICE_END
        vs = vs_full[:, i0:i1]
        zeta = zeta_full[:, i0:i1]
        nz, nx = vs.shape
        soil_nz = max(1, int(round(H / dz)))
        soil_nz = min(soil_nz, nz - 1)

        z_bed = bedrock_interface_depth(vs, vs_rock=vs2, dz=dz)
        dip = interface_dip(z_bed, dx=dx)
        dip_rms = float(np.sqrt(np.mean(dip**2)))
        imp_g = impedance_gradient_field(vs, rho=config.DEFAULT_RHO, dx=dx)
        dip_2d = dip_field_broadcast(dip, nz)

        vs_pad = _pad_depth(vs, config.NZ_MAX)
        zeta_pad = _pad_depth(zeta, config.NZ_MAX)
        dip_pad = _pad_depth(dip_2d, config.NZ_MAX)
        imp_pad = _pad_depth(imp_g, config.NZ_MAX)
        lz = nz * dz
        x_coord, z_coord = _coord_grids(nz, nx, config.NZ_MAX, lz)

        vs_n = _normalize_vs(vs_pad)
        zeta_n = _normalize_zeta(zeta_pad, nz)
        x_in = stack_delta_input_channels(vs_n, zeta_n, x_coord, z_coord, dip_pad, imp_pad)

        # Haskell at center recorder only (RV OpenSees H5 stores center AF)
        center = int(self.recorder_x[len(self.recorder_x) // 2])
        af_h_c = haskell_af_within(
            freq,
            vs[:, center],
            zeta[:, center],
            dz=dz,
            vs_rock=vs2,
            soil_nz=soil_nz,
            rho=config.DEFAULT_RHO,
        ).astype(np.float32)
        model_freq = np.logspace(-1, 1, config.N_FREQ)
        if len(freq) != config.N_FREQ or not np.allclose(freq, model_freq, rtol=1e-3, atol=1e-5):
            af_h_i = np.interp(model_freq, freq, af_h_c).astype(np.float32)
            af_ops_i = np.interp(model_freq, freq, af).astype(np.float32)
        else:
            af_h_i = af_h_c
            af_ops_i = af.astype(np.float32)

        haskell_grid = np.zeros((config.NX, config.N_FREQ), dtype=np.float32)
        haskell_grid[center] = af_h_i
        target = np.zeros((config.NX, config.N_FREQ), dtype=np.float32)
        target[center] = af_ops_i
        mask = np.zeros(config.NX, dtype=np.float32)
        mask[center] = 1.0

        return (
            torch.from_numpy(x_in),
            torch.from_numpy(haskell_grid),
            torch.from_numpy(target),
            torch.from_numpy(mask),
            torch.tensor(cov, dtype=torch.float32),
            torch.tensor(dip_rms, dtype=torch.float32),
        )


def get_rv_loaders(
    max_seeds: int | None = 10,
    batch_size: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    bs = batch_size or config.BATCH_SIZE
    train_ds = RVOpenSeesDeltaDataset(config.RV_TRAIN_SOBOL, max_seeds=max_seeds)
    val_ds = RVOpenSeesDeltaDataset(config.RV_VAL_SOBOL, max_seeds=max_seeds)
    test_ds = RVOpenSeesDeltaDataset(config.RV_TEST_SOBOL, max_seeds=max_seeds)
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=config.NUM_WORKERS),
        DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=config.NUM_WORKERS),
        DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=config.NUM_WORKERS),
    )
