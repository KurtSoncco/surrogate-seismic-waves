"""Overlay Box ood_dipping Toro and Passeri spectra onto a held-out pack.

``toro_comparison`` and ``passeri_comparison`` store one geomean per Sobol
physics case (fixed interface depth, and depth along the dip). Runs share a
physics id. Dmult is not in these files and is not written here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import config

BOX_OOD_DIPPING = config.data_root() / "ood_dipping"
_DMULT_KEYS = ("tf_dmult", "tf_dmult_p84", "sigma_ln_dmult")
_REPLACED_KEYS = (
    "tf_toro",
    "tf_passeri",
    "sigma_ln_toro",
    "sigma_ln_passeri",
    *_DMULT_KEYS,
)


def _rows_for_runs(ens_path: Path, sobol_ids: np.ndarray) -> np.ndarray:
    import h5py

    with h5py.File(ens_path) as f:
        ens_sid = np.asarray(f["sobol_id"], dtype=int)
    row_of = np.full(int(ens_sid.max()) + 1, -1, dtype=int)
    row_of[ens_sid] = np.arange(ens_sid.shape[0])
    rows = row_of[np.asarray(sobol_ids, dtype=int)]
    if np.any(rows < 0):
        missing = np.unique(sobol_ids[rows < 0])
        raise KeyError(f"{ens_path.name} has no sobol_id {missing.tolist()}")
    return rows


def _take(
    ens_path: Path, rows: np.ndarray, keys: tuple[str, ...]
) -> dict[str, np.ndarray]:
    import h5py

    with h5py.File(ens_path) as f:
        return {key: np.asarray(f[key], dtype=np.float64)[rows] for key in keys}


def sobol_ids_for_runs(
    runs: np.ndarray, *, box_root: Path = BOX_OOD_DIPPING
) -> np.ndarray:
    """Map corpus run index to Sobol physics id via ``toro_comparison/pearson.h5``."""
    import h5py

    runs = np.asarray(runs, dtype=int)
    with h5py.File(box_root / "toro_comparison" / "pearson.h5") as f:
        index = np.asarray(f["index"], dtype=int)
        sobol = np.asarray(f["sobol_id"], dtype=int)
    if index.shape != sobol.shape:
        raise ValueError("pearson.h5 index and sobol_id differ in length")
    sobol_of_run = np.full(int(index.max()) + 1, -1, dtype=int)
    sobol_of_run[index] = sobol
    if (
        np.any(runs < 0)
        or np.any(runs >= sobol_of_run.shape[0])
        or np.any(sobol_of_run[runs] < 0)
    ):
        raise KeyError("sample_idx is outside the ood_dipping pearson index")
    return sobol_of_run[runs]


def apply_ood_dipping_toro_passeri(
    pack: dict[str, np.ndarray],
    *,
    box_root: Path = BOX_OOD_DIPPING,
) -> dict[str, np.ndarray]:
    """Replace Toro / Passeri with Box fixed-H and dip-depth geomeans.

    Drops any Dmult arrays and the previous single-arm Toro / Passeri spectra.
    The OpenSees center recorder is the Box ``tf_2d_center`` used to score
    those geomeans.
    """
    import h5py

    if "sample_idx" not in pack:
        raise KeyError("pack needs sample_idx (ood_dipping run index)")
    runs = np.asarray(pack["sample_idx"], dtype=int)
    box_root = Path(box_root)
    sobol_ids = sobol_ids_for_runs(runs, box_root=box_root)
    toro_path = box_root / "toro_comparison" / "ensembles.h5"
    pas_path = box_root / "passeri_comparison" / "ensembles.h5"
    toro_rows = _rows_for_runs(toro_path, sobol_ids)
    pas_rows = _rows_for_runs(pas_path, sobol_ids)
    toro = _take(
        toro_path,
        toro_rows,
        (
            "toro_fixed_geomean",
            "toro_fixed_sigma_ln",
            "toro_dip_geomean",
            "toro_dip_sigma_ln",
        ),
    )
    pas = _take(
        pas_path,
        pas_rows,
        (
            "passeri_fixed_geomean",
            "passeri_fixed_sigma_ln",
            "passeri_dip_geomean",
            "passeri_dip_sigma_ln",
        ),
    )
    with h5py.File(box_root / "toro_comparison" / "tf_2d_center.h5") as f:
        center = np.asarray(f["tf_center"], dtype=np.float64)[runs]

    out = {key: pack[key] for key in pack if key not in _REPLACED_KEYS}
    out["sobol_id"] = sobol_ids
    out["tf_toro_fixed"] = toro["toro_fixed_geomean"]
    out["sigma_ln_toro_fixed"] = toro["toro_fixed_sigma_ln"]
    out["tf_toro_dip"] = toro["toro_dip_geomean"]
    out["sigma_ln_toro_dip"] = toro["toro_dip_sigma_ln"]
    out["tf_passeri_fixed"] = pas["passeri_fixed_geomean"]
    out["sigma_ln_passeri_fixed"] = pas["passeri_fixed_sigma_ln"]
    out["tf_passeri_dip"] = pas["passeri_dip_geomean"]
    out["sigma_ln_passeri_dip"] = pas["passeri_dip_sigma_ln"]

    ops = np.array(out["tf_opensees"], dtype=np.float64, copy=True)
    if ops.ndim == 3:
        ops[:, ops.shape[1] // 2, :] = center
    elif ops.ndim == 2:
        ops = center
    else:
        raise ValueError(f"tf_opensees ndim {ops.ndim} is not 2 or 3")
    out["tf_opensees"] = ops
    return out
