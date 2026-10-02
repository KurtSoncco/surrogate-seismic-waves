"""Overlay Box ``hallal_vs_2d`` spectra onto a pack, keyed by run index.

Toro, Passeri, and Dmult live once per Sobol ``sample_id`` (256 cases).
Pretell and the 2D center spectrum are stored per run (7,680). Pretell rows
marked invalid are left as NaN.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

BOX_DIR = Path("/mnt/box/GIG Lab - UC Berkeley/Projects/Neural Operator/data/hallal_vs_2d")


def apply_hallal_ground_truth(
    pack: dict[str, np.ndarray],
    *,
    box_dir: Path = BOX_DIR,
) -> dict[str, np.ndarray]:
    """Return a copy of ``pack`` whose classical spectra come from ``box_dir``."""
    import h5py

    if "sample_idx" not in pack:
        raise KeyError("pack needs sample_idx (global run index into the IID corpus)")
    runs = np.asarray(pack["sample_idx"], dtype=int)
    n = int(runs.shape[0])
    box_dir = Path(box_dir)

    with h5py.File(box_dir / "pearson_center.h5") as f:
        order = np.asarray(f["index"], dtype=int)
        sample_of_run = np.empty(int(order.max()) + 1, dtype=int)
        sample_of_run[order] = np.asarray(f["sample_id"], dtype=int)
    sample_ids = sample_of_run[runs]

    with h5py.File(box_dir / "ensembles.h5") as f:
        ens_sid = np.asarray(f["sample_id"], dtype=int)
        row_of = np.empty(int(ens_sid.max()) + 1, dtype=int)
        row_of[ens_sid] = np.arange(ens_sid.shape[0])
        rows = row_of[sample_ids]
        toro_all = np.asarray(f["toro_geomean"], dtype=np.float64)
        toro_sig_all = np.asarray(f["toro_sigma_ln"], dtype=np.float64)
        pas_all = np.asarray(f["passeri_geomean"], dtype=np.float64)
        pas_sig_all = np.asarray(f["passeri_sigma_ln"], dtype=np.float64)
        dmult_all = np.asarray(f["dmult_tf"], dtype=np.float64)
        freq = np.asarray(f["freq"], dtype=np.float64)
    toro = toro_all[rows]
    toro_sig = toro_sig_all[rows]
    pas = pas_all[rows]
    pas_sig = pas_sig_all[rows]
    dmult = dmult_all[rows]

    with h5py.File(box_dir / "tf_2d_center.h5") as f:
        tf_center = np.asarray(f["tf_center"], dtype=np.float64)[runs]

    with h5py.File(box_dir / "pretell_ensembles.h5") as f:
        valid_all = np.asarray(f["valid"], dtype=bool)
        pretell_all = np.asarray(f["geomean"], dtype=np.float64)
        pretell_sig_all = np.asarray(f["sigma_ln"], dtype=np.float64)
    valid = valid_all[runs]
    pretell = pretell_all[runs]
    pretell_sig = pretell_sig_all[runs]
    pretell[~valid] = np.nan
    pretell_sig[~valid] = np.nan

    out = dict(pack)
    out["tf_toro"] = toro
    out["sigma_ln_toro"] = toro_sig
    out["tf_passeri"] = pas
    out["sigma_ln_passeri"] = pas_sig
    out["tf_dmult"] = dmult
    out["tf_pretell"] = pretell
    out["sigma_ln_pretell"] = pretell_sig
    if "sigma_ln_dmult" in out:
        out["sigma_ln_dmult"] = np.full((n, freq.shape[0]), np.nan, dtype=np.float64)
    if "tf_dmult_p84" in out:
        out["tf_dmult_p84"] = np.full_like(dmult, np.nan)

    ops = np.array(out["tf_opensees"], dtype=np.float64, copy=True)
    if ops.ndim == 3:
        ops[:, ops.shape[1] // 2, :] = tf_center
    elif ops.ndim == 2:
        ops = tf_center
    else:
        raise ValueError(f"tf_opensees ndim {ops.ndim} is not 2 or 3")
    out["tf_opensees"] = ops
    out["freq"] = np.asarray(out.get("freq", freq), dtype=np.float64)
    out["hallal_sample_id"] = sample_ids
    return out
