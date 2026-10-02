"""Separate the Pearson<0.9 tail: joint-corner occupancy vs seed-to-seed ceiling.

No new OpenSees. Nested packs + signed-cache meta only.

    uv run python experiments/DeepONet-Residual/response_variability/diagnostics/tail_a_vs_b.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_EXP = Path(__file__).resolve().parents[2]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import config  # noqa: E402

from response_variability.gino_bias import _as_3d, central_slice  # noqa: E402
from response_variability.metrics import band_pearson  # noqa: E402

PACK_6D = ("vs1", "H", "cov", "rH", "aHV", "vs2")
META_6D = ("Vs1", "H", "CoV", "rH", "aHV", "Vs2")
DECIMALS = 8
PEARSON_LO = 0.1
PEARSON_HI = 10.0
TAIL_THR = 0.9
OUT_DIR = config.RESULTS_DIR / "response_variability" / "eval_bias"
PACK_DIR = config.RESULTS_DIR / "presentation"
PER_SAMPLE_CSV = config.RESULTS_DIR / "response_variability" / "gino_bias" / "per_sample.csv"

# 6D columns: Vs1, H, CoV, rH, aHV, Vs2
COL_COV, COL_RH = 2, 3
COLS_COV_RH = (COL_COV, COL_RH)
COLS_5AXIS = (0, 1, 2, 3, 4)  # drop Vs2


def param_matrix(
    arrays: dict[str, np.ndarray],
    keys: tuple[str, ...] = PACK_6D,
    *,
    idx: np.ndarray | None = None,
    decimals: int = DECIMALS,
) -> np.ndarray:
    """Stack named 1-D fields into a rounded (n, d) matrix."""
    n = int(np.asarray(arrays[keys[0]]).shape[0]) if idx is None else int(len(idx))
    cols = []
    loc = None if idx is None else np.asarray(idx, dtype=int)
    for k in keys:
        if k not in arrays:
            cols.append(np.full(n, np.nan, dtype=float))
            continue
        v = np.asarray(arrays[k], dtype=float)
        cols.append(v if loc is None else v[loc])
    return np.round(np.column_stack(cols), decimals)


def cell_ids(x: np.ndarray) -> np.ndarray:
    """Integer cell id per row; NaNs are a distinct sentinel per column."""
    y = np.asarray(x, dtype=float).copy()
    for j in range(y.shape[1]):
        col = y[:, j]
        finite = np.isfinite(col)
        fill = (np.nanmax(col[finite]) + 1.0e6) if np.any(finite) else 1.0e12
        col[~finite] = fill
        y[:, j] = col
    _, inv = np.unique(y, axis=0, return_inverse=True)
    return inv.astype(int)


def cell_occupancy(x: np.ndarray) -> dict[str, float]:
    ids = cell_ids(x)
    n_id = int(len(np.unique(ids)))
    counts = np.bincount(ids)
    return {
        "n_files": float(len(ids)),
        "n_unique_6d": float(n_id),
        "mean_files_per_id": float(np.mean(counts)) if n_id else float("nan"),
        "n_ids_ge2": float(int(np.sum(counts >= 2))),
        "n_files_in_replicated": float(int(np.sum(counts[counts >= 2]))),
    }


def quantile_cuts(train: np.ndarray, cols: tuple[int, ...], q: float) -> np.ndarray:
    t = np.asarray(train, dtype=float)[:, list(cols)]
    finite = np.all(np.isfinite(t), axis=1)
    t = t[finite]
    if t.shape[0] < 4:
        return np.full(len(cols), np.nan)
    return np.quantile(t, q, axis=0)


def in_joint_cell(
    x: np.ndarray, cuts: np.ndarray, cols: tuple[int, ...]
) -> np.ndarray:
    if not np.all(np.isfinite(cuts)):
        return np.zeros(len(x), dtype=bool)
    sl = np.asarray(x, dtype=float)[:, list(cols)]
    finite = np.all(np.isfinite(sl), axis=1)
    return finite & np.all(sl >= cuts[None, :], axis=1)


def occupancy_row(
    *,
    domain: str,
    split: str,
    cell: str,
    x: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    occ = cell_occupancy(x)
    if mask is None:
        mask = np.ones(len(x), dtype=bool)
    xin = x[mask]
    inside = cell_occupancy(xin) if np.any(mask) else {
        "n_files": 0.0,
        "n_unique_6d": 0.0,
        "mean_files_per_id": float("nan"),
        "n_ids_ge2": 0.0,
        "n_files_in_replicated": 0.0,
    }
    n_files = occ["n_files"]
    n_unique = occ["n_unique_6d"]
    return {
        "domain": domain,
        "split": split,
        "cell": cell,
        "n_files": int(n_files),
        "n_unique_6d": int(n_unique),
        "n_files_in_cell": int(inside["n_files"]),
        "n_unique_6d_in_cell": int(inside["n_unique_6d"]),
        "frac_files": float(inside["n_files"] / n_files) if n_files else float("nan"),
        "frac_unique": float(inside["n_unique_6d"] / n_unique) if n_unique else float("nan"),
        "mean_files_per_id": occ["mean_files_per_id"],
        "n_ids_ge2": int(occ["n_ids_ge2"]),
    }


def central_pearson(pred: np.ndarray, true: np.ndarray, freq: np.ndarray) -> np.ndarray:
    p = central_slice(pred)
    t = central_slice(true)
    n = t.shape[0]
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        out[i] = band_pearson(p[i], t[i], freq, lo=PEARSON_LO, hi=PEARSON_HI)
    return out


def array_pearson(pred: np.ndarray, true: np.ndarray, freq: np.ndarray) -> np.ndarray:
    p = _as_3d(pred)
    t = _as_3d(true)
    n = t.shape[0]
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        out[i] = band_pearson(p[i], t[i], freq, lo=PEARSON_LO, hi=PEARSON_HI)
    return out


def recorder_pearson(
    pred: np.ndarray, true: np.ndarray, freq: np.ndarray, rec: int
) -> np.ndarray:
    p = _as_3d(pred)
    t = _as_3d(true)
    n = t.shape[0]
    rec = int(np.clip(rec, 0, t.shape[1] - 1))
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        out[i] = band_pearson(p[i, rec], t[i, rec], freq, lo=PEARSON_LO, hi=PEARSON_HI)
    return out


def pairwise_ops_pearson(
    tf_ops: np.ndarray, freq: np.ndarray, members: np.ndarray
) -> dict[str, float]:
    """Median OpenSees–OpenSees Pearson over unordered pairs in ``members``."""
    idx = np.asarray(members, dtype=int)
    ops = _as_3d(tf_ops)
    central: list[float] = []
    array: list[float] = []
    mid = ops.shape[1] // 2
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = int(idx[a]), int(idx[b])
            central.append(
                band_pearson(
                    ops[i, mid], ops[j, mid], freq, lo=PEARSON_LO, hi=PEARSON_HI
                )
            )
            array.append(
                band_pearson(ops[i], ops[j], freq, lo=PEARSON_LO, hi=PEARSON_HI)
            )
    c = np.asarray(central, dtype=float)
    a = np.asarray(array, dtype=float)
    return {
        "n_pairs": float(len(c)),
        "ops_ops_pearson_median_central": float(np.nanmedian(c)) if c.size else float("nan"),
        "ops_ops_pearson_p16_central": float(np.nanpercentile(c, 16)) if c.size else float("nan"),
        "ops_ops_pearson_p84_central": float(np.nanpercentile(c, 84)) if c.size else float("nan"),
        "ops_ops_pearson_median_array": float(np.nanmedian(a)) if a.size else float("nan"),
    }


def seed_ceiling_rows(
    pack: dict[str, np.ndarray],
    *,
    domain: str,
    gino_ops: np.ndarray,
) -> list[dict[str, Any]]:
    x = param_matrix(pack)
    ids = cell_ids(x)
    freq = np.asarray(pack["freq"], dtype=float)
    tf_ops = pack["tf_opensees"]
    rows: list[dict[str, Any]] = []
    for cid in np.unique(ids):
        members = np.where(ids == cid)[0]
        if members.size < 2:
            continue
        pair = pairwise_ops_pearson(tf_ops, freq, members)
        gp = gino_ops[members]
        med_g = float(np.nanmedian(gp))
        med_o = pair["ops_ops_pearson_median_central"]
        proto = x[int(members[0])]
        rows.append(
            {
                "domain": domain,
                "cell_id": int(cid),
                "n_seeds": int(members.size),
                "vs1": float(proto[0]),
                "H": float(proto[1]),
                "cov": float(proto[2]),
                "rH": float(proto[3]),
                "aHV": float(proto[4]),
                "vs2": float(proto[5]),
                "pack_indices": ",".join(str(int(i)) for i in members),
                "gino_ops_pearson_mean_central": float(np.nanmean(gp)),
                "gino_ops_pearson_median_central": med_g,
                "gino_ops_pearson_min_central": float(np.nanmin(gp)),
                "gap_median_central": med_g - med_o,
                **pair,
            }
        )
    rows.sort(key=lambda r: r["gino_ops_pearson_min_central"])
    return rows


def recorder_sensitivity_rows(
    pack: dict[str, np.ndarray],
    *,
    domain: str,
    gino_ops_central: np.ndarray,
) -> list[dict[str, Any]]:
    freq = np.asarray(pack["freq"], dtype=float)
    gino = pack["tf_gino"]
    ops = pack["tf_opensees"]
    n_rec = int(_as_3d(ops).shape[1])
    mid = n_rec // 2
    arr = array_pearson(gino, ops, freq)
    rec0 = recorder_pearson(gino, ops, freq, 0)
    rec_last = recorder_pearson(gino, ops, freq, n_rec - 1)
    rec_m2 = recorder_pearson(gino, ops, freq, max(mid - 2, 0))
    rec_p2 = recorder_pearson(gino, ops, freq, min(mid + 2, n_rec - 1))
    n = int(ops.shape[0])
    rows = []
    for i in range(n):
        pc = float(gino_ops_central[i])
        pa = float(arr[i])
        rows.append(
            {
                "domain": domain,
                "sample": i,
                "n_rec": n_rec,
                "pearson_central": pc,
                "pearson_array": pa,
                "pearson_rec0": float(rec0[i]),
                "pearson_rec_last": float(rec_last[i]),
                "pearson_rec_m2": float(rec_m2[i]),
                "pearson_rec_p2": float(rec_p2[i]),
                "below_0.9_central": int(pc < TAIL_THR),
                "below_0.9_array": int(pa < TAIL_THR),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return path
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


def _load_meta(cache_dir: Path) -> dict[str, np.ndarray]:
    return dict(np.load(Path(cache_dir) / "meta.npz", allow_pickle=True))


def _split_indices(name: str) -> dict[str, np.ndarray]:
    from domain_splits import load_split

    if name == "iid":
        path = config.CACHE_DIR / "splits" / "iid_n1000_seed42.npz"
    else:
        path = config.CACHE_DIR / "splits" / f"{name}_seed42.npz"
    return load_split(path)


def _cloud_6d(cache_dir: Path, idx: np.ndarray) -> np.ndarray:
    meta = _load_meta(cache_dir)
    return param_matrix(meta, META_6D, idx=idx)


def _cell_defs(train: np.ndarray) -> list[tuple[str, np.ndarray, tuple[int, ...]]]:
    defs = []
    for q, name in ((0.50, "cov_rh_q50"), (0.75, "cov_rh_q75"), (0.80, "cov_rh_q80")):
        cuts = quantile_cuts(train, COLS_COV_RH, q)
        defs.append((name, cuts, COLS_COV_RH))
    cuts5 = quantile_cuts(train, COLS_5AXIS, 0.80)
    defs.append(("top20_5axis", cuts5, COLS_5AXIS))
    return defs


def build_occupancy(
    *,
    iid_train: np.ndarray,
    iid_val: np.ndarray,
    iid_test: np.ndarray,
    iid_n3000: np.ndarray | None,
    dip_train: np.ndarray,
    dip_test: np.ndarray,
    iid_tail: np.ndarray,
    dip_tail: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    iid_cells = _cell_defs(iid_train)
    dip_cells = _cell_defs(dip_train) if len(dip_train) else iid_cells
    splits = [
        ("iid", "train", iid_train, iid_cells),
        ("iid", "val", iid_val, iid_cells),
        ("iid", "test", iid_test, iid_cells),
        ("iid", "test_pearson_lt_0.9", iid_tail, iid_cells),
        ("dipping", "train", dip_train, dip_cells),
        ("dipping", "test", dip_test, dip_cells),
        ("dipping", "test_pearson_lt_0.9", dip_tail, dip_cells),
    ]
    if iid_n3000 is not None and len(iid_n3000):
        splits.append(("iid", "n3000_screen", iid_n3000, iid_cells))
    for domain, split, x, cells in splits:
        if len(x) == 0:
            continue
        rows.append(
            occupancy_row(
                domain=domain, split=split, cell="all", x=x, mask=np.ones(len(x), dtype=bool)
            )
        )
        for name, cuts, cols in cells:
            rows.append(
                occupancy_row(
                    domain=domain,
                    split=split,
                    cell=name,
                    x=x,
                    mask=in_joint_cell(x, cuts, cols),
                )
            )
    return rows


def tail_clusters(
    pack: dict[str, np.ndarray],
    tail_mask: np.ndarray,
    *,
    domain: str,
    pearson: np.ndarray,
) -> list[dict[str, Any]]:
    x = param_matrix(pack)
    ids = cell_ids(x)
    groups: dict[int, list[int]] = defaultdict(list)
    for i, flag in enumerate(tail_mask):
        if flag:
            groups[int(ids[i])].append(i)
    rows = []
    for cid, members in sorted(groups.items(), key=lambda kv: min(pearson[j] for j in kv[1])):
        proto = x[members[0]]
        rows.append(
            {
                "domain": domain,
                "cell_id": cid,
                "n_tail": len(members),
                "pack_indices": ",".join(str(i) for i in members),
                "pearson_min": float(np.min(pearson[members])),
                "pearson_median": float(np.median(pearson[members])),
                "vs1": float(proto[0]),
                "H": float(proto[1]),
                "cov": float(proto[2]),
                "rH": float(proto[3]),
                "aHV": float(proto[4]),
                "vs2": float(proto[5]),
                "n_cell_in_test": int(np.sum(ids == cid)),
            }
        )
    return rows


def load_per_sample_pearson(path: Path, domain: str, n: int) -> np.ndarray | None:
    if not Path(path).is_file():
        return None
    import pandas as pd

    df = pd.read_csv(path)
    sub = df[df["domain"] == domain].sort_values("sample")
    if len(sub) != n:
        return None
    return np.asarray(sub["pearson"], dtype=float)


def plot_sample50(pack: dict[str, np.ndarray], out_path: Path, *, i: int = 50) -> Path:
    import matplotlib.pyplot as plt

    from response_variability.plots.plot_presentation import (
        _plot_diff_panel,
        _plot_tf_panel,
        attach_stoch_from_cache,
    )
    from response_variability.style import apply_nature_style, figsize, panel_letter, savefig

    apply_nature_style()
    pack = attach_stoch_from_cache(pack, "iid")
    fig, axes = plt.subplots(1, 3, figsize=figsize("double", height_mm=72))
    vs = np.asarray(pack["vs_2d"][i], dtype=float) if "vs_2d" in pack else None
    if vs is not None and np.isfinite(vs).any():
        soil = int(pack["soil_nz"][i]) if "soil_nz" in pack else vs.shape[0]
        crop = vs[: max(soil, 1)]
        axes[0].imshow(crop, aspect="auto", origin="upper", cmap="viridis")
        axes[0].set_xlabel(r"$x$ (cells)")
        axes[0].set_ylabel("depth (cells)")
        axes[0].set_title(r"$V_s$ strip")
    else:
        axes[0].text(0.5, 0.5, "no vs_2d", ha="center", va="center")
        axes[0].set_axis_off()
    _plot_tf_panel(axes[1], pack, i)
    _plot_diff_panel(axes[2], pack, i)
    cov = float(pack["cov"][i])
    rh = float(pack["rH"][i]) if "rH" in pack else float("nan")
    ahv = float(pack["aHV"][i]) if "aHV" in pack else float("nan")
    vs1 = float(pack["vs1"][i])
    H = float(pack["H"][i])
    axes[1].set_title(
        rf"IID sample {i}: $V_{{s1}}$={vs1:.0f}, $H$={H:.0f}, "
        rf"CoV={cov:.2f}, $r_H$={rh:.0f} m, $a_{{HV}}$={ahv:.1f}",
        fontsize=6.5,
    )
    for ax, letter in zip(axes, "abc"):
        panel_letter(ax, letter, x=0.02, y=0.98)
    fig.tight_layout(w_pad=0.7)
    return savefig(fig, out_path)


def plot_recorder_sensitivity(rows: list[dict[str, Any]], out_path: Path) -> Path:
    import matplotlib.pyplot as plt

    from response_variability.style import apply_nature_style, figsize, savefig

    apply_nature_style()
    dip = [r for r in rows if r["domain"] == "dipping" and r["below_0.9_central"]]
    fig, ax = plt.subplots(figsize=figsize("single", height_mm=70))
    if dip:
        c = np.array([r["pearson_central"] for r in dip])
        a = np.array([r["pearson_array"] for r in dip])
        y = np.arange(len(dip))
        order = np.argsort(c)
        ax.plot(c[order], y, "o", color="#0072B2", ms=3.2, label="central")
        ax.plot(a[order], y, "s", color="#D55E00", ms=3.0, label="array mean")
        for yi, ci, ai in zip(y, c[order], a[order]):
            ax.plot([ci, ai], [yi, yi], color="0.75", lw=0.5, zorder=0)
        ax.axvline(TAIL_THR, color="0.4", ls="--", lw=0.6)
        ax.set_yticks([])
        ax.set_xlabel("Pearson")
        ax.set_ylabel("dipping tail cases (central $<0.9$)")
        ax.legend(loc="lower right")
    fig.tight_layout()
    return savefig(fig, out_path)


def _attach_pack(domain: str, pack_dir: Path) -> dict[str, np.ndarray]:
    from response_variability.plots.plot_presentation import (
        attach_stoch_from_cache,
        load_pack,
        pack_path,
    )

    pack = load_pack(pack_path(pack_dir, domain))
    return attach_stoch_from_cache(pack, domain)


def run(
    *,
    pack_dir: Path = PACK_DIR,
    out_dir: Path = OUT_DIR,
    per_sample_csv: Path = PER_SAMPLE_CSV,
    sample_i: int = 50,
) -> dict[str, Path]:
    from mix_ladder import iid_n1000_split
    from ood_signed_cache import cache_dir_for

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    iid_pack = _attach_pack("iid", pack_dir)
    dip_pack = _attach_pack("dipping", pack_dir)

    iid_p = load_per_sample_pearson(per_sample_csv, "iid", len(iid_pack["vs1"]))
    dip_p = load_per_sample_pearson(per_sample_csv, "dipping", len(dip_pack["vs1"]))
    if iid_p is None:
        iid_p = central_pearson(iid_pack["tf_gino"], iid_pack["tf_opensees"], iid_pack["freq"])
    if dip_p is None:
        dip_p = central_pearson(dip_pack["tf_gino"], dip_pack["tf_opensees"], dip_pack["freq"])

    iid_split = iid_n1000_split()
    n1000 = config.CACHE_DIR / "n1000_seed42"
    iid_train = _cloud_6d(n1000, np.asarray(iid_split["train"], dtype=int))
    iid_val = _cloud_6d(n1000, np.asarray(iid_split["val"], dtype=int))
    iid_test = param_matrix(iid_pack)
    n3000 = config.CACHE_DIR / "n3000_seed42"
    iid_n3000 = None
    if (n3000 / "meta.npz").is_file():
        meta = _load_meta(n3000)
        n_all = int(np.asarray(meta["Vs1"]).shape[0])
        iid_n3000 = param_matrix(meta, META_6D, idx=np.arange(n_all))

    dip_cache = cache_dir_for("ood_dipping")
    dip_split = _split_indices("ood_dipping")
    dip_train = _cloud_6d(dip_cache, np.asarray(dip_split["train"], dtype=int))
    dip_test = param_matrix(dip_pack)

    iid_tail = iid_test[iid_p < TAIL_THR]
    dip_tail = dip_test[dip_p < TAIL_THR]
    occ = build_occupancy(
        iid_train=iid_train,
        iid_val=iid_val,
        iid_test=iid_test,
        iid_n3000=iid_n3000,
        dip_train=dip_train,
        dip_test=dip_test,
        iid_tail=iid_tail,
        dip_tail=dip_tail,
    )
    n7680 = config.CACHE_DIR / "n7680_seed42"
    if not (n7680 / "meta.npz").is_file():
        occ.append(
            {
                "domain": "iid",
                "split": "n7680_extras",
                "cell": "unavailable",
                "n_files": -1,
                "n_unique_6d": -1,
                "n_files_in_cell": -1,
                "n_unique_6d_in_cell": -1,
                "frac_files": float("nan"),
                "frac_unique": float("nan"),
                "mean_files_per_id": float("nan"),
                "n_ids_ge2": -1,
            }
        )

    paths: dict[str, Path] = {}
    paths["corner_occupancy"] = _write_csv(out_dir / "corner_occupancy.csv", occ)
    clusters = tail_clusters(
        iid_pack, iid_p < TAIL_THR, domain="iid", pearson=iid_p
    ) + tail_clusters(dip_pack, dip_p < TAIL_THR, domain="dipping", pearson=dip_p)
    paths["tail_clusters"] = _write_csv(out_dir / "tail_6d_clusters.csv", clusters)

    ceil = seed_ceiling_rows(iid_pack, domain="iid", gino_ops=iid_p) + seed_ceiling_rows(
        dip_pack, domain="dipping", gino_ops=dip_p
    )
    paths["seed_ceiling"] = _write_csv(out_dir / "seed_ceiling.csv", ceil)

    rec = recorder_sensitivity_rows(
        iid_pack, domain="iid", gino_ops_central=iid_p
    ) + recorder_sensitivity_rows(dip_pack, domain="dipping", gino_ops_central=dip_p)
    paths["recorder_sensitivity"] = _write_csv(out_dir / "recorder_sensitivity.csv", rec)
    paths["recorder_plot"] = plot_recorder_sensitivity(
        rec, out_dir / "recorder_sensitivity_dipping.png"
    )
    paths["sample50"] = plot_sample50(iid_pack, out_dir / "sample50_iid.png", i=sample_i)

    x_ids = cell_ids(param_matrix(iid_pack))
    n_same = int(np.sum(x_ids == x_ids[sample_i]))
    summary = {
        "iid_n_tail": int(np.sum(iid_p < TAIL_THR)),
        "dip_n_tail": int(np.sum(dip_p < TAIL_THR)),
        "iid_n_unique_tail": int(len({int(i) for i in x_ids[iid_p < TAIL_THR]})),
        "sample50_n_in_cell": n_same,
        "n7680_meta": bool((n7680 / "meta.npz").is_file()),
        "iid_n_replicated_cells": sum(1 for r in ceil if r["domain"] == "iid"),
        "dip_n_replicated_cells": sum(1 for r in ceil if r["domain"] == "dipping"),
    }
    paths["summary_json"] = out_dir / "tail_a_vs_b_summary.json"
    import json

    paths["summary_json"].write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    for k, p in paths.items():
        print(f"  {k}: {p}", flush=True)
    return paths


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack-dir", type=Path, default=PACK_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--per-sample-csv", type=Path, default=PER_SAMPLE_CSV)
    p.add_argument("--sample", type=int, default=50)
    args = p.parse_args(argv)
    run(
        pack_dir=args.pack_dir,
        out_dir=args.out_dir,
        per_sample_csv=args.per_sample_csv,
        sample_i=args.sample,
    )


if __name__ == "__main__":
    main()
