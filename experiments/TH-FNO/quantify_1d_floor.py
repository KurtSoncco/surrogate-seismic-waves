#!/usr/bin/env python3
"""Quantify 1D residual floor: local Haskell (+ Pretell) vs OpenSees-2D on RV.

Reports hold-out-style rel L2 / Pearson / peak bias for:
  - local-column Thomson–Haskell AF_within (center recorder)
  - single-layer trend Haskell (Vs1, H, Vs2)
  - Pretell (from existing analysis CSV when available)

Also attempts GIFNO corpus hold-out if TF cache + H5 are mounted.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError as e:
    raise SystemExit("h5py required") from e
try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

_EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP))
import config

config.setup_import_paths()

from context_features import residual_gate_scalar  # noqa: E402
from haskell_baseline import haskell_af_within, haskell_trend_af_within  # noqa: E402

_EPS = 1e-12


def _rel_l2(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, float).ravel()
    true = np.asarray(true, float).ravel()
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + _EPS))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if a.size < 2 or np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _peak(freq: np.ndarray, af: np.ndarray) -> tuple[float, float]:
    i = int(np.argmax(af))
    return float(freq[i]), float(af[i])


def _stats(xs: list[float]) -> dict[str, float]:
    v = np.asarray(xs, float)
    v = v[np.isfinite(v)]
    return {
        "n": int(v.size),
        "mean": float(v.mean()) if v.size else float("nan"),
        "median": float(np.median(v)) if v.size else float("nan"),
        "p10": float(np.percentile(v, 10)) if v.size else float("nan"),
        "p90": float(np.percentile(v, 90)) if v.size else float("nan"),
    }


def _load_af(h5_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not h5_path.is_file():
        return None
    with h5py.File(h5_path, "r") as f:
        return (
            f["transfer_function"]["AF"][:].astype(np.float64),
            f["transfer_function"]["freq"][:].astype(np.float64),
        )


def quantify_rv(h5_dir: Path, out_dir: Path, max_seeds: int | None = None) -> dict:
    rv_root = config.RV_ROOT
    sys.path.insert(0, str(rv_root))
    os.environ.pop("RV_SMOKE", None)
    from manifest import (  # noqa: WPS433
        _hallal_block_size,
        active_rf_seeds,
        active_sobol_count,
        index_to_params,
        total_combinations,
    )

    hallal0 = _hallal_block_size(active_sobol_count())
    lookup: dict[tuple[int, str, int], int] = {}
    for i in range(hallal0, total_combinations()):
        p = index_to_params(i)
        if p.method in ("grf_2d", "opensees_2d", "pretell"):
            lookup[(p.sobol_id, p.method, p.seed)] = i

    seeds = active_rf_seeds()
    if max_seeds is not None:
        seeds = seeds[: max_seeds]

    rows: list[dict] = []
    per_seed = {
        "local_rel": [],
        "local_pear": [],
        "trend_rel": [],
        "trend_pear": [],
        "pretell_rel": [],
        "pretell_pear": [],
        "local_df": [],
        "local_dlnA": [],
        "trend_df": [],
        "trend_dlnA": [],
    }
    geo = {
        "local_rel": [],
        "local_pear": [],
        "trend_rel": [],
        "trend_pear": [],
        "pretell_rel": [],
        "pretell_pear": [],
    }

    center_full = config.BC_WIDTH + config.NX // 2  # column on 1500-wide grid

    for sid in range(active_sobol_count()):
        stacks = {k: [] for k in ("ops", "local", "trend", "pretell")}
        for seed in seeds:
            io = lookup.get((sid, "opensees_2d", seed))
            if io is None:
                continue
            ops_path = h5_dir / f"run_{io}.h5"
            loaded = _load_af(ops_path)
            if loaded is None:
                continue
            af_ops, freq = loaded

            with h5py.File(ops_path, "r") as f:
                vs = f["Vs_field"][:]
                zeta = f["Damping_zeta"][:]
                dz = float(f["grid"].attrs.get("dz", 1.0))
                vs1 = float(f["params"].attrs["Vs1"])
                H = float(f["params"].attrs["H"])
                vs2 = float(f["params"].attrs["Vs2"])
                cov = float(f["params"].attrs["CoV"])
                xi_surf = float(zeta[0, center_full]) if center_full < zeta.shape[1] else 0.05

            soil_nz = max(1, int(round(H / dz)))
            col = min(center_full, vs.shape[1] - 1)
            af_local = haskell_af_within(
                freq,
                vs[:, col],
                zeta[:, col],
                dz=dz,
                vs_rock=vs2,
                soil_nz=soil_nz,
            )
            af_trend = haskell_trend_af_within(
                freq, vs1=vs1, H=H, vs2=vs2, xi=max(xi_surf, 0.01)
            )

            af_pretell = None
            ip = lookup.get((sid, "pretell", seed))
            if ip is not None:
                pret = _load_af(h5_dir / f"run_{ip}.h5")
                if pret is not None:
                    af_p, fp = pret
                    af_pretell = (
                        af_p
                        if fp.shape == freq.shape and np.allclose(fp, freq, atol=1e-12)
                        else np.interp(freq, fp, af_p)
                    )

            def _pack(name: str, af: np.ndarray) -> None:
                r = _rel_l2(af, af_ops)
                pe = _pearson(af, af_ops)
                f_p, a_p = _peak(freq, af)
                f_t, a_t = _peak(freq, af_ops)
                per_seed[f"{name}_rel"].append(r)
                per_seed[f"{name}_pear"].append(pe)
                if name in ("local", "trend"):
                    per_seed[f"{name}_df"].append(abs(f_p - f_t))
                    per_seed[f"{name}_dlnA"].append(float(np.log(max(a_p, _EPS)) - np.log(max(a_t, _EPS))))
                rows.append(
                    {
                        "sobol_id": sid,
                        "seed": seed,
                        "method": name,
                        "rel_l2": r,
                        "pearson": pe,
                        "delta_f_peak": f_p - f_t,
                        "delta_ln_A_peak": float(np.log(max(a_p, _EPS)) - np.log(max(a_t, _EPS))),
                        "cov": cov,
                        "gate": residual_gate_scalar(cov, 0.0),
                    }
                )
                stacks[name].append(af)

            _pack("local", af_local)
            _pack("trend", af_trend)
            stacks["ops"].append(af_ops)
            if af_pretell is not None:
                _pack("pretell", af_pretell)
                stacks["pretell"].append(af_pretell)

        if len(stacks["ops"]) >= 2:
            g_ops = np.exp(np.mean(np.log(np.maximum(stacks["ops"], _EPS)), axis=0))
            for name in ("local", "trend", "pretell"):
                if len(stacks[name]) == len(stacks["ops"]):
                    g = np.exp(np.mean(np.log(np.maximum(stacks[name], _EPS)), axis=0))
                    geo[f"{name}_rel"].append(_rel_l2(g, g_ops))
                    geo[f"{name}_pear"].append(_pearson(g, g_ops))

    out_dir.mkdir(parents=True, exist_ok=True)
    per_path = out_dir / "1d_floor_per_seed.csv"
    with per_path.open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    summary: dict[str, dict] = {}
    for prefix, store in (("per_seed", per_seed), ("geomean", geo)):
        for key, vals in store.items():
            if vals:
                summary[f"{prefix}_{key}"] = _stats(vals)

    sum_path = out_dir / "1d_floor_summary.csv"
    with sum_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "n", "mean", "median", "p10", "p90"])
        for k, s in sorted(summary.items()):
            w.writerow([k, s["n"], s["mean"], s["median"], s["p10"], s["p90"]])

    # Existing Pretell-vs-OS geomean CSV cross-check
    band_csv = rv_root / "results" / "analysis" / "tf_band_misfit_vs_opensees.csv"
    if band_csv.is_file():
        with band_csv.open() as f:
            band_rows = list(csv.DictReader(f))
        for method in ("grf_2d", "pretell"):
            vals = [float(r["rel_l2_geomean"]) for r in band_rows if r["method"] == method]
            if vals:
                summary[f"bandcsv_{method}_rel_l2_geomean"] = _stats(vals)

    print("=== 1D residual floor vs OpenSees-2D (Response_Variability) ===")
    for k in (
        "per_seed_local_rel",
        "per_seed_local_pear",
        "per_seed_trend_rel",
        "per_seed_trend_pear",
        "per_seed_pretell_rel",
        "per_seed_pretell_pear",
        "geomean_local_rel",
        "geomean_local_pear",
        "geomean_pretell_rel",
        "geomean_pretell_pear",
    ):
        if k in summary:
            s = summary[k]
            print(
                f"  {k}: n={s['n']} mean={s['mean']:.4f} med={s['median']:.4f} "
                f"p10={s['p10']:.4f} p90={s['p90']:.4f}"
            )
    print(f"Wrote {per_path}")
    print(f"Wrote {sum_path}")
    return summary


def quantify_gifno_holdout(out_dir: Path, limit: int | None = 200) -> dict | None:
    """If GIFNO TF cache exists, compare center-column Haskell to TF targets."""
    if not config.TF_PER_SAMPLE_PATH.is_file() or not config.MANIFEST_PATH.is_file():
        print("[gifno-holdout] TF cache not found — skip (set GIFNO_DATA_ROOT).")
        return None

    import pandas as pd

    from data_loader import get_data_loaders  # noqa: WPS433

    # Use test split only
    _, _, test_loader, freq = get_data_loaders(limit=limit)
    # data_loader returns (x, target, mask); we need raw H5 for Haskell — use manifest
    manifest = pd.read_csv(config.MANIFEST_PATH)
    n = len(manifest) if limit is None else min(limit, len(manifest))
    # deterministic test indices matching GIFNO split
    rng = np.random.RandomState(config.SEED)
    idx = rng.permutation(n)
    n_train = int(config.TRAIN_SPLIT * n)
    n_val = int(config.VAL_SPLIT * n)
    test_idx = idx[n_train + n_val :]

    rec = config.recorder_x_indices()
    center_rec = rec[len(rec) // 2]
    rels, pears = [], []

    # Without reading full corpus H5 paths reliably when Box is down, skip heavy path
    # if H5_DIR empty.
    h5_dir = config.H5_DIR
    if not h5_dir.is_dir() or not any(h5_dir.glob("run_*.h5")):
        print("[gifno-holdout] No H5 runs under", h5_dir, "— skip.")
        return None

    print(f"[gifno-holdout] Evaluating {len(test_idx)} test samples (limit={limit})")
    # Lightweight: iterate test_loader targets at center recorder vs Haskell from batch inputs
    # Inputs are normalized Vs — cannot invert for Haskell. Must load raw H5.
    from data_loader import _resolve_h5_path  # noqa: WPS433

    for ti in test_idx[: min(len(test_idx), limit or len(test_idx))]:
        row = manifest.iloc[int(ti)]
        h5_path = _resolve_h5_path(row["h5_path"])
        if not Path(h5_path).is_file():
            continue
        with h5py.File(h5_path, "r") as f:
            vs = f["Vs_realization_2D"][:]
            zeta = f["Damping_zeta"][:]
            dz = float(f["grid"].attrs.get("dz", config.DZ))
            sl = slice(config.X_SLICE_START, config.X_SLICE_END)
            vs_s = vs[:, sl]
            zeta_s = zeta[:, sl]
        tf = np.load(config.TF_PER_SAMPLE_PATH, mmap_mode="r")[int(ti)]
        # tf: (21, F); center channel
        c = len(tf) // 2
        af_true = np.asarray(tf[c], dtype=float)
        freq_arr = np.load(config.TF_FREQ_PATH) if config.TF_FREQ_PATH.is_file() else freq
        # rock ~ deep Vs
        vs_rock = float(np.median(vs_s[-5:, :]))
        soil_nz = max(1, vs_s.shape[0] - 5)
        af_h = haskell_af_within(
            freq_arr,
            vs_s[:, int(center_rec)],
            zeta_s[:, int(center_rec)],
            dz=dz,
            vs_rock=vs_rock,
            soil_nz=soil_nz,
        )
        if len(af_h) != len(af_true):
            af_h = np.interp(np.arange(len(af_true)), np.linspace(0, len(af_true) - 1, len(af_h)), af_h)
        rels.append(_rel_l2(af_h, af_true))
        pears.append(_pearson(af_h, af_true))

    if not rels:
        print("[gifno-holdout] No samples evaluated.")
        return None
    summary = {"gifno_test_local_rel": _stats(rels), "gifno_test_local_pear": _stats(pears)}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "1d_floor_gifno_holdout.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "n", "mean", "median", "p10", "p90"])
        for k, s in summary.items():
            w.writerow([k, s["n"], s["mean"], s["median"], s["p10"], s["p90"]])
    print("=== GIFNO hold-out local Haskell floor ===")
    for k, s in summary.items():
        print(f"  {k}: mean={s['mean']:.4f} med={s['median']:.4f}")
    print(f"Wrote {path}")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5-dir", type=Path, default=config.RV_H5_DIR)
    p.add_argument("--out-dir", type=Path, default=config.RESULTS_SAVE_DIR / "1d_floor")
    p.add_argument("--max-seeds", type=int, default=None)
    p.add_argument("--skip-rv", action="store_true")
    p.add_argument("--gifno-limit", type=int, default=200)
    p.add_argument("--skip-gifno", action="store_true")
    args = p.parse_args()

    if not args.skip_rv:
        quantify_rv(args.h5_dir, args.out_dir, max_seeds=args.max_seeds)
    if not args.skip_gifno:
        quantify_gifno_holdout(args.out_dir, limit=args.gifno_limit)


if __name__ == "__main__":
    main()
