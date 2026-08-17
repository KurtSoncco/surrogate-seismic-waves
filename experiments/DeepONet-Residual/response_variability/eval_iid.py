#!/usr/bin/env python3
"""Score GINO vs OpenSees 2-D and Haskell baselines on the nested IID test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import config  # noqa: E402
from data import ResidualDeepONetDataset  # noqa: E402
from eval_ood import _load_residual_model  # noqa: E402
from mix_ladder import iid_n1000_split  # noqa: E402
from train import _device, _forward, apply_norms  # noqa: E402

from response_variability.metrics import (  # noqa: E402
    FREQ_BANDS,
    band_rel_l2,
    method_vs_reference,
    peak_af,
    spatial_sigma_ln,
    theoretical_f0,
)
from response_variability.names import (  # noqa: E402
    COMPARE_METHODS,
    GINO,
    HASKELL_COLUMN,
    HASKELL_NOMINAL,
    OPENSEES,
    TF_KEYS,
)

OUT_DIR = config.RESULTS_DIR / "response_variability"
CENTRAL_REC = config.N_LATERAL // 2
IID_CACHE = config.CACHE_DIR / "n1000_seed42"


def _require_files(*paths: Path) -> None:
    missing = [p for p in paths if not p.is_file()]
    if missing:
        lines = "\n".join(f"  {p}" for p in missing)
        raise FileNotFoundError(
            "Missing required files for the IID Response_Variability check:\n"
            f"{lines}"
        )


def _meta_col(meta: dict[str, Any], key: str, idx: np.ndarray) -> np.ndarray:
    arr = np.asarray(meta[key])
    return arr[idx]


def load_iid_arrays(
    *,
    cache_dir: Path,
    test_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    meta = dict(np.load(cache_dir / "meta.npz", allow_pickle=True))
    tf2d_path = cache_dir / "tf2d.npy"
    if tf2d_path.is_file():
        tf_ops = np.asarray(np.load(tf2d_path, mmap_mode="r")[test_idx], dtype=np.float64)
    else:
        tf_all = np.load(config.TF_PER_SAMPLE_PATH, mmap_mode="r")
        sidx = np.load(cache_dir / "sample_indices.npy")[test_idx]
        tf_ops = np.asarray(tf_all[sidx], dtype=np.float64)
    freq_path = cache_dir / "freq.npy"
    freq = np.load(freq_path if freq_path.is_file() else config.TF_FREQ_PATH)
    vs1 = _meta_col(meta, "Vs1", test_idx).astype(float)
    H = _meta_col(meta, "H", test_idx).astype(float)
    return {
        "tf_opensees": tf_ops,
        "tf_haskell_nominal": np.asarray(
            np.load(cache_dir / "tf1d_nom.npy", mmap_mode="r")[test_idx], dtype=np.float64
        ),
        "tf_haskell_column": np.asarray(
            np.load(cache_dir / "tf1d_col.npy", mmap_mode="r")[test_idx], dtype=np.float64
        ),
        "freq": np.asarray(freq, dtype=float),
        "vs1": vs1,
        "H": H,
        "cov": _meta_col(meta, "CoV", test_idx).astype(float),
        "vs2": _meta_col(meta, "Vs2", test_idx).astype(float),
        "rf_seed": _meta_col(meta, "rf_seed", test_idx).astype(int),
        "sample_idx": _meta_col(meta, "sample_idx", test_idx).astype(int),
        "local_idx": np.asarray(test_idx, dtype=int),
        "f0": np.array([theoretical_f0(v, h) for v, h in zip(vs1, H)], dtype=float),
    }


def predict_gino(
    *,
    cache_dir: Path,
    test_idx: np.ndarray,
    ckpt_path: Path,
    batch_size: int,
) -> np.ndarray:
    """Return GINO |TF| reconstructions, shape (n_test, n_rec, n_freq)."""
    import torch

    device = _device()
    model, blob, stats, trunk_set = _load_residual_model(ckpt_path, device)
    serial = bool(blob.get("serial_tf1d", True))
    ds = ResidualDeepONetDataset(
        cache_dir,
        test_idx,
        target="R_nom",
        trunk_set=trunk_set,
        n_freq=config.N_FREQ_EVAL,
        serial_tf1d=serial,
    )
    apply_norms(ds, stats)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    n_rec = ds.n_rec
    n_freq = len(ds.f_idx)
    t_mean = stats["target_mean"].to(device)
    t_std = stats["target_std"].to(device)
    mode = blob.get("branch_mode", "single")
    hats: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="GINO iid test", leave=False):
            pred_n = _forward(
                model,
                batch["fields"].to(device),
                batch["stoch"].to(device),
                batch["trunk_y"].to(device),
                mode,
            )
            pred = pred_n * t_std + t_mean
            tf1d = batch["tf1d"].to(device)
            hats.append((tf1d + pred).cpu().numpy())
    stacked = np.concatenate(hats, axis=0)
    return stacked.reshape(len(ds), n_rec, n_freq).astype(np.float64)


def _as_central(tf_i: np.ndarray) -> np.ndarray:
    a = np.asarray(tf_i, dtype=np.float64)
    if a.ndim == 1:
        return a
    return a[..., CENTRAL_REC, :] if a.ndim == 3 else a[CENTRAL_REC]


def pack_method_tfs(pack: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Candidate methods present in ``pack`` (excludes OpenSees 2-D)."""
    out = {}
    for method, key in TF_KEYS.items():
        if method == OPENSEES:
            continue
        if key in pack:
            out[method] = pack[key]
    return out


def summarize_methods(pack: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    freq = pack["freq"]
    tf_ops = pack["tf_opensees"]
    method_tfs = pack_method_tfs(pack)
    n = tf_ops.shape[0]
    summary_rows: list[dict[str, Any]] = []
    peak_rows: list[dict[str, Any]] = []
    for i in range(n):
        af_ref = _as_central(tf_ops[i])
        shared = {
            "sample": i,
            "local_idx": int(pack["local_idx"][i]),
            "sample_idx": int(pack["sample_idx"][i]),
            "rf_seed": int(pack["rf_seed"][i]),
            "Vs1": float(pack["vs1"][i]),
            "H": float(pack["H"][i]),
            "CoV": float(pack["cov"][i]),
            "Vs2": float(pack["vs2"][i]),
            "f0": float(pack["f0"][i]),
            "reference": OPENSEES,
        }
        f_ops, a_ops = peak_af(freq, af_ref)
        peak_rows.append(
            {
                **shared,
                "method": OPENSEES,
                "f_peak": f_ops,
                "A_peak": a_ops,
                "delta_f_peak": 0.0,
                "delta_ln_A_peak": 0.0,
                "gof_af": 0.0,
                "rel_l2": 0.0,
                "pearson": 1.0,
                "sigma_ln_spatial_mean": float(np.mean(spatial_sigma_ln(tf_ops[i]))),
            }
        )
        for method, tf in method_tfs.items():
            cand = np.asarray(tf[i], dtype=np.float64)
            spatial = cand if cand.ndim == 2 else None
            mets = method_vs_reference(
                freq=freq,
                af_ref=af_ref,
                af_cand=_as_central(cand),
                af_ref_spatial=tf_ops[i] if spatial is not None else None,
                af_cand_spatial=spatial,
            )
            row = {**shared, "method": method, **mets}
            summary_rows.append(row)
            peak_rows.append(
                {
                    **shared,
                    "method": method,
                    "f_peak": mets["f_peak"],
                    "A_peak": mets["A_peak"],
                    "delta_f_peak": mets["delta_f_peak"],
                    "delta_ln_A_peak": mets["delta_ln_A_peak"],
                    "gof_af": mets["gof_af"],
                    "rel_l2": mets["rel_l2"],
                    "pearson": mets["pearson"],
                    "sigma_ln_spatial_mean": mets.get(
                        "sigma_ln_spatial_mean", float("nan")
                    ),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(peak_rows)


def band_misfit_table(pack: dict[str, np.ndarray]) -> pd.DataFrame:
    freq = pack["freq"]
    tf_ops = pack["tf_opensees"]
    method_tfs = pack_method_tfs(pack)
    rows: list[dict[str, Any]] = []
    for i in range(tf_ops.shape[0]):
        ref_full = np.asarray(tf_ops[i], dtype=np.float64)
        ref_c = _as_central(ref_full)
        for method, tf in method_tfs.items():
            cand = np.asarray(tf[i], dtype=np.float64)
            row: dict[str, Any] = {
                "sample": i,
                "local_idx": int(pack["local_idx"][i]),
                "sample_idx": int(pack["sample_idx"][i]),
                "method": method,
                "reference": OPENSEES,
                "Vs1": float(pack["vs1"][i]),
                "H": float(pack["H"][i]),
                "CoV": float(pack["cov"][i]),
                "Vs2": float(pack["vs2"][i]),
            }
            for band, (lo, hi) in FREQ_BANDS.items():
                if cand.ndim == 1:
                    row[f"rel_l2_{band}"] = band_rel_l2(
                        cand, ref_c, freq, lo=lo, hi=hi
                    )
                    row[f"rel_l2_{band}_central"] = row[f"rel_l2_{band}"]
                else:
                    row[f"rel_l2_{band}"] = band_rel_l2(
                        cand, ref_full, freq, lo=lo, hi=hi
                    )
                    row[f"rel_l2_{band}_central"] = band_rel_l2(
                        _as_central(cand), ref_c, freq, lo=lo, hi=hi
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_json(summary: pd.DataFrame, misfit: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"n_test": int(summary["sample"].nunique())}
    methods = [m for m in summary["method"].unique() if m != OPENSEES]
    for method in methods:
        sub = summary[summary["method"] == method]
        mis = misfit[misfit["method"] == method]
        rec = {
            "rel_l2_central_mean": float(sub["rel_l2"].mean()),
            "rel_l2_central_median": float(sub["rel_l2"].median()),
            "pearson_central_mean": float(sub["pearson"].mean()),
            "gof_af_mean": float(sub["gof_af"].mean()),
            "delta_f_peak_median": float(sub["delta_f_peak"].median()),
            "delta_ln_A_peak_median": float(sub["delta_ln_A_peak"].median()),
            "rel_l2_low_mean": float(mis["rel_l2_low"].mean()),
            "rel_l2_mid_mean": float(mis["rel_l2_mid"].mean()),
            "rel_l2_high_mean": float(mis["rel_l2_high"].mean()),
        }
        if "rel_l2_spatial" in sub.columns:
            rec["rel_l2_spatial_mean"] = float(sub["rel_l2_spatial"].mean())
        out[method] = rec
    return out


def run(
    *,
    checkpoint: Path,
    cache_dir: Path,
    out_dir: Path,
    batch_size: int,
    skip_predict: bool,
) -> dict[str, Path]:
    split = iid_n1000_split()
    test_idx = np.asarray(split["test"], dtype=int)
    pred_path = out_dir / "predictions.npz"
    out_dir.mkdir(parents=True, exist_ok=True)

    if skip_predict and pred_path.is_file():
        blob = np.load(pred_path, allow_pickle=True)
        pack = {k: blob[k] for k in blob.files}
    else:
        _require_files(
            checkpoint,
            cache_dir / "meta.npz",
            cache_dir / "tf1d_nom.npy",
            cache_dir / "tf1d_col.npy",
            cache_dir / "r_nom_signed.npy",
            cache_dir / "sample_indices.npy",
        )
        pack = load_iid_arrays(cache_dir=cache_dir, test_idx=test_idx)
        pack["tf_gino"] = predict_gino(
            cache_dir=cache_dir,
            test_idx=test_idx,
            ckpt_path=checkpoint,
            batch_size=batch_size,
        )
        np.savez_compressed(pred_path, **pack)

    summary, peaks = summarize_methods(pack)
    misfit = band_misfit_table(pack)
    summary_path = out_dir / "method_comparison_summary.csv"
    peaks_path = out_dir / "per_sample_peaks.csv"
    misfit_path = out_dir / "tf_band_misfit.csv"
    summary.to_csv(summary_path, index=False)
    peaks.to_csv(peaks_path, index=False)
    misfit.to_csv(misfit_path, index=False)
    agg = aggregate_json(summary, misfit)
    agg_path = out_dir / "aggregate.json"
    agg_path.write_text(json.dumps(agg, indent=2))
    print(json.dumps(agg, indent=2), flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {peaks_path}", flush=True)
    print(f"Wrote {misfit_path}", flush=True)
    print(f"Wrote {pred_path}", flush=True)
    return {
        "summary": summary_path,
        "peaks": peaks_path,
        "misfit": misfit_path,
        "predictions": pred_path,
        "aggregate": agg_path,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=config.DEFAULT_CHECKPOINT)
    p.add_argument("--cache-dir", type=Path, default=IID_CACHE)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument(
        "--skip-predict",
        action="store_true",
        help="Reuse predictions.npz if present (skip GINO inference).",
    )
    p.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    run(
        checkpoint=args.checkpoint,
        cache_dir=args.cache_dir,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        skip_predict=args.skip_predict,
    )
    if args.plot:
        from response_variability.plot_iid import plot_all

        plot_all(args.out_dir)


if __name__ == "__main__":
    main()
