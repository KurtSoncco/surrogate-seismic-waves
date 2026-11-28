#!/usr/bin/env python3
"""Shared log-POD reconstruction ceiling on a mix train/holdout pair.

Fits (K, F) modes on labeled train columns, projects holdout labeled columns,
reports trusted-band Pearson of log |TF| (residual labels are multiplied back
by TF_1D so the number is comparable to the holdout scorer and to 1D).

    python pod_ceiling_mix.py --train DIR/mix_pretrain --holdout DIR/mix_pretrain/holdout \
        --modes 32,64,96,128 --out-dir DIR/pod_ceiling
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

EPS = 1e-8
TRUSTED = (0.1, 2.5)


def _pod():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "GIFNO" / "preprocess" / "compute_pod_basis.py"
    spec = importlib.util.spec_from_file_location("compute_pod_basis_mix", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pearson_row(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / den) if den > 0 else float("nan")


def _valid_curves(tf: np.ndarray, valid: np.ndarray, idx: np.ndarray) -> np.ndarray:
    parts = []
    for i in idx:
        ok = np.asarray(valid[i], dtype=bool)
        if ok.any():
            parts.append(np.asarray(tf[i, ok], dtype=np.float64))
    if not parts:
        raise RuntimeError("no labeled columns")
    return np.concatenate(parts, axis=0)


def _fit_shared(curves: np.ndarray, k: int, space: str, pod) -> tuple[np.ndarray, np.ndarray]:
    work = pod.curves_in_space(curves, space)
    mean = work.mean(axis=0).astype(np.float64)
    centered = work - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    k_use = min(k, vt.shape[0])
    modes = np.zeros((k, work.shape[1]), dtype=np.float64)
    for j in range(k_use):
        mode = vt[j]
        norm = np.linalg.norm(mode)
        if norm > 1e-12:
            mode = mode / norm
        modes[j] = mode
    return modes, mean


def _project(tf_row: np.ndarray, ok: np.ndarray, modes: np.ndarray, mean: np.ndarray, space: str, pod):
    curves = np.asarray(tf_row[ok], dtype=np.float64)
    work = pod.curves_in_space(curves, space)
    coeffs = (work - mean) @ modes.T
    recon = mean + coeffs @ modes
    if space == "log":
        recon = np.exp(recon)
    return recon


def _sample_pearson(pred, true, freq, band) -> float:
    m = (freq >= band[0]) & (freq <= band[1])
    rs = [_pearson_row(np.log(np.maximum(pred[j, m], EPS)), np.log(np.maximum(true[j, m], EPS))) for j in range(len(pred))]
    return float(np.nanmean(rs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--holdout", type=Path, required=True)
    ap.add_argument("--modes", default="32,64,96,128")
    ap.add_argument("--max-train", type=int, default=2000, help="subsample train rows for the SVD")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pod = _pod()
    ks = [int(x) for x in args.modes.split(",") if x.strip()]
    meta = json.loads((args.train / "dataset.json").read_text())
    target = meta.get("target", "outcrop")
    tf_tr = np.load(args.train / "tf_per_sample.npy", mmap_mode="r")
    tf1d_tr = np.load(args.train / "tf1d_per_sample.npy", mmap_mode="r")
    valid_tr = np.load(args.train / "collocation_valid.npy", mmap_mode="r")
    tf_ho = np.load(args.holdout / "tf_per_sample.npy", mmap_mode="r")
    tf1d_ho = np.load(args.holdout / "tf1d_per_sample.npy", mmap_mode="r")
    valid_ho = np.load(args.holdout / "collocation_valid.npy", mmap_mode="r")
    freq = np.load(args.holdout / "freq.npy")
    rows = list(csv.DictReader(open(args.holdout / "manifest.csv")))

    rng = np.random.default_rng(args.seed)
    n_tr = tf_tr.shape[0]
    train_idx = np.arange(n_tr)
    if n_tr > args.max_train:
        train_idx = np.sort(rng.choice(n_tr, size=args.max_train, replace=False))

    # Fit POD in the label space (outcrop |TF| or residual ratio).
    curves = _valid_curves(tf_tr, valid_tr, train_idx)
    print(f"[pod] target={target} train_rows={len(train_idx)} labeled_curves={len(curves)} holdout={len(rows)}", flush=True)

    hold_idx = np.arange(len(rows))
    # 1D prior on holdout in |TF| space
    prior_p = []
    for i in hold_idx:
        ok = np.asarray(valid_ho[i], dtype=bool)
        true = np.asarray(tf_ho[i, ok]) * np.asarray(tf1d_ho[i, ok]) if target == "residual" else np.asarray(tf_ho[i, ok])
        prior = np.asarray(tf1d_ho[i, ok])
        prior_p.append(_sample_pearson(prior, true, freq, TRUSTED))
    prior_med = float(np.median(prior_p))
    print(f"[pod] 1D prior trusted Pearson median={prior_med:.3f}", flush=True)

    results = []
    for k in ks:
        print(f"[pod] K={k} SVD...", flush=True)
        modes, mean = _fit_shared(curves, k, "log", pod)
        pears = []
        for i in hold_idx:
            ok = np.asarray(valid_ho[i], dtype=bool)
            if not ok.any():
                continue
            recon = _project(tf_ho[i], ok, modes, mean, "log", pod)
            true = np.asarray(tf_ho[i, ok])
            if target == "residual":
                recon = recon * np.asarray(tf1d_ho[i, ok])
                true = true * np.asarray(tf1d_ho[i, ok])
            pears.append(_sample_pearson(recon, true, freq, TRUSTED))
        row = {
            "n_modes": k,
            "space": "log",
            "n_train_rows": int(len(train_idx)),
            "n_holdout": int(len(pears)),
            "test_pearson_median": float(np.median(pears)),
            "test_pearson_mean": float(np.mean(pears)),
            "tf1d_pearson_median": prior_med,
        }
        results.append(row)
        print(
            f"  K={k}: recon r_med={row['test_pearson_median']:.3f}  "
            f"mean={row['test_pearson_mean']:.3f}  1D={prior_med:.3f}",
            flush=True,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "train": str(args.train),
        "holdout": str(args.holdout),
        "target": target,
        "tf1d_pearson_median": prior_med,
        "by_modes": results,
        "interpretation": (
            "If K=64 recon Pearson is already ~0.95, rank-K is not the wall. "
            "If it sits ~0.7 and jumps at K=96, raise GIFNO_POD_NUM_MODES."
        ),
    }
    (args.out_dir / "pod_ceiling.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", args.out_dir / "pod_ceiling.json", flush=True)


if __name__ == "__main__":
    main()
