#!/usr/bin/env python3
"""
Oracle POD reconstruction ceiling on the in-distribution GIFNO hold-out.

Projects test-split TFs onto the *train-split* POD basis at K ∈ {32,48,64,96}
and reports recon rel-L2 / Pearson. If K=64 recon ≪ model test_rel_l2 (~0.30),
the readout is not the wall; if recon ≈ 0.25–0.30, the POD head is the bottleneck.

Example:
  export GIFNO_DATA_ROOT="/mnt/box/GIG Lab - UC Berkeley/Projects/Neural Operator/data"
  cd experiments/GIFNO-FDO-XT-LOGLO-POD
  uv run python pod_recon_ceiling.py
  uv run python pod_recon_ceiling.py --limit 2000   # match screen subset
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config

config.setup_import_paths()

import importlib.util

_POD_SCRIPT = (
    Path(__file__).resolve().parents[1] / "GIFNO" / "preprocess" / "compute_pod_basis.py"
)
_spec = importlib.util.spec_from_file_location("compute_pod_basis", _POD_SCRIPT)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load {_POD_SCRIPT}")
_pod_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pod_mod)
compute_pod_modes = _pod_mod.compute_pod_modes

_EPS = 1e-12
DEFAULT_OUT = (
    Path.home()
    / "surrogate-seismic-waves"
    / "checkpoints"
    / "pod_recon_ceiling"
    / "results"
)


def _rel_l2(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + _EPS))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def reconstruct(
    tf: np.ndarray, modes: np.ndarray, mean: np.ndarray
) -> np.ndarray:
    """tf (R,F), modes (R,K,F), mean (R,F) -> recon (R,F)."""
    centered = tf - mean
    # coeffs[r,k] = <centered[r], mode[r,k]>
    coeffs = np.einsum("rf,rkf->rk", centered, modes)
    recon = mean + np.einsum("rk,rkf->rf", coeffs, modes)
    return recon.astype(np.float32)


def eval_modes(
    tf_array: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    recorder_x: np.ndarray,
    n_modes: int,
) -> dict:
    modes, mean = compute_pod_modes(tf_array, train_idx, recorder_x, n_modes)
    # energy: fraction of train variance captured
    train_tf = tf_array[train_idx]
    energy_fracs = []
    for r in range(len(recorder_x)):
        curves = train_tf[:, r, :].astype(np.float64)
        mu = curves.mean(axis=0)
        centered = curves - mu
        total = float(np.sum(centered**2))
        if total < 1e-18:
            energy_fracs.append(1.0)
            continue
        # project
        coeffs = centered @ modes[r].T  # (N, K)
        recon = coeffs @ modes[r]
        captured = float(np.sum(recon**2))
        energy_fracs.append(captured / total)

    rels, pears = [], []
    for i in test_idx:
        recon = reconstruct(tf_array[i], modes, mean)
        true = tf_array[i]
        # per-recorder then mean (matches capability style)
        r_rel = [_rel_l2(recon[r], true[r]) for r in range(true.shape[0])]
        r_pear = [_pearson(recon[r], true[r]) for r in range(true.shape[0])]
        rels.append(float(np.mean(r_rel)))
        pears.append(float(np.nanmean(r_pear)))

    return {
        "n_modes": n_modes,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "train_energy_frac_mean": float(np.mean(energy_fracs)),
        "test_rel_l2_mean": float(np.mean(rels)),
        "test_rel_l2_median": float(np.median(rels)),
        "test_rel_l2_p90": float(np.percentile(rels, 90)),
        "test_pearson_mean": float(np.mean(pears)),
        "test_pearson_median": float(np.median(pears)),
        "per_sample_rel_l2": rels,
    }


def split_indices(
    n: int, train_split: float, val_split: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match data_loader torch split exactly."""
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    train_idx = np.asarray(perm[:n_train], dtype=np.int64)
    val_idx = np.asarray(perm[n_train : n_train + n_val], dtype=np.int64)
    test_idx = np.asarray(perm[n_train + n_val :], dtype=np.int64)
    return train_idx, val_idx, test_idx


def main() -> int:
    parser = argparse.ArgumentParser(description="POD reconstruction ceiling")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--modes",
        type=str,
        default="32,48,64,96",
        help="Comma-separated POD mode counts",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    mode_list = [int(x) for x in args.modes.split(",") if x.strip()]
    manifest_path = config.MANIFEST_PATH
    tf_path = config.TF_PER_SAMPLE_PATH
    if not tf_path.is_file():
        print(f"ERROR: missing {tf_path}", file=sys.stderr)
        return 1

    with manifest_path.open(newline="", encoding="utf-8") as f:
        n_manifest = sum(1 for _ in csv.DictReader(f))
    tf_array = np.load(tf_path, mmap_mode="r")
    n = min(n_manifest, tf_array.shape[0])
    if args.limit is not None:
        n = min(n, args.limit)
    print(f"[pod] n={n} tf={tf_path} modes={mode_list}")

    recorder_x = np.load(config.TF_RESULTS_DIR / "recorder_x_idx.npy")
    train_idx, val_idx, test_idx = split_indices(
        n, config.TRAIN_SPLIT, config.VAL_SPLIT, args.seed
    )
    # Restrict arrays via index lists into first n samples
    # compute_pod_modes indexes into tf_array directly — ensure indices < n
    assert train_idx.max() < n and test_idx.max() < n

    results = []
    for k in mode_list:
        print(f"[pod] Evaluating K={k} ...")
        res = eval_modes(tf_array, train_idx, test_idx, recorder_x, k)
        results.append({kk: vv for kk, vv in res.items() if kk != "per_sample_rel_l2"})
        print(
            f"  K={k}: recon rel_l2={res['test_rel_l2_mean']:.4f}  "
            f"pearson={res['test_pearson_mean']:.4f}  "
            f"train_energy={res['train_energy_frac_mean']:.4f}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "n": n,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "seed": args.seed,
        "reference_model_test_rel_l2": 0.30,
        "interpretation": (
            "If oracle rel_l2 << 0.30, encoder/2D coupling is the wall. "
            "If oracle rel_l2 ≈ 0.25–0.30, POD readout capacity is the wall."
        ),
        "by_modes": results,
    }
    (args.out_dir / "pod_recon_ceiling.json").write_text(json.dumps(out, indent=2))

    ks = [r["n_modes"] for r in results]
    rels = [r["test_rel_l2_mean"] for r in results]
    pears = [r["test_pearson_mean"] for r in results]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ks, rels, "o-", label="oracle recon rel_l2")
    ax.axhline(0.30, color="C1", ls="--", label="model test_rel_l2 ≈ 0.30")
    ax.set_xlabel("POD modes K")
    ax.set_ylabel("Test mean rel L2")
    ax.set_title(f"POD reconstruction ceiling (n={n})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "pod_recon_ceiling.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ks, pears, "s-", color="C2")
    ax.set_xlabel("POD modes K")
    ax.set_ylabel("Test mean Pearson")
    ax.set_title(f"POD recon Pearson (n={n})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "pod_recon_pearson.png", dpi=140)
    plt.close(fig)

    print(f"[pod] Wrote {args.out_dir / 'pod_recon_ceiling.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
