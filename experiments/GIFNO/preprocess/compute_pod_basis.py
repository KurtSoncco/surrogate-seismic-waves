#!/usr/bin/env python3
"""Compute per-recorder POD modes of training-split transfer functions."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import config  # noqa: E402


def load_manifest(manifest_path: Path) -> list[dict]:
    with open(manifest_path, newline="") as f:
        return list(csv.DictReader(f))


def train_indices(
    n: int,
    train_split: float,
    val_split: float,
    seed: int,
) -> np.ndarray:
    """Train indices matching data_loader.get_data_loaders exactly (torch split).

    Using the same torch.Generator permutation guarantees the POD basis is
    built only from the model's train split (no val/test leakage).
    """
    import torch

    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    n_train = int(n * train_split)
    return np.asarray(perm[:n_train], dtype=np.int64)


def compute_pod_modes(
    tf_array: np.ndarray,
    train_idx: np.ndarray,
    recorder_x: np.ndarray,
    n_modes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """SVD per recorder on centered train TFs -> (R,K,F) modes and (R,F) mean."""
    n_rec = len(recorder_x)
    n_freq = tf_array.shape[-1]
    modes = np.zeros((n_rec, n_modes, n_freq), dtype=np.float32)
    mean = np.zeros((n_rec, n_freq), dtype=np.float32)

    train_tf = tf_array[train_idx]
    for r in range(n_rec):
        curves = train_tf[:, r, :].astype(np.float64)
        mu = curves.mean(axis=0)
        mean[r] = mu.astype(np.float32)
        centered = curves - mu
        if centered.shape[0] < 2:
            modes[r, 0] = 1.0
            continue
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        k = min(n_modes, vt.shape[0])
        for j in range(k):
            mode = vt[j]
            norm = np.linalg.norm(mode)
            if norm > 1e-12:
                mode = mode / norm
            modes[r, j] = mode.astype(np.float32)
    return modes, mean


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute POD basis for GIFNO TFs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-modes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--train-split",
        type=float,
        default=config.TRAIN_SPLIT,
        help="Must match the model's data_loader train split (no leakage)",
    )
    parser.add_argument("--val-split", type=float, default=config.VAL_SPLIT)
    parser.add_argument(
        "--out-modes",
        type=Path,
        default=None,
        help="Explicit output path for pod_modes.npy (overrides config)",
    )
    parser.add_argument("--out-mean", type=Path, default=None)
    args = parser.parse_args()

    n_modes = args.n_modes or getattr(config, "POD_NUM_MODES", 32)
    manifest = load_manifest(config.MANIFEST_PATH)
    if args.limit is not None:
        manifest = manifest[: args.limit]

    tf_array = np.load(config.TF_PER_SAMPLE_PATH, mmap_mode="r")
    recorder_x = np.load(config.TF_RESULTS_DIR / "recorder_x_idx.npy")
    n = len(manifest)
    train_idx = train_indices(n, args.train_split, args.val_split, args.seed)

    modes, mean = compute_pod_modes(tf_array, train_idx, recorder_x, n_modes)

    out_modes = args.out_modes or getattr(
        config, "POD_MODES_PATH", config.TF_RESULTS_DIR / "pod_modes.npy"
    )
    out_mean = args.out_mean or getattr(
        config, "POD_MEAN_PATH", config.TF_RESULTS_DIR / "pod_mean.npy"
    )
    out_modes = Path(out_modes)
    out_mean = Path(out_mean)
    out_modes.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_modes, modes)
    np.save(out_mean, mean)
    print(f"Saved POD basis: modes {modes.shape} -> {out_modes}")
    print(f"Saved POD mean: {mean.shape} -> {out_mean}")


if __name__ == "__main__":
    main()
