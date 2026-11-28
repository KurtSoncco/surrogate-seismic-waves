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

# Same floor as experiments/GIFNO/losses.py so log|TF| matches the training loss.
LOG_TF_EPS = 1e-8


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


def curves_in_space(curves: np.ndarray, space: str) -> np.ndarray:
    """Map recorder curves into the space the SVD is built in."""
    if space == "linear":
        return curves
    if space == "log":
        return np.log(np.clip(np.abs(curves), LOG_TF_EPS, None))
    raise ValueError(f"POD space must be linear or log, got {space!r}")


def compute_shared_pod_modes(
    tf_array: np.ndarray,
    train_idx: np.ndarray,
    n_modes: int,
    space: str = "linear",
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One frequency POD for every x: (K,F) modes and (F,) mean.

    Pools every labeled station (or dense column) of the train split so the
    branch can query an arbitrary column with the same basis. Unlabeled
    union-grid zeros must not enter the SVD: in log space they are log(eps)
    and pull the mean by ~8 nats, so recon starts at 5e-4 against residual ~1.
    """
    n_freq = int(tf_array.shape[-1])
    train_tf = np.asarray(tf_array[train_idx], dtype=np.float64)
    if valid is not None:
        train_ok = np.asarray(valid[train_idx], dtype=bool)
        if train_ok.shape[:2] != train_tf.shape[:2]:
            raise ValueError(
                "collocation_valid shape "
                f"{train_ok.shape} does not match TF {train_tf.shape}"
            )
        curves = train_tf[train_ok]
    else:
        flat = train_tf.reshape(-1, n_freq)
        curves = flat[np.any(np.abs(flat) > LOG_TF_EPS, axis=1)]
    if curves.shape[0] < 1:
        raise RuntimeError("shared POD: no labeled columns in the train split")
    curves = curves_in_space(curves, space)
    mean = curves.mean(axis=0).astype(np.float32)
    modes = np.zeros((n_modes, n_freq), dtype=np.float32)
    centered = curves - mean
    if centered.shape[0] < 2:
        modes[0] = 1.0
        return modes, mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(n_modes, vt.shape[0])
    for j in range(k):
        mode = vt[j]
        norm = np.linalg.norm(mode)
        if norm > 1e-12:
            mode = mode / norm
        modes[j] = mode.astype(np.float32)
    return modes, mean


def compute_pod_modes(
    tf_array: np.ndarray,
    train_idx: np.ndarray,
    recorder_x: np.ndarray,
    n_modes: int,
    space: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """SVD per recorder on centered train TFs -> (R,K,F) modes and (R,F) mean.

    ``space="log"`` decomposes ``log(|TF|)`` clipped at ``LOG_TF_EPS``. The
    returned mean is in that space. Callers that want linear amplitude
    exponentiate ``mean + coeffs @ modes``.
    """
    n_rec = len(recorder_x)
    n_freq = tf_array.shape[-1]
    modes = np.zeros((n_rec, n_modes, n_freq), dtype=np.float32)
    mean = np.zeros((n_rec, n_freq), dtype=np.float32)

    train_tf = tf_array[train_idx]
    for r in range(n_rec):
        curves = curves_in_space(train_tf[:, r, :].astype(np.float64), space)
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
    parser.add_argument(
        "--space",
        choices=("linear", "log"),
        default="linear",
        help="SVD linear |TF| or log(|TF|). Log outputs stay in log space.",
    )
    parser.add_argument(
        "--layout",
        choices=("shared", "per_recorder"),
        default="per_recorder",
        help="shared: one (K,F) basis for every x. per_recorder: legacy (R,K,F).",
    )
    args = parser.parse_args()

    n_modes = args.n_modes or getattr(config, "POD_NUM_MODES", 32)
    manifest = load_manifest(config.MANIFEST_PATH)
    if args.limit is not None:
        manifest = manifest[: args.limit]

    tf_array = np.load(config.TF_PER_SAMPLE_PATH, mmap_mode="r")
    recorder_x = np.load(config.TF_RESULTS_DIR / "recorder_x_idx.npy")
    valid_path = config.TF_RESULTS_DIR / "collocation_valid.npy"
    valid = np.load(valid_path, mmap_mode="r") if valid_path.is_file() else None
    n = len(manifest)
    train_idx = train_indices(n, args.train_split, args.val_split, args.seed)

    if args.layout == "shared":
        modes, mean = compute_shared_pod_modes(
            tf_array, train_idx, n_modes, space=args.space, valid=valid
        )
    else:
        modes, mean = compute_pod_modes(
            tf_array, train_idx, recorder_x, n_modes, space=args.space
        )

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
    space_path = out_modes.parent / "pod_space.txt"
    space_path.write_text(args.space + "\n")
    print(f"Saved POD basis ({args.space}): modes {modes.shape} -> {out_modes}")
    print(f"Saved POD mean: {mean.shape} -> {out_mean}")


if __name__ == "__main__":
    main()
