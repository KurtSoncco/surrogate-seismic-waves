"""POD basis of leftover R on the (recorder, frequency) grid.

Used when ``--pod-readout`` trains a coefficient head instead of a DeepONet
query product. The shipped GINO rebal FT checkpoint does not use this path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_EPS = 1e-12


def fit_residual_pod(
    residuals: np.ndarray, n_modes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-recorder SVD of leftover R.

    Parameters
    ----------
    residuals
        Array of shape ``(N, n_rec, n_freq)``.
    n_modes
        Number of modes ``K`` kept at each recorder.

    Returns
    -------
    modes, mean
        ``modes`` is ``(n_rec, K, n_freq)``; ``mean`` is ``(n_rec, n_freq)``.
    """
    r = np.asarray(residuals, dtype=np.float64)
    if r.ndim != 3:
        raise ValueError(f"residuals must be (N, n_rec, n_freq), got {r.shape}")
    n, n_rec, n_freq = r.shape
    k = max(1, min(int(n_modes), n, n_freq))
    mean = r.mean(axis=0)
    modes = np.zeros((n_rec, k, n_freq), dtype=np.float32)
    for rec in range(n_rec):
        x = r[:, rec, :] - mean[rec]
        _u, _s, vt = np.linalg.svd(x, full_matrices=False)
        modes[rec] = np.asarray(vt[:k], dtype=np.float32)
    return modes, np.asarray(mean, dtype=np.float32)


def fit_pod_from_caches(
    parts: list[tuple[Any, Path, np.ndarray]],
    n_modes: int,
    *,
    log_residual: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack ``r_nom_signed`` from mix-ladder train parts and fit POD."""
    stacks: list[np.ndarray] = []
    for _name, cache_dir, idx in parts:
        cache = Path(cache_dir)
        idx = np.asarray(idx, dtype=int)
        r = np.asarray(np.load(cache / "r_nom_signed.npy", mmap_mode="r")[idx])
        if log_residual:
            tf1d = np.asarray(np.load(cache / "tf1d_nom.npy", mmap_mode="r")[idx])
            tf2d = tf1d + r
            r = np.log(np.clip(tf2d, _EPS, None)) - np.log(np.clip(tf1d, _EPS, None))
        stacks.append(np.asarray(r, dtype=np.float64))
    if not stacks:
        raise ValueError("no cache parts to fit leftover POD")
    return fit_residual_pod(np.concatenate(stacks, axis=0), n_modes=n_modes)
