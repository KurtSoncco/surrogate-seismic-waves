"""Confidence intervals that resample records, not stations.

Stations inside one record are one draw. Treasure Island is a single section,
so its interval is a moving block along x.
"""

from __future__ import annotations

import numpy as np

N_DRAW = 2000
BLOCK_M = 100.0


def _rng(rng: np.random.Generator | None) -> np.random.Generator:
    return np.random.default_rng() if rng is None else rng


def _groups(values: np.ndarray, record_ids: np.ndarray) -> list[np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    record_ids = np.asarray(record_ids)
    if values.shape[0] != record_ids.shape[0]:
        raise ValueError("values and record_ids must have the same length")
    if values.size == 0:
        raise ValueError("no records")
    _ids, inverse = np.unique(record_ids, return_inverse=True)
    return [values[inverse == i] for i in range(int(inverse.max()) + 1)]


def bootstrap_statistic(
    values,
    record_ids,
    statistic,
    *,
    n_draw: int = N_DRAW,
    rng: np.random.Generator | None = None,
    alpha: float = 0.05,
) -> dict:
    """Percentile interval of ``statistic`` on station values pooled by resampled records."""
    groups = _groups(values, record_ids)
    n = len(groups)
    rng = _rng(rng)
    draws = np.empty(n_draw, dtype=np.float64)
    for i in range(n_draw):
        pick = rng.integers(0, n, size=n)
        draws[i] = float(statistic(np.concatenate([groups[j] for j in pick])))
    low, high = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    point = float(statistic(np.concatenate(groups)))
    return {
        "point": point,
        "low": float(low),
        "high": float(high),
        "n_records": int(n),
        "n_draw": int(n_draw),
    }


def bootstrap_median(values, record_ids, **kwargs) -> dict:
    """Percentile interval of the median. Records are the resampled units."""
    return bootstrap_statistic(values, record_ids, np.median, **kwargs)


def bootstrap_cluster_statistic(
    values,
    cluster_ids,
    statistic,
    *,
    n_draw: int = N_DRAW,
    rng: np.random.Generator | None = None,
    alpha: float = 0.05,
) -> dict:
    """Percentile interval when the resampled unit is a site, not a record.

    A draw picks sites with replacement and keeps every record that belongs to
    a drawn site. ``statistic`` then sees that pooled set of record scores.
    """
    groups = _groups(np.asarray(values, dtype=np.float64), np.asarray(cluster_ids))
    n = len(groups)
    rng = _rng(rng)
    draws = np.empty(n_draw, dtype=np.float64)
    for i in range(n_draw):
        pick = rng.integers(0, n, size=n)
        draws[i] = float(statistic(np.concatenate([groups[j] for j in pick])))
    low, high = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    pooled = np.concatenate(groups)
    return {
        "point": float(statistic(pooled)),
        "low": float(low),
        "high": float(high),
        "n_clusters": int(n),
        "n_records": int(pooled.size),
        "n_draw": int(n_draw),
    }


def bootstrap_win_rate(
    score_a,
    score_b,
    record_ids,
    *,
    lower_is_better: bool = True,
    n_draw: int = N_DRAW,
    rng: np.random.Generator | None = None,
    alpha: float = 0.05,
) -> dict:
    """Fraction of records where ``score_a`` beats ``score_b``. Ties are not wins.

    Several stations of one record collapse to that record's median before the draw.
    """
    a = np.asarray(score_a, dtype=np.float64)
    b = np.asarray(score_b, dtype=np.float64)
    ids = np.asarray(record_ids)
    if a.shape != b.shape or a.shape != ids.shape:
        raise ValueError("score_a, score_b, and record_ids must be aligned")
    groups = _groups(np.arange(a.size, dtype=np.float64), ids)
    aa = np.array([float(np.median(a[np.asarray(g, dtype=int)])) for g in groups])
    bb = np.array([float(np.median(b[np.asarray(g, dtype=int)])) for g in groups])
    wins = aa < bb if lower_is_better else aa > bb
    n = int(wins.size)
    rng = _rng(rng)
    draws = np.empty(n_draw, dtype=np.float64)
    for i in range(n_draw):
        draws[i] = float(np.mean(wins[rng.integers(0, n, size=n)]))
    low, high = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "point": float(np.mean(wins)),
        "low": float(low),
        "high": float(high),
        "n_records": n,
        "n_draw": int(n_draw),
    }


def block_bootstrap_median(
    values,
    x_m,
    *,
    block_m: float = BLOCK_M,
    n_draw: int = N_DRAW,
    rng: np.random.Generator | None = None,
    alpha: float = 0.05,
) -> dict:
    """Moving-block bootstrap of the median along one section.

    Blocks are contiguous in x and about ``block_m`` metres long. The default
    is the 100 m consistency margin used in evaluation.
    """
    values = np.asarray(values, dtype=np.float64)
    x = np.asarray(x_m, dtype=np.float64)
    if values.shape != x.shape or values.size == 0:
        raise ValueError("values and x_m must be the same nonempty shape")
    order = np.argsort(x)
    values, x = values[order], x[order]
    if values.size == 1:
        step = 1
    else:
        dx = float(np.median(np.diff(x)))
        step = max(1, int(round(float(block_m) / (dx if dx > 0 else 1.0))))
    blocks = [values[s : s + step] for s in range(0, values.size, step)]
    n = len(blocks)
    rng = _rng(rng)
    draws = np.empty(n_draw, dtype=np.float64)
    for i in range(n_draw):
        pick = rng.integers(0, n, size=n)
        draws[i] = float(np.median(np.concatenate([blocks[j] for j in pick])))
    low, high = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "point": float(np.median(values)),
        "low": float(low),
        "high": float(high),
        "n_stations": int(values.size),
        "block_m": float(block_m),
        "n_blocks": int(n),
        "n_draw": int(n_draw),
    }
