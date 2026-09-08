"""Sobol covering and frequency-mask helpers for residual GINO probes.

4D Response_Variability cases match seiskit
``comparison/Response_Variability/sobol_base_cases.py`` (seed 42, fixed
``rH=10 m``, ``aHV=50``). 6D bounds match ``neural-operator/data/sobol.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import lognorm, norm, qmc

DEFAULT_CONFIDENCE = 0.99
DEFAULT_SAMPLER_SEED = 42
RH_FIXED = 10.0
AHV_FIXED = 50.0
BEDROCK_THICKNESS = 10.0
VS1_BOUNDS = (100.0, 360.0)
VS2_BOUNDS = (760.0, 1500.0)
BOUNDS_H = (15.0, 100.0)
BOUNDS_COV = (0.1, 0.3)
BOUNDS_RH = (10.0, 100.0)
BOUNDS_AHV = (10.0, 50.0)
PARAM_4D = ("Vs1", "H", "CoV", "Vs2")
PARAM_6D = ("Vs1", "H", "CoV", "rH", "aHV", "Vs2")
DEFAULT_RV_COUNT = 64


def lognormal_parameter(
    lower: float, upper: float, confidence: float = DEFAULT_CONFIDENCE
) -> tuple[float, float]:
    ln_lower, ln_upper = np.log(lower), np.log(upper)
    z_score = float(norm.ppf(1 - (1 - confidence) / 2))
    mu = (ln_lower + ln_upper) / 2
    sigma = (ln_upper - ln_lower) / (2 * z_score)
    return float(np.exp(mu)), float(sigma)


scale_Vs1, sigma_Vs1 = lognormal_parameter(*VS1_BOUNDS)
scale_Vs2, sigma_Vs2 = lognormal_parameter(*VS2_BOUNDS)
scale_aHV, sigma_aHV = lognormal_parameter(*BOUNDS_AHV)


@dataclass(frozen=True)
class SobolBaseCase:
    sobol_id: int
    vs1: float
    H: float
    cov: float
    vs2: float
    rH: float = RH_FIXED
    aHV: float = AHV_FIXED
    bedrock_thickness: float = BEDROCK_THICKNESS

    def as_4d(self) -> np.ndarray:
        return np.array([self.vs1, self.H, self.cov, self.vs2], dtype=float)

    def as_6d(self) -> np.ndarray:
        return np.array(
            [self.vs1, self.H, self.cov, self.rH, self.aHV, self.vs2], dtype=float
        )


def unit_to_physical_4d(unit_samples: np.ndarray) -> np.ndarray:
    raw = np.asarray(unit_samples, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 4:
        raise ValueError(f"Expected shape (n, 4), got {raw.shape}")
    phys = np.zeros_like(raw)
    phys[:, 0] = lognorm.ppf(raw[:, 0], s=sigma_Vs1, scale=scale_Vs1)
    phys[:, 1] = BOUNDS_H[0] + raw[:, 1] * (BOUNDS_H[1] - BOUNDS_H[0])
    phys[:, 2] = BOUNDS_COV[0] + raw[:, 2] * (BOUNDS_COV[1] - BOUNDS_COV[0])
    phys[:, 3] = lognorm.ppf(raw[:, 3], s=sigma_Vs2, scale=scale_Vs2)
    return phys


def physical_to_unit_6d(physical: np.ndarray) -> np.ndarray:
    p = np.asarray(physical, dtype=float)
    if p.ndim != 2 or p.shape[1] != 6:
        raise ValueError(f"Expected shape (n, 6), got {p.shape}")
    unit = np.zeros_like(p)
    unit[:, 0] = lognorm.cdf(p[:, 0], s=sigma_Vs1, scale=scale_Vs1)
    unit[:, 1] = (p[:, 1] - BOUNDS_H[0]) / (BOUNDS_H[1] - BOUNDS_H[0])
    unit[:, 2] = (p[:, 2] - BOUNDS_COV[0]) / (BOUNDS_COV[1] - BOUNDS_COV[0])
    unit[:, 3] = (p[:, 3] - BOUNDS_RH[0]) / (BOUNDS_RH[1] - BOUNDS_RH[0])
    unit[:, 4] = lognorm.cdf(p[:, 4], s=sigma_aHV, scale=scale_aHV)
    unit[:, 5] = lognorm.cdf(p[:, 5], s=sigma_Vs2, scale=scale_Vs2)
    return unit


def _bounds_mask_4d(physical: np.ndarray) -> np.ndarray:
    return (
        (physical[:, 0] >= VS1_BOUNDS[0])
        & (physical[:, 0] <= VS1_BOUNDS[1])
        & (physical[:, 1] >= BOUNDS_H[0])
        & (physical[:, 1] <= BOUNDS_H[1])
        & (physical[:, 2] >= BOUNDS_COV[0])
        & (physical[:, 2] <= BOUNDS_COV[1])
        & (physical[:, 3] >= VS2_BOUNDS[0])
        & (physical[:, 3] <= VS2_BOUNDS[1])
    )


def generate_rv_base_cases(
    target_count: int,
    *,
    sampler_seed: int = DEFAULT_SAMPLER_SEED,
) -> list[SobolBaseCase]:
    """Scrambled Sobol 4D cases; prefixes of larger ``target_count`` are nested."""
    if target_count <= 0:
        return []
    sampler = qmc.Sobol(d=4, scramble=True, seed=sampler_seed)
    # Fixed batch so prefixes of the 64-case campaign stay nested.
    batch = 64
    collected: list[SobolBaseCase] = []
    sobol_id = 0
    while len(collected) < target_count:
        physical = unit_to_physical_4d(sampler.random(batch))
        for row in physical[_bounds_mask_4d(physical)]:
            if len(collected) >= target_count:
                break
            collected.append(
                SobolBaseCase(
                    sobol_id=sobol_id,
                    vs1=float(row[0]),
                    H=float(row[1]),
                    cov=float(row[2]),
                    vs2=float(row[3]),
                )
            )
            sobol_id += 1
    return collected


def cases_matrix(cases: list[SobolBaseCase], *, kind: str = "6d") -> np.ndarray:
    rows = [c.as_6d() if kind == "6d" else c.as_4d() for c in cases]
    return np.asarray(rows, dtype=float)


def unique_rows(X: np.ndarray, *, decimals: int = 8) -> np.ndarray:
    r = np.round(np.asarray(X, dtype=float), decimals)
    if r.size == 0:
        return r.reshape(0, X.shape[1] if X.ndim == 2 else 0)
    return np.unique(r, axis=0)


def standardize(train: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(train, dtype=float)
    q = np.asarray(query, dtype=float)
    mu = t.mean(axis=0)
    sd = np.clip(t.std(axis=0), 1e-8, None)
    return (t - mu) / sd, (q - mu) / sd


def knn_distance(train: np.ndarray, query: np.ndarray, *, k: int = 1) -> np.ndarray:
    """L2 distance from each query row to its k-th nearest train row."""
    t = np.asarray(train, dtype=float)
    q = np.asarray(query, dtype=float)
    if q.size == 0:
        return np.zeros(0, dtype=float)
    if t.size == 0:
        return np.full(len(q), np.inf)
    d = np.linalg.norm(q[:, None, :] - t[None, :, :], axis=-1)
    d.sort(axis=1)
    kk = min(max(int(k), 1), d.shape[1]) - 1
    return d[:, kk]


def aabb_inside(
    train: np.ndarray, query: np.ndarray, *, atol: float = 0.0
) -> np.ndarray:
    t = np.asarray(train, dtype=float)
    q = np.asarray(query, dtype=float)
    lo = t.min(axis=0) - atol
    hi = t.max(axis=0) + atol
    return np.all((q >= lo) & (q <= hi), axis=1)


def fill_distance(train: np.ndarray, query: np.ndarray) -> float:
    """Mean 1-NN distance of query points to the train set (covering gap)."""
    d = knn_distance(train, query, k=1)
    finite = np.isfinite(d)
    if not np.any(finite):
        return float("inf")
    return float(np.mean(d[finite]))


def nested_permutation_prefixes(
    n_total: int, sizes: tuple[int, ...], *, seed: int = DEFAULT_SAMPLER_SEED
) -> dict[int, np.ndarray]:
    """Nested index prefixes of a permutation (space-filling stand-in without IDs)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(int(n_total))
    out: dict[int, np.ndarray] = {}
    for n in sizes:
        k = min(int(n), int(n_total))
        out[k] = perm[:k]
    return out


def freq_train_mask(freq: np.ndarray, n_train: int) -> np.ndarray:
    """Boolean mask of the log-spaced trunk queries used at train time."""
    from data import freq_screen_indices

    f = np.asarray(freq, dtype=float).ravel()
    idx = np.asarray(freq_screen_indices(f, int(n_train)), dtype=int)
    mask = np.zeros(len(f), dtype=bool)
    mask[idx] = True
    return mask


def f_over_f0(freq: np.ndarray, f0: np.ndarray) -> np.ndarray:
    """``(n_sample, n_freq)`` array of f / f0, NaN when f0 is not positive."""
    f = np.asarray(freq, dtype=float).ravel()
    f0 = np.asarray(f0, dtype=float).ravel()
    out = np.full((len(f0), len(f)), np.nan, dtype=float)
    good = np.isfinite(f0) & (f0 > 0)
    out[good] = f[None, :] / f0[good, None]
    return out
