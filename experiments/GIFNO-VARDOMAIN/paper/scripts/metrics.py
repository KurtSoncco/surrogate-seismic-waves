"""Band metrics shared by the paper figures.

Bands match ``evaluate.BANDS``. Anderson misfit matches ``score_test_methods._scores``:
a Gaussian weight with σ = 1.5 Hz about f0, then the mean absolute log error.
Skill is ``1 - log_l1(model) / log_l1(column)``.
"""

from __future__ import annotations

import numpy as np

BANDS = {"low": (0.1, 0.4), "mid": (0.4, 2.5), "high": (2.5, 10.0)}
DZ_M = 1.0
ROCK_VS = 760.0


def band_mask(freq: np.ndarray, name: str) -> np.ndarray:
    lo, hi = BANDS[name]
    freq = np.asarray(freq, dtype=np.float64)
    return (freq >= lo) & (freq <= hi)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two curves. A constant curve scores 0."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom < 1e-12:
        return 0.0
    return float((a * b).sum() / denom)


def anderson(freq: np.ndarray, ref: np.ndarray, cand: np.ndarray, f0: float) -> float:
    """Anderson misfit: Gaussian-weighted mean |log cand − log ref|, σ = 1.5 Hz about f0."""
    freq = np.asarray(freq, dtype=np.float64)
    weight = np.exp(-0.5 * ((freq - float(f0)) / 1.5) ** 2)
    weight /= max(float(weight.sum()), 1e-12)
    r = np.clip(np.asarray(ref, dtype=np.float64), 1e-12, None)
    c = np.clip(np.asarray(cand, dtype=np.float64), 1e-12, None)
    return float(np.sum(weight * np.abs(np.log(c) - np.log(r))))


def log_l1(log_a: np.ndarray, log_b: np.ndarray, freq: np.ndarray | None = None, band: str | None = None) -> float:
    err = np.abs(np.asarray(log_a, dtype=np.float64) - np.asarray(log_b, dtype=np.float64))
    if band is not None:
        if freq is None:
            raise ValueError("freq is required when a band is set")
        err = err[..., band_mask(freq, band)]
    return float(np.mean(err))


def skill(model_l1: float, column_l1: float) -> float:
    return float(1.0 - model_l1 / max(float(column_l1), 1e-6))


FLOOR_LOG_L1 = 0.014


def above_floor_skill(column_l1, skill_values, floor: float = FLOOR_LOG_L1) -> dict:
    """Median skill on rows whose column log-L1 exceeds the consistency floor.

    Rows at or below the floor stay in the headline skill. This second number
    is the correction only where the column leftover is larger than that floor.
    """
    column_l1 = np.asarray(column_l1, dtype=np.float64)
    skill_values = np.asarray(skill_values, dtype=np.float64)
    if column_l1.shape != skill_values.shape:
        raise ValueError("column_l1 and skill_values must be aligned")
    mask = column_l1 > float(floor)
    n = int(mask.sum())
    return {
        "floor": float(floor),
        "n": int(column_l1.size),
        "n_above_floor": n,
        "skill": None if n == 0 else float(np.median(skill_values[mask])),
    }


def f_max_hz(vs_min: float, dz_m: float = DZ_M) -> float:
    """Trusted frequency Vs_min / (10 Δz)."""
    return float(vs_min) / (10.0 * float(dz_m))


def soil_rows(vs: np.ndarray, rock_vs: float = ROCK_VS) -> np.ndarray:
    """Rows whose median Vs is soil. The bedrock pad is left out so it cannot dilute a lateral std."""
    field = np.asarray(vs, dtype=np.float64)
    if field.ndim == 1:
        soil = field[field < rock_vs]
        return soil if soil.size else field
    med = np.median(field, axis=1)
    soil = field[med < rock_vs]
    return soil if soil.shape[0] else field
