"""Numerical learning-curve rules for the audit protocol.

``E(n) = c + a n^(-b)``. The labels are assigned before any curve is fit.
An IID curve does not classify dipping, three-layer, or realistic sections.
"""

from __future__ import annotations

import numpy as np

DELTA_LOG_L1 = 0.02
FLOOR_FRACTION = 0.10


def power_law(n: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return c + a * np.power(np.asarray(n, dtype=np.float64), -b)


def fit_power_law(n: np.ndarray, err: np.ndarray) -> np.ndarray:
    from scipy.optimize import curve_fit

    size = np.asarray(n, dtype=np.float64)
    error = np.asarray(err, dtype=np.float64)
    if size.size < 3:
        raise ValueError("a power law needs at least three subset sizes")
    floor = max(float(np.min(error)), 0.0)
    amplitude = max(float(np.max(error) - floor), 1e-6)
    coefficients, _ = curve_fit(
        power_law, size, error, p0=(amplitude, 0.5, floor),
        bounds=([0.0, 0.0, 0.0], [np.inf, 5.0, np.inf]), maxfev=20000,
    )
    return np.asarray(coefficients, dtype=np.float64)


def projected_gain(coefficients: np.ndarray, n_max: float) -> float:
    """Error reduction expected from doubling the largest observed subset."""
    return float(power_law(n_max, *coefficients) - power_law(2.0 * n_max, *coefficients))


def _bootstrap_index(n_rows: int, groups: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
    if groups is None:
        return rng.integers(0, n_rows, n_rows)
    labels = np.asarray(groups)
    chosen = rng.choice(np.unique(labels), size=np.unique(labels).size, replace=True)
    return np.concatenate([np.flatnonzero(labels == group) for group in chosen])


def classify_learning_curve(
    n: np.ndarray, err: np.ndarray, *, e_floor: float, e_1d: float,
    groups: np.ndarray | None = None, delta: float = DELTA_LOG_L1,
    floor_fraction: float = FLOOR_FRACTION, n_boot: int = 400, seed: int = 0,
) -> dict:
    """Return one of still_falling, flat_near_floor, flat_above_floor, or inconclusive."""
    size = np.asarray(n, dtype=np.float64)
    error = np.asarray(err, dtype=np.float64)
    coefficients = fit_power_law(size, error)
    n_max = float(np.max(size))
    gain = projected_gain(coefficients, n_max)
    fitted = float(power_law(n_max, *coefficients))
    rng = np.random.default_rng(int(seed))
    gains = []
    for _ in range(int(n_boot)):
        index = _bootstrap_index(error.size, None if groups is None else np.asarray(groups), rng)
        try:
            gains.append(projected_gain(fit_power_law(size[index], error[index]), n_max))
        except (ValueError, RuntimeError):
            continue
    if len(gains) < max(20, int(n_boot) // 5):
        low = high = float("nan")
    else:
        low, high = (float(v) for v in np.quantile(gains, [0.025, 0.975]))
    floor_gap = fitted - float(e_floor)
    near_floor = bool(np.isfinite(floor_gap) and floor_gap <= float(floor_fraction) * float(e_1d))
    if np.isfinite(low) and gain > float(delta) and low > 0.0:
        label = "still_falling"
    elif np.isfinite(high) and high <= float(delta) and near_floor:
        label = "flat_near_floor"
    elif np.isfinite(high) and high <= float(delta):
        label = "flat_above_floor"
    else:
        label = "inconclusive"
    return {
        "label": label,
        "coefficients": [float(v) for v in coefficients],
        "projected_gain": gain,
        "gain_interval": [low, high],
        "fitted_at_n": fitted,
        "floor_gap": floor_gap,
        "near_floor": near_floor,
        "delta": float(delta),
        "scope": "one family; an IID curve does not classify another family",
    }


def length_scaling_gate(errors: np.ndarray, measured_floor: float, placeholder: float = 0.0136) -> dict:
    """Pass only when every direct FE pair is inside the identity tolerance."""
    values = np.asarray(errors, dtype=np.float64)
    tolerance = max(float(measured_floor), float(placeholder))
    passes = bool(values.size >= 3 and np.all(values <= tolerance))
    return {"passes": passes, "tolerance": tolerance, "n_pairs": int(values.size)}
