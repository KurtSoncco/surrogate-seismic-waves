"""How long a section record must run before its surface / outcrop ratio is steady.

With C = a0 M + a1 K in a region, every mode at angular frequency w decays at
least as fast as r(w) = (a0 + a1 w^2) / 2; radiation through the Lysmer base and
the lateral boundaries and TRBDF2's own dissipation only add to it. A record of
length T therefore leaves at most A exp(-(T - t_arr) / tau) of the response at
frequency f still ringing, tau = 1 / r. Soil-trapped resonances are bounded by
the soil region's coefficients; energy in rock leaves through the base.

    python decay.py --manifest sections/manifold.csv            # table per row
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

CASE7A_DIR = Path(__file__).resolve().parents[1] / "case7a"
if str(CASE7A_DIR) not in sys.path:
    sys.path.insert(0, str(CASE7A_DIR))

from section_1d import rayleigh_coefficients  # noqa: E402

MOTION_T_SHIFT_S = 0.5
BAND_HZ = (0.1, 10.0)


def decay_rate(freq: np.ndarray, a0: float, a1: float) -> np.ndarray:
    """Guaranteed amplitude decay rate (1/s) of a mode at ``freq`` for C = a0 M + a1 K."""
    w = 2.0 * np.pi * np.asarray(freq, dtype=np.float64)
    return 0.5 * (float(a0) + float(a1) * w**2)


def applied_zeta(row: dict, pad_m: float | None = None) -> tuple[float, float]:
    """(soil, rock) zeta OpenSees is given for this row on the padded 1 m mesh."""
    family = str(row.get("damping_family", "explicit") or "explicit")
    if family == "explicit":
        return float(row["zeta_soil"]), float(row["zeta_rock"])
    if family != "qvs":
        raise ValueError(f"damping_family must be 'explicit' or 'qvs', got {family!r}")
    from case7_crops import qvs_xi
    from run_sections import PAD_M, pad_free_field
    from section_generator import section_field

    vs, _, rock = section_field(row)
    pad = PAD_M if pad_m is None else float(pad_m)
    vs = pad_free_field(vs, pad, 1.0)
    rock = pad_free_field(rock.astype(np.float64), pad, 1.0) > 0.5
    return qvs_xi(vs, rock)


def row_coefficients(row: dict, zeta: tuple[float, float] | None = None) -> dict[str, tuple[float, float]]:
    soil, rock = applied_zeta(row) if zeta is None else zeta
    f1, f2 = float(row["rayleigh_f1_hz"]), float(row["rayleigh_f2_hz"])
    return {"soil": rayleigh_coefficients(soil, f1, f2), "rock": rayleigh_coefficients(rock, f1, f2)}


def crossing_time(lx_m: float, pad_m: float, vs_min: float) -> float:
    """Time for the slowest wave to cross the whole solved mesh (interior plus both pads)."""
    return (float(lx_m) + 2.0 * float(pad_m)) / float(vs_min)


def t_needed(
    freq: np.ndarray,
    a0: float,
    a1: float,
    *,
    eps: float,
    t_cross: float = 0.0,
    amplitude: float | np.ndarray = 1.0,
    k_cross: float = 1.0,
    t_shift: float = MOTION_T_SHIFT_S,
) -> np.ndarray:
    """Record length (s) after which the ringing at ``freq`` is below ``eps`` of ``amplitude``."""
    r = decay_rate(freq, a0, a1)
    ratio = np.maximum(np.asarray(amplitude, dtype=np.float64) / float(eps), 1.0)
    return float(t_shift) + float(k_cross) * float(t_cross) + np.log(ratio) / r


def trbdf2_numerics(freq: np.ndarray, dt: float, gamma: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """(numerical damping ratio, relative period elongation) of TRBDF2 on an undamped mode.

    OpenSees ``TRBDF2`` is Bathe's composite scheme: trapezoidal rule to
    t + gamma dt, then three-point backward difference to t + dt (gamma = 1/2).
    """
    omega_dt = 2.0 * np.pi * np.asarray(freq, dtype=np.float64) * float(dt)
    z = 1j * omega_dt
    g = float(gamma)
    y_g = (1.0 + g * z / 2.0) / (1.0 - g * z / 2.0)
    amp = (y_g / (g * (2.0 - g)) - (1.0 - g) ** 2 / (g * (2.0 - g))) / (1.0 - (1.0 - g) / (2.0 - g) * z)
    log_r, phase = np.log(np.abs(amp)), np.angle(amp)
    zeta_num = -log_r / np.sqrt(log_r**2 + phase**2)
    return zeta_num, omega_dt / phase - 1.0


def max_dt(f_top: float, zeta_floor: float, frac: float = 0.1, dt_hi: float = 0.05) -> float:
    """Largest dt whose TRBDF2 damping at ``f_top`` stays below ``frac * zeta_floor``."""
    lo, hi = 1e-5, float(dt_hi)
    if trbdf2_numerics(np.array([f_top]), hi)[0][0] <= frac * zeta_floor:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if trbdf2_numerics(np.array([f_top]), mid)[0][0] <= frac * zeta_floor:
            lo = mid
        else:
            hi = mid
    return lo


def row_summary(row: dict, eps: float = 0.02, pad_m: float | None = None) -> dict:
    """Decay bound at the band edges and the record length the whole band needs on one mesh."""
    from run_sections import PAD_M
    from section_generator import section_field

    pad = PAD_M if pad_m is None else float(pad_m)
    zeta = applied_zeta(row, pad)
    coef = row_coefficients(row, zeta)
    vs, _, rock = section_field(row)
    vs_min = float(vs[~rock].min()) if (~rock).any() else float(vs.min())
    t_cross = crossing_time(float(row["lx_m"]), pad, vs_min)
    a0, a1 = coef["soil"]
    freq = np.geomspace(*BAND_HZ, 200)
    t_band = t_needed(freq, a0, a1, eps=eps, t_cross=t_cross)
    return {
        "sample_id": str(row.get("sample_id", "")),
        "zeta_soil": zeta[0],
        "zeta_rock": zeta[1],
        "a0_soil": a0,
        "a1_soil": a1,
        "tau_max_s": 2.0 / a0 if a0 > 0 else math.inf,
        "t_cross_s": t_cross,
        "t_needed_full_band_s": float(t_band.max()),
        "f_slowest_hz": float(freq[int(np.argmax(t_band))]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--eps", type=float, default=0.02)
    args = ap.parse_args()
    from section_generator import read_rows

    for row in read_rows(args.manifest):
        s = row_summary(row, args.eps)
        print(
            f"{s['sample_id']:>14s} zeta {s['zeta_soil']:.4f}/{s['zeta_rock']:.5f} "
            f"tau_max {s['tau_max_s']:7.1f} s  cross {s['t_cross_s']:5.1f} s  "
            f"T_full {s['t_needed_full_band_s']:7.1f} s at {s['f_slowest_hz']:.2f} Hz"
        )


if __name__ == "__main__":
    main()
