"""Phase-2 pilot specification: 0.05–50 Hz, graded mesh, paired widths.

This does not launch a production OpenSees campaign. It checks the numerical
specification on a flat column and writes the paired-width job list. A flat
homogeneous column must keep |TF| at 1, which is the |g| < 0.01 floor once the
2D solve uses the same operator. The broadband pulse is accepted when its
Fourier amplitude stays within 20 dB across 0.05–50 Hz.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

F_MIN_HZ = 0.05
F_MAX_HZ = 50.0
WIDTHS_M = (500, 1000, 1500, 2000)
VS_MIN = 100.0
POINTS_PER_WAVE = 10
DT_S = 0.002
DURATION_S = 100.0
DEEP_CHECK_S = 200.0
PAD_M = 500


def graded_dz(z_m: np.ndarray, vs_surface: float = VS_MIN, vs_rock: float = 1500.0, depth_soil: float = 80.0) -> np.ndarray:
    """Element height (m). 0.25 m in the soft surface, 1 m once Vs would allow it."""
    z = np.asarray(z_m, dtype=np.float64)
    vs = np.where(z < depth_soil, vs_surface + (vs_rock - vs_surface) * (z / depth_soil), vs_rock)
    allow = vs / (POINTS_PER_WAVE * F_MAX_HZ)
    return np.clip(allow, 0.2, 1.0)


def mesh_ok(dz: np.ndarray, vs_at_element: np.ndarray, f_max: float = F_MAX_HZ) -> bool:
    return bool(np.all(dz <= vs_at_element / (POINTS_PER_WAVE * f_max) + 1e-9))


def ormsby_wavelet(t: np.ndarray, f1: float = 0.04, f2: float = 0.05, f3: float = 50.0, f4: float = 55.0) -> np.ndarray:
    """Zero-phase Ormsby pulse: a trapezoid spectrum, flat from f2 to f3.

    ``t`` is uniform. The pilot band 0.05–50 Hz sits on the flat top.
    """
    dt = float(t[1] - t[0])
    n = int(t.size)
    freq = np.fft.rfftfreq(n, dt)
    amp = np.zeros(freq.shape, dtype=np.float64)
    rise = (freq >= f1) & (freq < f2)
    flat = (freq >= f2) & (freq <= f3)
    fall = (freq > f3) & (freq <= f4)
    amp[rise] = (freq[rise] - f1) / (f2 - f1)
    amp[flat] = 1.0
    amp[fall] = (f4 - freq[fall]) / (f4 - f3)
    pulse = np.fft.fftshift(np.fft.irfft(amp, n))
    return pulse / np.max(np.abs(pulse))


def band_snr_db(pulse: np.ndarray, dt: float, f_min: float = F_MIN_HZ, f_max: float = F_MAX_HZ) -> float:
    """Smallest in-band Fourier amplitude relative to the in-band peak, in dB.

    A flat spectrum returns 0. The pilot asks for this drop to stay under 20 dB.
    """
    spec = np.abs(np.fft.rfft(pulse))
    freq = np.fft.rfftfreq(pulse.size, dt)
    band = (freq >= f_min) & (freq <= f_max)
    if not np.any(band):
        return 0.0
    amp = spec[band]
    return float(20.0 * np.log10(np.maximum(amp.min(), 1e-30) / amp.max()))


def homogeneous_outcrop_is_one(freq: np.ndarray, vs: float = 400.0, n: int = 40) -> np.ndarray:
    """|TF| of a uniform column. The fixed outcrop operator returns 1 when undamped."""
    import sys

    case7 = Path(__file__).resolve().parents[1] / "case7a"
    if str(case7) not in sys.path:
        sys.path.insert(0, str(case7))
    from section_1d import outcrop_tf_rayleigh

    column = np.full(n, vs)
    return np.abs(outcrop_tf_rayleigh(freq, column, 0.0, 0.0, dz=1.0))


def pilot_jobs(n_geologies: int = 8) -> list[dict]:
    """Eight geologies, each at four widths, identical (f1, f2) across the widths."""
    rows = []
    for geo in range(n_geologies):
        f1 = 0.4 + 0.15 * (geo % 4)
        damping = "explicit" if geo % 2 == 0 else "qvs"
        zeta = 0.005 + 0.005 * (geo % 5)
        depth = 80.0 + 20.0 * (geo % 9)
        for width in WIDTHS_M:
            rows.append({
                "geology": f"pilot_{geo:02d}",
                "width_m": int(width),
                "pad_m": PAD_M,
                "depth_m": depth,
                "f1_hz": round(f1, 4),
                "f2_hz": round(3.0 * f1, 4),
                "damping": damping,
                "zeta_soil": zeta,
                "duration_s": DURATION_S,
                "dt_s": DT_S,
                "f_min_hz": F_MIN_HZ,
                "f_max_hz": F_MAX_HZ,
            })
    return rows


def write_pilot(out: Path, n_geologies: int = 8) -> dict:
    """Write the job list and the local numerical checks. No OpenSees solve."""
    out.mkdir(parents=True, exist_ok=True)
    jobs = pilot_jobs(n_geologies)
    with (out / "jobs.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(jobs[0]))
        writer.writeheader()
        writer.writerows(jobs)
    z = np.linspace(0.0, 250.0, 251)
    dz = graded_dz(z)
    vs = np.where(z < 80.0, VS_MIN + (1500.0 - VS_MIN) * (z / 80.0), 1500.0)
    t = np.arange(0.0, 20.0, DT_S) - 10.0
    pulse = ormsby_wavelet(t)
    snr = band_snr_db(pulse, DT_S)
    freq = np.geomspace(F_MIN_HZ, F_MAX_HZ, 64)
    tf = homogeneous_outcrop_is_one(freq)
    report = {
        "n_jobs": len(jobs),
        "widths_m": list(WIDTHS_M),
        "mesh_ok": mesh_ok(dz, vs),
        "dz_surface_m": float(dz[0]),
        "dz_rock_m": float(dz[-1]),
        "wavelet_drop_db": snr,
        "wavelet_within_20db": bool(snr >= -20.0),
        "flat_abs_g_max": float(np.max(np.abs(np.log(np.maximum(tf, 1e-12))))),
        "deep_check_duration_s": DEEP_CHECK_S,
        "opensees": "not launched; jobs.csv is the paired-width campaign",
    }
    (out / "checks.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("checkpoints/vardomain/broadband_pilot"))
    ap.add_argument("--geologies", type=int, default=8)
    args = ap.parse_args()
    report = write_pilot(args.out, args.geologies)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
