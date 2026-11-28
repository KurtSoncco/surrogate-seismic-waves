#!/usr/bin/env python3
"""Bias that record length alone leaves in an outcrop label, on exact 1D records.

For a column with known Rayleigh coefficients the surface record is
irfft(H * X): H the exact outcrop transfer function (``outcrop_tf_rayleigh``),
X the spectrum of the seiskit Ricker. On a grid long enough for the response to
die out this is the steady record. Cutting it at T and running both through
``labels.tf_over_input`` with one fixed window gives bias(f, T) = log label_T -
log label_steady. The 2D excess on top of this comes from the pilots.

    python synthetic_truncation.py rows --manifest sections/manifold.csv --rows 0:40 --out OUT
    python synthetic_truncation.py iid --manifest .../transfer_function/manifest.csv \
        --h5-dir .../h5 --rows 0:200 --out OUT
    python synthetic_truncation.py tasks --mode rows --manifest M --chunk 20 --out OUT > tasks.txt

One ``OUT/<id>.npz`` per row or run; existing files are skipped, so a resubmitted
pylauncher array only does what is missing.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft, rfftfreq

CASE7A_DIR = Path(__file__).resolve().parents[1] / "case7a"
if str(CASE7A_DIR) not in sys.path:
    sys.path.insert(0, str(CASE7A_DIR))

from decay import MOTION_T_SHIFT_S, applied_zeta, decay_rate  # noqa: E402
from labels import LABEL_WINDOW_S, seiskit_input_acceleration, tf_over_input  # noqa: E402
from section_1d import outcrop_tf_rayleigh, rayleigh_coefficients  # noqa: E402

DT_S = 0.01
T_GRID_S = (15.0, 20.0, 30.0, 50.0, 80.0, 120.0, 200.0, 300.0, 500.0, 800.0, 1200.0)
N_TAU_STEADY = 6.0
# |log bias| below this is float32 noise in the label pipeline, not ringing.
BIAS_FLOOR = 1e-5
IID_RECORDER_COLS = 500 + 250 + 15 * np.arange(-10, 11)
IID_DAMPING_F2_HZ = 10.0


def _ttf():
    from dense_labels import _ttf as load

    return load()


def column_coefficients(rock_col: np.ndarray, zeta: tuple[float, float], f1: float, f2: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell (a0, a1): soil cells get the soil pair, rock cells the rock pair."""
    soil = rayleigh_coefficients(zeta[0], f1, f2)
    rock = rayleigh_coefficients(zeta[1], f1, f2)
    rock_col = np.asarray(rock_col, dtype=bool)
    return np.where(rock_col, rock[0], soil[0]), np.where(rock_col, rock[1], soil[1])


def steady_length(a0_soil: float, t_shift: float = MOTION_T_SHIFT_S, window_s: float = LABEL_WINDOW_S) -> tuple[float, bool]:
    """Grid length that lets the slowest soil mode decay N_TAU_STEADY times, capped by the window."""
    need = float(t_shift) + N_TAU_STEADY * 2.0 / max(float(a0_soil), 1e-12)
    return min(need, float(window_s)), need <= float(window_s)


def synthetic_records(
    vs_col: np.ndarray,
    a0: np.ndarray,
    a1: np.ndarray,
    *,
    dz: float,
    length_s: float,
    dt: float = DT_S,
    motion_freq: float = 3.0,
    t_shift: float = MOTION_T_SHIFT_S,
    ricker_is: str = "acceleration",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t, input acceleration, surface acceleration) of a 1D column, ``length_s`` long."""
    n = int(round(float(length_s) / float(dt)))
    n_fft = next_fast_len(2 * n)
    t = np.arange(n_fft) * float(dt)
    x = seiskit_input_acceleration(motion_freq, t_shift, t, ricker_is)
    freq = rfftfreq(n_fft, float(dt))
    h = np.ones(freq.size, dtype=np.complex128)
    h[1:] = outcrop_tf_rayleigh(freq[1:], vs_col, a0, a1, dz=dz)
    y = irfft(h * rfft(x), n_fft)
    return t[:n], x[:n], y[:n]


def truncation_bias(
    vs_col: np.ndarray,
    rock_col: np.ndarray,
    zeta: tuple[float, float],
    f1: float,
    f2: float,
    *,
    dz: float,
    t_grid: tuple[float, ...] | np.ndarray = T_GRID_S,
    dt: float = DT_S,
    motion_freq: float = 3.0,
    ricker_is: str = "acceleration",
    window_s: float = LABEL_WINDOW_S,
    ttf=None,
) -> dict:
    """bias (n_T, n_freq) and the diagnostics that say whether the reference is steady."""
    ttf = _ttf() if ttf is None else ttf
    a0, a1 = column_coefficients(rock_col, zeta, f1, f2)
    a0_soil = float(rayleigh_coefficients(zeta[0], f1, f2)[0])
    length, steady = steady_length(a0_soil, window_s=window_s)
    t, x, y = synthetic_records(vs_col, a0, a1, dz=dz, length_s=length, dt=dt, motion_freq=motion_freq, ricker_is=ricker_is)
    freq, ref = tf_over_input(y[:, None], x, dt, ttf, window_s=window_s)
    log_ref = np.log(np.maximum(ref[0], 1e-12))
    t_used = np.array([T for T in t_grid if T < length], dtype=np.float64)
    bias = np.empty((t_used.size, freq.size))
    for i, T in enumerate(t_used):
        n = int(round(T / dt))
        _, lab = tf_over_input(y[:n, None], x[:n], dt, ttf, window_s=window_s)
        bias[i] = np.log(np.maximum(lab[0], 1e-12)) - log_ref
    exact = np.abs(outcrop_tf_rayleigh(freq, vs_col, a0, a1, dz=dz))
    r = decay_rate(freq, a0_soil, float(rayleigh_coefficients(zeta[0], f1, f2)[1]))
    above = np.abs(bias) > BIAS_FLOOR
    log_env = np.where(above, np.log(np.maximum(np.abs(bias), BIAS_FLOOR)) + np.outer(t_used - MOTION_T_SHIFT_S, r), -np.inf)
    log_env = log_env.max(axis=0) if t_used.size else np.full(freq.size, -np.inf)
    tail = float(np.abs(y[int(0.9 * y.size):]).max() / max(np.abs(y).max(), 1e-30))
    return {
        "freq": freq.astype(np.float32),
        "t_grid_s": t_used.astype(np.float32),
        "bias": bias.astype(np.float32),
        "pipeline_minus_exact": (log_ref - np.log(np.maximum(exact, 1e-12))).astype(np.float32),
        "decay_rate": r.astype(np.float32),
        "log_amp_envelope": log_env.astype(np.float32),
        "steady_length_s": np.float32(length),
        "reference_steady": np.bool_(steady),
        "tail_ratio": np.float32(tail),
    }


def slowest_columns(vs: np.ndarray, rock: np.ndarray, dz: float) -> np.ndarray:
    """Indices of the column with the longest vertical soil travel time and of the center column."""
    slowness = np.where(rock, 0.0, dz / np.maximum(vs, 1e-9)).sum(axis=0)
    return np.unique([int(np.argmax(slowness)), vs.shape[1] // 2])


def _save(dest: Path, **arrays) -> None:
    tmp = dest.with_name(dest.stem + ".part.npz")
    np.savez(tmp, **arrays)
    tmp.replace(dest)


def run_rows(manifest: Path, lo: int, hi: int, out: Path, window_s: float) -> None:
    from section_generator import read_rows, section_field

    rows = read_rows(manifest)[lo:hi]
    out.mkdir(parents=True, exist_ok=True)
    ttf = _ttf()
    for row in rows:
        dest = out / f"{row['sample_id']}.npz"
        if dest.is_file():
            print(f"skip {dest.name}", flush=True)
            continue
        vs, _, rock = section_field(row)
        zeta = applied_zeta(row)
        f1, f2 = float(row["rayleigh_f1_hz"]), float(row["rayleigh_f2_hz"])
        cols = slowest_columns(vs, rock, 1.0)
        res = [
            truncation_bias(vs[:, c], rock[:, c], zeta, f1, f2, dz=1.0, motion_freq=float(row["motion_freq_hz"]),
                            window_s=window_s, ttf=ttf)
            for c in cols
        ]
        _save(dest, columns=cols, zeta=np.array(zeta), f1_f2=np.array([f1, f2]), freq=res[0]["freq"],
              t_grid_s=res[0]["t_grid_s"], **{k: np.stack([r[k] for r in res]) for k in res[0] if k not in ("freq", "t_grid_s")})
        print(f"wrote {dest.name} steady={bool(res[0]['reference_steady'])} L={float(res[0]['steady_length_s']):.0f}s", flush=True)


def run_iid(manifest: Path, h5_dir: Path, lo: int, hi: int, out: Path, window_s: float) -> None:
    """IID runs at their own record length, every recorder column, applied Q(Vs) zeta on the stored mesh."""
    import h5py
    import hdf5plugin  # noqa: F401
    from case7_crops import ROCK_VS_MPS, qvs_xi
    from relabel_outcrop import _h5_name

    rows = list(csv.DictReader(open(manifest)))[lo:hi]
    out.mkdir(parents=True, exist_ok=True)
    ttf = _ttf()
    for row in rows:
        path = h5_dir / _h5_name(row)
        dest = out / f"{path.stem}.npz"
        if dest.is_file() or not path.is_file():
            continue
        with h5py.File(path, "r") as f:
            vs = f["Vs_realization_2D"][:].astype(np.float64)
            time = f["recorders/accel/time"][:].astype(np.float64)
            motion_freq = float(f["grid"].attrs.get("motion_freq", 3.0))
            dz = float(f["grid"].attrs.get("dz", 1.0))
            f1 = float(f["params"].attrs["damping_freq_first"])
        rock = vs > ROCK_VS_MPS
        zeta = qvs_xi(vs, rock)
        dt = float(time[1] - time[0])
        T = float(time[-1] - time[0] + dt)
        res = [
            truncation_bias(vs[:, c], rock[:, c], zeta, f1, IID_DAMPING_F2_HZ, dz=dz, t_grid=(T,), dt=dt,
                            motion_freq=motion_freq, ricker_is="velocity", window_s=window_s, ttf=ttf)
            for c in IID_RECORDER_COLS
        ]
        _save(dest, zeta=np.array(zeta), f1_f2=np.array([f1, IID_DAMPING_F2_HZ]), record_s=np.float32(T),
              freq=res[0]["freq"], **{k: np.stack([r[k] for r in res]) for k in res[0] if k not in ("freq", "t_grid_s")})
        print(f"wrote {dest.name}", flush=True)


def write_tasks(args: argparse.Namespace) -> None:
    n = sum(1 for _ in csv.DictReader(open(args.manifest)))
    script = Path(__file__).resolve()
    extra = f" --h5-dir {args.h5_dir}" if args.mode == "iid" else ""
    for lo in range(0, n, args.chunk):
        hi = min(n, lo + args.chunk)
        print(f"python -u {script} {args.mode} --manifest {args.manifest}{extra} --rows {lo}:{hi} "
              f"--out {args.out} --window-s {args.window_s}")


def _span(text: str) -> tuple[int, int]:
    lo, hi = text.split(":")
    return int(lo), int(hi)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("rows", "iid"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--rows", type=_span, default=(0, 10**9), help="lo:hi slice of the manifest")
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--window-s", type=float, default=LABEL_WINDOW_S)
        if name == "iid":
            p.add_argument("--h5-dir", type=Path, required=True)
    p = sub.add_parser("tasks", help="pylauncher task file on stdout")
    p.add_argument("--mode", choices=("rows", "iid"), required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--h5-dir", type=Path)
    p.add_argument("--chunk", type=int, default=20)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--window-s", type=float, default=LABEL_WINDOW_S)
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if args.cmd == "tasks":
        write_tasks(args)
    elif args.cmd == "rows":
        run_rows(args.manifest, *args.rows, args.out, args.window_s)
    else:
        run_iid(args.manifest, args.h5_dir, *args.rows, args.out, args.window_s)


if __name__ == "__main__":
    main()
