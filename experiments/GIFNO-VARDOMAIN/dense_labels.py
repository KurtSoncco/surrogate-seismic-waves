#!/usr/bin/env python3
"""Outcrop labels at every STRIDE m of surface for whole-section runs.

One file per section, ``OUT/<sample_id>.npz``: ``x_m`` (n_nodes,), ``tf``
surface / (c * input) and ``tf1d`` (vertical-SH outcrop of the column under
each node), both (n_nodes, 1000), ``freq`` (1000,). Same conventions as
``experiments/case7a/section_windows.py``: acceleration Ricker, Rayleigh xi at
the run's own (damping_freq_first, damping_freq_second), and the input scale
``c`` read from an existing ``outcrop_calibration.json`` so these labels match
the 500 m windows the shipped models trained on.

    python dense_labels.py --h5-dir .../case7_sections/round0/h5 \
        --calibration .../case7_sections/round0/windows/outcrop_calibration.json \
        --out ../../data/vardomain/section_labels/round0 --workers 8

Sections whose npz already exists are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

CASE7A_DIR = Path(__file__).resolve().parents[1] / "case7a"
if str(CASE7A_DIR) not in sys.path:
    sys.path.insert(0, str(CASE7A_DIR))

MOTION_T_SHIFT = 0.5


def _ttf():
    import run_case7a

    root = os.environ.get("SEISKIT_ROOT")
    if root:
        pkg = Path(root) / "seiskit"
        run_case7a.SEISKIT_ROOT = pkg if (pkg / "ttf").is_dir() else Path(root)
    return run_case7a.load_ttf_batch_fast()


def label_section(
    path: Path,
    out_dir: Path,
    scale: float,
    stride_m: float,
    window_s: float | None = None,
    drop_pads: bool = False,
) -> str:
    import h5py
    import hdf5plugin  # noqa: F401
    from labels import seiskit_input_acceleration, tf_over_input
    from section_1d import RHO, rayleigh_xi_target, surface_over_outcrop

    with h5py.File(path, "r") as f:
        params = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f["params"].attrs.items()}
        sid = str(params.get("sample_id", path.stem))
        dest = out_dir / f"{sid}.npz"
        if dest.is_file():
            return f"skip {sid}"
        surface = f["recorders/accel/surface"]
        x_nodes = np.asarray(surface.attrs["x_m"], dtype=np.float64)
        grid = dict(f["grid"].attrs)
        dx = float(grid["dx"])
        step = max(1, int(round(stride_m / dx)))
        nodes = np.arange(0, x_nodes.size, step)
        pad_m = float(params.get("pad_m") or 0.0)
        lx_m = float(params.get("lx_interior_m") or params.get("lx_m") or 0.0)
        if drop_pads:
            if lx_m <= 0:
                raise ValueError(f"{sid}: drop_pads needs lx_interior_m or lx_m")
            x_all = x_nodes[nodes]
            keep = (x_all >= pad_m - 1e-3) & (x_all < pad_m + lx_m - 1e-3)
            nodes = nodes[keep]
            if nodes.size == 0:
                raise ValueError(f"{sid}: no interior nodes at stride {stride_m}")
        acc = surface[:, nodes].astype(np.float64)
        vs = f["Vs_realization_2D"][:].astype(np.float64)
        from applied_damping import applied_zeta

        zeta = applied_zeta(f)
        time = f["recorders/accel/time"][:].astype(np.float64)
    dt = float(time[1] - time[0])
    input_acc = seiskit_input_acceleration(
        float(grid["motion_freq"]), MOTION_T_SHIFT, time, str(params.get("ricker_is", "acceleration"))
    )
    freq, tf_raw = tf_over_input(acc, input_acc, dt, _ttf(), window_s=window_s)
    f1, f2 = float(params["damping_freq_first"]), float(params["damping_freq_second"])
    tf1d = np.empty_like(tf_raw)
    for j, col in enumerate(np.minimum(nodes, vs.shape[1] - 1)):
        xi = np.stack([rayleigh_xi_target(freq, z, f1, f2) for z in zeta[:, col]])
        tf1d[j] = surface_over_outcrop(freq, vs[:, col], dz=float(grid["dz"]), xi=xi, rho=RHO)
    x_m = x_nodes[nodes].astype(np.float64)
    if drop_pads:
        x_m = x_m - pad_m
    np.savez(
        dest,
        x_m=x_m.astype(np.float32),
        tf=(tf_raw / scale).astype(np.float32),
        tf1d=tf1d.astype(np.float32),
        freq=freq.astype(np.float32),
        split=np.array(str(params.get("split", "train"))),
        pad_m=np.float32(pad_m if drop_pads else 0.0),
        lx_m=np.float32(lx_m if drop_pads else x_nodes[-1]),
    )
    return f"wrote {dest.name} n={nodes.size}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, help="outcrop_calibration.json holding the input scale c")
    ap.add_argument("--scale", type=float, help="input scale c (overrides --calibration)")
    ap.add_argument("--stride", type=float, default=5.0, help="metres between labeled surface nodes")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--window-s", type=float, default=None,
                    help="fixed FFT window in seconds (labels.LABEL_WINDOW_S for vardomain); default: seiskit's per-record length")
    ap.add_argument("--drop-pads", action="store_true",
                    help="label only the interior (pad_m to pad_m+lx_interior_m) and store x from the interior origin")
    args = ap.parse_args()
    if args.scale is None and args.calibration is None:
        raise SystemExit("pass --calibration or --scale")
    scale = float(args.scale) if args.scale is not None else float(json.loads(args.calibration.read_text())["scale"])
    paths = sorted(p for p in args.h5_dir.glob("*.h5") if not p.name.startswith(("smoke", "timing")))
    if not paths:
        raise SystemExit(f"no section H5 in {args.h5_dir}")
    args.out.mkdir(parents=True, exist_ok=True)
    work = partial(
        label_section, out_dir=args.out, scale=scale, stride_m=args.stride,
        window_s=args.window_s, drop_pads=args.drop_pads,
    )
    with ProcessPoolExecutor(max_workers=min(args.workers, len(paths))) as pool:
        for msg in pool.map(work, paths):
            print(msg, flush=True)
    meta = {"scale": scale, "stride_m": args.stride, "h5_dir": str(args.h5_dir), "n_sections": len(paths),
            "window_s": args.window_s, "drop_pads": bool(args.drop_pads)}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
