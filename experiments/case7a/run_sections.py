#!/usr/bin/env python3
"""OpenSees runs for whole-section rows from section_generator.py.

One row is one 2000 m (training) or 4000 m (validation) interior, 150 m deep,
1 m cells. Each side then repeats the edge column for PAD_M (500 m), the same
``_extend_profile`` free-field pad as the IID basin runs, so ASDA sees a 1-D
column far from the feature. Linear algorithm, UmfPack by default
(BASIN_SOLVER overrides). Set SECTION_PAD_M=0 to skip the pad.

Recorders, x acceleration:
  * every surface node (x = 0, hx, ..., Lx) -> ``recorders/accel/surface``
  * columns every 60 m at 10 m depth steps, 0-150 m -> ``recorders/accel/columns``
    (n_time, n_depth, n_col), for the optional physics-informed column loss.

    python run_sections.py --manifest sections/round0.csv --index 3 --out-dir DATA
    python run_sections.py --manifest sections/round0.csv --smoke --out-dir /tmp/sec
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parent
SEISKIT_ROOT = Path(os.environ.get("SEISKIT_ROOT", "/home/kurt-/seiskit"))
for p in (CASE_DIR, SEISKIT_ROOT):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_basin_ensemble import _limit_blas_threads, _preload_opensees_libs  # noqa: E402
from section_generator import DEPTH_M, DZ_M, read_rows, section_field  # noqa: E402

DT_S = 0.01
DURATION_S = 15.0
MOTION_T_SHIFT_S = 0.5
COLUMN_SPACING_M = 60.0
COLUMN_DEPTHS_M = tuple(float(d) for d in range(0, 151, 10))
PAD_M = float(os.environ.get("SECTION_PAD_M", "500"))


def pad_free_field(arr: np.ndarray, pad_m: float, dx_m: float) -> np.ndarray:
    """Repeat the first and last columns ``pad_m`` metres each side."""
    n = int(round(float(pad_m) / float(dx_m)))
    if n <= 0:
        return np.asarray(arr)
    arr = np.asarray(arr)
    return np.hstack([np.repeat(arr[:, :1], n, axis=1), arr, np.repeat(arr[:, -1:], n, axis=1)])


def scored_surface_x(interior_m: float, pad_m: float, dx_m: float) -> np.ndarray:
    """Surface-node x (m) kept after the free-field pads are removed.

    ``pad_free_field`` repeats ``round(pad_m / dx)`` element columns on each
    side. Surface nodes lie at ``i * dx`` from the left edge of that padded
    mesh. The kept indices are ``[n_pad, n_pad + n_interior)``, the same span
    ``run_case7_crops.inner_surface`` writes out. Every kept station satisfies
    ``pad_m <= x < pad_m + interior_m`` and none lie inside a pad.
    """
    dx = float(dx_m)
    n_interior = int(round(float(interior_m) / dx))
    if n_interior < 1:
        raise ValueError(f"interior_m={interior_m}")
    padded = pad_free_field(np.zeros((1, n_interior), dtype=np.float64), float(pad_m), dx)
    n_pad_total = int(padded.shape[1] - n_interior)
    if n_pad_total % 2:
        raise ValueError("left and right pads differ; scored stations are undefined")
    n_pad = n_pad_total // 2
    return (n_pad + np.arange(n_interior, dtype=np.float64)) * dx


def coarsen_fields(
    vs: np.ndarray, zeta: np.ndarray, rock: np.ndarray, k: int, mapping: str = "harmonic"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map 1 m cells onto k x k element blocks.

    ``harmonic``: shear modulus harmonic-averaged down each block column (layers
    in series under vertical SH), then arithmetic-averaged across the block
    (columns in parallel); constant density, so Vs = sqrt(G / rho). Zeta is the
    block mean and rock the block majority. ``sample`` keeps the centre cell.
    """
    if k == 1:
        return vs, zeta, rock
    nz, nx = vs.shape
    if nz % k or nx % k:
        raise ValueError(f"{nz} x {nx} cells do not tile into {k} x {k} blocks")
    if mapping == "sample":
        return tuple(a[k // 2 :: k, k // 2 :: k] for a in (vs, zeta, rock))
    if mapping != "harmonic":
        raise ValueError(f"mapping must be 'harmonic' or 'sample', got {mapping!r}")

    def blocks(a: np.ndarray) -> np.ndarray:
        return np.asarray(a, dtype=np.float64).reshape(nz // k, k, nx // k, k)

    g_series = 1.0 / blocks(1.0 / np.asarray(vs, dtype=np.float64) ** 2).mean(axis=1)
    vs_c = np.sqrt(g_series.mean(axis=2))
    return vs_c, blocks(zeta).mean(axis=(1, 3)), blocks(rock).mean(axis=(1, 3)) > 0.5


def _analysis_config(row: dict, *, ly_m: float, lx_m: float, hx_m: float, duration_s: float, dt_s: float):
    import inspect

    from seiskit.config import AnalysisConfig

    depths = [d for d in COLUMN_DEPTHS_M if d <= ly_m]
    kwargs = dict(
        Ly=ly_m,
        Lx=lx_m,
        hx=hx_m,
        dt=dt_s,
        duration=duration_s,
        motion_freq=float(row["motion_freq_hz"]),
        motion_t_shift=MOTION_T_SHIFT_S,
        damping_freqs=(float(row["rayleigh_f1_hz"]), float(row["rayleigh_f2_hz"])),
        damping_zeta=float(row["zeta_soil"]),
        damping_method="global_avg",
        xi_soil_base=float(row["zeta_soil"]),
        xi_rock_base=float(row["zeta_rock"]),
        boundary_condition_type="2D",
        record_center_nodes=False,
        center_node_y_positions=[ly_m - d for d in depths],
        record_lateral_span_at_center_depths=(int((lx_m / 2) // COLUMN_SPACING_M), COLUMN_SPACING_M),
        record_all_surface_nodes=True,
        element_type="4node",
        solver_type=os.environ.get("BASIN_SOLVER", "UmfPack"),
        dynamic_algorithm="Linear",
        dynamic_tolerance=1.0e-3,
        max_dynamic_iter=25,
        max_time_per_batch=40.0 * 3600.0,
    )
    allowed = inspect.signature(AnalysisConfig.__init__).parameters
    dropped = sorted(set(kwargs) - set(allowed))
    if dropped:
        raise RuntimeError(f"installed seiskit does not accept {dropped}; update seiskit before solving")
    return AnalysisConfig(**kwargs)


def _load(path: Path) -> np.ndarray:
    arr = np.loadtxt(path)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def _load_recorders(rec_dir: Path, ly_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    surf = _load(rec_dir / "surface_nodes_dof1_accel.txt")
    files = sorted(rec_dir.glob("row_y*_dof1_accel.txt"), key=lambda p: -float(p.name.split("_dof")[0][5:]))
    if not files:
        raise RuntimeError(f"no column rows in {rec_dir}")
    rows = [_load(f) for f in files]
    n = min([surf.shape[0]] + [r.shape[0] for r in rows])
    depth = np.array([ly_m - float(f.name.split("_dof")[0][5:]) for f in files])
    cols = np.stack([r[:n, 1:] for r in rows], axis=1)
    return surf[:n, 0], surf[:n, 1:], cols, depth


def _write_h5(path: Path, *, row: dict, vs, zeta, rock, t, surface, columns, depth, lx_m, ly_m, hx_m, dt_s, duration_s, wall_s) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".h5.part")
    nx_nodes = surface.shape[1]
    step = max(1, round(COLUMN_SPACING_M / hx_m))
    i_rec = (int(round(lx_m / hx_m)) + 2) // 2 - 1
    half = int((lx_m / 2) // COLUMN_SPACING_M)
    col_idx = np.arange(max(0, i_rec - half * step), min(nx_nodes - 1, i_rec + half * step) + 1, step)[: columns.shape[2]]
    with h5py.File(tmp, "w") as h:
        h.attrs["sample_id"] = str(row["sample_id"])
        h.attrs["task_id"] = str(row["sample_id"])
        h.attrs["wall_s"] = float(wall_s)
        params = h.create_group("params")
        for k, v in row.items():
            try:
                params.attrs[k] = float(v)
            except (TypeError, ValueError):
                params.attrs[k] = str(v)
        params.attrs["H"] = float(row["soil_thickness_m"])
        params.attrs["duration"] = float(duration_s)
        params.attrs["damping_freq_first"] = float(row["rayleigh_f1_hz"])
        params.attrs["damping_freq_second"] = float(row["rayleigh_f2_hz"])
        params.attrs["motion_t_shift"] = MOTION_T_SHIFT_S
        params.attrs["ricker_is"] = "acceleration"
        params.attrs["damping_applied"] = "xi_base"
        params.attrs["damping_family"] = str(row.get("damping_family", "explicit"))
        params.attrs["pad_m"] = float(row.get("pad_m", PAD_M))
        params.attrs["lx_interior_m"] = float(row.get("lx_interior_m", row.get("lx_m", lx_m)))
        h.create_dataset("Vs_realization_2D", data=vs.astype(np.float32), compression="gzip")
        h.create_dataset("Damping_zeta", data=zeta.astype(np.float32), compression="gzip")
        h.create_dataset("rock_mask", data=rock.astype(np.uint8), compression="gzip")
        g = h.create_group("grid")
        g.attrs.update(Lx=lx_m, Lz=ly_m, dx=hx_m, dz=hx_m, dt=dt_s, motion_freq=float(row["motion_freq_hz"]))
        acc = h.create_group("recorders").create_group("accel")
        acc.create_dataset("time", data=t.astype(np.float32), compression="gzip")
        ds = acc.create_dataset("surface", data=surface.astype(np.float32), compression="gzip", chunks=(surface.shape[0], min(64, nx_nodes)))
        ds.attrs["x_m"] = np.arange(nx_nodes) * hx_m
        ds = acc.create_dataset("columns", data=columns.astype(np.float32), compression="gzip")
        ds.attrs["depth_m"] = depth
        ds.attrs["x_m"] = col_idx * hx_m
        ds.attrs["layout"] = "n_time x n_depth x n_column"
    tmp.replace(path)


def _resolve_damping(row: dict, vs: np.ndarray, zeta: np.ndarray, rock: np.ndarray) -> tuple[dict, np.ndarray]:
    """Honor ``damping_family``. ``qvs`` replaces the CSV zeta with seiskit global_avg."""
    family = str(row.get("damping_family", "explicit") or "explicit")
    if family == "explicit":
        return dict(row, damping_family="explicit"), zeta
    if family != "qvs":
        raise ValueError(f"damping_family must be 'explicit' or 'qvs', got {family!r}")
    from case7_crops import qvs_xi

    xi_soil, xi_rock = qvs_xi(vs, rock)
    zeta = np.where(rock, xi_rock, xi_soil)
    return dict(row, damping_family="qvs", zeta_soil=float(xi_soil), zeta_rock=float(xi_rock)), zeta


def h5_ok(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    import h5py

    try:
        with h5py.File(path, "r") as h:
            return h["recorders/accel/surface"].shape[0] >= 10 and "Vs_realization_2D" in h
    except Exception:
        return False


def _row_value(row: dict, key: str, default):
    v = row.get(key)
    return default if v in (None, "") else type(default)(v)


def simulate(row: dict, out_dir: Path, *, hx_m: float = DZ_M, lx_m: float | None = None,
             duration_s: float = DURATION_S, dt_s: float = DT_S, keep: bool = False,
             pad_m: float | None = None, mesh_mapping: str = "harmonic") -> Path:
    """Solve one generated row. Keyword arguments are the defaults; the row overrides them."""
    vs, zeta, rock = section_field(row)
    return _simulate_arrays(
        row, vs, zeta, rock, out_dir, hx_m=hx_m, lx_m=lx_m, duration_s=duration_s,
        dt_s=dt_s, keep=keep, pad_m=pad_m, mesh_mapping=mesh_mapping,
    )


def simulate_fields(vs, zeta, rock, row: dict, out_dir: Path, **kwargs) -> Path:
    """Solve imported 1 m arrays. The row must name a source hash and the Rayleigh anchors."""
    if not str(row.get("source_sha256") or ""):
        raise ValueError("imported sections need source_sha256")
    return _simulate_arrays(row, np.asarray(vs), np.asarray(zeta), np.asarray(rock), out_dir, **kwargs)


def _simulate_arrays(row: dict, vs, zeta, rock, out_dir: Path, *, hx_m: float = DZ_M, lx_m: float | None = None,
                     duration_s: float = DURATION_S, dt_s: float = DT_S, keep: bool = False,
                     pad_m: float | None = None, mesh_mapping: str = "harmonic") -> Path:
    """Pad, damp, and solve. Damping is resolved before coarsening."""
    _preload_opensees_libs()
    from seiskit.analysis import run_opensees_analysis
    from seiskit.builder import build_model_data

    hx_m = _row_value(row, "hx_m", float(hx_m))
    dt_s = _row_value(row, "dt_s", float(dt_s))
    duration_s = _row_value(row, "duration_s", float(duration_s))
    mesh_mapping = _row_value(row, "mesh_mapping", str(mesh_mapping))
    k = int(round(hx_m / DZ_M))
    if k < 1 or abs(k * DZ_M - hx_m) > 1e-9:
        raise ValueError(f"hx_m={hx_m} is not a whole number of {DZ_M} m cells")

    lx_interior = float(row["lx_m"]) if lx_m is None else lx_m
    pad_m = PAD_M if pad_m is None else float(pad_m)
    ly_m = float(row.get("depth_m", DEPTH_M))
    nx = int(round(lx_interior / DZ_M))
    vs, zeta, rock = (pad_free_field(a[:, :nx], pad_m, DZ_M) for a in (vs, zeta, rock))
    rock = rock > 0.5
    row, zeta = _resolve_damping(row, vs, zeta, rock)
    vs, zeta, rock = coarsen_fields(vs, zeta, rock, k, mesh_mapping)
    lx_m = vs.shape[1] * hx_m
    row = dict(row, pad_m=pad_m, lx_interior_m=lx_interior, hx_m=hx_m, dt_s=dt_s, mesh_mapping=mesh_mapping)
    config = _analysis_config(row, ly_m=ly_m, lx_m=lx_m, hx_m=hx_m, duration_s=duration_s, dt_s=dt_s)
    density = float(row.get("density_kg_m3") or 2000.0)
    poisson = float(row.get("poisson") or 0.3)
    model = build_model_data(config, vs, np.full_like(vs, density), np.full_like(vs, poisson), bedrock_mask=rock)
    sid = str(row["sample_id"])
    work = out_dir / "work" / sid
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    t0 = time.time()
    status = run_opensees_analysis(config, model, sid, str(work))
    wall = time.time() - t0
    print(f"{sid}: {status} wall={wall:.0f}s lx={lx_m:.0f} (interior {lx_interior:.0f} + {pad_m:.0f} m pads)", flush=True)
    if not str(status).startswith("Finished"):
        raise RuntimeError(status)
    t, surface, columns, depth = _load_recorders(work / sid, ly_m)
    dest = out_dir / "h5" / f"{sid}.h5"
    _write_h5(dest, row=row, vs=vs, zeta=zeta, rock=rock, t=t, surface=surface, columns=columns, depth=depth,
              lx_m=lx_m, ly_m=ly_m, hx_m=hx_m, dt_s=dt_s, duration_s=duration_s, wall_s=wall)
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--index", type=int, help="row of the manifest (default: all missing rows, in order)")
    ap.add_argument("--duration", type=float, default=DURATION_S)
    ap.add_argument("--smoke", action="store_true", help="row 0 on a 5 m, 400 m mesh for 2 s")
    ap.add_argument("--timing", type=float, default=0.0, help="run row --index for this many seconds of motion (timing only)")
    args = ap.parse_args()
    _limit_blas_threads()
    rows = read_rows(args.manifest)
    if args.smoke:
        row = dict(rows[0], sample_id="smoke")
        print(simulate(row, args.out_dir, hx_m=5.0, lx_m=400.0, duration_s=2.0, keep=True, pad_m=0.0))
        return
    if args.timing:
        row = dict(rows[args.index or 0])
        row["sample_id"] = f"timing_{row['sample_id']}"
        print(simulate(row, args.out_dir, duration_s=args.timing))
        return
    chosen = [rows[args.index]] if args.index is not None else rows
    for row in chosen:
        dest = args.out_dir / "h5" / f"{row['sample_id']}.h5"
        if h5_ok(dest):
            print(f"skip {dest}", flush=True)
            continue
        try:
            print(f"wrote {simulate(row, args.out_dir, duration_s=args.duration)}", flush=True)
        except Exception as exc:  # keep the queue going; failures are listed for resubmission
            print(f"FAILED {row['sample_id']}: {exc}", flush=True)
            with (args.out_dir / "failures.log").open("a") as fh:
                fh.write(f"{row['sample_id']}\t{exc}\n")


if __name__ == "__main__":
    main()
