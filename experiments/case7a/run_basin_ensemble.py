#!/usr/bin/env python3
"""OpenSees runs for the gradational basin manifest.

Train rows only. The Case 7a holdout is not simulated and is not written
into the H5 directory. Mesh matches the GIFNO cache: 1500 m wide, 150 m
deep, 1 m cells, 500 m of edge padding, Rayleigh damping at the two
manifest frequencies, and 21 recorders at 15 m. Downhole is at 120 m,
inside the rock. An existing run_*.h5 is left in place.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import multiprocessing as mp
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[1]
SEISKIT_ROOT = Path(os.environ.get("SEISKIT_ROOT", "/home/kurt-/seiskit"))
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))
if SEISKIT_ROOT.is_dir() and str(SEISKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(SEISKIT_ROOT))

from basin_ensemble import DEPTH_M, basin_rows, gradational_field  # noqa: E402

LX_M = 1500.0
DX_M = 1.0
DT_S = 0.01
DURATION_S = 15.0
N_EACH_SIDE = 10
RECORDER_SPACING_M = 15.0


def _preload_opensees_libs() -> None:
    """Load libquadmath before OpenSees. The .so is inside the wheel."""
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    lib_dir = (
        Path(sys.prefix)
        / "lib"
        / f"python{version}"
        / "site-packages"
        / "openseespylinux"
        / "lib"
    )
    if not lib_dir.is_dir():
        return
    preferred = ("libquadmath.so.0", "libgfortran.so.5", "libgomp.so.1")
    names = list(preferred) + sorted(path.name for path in lib_dir.glob("lib*.so*"))
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        path = lib_dir / name
        if not path.is_file():
            continue
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue


def _train_only(rows: list[dict]) -> list[dict]:
    return [row for row in rows if str(row["split"]) == "train" and int(row["holdout_case7a"]) == 0]


def _extend(vs: np.ndarray, lx_m: float, dx_m: float) -> np.ndarray:
    from seiskit.gaussian_field import _extend_profile

    extended, _ = _extend_profile(vs, Lx=lx_m, dx=dx_m)
    return extended


def _analysis_config(row: dict, *, ly_m: float, lx_m: float, hx_m: float, duration_s: float, dt_s: float):
    from seiskit.config import AnalysisConfig

    downhole_m = float(row["downhole_depth_m"])
    return AnalysisConfig(
        Ly=ly_m,
        Lx=lx_m,
        hx=hx_m,
        dt=dt_s,
        duration=duration_s,
        motion_freq=float(row["rayleigh_f1_hz"]),
        motion_t_shift=0.5,
        damping_freqs=(float(row["rayleigh_f1_hz"]), float(row["rayleigh_f2_hz"])),
        damping_zeta=float(row["zeta_soil"]),
        damping_method="global_avg",
        xi_soil_base=float(row["zeta_soil"]),
        xi_rock_base=float(row["zeta_rock"]),
        boundary_condition_type="2D",
        record_center_nodes=False,
        center_node_y_positions=[downhole_m, ly_m],
        record_lateral_span_at_center_depths=(N_EACH_SIDE, RECORDER_SPACING_M),
        record_all_surface_nodes=False,
        element_type="4node",
        solver_type=os.environ.get("BASIN_SOLVER", "UmfPack"),
        dynamic_algorithm="Linear",
        dynamic_tolerance=1.0e-3,
        max_dynamic_iter=25,
        max_time_per_batch=6.0 * 3600.0,
    )


def _load_recorder_txt(recorder_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def y_of(path: Path) -> float:
        stem = path.name
        token = stem.split("_dof", 1)[0].replace("row_y", "")
        return float(token)

    files = sorted(recorder_dir.glob("row_y*_dof1_accel.txt"), key=y_of)
    if len(files) != 2:
        raise RuntimeError(f"expected two recorder rows in {recorder_dir}, found {len(files)}")
    chunks = []
    time_arr = None
    ys = []
    for path in files:
        arr = np.loadtxt(path)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if time_arr is None:
            time_arr = arr[:, 0]
        chunks.append(arr[:, 1:])
        ys.append(y_of(path))
    data = np.hstack(chunks)
    if time_arr is None or data.shape[1] % 2 != 0:
        raise RuntimeError(f"recorder channels are not a surface/downhole pair: {data.shape}")
    return time_arr, data, np.asarray(ys, dtype=np.float64)


def _write_h5(
    path: Path,
    *,
    index: int,
    row: dict,
    vs: np.ndarray,
    zeta: np.ndarray,
    time_arr: np.ndarray,
    accel: np.ndarray,
    row_y_m: np.ndarray,
    lx_m: float,
    ly_m: float,
    dx_m: float,
    dt_s: float,
    duration_s: float,
) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".h5.part")
    center = vs.shape[1] // 2
    with h5py.File(tmp, "w") as handle:
        handle.attrs["index"] = index
        handle.attrs["sample_id"] = str(row["sample_id"])
        handle.attrs["task_id"] = str(row["sample_id"])
        params = handle.create_group("params")
        params.attrs["H_discretized"] = float(row["soil_thickness_m"])
        params.attrs["H"] = float(row["soil_thickness_m"])
        params.attrs["CoV"] = 0.0
        params.attrs["rf_seed"] = 0
        params.attrs["seed"] = 0
        params.attrs["f0_effective"] = float(row["rayleigh_f1_hz"])
        params.attrs["Vs1"] = float(row["vs_surface_mps"])
        params.attrs["Vs2"] = float(row["vs_rock_mps"])
        params.attrs["duration"] = float(duration_s)
        params.attrs["damping_freq_first"] = float(row["rayleigh_f1_hz"])
        params.attrs["damping_freq_second"] = float(row["rayleigh_f2_hz"])
        params.attrs["zeta_soil"] = float(row["zeta_soil"])
        params.attrs["zeta_rock"] = float(row["zeta_rock"])
        handle.create_dataset("Vs_profile_1D", data=vs[:, center].astype(np.float32), compression="gzip")
        handle.create_dataset("Vs_realization_2D", data=vs.astype(np.float32), compression="gzip")
        handle.create_dataset("Damping_zeta", data=zeta.astype(np.float32), compression="gzip")
        grid = handle.create_group("grid")
        grid.attrs["Lx"] = lx_m
        grid.attrs["Lz"] = ly_m
        grid.attrs["dx"] = dx_m
        grid.attrs["dz"] = dx_m
        grid.attrs["dt"] = dt_s
        grid.attrs["motion_freq"] = float(row["rayleigh_f1_hz"])
        accel_group = handle.create_group("recorders").create_group("accel")
        accel_group.create_dataset("time", data=time_arr.astype(np.float32), compression="gzip")
        ds = accel_group.create_dataset("data", data=accel.astype(np.float32), compression="gzip")
        ds.attrs["layout"] = "n_time x n_channels"
        ds.attrs["row_y_m"] = row_y_m
    tmp.replace(path)


def _zeta_grid(vs: np.ndarray, rock: np.ndarray, row: dict) -> np.ndarray:
    """Soil and rock ratios stored for the network. Rayleigh uses the same pair."""
    zeta = np.full(vs.shape, float(row["zeta_soil"]), dtype=np.float64)
    zeta[rock] = float(row["zeta_rock"])
    return zeta


def simulate_row(
    index: int,
    row: dict,
    out_dir: Path,
    *,
    ly_m: float = DEPTH_M,
    lx_m: float = LX_M,
    hx_m: float = DX_M,
    duration_s: float = DURATION_S,
    dt_s: float = DT_S,
    keep_recorders: bool = False,
) -> Path:
    """Run one column and write run_{index}.h5. Returns the H5 path."""
    _preload_opensees_libs()
    from seiskit.analysis import run_opensees_analysis
    from seiskit.builder import build_model_data

    vs_window, rock_window = gradational_field(row)
    if hx_m != DX_M or ly_m != DEPTH_M:
        vs_window, rock_window = _resample_window(vs_window, rock_window, ly_m, lx_m, hx_m)
    vs = _extend(vs_window, lx_m, hx_m)
    rock = _extend(rock_window.astype(np.float64), lx_m, hx_m) > 0.5
    config = _analysis_config(
        row, ly_m=ly_m, lx_m=lx_m, hx_m=hx_m, duration_s=duration_s, dt_s=dt_s
    )
    zeta = _zeta_grid(vs, rock, row)
    rho = np.full_like(vs, 2000.0)
    nu = np.full_like(vs, 0.3)
    model = build_model_data(config, vs, rho, nu, bedrock_mask=rock)
    work = out_dir / "work" / f"run_{index}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    task_id = str(row["sample_id"])
    status = run_opensees_analysis(config, model, task_id, str(work))
    print(status, flush=True)
    if not str(status).startswith("Finished"):
        raise RuntimeError(status)
    time_arr, accel, row_y = _load_recorder_txt(work / task_id)
    h5_path = out_dir / "h5" / f"run_{index}.h5"
    _write_h5(
        h5_path,
        index=index,
        row=row,
        vs=vs,
        zeta=zeta,
        time_arr=time_arr,
        accel=accel,
        row_y_m=row_y,
        lx_m=lx_m,
        ly_m=ly_m,
        dx_m=hx_m,
        dt_s=dt_s,
        duration_s=duration_s,
    )
    if not keep_recorders:
        shutil.rmtree(work, ignore_errors=True)
    return h5_path


def _resample_window(
    vs: np.ndarray, rock: np.ndarray, ly_m: float, lx_m: float, hx_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Used by the smoke mesh. The ensemble path keeps the 1 m field."""
    nz = int(round(ly_m / hx_m))
    nx = int(round(min(lx_m, 500.0) / hx_m))
    z_idx = np.clip(np.linspace(0, vs.shape[0] - 1, nz).astype(int), 0, vs.shape[0] - 1)
    x_idx = np.clip(np.linspace(0, vs.shape[1] - 1, nx).astype(int), 0, vs.shape[1] - 1)
    return vs[np.ix_(z_idx, x_idx)], rock[np.ix_(z_idx, x_idx)]


def _h5_ok(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    import h5py

    try:
        with h5py.File(path, "r") as handle:
            n_time, n_ch = handle["recorders/accel/data"].shape
            return n_time >= 10 and n_ch >= 2 and n_ch % 2 == 0 and "Vs_realization_2D" in handle
    except Exception:
        return False


def choose_workers(rss_gb: float, ram_gb: float, *, cap: int = 16) -> int:
    """How many columns to run at once from one column's RSS.

    ``min(cap, floor(0.7 * RAM / RSS))``. A missing measurement falls back to 8.
    """
    if rss_gb <= 0.0 or ram_gb <= 0.0:
        return min(8, cap)
    n = int((0.7 * ram_gb) // rss_gb)
    if n < 1:
        return 1
    return min(cap, n)


def select_jobs(
    rows: list[dict],
    out_dir: Path,
    *,
    limit: int | None,
    index: int | None,
    start: int = 0,
) -> list[tuple[int, dict]]:
    """Train rows still missing an H5. The Case 7a holdout is not in this list."""
    rows = _train_only(rows)
    if index is not None:
        chosen = [(index, rows[index])]
    else:
        chosen = [(i, row) for i, row in enumerate(rows) if i >= start]
        if limit is not None:
            chosen = chosen[:limit]
    pending: list[tuple[int, dict]] = []
    for i, row in chosen:
        dest = out_dir / "h5" / f"run_{i}.h5"
        if _h5_ok(dest):
            print(f"skip {row['sample_id']} ({dest})", flush=True)
            continue
        pending.append((i, row))
    return pending


def _record_failure(out_dir: Path, sample_id: str, exc: BaseException) -> None:
    print(f"FAILED {sample_id}: {exc}", flush=True)
    with (out_dir / "failures.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{sample_id}\t{exc}\n")


def _limit_blas_threads() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"


def _run_one(index: int, row: dict, out_dir: str, duration_s: float) -> str:
    """One column in a spawned process. OpenSees is loaded after the spawn."""
    _limit_blas_threads()
    written = simulate_row(index, row, Path(out_dir), duration_s=duration_s)
    return str(written)


def _run_logged(index: int, row: dict, out_dir: Path, duration_s: float) -> None:
    dest = out_dir / "h5" / f"run_{index}.h5"
    print(f"start {row['sample_id']} -> {dest}", flush=True)
    try:
        written = simulate_row(index, row, out_dir, duration_s=duration_s)
    except Exception as exc:
        _record_failure(out_dir, str(row["sample_id"]), exc)
        return
    print(f"wrote {written}", flush=True)


def run_queue(
    out_dir: Path,
    *,
    limit: int | None,
    index: int | None,
    duration_s: float,
    workers: int = 1,
    start: int = 0,
) -> None:
    jobs = select_jobs(
        basin_rows(),
        out_dir,
        limit=limit,
        index=index,
        start=start,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    if workers <= 1 or len(jobs) <= 1:
        for i, row in jobs:
            _run_logged(i, row, out_dir, duration_s)
        return
    ctx = mp.get_context("spawn")
    print(f"pool workers={workers} jobs={len(jobs)}", flush=True)
    for i, row in jobs:
        print(f"start {row['sample_id']} -> {out_dir / 'h5' / f'run_{i}.h5'}", flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        futures = {
            executor.submit(_run_one, i, row, str(out_dir), duration_s): row
            for i, row in jobs
        }
        for fut in as_completed(futures):
            row = futures[fut]
            sample_id = str(row["sample_id"])
            try:
                written = fut.result()
            except Exception as exc:
                _record_failure(out_dir, sample_id, exc)
                continue
            print(f"wrote {written}", flush=True)


def run_smoke(out_dir: Path) -> Path:
    """Tiny mesh to confirm OpenSees, recorders, and the H5 schema."""
    row = basin_rows()[0]
    row = dict(row)
    row["downhole_depth_m"] = 5.0
    row["sample_id"] = "smoke"
    path = simulate_row(
        0,
        row,
        out_dir / "smoke",
        ly_m=10.0,
        lx_m=40.0,
        hx_m=5.0,
        duration_s=0.05,
        dt_s=0.01,
    )
    print(f"smoke wrote {path}", flush=True)
    return path


def _write_index(out_dir: Path) -> None:
    rows = _train_only(basin_rows())
    path = out_dir / "train_index.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_index", "sample_id", "h5_name"])
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow(
                {"run_index": i, "sample_id": row["sample_id"], "h5_name": f"run_{i}.h5"}
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "basin_ensemble",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--start", type=int, default=0, help="First train-row index to run")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--duration", type=float, default=DURATION_S)
    args = parser.parse_args()
    _write_index(args.out_dir)
    if args.smoke:
        run_smoke(args.out_dir)
        return
    run_queue(
        args.out_dir,
        limit=args.limit,
        index=args.index,
        duration_s=args.duration,
        workers=args.workers,
        start=args.start,
    )


if __name__ == "__main__":
    main()
