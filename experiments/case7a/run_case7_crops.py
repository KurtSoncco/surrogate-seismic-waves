#!/usr/bin/env python3
"""OpenSees on one extended Case 7 crop.

The mesh is the 500 m crop plus 500 m of repeated edge column on each side.
Damping is seiskit ``global_avg`` with the Q–Vs xi stored in the mesh json
(soil harmonic mean, rock from the median rock Vs). Rayleigh frequencies are
that crop's center-column f0 and 3 f0. The H5 keeps the inner 500 m only.

    python run_case7_crops.py --prepare --meshes-dir data/case7a/opensees_crops/meshes
    python run_case7_crops.py --meshes-dir MESHES --index 0 --out-dir $SCRATCH/case7_opensees_crops
"""

from __future__ import annotations

import argparse
import json
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

from case7_crops import (  # noqa: E402
    DT_S,
    DURATION_S,
    DX_M,
    INNER_NX,
    LX_SOLVE_M,
    MOTION_FREQ_HZ,
    MOTION_T_SHIFT_S,
    PAD_NX,
    ROCK_VS_MPS,
    prepare_all,
    prepare_context,
)

DEPTH_M = 150.0


def h5_ok(path: Path, n_inner: int = INNER_NX) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    import h5py

    try:
        with h5py.File(path, "r") as handle:
            surf = handle["recorders/accel/surface"]
            return surf.shape[0] >= 10 and surf.shape[1] == int(n_inner) and "Vs_realization_2D" in handle
    except Exception:
        return False


def job_list(meshes_dir: Path) -> list[dict]:
    """jobs.json when this directory is a context study; otherwise the five 500 m crops."""
    path = meshes_dir / "jobs.json"
    if path.is_file():
        return json.loads(path.read_text())
    from case7_crops import CROPS

    return [dict(crop) for crop in CROPS]


def _geom(meta: dict) -> tuple[float, float, float]:
    inner = float(meta.get("inner_m", INNER_NX))
    pad = float(meta.get("pad_m", PAD_NX))
    lx = float(meta.get("lx_solve_m", inner + 2.0 * pad))
    return inner, pad, lx


def load_mesh(meshes_dir: Path, sample_id: str) -> dict:
    meta = json.loads((meshes_dir / f"{sample_id}.json").read_text())
    with np.load(meshes_dir / f"{sample_id}.npz") as pack:
        meta["vs_extended"] = pack["vs_extended"].astype(np.float64)
        meta["rock_extended"] = pack["rock_extended"].astype(bool)
        meta["vs_inner"] = pack["vs_inner"].astype(np.float64)
        meta["zeta_inner"] = pack["zeta_inner"].astype(np.float64)
    return meta


def _analysis_config(meta: dict, *, duration_s: float, dt_s: float):
    import inspect

    from seiskit.config import AnalysisConfig

    _inner, _pad, lx_m = _geom(meta)
    kwargs = dict(
        Ly=DEPTH_M,
        Lx=lx_m,
        hx=DX_M,
        dt=dt_s,
        duration=duration_s,
        motion_freq=MOTION_FREQ_HZ,
        motion_t_shift=MOTION_T_SHIFT_S,
        damping_freqs=(float(meta["rayleigh_f1_hz"]), float(meta["rayleigh_f2_hz"])),
        damping_zeta=float(meta["xi_soil"]),
        damping_method="global_avg",
        xi_soil_base=float(meta["xi_soil"]),
        xi_rock_base=float(meta["xi_rock"]),
        boundary_condition_type="2D",
        record_center_nodes=False,
        record_all_surface_nodes=True,
        element_type="4node",
        solver_type=os.environ.get("BASIN_SOLVER", "UmfPack"),
        dynamic_algorithm="Linear",
        dynamic_tolerance=1.0e-3,
        max_dynamic_iter=25,
        max_time_per_batch=5.5 * 3600.0,
    )
    allowed = inspect.signature(AnalysisConfig.__init__).parameters
    dropped = sorted(set(kwargs) - set(allowed))
    if dropped:
        raise RuntimeError(f"installed seiskit does not accept {dropped}; update seiskit before solving")
    return AnalysisConfig(**kwargs)


def _load_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr[:, 0], arr[:, 1:]


def _write_h5(path: Path, meta: dict, t: np.ndarray, surface: np.ndarray, *, duration_s: float, dt_s: float, wall_s: float) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".h5.part")
    vs = meta["vs_inner"]
    zeta = meta["zeta_inner"]
    rock = vs > ROCK_VS_MPS
    x0 = float(meta["crop_x0_m"])
    with h5py.File(tmp, "w") as handle:
        handle.attrs["sample_id"] = meta["sample_id"]
        handle.attrs["wall_s"] = float(wall_s)
        params = handle.create_group("params")
        for key in (
            "sample_id",
            "name",
            "center_x_m",
            "crop_x0_m",
            "crop_x1_m",
            "xi_soil",
            "xi_rock",
            "f0_hz",
            "rayleigh_f1_hz",
            "rayleigh_f2_hz",
            "inner_m",
            "pad_m",
            "lx_solve_m",
            "mesh_shift_m",
        ):
            if key not in meta:
                continue
            value = meta[key]
            params.attrs[key] = value if isinstance(value, str) else float(value)
        params.attrs["damping"] = str(meta.get("damping", "qvs"))
        params.attrs["damping_applied"] = "xi_base"
        params.attrs["duration"] = float(duration_s)
        params.attrs["damping_freq_first"] = float(meta["rayleigh_f1_hz"])
        params.attrs["damping_freq_second"] = float(meta["rayleigh_f2_hz"])
        params.attrs["motion_t_shift"] = MOTION_T_SHIFT_S
        params.attrs["ricker_is"] = "acceleration"
        params.attrs["damping_method"] = "global_avg"
        inner_m, pad_m, lx_m = _geom(meta)
        params.attrs["pad_m"] = float(pad_m)
        params.attrs["inner_m"] = float(inner_m)
        handle.create_dataset("Vs_realization_2D", data=vs.astype(np.float32), compression="gzip")
        handle.create_dataset("Damping_zeta", data=zeta.astype(np.float32), compression="gzip")
        handle.create_dataset("rock_mask", data=rock.astype(np.uint8), compression="gzip")
        grid = handle.create_group("grid")
        inner_m, _pad_m, lx_m = _geom(meta)
        grid.attrs.update(
            Lx=float(inner_m),
            Lz=DEPTH_M,
            dx=DX_M,
            dz=DX_M,
            dt=dt_s,
            motion_freq=MOTION_FREQ_HZ,
            solved_Lx=float(lx_m),
        )
        acc = handle.create_group("recorders").create_group("accel")
        acc.create_dataset("time", data=t.astype(np.float32), compression="gzip")
        ds = acc.create_dataset(
            "surface",
            data=surface.astype(np.float32),
            compression="gzip",
            chunks=(surface.shape[0], min(64, surface.shape[1])),
        )
        ds.attrs["x_m"] = x0 + np.arange(surface.shape[1]) * DX_M
    tmp.replace(path)


def inner_surface(
    surface: np.ndarray,
    *,
    pad_m: float = PAD_NX,
    inner_m: float = INNER_NX,
    lx_m: float = LX_SOLVE_M,
) -> np.ndarray:
    """Surface nodes on the left edge of each inner element.

    The recorder file starts at x = 0 (absorbing nodes are omitted), so column
    ``k`` is the soil node at ``k`` metres. Defaults are the 500 m crop.
    """
    n_nodes = int(round(lx_m / DX_M)) + 1
    pad_n = int(round(pad_m / DX_M))
    inner_n = int(round(inner_m / DX_M))
    if surface.shape[1] != n_nodes:
        raise ValueError(f"expected {n_nodes} surface nodes, got {surface.shape[1]}")
    sl = surface[:, pad_n : pad_n + inner_n]
    if sl.shape[1] != inner_n:
        raise ValueError(f"surface {surface.shape} has no {inner_n}-node inner span at pad {pad_n}")
    return sl


def simulate(meta: dict, out_dir: Path, *, duration_s: float = DURATION_S, dt_s: float = DT_S) -> Path:
    from run_basin_ensemble import _preload_opensees_libs

    _preload_opensees_libs()
    from seiskit.analysis import run_opensees_analysis
    from seiskit.builder import build_model_data

    vs = meta["vs_extended"]
    rock = meta["rock_extended"]
    if "duration_s" in meta:
        duration_s = float(meta["duration_s"])
    config = _analysis_config(meta, duration_s=duration_s, dt_s=dt_s)
    model = build_model_data(config, vs, np.full_like(vs, 2000.0), np.full_like(vs, 0.3), bedrock_mask=rock)
    sid = str(meta["sample_id"])
    work = out_dir / "work" / sid
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    t0 = time.time()
    status = run_opensees_analysis(config, model, sid, str(work))
    wall = time.time() - t0
    print(f"{sid}: {status} wall={wall:.0f}s", flush=True)
    if not str(status).startswith("Finished"):
        raise RuntimeError(status)
    t, surface = _load_surface(work / sid / "surface_nodes_dof1_accel.txt")
    dest = out_dir / "h5" / f"{sid}.h5"
    inner_m, pad_m, lx_m = _geom(meta)
    kept = inner_surface(surface, pad_m=pad_m, inner_m=inner_m, lx_m=lx_m)
    _write_h5(dest, meta, t, kept, duration_s=duration_s, dt_s=dt_s, wall_s=wall)
    shutil.rmtree(work, ignore_errors=True)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meshes-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--index", type=int, default=None, help="crop index; default is every crop still missing an H5")
    ap.add_argument("--prepare", action="store_true", help="cut the five 500 m crops and write mesh files, no OpenSees")
    ap.add_argument("--prepare-context", action="store_true", help="cut the 1000/1500 m crops and the two full sections")
    ap.add_argument("--prepare-record-pilot", action="store_true", help="flat-west record-length crops plus a copy of full_svlab")
    ap.add_argument("--context-meshes", type=Path, default=None, help="context meshes holding full_svlab, for --prepare-record-pilot")
    ap.add_argument("--duration", type=float, default=DURATION_S)
    args = ap.parse_args()
    if args.prepare_record_pilot:
        if args.context_meshes is None:
            ap.error("--prepare-record-pilot needs --context-meshes")
        from case7_crops import prepare_record_pilot

        prepare_record_pilot(args.meshes_dir, args.context_meshes)
        return
    if args.prepare_context:
        prepare_context(args.meshes_dir)
        return
    if args.prepare:
        prepare_all(args.meshes_dir)
        return
    if args.out_dir is None:
        ap.error("--out-dir is required unless --prepare")
    from run_basin_ensemble import _limit_blas_threads

    _limit_blas_threads()
    jobs = job_list(args.meshes_dir)
    chosen = [jobs[args.index]] if args.index is not None else list(jobs)
    for crop in chosen:
        n_inner = int(round(float(crop.get("inner_m", INNER_NX))))
        dest = args.out_dir / "h5" / f"{crop['sample_id']}.h5"
        if h5_ok(dest, n_inner):
            print(f"skip {dest}", flush=True)
            continue
        try:
            meta = load_mesh(args.meshes_dir, crop["sample_id"])
            print(f"wrote {simulate(meta, args.out_dir, duration_s=args.duration)}", flush=True)
        except Exception as exc:
            print(f"FAILED {crop['sample_id']}: {exc}", flush=True)
            args.out_dir.mkdir(parents=True, exist_ok=True)
            with (args.out_dir / "failures.log").open("a") as handle:
                handle.write(f"{crop['sample_id']}\t{exc}\n")


if __name__ == "__main__":
    main()
