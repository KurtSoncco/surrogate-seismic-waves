"""Five 500 m Case 7 crops, each extended to 1500 m for an OpenSees run.

The inner 500 m is the LOGLO window. The extra 500 m on each side repeats the
edge column, the same padding as seiskit ``_extend_profile`` and the basin
ensemble. Damping is seiskit ``global_avg`` with no zeta override: soil xi is
the harmonic mean of Taborda–Bielak Q(Vs), and rock xi is Q of the median
rock Vs. Both are computed on the extended mesh, which is what OpenSees
averages. The Rayleigh pair is the center-column soil quarter-wavelength
frequency and three times that frequency.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROCK_VS_MPS = 760.0
INNER_NX = 500
PAD_NX = 500
LX_SOLVE_M = 1500.0
DX_M = 1.0
DEPTH_M = 150.0
MOTION_FREQ_HZ = 3.0
MOTION_T_SHIFT_S = 0.5
DURATION_S = 15.0
DT_S = 0.01
RAYLEIGH_F2_OVER_F1 = 3.0
F1_FLOOR_HZ = 0.2

CROPS = (
    {"sample_id": "crop_west", "name": "flat_west", "center_x_m": -1500.0},
    {"sample_id": "crop_center", "name": "center", "center_x_m": 0.0},
    {"sample_id": "crop_thin", "name": "thinning", "center_x_m": 800.0},
    {"sample_id": "crop_ramp", "name": "ramp", "center_x_m": 1175.0},
    {"sample_id": "crop_east", "name": "flat_east", "center_x_m": 1605.0},
)


def quality_factor(vs: np.ndarray) -> np.ndarray:
    """Taborda and Bielak (2013) Q(Vs), Vs in m/s. Matches seiskit."""
    x = np.asarray(vs, dtype=np.float64) / 1000.0
    return np.polyval([0.31, -5.29, 34.7, -103.0, 153.0, -16.0, 10.5], x)


def qvs_xi(vs: np.ndarray, rock: np.ndarray) -> tuple[float, float]:
    """Soil harmonic-mean Q and rock Q(median Vs), as seiskit ``global_avg``."""
    soil = np.asarray(vs, dtype=np.float64)[~np.asarray(rock, dtype=bool)]
    bedrock = np.asarray(vs, dtype=np.float64)[np.asarray(rock, dtype=bool)]
    if soil.size == 0 or bedrock.size == 0:
        raise ValueError("Q–Vs global_avg needs both soil and rock elements")
    q_soil = quality_factor(soil)
    q_harm = float(q_soil.size / np.sum(1.0 / q_soil))
    q_rock = float(quality_factor(np.array(np.median(bedrock))))
    return 1.0 / (2.0 * q_harm), 1.0 / (2.0 * q_rock)


def extend_profile(vs: np.ndarray, lx_m: float = LX_SOLVE_M, dx_m: float = DX_M) -> np.ndarray:
    """Repeat edge columns out to ``lx_m``. Same split as seiskit ``_extend_profile``."""
    vs = np.asarray(vs)
    nx_total = int(round(lx_m / dx_m))
    n_pad = nx_total - vs.shape[1]
    if n_pad < 0:
        raise ValueError(f"profile width {vs.shape[1]} exceeds {nx_total} columns")
    if n_pad == 0:
        return np.array(vs, copy=True)
    n_left = n_pad // 2
    n_right = n_pad - n_left
    left = np.repeat(vs[:, :1], n_left, axis=1)
    right = np.repeat(vs[:, -1:], n_right, axis=1)
    return np.hstack([left, vs, right])


def soil_quarter_wave_hz(vs_col: np.ndarray, rock_vs: float) -> float:
    """f0 = 0.25 / travel time through cells with Vs at or below ``rock_vs``."""
    col = np.asarray(vs_col, dtype=np.float64)
    soil = col[col <= rock_vs]
    use = soil if soil.size else col
    travel = float(np.sum(DX_M / np.maximum(use, 1e-12)))
    return 0.25 / travel


SVLAB_XI_SOIL = 0.0075
SVLAB_ROCK_FRAC = 0.1
SECTION_LX_M = 4000.0
CONTEXT_WIDTHS_M = (1000.0, 1500.0)


def rayleigh_pair(vs_inner: np.ndarray, rock_vs: float) -> tuple[float, float, float]:
    """Center-column soil f0, then (max(f0, 0.2), 3 f1) as in the section runs."""
    f0 = soil_quarter_wave_hz(vs_inner[:, vs_inner.shape[1] // 2], rock_vs)
    f1 = max(F1_FLOOR_HZ, f0)
    return f0, f1, RAYLEIGH_F2_OVER_F1 * f1


def span_columns(center_x_m: float, inner_m: float) -> tuple[int, int, float]:
    """Column start, count, and Case 7 x of ``[center ± inner/2)``, shifted onto the mesh.

    A window that would hang off ±2000 m is slid inward so the requested center
    stays inside it. ``mesh_shift_m`` is how far the left edge moved.
    """
    from case7a import NX_1M, X_LEFT_M

    n = int(round(float(inner_m) / DX_M))
    if n <= 0 or n > NX_1M:
        raise ValueError(f"inner width {inner_m} m is outside 1..{NX_1M}")
    x0_ideal = float(center_x_m) - n * DX_M / 2.0
    c0_ideal = int(round(x0_ideal - X_LEFT_M))
    c0 = int(np.clip(c0_ideal, 0, NX_1M - n))
    return c0, n, (c0 - c0_ideal) * DX_M


def cut_inner(vs_1m: np.ndarray, center_x_m: float, inner_m: float | None = None) -> tuple[np.ndarray, int]:
    """Columns of the 1 m Case 7 grid covering ``inner_m`` about ``center_x_m``.

    The default width is the 500 m LOGLO window, cut as
    ``[center - 250, center + 250)``.
    """
    from case7a import NX, window_column_start

    if inner_m is None:
        if NX != INNER_NX:
            raise ValueError(f"LOGLO window is {NX} m, crops are {INNER_NX} m")
        c0 = window_column_start(float(center_x_m))
        return np.asarray(vs_1m[:, c0 : c0 + INNER_NX], dtype=np.float64), c0
    c0, n, _shift = span_columns(center_x_m, inner_m)
    return np.asarray(vs_1m[:, c0 : c0 + n], dtype=np.float64), c0


def prepare_crop(
    vs_1m: np.ndarray,
    crop: dict,
    rock_vs: float,
    *,
    inner_m: float | None = None,
    pad_m: float = PAD_NX,
    damping: str = "qvs",
) -> dict:
    """Inner window, 500 m free-field pads, and the Rayleigh numbers for one crop.

    ``damping='qvs'`` is seiskit global_avg on the extended mesh.
    ``damping='svlab'`` is soil ζ 0.0075 and rock one tenth of that.
    """
    from case7a import X_LEFT_M

    width = None if inner_m is None else float(inner_m)
    if "inner_m" in crop and width is None:
        width = float(crop["inner_m"])
    damping = str(crop.get("damping", damping))
    inner, c0 = cut_inner(vs_1m, float(crop["center_x_m"]), width)
    n = inner.shape[1]
    if inner.shape[0] != int(DEPTH_M):
        raise ValueError(f"{crop['sample_id']} inner shape {inner.shape}")
    pad_n = int(round(float(pad_m) / DX_M))
    lx_solve = float(n + 2 * pad_n)
    extended = extend_profile(inner, lx_m=lx_solve)
    rock_ext = extended > rock_vs
    if damping == "qvs":
        xi_soil, xi_rock = qvs_xi(extended, rock_ext)
    elif damping == "svlab":
        xi_soil, xi_rock = SVLAB_XI_SOIL, SVLAB_XI_SOIL * SVLAB_ROCK_FRAC
    else:
        raise ValueError(f"damping must be 'qvs' or 'svlab', got {damping!r}")
    zeta = np.where(inner > rock_vs, xi_rock, xi_soil)
    f0, f1, f2 = rayleigh_pair(inner, rock_vs)
    x0 = X_LEFT_M + c0 * DX_M
    ideal = float(crop["center_x_m"]) - n * DX_M / 2.0
    return {
        "sample_id": crop["sample_id"],
        "name": crop["name"],
        "center_x_m": float(crop["center_x_m"]),
        "inner_m": float(n),
        "pad_m": float(pad_n),
        "lx_solve_m": lx_solve,
        "damping": damping,
        "mesh_shift_m": float(x0 - ideal),
        "crop_x0_m": float(x0),
        "crop_x1_m": float(x0 + n * DX_M),
        "column0": int(c0),
        "vs_inner": inner,
        "vs_extended": extended,
        "rock_extended": rock_ext,
        "zeta_inner": zeta,
        "xi_soil": xi_soil,
        "xi_rock": xi_rock,
        "f0_hz": f0,
        "rayleigh_f1_hz": f1,
        "rayleigh_f2_hz": f2,
    }


def context_jobs() -> list[dict]:
    """Ten wider crops plus the full section at Q–Vs and at SVLAB damping."""
    jobs = []
    for width in CONTEXT_WIDTHS_M:
        for crop in CROPS:
            jobs.append(
                {
                    "sample_id": f"{crop['sample_id']}_w{int(width)}",
                    "name": crop["name"],
                    "center_x_m": crop["center_x_m"],
                    "inner_m": float(width),
                    "damping": "qvs",
                }
            )
    for damping in ("qvs", "svlab"):
        jobs.append(
            {
                "sample_id": f"full_{damping}",
                "name": "full_section",
                "center_x_m": 0.0,
                "inner_m": SECTION_LX_M,
                "damping": damping,
            }
        )
    return jobs


def write_mesh(pack: dict, meshes_dir: Path) -> Path:
    meshes_dir.mkdir(parents=True, exist_ok=True)
    sid = pack["sample_id"]
    np.savez_compressed(
        meshes_dir / f"{sid}.npz",
        vs_extended=pack["vs_extended"].astype(np.float32),
        rock_extended=pack["rock_extended"].astype(np.uint8),
        vs_inner=pack["vs_inner"].astype(np.float32),
        zeta_inner=pack["zeta_inner"].astype(np.float32),
    )
    meta = {k: pack[k] for k in pack if not isinstance(pack[k], np.ndarray)}
    (meshes_dir / f"{sid}.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meshes_dir / f"{sid}.json"


def _write_jobs(meshes_dir: Path, rows: list[dict]) -> None:
    (meshes_dir / "jobs.json").write_text(json.dumps(rows, indent=2) + "\n")


def prepare_jobs(meshes_dir: Path, jobs: list[dict]) -> list[dict]:
    """Cut ``jobs`` from the cached Case 7 grid and write meshes plus jobs.json."""
    from case7a import ROCK_VS, ensure_cache, paint_section_1m

    vs_1m = paint_section_1m(ensure_cache()["vs"])
    written = []
    for crop in jobs:
        pack = prepare_crop(vs_1m, crop, ROCK_VS)
        if "duration_s" in crop:
            pack["duration_s"] = float(crop["duration_s"])
        write_mesh(pack, meshes_dir)
        row = {k: pack[k] for k in pack if not isinstance(pack[k], np.ndarray)}
        written.append(row)
        print(
            f"{pack['sample_id']}: x=[{pack['crop_x0_m']:.0f},{pack['crop_x1_m']:.0f}) "
            f"inner={pack['inner_m']:.0f} solve={pack['lx_solve_m']:.0f} shift={pack['mesh_shift_m']:.0f} "
            f"damp={pack['damping']} f0={pack['f0_hz']:.3f} "
            f"xi=({pack['xi_soil']:.4f},{pack['xi_rock']:.4f})",
            flush=True,
        )
    _write_jobs(meshes_dir, written)
    return written


def prepare_all(meshes_dir: Path) -> list[dict]:
    """Cut the five 500 m crops. No jobs.json, so the original runner list stays CROPS."""
    from case7a import ROCK_VS, ensure_cache, paint_section_1m

    vs_1m = paint_section_1m(ensure_cache()["vs"])
    written = []
    for crop in CROPS:
        pack = prepare_crop(vs_1m, crop, ROCK_VS)
        write_mesh(pack, meshes_dir)
        written.append({k: pack[k] for k in pack if not isinstance(pack[k], np.ndarray)})
        print(
            f"{pack['sample_id']}: x=[{pack['crop_x0_m']:.0f},{pack['crop_x1_m']:.0f}) "
            f"f0={pack['f0_hz']:.3f} Hz Rayleigh=({pack['rayleigh_f1_hz']:.3f},{pack['rayleigh_f2_hz']:.3f}) "
            f"xi_soil={pack['xi_soil']:.4f} xi_rock={pack['xi_rock']:.4f}",
            flush=True,
        )
    return written


def prepare_context(meshes_dir: Path) -> list[dict]:
    """Wider crops and the two full-section meshes. Writes jobs.json."""
    return prepare_jobs(meshes_dir, context_jobs())


def record_pilot_jobs() -> list[dict]:
    """Flat-west 500 m crops at two dampings and three record lengths.

    Q–Vs at 15 s already exists, so that cell is not repeated.
    """
    west = next(crop for crop in CROPS if crop["name"] == "flat_west")
    jobs = []
    for damping, durations in (("svlab", (15.0, 30.0, 60.0)), ("qvs", (30.0, 60.0))):
        for duration_s in durations:
            jobs.append(
                {
                    "sample_id": f"crop_west_{damping}_d{int(duration_s)}",
                    "name": west["name"],
                    "center_x_m": west["center_x_m"],
                    "inner_m": float(INNER_NX),
                    "damping": damping,
                    "duration_s": float(duration_s),
                }
            )
    return jobs


def prepare_record_pilot(meshes_dir: Path, context_meshes: Path) -> list[dict]:
    """Pilot crops plus the existing full_svlab mesh. Writes jobs.json."""
    import shutil

    written = prepare_jobs(meshes_dir, record_pilot_jobs())
    for name in ("full_svlab.json", "full_svlab.npz"):
        src = context_meshes / name
        if not src.is_file():
            raise FileNotFoundError(src)
        shutil.copy(src, meshes_dir / name)
    full = json.loads((meshes_dir / "full_svlab.json").read_text())
    full["duration_s"] = float(DURATION_S)
    (meshes_dir / "full_svlab.json").write_text(json.dumps(full, indent=2) + "\n")
    rows = [full, *written]
    _write_jobs(meshes_dir, rows)
    return rows
