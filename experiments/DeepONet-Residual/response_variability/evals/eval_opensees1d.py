#!/usr/bin/env python3
"""Literal OpenSees 1-D Rayleigh column vs Haskell ξ_soil vs OpenSees 2-D.

Isolates damping *model shape* (Rayleigh vs hysteretic) after
``eval_haskell_xi.py`` isolated the ξ *value*. Does not retrain GINO.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import site
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass
import h5py

_EXP = Path(__file__).resolve().parents[2]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

_SEISKIT = Path.home() / "seiskit"
if _SEISKIT.is_dir() and str(_SEISKIT) not in sys.path:
    sys.path.insert(0, str(_SEISKIT))

import config  # noqa: E402
from ood_io import jsonable  # noqa: E402

from response_variability.evals.eval_haskell_xi import (  # noqa: E402
    read_soil_xi,
    resolve_h5,
)
from response_variability.evals.eval_iid import _as_central  # noqa: E402
from response_variability.metrics import (  # noqa: E402
    fmt_iqr,
    median_iqr,
    method_vs_reference,
    odd_quarter_wave_peaks,
    pearson,
    theoretical_f0,
)
from response_variability.plots.plot_presentation import (  # noqa: E402
    DOMAIN_SPECS,
    load_pack,
    pack_path,
)
from response_variability.style import (  # noqa: E402
    apply_nature_style,
    figsize,
    panel_letter,
    savefig,
)

OUT_DIR = config.RESULTS_DIR / "response_variability" / "eval_bias"
PACK_DIR = config.RESULTS_DIR / "presentation"
CACHE_DIR = OUT_DIR / "cache" / "opensees1d"
SCORE_DOMAINS = ("iid", "dipping")
N_MODES = 3
DT_ANALYSIS = 0.01  # 2-D solver dt; H5 recorders are often downsampled to 0.02
DAMPING_F2 = 10.0
MOTION_T_SHIFT = 0.5  # matches 2-D GIFNO data gen
PACK_BASE_Y = 2.0  # 2-D pack AF_within uses this elevation
_EPS = 1e-12

ARM_HASKELL_VS_2D = "Haskell ξ_soil vs 2-D (within)"
ARM_OS1D_VS_2D = "OpenSees 1-D vs 2-D (pack base)"
ARM_SHAPE_WITHIN = "Haskell ξ_soil vs OpenSees 1-D (within)"
ARM_SHAPE_OUTCROP = "Haskell ξ_soil vs OpenSees 1-D (outcrop)"
SCORE_ARMS = (ARM_HASKELL_VS_2D, ARM_OS1D_VS_2D, ARM_SHAPE_WITHIN, ARM_SHAPE_OUTCROP)


def prepend_opensees_lib() -> Path | None:
    """Put bundled openseespylinux/lib first on LD_LIBRARY_PATH."""
    search: list[Path] = []
    try:
        search.extend(Path(p) for p in site.getsitepackages())
    except Exception:
        pass
    try:
        search.append(Path(site.getusersitepackages()))
    except Exception:
        pass
    for base in search:
        lib = base / "openseespylinux" / "lib"
        if lib.is_dir():
            cur = os.environ.get("LD_LIBRARY_PATH", "")
            prefix = str(lib)
            if prefix not in cur.split(":"):
                os.environ["LD_LIBRARY_PATH"] = prefix + (":" + cur if cur else "")
            return lib
    return None


def worker_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    prepend_opensees_lib()
    if _SEISKIT.is_dir() and str(_SEISKIT) not in sys.path:
        sys.path.insert(0, str(_SEISKIT))


def cache_path(cache_dir: Path, domain: str, sample: int) -> Path:
    return Path(cache_dir) / f"{domain}_s{sample:04d}.npz"


def cache_matches(path: Path, spec: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        z = np.load(path)
    except (OSError, ValueError):
        return False
    keys = ("vs1", "H", "vs2", "xi_soil", "dt", "duration", "f1", "f2", "hx")
    for k in keys:
        if k not in z.files:
            return False
        if abs(float(z[k]) - float(spec[k])) > 1e-8:
            return False
    return all(
        n in z.files
        for n in ("freq", "af_within_iface", "af_within_packbase", "af_outcrop")
    )


def read_run_params(h5_path: Path) -> dict[str, float]:
    with h5py.File(h5_path, "r") as f:
        g = {k: f["grid"].attrs[k] for k in f["grid"].attrs}
        p = {k: f["params"].attrs[k] for k in f["params"].attrs}
    return {
        "dt_stored": float(g.get("dt", 0.02)),
        "dx": float(g.get("dx", 1.0)),
        "dz": float(g.get("dz", 1.0)),
        "motion_freq": float(g.get("motion_freq", 3.0)),
        "duration": float(p.get("duration", 30.0)),
        "damping_freq_first": float(p.get("damping_freq_first", float("nan"))),
        "bedrock_thickness": float(g.get("bedrock_thickness", 10.0)),
        "f0_effective": float(p.get("f0_effective", float("nan"))),
    }


def haskell_nominal_af_both(
    freq: np.ndarray,
    *,
    vs1: float,
    H: float,
    vs2: float,
    xi: float,
    rho: float = config.RHO,
) -> tuple[np.ndarray, np.ndarray]:
    """Hysteretic Haskell AF_within and AF_outcrop on ``freq``."""
    from seiskit.theory import Layer, RockHalfspace, layered_transfer_function

    _, within, outcrop = layered_transfer_function(
        np.asarray(freq, dtype=float),
        [Layer(float(H), float(vs1), float(rho), float(xi))],
        RockHalfspace(float(vs2), float(rho), 0.0),
    )
    return np.asarray(within, dtype=np.float64), np.asarray(outcrop, dtype=np.float64)


def interp_af(freq_src: np.ndarray, af: np.ndarray, freq_dst: np.ndarray) -> np.ndarray:
    src = np.asarray(freq_src, dtype=float).ravel()
    dst = np.asarray(freq_dst, dtype=float).ravel()
    y = np.asarray(af, dtype=float).ravel()
    if src.size == dst.size and np.allclose(src, dst, rtol=0, atol=1e-6):
        return y.astype(np.float64)
    return np.interp(dst, src, y, left=float("nan"), right=float("nan"))


def geomean_af(stack: np.ndarray) -> np.ndarray:
    x = np.asarray(stack, dtype=np.float64)
    x = np.clip(x, _EPS, None)
    return np.exp(np.nanmean(np.log(x), axis=0))


def mode_windows(f0: float, n_modes: int = N_MODES) -> list[tuple[float, float]]:
    return [((2 * (k - 1)) * f0, (2 * k) * f0) for k in range(1, n_modes + 1)]


def score_pair(
    freq: np.ndarray,
    ref: np.ndarray,
    cand: np.ndarray,
    *,
    f0: float,
) -> dict[str, float]:
    out = method_vs_reference(freq=freq, af_ref=ref, af_cand=cand)
    modes_r = odd_quarter_wave_peaks(freq, ref, f0=f0, n_modes=N_MODES)
    modes_c = odd_quarter_wave_peaks(freq, cand, f0=f0, n_modes=N_MODES)
    f = np.asarray(freq, dtype=float)
    for k, ((fr, ar), (fc, ac)) in enumerate(zip(modes_r, modes_c), start=1):
        out[f"f_mode{k}"] = fc
        out[f"A_mode{k}"] = ac
        out[f"ref_f_mode{k}"] = fr
        out[f"ref_A_mode{k}"] = ar
        if np.isfinite(ac) and np.isfinite(ar) and ar > 0 and ac > 0:
            out[f"delta_ln_A_mode{k}"] = float(np.log(ac / ar))
        else:
            out[f"delta_ln_A_mode{k}"] = float("nan")
        out[f"delta_f_mode{k}"] = (
            float(fc - fr) if np.isfinite(fc) and np.isfinite(fr) else float("nan")
        )
        lo, hi = mode_windows(f0)[k - 1]
        mask = (f >= lo) & (f <= min(hi, 10.0))
        out[f"pearson_mode{k}"] = float(pearson(cand, ref, mask=mask))
    return out


def _load_center_accels(run_dir: Path) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    files = sorted(run_dir.glob("center_node_y*_dof1_accel.txt"))
    if len(files) < 2:
        raise FileNotFoundError(f"Need ≥2 center recorders in {run_dir}")

    def y_of(p: Path) -> float:
        stem = p.name.split("_dof")[0]
        return float(stem.replace("center_node_y", ""))

    by_y: dict[float, np.ndarray] = {}
    t = None
    for p in files:
        arr = np.loadtxt(p)
        by_y[y_of(p)] = np.asarray(arr[:, 1], dtype=float)
        t = np.asarray(arr[:, 0], dtype=float)
    assert t is not None
    return t, by_y


def _nearest_y(
    by_y: dict[float, np.ndarray], target: float
) -> tuple[float, np.ndarray]:
    y = min(by_y, key=lambda v: abs(v - target))
    return y, by_y[y]


def _silence_fds() -> tuple[int, int]:
    """Mute C++ OpenSees chatter (ld.so-level stdout/stderr)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_out, old_err = os.dup(1), os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)
    return old_out, old_err


def _restore_fds(old_out: int, old_err: int) -> None:
    os.dup2(old_out, 1)
    os.dup2(old_err, 2)
    os.close(old_out)
    os.close(old_err)


def run_one_column(spec: dict[str, Any]) -> dict[str, Any]:
    """Run one 1-D OpenSees column; write TF cache. Spawn-safe."""
    worker_init()
    dest = Path(spec["cache_path"])
    if not spec.get("force") and cache_matches(dest, spec):
        return {
            "ok": True,
            "cache": str(dest),
            "skipped": True,
            "sample": spec["sample"],
            "domain": spec["domain"],
        }

    from seiskit.analysis import run_opensees_analysis
    from seiskit.builder import build_model_data
    from seiskit.config import AnalysisConfig
    from seiskit.ttf.TTF import TTF
    from seiskit.utils import compute_ricker

    hx = float(spec["hx"])
    rock_buffer = float(spec["rock_buffer"])
    n_soil = max(1, int(round(float(spec["H"]) / hx)))
    n_rock = max(1, int(round(rock_buffer / hx)))
    vs_rows = [float(spec["vs1"])] * n_soil + [float(spec["vs2"])] * n_rock
    vs = np.asarray(vs_rows, dtype=float).reshape(-1, 1)
    rho = np.full_like(vs, float(spec["rho"]))
    nu = np.full_like(vs, 0.3)
    ly = float(len(vs_rows) * hx)
    interface_y = float(n_rock * hx)
    f1 = float(spec["f1"])
    f2 = float(spec["f2"])
    dt = float(spec["dt"])
    duration = float(spec["duration"])
    motion_freq = float(spec["motion_freq"])

    cfg = AnalysisConfig(
        Ly=ly,
        Lx=hx,
        hx=hx,
        dt=dt,
        duration=duration,
        motion_freq=motion_freq,
        motion_t_shift=float(spec["motion_t_shift"]),
        damping_method="uniform_soil_only",
        damping_zeta=float(spec["xi_soil"]),
        damping_freqs=(f1, f2),
        boundary_condition_type="1D",
        record_center_nodes=True,
        center_node_y_positions=[PACK_BASE_Y, interface_y, ly],
        record_all_surface_nodes=False,
        element_type="4node",
        solver_type="UmfPack",
    )
    run_id = f"{spec['domain']}_s{int(spec['sample']):04d}"
    raw_root = Path(spec["raw_dir"])
    raw_root.mkdir(parents=True, exist_ok=True)
    model = build_model_data(
        cfg, vs, rho, nu, bedrock_mask=(vs >= float(spec["vs2"]) * 0.99)
    )
    run_dir = raw_root / run_id
    try:
        old_fds = _silence_fds()
        try:
            run_opensees_analysis(cfg, model, run_id=run_id, output_dir=str(raw_root))
        finally:
            _restore_fds(*old_fds)
        t, by_y = _load_center_accels(run_dir)
        dt_rec = float(t[1] - t[0]) if t.size > 1 else dt
        duration_eff = float(t[-1]) if t.size else duration
        _, surf = _nearest_y(by_y, ly)
        _, iface = _nearest_y(by_y, interface_y)
        _, packbase = _nearest_y(by_y, PACK_BASE_Y)
        freq_w, af_iface = TTF(
            surf,
            iface,
            dt=dt_rec,
            n_points=int(config.N_FREQ),
            Vsmin=None,
            dz=hx,
            smooth_coeff=int(config.SMOOTH_COEFF),
        )
        _, af_base = TTF(
            surf,
            packbase,
            dt=dt_rec,
            n_points=int(config.N_FREQ),
            Vsmin=None,
            dz=hx,
            smooth_coeff=int(config.SMOOTH_COEFF),
        )
        a_inc = compute_ricker(
            motion_freq, float(spec["motion_t_shift"]), duration_eff, dt
        )
        n = min(len(surf), len(a_inc))
        freq_o, af_out = TTF(
            surf[:n],
            2.0 * a_inc[:n],
            dt=dt_rec,
            n_points=int(config.N_FREQ),
            Vsmin=None,
            dz=hx,
            smooth_coeff=int(config.SMOOTH_COEFF),
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dest,
            freq=np.asarray(freq_w, dtype=np.float64),
            freq_outcrop=np.asarray(freq_o, dtype=np.float64),
            af_within_iface=np.asarray(af_iface, dtype=np.float64),
            af_within_packbase=np.asarray(af_base, dtype=np.float64),
            af_outcrop=np.asarray(af_out, dtype=np.float64),
            vs1=float(spec["vs1"]),
            H=float(spec["H"]),
            vs2=float(spec["vs2"]),
            xi_soil=float(spec["xi_soil"]),
            dt=dt,
            duration=duration,
            f1=f1,
            f2=f2,
            hx=hx,
            interface_y=interface_y,
            ly=ly,
        )
        return {
            "ok": True,
            "cache": str(dest),
            "skipped": False,
            "sample": spec["sample"],
            "domain": spec["domain"],
        }
    except Exception as exc:  # noqa: BLE001 — isolate OpenSees failures per case
        import traceback

        return {
            "ok": False,
            "cache": str(dest),
            "sample": spec["sample"],
            "domain": spec["domain"],
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    finally:
        if run_dir.is_dir():
            shutil.rmtree(run_dir, ignore_errors=True)


def collect_specs(
    domain: str,
    pack: dict[str, np.ndarray],
    *,
    cache_dir: Path,
    raw_dir: Path,
    force: bool,
    limit: int | None,
    dt: float,
) -> list[dict[str, Any]]:
    n = (
        int(pack["tf_opensees"].shape[0])
        if limit is None
        else min(limit, int(pack["tf_opensees"].shape[0]))
    )
    specs: list[dict[str, Any]] = []
    for i in range(n):
        h5 = resolve_h5(str(pack["h5_path"][i]), domain)
        soil_nz = (
            int(pack["soil_nz"][i])
            if "soil_nz" in pack
            else int(round(float(pack["H"][i])))
        )
        xi_soil, _ = read_soil_xi(h5, soil_nz)
        params = read_run_params(h5)
        vs1 = float(pack["vs1"][i])
        H = float(pack["H"][i])
        f0 = theoretical_f0(vs1, H)
        f1 = params["damping_freq_first"]
        if not np.isfinite(f1):
            f1 = (
                min(f0, params["motion_freq"])
                if np.isfinite(f0)
                else params["motion_freq"]
            )
        spec = {
            "domain": domain,
            "sample": i,
            "cache_path": str(cache_path(cache_dir, domain, i)),
            "raw_dir": str(raw_dir),
            "force": bool(force),
            "vs1": vs1,
            "H": H,
            "vs2": float(pack["vs2"][i]),
            "xi_soil": float(xi_soil),
            "dt": float(dt),
            "duration": float(params["duration"]),
            "motion_freq": float(params["motion_freq"]),
            "motion_t_shift": MOTION_T_SHIFT,
            "f1": float(f1),
            "f2": DAMPING_F2,
            "hx": float(params["dz"]),
            "rock_buffer": float(params["bedrock_thickness"]),
            "rho": float(config.RHO),
        }
        specs.append(spec)
    return specs


def run_specs(specs: list[dict[str, Any]], *, n_workers: int) -> list[dict[str, Any]]:
    todo = [
        s
        for s in specs
        if s.get("force") or not cache_matches(Path(s["cache_path"]), s)
    ]
    done = [
        {
            "ok": True,
            "cache": s["cache_path"],
            "skipped": True,
            "sample": s["sample"],
            "domain": s["domain"],
        }
        for s in specs
        if s not in todo
    ]
    if not todo:
        return done
    prepend_opensees_lib()
    # Always spawn: LD_LIBRARY_PATH is only honored by ld.so at process start.
    n_workers = max(1, int(n_workers))
    ctx = get_context("spawn")
    out: list[dict[str, Any]] = list(done)
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=worker_init,
    ) as pool:
        futs = {pool.submit(run_one_column, s): s for s in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            out.append(fut.result())
            if i == 1 or i % 10 == 0 or i == len(futs):
                n_ok = sum(1 for r in out if r.get("ok"))
                print(f"[opensees1d] {i}/{len(futs)} finished ({n_ok} ok)", flush=True)
    return out


def _load_cached_tf(
    spec: dict[str, Any], freq: np.ndarray
) -> dict[str, np.ndarray] | None:
    path = Path(spec["cache_path"])
    if not cache_matches(path, spec):
        return None
    z = np.load(path)
    freq_w = np.asarray(z["freq"], dtype=float)
    freq_o = (
        np.asarray(z["freq_outcrop"], dtype=float)
        if "freq_outcrop" in z.files
        else freq_w
    )
    return {
        "within_iface": interp_af(freq_w, z["af_within_iface"], freq),
        "within_packbase": interp_af(freq_w, z["af_within_packbase"], freq),
        "outcrop": interp_af(freq_o, z["af_outcrop"], freq),
    }


def score_domain(
    domain: str,
    pack: dict[str, np.ndarray],
    specs: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    freq = np.asarray(pack["freq"], dtype=float)
    ops = np.asarray(pack["tf_opensees"], dtype=np.float64)
    n = len(specs)
    n_f = freq.size
    haskell_w = np.full((n, n_f), np.nan)
    haskell_o = np.full((n, n_f), np.nan)
    os_iface = np.full((n, n_f), np.nan)
    os_base = np.full((n, n_f), np.nan)
    os_out = np.full((n, n_f), np.nan)
    ops_c = np.full((n, n_f), np.nan)
    rows: list[dict[str, Any]] = []
    n_missing = 0
    for i, spec in enumerate(specs):
        vs1, H, vs2 = spec["vs1"], spec["H"], spec["vs2"]
        xi = spec["xi_soil"]
        f0 = theoretical_f0(vs1, H)
        hw, ho = haskell_nominal_af_both(freq, vs1=vs1, H=H, vs2=vs2, xi=xi)
        haskell_w[i], haskell_o[i] = hw, ho
        ops_c[i] = _as_central(ops[i])
        cached = _load_cached_tf(spec, freq)
        shared = {
            "domain": domain,
            "sample": spec["sample"],
            "vs1": vs1,
            "H": H,
            "vs2": vs2,
            "f0": f0,
            "xi_soil": xi,
            "f1": spec["f1"],
        }
        if cached is None:
            n_missing += 1
            continue
        os_iface[i] = cached["within_iface"]
        os_base[i] = cached["within_packbase"]
        os_out[i] = cached["outcrop"]
        pairs = (
            (ARM_HASKELL_VS_2D, ops_c[i], hw),
            (ARM_OS1D_VS_2D, ops_c[i], os_base[i]),
            (ARM_SHAPE_WITHIN, hw, os_iface[i]),
            (ARM_SHAPE_OUTCROP, ho, os_out[i]),
        )
        for method, ref, cand in pairs:
            met = score_pair(freq, ref, cand, f0=f0)
            rows.append({**shared, "method": method, **met})
    df = pd.DataFrame(rows)
    rec: dict[str, Any] = {
        "n": n,
        "n_scored": int(n - n_missing),
        "n_missing": n_missing,
        "dt": DT_ANALYSIS,
        "damping_f2": DAMPING_F2,
        "methods": {},
    }
    for method in SCORE_ARMS:
        sub = df[df["method"] == method] if not df.empty else df
        block: dict[str, Any] = {
            "pearson": median_iqr(sub["pearson"].to_numpy())
            if not sub.empty
            else median_iqr(np.array([])),
            "anderson": median_iqr(sub["gof_af"].to_numpy())
            if not sub.empty
            else median_iqr(np.array([])),
            "delta_ln_A_peak": median_iqr(sub["delta_ln_A_peak"].to_numpy())
            if not sub.empty
            else median_iqr(np.array([])),
        }
        for k in range(1, N_MODES + 1):
            col = f"delta_ln_A_mode{k}"
            pcol = f"pearson_mode{k}"
            block[col] = (
                median_iqr(sub[col].to_numpy())
                if not sub.empty and col in sub
                else median_iqr(np.array([]))
            )
            block[pcol] = (
                median_iqr(sub[pcol].to_numpy())
                if not sub.empty and pcol in sub
                else median_iqr(np.array([]))
            )
        rec["methods"][method] = block
    arrays = {
        "freq": freq,
        "tf_haskell_within": haskell_w,
        "tf_haskell_outcrop": haskell_o,
        "tf_os1d_within_iface": os_iface,
        "tf_os1d_within_packbase": os_base,
        "tf_os1d_outcrop": os_out,
        "tf_opensees2d": ops_c,
        "vs1": np.asarray([s["vs1"] for s in specs], dtype=float),
        "H": np.asarray([s["H"] for s in specs], dtype=float),
        "xi_soil": np.asarray([s["xi_soil"] for s in specs], dtype=float),
    }
    return df, rec, arrays


def write_markdown(agg: dict[str, Any], dest: Path) -> None:
    lines = [
        "# OpenSees 1-D vs Haskell ξ_soil vs OpenSees 2-D",
        "",
        "Third arm after `HASKELL_XI.md`: a **literal 1-D OpenSees column** "
        "(Rayleigh, not hysteretic Haskell). Isolates damping *model shape* "
        "the same way that note isolated the ξ *value*.",
        "",
        'Setup: `boundary_condition_type="1D"` simple-shear column, '
        "`uniform_soil_only` with ζ = H5 soil-mean `Damping_zeta`, Rayleigh "
        "matched at `(damping_freq_first, 10 Hz)` — the 2-D data-gen convention, "
        "not the 1-D validation `(f_0, 3f_0)`. Analysis `dt = 0.01` s (2-D "
        "solver step; stored H5 recorders are often 0.02 s), `hx = 1` m, "
        "`motion_t_shift = 0.5` s, 10 m rock buffer. **Anderson** is the same "
        "Gaussian-weighted $L_1$ on $\\ln|\\mathrm{TF}|$ as in `HASKELL_XI.md`.",
        "",
        "Two within motions: **interface** is the soil–rock contact (Haskell "
        "`AF_within`); **pack base** is the $y=2$ m recorder used to build the "
        "OpenSees 2-D pack TFs. Outcrop is $|\\mathrm{FAS}_{\\mathrm{surf}}/"
        "(2 a_{\\mathrm{incident}})|$. Mode $k$ is the peak in "
        "$[2(k-1)f_0,\\,2k f_0]$ clipped to 0.1–10 Hz.",
        "",
        "| Domain | n | Arm | Pearson | Anderson | "
        "$\\Delta\\ln A_1$ | $\\Delta\\ln A_2$ | $\\Delta\\ln A_3$ |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for domain in SCORE_DOMAINS:
        rec = agg.get(domain)
        if not rec:
            continue
        for method in SCORE_ARMS:
            m = rec["methods"][method]
            lines.append(
                f"| {domain} | {rec['n_scored']} | {method} | "
                f"{fmt_iqr(m['pearson'])} | {fmt_iqr(m['anderson'])} | "
                f"{fmt_iqr(m['delta_ln_A_mode1'])} | {fmt_iqr(m['delta_ln_A_mode2'])} | "
                f"{fmt_iqr(m['delta_ln_A_mode3'])} |"
            )
    lines += [
        "",
        "## Reading",
        "",
        "- **Haskell vs OpenSees 1-D, within, mode 1** is *not* the "
        r"$\Delta\ln A \approx -0.22$ from the 1-D validation column. "
        "That column matched Rayleigh at $(f_0, 3f_0)$. Here $f_2=10$ Hz "
        "(the 2-D data-gen convention), so mode 1 sits near a match "
        "frequency and the two 1-D models agree "
        r"(Pearson $\sim 0.98$, $\Delta\ln A_1 \sim -0.02$). "
        "Rayleigh sags between the two match frequencies, so OpenSees 1-D "
        "is *taller* than hysteretic Haskell at modes 2–3.",
        "- **Outcrop** is a different story: OpenSees 1-D peaks fall well "
        "below Haskell (mode-1 $\\Delta\\ln A$ of order $-0.7$), matching "
        "the attached 1-D validation outcrop panel. Pack 2-D TFs are "
        "*within* motion, so outcrop numerics do not enter the 0.77 / 0.69 "
        "Pearson numbers.",
        "- **OpenSees 1-D vs OpenSees 2-D** (pack-base within) is the "
        "leftover after matching the FE / Rayleigh scheme. That is the "
        "fraction that can still be attributed to genuine spatial content "
        "before GINO. 2-D itself uses `global_avg` on a GRF, so this 1-D "
        "arm (uniform soil ζ) is not a bit-identical twin. Residual GINO "
        "is not retrained.",
        "",
    ]
    for domain in SCORE_DOMAINS:
        rec = agg.get(domain)
        if not rec:
            continue
        shape = rec["methods"][ARM_SHAPE_WITHIN]
        outc = rec["methods"][ARM_SHAPE_OUTCROP]
        vs2d = rec["methods"][ARM_OS1D_VS_2D]
        h2d = rec["methods"][ARM_HASKELL_VS_2D]
        lines.append(
            f"- **{domain}:** Haskell vs 1-D OpenSees Pearson (within) "
            f"{fmt_iqr(shape['pearson'])}, mode-1 $\\Delta\\ln A$ "
            f"{fmt_iqr(shape['delta_ln_A_mode1'])} "
            f"(outcrop Pearson {fmt_iqr(outc['pearson'])}, "
            f"$\\Delta\\ln A_1$ {fmt_iqr(outc['delta_ln_A_mode1'])}). "
            f"OpenSees 1-D vs 2-D Pearson {fmt_iqr(vs2d['pearson'])} "
            f"vs Haskell vs 2-D {fmt_iqr(h2d['pearson'])}."
            + (
                " Dipping 2-D high-$f$ rise is shared with OpenSees 1-D and "
                "absent from Haskell — that is most of the 0.67→0.84 Pearson lift."
                if domain == "dipping"
                else ""
            )
        )
    dest.write_text("\n".join(lines) + "\n")


def plot_opensees1d(
    agg: dict[str, Any],
    arrays_by_domain: dict[str, dict[str, np.ndarray]],
    dest: Path,
) -> None:
    apply_nature_style()
    fig, axes = plt_axes()
    letters = "abcd"
    for col, domain in enumerate(SCORE_DOMAINS):
        arr = arrays_by_domain.get(domain)
        rec = agg.get(domain)
        ax = axes[0, col]
        if arr is None:
            ax.set_visible(False)
            continue
        freq = arr["freq"]
        ax.semilogx(
            freq,
            geomean_af(arr["tf_opensees2d"]),
            color="#000000",
            lw=1.2,
            label="OpenSees 2-D",
        )
        ax.semilogx(
            freq,
            geomean_af(arr["tf_os1d_within_packbase"]),
            color="#0072B2",
            lw=1.1,
            label="OpenSees 1-D (pack base)",
        )
        ax.semilogx(
            freq,
            geomean_af(arr["tf_haskell_within"]),
            color="#999999",
            ls="-.",
            lw=1.1,
            label="Haskell ξ_soil",
        )
        ax.set_xlim(0.1, 10)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("|TF| within (geomean)")
        ax.set_title(DOMAIN_SPECS[domain]["title"])
        ax.legend(loc="upper right")
        panel_letter(ax, letters[col])

        axb = axes[1, col]
        if rec is None:
            continue
        x = np.arange(1, N_MODES + 1)
        width = 0.25
        series = [
            (ARM_SHAPE_WITHIN, "#0072B2", "OS 1-D − Haskell"),
            (ARM_OS1D_VS_2D, "#D55E00", "OS 1-D − 2-D"),
            (ARM_SHAPE_OUTCROP, "#009E73", "OS 1-D − Haskell outcrop"),
        ]
        for j, (method, color, _lab) in enumerate(series):
            meds = [
                rec["methods"][method][f"delta_ln_A_mode{k}"]["median"]
                for k in range(1, N_MODES + 1)
            ]
            axb.bar(
                x + (j - 1) * width,
                meds,
                width=width,
                color=color,
                label=_lab if col == 0 else None,
            )
        axb.axhline(0.0, color="0.6", lw=0.6)
        axb.set_xticks(x)
        axb.set_xticklabels(["1", "2", "3"])
        axb.set_xlabel("Mode")
        axb.set_ylabel(r"median $\Delta\ln A$")
        panel_letter(axb, letters[col + 2])
        if col == 0:
            axb.legend(loc="best")
    fig.tight_layout()
    savefig(fig, dest)


def plt_axes():
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=figsize("double", height_mm=120))
    return fig, axes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack-dir", type=Path, default=PACK_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    p.add_argument(
        "--domains", nargs="+", default=list(SCORE_DOMAINS), choices=list(DOMAIN_SPECS)
    )
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dt", type=float, default=DT_ANALYSIS)
    p.add_argument("--skip-opensees", action="store_true", help="Score from cache only")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.cache_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    agg: dict[str, Any] = {}
    arrays_by_domain: dict[str, dict[str, np.ndarray]] = {}
    run_reports: list[dict[str, Any]] = []
    for domain in args.domains:
        src = pack_path(args.pack_dir, domain)
        if not src.is_file():
            raise FileNotFoundError(src)
        pack = load_pack(src)
        specs = collect_specs(
            domain,
            pack,
            cache_dir=args.cache_dir,
            raw_dir=raw_dir,
            force=args.force,
            limit=args.limit,
            dt=args.dt,
        )
        if not args.skip_opensees:
            run_reports.extend(run_specs(specs, n_workers=args.n_workers))
        df, rec, arrays = score_domain(domain, pack, specs)
        rec["n_failed"] = sum(
            1
            for r in run_reports
            if r.get("domain") == domain and not r.get("ok", True)
        )
        frames.append(df)
        agg[domain] = rec
        arrays_by_domain[domain] = arrays
        np.savez_compressed(args.out_dir / f"{domain}_opensees1d.npz", **arrays)

    csv_path = args.out_dir / "opensees1d_vs_haskell.csv"
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(csv_path, index=False)
    json_path = args.out_dir / "opensees1d_vs_haskell_summary.json"
    json_path.write_text(json.dumps(jsonable(agg), indent=2))
    write_markdown(agg, args.out_dir / "OPENSEES1D.md")
    plot_opensees1d(agg, arrays_by_domain, args.out_dir / "opensees1d_vs_haskell.png")
    n_fail = sum(1 for r in run_reports if not r.get("ok", True))
    if n_fail:
        err_path = args.out_dir / "opensees1d_failures.json"
        err_path.write_text(
            json.dumps([r for r in run_reports if not r.get("ok", True)], indent=2)
        )
        print(f"OpenSees failures: {n_fail} (see {err_path})", flush=True)
    print(json.dumps(jsonable(agg), indent=2), flush=True)
    print(f"Wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
