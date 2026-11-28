#!/usr/bin/env python3
"""OpenSees sensitivity of IID validation fields.

The unperturbed rerun uses the stored H5 Vs, the campaign mesh, and the
1310.72 s outcrop window. It has to land within 1e-3 of the float32 fixed-window
label before the perturbations are trusted. Each field is then solved four
times: the rank-128 column PCA reconstruction, a 10 m lateral low-pass, a
42 m lateral low-pass, and 7.5% band-limited Vs noise between 5 m and 20 m.

    NUMBA_CACHE_DIR=/tmp/numba_cache PYTHONPATH=/home/kurt-/seiskit \
      python sensitivity_opensees.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SEISKIT = Path("/home/kurt-/seiskit")
CASE7A = REPO / "experiments" / "case7a"
GEN = SEISKIT / "neural-operator" / "data"
for path in (SEISKIT, CASE7A, GEN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
_venv = REPO / ".venv/lib/python3.11/site-packages"
if str(_venv) not in sys.path:
    sys.path.append(str(_venv))

import hdf5plugin  # noqa: E402,F401
import h5py  # noqa: E402

from seiskit.builder import build_model_data  # noqa: E402
from seiskit.config import AnalysisConfig  # noqa: E402
from seiskit.plot_results import get_damping_zeta_grid  # noqa: E402

import run_experiment as gen  # noqa: E402
from relabel_outcrop import DAMPING_FREQ_SECOND, N_LATERAL, relabel_one  # noqa: E402

H5_DIR = REPO / "data" / "gifno_screen" / "h5"
LABELS = REPO / "data" / "vardomain" / "outcrop_fixed_window" / "iid"
CACHE = REPO / "data" / "vardomain" / "cache" / "iid"
PCA_DIR = REPO / "checkpoints" / "vardomain" / "v2_ablations" / "P128n128"
OUT = REPO / "experiments" / "GIFNO-VARDOMAIN" / "paper" / "diagnostics" / "opensees"
WORK = Path("/tmp/opensees_sensitivity")
STRIP = (500, 1000)
VS_REF = 400.0
ZETA_REF = 0.02
WINDOW_S = 1310.72
SCALE = 1.982620120048523
BANDS = (("low", 0.1, 0.4), ("mid", 0.4, 2.5), ("high", 2.5, 10.0), ("full", 0.1, 10.0))
P128N_HIGH = 0.121


def _load_h5(run_index: int) -> dict:
    path = H5_DIR / f"run_{run_index}.h5"
    with h5py.File(path, "r") as handle:
        vs = np.asarray(handle["Vs_realization_2D"], dtype=np.float64)
        zeta = np.asarray(handle["Damping_zeta"], dtype=np.float64)
        grid = {k: handle["grid"].attrs[k] for k in handle["grid"].attrs}
        params = {k: handle["params"].attrs[k] for k in handle["params"].attrs}
    soil = int(params["soil_layer_count"])
    mask = np.zeros(vs.shape, dtype=bool)
    mask[soil:, :] = True
    return {"run_index": int(run_index), "vs": vs, "zeta": zeta, "grid": grid, "params": params, "mask": mask, "soil": soil}


def _extend(strip: np.ndarray, width: int) -> np.ndarray:
    left = np.repeat(strip[:, :1], STRIP[0], axis=1)
    right = np.repeat(strip[:, -1:], width - STRIP[1], axis=1)
    return np.concatenate([left, strip, right], axis=1)


def _lowpass(strip: np.ndarray, min_wavelength: float) -> np.ndarray:
    spec = np.fft.rfft(np.log(np.maximum(strip, 1e-6)), axis=1)
    freq = np.fft.rfftfreq(strip.shape[1], d=1.0)
    spec[:, freq > 1.0 / min_wavelength] = 0.0
    return np.exp(np.fft.irfft(spec, n=strip.shape[1], axis=1))


def _band_noise(strip: np.ndarray, seed: int) -> np.ndarray:
    """7.5% Vs, wavelengths 5 m to 20 m. That is the sub-resolution band."""
    rng = np.random.default_rng(seed)
    spec = np.fft.rfft(rng.normal(size=strip.shape), axis=1)
    freq = np.fft.rfftfreq(strip.shape[1], d=1.0)
    keep = (freq >= 1.0 / 20.0) & (freq <= 1.0 / 5.0)
    spec[:, ~keep] = 0.0
    noise = np.fft.irfft(spec, n=strip.shape[1], axis=1)
    noise = noise / max(float(noise.std()), 1e-12)
    return strip * np.exp(0.075 * noise)


def _pca_strip(vs: np.ndarray, zeta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.load(PCA_DIR / "pca_mean.npy").astype(np.float64)
    components = np.load(PCA_DIR / "pca_components.npy").astype(np.float64)
    height = int(components.shape[1] // 2)
    rows = min(height, vs.shape[0])
    feat = np.repeat(mean[None, :], vs.shape[1], axis=0)
    paired = np.empty((rows, vs.shape[1], 2), dtype=np.float64)
    paired[:, :, 0] = np.log(np.maximum(vs[:rows], 1e-6) / VS_REF)
    paired[:, :, 1] = np.log(np.maximum(zeta[:rows], 1e-8) / ZETA_REF)
    feat[:, : 2 * rows] = paired.transpose(1, 0, 2).reshape(vs.shape[1], 2 * rows)
    code = (feat - mean) @ components.T
    recon = code @ components + mean
    out_vs = vs.copy()
    out_zeta = zeta.copy()
    out_vs[:rows] = VS_REF * np.exp(recon[:, 0 : 2 * rows : 2].T)
    out_zeta[:rows] = ZETA_REF * np.exp(recon[:, 1 : 2 * rows : 2].T)
    return out_vs, out_zeta


def perturb(field: dict, kind: str) -> dict:
    vs = field["vs"].copy()
    zeta = field["zeta"].copy()
    lo, hi = STRIP
    if kind == "raw":
        return field
    if kind == "pca128":
        vs_s, zeta_s = _pca_strip(vs[:, lo:hi], zeta[:, lo:hi])
        vs = _extend(vs_s, vs.shape[1])
        zeta = _extend(zeta_s, zeta.shape[1])
    elif kind == "lp10":
        vs = _extend(_lowpass(vs[:, lo:hi], 10.0), vs.shape[1])
    elif kind == "lp42":
        vs = _extend(_lowpass(vs[:, lo:hi], 42.0), vs.shape[1])
    elif kind == "subres":
        vs = _extend(_band_noise(vs[:, lo:hi], seed=field["run_index"]), vs.shape[1])
    else:
        raise ValueError(kind)
    out = dict(field)
    out["vs"] = vs
    out["zeta"] = zeta
    return out


def _config(field: dict, dt: float) -> AnalysisConfig:
    grid, params = field["grid"], field["params"]
    return AnalysisConfig(
        Ly=float(grid["Lz"]), Lx=float(grid["Lx"]), hx=float(grid["dx"]), dt=float(dt),
        duration=float(params["duration"]), motion_freq=float(grid["motion_freq"]), motion_t_shift=0.5,
        damping_freqs=(float(params["damping_freq_first"]), DAMPING_FREQ_SECOND),
        damping_zeta=0.025, damping_method="global_avg", boundary_condition_type="2D",
        record_center_nodes=False, center_node_y_positions=[2.0, float(grid["Lz"])],
        record_lateral_span_at_center_depths=(10, 15.0), record_all_surface_nodes=False,
        element_type="4node", solver_type="Mumps", max_time_per_batch=8 * 3600.0,
    )


def _relabel(field: dict, recorder_dir: Path) -> dict:
    time_arr, data = gen._load_recorder_txt(recorder_dir)
    if data.ndim != 2 or data.shape[1] < 2 * N_LATERAL:
        raise RuntimeError(f"{recorder_dir}: recorder channels {getattr(data, 'shape', None)}")
    data = data[:, : 2 * N_LATERAL]
    path = recorder_dir / "relabel.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Vs_realization_2D", data=field["vs"].astype(np.float32))
        handle.create_dataset("Damping_zeta", data=field["zeta"].astype(np.float32))
        grid = handle.create_group("grid")
        for key, value in field["grid"].items():
            grid.attrs[key] = value
        params = handle.create_group("params")
        for key, value in field["params"].items():
            params.attrs[key] = value
        acc = handle.create_group("recorders").create_group("accel")
        acc.create_dataset("time", data=np.asarray(time_arr, dtype=np.float64))
        acc.create_dataset("data", data=np.asarray(data, dtype=np.float64))
    row = {"motion_freq": float(field["grid"]["motion_freq"]), "f0_effective": float(field["params"]["f0_effective"])}
    freq, tf_raw, tf1d = relabel_one((str(path), row, "velocity", WINDOW_S))
    path.unlink()
    tf = np.asarray(tf_raw, dtype=np.float64) / SCALE
    return {"freq": np.asarray(freq, dtype=np.float64), "tf": tf, "tf1d": np.asarray(tf1d, dtype=np.float64)}


def _bands(freq: np.ndarray, log_a: np.ndarray, log_b: np.ndarray) -> dict:
    out = {}
    for name, lo, hi in BANDS:
        mask = (freq >= lo) & (freq <= hi)
        out[name] = float(np.mean(np.abs(log_a[:, mask] - log_b[:, mask])))
    return out


def solve_one(job: tuple) -> dict:
    run_index, kind, dt, cache_row = job
    tag = f"run{run_index}_{kind}_dt{dt:.3f}"
    dest = OUT / f"{tag}.json"
    if dest.is_file():
        return json.loads(dest.read_text())
    field = perturb(_load_h5(run_index), kind)
    work = WORK / tag
    if work.exists():
        import shutil
        shutil.rmtree(work)
    work.mkdir(parents=True)
    config = _config(field, dt)
    rho = np.ones_like(field["vs"]) * 2000.0
    nu = np.ones_like(field["vs"]) * 0.3
    model = build_model_data(config, field["vs"], rho, nu, bedrock_mask=field["mask"])
    status = gen.run_opensees_analysis(config, model, tag, str(work))
    recorder = work / tag
    result = {"run_index": run_index, "kind": kind, "dt": dt, "cache_row": cache_row, "status": str(status)}
    if not recorder.is_dir():
        result["error"] = "no recorder directory"
        dest.write_text(json.dumps(result))
        return result
    labeled = _relabel(field, recorder)
    stored_freq = np.load(LABELS / "outcrop_freq.npy")
    if not np.allclose(labeled["freq"], stored_freq, rtol=1e-5, atol=1e-5):
        raise RuntimeError("relabel frequency grid does not match the fixed-window labels")
    log_hat = np.log(np.maximum(labeled["tf"], 1e-8))
    reference = np.log(np.maximum(np.load(LABELS / "tf_outcrop_per_sample.npy", mmap_mode="r")[cache_row], 1e-8))
    if log_hat.shape != reference.shape:
        raise RuntimeError(f"relabel {log_hat.shape} != label {reference.shape}")
    cached = np.asarray(np.load(CACHE / "log_tf.npy", mmap_mode="r")[cache_row], dtype=np.float64)
    result["versus_fixed_window"] = _bands(labeled["freq"], log_hat, reference)
    result["versus_cache_float16"] = _bands(labeled["freq"], log_hat, cached)
    result["max_abs_fixed"] = float(np.max(np.abs(log_hat - reference)))
    if kind != "raw":
        raw_tf = OUT / f"run{run_index}_raw_dt{dt:.3f}_logtf.npy"
        if raw_tf.is_file():
            result["versus_raw"] = _bands(labeled["freq"], log_hat, np.load(raw_tf))
    np.save(OUT / f"{tag}_logtf.npy", log_hat.astype(np.float32))
    dest.write_text(json.dumps(result))
    print(f"{tag} max {result['max_abs_fixed']:.4e} status {status}", flush=True)
    import shutil
    shutil.rmtree(work, ignore_errors=True)
    return result


def select_runs() -> list[dict]:
    split = np.load(CACHE / "split.npy").astype(str)
    sample = np.load(CACHE / "sample_id.npy")
    run_index = np.load(CACHE / "run_index.npy")
    vs = np.load(CACHE / "vs.npy", mmap_mode="r")
    n_active = np.load(CACHE / "n_active.npy")
    log_tf = np.load(CACHE / "log_tf.npy", mmap_mode="r")
    log_1d = np.load(CACHE / "log_tf1d.npy", mmap_mode="r")
    freq = np.load(REPO / "data" / "vardomain" / "cache" / "freq.npy")
    high = (freq >= 2.5) & (freq <= 10.0)
    val = np.flatnonzero(split == "val")
    soil = np.array([float(np.mean(vs[i, : int(n_active[i])][vs[i, : int(n_active[i])] < 600.0])) for i in val])
    gee = np.array([float(np.mean(np.abs(log_tf[i][:, high] - log_1d[i][:, high]))) for i in val])
    rows = []
    for local, row in enumerate(val):
        rows.append({
            "cache_row": int(row), "run_index": int(run_index[row]), "sample_id": int(sample[row]),
            "soil_vs": float(soil[local]), "high_abs_g": float(gee[local]),
        })
    soft_cut, stiff_cut = np.quantile(soil, [0.25, 0.75])
    g_med = float(np.median(gee))
    picks = []
    used: set[int] = set()

    def take(pool: list[dict], prefer_high: bool) -> None:
        pool = [r for r in pool if r["sample_id"] not in used]
        if not pool:
            return
        if prefer_high:
            chosen = max(pool, key=lambda r: r["high_abs_g"])
        else:
            chosen = min(pool, key=lambda r: abs(r["high_abs_g"] - g_med))
        used.add(chosen["sample_id"])
        picks.append(chosen)

    soft = [r for r in rows if r["soil_vs"] <= soft_cut]
    stiff = [r for r in rows if r["soil_vs"] >= stiff_cut]
    mid = [r for r in rows if soft_cut < r["soil_vs"] < stiff_cut]
    for pool in (soft, stiff, mid):
        take(pool, False)
        take(pool, True)
    return picks


def main() -> None:
    os.environ.setdefault("NUMBA_CACHE_DIR", str(REPO / ".numba_cache"))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    picks = select_runs()
    (OUT / "selection.json").write_text(json.dumps(picks, indent=2))
    print("selected", json.dumps(picks), flush=True)
    first = picks[0]
    field = _load_h5(first["run_index"])
    stored_dt = float(field["grid"]["dt"])
    repros = []
    trusted_dt = None
    for dt in (stored_dt, 0.01 if abs(stored_dt - 0.01) > 1e-6 else 0.02):
        repro = solve_one((first["run_index"], "raw", dt, first["cache_row"]))
        repros.append(repro)
        if repro.get("max_abs_fixed", 1.0) < 1e-3:
            trusted_dt = dt
            break
    (OUT / "repro.json").write_text(json.dumps({"tries": repros, "trusted_dt": trusted_dt, "p128n128_high_log_l1": P128N_HIGH}, indent=2))
    if trusted_dt is None:
        print("unperturbed rerun did not match the fixed-window label within 1e-3", flush=True)
        return
    print(f"repro matched at dt {trusted_dt}", flush=True)
    jobs = []
    for pick in picks:
        for kind in ("raw", "pca128", "lp10", "lp42", "subres"):
            if kind == "raw" and pick["run_index"] == first["run_index"]:
                continue
            jobs.append((pick["run_index"], kind, trusted_dt, pick["cache_row"]))
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
        list(pool.map(solve_one, jobs))
    summary = []
    for path in sorted(OUT.glob("run*_dt*.json")):
        summary.append(json.loads(path.read_text()))
    (OUT / "summary.json").write_text(json.dumps({
        "trusted_dt": trusted_dt, "p128n128_high_log_l1": P128N_HIGH, "runs": summary,
    }, indent=2))
    print("wrote summary.json", flush=True)


if __name__ == "__main__":
    main()
