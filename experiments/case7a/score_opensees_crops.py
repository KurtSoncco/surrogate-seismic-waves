#!/usr/bin/env python3
"""Score the residual LOGLO head on the five OpenSees Case 7 crops.

The head sees the inner 500 m and the Q–Vs zeta stored in the H5. Its prior
channels and the residual multiply-back use the training Rayleigh propagator
(column f0 and 10 Hz). The 1D curve uses the Rayleigh pair the OpenSees run
actually applied (crop f0 and 3 f0). Pearson is on log |TF| over 0.1–2.5 Hz.
A single input scale, fit on the flat west and flat east crops, removes the
Lysmer injection constant before the plots.

    python score_opensees_crops.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CASE_DIR = Path(__file__).resolve().parent
REPO = CASE_DIR.parents[1]
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

from case7_crops import CROPS, INNER_NX  # noqa: E402
from labels import CALIBRATION_BAND_HZ, TRUSTED_BAND_HZ, band_mask, calibrate_input_scale, seiskit_input_acceleration, tf_over_input  # noqa: E402
from score_svlab import _pearson, _ylim_in_band  # noqa: E402
from section_1d import RHO, rayleigh_xi_target, surface_over_outcrop  # noqa: E402

H5_DIR = REPO / "data" / "case7a" / "opensees_crops" / "h5"
MODEL_DIR = REPO / "checkpoints" / "case7_generalization" / "residual" / "pretrain_s0"
OUT_DIR = REPO / "checkpoints" / "case7_generalization" / "svlab_diagnosis" / "plots" / "opensees_crops"
CENTER_COL = INNER_NX // 2


def _ttf():
    from run_case7a import load_ttf_batch_fast

    return load_ttf_batch_fast()


def run_convention_tf(freq: np.ndarray, vs: np.ndarray, zeta: np.ndarray, f1: float, f2: float) -> np.ndarray:
    """|surface / outcrop| with the crop's OpenSees Rayleigh pair. Shape (nx, F)."""
    out = np.empty((vs.shape[1], freq.size), dtype=np.float64)
    for j in range(vs.shape[1]):
        xi = np.stack([rayleigh_xi_target(freq, float(z), f1, f2) for z in zeta[:, j]])
        out[j] = surface_over_outcrop(freq, vs[:, j], dz=1.0, xi=xi, rho=RHO)
    return out


def training_tf(freq: np.ndarray, vs: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    """Training residual denominator: per-column f0 and 10 Hz. Shape (nx, F)."""
    gifno = str(REPO / "experiments" / "GIFNO")
    if gifno not in sys.path:
        sys.path.append(gifno)
    from absolute_inputs import column_tf1d_rayleigh

    return column_tf1d_rayleigh(vs, zeta, 1.0, freq).T


def predict_window(model, device, config, vs: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    """Linear |TF| ratio from the shared head. Shape (nx, F)."""
    import torch

    gifno = str(REPO / "experiments" / "GIFNO")
    if gifno not in sys.path:
        sys.path.append(gifno)
    from absolute_inputs import encode_absolute

    encoded = encode_absolute(
        vs,
        zeta,
        vs.shape[0],
        config.NZ_MAX,
        n_prior=int(config.PRIOR_CHANNELS),
        prior_mode="rayleigh",
    )
    with torch.no_grad():
        grid = model(torch.from_numpy(encoded)[None].to(device)).cpu().numpy()
    grid = np.squeeze(grid, axis=0)
    if grid.shape != (vs.shape[1], grid.shape[-1]):
        raise ValueError(f"model output {grid.shape} is not ({vs.shape[1]}, F)")
    return grid


def trusted_log_pearson(pred: np.ndarray, true: np.ndarray, freq: np.ndarray) -> float:
    """One Pearson of log amplitude over every column and the trusted band."""
    m = band_mask(freq, TRUSTED_BAND_HZ)
    return _pearson(np.log(np.maximum(pred[:, m], 1e-12)), np.log(np.maximum(true[:, m], 1e-12)))


def pearson_along_x(pred: np.ndarray, true: np.ndarray, freq: np.ndarray) -> np.ndarray:
    m = band_mask(freq, TRUSTED_BAND_HZ)
    out = np.empty(pred.shape[0], dtype=np.float64)
    for j in range(pred.shape[0]):
        out[j] = _pearson(np.log(np.maximum(pred[j, m], 1e-12)), np.log(np.maximum(true[j, m], 1e-12)))
    return out


def rel_l1_along_x(pred: np.ndarray, true: np.ndarray, freq: np.ndarray) -> np.ndarray:
    """Per column, Σ|pred−true| / Σ|true| on the trusted band. Same formula as DeepONet ``rel_l1_rows``."""
    m = band_mask(freq, TRUSTED_BAND_HZ)
    num = np.sum(np.abs(pred[:, m] - true[:, m]), axis=1)
    den = np.sum(np.abs(true[:, m]), axis=1)
    return num / np.maximum(den, 1e-12)


def _attrs(handle, group: str) -> dict:
    out = {}
    for key, value in handle[group].attrs.items():
        if isinstance(value, bytes):
            value = value.decode()
        elif hasattr(value, "item"):
            value = value.item()
        out[key] = value
    return out


def load_run(path: Path) -> dict:
    import h5py

    with h5py.File(path, "r") as handle:
        time = handle["recorders/accel/time"][:].astype(np.float64)
        surface = handle["recorders/accel/surface"][:].astype(np.float64)
        vs = handle["Vs_realization_2D"][:].astype(np.float64)
        zeta = handle["Damping_zeta"][:].astype(np.float64)
        params = _attrs(handle, "params")
        grid = _attrs(handle, "grid")
    dt = float(time[1] - time[0])
    input_acc = seiskit_input_acceleration(
        float(grid["motion_freq"]), float(params["motion_t_shift"]), time, str(params.get("ricker_is", "acceleration"))
    )
    freq, tf_raw = tf_over_input(surface, input_acc, dt, _ttf())
    if tf_raw.shape != (vs.shape[1], freq.size):
        raise ValueError(f"surface TF {tf_raw.shape} != ({vs.shape[1]}, {freq.size})")
    f1, f2 = float(params["damping_freq_first"]), float(params["damping_freq_second"])
    tf1d = run_convention_tf(freq, vs, zeta, f1, f2)
    return {
        "path": path,
        "params": params,
        "freq": freq,
        "vs": vs,
        "zeta": zeta,
        "tf_raw": tf_raw,
        "tf1d": tf1d,
        "x_m": float(params["crop_x0_m"]) + np.arange(vs.shape[1]),
    }


def _plot(runs: list[dict], out_dir: Path) -> None:
    freq = runs[0]["freq"]
    fig, axes = plt.subplots(1, len(runs), figsize=(3.2 * len(runs), 3.6), sharey=True, constrained_layout=True)
    for ax, run in zip(np.atleast_1d(axes), runs):
        p = run["params"]
        series = [run["tf_ops"][CENTER_COL], run["tf1d"][CENTER_COL], run["tf_model"][CENTER_COL]]
        ax.semilogx(freq, series[0], color="k", lw=1.4, label="OpenSees")
        ax.semilogx(freq, series[1], color="C3", lw=1.2, label="1D")
        ax.semilogx(freq, series[2], color="C0", lw=1.2, label="LOGLO")
        ax.set_xlim(*TRUSTED_BAND_HZ)
        ax.set_ylim(*_ylim_in_band(freq, series))
        ax.set_title(f"{p['name']}\nx = {float(p['center_x_m']):.0f} m")
        ax.set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("|TF|")
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(out_dir / "tf_stations.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.0), constrained_layout=True)
    for run in runs:
        ax.plot(run["x_m"], run["pearson_x_model"], color="C0", lw=1.2)
        ax.plot(run["x_m"], run["pearson_x_1d"], color="C3", lw=1.0, alpha=0.85)
    ax.plot([], [], color="C0", label="LOGLO")
    ax.plot([], [], color="C3", label="1D")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Pearson of log |TF|, 0.1–2.5 Hz")
    ax.legend(frameon=False)
    fig.savefig(out_dir / "pearson_vs_x.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.0), constrained_layout=True)
    for run in runs:
        ax.plot(run["x_m"], run["rel_l1_x_model"], color="C0", lw=1.2)
        ax.plot(run["x_m"], run["rel_l1_x_1d"], color="C3", lw=1.0, alpha=0.85)
    ax.plot([], [], color="C0", label="LOGLO")
    ax.plot([], [], color="C3", label="1D")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("relative L1 of |TF|, 0.1–2.5 Hz")
    ax.legend(frameon=False)
    fig.savefig(out_dir / "rel_l1_vs_x.png", dpi=150)
    plt.close(fig)


def main() -> None:
    paths = [H5_DIR / f"{crop['sample_id']}.h5" for crop in CROPS]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise SystemExit("missing OpenSees crops:\n" + "\n".join(missing))

    from score_section import load_models

    models, device, config, target = load_models([MODEL_DIR])
    if target != "residual":
        raise SystemExit(f"expected the residual head, got TF_TARGET={target}")
    model = models[0]

    runs = [load_run(path) for path in paths]
    freq = runs[0]["freq"]
    flat = [r for r in runs if r["params"]["name"] in ("flat_west", "flat_east")]
    scale = calibrate_input_scale(
        np.concatenate([r["tf_raw"] for r in flat], axis=0),
        np.concatenate([r["tf1d"] for r in flat], axis=0),
        freq,
        CALIBRATION_BAND_HZ,
    )
    print(f"input scale c = {scale:.4f} from flat west and flat east", flush=True)

    metrics = {"input_scale": scale, "crops": {}}
    stored = {}
    for run in runs:
        run["tf_ops"] = run["tf_raw"] / scale
        ratio = predict_window(model, device, config, run["vs"], run["zeta"])
        if ratio.shape[-1] != freq.size:
            raise SystemExit(f"model frequencies {ratio.shape[-1]} != OpenSees TF {freq.size}")
        tf_train = training_tf(freq, run["vs"], run["zeta"])
        run["tf_model"] = np.exp(np.log(np.maximum(ratio, 1e-6)) + np.log(np.maximum(tf_train, 1e-6)))
        run["pearson_x_model"] = pearson_along_x(run["tf_model"], run["tf_ops"], freq)
        run["pearson_x_1d"] = pearson_along_x(run["tf1d"], run["tf_ops"], freq)
        run["rel_l1_x_model"] = rel_l1_along_x(run["tf_model"], run["tf_ops"], freq)
        run["rel_l1_x_1d"] = rel_l1_along_x(run["tf1d"], run["tf_ops"], freq)
        name = str(run["params"]["name"])
        row = {
            "sample_id": run["params"]["sample_id"],
            "center_x_m": float(run["params"]["center_x_m"]),
            "f0_hz": float(run["params"]["f0_hz"]),
            "rayleigh_hz": [float(run["params"]["rayleigh_f1_hz"]), float(run["params"]["rayleigh_f2_hz"])],
            "xi_soil": float(run["params"]["xi_soil"]),
            "xi_rock": float(run["params"]["xi_rock"]),
            "pearson_trusted_model": trusted_log_pearson(run["tf_model"], run["tf_ops"], freq),
            "pearson_trusted_1d": trusted_log_pearson(run["tf1d"], run["tf_ops"], freq),
            "pearson_center_model": float(run["pearson_x_model"][CENTER_COL]),
            "pearson_center_1d": float(run["pearson_x_1d"][CENTER_COL]),
            "rel_l1_trusted_model": float(np.mean(run["rel_l1_x_model"])),
            "rel_l1_trusted_1d": float(np.mean(run["rel_l1_x_1d"])),
            "rel_l1_center_model": float(run["rel_l1_x_model"][CENTER_COL]),
            "rel_l1_center_1d": float(run["rel_l1_x_1d"][CENTER_COL]),
        }
        metrics["crops"][name] = row
        print(
            f"{name}: model {row['pearson_trusted_model']:.3f}  1D {row['pearson_trusted_1d']:.3f}  "
            f"relL1 model {row['rel_l1_trusted_model']:.3f}  1D {row['rel_l1_trusted_1d']:.3f}",
            flush=True,
        )
        stored[name] = run

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(
        OUT_DIR / "tf_compare.npz",
        freq=freq,
        **{f"{name}_x": run["x_m"] for name, run in stored.items()},
        **{f"{name}_ops": run["tf_ops"].astype(np.float32) for name, run in stored.items()},
        **{f"{name}_1d": run["tf1d"].astype(np.float32) for name, run in stored.items()},
        **{f"{name}_model": run["tf_model"].astype(np.float32) for name, run in stored.items()},
    )
    _plot(list(stored.values()), OUT_DIR)
    print(f"wrote {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
