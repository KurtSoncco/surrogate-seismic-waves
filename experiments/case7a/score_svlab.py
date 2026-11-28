#!/usr/bin/env python3
"""Score 1D theoretical (and LOGLO if checkpoints exist) against SVLAB ground truth.

Writes named folders under experiments/case7a/results/. Legends are
``SVLAB ground truth`` and ``1D theoretical``; never ``reference`` / ``ref/input``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from case7a import (
    DEFAULT_CACHE,
    DEFAULT_SVLAB_DIR,
    NODES_EACH_SIDE,
    REPO,
    ZONES,
    ensure_cache,
    install_svlab,
    paint_section_1m,
    window_center_x,
    window_node_index,
    zone_of,
)
from labels import CORE_BAND_HZ, TRUSTED_BAND_HZ, band_mask
from run_case7a import OUT_DIR, outcrop_reference_transfer_function
from section_1d import section_tf_1d

LOW_BAND_HZ = (0.1, 0.4)
BANDS = {"full": (0.1, 10.0), "trusted": TRUSTED_BAND_HZ, "low": LOW_BAND_HZ, "core": CORE_BAND_HZ}
LABEL_GT = "SVLAB ground truth"
LABEL_1D = "1D theoretical"
LABEL_REL = "LOGLO 500 m (relative Vs)"
LABEL_RES = "LOGLO residual (1D prior)"
YLABEL = "|TF| (surface / input)"
FOUR_X_M = (-1500.0, 0.0, 495.0, 1605.0)
EPS = 1e-6

RELATIVE_CKPT = REPO / "checkpoints" / "tier2_pod64_full7680" / "best_model.pt"
RESIDUAL_DIRS = (
    REPO / "checkpoints" / "case7_generalization" / "residual" / "pretrain_s0",
    REPO / "checkpoints" / "loglo_abs_residual_cpu_s0",
    REPO / "checkpoints" / "loglo_abs_residual_cpu_s1",
    REPO / "checkpoints" / "loglo_abs_residual_s0",
    REPO / "checkpoints" / "loglo_abs_residual_s1",
)
MIX_HOLD = (
    REPO / "data" / "gifno_screen" / "outcrop_tf" / "residual" / "mix_pretrain" / "holdout"
)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel() - np.mean(a)
    b = np.asarray(b, dtype=np.float64).ravel() - np.mean(b)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / den) if den > 0 else float("nan")


def _rel_l2(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(pred - true) / max(np.linalg.norm(true), 1e-12))


def _log_rms(pred: np.ndarray, true: np.ndarray) -> float:
    lp = np.log(np.maximum(np.asarray(pred, dtype=np.float64), EPS))
    lt = np.log(np.maximum(np.asarray(true, dtype=np.float64), EPS))
    return float(np.sqrt(np.mean((lp - lt) ** 2)))


def _band_metrics(pred: np.ndarray, true: np.ndarray, freq: np.ndarray) -> dict:
    out = {}
    for name, rng in BANDS.items():
        m = band_mask(freq, rng)
        out[name] = {
            "pearson": _pearson(pred[m], true[m]),
            "rel_l2": _rel_l2(pred[m], true[m]),
            "log_rms": _log_rms(pred[m], true[m]),
        }
    return out


def _first_peak_hz(freq: np.ndarray, mag: np.ndarray, band: tuple[float, float] = TRUSTED_BAND_HZ) -> float:
    m = band_mask(freq, band)
    return float(freq[m][int(np.argmax(np.asarray(mag)[m]))])


def zone_scores(x: np.ndarray, pred: np.ndarray, true: np.ndarray, freq: np.ndarray) -> dict:
    zones = zone_of(x)
    out: dict = {}
    for zone in ZONES:
        rows = np.flatnonzero(zones == zone)
        out[zone] = {"n_windows": int(rows.size)}
        for band, rng in BANDS.items():
            m = band_mask(freq, rng)
            p = [_pearson(pred[i, m], true[i, m]) for i in rows]
            r = [_rel_l2(pred[i, m], true[i, m]) for i in rows]
            g = [_log_rms(pred[i, m], true[i, m]) for i in rows]
            out[zone][band] = {
                "pearson_median": float(np.median(p)),
                "rel_l2_median": float(np.median(r)),
                "log_rms_median": float(np.median(g)),
            }
    return out


def wipe_results(root: Path) -> None:
    if not root.is_dir():
        return
    for child in root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _write_skip(folder: Path, reason: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    reason = " ".join(reason.split())
    if len(reason) > 240:
        reason = reason[:237] + "..."
    (folder / "metrics.json").write_text(json.dumps({"skipped": reason}, indent=2) + "\n")


def _ylim_in_band(freq: np.ndarray, series: list[np.ndarray], band: tuple[float, float] = TRUSTED_BAND_HZ) -> tuple[float, float]:
    m = band_mask(freq, band)
    hi = max((float(np.max(s[m])) for s in series), default=1.0)
    return 0.0, max(1.05 * hi, 1.0)


def plot_center_x0(freq: np.ndarray, tf_gt: np.ndarray, tf_1d: np.ndarray, extra: list, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.2), constrained_layout=True)
    ax.semilogx(freq, tf_gt, color="k", lw=1.6, label=LABEL_GT)
    ax.semilogx(freq, tf_1d, color="C3", lw=1.4, label=LABEL_1D)
    curves = [tf_gt, tf_1d]
    for series, color, label in extra:
        ax.semilogx(freq, series, color=color, lw=1.2, label=label)
        curves.append(series)
    ax.set_xlim(*TRUSTED_BAND_HZ)
    ax.set_ylim(*_ylim_in_band(freq, curves))
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel(YLABEL)
    ax.set_title("Case 7, x = 0 m, 0.1–2.5 Hz")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_four_x(
    freq: np.ndarray,
    x: np.ndarray,
    tf_gt: np.ndarray,
    tf_1d: np.ndarray,
    extra: list,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), sharex=True, constrained_layout=True)
    for ax, x_m in zip(axes.ravel(), FOUR_X_M):
        i = int(np.argmin(np.abs(x - x_m)))
        ax.semilogx(freq, tf_gt[i], color="k", lw=1.4, label=LABEL_GT)
        ax.semilogx(freq, tf_1d[i], color="C3", lw=1.2, label=LABEL_1D)
        curves = [tf_gt[i], tf_1d[i]]
        for series, color, label in extra:
            ax.semilogx(freq, series[i], color=color, lw=1.1, label=label)
            curves.append(series[i])
        ax.set_xlim(*TRUSTED_BAND_HZ)
        ax.set_ylim(*_ylim_in_band(freq, curves))
        ax.set_title(f"Case 7, x = {x[i]:.0f} m, 0.1–2.5 Hz")
        ax.set_ylabel(YLABEL)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].set_xlabel("frequency (Hz)")
    axes[1, 1].set_xlabel("frequency (Hz)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pearson_vs_x(x: np.ndarray, freq: np.ndarray, pred: np.ndarray, true: np.ndarray, extra: list, out_path: Path) -> None:
    m = band_mask(freq, TRUSTED_BAND_HZ)
    fig, ax = plt.subplots(figsize=(9.0, 3.8), constrained_layout=True)
    ax.plot(x, [_pearson(pred[i, m], true[i, m]) for i in range(len(x))], color="C3", lw=1.4, label=LABEL_1D)
    for series, color, label in extra:
        ax.plot(x, [_pearson(series[i, m], true[i, m]) for i in range(len(x))], color=color, lw=1.2, label=label)
    for name, (lo, hi) in ZONES.items():
        ax.axvspan(max(lo, float(x[0])), min(hi, float(x[-1])), alpha=0.06 if name != "geologic" else 0.14, color="0.3")
    ax.set_ylim(-0.2, 1.02)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Pearson r")
    title = "Case 7, 1D theoretical vs SVLAB ground truth, 0.1–2.5 Hz"
    if extra:
        title = "Case 7 vs SVLAB ground truth, 0.1–2.5 Hz"
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_window_median(freq: np.ndarray, tf_gt: np.ndarray, pred: np.ndarray, pred_label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.2), constrained_layout=True)
    ax.fill_between(
        freq,
        np.percentile(tf_gt, 16, axis=0),
        np.percentile(tf_gt, 84, axis=0),
        color="k",
        alpha=0.12,
        linewidth=0,
    )
    med_gt = np.median(tf_gt, axis=0)
    med_pred = np.median(pred, axis=0)
    ax.semilogx(freq, med_gt, color="k", lw=1.5, label=LABEL_GT)
    ax.semilogx(freq, med_pred, color="C0", lw=1.3, label=pred_label)
    ax.set_xlim(*TRUSTED_BAND_HZ)
    ax.set_ylim(*_ylim_in_band(freq, [med_gt, med_pred, np.percentile(tf_gt, 84, axis=0)]))
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel(YLABEL)
    ax.set_title("median over 21 stations in the center 500 m window (−250 to 250 m)")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _index_at(x: np.ndarray, value: float) -> int:
    hits = np.flatnonzero(np.isclose(x, value))
    if hits.size != 1:
        raise RuntimeError(f"expected one station at x={value}, got {hits.size}")
    return int(hits[0])


def score_1d(pack: dict) -> dict:
    x = window_center_x()
    nodes = np.array([window_node_index(float(v)) for v in x], dtype=np.int64)
    freq, tf_gt, scale = outcrop_reference_transfer_function(pack, nodes)
    tf_1d = section_tf_1d(freq, pack["vs"], x, "outcrop")
    i0 = _index_at(x, 0.0)
    metrics = {
        "ground_truth": LABEL_GT,
        "model": LABEL_1D,
        "input_scale": float(scale),
        "calibration": "median(SVLAB / 1D theoretical) over flat_east, 0.4-2.0 Hz",
        "dt_s": float(pack["dt"]),
        "n_time": int(pack["surface_h"].shape[0]),
        "center_x0": {
            **_band_metrics(tf_1d[i0], tf_gt[i0], freq),
            "first_peak_hz_svlab": _first_peak_hz(freq, tf_gt[i0]),
            "first_peak_hz_1d": _first_peak_hz(freq, tf_1d[i0]),
        },
        "zones": zone_scores(x, tf_1d, tf_gt, freq),
        "four_x_m": list(FOUR_X_M),
    }
    return {
        "x": x,
        "freq": freq,
        "tf_gt": tf_gt,
        "tf_1d": tf_1d,
        "input_scale": float(scale),
        "metrics": metrics,
        "i0": i0,
    }


def write_1d(res: dict, out_dir: Path, extra: list | None = None) -> None:
    extra = extra or []
    out_dir.mkdir(parents=True, exist_ok=True)
    i0 = res["i0"]
    plot_center_x0(res["freq"], res["tf_gt"][i0], res["tf_1d"][i0], [(s[i0], c, n) for s, c, n in extra], out_dir / "center_x0_tf.png")
    plot_four_x(res["freq"], res["x"], res["tf_gt"], res["tf_1d"], extra, out_dir / "four_x_tf.png")
    plot_pearson_vs_x(res["x"], res["freq"], res["tf_1d"], res["tf_gt"], extra, out_dir / "pearson_vs_x.png")
    (out_dir / "metrics.json").write_text(json.dumps(res["metrics"], indent=2) + "\n")


def try_residual(pack: dict, res_1d: dict, out_dir: Path, model_dirs: list[Path] | None = None) -> None:
    model_dirs = [d for d in (model_dirs or RESIDUAL_DIRS) if (d / "best_model.pt").is_file()]
    if not model_dirs:
        _write_skip(out_dir, "checkpoint missing")
        return
    try:
        from score_section import load_models, predict_windows

        models, device, config, target = load_models(model_dirs)
        log_w = predict_windows(models, device, config, paint_section_1m(pack["vs"]), res_1d["x"])
    except (ImportError, FileNotFoundError, RuntimeError, ValueError) as exc:
        _write_skip(out_dir, f"{type(exc).__name__}: {exc}")
        return
    tf1d = np.asarray(res_1d["tf_1d"], dtype=np.float64)
    center_i = log_w.shape[2] // 2 if getattr(models[0].head, "shared", False) else NODES_EACH_SIDE
    log_center = log_w.mean(axis=0)[:, center_i]
    if target == "residual":
        log_center = log_center + np.log(np.maximum(tf1d, EPS))
    tf_pred = np.exp(log_center)
    i0 = res_1d["i0"]
    metrics = {
        "ground_truth": LABEL_GT,
        "model": LABEL_RES,
        "target": target,
        "models": [str(d) for d in model_dirs],
        "center_x0": _band_metrics(tf_pred[i0], res_1d["tf_gt"][i0], res_1d["freq"]),
        "zones": zone_scores(res_1d["x"], tf_pred, res_1d["tf_gt"], res_1d["freq"]),
    }
    extra = [(tf_pred, "C0", LABEL_RES)]
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_center_x0(
        res_1d["freq"],
        res_1d["tf_gt"][i0],
        res_1d["tf_1d"][i0],
        [(tf_pred[i0], "C0", LABEL_RES)],
        out_dir / "center_x0_tf.png",
    )
    plot_pearson_vs_x(res_1d["x"], res_1d["freq"], res_1d["tf_1d"], res_1d["tf_gt"], extra, out_dir / "pearson_vs_x.png")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def _reload_relative_config() -> None:
    """Drop residual GIFNO_* so the 4-channel 500 m checkpoint can load."""
    for key in list(os.environ):
        if key.startswith("GIFNO_") or key == "TF_TARGET":
            del os.environ[key]
    os.environ.update(
        {
            "GIFNO_LATENT_CHANNELS": "128",
            "GIFNO_POD_NUM_MODES": "64",
            "GIFNO_NUM_FNO_LAYERS": "5",
            "GIFNO_DEEPONET_LATENT_DIM": "128",
            "GIFNO_INPUT_ENCODING": "relative",
            "GIFNO_DIMNORM_FILM": "false",
            "GIFNO_PRIOR_CHANNELS": "0",
            "GIFNO_NZ_MAX": "128",
            "GIFNO_POD_SPACE": "linear",
            "GIFNO_MODEL_DIR": str(RELATIVE_CKPT.parent),
            "GIFNO_WINDOW_SHIFT_MAX": "0",
            "GIFNO_WINDOW_CONSISTENCY_WEIGHT": "0",
            "GIFNO_DATA_ROOT": str(REPO / "data" / "gifno_screen"),
        }
    )
    for name in ("config", "model"):
        sys.modules.pop(name, None)


def try_relative(pack: dict, res_1d: dict, out_dir: Path) -> None:
    if not RELATIVE_CKPT.is_file():
        _write_skip(out_dir, "checkpoint missing")
        return
    pod_dir = RELATIVE_CKPT.parent
    if not (pod_dir / "pod_modes.npy").is_file() or not (pod_dir / "pod_mean.npy").is_file():
        _write_skip(out_dir, "pod_modes.npy missing next to tier2_pod64_full7680/best_model.pt")
        return
    _reload_relative_config()
    try:
        from run_case7a import build_model_input, predict_transfer_function, station_x_m

        pred = predict_transfer_function(
            build_model_input(pack["vs"]),
            RELATIVE_CKPT,
            pod_modes_path=pod_dir / "pod_modes.npy",
            pod_mean_path=pod_dir / "pod_mean.npy",
        )
    except (ImportError, FileNotFoundError, RuntimeError, ValueError) as exc:
        _write_skip(out_dir, f"{type(exc).__name__}: {exc}")
        return
    x_st = station_x_m()
    freq, tf_gt, _ = outcrop_reference_transfer_function(pack, nodes)
    tf_1d = section_tf_1d(freq, pack["vs"], x_st, "outcrop")
    i0 = int(np.flatnonzero(np.isclose(x_st, 0.0))[0])
    metrics = {
        "ground_truth": LABEL_GT,
        "model": LABEL_REL,
        "checkpoint": str(RELATIVE_CKPT),
        "center_x0": _band_metrics(pred[i0], tf_gt[i0], freq),
        "window_median_note": "median over 21 stations in the center 500 m window (−250 to 250 m)",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_center_x0(freq, tf_gt[i0], tf_1d[i0], [(pred[i0], "C0", LABEL_REL)], out_dir / "center_x0_tf.png")
    plot_window_median(freq, tf_gt, pred, LABEL_REL, out_dir / "window_median_tf.png")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def try_campaigns(out_dir: Path, model_dirs: list[Path] | None = None) -> None:
    model_dirs = [d for d in (model_dirs or RESIDUAL_DIRS) if (d / "best_model.pt").is_file()]
    if not MIX_HOLD.is_dir():
        _write_skip(out_dir, "mix holdout H5s missing")
        return
    if not model_dirs:
        _write_skip(out_dir, "checkpoint missing")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "score_campaigns.py"),
        "--data",
        str(MIX_HOLD),
        "--label",
        "loglo_residual",
        "--max-per-campaign",
        "24",
        "--out",
        str(out_dir / "metrics.json"),
    ]
    for d in model_dirs:
        cmd.extend(["--model-dir", str(d)])
    try:
        subprocess.check_call(cmd, cwd=str(Path(__file__).resolve().parent))
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        _write_skip(out_dir, f"{type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--no-wipe", action="store_true")
    parser.add_argument(
        "--model-dir",
        type=Path,
        action="append",
        default=None,
        help="Residual checkpoint dir (repeatable). Default: first existing RESIDUAL_DIRS.",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Score residual and Bet B on SVLAB (zones, low band, conventions, spatial gate).",
    )
    parser.add_argument("--diagnose-out", type=Path, default=None)
    args = parser.parse_args()
    if args.diagnose:
        from diagnose_svlab import run

        out = args.diagnose_out or (REPO / "checkpoints" / "case7_generalization" / "svlab_diagnosis")
        run(out)
        return
    model_dirs = [p.resolve() for p in args.model_dir] if args.model_dir else None

    if not DEFAULT_CACHE.is_file() and not (DEFAULT_SVLAB_DIR / "Surface.0.out").is_file():
        install_svlab()
    if not args.no_wipe:
        wipe_results(args.out_dir)
    pack = ensure_cache(force=False)
    res = score_1d(pack)
    write_1d(res, args.out_dir / "1d_vs_svlab")
    try_residual(pack, res, args.out_dir / "loglo_residual", model_dirs)
    try_relative(pack, res, args.out_dir / "loglo_relative_500m")
    try_campaigns(args.out_dir / "campaigns", model_dirs)

    center = res["metrics"]["center_x0"]
    print(json.dumps({"input_scale": res["metrics"]["input_scale"], "center_x0": center}, indent=2))
    print("zones trusted")
    for zone, s in res["metrics"]["zones"].items():
        t = s["trusted"]
        print(
            f"  {zone:10s} P {t['pearson_median']:.3f} L2 {t['rel_l2_median']:.3f} "
            f"logRMS {t['log_rms_median']:.3f}"
        )


if __name__ == "__main__":
    main()
