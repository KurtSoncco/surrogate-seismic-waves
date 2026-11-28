#!/usr/bin/env python3
"""SVLAB zone metrics for the stage-1 metric residual.

Stations and bands match ``case7a/score_svlab.py``. Each station is the center
of a 500 m window cut from the Case 7 Q–Vs mesh, with the damping OpenSees
applied. The column prior uses that same zeta, and the network adds the residual.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CASE7A = HERE.parent / "case7a"
sys.path.insert(0, str(CASE7A))
sys.path.insert(0, str(HERE))

from applied_damping import applied_zeta  # noqa: E402
from case7_crops import span_columns  # noqa: E402
from case7a import ensure_cache, window_center_x, window_node_index  # noqa: E402
from encoding import encode, query_columns  # noqa: E402
from evaluate import load_model  # noqa: E402
from provenance import full_stamp  # noqa: E402
from run_case7a import outcrop_reference_transfer_function  # noqa: E402
from score_svlab import _band_metrics, _pearson, zone_scores  # noqa: E402
from section_1d import RHO, rayleigh_xi_target, surface_over_outcrop  # noqa: E402

MODEL_DIR = REPO / "checkpoints" / "vardomain" / "early_width" / "stage1"
CACHE_FREQ = REPO / "data" / "vardomain" / "cache" / "freq.npy"
FULL_QVS = REPO / "data" / "case7a" / "opensees_crops" / "context" / "h5" / "full_qvs.h5"
OUT = HERE / "figures" / "stage1" / "svlab_metrics.json"
FIG = HERE / "figures" / "stage1" / "svlab_pearson_vs_x.png"
WIDTH = 500
BANDS = (("0.1–2.5 Hz", 0.1, 2.5), ("0.4–2.5 Hz", 0.4, 2.5))


def _column_tf(freq: np.ndarray, vs: np.ndarray, zeta: np.ndarray, f1: float, f2: float) -> np.ndarray:
    xi = np.stack([rayleigh_xi_target(freq, float(z), f1, f2) for z in zeta])
    return surface_over_outcrop(freq, vs, dz=1.0, xi=xi, rho=RHO)


def _along(freq: np.ndarray, pred: np.ndarray, true: np.ndarray, lo: float, hi: float) -> np.ndarray:
    band = (freq >= lo) & (freq <= hi)
    return np.array([_pearson(pred[i, band], true[i, band]) for i in range(pred.shape[0])])


def _plot(x: np.ndarray, curves: dict[str, dict[str, np.ndarray]], fig_path: Path, label: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.6), sharex=True, constrained_layout=True)
    colors = {label: "#C2410C", "Column": "#0072B2"}
    for ax, (title, _lo, _hi) in zip(axes, BANDS):
        for name, series in curves.items():
            ax.plot(x, series[title], color=colors[name], lw=1.2, label=name)
        ax.axvspan(-650, 1050, color="0.5", alpha=0.08, lw=0)
        ax.set_ylim(-0.2, 1.02)
        ax.set_ylabel("Pearson of |TF|")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("x (m), 500 m window inside the 4000 m section")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--fig", type=Path, default=FIG)
    ap.add_argument("--label", default="Stage 1")
    args = ap.parse_args()
    pack = ensure_cache(force=False)
    x = window_center_x()
    nodes = np.array([window_node_index(float(v)) for v in x], dtype=np.int64)
    freq_gt, tf_gt, scale = outcrop_reference_transfer_function(pack, nodes)
    freq_m = np.load(CACHE_FREQ)
    with h5py.File(FULL_QVS, "r") as handle:
        vs = np.asarray(handle["Vs_realization_2D"][:], dtype=np.float64)
        zeta = applied_zeta(handle)
        f1 = float(handle["params"].attrs["damping_freq_first"])
        f2 = float(handle["params"].attrs["damping_freq_second"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.model_dir, device)
    model.eval()
    stamp = full_stamp(args.model_dir, device, bool(cfg.get("amp", True)))
    query = torch.tensor(query_columns("metric", np.array([WIDTH / 2.0]), WIDTH), dtype=torch.float32).view(1, 1)
    pred = np.empty((x.size, freq_gt.size), dtype=np.float64)
    prior = np.empty_like(pred)
    with torch.no_grad():
        for i, center in enumerate(x):
            c0, n, _shift = span_columns(float(center), WIDTH)
            vs_w = vs[:, c0 : c0 + n]
            zeta_w = zeta[:, c0 : c0 + n]
            mid = n // 2
            tf1d = _column_tf(freq_m, vs_w[:, mid], zeta_w[:, mid], f1, f2)
            enc = torch.from_numpy(encode("metric", vs_w, zeta_w, vs_w.shape[0], f1_hz=f1, f2_hz=f2)).unsqueeze(0)
            g = model(enc.to(device), query.to(device)).float().cpu().numpy()[0, 0]
            prior[i] = np.interp(freq_gt, freq_m, tf1d)
            pred[i] = np.interp(freq_gt, freq_m, tf1d * np.exp(g))
            if i % 40 == 0:
                print(f"station {i}/{x.size} g {float(g.mean()):+.3f}", flush=True)
    i0 = int(np.flatnonzero(np.isclose(x, 0.0))[0])
    along = {
        args.label: {title: _along(freq_gt, pred, tf_gt, lo, hi) for title, lo, hi in BANDS},
        "Column": {title: _along(freq_gt, prior, tf_gt, lo, hi) for title, lo, hi in BANDS},
    }
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    _plot(x, along, args.fig, args.label)
    metrics = {
        "ground_truth": "SVLAB ground truth",
        "model": args.label,
        "checkpoint": str(args.model_dir),
        "provenance": stamp,
        "mesh": str(FULL_QVS),
        "target": cfg.get("target"),
        "window_m": WIDTH,
        "f1_hz": f1,
        "f2_hz": f2,
        "input_scale": float(scale),
        "n_stations": int(x.size),
        "center_x0": _band_metrics(pred[i0], tf_gt[i0], freq_gt),
        "center_x0_1d": _band_metrics(prior[i0], tf_gt[i0], freq_gt),
        "zones": zone_scores(x, pred, tf_gt, freq_gt),
        "zones_1d": zone_scores(x, prior, tf_gt, freq_gt),
        "x_m": x.tolist(),
        "pearson_vs_x": {
            name: {title: series[title].tolist() for title, _lo, _hi in BANDS} for name, series in along.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)
    print(f"wrote {args.fig}", flush=True)
    print(f"x from {x[0]:.0f} to {x[-1]:.0f} m, stations whose 500 m window stays inside the section")
    print(f"{'zone':12} {'model 0.1-2.5':>12} {'1D 0.1-2.5':>12} {'model 0.4-2.5':>12} {'1D 0.4-2.5':>12}")
    for zone in metrics["zones"]:
        m = metrics["zones"][zone]
        d = metrics["zones_1d"][zone]
        print(
            f"{zone:12} {m['trusted']['pearson_median']:12.3f} {d['trusted']['pearson_median']:12.3f} "
            f"{m['core']['pearson_median']:12.3f} {d['core']['pearson_median']:12.3f}"
        )


if __name__ == "__main__":
    main()
