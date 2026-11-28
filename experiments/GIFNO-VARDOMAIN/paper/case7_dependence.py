#!/usr/bin/env python3
"""Agreement of Case 7 log |TF| with OpenSees, beyond Pearson.

Pearson, Spearman, and Lin's concordance use every frequency in the band.
Distance correlation and mutual information use 256 evenly spaced frequencies
in that band, the same indices for every model. Each number is the mean across
stations. Concordance is Lin's rho_c on log |TF|, so a shift off the
45-degree line lowers it even when the shape correlation stays high.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_selection import mutual_info_regression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import paper.p128n128_case7 as c
from case7_crops import CROPS

ROOT = HERE.parents[2] / "checkpoints" / "vardomain"
MODELS = {
    "stage2": ROOT / "early_width" / "stage2_rise50",
    "P128": ROOT / "v2_ablations" / "P128",
    "P128s1": ROOT / "v2_ablations" / "P128s1",
    "P128n128": ROOT / "v2_ablations" / "P128n128",
}


def _ccc(x: np.ndarray, y: np.ndarray) -> float:
    vx, vy = x.var(), y.var()
    cov = ((x - x.mean()) * (y - y.mean())).mean()
    return float(2.0 * cov / (vx + vy + (x.mean() - y.mean()) ** 2 + 1e-12))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return c.pr._pearson(rx, ry)


def _dcor(x: np.ndarray, y: np.ndarray) -> float:
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    a = a - a.mean(axis=1, keepdims=True) - a.mean(axis=0, keepdims=True) + a.mean()
    b = b - b.mean(axis=1, keepdims=True) - b.mean(axis=0, keepdims=True) + b.mean()
    cov = float((a * b).mean())
    vx = float((a * a).mean())
    vy = float((b * b).mean())
    if vx <= 0.0 or vy <= 0.0 or cov <= 0.0:
        return 0.0
    return float(np.sqrt(cov / np.sqrt(vx * vy)))


def _nmi(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    joint, _, _ = np.histogram2d(x, y, bins=bins)
    p = joint / max(joint.sum(), 1.0)
    px, py = p.sum(1), p.sum(0)
    def ent(q):
        q = q[q > 0]
        return float(-(q * np.log(q)).sum())
    hx, hy = ent(px), ent(py)
    hxy = ent(p.ravel())
    mi = max(hx + hy - hxy, 0.0)
    denom = np.sqrt(hx * hy)
    return float(mi / denom) if denom > 0 else 0.0


def _station_means(pred: np.ndarray, truth: np.ndarray, idx: np.ndarray) -> dict:
    pearson, spearman, ccc, dcor, mi, nmi = [], [], [], [], [], []
    for i in range(pred.shape[0]):
        x = pred[i, idx]
        y = truth[i, idx]
        thin = idx[:: max(idx.size // 256, 1)][:256]
        xs, ys = pred[i, thin], truth[i, thin]
        pearson.append(c.pr._pearson(x, y))
        spearman.append(_spearman(x, y))
        ccc.append(_ccc(x, y))
        dcor.append(_dcor(xs, ys))
        mi.append(float(mutual_info_regression(xs.reshape(-1, 1), ys, random_state=0, n_neighbors=3)[0]))
        nmi.append(_nmi(xs, ys))
    return {k: float(np.mean(v)) for k, v in {
        "pearson": pearson, "spearman": spearman, "ccc": ccc, "dcor": dcor, "mi_nats": mi, "nmi": nmi,
    }.items()}


def score_model(model, device) -> dict:
    out = {}
    for crop in CROPS:
        name = crop["name"]
        out[name] = {}
        for width in (500, 1000, 1500, 2000):
            if width == 2000:
                mesh = c.pr._window_from_full(float(crop["center_x_m"]), width)
                sensor = mesh.pop("sensor")
            else:
                path = c.pr._crop_path(crop["sample_id"], width)
                if path is None:
                    continue
                mesh = c.pr._read_mesh(path)
                sensor = mesh["vs"].shape[1] // 2
            freq, truth, tf1d, pred, _sensor = c._forward(model, device, mesh, sensor)
            log_t = np.log(np.maximum(truth, 1e-6))
            log_p = np.log(np.maximum(pred, 1e-6))
            log_d = np.log(np.maximum(tf1d, 1e-6))
            bands = {}
            for label, lo, hi in (("mid", 0.4, 2.5), ("full", 0.1, 10.0)):
                idx = np.flatnonzero((freq >= lo) & (freq <= hi))
                bands[label] = {
                    "model": _station_means(log_p, log_t, idx),
                    "column": _station_means(log_d, log_t, idx),
                    "n_freq": int(idx.size),
                    "n_stations": int(pred.shape[0]),
                }
            out[name][str(width)] = bands
            mid = bands["mid"]["model"]
            print(
                f"{name:12} w{width} pearson {mid['pearson']:.3f} spearman {mid['spearman']:.3f} "
                f"ccc {mid['ccc']:.3f} dcor {mid['dcor']:.3f} mi {mid['mi_nats']:.3f} nmi {mid['nmi']:.3f}",
                flush=True,
            )
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)
    got = {}
    for label, path in MODELS.items():
        print(label, flush=True)
        model = c.load_fp32(path, device)
        got[label] = score_model(model, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    dest = HERE / "p128n128" / "case7_dependence.json"
    dest.write_text(json.dumps(got, indent=2))
    print("wrote", dest, flush=True)


if __name__ == "__main__":
    main()
