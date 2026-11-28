#!/usr/bin/env python3
"""Absolute lateral RMS of log Vs, and L1 of the seed-to-seed residual."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from data import Cache
from encoding import NX_METRIC

CACHE = Path("/home/ubuntu/surrogate-seismic-waves/data/vardomain/cache")
POD = Path("/home/ubuntu/surrogate-seismic-waves/checkpoints/vardomain/v2_ablations/A")
SOIL_VS = 1500.0


def band_var(field: np.ndarray) -> dict[str, float]:
    centered = field - field.mean(axis=1, keepdims=True)
    spec = np.fft.rfft(centered, axis=1)
    n = field.shape[1]
    k = np.arange(spec.shape[1])
    wavelength = np.full(k.shape, np.inf)
    wavelength[1:] = n / k[1:]
    weight = np.full(k.shape, 2.0)
    weight[0] = 0.0
    if n % 2 == 0:
        weight[-1] = 1.0
    mode = (weight * np.abs(spec) ** 2) / (n * n)
    bands = {
        "<10": wavelength < 10.0,
        "10-20": (wavelength >= 10.0) & (wavelength < 20.0),
        "20-42": (wavelength >= 20.0) & (wavelength < NX_METRIC / 96.0),
        ">42": wavelength >= NX_METRIC / 96.0,
    }
    out = {name: float(mode[:, mask].sum(axis=1).mean()) for name, mask in bands.items()}
    out["lateral"] = float(np.mean(centered ** 2))
    return out


def main() -> None:
    cache = Cache(CACHE)
    mean = np.load(POD / "pca_mean.npy").astype(np.float64)
    components = np.load(POD / "pca_components.npy").astype(np.float64)
    height = int(components.shape[1] // 2)
    freq = np.asarray(cache.freq, dtype=np.float64)
    high = (freq >= 2.5) & (freq <= 10.0)
    acc: dict[str, float] = defaultdict(float)
    n_sec = 0
    sq = row_sq = cells = 0.0
    for j in cache.section_indices("val"):
        sec = cache.sections[int(j)]
        vs = np.asarray(sec.vs, dtype=np.float64)
        zeta = np.asarray(sec.zeta, dtype=np.float64)
        rows = min(int(vs.shape[0]), height)
        soil = np.median(vs[:rows], axis=1) < SOIL_VS
        if not np.any(soil):
            soil = np.ones(rows, dtype=bool)
        log_vs = np.log(np.maximum(vs[:rows], 1e-6) / 400.0)
        log_zeta = np.log(np.maximum(zeta[:rows], 1e-8) / 0.02)
        feat = np.broadcast_to(mean, (vs.shape[1], mean.shape[0])).copy()
        feat[:, 0 : 2 * rows : 2] = log_vs.T
        feat[:, 1 : 2 * rows : 2] = log_zeta.T
        hat = (mean + ((feat - mean) @ components.T) @ components)[:, 0 : 2 * rows : 2].T
        true = log_vs[soil]
        recon = hat[soil]
        for key, value in band_var(true).items():
            acc["t_" + key] += value
        for key, value in band_var(true - recon).items():
            acc["e_" + key] += value
        sq += float(np.sum((true - recon) ** 2))
        row_sq += float(np.sum((true.mean(1) - recon.mean(1)) ** 2) * true.shape[1])
        cells += true.size
        n_sec += 1
    print("sections", n_sec)
    print("logVs RMSE", round(float(np.sqrt(sq / cells)), 4))
    print("row-mean RMSE", round(float(np.sqrt(row_sq / cells)), 4))
    for key in ("<10", "10-20", "20-42", ">42", "lateral"):
        true_rms = float(np.sqrt(acc["t_" + key] / n_sec))
        err_rms = float(np.sqrt(acc["e_" + key] / n_sec))
        print(key, "true_rms", round(true_rms, 4), "error_rms", round(err_rms, 4))

    def l1(groups: dict) -> None:
        devs, mags = [], []
        for curves in groups.values():
            shapes = {c.shape for c in curves}
            if len(curves) < 2 or len(shapes) != 1:
                continue
            stack = np.stack(curves, axis=0)
            devs.append(float(np.mean(np.abs(stack - stack.mean(axis=0)))))
            mags.append(float(np.mean(np.abs(stack[..., high]))))
            devs[-1] = float(np.mean(np.abs(stack[..., high] - stack[..., high].mean(axis=0))))
        if not devs:
            print("  none")
            return
        print("  groups", len(devs), "L1_dev", round(float(np.median(devs)), 4), "mean_abs_g", round(float(np.median(mags)), 4))

    for family in ("iid", "ood_dipping", "ood_three_layer"):
        grouped: dict[int, list] = defaultdict(list)
        for flat in cache.runs("val", (family,)):
            camp, j = cache.campaign(int(flat))
            g = np.asarray(camp.log_tf[j], dtype=np.float64) - np.asarray(camp.log_tf1d[j], dtype=np.float64)
            grouped[int(camp.group[j])].append(g)
        print(family, "val high-band")
        l1(grouped)
    for split in ("val", "train"):
        grouped: dict[tuple, list] = defaultdict(list)
        for j in cache.section_indices(split):
            sec = cache.sections[int(j)]
            g = np.asarray(sec.log_tf, dtype=np.float64) - np.asarray(sec.log_tf1d, dtype=np.float64)
            x = np.asarray(sec.x_m, dtype=np.float64)
            keep = (x >= 5.0) & (x <= sec.lx - 5.0) & np.isclose(np.mod(x, 20.0), 0.0, atol=0.2)
            if int(np.sum(keep)) < 2:
                continue
            grouped[(int(sec.group), int(sec.lx))].append(g[keep])
        print("manifold", split, "high-band")
        l1(grouped)


if __name__ == "__main__":
    main()
