#!/usr/bin/env python3
"""Is 16 PCA components the limit of a linear column code, or of the regressor?

Same target and same ridge readout as depth_probe.py. The column is still
mean-imputed past n_active. Three extra inputs are scored on the validation
receivers:

- the raw mean-imputed column, the linear upper bound
- a 64-component masked PCA of that column
- the 16-component code concatenated with the four physical features

    python depth_bound.py --cache ... --out .../depth_probe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data import Cache
from depth_probe import (
    DZ_M,
    NEAR_SURFACE_M,
    PCA_RANK,
    POD_RANK,
    RIDGE,
    TIE_MARGIN,
    NormalEquations,
    curve_scores,
    fit_pod,
    masked_pca,
    pod_coefficients,
    reconstruct,
)
from pretrain_ae import VS_REF, ZETA_REF, _refuse_shipped

HIGH_RANK = 64
CHUNK = 256


def _runs(cache: Cache, split: str) -> tuple[np.ndarray, object]:
    flats = cache.runs(split, ("iid",))
    if flats.size == 0:
        raise RuntimeError(f"no {split} IID runs")
    camp, _j = cache.campaign(int(flats[0]))
    index = np.array([cache.campaign(int(flat))[1] for flat in flats], dtype=np.int64)
    return index, camp


def _paired(vs: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    """(N, H, W) soil arrays to (N*W, 2H), feature 2*row+channel, float32-rounded logs."""
    logged_vs = np.log(np.maximum(vs, 1e-6) / VS_REF).astype(np.float32).astype(np.float64)
    logged_zeta = np.log(np.maximum(zeta, 1e-8) / ZETA_REF).astype(np.float32).astype(np.float64)
    count, height, width = logged_vs.shape
    paired = np.empty((count * width, height * 2), dtype=np.float64)
    paired[:, 0::2] = logged_vs.transpose(0, 2, 1).reshape(count * width, height)
    paired[:, 1::2] = logged_zeta.transpose(0, 2, 1).reshape(count * width, height)
    return paired


def _physical(vs: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    """(N, H, W) to (N, W, 4) logs. Matches depth_probe.physical_features."""
    top = min(NEAR_SURFACE_M, vs.shape[1])
    travel = np.sum(DZ_M / np.maximum(vs, 1e-6), axis=1)
    raw = np.stack(
        [
            1.0 / (4.0 * travel),
            np.full(vs.shape[0::2], vs.shape[1] * DZ_M, dtype=np.float64),
            vs[:, :top].mean(axis=1),
            zeta.mean(axis=1),
        ],
        axis=-1,
    )
    return np.log(np.maximum(raw, 1e-8))


def accumulate(cache: Cache) -> tuple[dict, dict, int]:
    index, camp = _runs(cache, "train")
    height = int(camp.n_active[index].max())
    features = 2 * height
    cross = np.zeros((features, features), dtype=np.float64)
    cross_n = np.zeros((features, features), dtype=np.float64)
    total = np.zeros(features, dtype=np.float64)
    count = np.zeros(features, dtype=np.float64)
    freq = int(np.asarray(cache.freq).size)
    pod_cross = np.zeros((freq, freq), dtype=np.float64)
    pod_sum = np.zeros(freq, dtype=np.float64)
    pod_n = 0
    heights = camp.n_active[index]
    for soil in np.unique(heights):
        chosen = index[heights == soil]
        width = 2 * int(soil)
        for start in range(0, chosen.size, CHUNK):
            rows = chosen[start : start + CHUNK]
            paired = _paired(
                np.asarray(camp.vs[rows, :soil], dtype=np.float64),
                np.asarray(camp.zeta[rows, :soil], dtype=np.float64),
            )
            cross[:width, :width] += paired.T @ paired
            cross_n[:width, :width] += paired.shape[0]
            total[:width] += paired.sum(axis=0)
            count[:width] += paired.shape[0]
            curves = np.asarray(camp.log_tf1d[rows], dtype=np.float64).reshape(-1, freq)
            pod_cross += curves.T @ curves
            pod_sum += curves.sum(axis=0)
            pod_n += int(curves.shape[0])
        print(f"height {int(soil)} runs {chosen.size}", flush=True)
    pca = masked_pca(cross, cross_n, total, count, HIGH_RANK)
    pod = fit_pod(pod_cross, pod_sum, pod_n, POD_RANK)
    return pca, pod, pod_n


def _impute(paired: np.ndarray, mean: np.ndarray) -> np.ndarray:
    full = np.repeat(mean[None, :], paired.shape[0], axis=0)
    full[:, : paired.shape[1]] = paired
    return full


def _codes(paired: np.ndarray, physical: np.ndarray, pca16: dict, pca64: dict) -> dict[str, np.ndarray]:
    raw = _impute(paired, pca64["mean"])
    centered = raw - pca64["mean"]
    pca16_code = centered @ pca16["components"].T
    return {
        "pca16": pca16_code,
        "pca64": centered @ pca64["components"].T,
        "raw": raw,
        "combined": np.concatenate([pca16_code, physical.reshape(-1, 4)], axis=1),
    }


def _receiver_batch(camp, rows: np.ndarray, soil: int, cols: np.ndarray):
    vs = np.asarray(camp.vs[rows, :soil][:, :, cols], dtype=np.float64)
    zeta = np.asarray(camp.zeta[rows, :soil][:, :, cols], dtype=np.float64)
    return _paired(vs, zeta), _physical(vs, zeta)


def fit_readouts(cache: Cache, pca16: dict, pca64: dict, pod: dict) -> dict[str, NormalEquations]:
    index, camp = _runs(cache, "train")
    cols = np.asarray(camp.rec_x, dtype=np.int64)
    targets = int(pod["modes"].shape[0])
    widths = {"pca16": PCA_RANK, "pca64": HIGH_RANK, "raw": int(pca64["mean"].size), "combined": PCA_RANK + 4}
    eqs = {name: NormalEquations(width, targets) for name, width in widths.items()}
    freq = int(np.asarray(cache.freq).size)
    heights = camp.n_active[index]
    for soil in np.unique(heights):
        chosen = index[heights == soil]
        for start in range(0, chosen.size, CHUNK):
            rows = chosen[start : start + CHUNK]
            paired, physical = _receiver_batch(camp, rows, int(soil), cols)
            curves = np.asarray(camp.log_tf1d[rows], dtype=np.float64).reshape(-1, freq)
            coeff = pod_coefficients(curves, pod)
            codes = _codes(paired, physical, pca16, pca64)
            for name, eq in eqs.items():
                eq.add(codes[name], coeff)
    return eqs


def score(cache: Cache, pca16: dict, pca64: dict, pod: dict, equations: dict[str, NormalEquations], solutions: dict) -> dict:
    index, camp = _runs(cache, "val")
    cols = np.asarray(camp.rec_x, dtype=np.int64)
    freq = np.asarray(cache.freq, dtype=np.float64)
    buckets: dict[str, list[np.ndarray]] = {name: [] for name in ("mean", *solutions)}
    targets: list[np.ndarray] = []
    heights = camp.n_active[index]
    for soil in np.unique(heights):
        chosen = index[heights == soil]
        for start in range(0, chosen.size, CHUNK):
            rows = chosen[start : start + CHUNK]
            paired, physical = _receiver_batch(camp, rows, int(soil), cols)
            curves = np.asarray(camp.log_tf1d[rows], dtype=np.float64).reshape(-1, freq.size)
            codes = _codes(paired, physical, pca16, pca64)
            buckets["mean"].append(np.repeat(pod["mean"][None, :], curves.shape[0], axis=0))
            for name, solution in solutions.items():
                buckets[name].append(reconstruct(equations[name].predict(codes[name], solution), pod))
            targets.append(curves)
    target = np.concatenate(targets, axis=0)
    scored = {name: curve_scores(np.concatenate(rows, axis=0), target, freq) for name, rows in buckets.items()}
    scored["n_curves"] = int(target.shape[0])
    scored["n_runs"] = int(index.size)
    return scored


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    _refuse_shipped(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    cache = Cache(args.cache)
    pca64, pod, n_train = accumulate(cache)
    taken = min(PCA_RANK, pca64["rank"])
    pca16 = {"mean": pca64["mean"], "components": pca64["components"][:taken], "rank": taken}
    print(
        f"pca64 rank {pca64['rank']} explained {pca64['explained']:.3f} "
        f"negative {pca64['negative_eigenvalues']} pod {pod['explained']:.5f} curves {n_train}",
        flush=True,
    )
    equations = fit_readouts(cache, pca16, pca64, pod)
    solutions = {name: eq.solve(RIDGE) for name, eq in equations.items()}
    conds = {}
    for name, eq in equations.items():
        mean = eq.sum_x / eq.n
        centered = eq.sum_xx - eq.n * np.outer(mean, mean)
        scale = np.sqrt(np.maximum(np.diag(centered) / max(eq.n - 1, 1), 1e-12))
        conds[name] = float(np.linalg.cond(centered / scale[:, None] / scale[None, :]))
        print(f"readout {name} columns {eq.n} width {eq.sum_x.size} cond {conds[name]:.3e}", flush=True)
    val = score(cache, pca16, pca64, pod, equations, solutions)
    pca_mae = val["pca16"]["mae"]
    combined_mae = val["combined"]["mae"]
    margin = pca_mae * (1.0 - TIE_MARGIN)
    freeze = "pca16+physical" if combined_mae < margin else "pca16"
    report = {
        "target": "log|TF_1D| at each receiver column",
        "holdout": "validation IID runs",
        "ridge_alpha": RIDGE,
        "tie_margin": TIE_MARGIN,
        "pca64_rank": pca64["rank"],
        "pca64_variance_explained": pca64["explained"],
        "pca64_negative_eigenvalues": pca64["negative_eigenvalues"],
        "pod_rank": pod["rank"],
        "pod_variance_explained": pod["explained"],
        "n_train_curves": n_train,
        "combined_beats_pca16_by_more_than_tie_margin": combined_mae < margin,
        "tie_threshold_mae": margin,
        "freeze": freeze,
        "standardized_condition": conds,
        "scores": val,
    }
    (args.out / "bound.json").write_text(json.dumps(report))
    print(f"freeze {freeze}", flush=True)
    for name in ("mean", "pca16", "pca64", "raw", "combined"):
        row = val[name]
        bands = " ".join(f"{key} {value:.4f}" for key, value in row["bands"].items())
        print(f"{name} mae {row['mae']:.4f} L2 {row['frobenius']:.4f} {bands}", flush=True)


if __name__ == "__main__":
    main()
