#!/usr/bin/env python3
"""CPU floor measurement and the feasibility stop decision.

This does not train a network, open the test split, or decrypt an external label.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path[:0] = [str(ROOT), str(REPO / "experiments" / "GIFNO-VARDOMAIN")]

from diagnosis import length_scaling_gate  # noqa: E402
from external.decision import paper_decision  # noqa: E402
from model import fit_weighted_pod, spectral_weight  # noqa: E402
from representation import fit_input_pod  # noqa: E402


def _envelope(vs: np.ndarray, train: np.ndarray) -> dict:
    sample = train[:: max(len(train) // 200, 1)]
    lows, highs, ratios = [], [], []
    for index in sample:
        field = np.asarray(vs[int(index)], dtype=np.float64)
        column = field[:, ::10]
        lows.append(float(column.min()))
        highs.append(float(column.max()))
        ratios.append(float(column.max() / max(column.min(), 1.0)))
    return {
        "vs_min": float(np.min(lows)),
        "vs_max": float(np.max(highs)),
        "impedance_ratio_max": float(np.quantile(ratios, 0.99)),
        "train_runs_sampled": int(sample.size),
    }


def _floors(cache: Path) -> dict:
    split = np.load(cache / "iid" / "split.npy")
    train = np.flatnonzero(split == "train")
    if train.size == 0:
        raise RuntimeError("IID cache has no training split")
    freq = np.load(cache / "freq.npy")
    weight = spectral_weight(freq)
    log_tf = np.load(cache / "iid" / "log_tf.npy", mmap_mode="r")
    prior = np.load(cache / "iid" / "log_tf1d.npy", mmap_mode="r")
    fit_runs = train[:32]
    eval_runs = train[32:48]
    def curves(indexes):
        rows = []
        for index in indexes:
            residual = np.asarray(log_tf[int(index), ::5], dtype=np.float64)
            residual = residual - np.asarray(prior[int(index), ::5], dtype=np.float64)
            rows.append(residual)
        return np.concatenate(rows)
    fitted = curves(fit_runs)
    modes, mean = fit_weighted_pod(fitted, min(128, fitted.shape[0] - 1), weight)
    held = curves(eval_runs)
    centered = held - mean
    recon = mean + ((centered * weight) @ modes.T) @ modes / weight
    bands = {}
    for name, lo, hi in (("low", 0.1, 0.4), ("mid", 0.4, 2.5), ("high", 2.5, 10.0), ("overall", 0.1, 10.0)):
        mask = (freq >= lo) & (freq <= hi)
        bands[name] = float(np.abs(recon - held)[:, mask].mean())
    vs = np.load(cache / "iid" / "vs.npy", mmap_mode="r")
    zeta = np.load(cache / "iid" / "zeta.npy", mmap_mode="r")
    columns_vs, columns_zeta = [], []
    for index in fit_runs[::2]:
        field = np.asarray(vs[int(index), :, ::25], dtype=np.float64)
        damp = np.asarray(zeta[int(index), :, ::25], dtype=np.float64)
        columns_vs.append(field.T)
        columns_zeta.append(damp.T)
    vs_rows = np.concatenate(columns_vs)
    zeta_rows = np.concatenate(columns_zeta)
    observed = np.ones(vs_rows.shape, dtype=bool)
    pod = fit_input_pod(vs_rows, zeta_rows, observed, 64)
    centered_vs = np.log(np.maximum(vs_rows, 1e-6) / 400.0)
    # Unweighted PCA on the Vs column only, for the control floor.
    pca_mean = centered_vs.mean(axis=0)
    _, _, vt = np.linalg.svd(centered_vs - pca_mean, full_matrices=False)
    pca_recon = pca_mean + (centered_vs - pca_mean) @ vt[:16].T @ vt[:16]
    return {
        "output_pod_floor_by_band": bands,
        "output_pod_fit_runs": int(fit_runs.size),
        "output_pod_eval_runs": int(eval_runs.size),
        "eval_split": "iid_train_held_out_of_the_basis_fit",
        "test_split_opened": False,
        "input_weighted_pod_reconstruction": pod["reconstruction"],
        "input_weighted_pod_rank": pod["rank"],
        "input_pca16_vs_reconstruction": float(np.mean(np.abs(pca_recon - centered_vs))),
        "envelope": _envelope(vs, train),
    }


def main() -> None:
    cache = REPO / "data" / "vardomain" / "cache"
    out = ROOT / "analysis" / "preregistration"
    out.mkdir(parents=True, exist_ok=True)
    floors = _floors(cache)
    decision = paper_decision(timing_measured=False, custodian="", cuda=False)
    decision["length_scaling_gate"] = length_scaling_gate(np.array([]), floors["output_pod_floor_by_band"]["overall"])
    decision["learning_curve"] = {
        "status": "not_run",
        "label": "inconclusive",
        "reason": "No CUDA is available, and the feasibility gate blocked confirmatory training.",
        "families_not_classified": ["iid", "ood_dipping", "ood_three_layer", "manifold"],
    }
    (out / "local_floors.json").write_text(json.dumps(floors, indent=2) + "\n")
    (out / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps({"paper": decision["paper"], "blocked": decision["blocked_reasons"], "floors": floors["output_pod_floor_by_band"]}, indent=2))


if __name__ == "__main__":
    main()
