#!/usr/bin/env python3
"""Offline test: Pearson |TF|(f) at ALL recorder positions on GIFNO test set.

Does not train. Loads a checkpoint and reports per-position and pooled
distributions (mean/std/quantiles + optional histogram CSV).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXP))
import config

config.setup_import_paths()

from gifno_dataset import get_gifno_loaders  # noqa: E402
from model import create_model  # noqa: E402


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if a.size < 2 or np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


@torch.no_grad()
def collect(model, loader, device):
    rec = config.recorder_x_indices()
    # pearson[sample, recorder], rel same
    pears, rels, offsets = [], [], []
    half = (config.NX - 1) / 2.0
    for c in rec:
        offsets.append(float((int(c) - half) * config.DX))  # m from strip center

    for batch in loader:
        x, haskell, target, mask, cov, dip, physics = [t.to(device) for t in batch]
        pred = model(x, haskell, cov, dip, physics=physics)
        for b in range(pred.shape[0]):
            row_p, row_r = [], []
            for c in rec:
                c = int(c)
                p = pred[b, c].detach().cpu().numpy()
                t = target[b, c].detach().cpu().numpy()
                row_p.append(_pearson(p, t))
                row_r.append(_rel_l2(p, t))
            pears.append(row_p)
            rels.append(row_r)
    return (
        np.asarray(pears, float),
        np.asarray(rels, float),
        np.asarray(offsets, float),
        np.asarray(rec, int),
    )


def summarize(pears: np.ndarray, rels: np.ndarray, offsets: np.ndarray, rec: np.ndarray):
    # per-position across samples
    per_pos = []
    for j in range(pears.shape[1]):
        pj = pears[:, j]
        rj = rels[:, j]
        per_pos.append(
            {
                "recorder_idx": int(j),
                "x_index": int(rec[j]),
                "x_offset_m": float(offsets[j]),
                "pearson_mean": float(np.nanmean(pj)),
                "pearson_std": float(np.nanstd(pj)),
                "pearson_p10": float(np.nanpercentile(pj, 10)),
                "pearson_p50": float(np.nanpercentile(pj, 50)),
                "pearson_p90": float(np.nanpercentile(pj, 90)),
                "rel_l2_mean": float(np.nanmean(rj)),
                "rel_l2_p50": float(np.nanpercentile(rj, 50)),
                "n": int(np.sum(np.isfinite(pj))),
            }
        )

    flat_p = pears.ravel()
    flat_r = rels.ravel()
    # sample-mean across all recorders
    sample_mean_p = np.nanmean(pears, axis=1)
    pooled = {
        "n_samples": int(pears.shape[0]),
        "n_recorders": int(pears.shape[1]),
        "n_curves": int(flat_p.size),
        "pearson_all_mean": float(np.nanmean(flat_p)),
        "pearson_all_std": float(np.nanstd(flat_p)),
        "pearson_all_p10": float(np.nanpercentile(flat_p, 10)),
        "pearson_all_p50": float(np.nanpercentile(flat_p, 50)),
        "pearson_all_p90": float(np.nanpercentile(flat_p, 90)),
        "pearson_sample_mean_of_rec_means": float(np.nanmean(sample_mean_p)),
        "pearson_center_mean": float(np.nanmean(pears[:, pears.shape[1] // 2])),
        "pearson_edge_mean": float(
            np.nanmean(np.concatenate([pears[:, 0], pears[:, -1]]))
        ),
        "rel_l2_all_mean": float(np.nanmean(flat_r)),
        "rel_l2_center_mean": float(np.nanmean(rels[:, rels.shape[1] // 2])),
        "rel_l2_edge_mean": float(
            np.nanmean(np.concatenate([rels[:, 0], rels[:, -1]]))
        ),
    }
    return per_pos, pooled


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--predict-mode", choices=("direct", "residual"), default="direct")
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=config.DIAGNOSTICS_DIR / "pearson_all_recorders",
    )
    args = p.parse_args()

    config.PREDICT_MODE = args.predict_mode
    device = config.DEVICE
    _, _, test_loader = get_gifno_loaders(limit=args.limit, batch_size=args.batch_size)
    model = create_model(predict_mode=args.predict_mode).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    pears, rels, offsets, rec = collect(model, test_loader, device)
    per_pos, pooled = summarize(pears, rels, offsets, rec)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.predict_mode
    with (args.out_dir / f"per_position_{tag}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_pos[0].keys()))
        w.writeheader()
        w.writerows(per_pos)
    # raw distribution for histogramming
    np.savez_compressed(
        args.out_dir / f"pearson_raw_{tag}.npz",
        pearson=pears,
        rel_l2=rels,
        offsets_m=offsets,
        recorder_x=rec,
    )
    (args.out_dir / f"pooled_{tag}.json").write_text(json.dumps(pooled, indent=2) + "\n")

    # simple ASCII hist of all-recorder pearson
    hist, edges = np.histogram(pears.ravel()[np.isfinite(pears.ravel())], bins=20, range=(-0.2, 1.0))
    hist_rows = [
        {"bin_lo": float(edges[i]), "bin_hi": float(edges[i + 1]), "count": int(hist[i])}
        for i in range(len(hist))
    ]
    with (args.out_dir / f"pearson_hist_{tag}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bin_lo", "bin_hi", "count"])
        w.writeheader()
        w.writerows(hist_rows)

    print(json.dumps(pooled, indent=2))
    print("\nPer-position pearson_mean (x_offset_m):")
    for row in per_pos:
        print(
            f"  {row['x_offset_m']:+7.1f} m  pearson={row['pearson_mean']:.4f} "
            f"(p50={row['pearson_p50']:.4f})  rel={row['rel_l2_mean']:.4f}"
        )
    print(f"\nWrote {args.out_dir}")


if __name__ == "__main__":
    main()
