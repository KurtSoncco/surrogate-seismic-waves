#!/usr/bin/env python3
"""Regression check on held-out IID, dipping, three-layer and section windows (plan step 10).

Scores in TF space, not in the residual space main.py reports: the
prediction is TF_1D * exp(mean_m log head_m), the truth is the outcrop label
(label * TF_1D for residual datasets). The ``tf1d_prior`` arm is the 1D
estimate alone. Metrics per campaign, 0.1-2.5 Hz (trusted) and 0.4-2.5 Hz
(core), all 21 stations: median over windows of the log RMS error, of the
relative L2 of |TF| and of the Pearson correlation of log |TF| over frequency.

    python score_campaigns.py --data DIR/holdout --model-dir CK_A [--model-dir CK_B ...] \
        --label iid_only --out results/campaigns/iid_only.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from labels import CORE_BAND_HZ, TRUSTED_BAND_HZ, band_mask


def _resolve_h5(stored: str, data_dir: Path) -> Path:
    p = Path(stored)
    if p.is_file():
        return p
    for d in (data_dir / "h5", data_dir.parent / "h5", Path(os.environ.get("GIFNO_H5_DIR", ""))):
        cand = d / p.name
        if cand.is_file():
            return cand
    raise FileNotFoundError(stored)

BANDS = {"trusted": TRUSTED_BAND_HZ, "core": CORE_BAND_HZ}


def _metrics(log_pred: np.ndarray, log_true: np.ndarray, m: np.ndarray) -> dict:
    p, t = log_pred[..., m], log_true[..., m]
    rms = np.sqrt(np.mean((p - t) ** 2, axis=(1, 2)))
    ep, et = np.exp(p), np.exp(t)
    rel = np.linalg.norm((ep - et).reshape(len(p), -1), axis=1) / np.linalg.norm(et.reshape(len(t), -1), axis=1)
    pc = p - p.mean(axis=-1, keepdims=True)
    tc = t - t.mean(axis=-1, keepdims=True)
    r = (pc * tc).sum(-1) / np.sqrt((pc**2).sum(-1) * (tc**2).sum(-1) + 1e-12)
    return {"log_rms_p50": float(np.median(rms)), "rel_l2_p50": float(np.median(rel)), "pearson_p50": float(np.median(r.mean(axis=1))), "n": int(len(p))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="build_outcrop_dataset.py output (usually .../holdout)")
    ap.add_argument("--model-dir", type=Path, action="append", required=True)
    ap.add_argument("--label", default="model")
    ap.add_argument("--max-per-campaign", type=int, default=200)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import hdf5plugin  # noqa: F401
    import h5py

    from acquire_sections import Ensemble

    ens = Ensemble(args.model_dir, args.data / "freq.npy")
    c = ens.config
    meta = json.loads((args.data / "dataset.json").read_text())
    rows = list(csv.DictReader(open(args.data / "manifest.csv")))
    label = np.load(args.data / "tf_per_sample.npy", mmap_mode="r")
    tf1d = np.load(args.data / "tf1d_per_sample.npy", mmap_mode="r")
    freq = np.load(args.data / "freq.npy")
    masks = {k: band_mask(freq, v) for k, v in BANDS.items()}

    by_campaign: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_campaign[r.get("campaign", "all")].append(i)
    rng = np.random.default_rng(0)
    out: dict = {"label": args.label, "models": [str(d) for d in args.model_dir], "data": str(args.data), "campaigns": {}}
    for camp, idx in sorted(by_campaign.items()):
        if len(idx) > args.max_per_campaign:
            idx = sorted(rng.choice(idx, size=args.max_per_campaign, replace=False).tolist())
        log_head = []
        for i in idx:
            r = rows[i]
            with h5py.File(_resolve_h5(r["h5_path"], args.data), "r") as f:
                vs = f["Vs_realization_2D"][:].astype(np.float64)
                zeta = f["Damping_zeta"][:].astype(np.float64)
            start = int(float(r["x_start"])) if r.get("x_start") not in (None, "") else int(c.X_SLICE_START)
            log_head.append(ens.log_residual(vs, zeta, np.array([start])).mean(axis=0)[0])
        log_head = np.stack(log_head)
        log_1d = np.log(np.maximum(np.asarray(tf1d[idx]), 1e-6))
        lab = np.asarray(label[idx])
        log_true = np.log(np.maximum(lab * tf1d[idx] if meta["target"] == "residual" else lab, 1e-6))
        out["campaigns"][camp] = {
            arm: {b: _metrics(lp, log_true, m) for b, m in masks.items()}
            for arm, lp in (("tf1d_prior", log_1d), (args.label, log_1d + log_head))
        }
        t = out["campaigns"][camp]
        print(f"{camp:16s} n={len(idx):4d} trusted log-RMS prior {t['tf1d_prior']['trusted']['log_rms_p50']:.3f} "
              f"model {t[args.label]['trusted']['log_rms_p50']:.3f} | Pearson prior {t['tf1d_prior']['trusted']['pearson_p50']:.3f} "
              f"model {t[args.label]['trusted']['pearson_p50']:.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
