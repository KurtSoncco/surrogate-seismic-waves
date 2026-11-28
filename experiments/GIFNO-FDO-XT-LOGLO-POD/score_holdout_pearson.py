#!/usr/bin/env python3
"""Pearson of log |TF| on a mix holdout, by campaign, vs the 1D prior.

    python score_holdout_pearson.py --model-dir CK --data DIR/holdout --out CK/holdout_pearson.json

Residual datasets store TF/TF_1D in tf_per_sample.npy; the scorer multiplies
back by tf1d so the metric is always Pearson of log |TF| vs outcrop GT.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EPS = 1e-6
BANDS = {"trusted": (0.1, 2.5), "core": (0.4, 2.5), "full": (0.1, 10.0)}
REPO = Path(__file__).resolve().parents[2]


def band_mask(freq, rng):
    lo, hi = rng
    return (freq >= lo) & (freq <= hi)


def pearson_per_col(log_p, log_t):
    pc = log_p - log_p.mean(axis=-1, keepdims=True)
    tc = log_t - log_t.mean(axis=-1, keepdims=True)
    num = (pc * tc).sum(axis=-1)
    den = np.sqrt((pc**2).sum(axis=-1) * (tc**2).sum(axis=-1) + 1e-12)
    return num / den


def summarize(vals):
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {
            "n": 0,
            "median": float("nan"),
            "mean": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
        }
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
    }


def scores_from_arrays(pred, true, valid, freq):
    """pred/true (N, C, F), valid (N, C). Sample score = mean Pearson over valid cols."""
    out = {}
    log_p = np.log(np.maximum(pred, EPS))
    log_t = np.log(np.maximum(true, EPS))
    for name, rng in BANDS.items():
        m = band_mask(freq, rng)
        per_sample = []
        for i in range(len(pred)):
            ok = np.asarray(valid[i], dtype=bool)
            if not ok.any():
                continue
            r = pearson_per_col(log_p[i, ok][:, m], log_t[i, ok][:, m])
            per_sample.append(float(np.mean(r)))
        out[name] = summarize(per_sample)
    return out


def _to_outcrop_tf(arr, tf1d, target: str):
    if target == "residual":
        return np.asarray(arr) * np.asarray(tf1d)
    return np.asarray(arr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="mix holdout directory")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    hold = args.data
    ckpt = args.model_dir
    arm = args.arm or ckpt.name
    out_path = args.out or (ckpt / "holdout_pearson.json")
    meta = json.loads((hold / "dataset.json").read_text())
    target = meta.get("target", "outcrop")

    rows = list(csv.DictReader(open(hold / "manifest.csv")))
    label = np.load(hold / "tf_per_sample.npy", mmap_mode="r")
    tf1d = np.load(hold / "tf1d_per_sample.npy", mmap_mode="r")
    valid = np.load(hold / "collocation_valid.npy", mmap_mode="r")
    freq = np.load(hold / "freq.npy")
    rec = np.load(hold / "recorder_x_idx.npy")
    true_tf = _to_outcrop_tf(label, tf1d, target)
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r.get("campaign", "all")].append(i)

    print("=== 1D prior (holdout) ===", flush=True)
    result = {
        "model": str(ckpt),
        "data": str(hold),
        "target": target,
        "metric": "median over samples of mean-over-valid-columns Pearson of log|TF| vs GT",
        "campaigns": {},
    }
    for camp, idx in sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        sl = np.asarray(idx)
        prior = scores_from_arrays(np.asarray(tf1d[sl]), true_tf[sl], np.asarray(valid[sl]), freq)
        result["campaigns"][camp] = {"n": int(len(idx)), "tf1d_prior": prior}
        t = prior["trusted"]
        print(
            f"{camp:16s} n={len(idx):4d}  1D prior trusted r_med={t['median']:.3f}  mean={t['mean']:.3f}",
            flush=True,
        )

    print(f"=== loading {arm} from {ckpt} ===", flush=True)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.30, 0)

    # load_models imports LOGLO config; GIFNODataset must come after that so
    # sys.path is GIFNO + LOGLO, not case7a-only (which has no data_loader).
    sys.path.insert(0, str(REPO / "experiments/case7a"))
    from score_section import load_models

    models, device, _cfg, model_target = load_models([ckpt])
    from data_loader import GIFNODataset, load_manifest
    model = models[0]
    print(
        f"device={device} target={model_target} shared={getattr(model.head, 'shared', False)}",
        flush=True,
    )

    ds = GIFNODataset(
        load_manifest(hold / "manifest.csv"),
        label,
        rec,
        collocation_valid=valid,
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    pred_slots = np.zeros((len(ds), rec.size, freq.size), dtype=np.float32)
    i0 = 0
    rec_t = torch.as_tensor(rec, device=device, dtype=torch.long)
    with torch.no_grad():
        for xb, _yb, _mb in loader:
            n = xb.shape[0]
            grid = model(xb.to(device, non_blocking=True)).float()
            take = grid.index_select(1, rec_t).cpu().numpy()
            pred_slots[i0 : i0 + n] = take
            i0 += n
            if i0 % 200 < 8:
                print(f"  scored {i0}/{len(ds)}", flush=True)
    assert i0 == len(ds)
    pred_tf = _to_outcrop_tf(pred_slots, tf1d, model_target)

    print(f"=== model ({arm}) ===", flush=True)
    for camp, idx in sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        sl = np.asarray(idx)
        model_sc = scores_from_arrays(pred_tf[sl], true_tf[sl], np.asarray(valid[sl]), freq)
        result["campaigns"][camp][arm] = model_sc
        p = result["campaigns"][camp]["tf1d_prior"]["trusted"]
        m = model_sc["trusted"]
        beat = "WINS" if m["median"] > p["median"] else "loses"
        print(
            f"{camp:16s} n={len(idx):4d}  trusted Pearson  1D {p['median']:.3f}  "
            f"model {m['median']:.3f}  ({beat})",
            flush=True,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
