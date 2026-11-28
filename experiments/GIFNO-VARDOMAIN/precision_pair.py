#!/usr/bin/env python3
"""Score one checkpoint twice: bfloat16 autocast, then full float32.

The number is the 0.1–10 Hz log-L1, averaged inside each width bin and then
across bins, which is the log-L1 the trainer prints.

    python precision_pair.py --ckpt checkpoints/vardomain/v2_ablations/P64/best_model.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data import Cache
from evaluate import evaluate
from model import load_vardomain_state, model_from_config


def family_log_l1(res: dict, family: str) -> float:
    vals = [block["log_l1"] for key, block in res["bins"].items() if key.startswith(family + "_w")]
    if not vals:
        raise RuntimeError(f"no {family} bins in {sorted(res['bins'])[:8]}")
    return float(np.mean(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--only", choices=("both", "bf16", "fp32"), default="both")
    args = ap.parse_args()
    pod = args.ckpt.parent
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = model_from_config(
        cfg,
        np.load(pod / "pod_modes.npy"),
        np.load(pod / "pod_mean.npy"),
        np.load(pod / "pca_mean.npy"),
        np.load(pod / "pca_components.npy"),
    )
    load_vardomain_state(model, ckpt["state_dict"])
    device = torch.device("cuda")
    model.to(device).eval()
    cache = Cache(args.cache)
    print(f"checkpoint epoch {ckpt.get('epoch')} pca_rank {cfg.get('pca_rank')}", flush=True)
    scores = {}
    passes = (True, False) if args.only == "both" else (args.only == "bf16",)
    for amp in passes:
        res = evaluate(
            model, cache, "metric", "val", device, batch=args.batch, workers=2,
            positions=1, amp=amp, target="residual",
            families=("iid", "ood_dipping", "ood_three_layer"), stage=3,
        )
        iid = family_log_l1(res, "iid")
        man = family_log_l1(res, "manifold")
        scores[amp] = (iid, man)
        label = "bf16" if amp else "fp32"
        print(f"{label} iid {iid:.6f} manifold {man:.6f} mean {0.5 * (iid + man):.6f}", flush=True)
    if True in scores and False in scores:
        for name, index in (("iid", 0), ("manifold", 1)):
            bf, fp = scores[True][index], scores[False][index]
            rel = abs(bf - fp) / max(abs(fp), 1e-8)
            print(f"{name} abs {abs(bf - fp):.6e} rel {100 * rel:.4f}%", flush=True)


if __name__ == "__main__":
    main()
