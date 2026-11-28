#!/usr/bin/env python3
"""Retrain the soil-column autoencoder with a wider code and score it by depth.

The previous fit used a code of length 2, the same width as log Vs and log zeta.
A contact depth does not fit in that code. This run uses 32 codes and still has
no parameter whose length is the number of depth rows.

    python fit_column_ae.py --cache ... --out .../depth_code32 --baseline .../depth_ae.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data import Cache
from pretrain_ae import _refuse_shipped, column_splits, train_depth_indexed
from screen_hifreq import score_depth_columns

CODE_DIM = 32


def _bands(report: dict) -> dict[str, dict[str, float]]:
    edges = ((0, 10), (10, 30), (30, 60), (60, 110))
    out = {}
    idx = report["row_index"]
    ae = report["row_mae_ae"]
    lin = report["row_mae_linear"]
    count = report["row_count"]
    base = report.get("row_mae_baseline")
    for lo, hi in edges:
        totals = {"ae": 0.0, "linear": 0.0, "n": 0.0}
        if base is not None:
            totals["baseline"] = 0.0
        for i, a, l, c in zip(idx, ae, lin, count):
            if not (lo <= i < hi) or not c:
                continue
            totals["ae"] += a * c
            totals["linear"] += l * c
            totals["n"] += c
            if base is not None:
                totals["baseline"] += base[i] * c
        n = totals.pop("n")
        out[f"{lo}-{hi}m"] = {key: (value / n if n else 0.0) for key, value in totals.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True, help="the code-length-2 checkpoint")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--code-dim", type=int, default=CODE_DIM)
    args = ap.parse_args()
    _refuse_shipped(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = Cache(args.cache)
    train_cols, hold_cols = column_splits(cache, 0, args.seed)
    print(f"columns train {len(train_cols)} hold {len(hold_cols)} code {args.code_dim} device {device}", flush=True)
    model = train_depth_indexed(
        cache, train_cols, epochs=args.epochs, batch=args.batch, lr=args.lr, seed=args.seed,
        device=device, code_dim=args.code_dim, hold_pairs=hold_cols,
    )
    checkpoint = args.out / "depth_ae.pt"
    torch.save(model.state_dict(), checkpoint)
    report = score_depth_columns(cache, hold_cols, checkpoint, device, batch=args.batch, code_dim=args.code_dim)
    # score_depth_columns loads a raw state dict. The wider run stores a dict with the code width.
    baseline = score_depth_columns(cache, hold_cols, args.baseline, device, batch=args.batch, code_dim=2)
    report["baseline_masked_l1"] = baseline["holdout_masked_l1"]
    report["row_mae_baseline"] = baseline["row_mae_ae"]
    report["code_dim"] = args.code_dim
    report["bands_m"] = _bands(report)
    (args.out / "report.json").write_text(json.dumps(report))
    print(
        f"masked L1 wide {report['holdout_masked_l1']:.4f} "
        f"code2 {report['baseline_masked_l1']:.4f} linear {report['linear_masked_l1']:.4f}",
        flush=True,
    )
    for name, row in report["bands_m"].items():
        print(
            f"  {name} wide {row['ae']:.4f} code2 {row['baseline']:.4f} linear {row['linear']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
