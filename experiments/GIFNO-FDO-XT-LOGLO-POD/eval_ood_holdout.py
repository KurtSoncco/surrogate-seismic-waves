#!/usr/bin/env python3
"""
Evaluate a checkpoint on the OOD hold-out split from prepare_ood_mix_data.py,
and compare to zero-shot metrics from score_ood_campaign.py when available.

Example:
  export GIFNO_MODEL_DIR=~/surrogate-seismic-waves/checkpoints/mix_finetune_ood
  export GIFNO_POD_NUM_MODES=64 ...
  uv run python eval_ood_holdout.py \
    --mix-data ~/surrogate-seismic-waves/checkpoints/mix_finetune_data \
    --checkpoint $GIFNO_MODEL_DIR/best_model.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

import config

config.setup_import_paths()

from capability_check import build_input_from_h5, load_model, predict_tf  # noqa: E402
from score_ood_campaign import metrics_from_tfs  # noqa: E402

_EPS = 1e-12
DEFAULT_ZERO_SHOT = (
    Path.home() / "surrogate-seismic-waves" / "checkpoints" / "ood_campaign_scores"
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mix-data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--zero-shot-root", type=Path, default=DEFAULT_ZERO_SHOT)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    ckpt = Path(args.checkpoint or config.MODEL_SAVE_PATH)
    if not ckpt.is_file():
        print(f"ERROR: missing checkpoint {ckpt}", file=sys.stderr)
        return 1

    hold_man = args.mix_data / "holdout_manifest.csv"
    hold_tf = np.load(args.mix_data / "holdout_tf_per_sample.npy")
    freq = np.load(args.mix_data / "freq.npy")
    with hold_man.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == hold_tf.shape[0]

    device = torch.device(args.device if args.device else str(config.DEVICE))
    model = load_model(ckpt, device)

    out_dir = args.out_dir or (ckpt.parent / "holdout_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, row in enumerate(rows):
        h5_path = Path(row["h5_path"])
        x = build_input_from_h5(h5_path)
        pred = predict_tf(model, x, device)
        m = metrics_from_tfs(pred, hold_tf[i], freq)
        camp = row.get("ood_campaign", "")
        idx = int(row.get("ood_index", row.get("run_index", i)))
        # zero-shot from campaign CSV if present
        zs = None
        zs_csv = args.zero_shot_root / camp / "per_case_metrics.csv"
        if zs_csv.is_file():
            with zs_csv.open(newline="", encoding="utf-8") as f:
                for zr in csv.DictReader(f):
                    if int(zr["index"]) == idx:
                        zs = float(zr["rel_l2_mean"])
                        break
        results.append(
            {
                "ood_campaign": camp,
                "ood_index": idx,
                "rel_l2_mean": m["rel_l2_mean"],
                "pearson_mean": m["pearson_mean"],
                "zero_shot_rel_l2": zs,
                "delta_rel_l2": (m["rel_l2_mean"] - zs) if zs is not None else None,
            }
        )
        if (i + 1) % 50 == 0:
            print(f"  holdout {i + 1}/{len(rows)}")

    by_camp: dict[str, list] = {}
    for r in results:
        by_camp.setdefault(r["ood_campaign"], []).append(r)

    summary = {"n": len(results), "campaigns": {}}
    for camp, group in by_camp.items():
        entry = {
            "n": len(group),
            "rel_l2_mean": float(np.mean([g["rel_l2_mean"] for g in group])),
            "pearson_mean": float(np.mean([g["pearson_mean"] for g in group])),
        }
        zs_vals = [g["zero_shot_rel_l2"] for g in group if g["zero_shot_rel_l2"] is not None]
        if zs_vals:
            entry["zero_shot_rel_l2_mean"] = float(np.mean(zs_vals))
            entry["delta_rel_l2_mean"] = float(
                np.mean(
                    [
                        g["delta_rel_l2"]
                        for g in group
                        if g["delta_rel_l2"] is not None
                    ]
                )
            )
        summary["campaigns"][camp] = entry

    with (out_dir / "holdout_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    (out_dir / "holdout_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
