#!/usr/bin/env python3
"""
Prepare OOD TF cache + 80/20 hold-out split for mix-finetune.

Reads GT TFs already cached by score_ood_campaign.py (or computes them),
writes:
  out_dir/
    ood_tf_per_sample.npy   (N, 21, 1000)
    ood_manifest.csv
    ood_split.json          {train_indices, holdout_indices} within OOD block
    mix_manifest.csv        original GIFNO rows + OOD train rows (absolute h5 paths)
    mix_tf_per_sample.npy   concat(original[:n_id], ood[train])
    holdout_manifest.csv
    holdout_tf_per_sample.npy

Example:
  export GIFNO_DATA_ROOT="..."
  uv run python prepare_ood_mix_data.py --holdout-frac 0.2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

import config

config.setup_import_paths()

from capability_check import compute_ground_truth_tf  # noqa: E402

DEFAULT_BOX = Path("/mnt/box/GIG Lab - UC Berkeley/Projects/Neural Operator/data")
DEFAULT_OOD_SCORES = (
    Path.home() / "surrogate-seismic-waves" / "checkpoints" / "ood_campaign_scores"
)
DEFAULT_OUT = (
    Path.home() / "surrogate-seismic-waves" / "checkpoints" / "mix_finetune_data"
)


def _load_or_compute_tf(h5_path: Path, cache_dir: Path) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tf_path = cache_dir / "tf_true.npy"
    if tf_path.is_file():
        return np.load(tf_path)
    tf, freq = compute_ground_truth_tf(h5_path)
    np.save(tf_path, tf)
    np.save(cache_dir / "freq.npy", freq)
    return tf


def collect_campaign(
    name: str,
    camp_dir: Path,
    scores_root: Path,
) -> tuple[list[dict], np.ndarray]:
    man_path = camp_dir / "manifest.csv"
    with man_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    h5_dir = camp_dir / "h5"
    tfs = []
    out_rows = []
    for row in rows:
        idx = int(row["index"])
        h5_path = h5_dir / f"run_{idx}.h5"
        if not h5_path.is_file():
            print(f"[warn] skip missing {h5_path}")
            continue
        cache_dir = scores_root / name / f"run_{idx}"
        tf = _load_or_compute_tf(h5_path, cache_dir)
        tfs.append(tf.astype(np.float32))
        out_rows.append(
            {
                "sample_idx": f"ood_{name}_{idx}",
                "run_index": idx,
                "h5_path": str(h5_path.resolve()),
                "rf_seed": row.get("rf_seed", row.get("seed1", "")),
                "H_discretized": row.get("H_discretized", row.get("H1_discretized", "")),
                "CoV": row.get("CoV", row.get("CoV1", "")),
                "f0_effective": row.get("f0_effective", ""),
                "nz_actual": "",
                "n_lateral": "21",
                "ood_campaign": name,
                "ood_index": str(idx),
            }
        )
    arr = np.stack(tfs, axis=0)
    print(f"[{name}] collected {arr.shape}")
    return out_rows, arr


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--box-root", type=Path, default=DEFAULT_BOX)
    p.add_argument("--ood-scores-root", type=Path, default=DEFAULT_OOD_SCORES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--id-limit",
        type=int,
        default=None,
        help="Optional cap on in-dist GIFNO rows used in the mix",
    )
    args = p.parse_args()

    ood_rows: list[dict] = []
    ood_tfs: list[np.ndarray] = []
    for name in ("dipping", "three_layer"):
        camp = args.box_root / f"ood_{name}"
        rows, arr = collect_campaign(name, camp, args.ood_scores_root)
        ood_rows.extend(rows)
        ood_tfs.append(arr)
    ood_tf = np.concatenate(ood_tfs, axis=0)
    assert len(ood_rows) == ood_tf.shape[0]

    rng = np.random.default_rng(args.seed)
    # Stratify holdout by campaign
    train_idx: list[int] = []
    hold_idx: list[int] = []
    offset = 0
    for name, arr in zip(("dipping", "three_layer"), ood_tfs):
        n = arr.shape[0]
        perm = rng.permutation(n)
        n_hold = max(1, int(round(args.holdout_frac * n)))
        hold_local = set(perm[:n_hold].tolist())
        for i in range(n):
            gi = offset + i
            if i in hold_local:
                hold_idx.append(gi)
            else:
                train_idx.append(gi)
        offset += n
    train_idx = np.asarray(sorted(train_idx), dtype=np.int64)
    hold_idx = np.asarray(sorted(hold_idx), dtype=np.int64)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "ood_tf_per_sample.npy", ood_tf)
    with (args.out_dir / "ood_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ood_rows[0].keys()))
        w.writeheader()
        w.writerows(ood_rows)

    split = {
        "seed": args.seed,
        "holdout_frac": args.holdout_frac,
        "n_ood": int(len(ood_rows)),
        "train_indices": train_idx.tolist(),
        "holdout_indices": hold_idx.tolist(),
        "n_train": int(len(train_idx)),
        "n_holdout": int(len(hold_idx)),
    }
    (args.out_dir / "ood_split.json").write_text(json.dumps(split, indent=2))

    # Holdout arrays
    hold_rows = [ood_rows[i] for i in hold_idx]
    hold_tf = ood_tf[hold_idx]
    np.save(args.out_dir / "holdout_tf_per_sample.npy", hold_tf)
    with (args.out_dir / "holdout_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(hold_rows[0].keys()))
        w.writeheader()
        w.writerows(hold_rows)

    # In-dist
    id_man_path = config.MANIFEST_PATH
    id_tf_path = config.TF_PER_SAMPLE_PATH
    with id_man_path.open(newline="", encoding="utf-8") as f:
        id_rows = list(csv.DictReader(f))
    id_tf = np.load(id_tf_path, mmap_mode="r")
    n_id = len(id_rows)
    if args.id_limit is not None:
        n_id = min(n_id, args.id_limit)
    id_rows = id_rows[:n_id]
    # Rewrite h5 paths to GIFNO_H5_DIR if set
    h5_dir = os.environ.get("GIFNO_H5_DIR")
    fixed_id = []
    for r in id_rows:
        rr = dict(r)
        if h5_dir:
            rr["h5_path"] = str(Path(h5_dir) / Path(r["h5_path"]).name)
        rr.setdefault("ood_campaign", "")
        rr.setdefault("ood_index", "")
        fixed_id.append(rr)

    ood_train_rows = [ood_rows[i] for i in train_idx]
    # Align columns
    all_keys = list(
        dict.fromkeys(
            list(fixed_id[0].keys()) + list(ood_train_rows[0].keys())
        )
    )
    mix_rows = []
    for r in fixed_id + ood_train_rows:
        mix_rows.append({k: r.get(k, "") for k in all_keys})

    mix_tf = np.concatenate(
        [np.asarray(id_tf[:n_id]), ood_tf[train_idx]], axis=0
    ).astype(np.float32)
    assert mix_tf.shape[0] == len(mix_rows)

    np.save(args.out_dir / "mix_tf_per_sample.npy", mix_tf)
    with (args.out_dir / "mix_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        w.writerows(mix_rows)

    # Copy freq + recorder idx for convenience
    np.save(args.out_dir / "freq.npy", np.load(config.TF_FREQ_PATH))
    np.save(
        args.out_dir / "recorder_x_idx.npy",
        np.load(config.TF_RESULTS_DIR / "recorder_x_idx.npy"),
    )

    meta = {
        "n_id": n_id,
        "n_ood_train": int(len(train_idx)),
        "n_ood_holdout": int(len(hold_idx)),
        "mix_n": int(mix_tf.shape[0]),
        "mix_tf_shape": list(mix_tf.shape),
    }
    (args.out_dir / "prep_meta.json").write_text(json.dumps({**split, **meta}, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"[prep] Wrote mix data to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
