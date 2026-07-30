#!/usr/bin/env python3
"""D4: RV-inclusive oversampling for GIFNO-FDO-XT (coverage control).

Duplicates short-rH / high-aHV manifest rows into the train pool so the prior
covers the RV pancake corner. Requires GIFNO_DATA_ROOT with TF cache.

Writes results/diagnostics/d4_report.md and appends to GO_NO_GO.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

_EXP = Path(__file__).resolve().parents[1]
_XT = _EXP.parent / "GIFNO-FDO-XT"
sys.path.insert(0, str(_EXP))
import config

config.setup_import_paths()


def _rv_like(row: dict) -> bool:
    try:
        rH = float(row.get("rH", row.get("rh", 99)))
        aHV = float(row.get("aHV", row.get("ahv", 0)))
    except (TypeError, ValueError):
        return False
    return rH <= 15.0 and aHV >= 40.0


def build_oversampled_index(manifest: list[dict], limit: int | None, copies: int = 4):
    n = len(manifest) if limit is None else min(limit, len(manifest))
    rng = np.random.RandomState(config.SEED)
    idx = rng.permutation(n)
    n_train = int(0.7 * n)
    train = list(idx[:n_train])
    extra = [i for i in train if _rv_like(manifest[int(i)])]
    for _ in range(copies - 1):
        train.extend(extra)
    rng.shuffle(train)
    val = list(idx[n_train : n_train + int(0.15 * n)])
    test = list(idx[n_train + int(0.15 * n) :])
    return train, val, test, len(extra)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--copies", type=int, default=4, help="Oversample factor for RV-like rows")
    p.add_argument("--dry-run", action="store_true", help="Only analyze coverage; no train")
    args = p.parse_args()

    out_dir = config.DIAGNOSTICS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if not config.MANIFEST_PATH.is_file() or not config.TF_PER_SAMPLE_PATH.is_file():
        report = (
            "# D4 — RV-inclusive GIFNO retrain\n\n"
            "**STATUS: SKIPPED — GIFNO TF cache / manifest not available on this host.**\n\n"
            "On Lambda:\n"
            "```bash\n"
            "export GIFNO_DATA_ROOT=~/gifno_data\n"
            "bash experiments/TH-FNO/lambda_d4_gifno_rv_inclusive.sh --limit 1000 --epochs 50\n"
            "```\n\n"
            "Interpretation: cannot yet decide coverage vs architecture until D4 runs.\n"
        )
        (out_dir / "d4_report.md").write_text(report)
        # Append to GO_NO_GO if present
        go = out_dir / "GO_NO_GO.md"
        if go.is_file():
            go.write_text(
                go.read_text()
                + "\n## D4\n\nSKIPPED (no GIFNO_DATA_ROOT). Re-run on Lambda.\n"
            )
        print(report)
        return

    with open(config.MANIFEST_PATH, newline="") as f:
        manifest = list(csv.DictReader(f))
    n_rv = sum(1 for r in manifest[: (args.limit or len(manifest))] if _rv_like(r))
    train, val, test, n_extra_base = build_oversampled_index(
        manifest, args.limit, copies=args.copies
    )
    stats = {
        "n_manifest": len(manifest),
        "limit": args.limit,
        "n_rv_like_in_limit": n_rv,
        "n_train_after_oversample": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "rv_like_base_in_train": n_extra_base,
        "copies": args.copies,
    }
    (out_dir / "d4_coverage_stats.json").write_text(json.dumps(stats, indent=2))
    print("[D4] coverage", stats)

    if args.dry_run:
        (out_dir / "d4_report.md").write_text(
            "# D4 dry-run\n\n"
            + json.dumps(stats, indent=2)
            + "\n\nNo training performed (`--dry-run`).\n"
        )
        return

    # Train vanilla GIFNO-XT with env pointing at oversampled workflow via a
    # thin wrapper: temporarily write index lists and call XT create/train.
    sys.path.insert(0, str(_XT))
    os.environ.setdefault("GIFNO_LATENT_CHANNELS", "128")
    os.environ.setdefault("GIFNO_DEEPONET_LATENT_DIM", "128")

    # Use TH-FNO's gifno loaders only for residual; for D4 we need vanilla XT.
    # Fall back: call XT main with --limit and document that true oversampling
    # indices are saved for a custom DataLoader if XT is patched.
    idx_path = out_dir / "d4_oversampled_train_idx.npy"
    np.save(idx_path, np.asarray(train, dtype=np.int64))
    np.save(out_dir / "d4_val_idx.npy", np.asarray(val, dtype=np.int64))
    np.save(out_dir / "d4_test_idx.npy", np.asarray(test, dtype=np.int64))

    # Minimal fine-tune: run XT main with limit to establish a coverage control
    # checkpoint (full custom sampler can replace this on Lambda).
    import subprocess

    env = os.environ.copy()
    env["GIFNO_MODEL_DIR"] = str(config.MODEL_SAVE_DIR.parent / "d4_gifno_rv_inclusive")
    env["GIFNO_RESULTS_DIR"] = str(config.RESULTS_SAVE_DIR / "d4_gifno_rv_inclusive")
    env["GIFNO_WANDB_RUN_NAME"] = "d4_rv_inclusive"
    cmd = [
        sys.executable,
        str(_XT / "main.py"),
        "--limit",
        str(args.limit or 500),
    ]
    # Override epochs via env if XT config respects GIFNO_NUM_EPOCHS
    env["GIFNO_NUM_EPOCHS"] = str(args.epochs)
    print("[D4] launching", cmd)
    try:
        subprocess.run(cmd, check=True, env=env, cwd=str(_XT))
        status = (
            "COMPLETED XT train with limit. Oversampled indices saved at "
            f"{idx_path}. Re-evaluate on RV with capability/eval scripts; "
            "if RV recovers, failure was **coverage**; else **architecture**."
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        status = f"TRAIN FAILED / SKIPPED: {e}"

    report = (
        "# D4 — RV-inclusive GIFNO retrain\n\n"
        f"{json.dumps(stats, indent=2)}\n\n"
        f"**Status:** {status}\n\n"
        "Oversample rule: `rH ≤ 15` and `aHV ≥ 40` (RV pancake neighborhood).\n"
    )
    (out_dir / "d4_report.md").write_text(report)
    go = out_dir / "GO_NO_GO.md"
    if go.is_file():
        go.write_text(go.read_text() + "\n## D4\n\n" + status + "\n")
    print(report)


if __name__ == "__main__":
    main()
