#!/usr/bin/env python3
"""
Evaluate a saved HFNO checkpoint on the test split; log metrics and plots to W&B.

Use after a training job timed out but left best_model.pt on disk.

Example (interactive GPU node):
  export GIFNO_H5_DIR=.../h5
  export GIFNO_TF_DIR=.../transfer_function
  export GIFNO_MODEL_DIR=.../models/fdo_xt_hfno/sweep/n2000/hfno_depth
  export GIFNO_BRANCH_MODE=depth
  export GIFNO_LATENT_CHANNELS=128
  export GIFNO_DEEPONET_LATENT_DIM=128

  python eval_checkpoint.py --limit 2000 --wandb-run-id 5wp5nyin

Delta:
  sbatch delta_eval.sh --limit 2000 --wandb-run-id 5wp5nyin
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import wandb

import config

config.setup_import_paths()

from data_loader import get_data_loaders  # noqa: E402
from evaluate import evaluate_model  # noqa: E402
from model import create_model  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate HFNO checkpoint and log test metrics/plots to W&B"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to best_model.pt (default: config.MODEL_SAVE_PATH)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Dataset limit (use same value as the sweep, e.g. 2000)",
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="W&B run id to resume (e.g. 5wp5nyin)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional new run name when not resuming by id",
    )
    parser.add_argument(
        "--wandb-resume",
        type=str,
        choices=("must", "allow", "never"),
        default="must",
        help="W&B resume mode when --wandb-run-id is set (default: must)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Skip W&B logging (still writes test_metrics.json)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Test DataLoader batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader workers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: config.SEED)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    seed = args.seed if args.seed is not None else config.SEED

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ckpt = Path(args.checkpoint or config.MODEL_SAVE_PATH)
    if not ckpt.is_file():
        print(f"ERROR: checkpoint not found: {ckpt}", file=sys.stderr)
        return 1

    _, _, test_loader, freq = get_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=args.limit,
    )

    model = create_model().to(config.DEVICE)
    model.load_state_dict(
        torch.load(ckpt, map_location=config.DEVICE, weights_only=True)
    )
    print(f"Loaded checkpoint: {ckpt}")

    run = None
    if not args.no_wandb:
        init_kwargs: dict = {
            "project": config.WANDB_PROJECT,
            "config": {k: v for k, v in vars(config).items() if k.isupper()},
        }
        if args.wandb_run_id:
            init_kwargs["id"] = args.wandb_run_id
            init_kwargs["resume"] = args.wandb_resume
        if args.wandb_run_name:
            init_kwargs["name"] = args.wandb_run_name
        elif not args.wandb_run_id:
            init_kwargs["name"] = config.WANDB_RUN_NAME or "hfno_eval"
        run = wandb.init(**init_kwargs)

    metrics = evaluate_model(
        model,
        test_loader,
        freq_data=freq,
        save_dir=config.RESULTS_SAVE_DIR,
        run=run,
        seed=seed,
    )

    out_json = Path(config.RESULTS_SAVE_DIR) / "test_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {out_json}")
    print("Test metrics:", metrics)

    if run is not None:
        wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
