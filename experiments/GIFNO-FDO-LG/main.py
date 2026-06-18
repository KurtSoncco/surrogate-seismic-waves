# main.py
"""GIFNO-FDO-LG: FNO + U-Net local branch transfer training."""

import os
import random
import subprocess
import sys

import numpy as np
import torch

import config

config.setup_import_paths()
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from data_loader import get_data_loaders  # noqa: E402
from evaluate import evaluate_model  # noqa: E402
from model import create_model  # noqa: E402
from train import train_model  # noqa: E402


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_tf_cache():
    if config.TF_PER_SAMPLE_PATH.exists() and config.MANIFEST_PATH.exists():
        return
    print("[GIFNO-FDO-LG] TF cache not found — running GIFNO preprocessing...")
    subprocess.run([sys.executable, str(config.PREPROCESS_SCRIPT)], check=True)


def run_pipeline(limit: int | None = None):
    set_seed(config.SEED)
    ensure_tf_cache()

    train_loader, val_loader, test_loader, freq = get_data_loaders(limit=limit)

    run = train_model(train_loader, val_loader)

    model = create_model().to(config.DEVICE)
    model.load_state_dict(
        torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    )

    metrics = evaluate_model(
        model,
        test_loader,
        freq_data=freq,
        save_dir=config.RESULTS_SAVE_DIR,
        run=run,
        seed=config.SEED,
    )
    print("Test metrics:", metrics)

    import wandb

    wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GIFNO-FDO-LG training pipeline")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit dataset size (for smoke tests)",
    )
    args = parser.parse_args()
    run_pipeline(limit=args.limit)
