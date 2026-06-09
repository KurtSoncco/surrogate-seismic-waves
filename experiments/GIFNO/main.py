# main.py
"""Main entry point for GIFNO grid-direct FNO training and evaluation."""

import os
import random
import subprocess
import sys

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import config
from data_loader import get_data_loaders
from evaluate import evaluate_model
from model import create_model
from train import train_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_tf_cache():
    """Run TF preprocessing if cache is missing."""
    if config.TF_PER_SAMPLE_PATH.exists() and config.MANIFEST_PATH.exists():
        return
    print("[GIFNO] TF cache not found — running preprocessing...")
    preprocess_script = (
        config.EXPERIMENT_DIR / "preprocess" / "compute_transfer_function.py"
    )
    subprocess.run([sys.executable, str(preprocess_script)], check=True)


def run_pipeline(limit: int | None = None):
    set_seed(config.SEED)
    ensure_tf_cache()

    train_loader, val_loader, test_loader, freq = get_data_loaders(limit=limit)

    run = train_model(train_loader, val_loader)

    model = create_model(
        in_channels=config.IN_CHANNELS,
        latent_channels=config.LATENT_CHANNELS,
        n_freq=config.N_FREQ,
        fno_modes=config.FNO_MODES,
        num_fno_layers=config.NUM_FNO_LAYERS,
    ).to(config.DEVICE)
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

    parser = argparse.ArgumentParser(description="GIFNO training pipeline")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit dataset size (for smoke tests)",
    )
    args = parser.parse_args()
    run_pipeline(limit=args.limit)
