# main.py
"""Main script for rf_seed 2D Vs to transfer function FNO training and evaluation."""

import os
import random
import sys

import numpy as np
import torch

# Ensure rf_seed is on path when running from project root or rf_seed
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ruff: noqa: E402
import config
from data_loader import get_data_loaders, load_frequency_data
from evaluate import evaluate_model
from train import train_model


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_pipeline():
    """Execute the full training and evaluation pipeline."""
    set_seed(config.SEED)

    train_loader, val_loader, test_loader = get_data_loaders(
        data_path=config.DATA_PATH,
        vs_shape=config.INPUT_SHAPE,
        tf_len=config.OUTPUT_SIZE,
        vs_keys=config.VS_KEYS,
        tf_keys=config.TF_KEYS,
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
        test_split=config.TEST_SPLIT,
        batch_size=config.BATCH_SIZE,
        seed=config.SEED,
    )

    run = train_model(train_loader, val_loader)

    # Load frequency data for evaluation plots
    freq_data = load_frequency_data(config.DATA_PATH)

    # Load best model and evaluate
    from model import create_model

    model = create_model(
        input_shape=config.INPUT_SHAPE,
        output_size=config.OUTPUT_SIZE,
        latent_dim=config.LATENT_DIM,
        encoder_channels=config.ENCODER_CHANNELS,
        encoder_kernel_size=config.ENCODER_KERNEL_SIZE,
        encoder_pool_size=config.ENCODER_POOL_SIZE,
        fno_modes=config.FNO_MODES,
        fno_width=config.FNO_WIDTH,
        num_fno_layers=config.NUM_FNO_LAYERS,
    ).to(config.DEVICE)
    model.load_state_dict(
        torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    )

    metrics = evaluate_model(
        model,
        test_loader,
        freq_data=freq_data,
        save_dir=config.RESULTS_SAVE_DIR,
        run=run,
        seed=config.SEED,
    )
    print("Test metrics:", metrics)

    import wandb

    wandb.finish()


if __name__ == "__main__":
    run_pipeline()
