# train.py
"""Training script for the rf_seed 2D Vs to transfer function FNO model."""

import config
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

import wandb

from model import create_model


def train_model(train_loader, val_loader):
    """Main training and validation loop."""
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

    dummy_input = torch.randn(1, 1, *config.INPUT_SHAPE).to(config.DEVICE)
    dummy_output = model(dummy_input)
    assert dummy_output.shape == (1, config.OUTPUT_SIZE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=80, factor=0.9
    )

    run = wandb.init(
        project=config.WANDB_PROJECT,
        name=config.WANDB_RUN_NAME,
        config={k: v for k, v in vars(config).items() if k.isupper()},
    )
    wandb.watch(model, log="all")

    best_val_loss = float("inf")
    early_stop_counter = 0

    t = trange(config.NUM_EPOCHS, desc="Training")
    for epoch in t:
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(config.DEVICE)
            targets = targets.to(config.DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(config.DEVICE)
                targets = targets.to(config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        # --- Logging and Early Stopping ---
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        t.set_postfix(train_loss=train_loss, val_loss=val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= config.EARLY_STOP_PATIENCE:
            break

    return run
