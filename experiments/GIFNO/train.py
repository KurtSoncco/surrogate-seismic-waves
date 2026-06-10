# train.py
"""Training script for the GIFNO grid-direct FNO model."""

import config
import torch
import torch.optim as optim
from tqdm import trange

import wandb

from losses import MaskedLpLoss
from model import create_model


def train_model(train_loader, val_loader):
    """Main training and validation loop."""
    model = create_model(
        in_channels=config.IN_CHANNELS,
        latent_channels=config.LATENT_CHANNELS,
        n_freq=config.N_FREQ,
        fno_modes=config.FNO_MODES,
        num_fno_layers=config.NUM_FNO_LAYERS,
    ).to(config.DEVICE)

    dummy = torch.randn(1, config.IN_CHANNELS, config.NZ_MAX, config.NX).to(
        config.DEVICE
    )
    assert model(dummy).shape == (1, config.NX, config.N_FREQ)

    criterion = MaskedLpLoss(d=2, p=2)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=20, factor=0.5
    )

    run = wandb.init(
        project=config.WANDB_PROJECT,
        name=config.WANDB_RUN_NAME,
        config={k: v for k, v in vars(config).items() if k.isupper()},
    )
    # Skip wandb.watch: FNOBlocks use spectral (complex) weights; watch casts them
    # to float and emits "Casting complex values to real" every step. Loss/lr/eval
    # plots are logged explicitly below and in evaluate.py.

    best_val_loss = float("inf")
    early_stop_counter = 0

    t = trange(config.NUM_EPOCHS, desc="Training")
    for epoch in t:
        model.train()
        train_loss = 0.0
        n_train = 0
        for inputs, targets, masks in train_loader:
            inputs = inputs.to(config.DEVICE)
            targets = targets.to(config.DEVICE)
            masks = masks.to(config.DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, targets, masks)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            n_train += inputs.size(0)

        train_loss /= max(n_train, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for inputs, targets, masks in val_loader:
                inputs = inputs.to(config.DEVICE)
                targets = targets.to(config.DEVICE)
                masks = masks.to(config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets, masks)
                val_loss += loss.item() * inputs.size(0)
                n_val += inputs.size(0)

        val_loss /= max(n_val, 1)
        scheduler.step(val_loss)

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
