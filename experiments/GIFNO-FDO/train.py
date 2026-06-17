# train.py
"""Training loop for GIFNO-FDO."""

import config
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange

import wandb

from losses import build_training_loss
from metrics import compute_val_tail_metrics_torch
from model import create_model


def build_optimizer(model: torch.nn.Module) -> optim.Optimizer:
    """Adam or AdamW with config-driven hyperparameters."""
    opt_name = config.OPTIMIZER.lower()
    kwargs = {
        "lr": config.LEARNING_RATE,
        "betas": (config.ADAM_BETA1, config.ADAM_BETA2),
        "eps": config.ADAM_EPS,
        "weight_decay": config.WEIGHT_DECAY,
        "amsgrad": config.AMSGRAD,
    }
    if opt_name == "adamw":
        return optim.AdamW(model.parameters(), **kwargs)
    if opt_name == "adam":
        return optim.Adam(model.parameters(), **kwargs)
    raise ValueError(f"Unsupported OPTIMIZER={config.OPTIMIZER!r}")


def _use_amp() -> bool:
    return config.USE_AMP and config.DEVICE.type == "cuda"


def train_model(train_loader, val_loader):
    model = create_model().to(config.DEVICE)

    if config.TORCH_COMPILE and hasattr(torch, "compile"):
        model = torch.compile(model)

    dummy = torch.randn(1, config.IN_CHANNELS, config.NZ_MAX, config.NX).to(
        config.DEVICE
    )
    assert model(dummy).shape == (1, config.NX, config.N_FREQ)

    amp_enabled = _use_amp()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    autocast = torch.autocast(device_type="cuda", enabled=amp_enabled)
    non_blocking = amp_enabled

    criterion = build_training_loss()
    optimizer = build_optimizer(model)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=20, factor=0.5
    )

    run = wandb.init(
        project=config.WANDB_PROJECT,
        name=config.WANDB_RUN_NAME,
        config={k: v for k, v in vars(config).items() if k.isupper()},
    )

    best_val_loss = float("inf")
    early_stop_counter = 0
    freq = np.load(config.TF_FREQ_PATH)

    t = trange(config.NUM_EPOCHS, desc="Training")
    for epoch in t:
        model.train()
        train_loss = 0.0
        n_train = 0
        for inputs, targets, masks in train_loader:
            inputs = inputs.to(config.DEVICE, non_blocking=non_blocking)
            targets = targets.to(config.DEVICE, non_blocking=non_blocking)
            masks = masks.to(config.DEVICE, non_blocking=non_blocking)

            optimizer.zero_grad(set_to_none=True)
            with autocast:
                outputs = model(inputs)
                loss = criterion(outputs, targets, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)
            n_train += inputs.size(0)

        train_loss /= max(n_train, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for inputs, targets, masks in val_loader:
                inputs = inputs.to(config.DEVICE, non_blocking=non_blocking)
                targets = targets.to(config.DEVICE, non_blocking=non_blocking)
                masks = masks.to(config.DEVICE, non_blocking=non_blocking)
                with autocast:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets, masks)
                val_loss += loss.item() * inputs.size(0)
                n_val += inputs.size(0)

        val_loss /= max(n_val, 1)
        scheduler.step(val_loss)

        val_tail = compute_val_tail_metrics_torch(
            model, val_loader, config.DEVICE, freq
        )

        log_payload = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **val_tail,
        }
        wandb.log(log_payload)
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
