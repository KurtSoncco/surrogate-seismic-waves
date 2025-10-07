# train.py
"""Training script for the DeepONet model with optional PI-Loss for extrapolation."""

import config
import torch
import torch.nn as nn
import torch.optim as optim
from model import DeepONetModel, build_deeponet_components
from tqdm import trange

import wandb
from wave_surrogate.logging_setup import setup_logging

logger = setup_logging()


def train_model(train_loader, val_loader):
    """Main training and validation loop for DeepONet."""

    # 1. Initialize Networks
    branch, trunk = build_deeponet_components(config)
    branch.to(config.DEVICE)
    trunk.to(config.DEVICE)

    # 2. Combine into DeepONet Model
    model = DeepONetModel(
        branch=branch, trunk=trunk, output_size=config.OUTPUT_SIZE
    ).to(config.DEVICE)

    # Check forward pass with dummy data (Vs, Frequencies)
    dummy_vs = torch.randn(1, 1, config.INPUT_SIZE).to(config.DEVICE)
    # Frequencies are the third item in the data loader output (taken from the batch)
    dummy_freq = next(iter(train_loader))[2][0].to(config.DEVICE).unsqueeze(0)
    dummy_output = model(dummy_vs, dummy_freq)
    assert dummy_output.shape == (1, config.OUTPUT_SIZE), (
        f"Output shape error: {dummy_output.shape}"
    )

    # 3. Setup Training Components
    criterion_data = nn.L1Loss()  # Data loss is L1 for robustness
    # criterion_pi = nn.MSELoss()  # PI-Loss is also MSE on the residual

    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=config.EARLY_STOP_PATIENCE // 3, factor=0.9
    )

    # 4. Initialize W&B
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
        train_loss_data = 0.0
        train_loss_total = 0.0

        # Data loader returns: Vs_profile, TTF_target, Frequencies
        for inputs, targets, freqs in train_loader:
            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
            freqs = freqs.to(config.DEVICE)  # Frequencies for the Trunk

            outputs = model(inputs, freqs)

            # 1. Data Loss (Supervised Loss)
            loss_data = criterion_data(outputs, targets)

            # 2. Physics-Informed Loss (for Extrapolation)
            # NOTE: For reliable extrapolation, the governing PDE residual L_PDE
            # would be computed here using auto-diff on 'outputs' w.r.t 'freqs' or space.
            loss_pi = torch.tensor(0.0).to(config.DEVICE)  # Placeholder
            if config.PI_LOSS_WEIGHT > 0.0:
                # --- Placeholder for L_PDE Implementation ---
                # loss_pi = compute_pde_residual(outputs, inputs, freqs)
                # ---------------------------------------------
                pass

            # Total Loss
            loss = loss_data + config.PI_LOSS_WEIGHT * loss_pi

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)

            optimizer.step()

            # Note: inputs.size(0) is the actual batch size for the current batch
            train_loss_data += loss_data.item() * inputs.size(0)
            train_loss_total += loss.item() * inputs.size(0)

        # len(train_loader.dataset) correctly gets the total number of samples in the subset
        train_loss_data /= len(train_loader.dataset)
        train_loss_total /= len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, freqs in val_loader:
                inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
                freqs = freqs.to(config.DEVICE)
                outputs = model(inputs, freqs)
                loss = criterion_data(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        # --- Logging and Early Stopping ---
        wandb.log(
            {
                "epoch": epoch,
                "train_loss_data": train_loss_data,
                "train_loss_total": train_loss_total,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "pi_loss_weight": config.PI_LOSS_WEIGHT,
            }
        )

        t.set_postfix(
            train_data_loss=f"{train_loss_data:.6f}", val_loss=f"{val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= config.EARLY_STOP_PATIENCE:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # wandb.finish()
    logger.info(f"Finished training. Best validation loss: {best_val_loss:.6f}")

    return run
