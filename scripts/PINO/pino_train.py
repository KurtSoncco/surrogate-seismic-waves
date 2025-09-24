# pino_train.py
"""
Training script for the v2 PINO with variable density and a hybrid loss approach.
"""

import numpy as np
import pino_config as config
import torch
import torch.nn as nn
import torch.optim as optim
from physics_loss import WaveEquationLoss
from pino_model import PINO
from tqdm import trange

import wandb
from wave_surrogate.logging_setup import setup_logging
from wave_surrogate.models.fno.model import Encoder
from wave_surrogate.ttf import TTF  # Import your trusted TTF function for validation

logger = setup_logging()


def get_spatiotemporal_grid(batch_size):
    """Creates the (z, t) grid required by the PINO."""
    t = torch.linspace(0, config.T_MAX, config.TIMESTEPS, device=config.DEVICE)
    # z is normalized to [0, 1], representing bottom to top of the soil column
    z = torch.linspace(0, 1, config.SPATIAL_POINTS, device=config.DEVICE)
    grid_z, grid_t = torch.meshgrid(z, t, indexing="ij")
    grid = torch.stack((grid_z, grid_t), dim=-1)
    return grid.unsqueeze(0).expand(batch_size, -1, -1, -1)


def ricker_wavelet(t, f=1.5):
    """
    Generate a Ricker wavelet to serve as a standard input ground motion.
    This provides the necessary forcing function for the time-domain simulation.
    """
    t_centered = t - 2.0 / f  # Center the wavelet
    pi2_f2_t2 = (np.pi * f * t_centered) ** 2
    return (1.0 - 2.0 * pi2_f2_t2) * np.exp(-pi2_f2_t2)


def train_pino_model(train_loader, val_loader, freq_data):
    """Main training and validation loop for the PINO."""
    # 1. --- Model, Loss, and Optimizer Initialization ---
    vs_encoder = Encoder(
        channels=config.ENCODER_CHANNELS, latent_dim=config.LATENT_DIM
    ).to(config.DEVICE)

    model = PINO(
        vs_encoder=vs_encoder,
        latent_dim=config.LATENT_DIM,
        fno_width=config.FNO_WIDTH,
        fno_modes=config.FNO_MODES,
        num_fno_layers=config.NUM_FNO_LAYERS,
    ).to(config.DEVICE)

    data_criterion = nn.MSELoss()
    physics_criterion = WaveEquationLoss(
        layer_thickness=config.LAYER_THICKNESS,
        total_depth=config.INPUT_SIZE * config.LAYER_THICKNESS,
        total_time=config.T_MAX,
        device=config.DEVICE,
    )
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=100)

    # Initialize lazy parameters by running a dummy forward pass
    with torch.no_grad():
        dummy_vs_rho = torch.randn(1, 2, config.INPUT_SIZE, device=config.DEVICE)
        dummy_motion = torch.randn(1, config.TIMESTEPS, device=config.DEVICE)
        dummy_grid = get_spatiotemporal_grid(1)
        _ = model(dummy_vs_rho, dummy_motion, dummy_grid)

    wandb.init(
        project=config.WANDB_PROJECT,
        name=config.WANDB_RUN_NAME,
        config={k: v for k, v in vars(config).items() if k.isupper()},
    )
    wandb.watch(model, log="all")

    # 2. --- Prepare Standard Inputs for Simulation ---
    time_pts = np.linspace(0, config.T_MAX, config.TIMESTEPS)
    input_motion_np = ricker_wavelet(time_pts)
    input_motion = torch.tensor(input_motion_np, dtype=torch.float32).to(config.DEVICE)
    base_accel_np = np.gradient(
        np.gradient(input_motion_np, config.DT), config.DT
    )  # Pre-calculate for validation

    best_val_loss = float("inf")
    t = trange(config.NUM_EPOCHS, desc="Training PINO v2")
    for epoch in t:
        # 3. --- Training Phase ---
        model.train()
        train_data_loss, train_pde_loss = 0.0, 0.0
        for vs_rho_profiles, targets_ttf in train_loader:
            vs_rho_profiles, targets_ttf = (
                vs_rho_profiles.to(config.DEVICE),
                targets_ttf.to(config.DEVICE),
            )

            # --- Forward Pass ---
            grid = get_spatiotemporal_grid(vs_rho_profiles.shape[0])
            grid.requires_grad = True  # Essential for autograd in physics loss
            u_pred = model(
                vs_rho_profiles,
                input_motion.expand(vs_rho_profiles.shape[0], -1),
                grid,
            )

            # --- Differentiable Data Loss (for Gradient signal) ---
            u_surface = u_pred[:, -1, :]
            u_bedrock = u_pred[:, 0, :]
            U_surface_diff = torch.fft.rfft(u_surface, dim=1)
            U_bedrock_diff = torch.fft.rfft(u_bedrock, dim=1)
            # Simple ratio for a differentiable loss proxy
            T_pred_diff = torch.abs(U_surface_diff) / (torch.abs(U_bedrock_diff) + 1e-8)
            # Interpolate to target frequencies
            # target_freq = torch.tensor(freq_data, device=config.DEVICE)
            # T_pred_diff has shape (B, T_fft). Add a channel dim for interpolate.
            T_pred_reshaped = T_pred_diff.unsqueeze(1)  # Shape: (B, 1, T_fft)

            # Interpolate the entire batch at once
            T_pred_resampled = torch.nn.functional.interpolate(
                T_pred_reshaped,
                size=targets_ttf.shape[1],
                mode="linear",
                align_corners=False,  # Often recommended for 1D interpolation
            ).squeeze(1)  # Shape: (B, num_target_freqs)
            loss_data = data_criterion(T_pred_resampled, targets_ttf)

            # --- Physics Loss ---
            loss_pde = physics_criterion(u_pred, vs_rho_profiles, grid)

            # --- Combined Loss and Backward Pass ---
            total_loss = loss_data + config.PHYSICS_LOSS_WEIGHT * loss_pde
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()

            train_data_loss += loss_data.item()
            train_pde_loss += loss_pde.item()

        # 4. --- Validation Phase ---
        model.eval()
        val_mse_numpy = 0.0
        with torch.no_grad():
            for vs_rho_profiles, targets_ttf in val_loader:
                vs_rho_profiles, targets_ttf = (
                    vs_rho_profiles.to(config.DEVICE),
                    targets_ttf.to(config.DEVICE),
                )

                # --- Forward pass for validation ---
                grid = get_spatiotemporal_grid(vs_rho_profiles.shape[0])
                u_pred = model(
                    vs_rho_profiles,
                    input_motion.expand(vs_rho_profiles.shape[0], -1),
                    grid,
                )

                # --- Validation using your non-differentiable TTF function ---
                # Convert to numpy and calculate acceleration from displacement
                surface_disp_np = u_pred[:, -1, :].cpu().numpy()
                surface_accel_np = np.gradient(
                    np.gradient(surface_disp_np, config.DT, axis=1),
                    config.DT,
                    axis=1,
                )

                for i in range(vs_rho_profiles.shape[0]):
                    # Use your trusted function for an apples-to-apples comparison
                    _, tf_pred_numpy = TTF(
                        surface_accel_np[i],
                        base_accel_np,
                        dt=config.DT,
                        n_points=targets_ttf.shape[1],
                    )
                    val_mse_numpy += np.mean(
                        (tf_pred_numpy - targets_ttf[i].cpu().numpy()) ** 2
                    )

        # 5. --- Logging and Model Checkpointing ---
        avg_train_data_loss = train_data_loss / len(train_loader)
        avg_train_pde_loss = train_pde_loss / len(train_loader)
        avg_val_mse_numpy = val_mse_numpy / len(val_loader.dataset)

        scheduler.step(avg_val_mse_numpy)

        wandb.log(
            {
                "epoch": epoch,
                "train_data_loss": avg_train_data_loss,
                "train_pde_loss": avg_train_pde_loss,
                "val_mse_numpy": avg_val_mse_numpy,  # This is our primary validation metric
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        t.set_postfix(
            data_loss=f"{avg_train_data_loss:.4e}",
            pde_loss=f"{avg_train_pde_loss:.4e}",
            val_mse=f"{avg_val_mse_numpy:.4e}",
        )

        if avg_val_mse_numpy < best_val_loss:
            best_val_loss = avg_val_mse_numpy
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            logger.info(
                f"New best model saved at epoch {epoch} with validation MSE: {best_val_loss:.6f}"
            )

    wandb.finish()
    logger.info(f"Finished training. Best validation MSE: {best_val_loss:.6f}")


if __name__ == "__main__":
    ## Let's do a simple one epoch training run to test everything works
    import pickle

    import pino_config as config
    from data_loader import get_data_loaders

    # Load the datasets
    vs_profiles = pickle.load(open(config.VS_PICKLE_PATH, "rb"))
    rho_profiles = pickle.load(open(config.RHO_PICKLE_PATH, "rb"))
    ttf_data = pickle.load(open(config.TTF_PICKLE_PATH, "rb"))
    freq_data = np.loadtxt(config.FREQ_PATH)

    train_loader, val_loader, _ = get_data_loaders(
        vs_profiles=vs_profiles,
        rho_profiles=rho_profiles,
        ttf_data=ttf_data,
        batch_size=config.BATCH_SIZE,
    )

    # Print shape of each batch to verify
    for vs_rho_profiles, targets_ttf in train_loader:
        logger.info(f"Vs/Rho profiles batch shape: {vs_rho_profiles.shape}")
        logger.info(f"TTF targets batch shape: {targets_ttf.shape}")
        break  # Just check the first batch

    # Create a simple model instance
    vs_encoder = Encoder(
        channels=config.ENCODER_CHANNELS, latent_dim=config.LATENT_DIM
    ).to(config.DEVICE)

    model = PINO(
        vs_encoder=vs_encoder,
        latent_dim=config.LATENT_DIM,
        fno_width=config.FNO_WIDTH,
        fno_modes=config.FNO_MODES,
        num_fno_layers=config.NUM_FNO_LAYERS,
    ).to(config.DEVICE)

    data_criterion = nn.MSELoss()
    physics_criterion = WaveEquationLoss(
        layer_thickness=config.LAYER_THICKNESS,
        total_depth=config.INPUT_SIZE * config.LAYER_THICKNESS,
        total_time=config.T_MAX,
        device=config.DEVICE,
    )
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Prepare standard inputs
    time_pts = np.linspace(0, config.T_MAX, config.TIMESTEPS)
    input_motion_np = ricker_wavelet(time_pts)
    input_motion = torch.tensor(input_motion_np, dtype=torch.float32).to(config.DEVICE)

    # Run a single training epoch
    model.train()
    for vs_rho_profiles, targets_ttf in train_loader:
        vs_rho_profiles, targets_ttf = (
            vs_rho_profiles.to(config.DEVICE),
            targets_ttf.to(config.DEVICE),
        )

        # Forward Pass
        grid = get_spatiotemporal_grid(vs_rho_profiles.shape[0])
        grid.requires_grad = True
        u_pred = model(
            vs_rho_profiles,
            input_motion.expand(vs_rho_profiles.shape[0], -1),
            grid,
        )

        # Data Loss
        u_surface = u_pred[:, -1, :]
        u_bedrock = u_pred[:, 0, :]
        U_surface_diff = torch.fft.rfft(u_surface, dim=1)
        U_bedrock_diff = torch.fft.rfft(u_bedrock, dim=1)
        T_pred_diff = torch.abs(U_surface_diff) / (torch.abs(U_bedrock_diff) + 1e-8)
        # Interpolate to target frequencies
        fft_freq = torch.fft.rfftfreq(config.TIMESTEPS, config.DT, device=config.DEVICE)
        target_freq = torch.tensor(freq_data, device=config.DEVICE)
        # T_pred_diff has shape (B, T_fft). Add a channel dim for interpolate.
        T_pred_reshaped = T_pred_diff.unsqueeze(1)  # Shape: (B, 1, T_fft)

        # Interpolate the entire batch at once
        T_pred_resampled = torch.nn.functional.interpolate(
            T_pred_reshaped,
            size=target_freq.shape[0],
            mode="linear",
            align_corners=False,  # Often recommended for 1D interpolation
        ).squeeze(1)  # Shape: (B, num_target_freqs)

        loss_data = data_criterion(T_pred_resampled, targets_ttf)

        # Physics Loss
        loss_pde = physics_criterion(u_pred, vs_rho_profiles, grid)

        # Combined Loss and Backward Pass
        total_loss = loss_data + config.PHYSICS_LOSS_WEIGHT * loss_pde
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
        optimizer.step()

        logger.info(
            f"Sample loss_data: {loss_data.item():.4e}, loss_pde: {loss_pde.item():.4e}"
        )
        break  # Just one batch for testing

    logger.info("Single epoch training test completed successfully.")
