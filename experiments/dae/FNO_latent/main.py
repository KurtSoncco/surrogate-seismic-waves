# train_fno.py

import numpy as np
import torch
import torch.nn as nn
from dataset import SoilProfileDataset
from OTEncoder import OTEncoder
from test_model import evaluate, plot_test_predictions, test_model
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm, trange
from utils import FNOLatentConfig

import wandb
from wave_surrogate.logging_setup import setup_logging
from wave_surrogate.models.fno.model import FNO1D

logger = setup_logging()
logger.name = "FNO_latent_trainer"


class SeismicSurrogateModel(nn.Module):
    """
    A full surrogate model that:
    1. Encodes a variable-length profile to a fixed latent grid (OTEncoder)
    2. Lifts features to a high-dimensional space (Lifting)
    3. Processes the latent field (FNO1D)
    4. Decodes the field into a 1D time series (MLP Head)
    """

    def __init__(
        self,
        config: FNOLatentConfig,
        input_channels: int = 3,  # Vs, Vp, Rho
        output_dim: int = 500,  # e.g., 500-point time series
        device: str = "cuda",
    ):
        super().__init__()

        # --- 1. OTEncoder parameters ---
        self.latent_dim = config.latent_dim  # e.g., 16
        self.ot_steps = config.ot_steps
        self.ot_lr = config.ot_lr

        # --- 2. FNO parameters ---
        fno_modes = 8
        fno_width = 64  # CORRECT: This is the internal processing dimension

        # --- 1. The Encoder Layer ---
        # Takes (B, 3, N) + (B, N) -> (B, 16, 3)
        self.ot_encoder = OTEncoder(
            latent_dim=self.latent_dim,
            ot_steps=self.ot_steps,
            ot_lr=self.ot_lr,
            device=device,
        )

        # --- 2. The "Lifting" Layer ---
        # Lifts from input channels (3) to FNO processing width (64)
        # Takes (B, 16, 3) -> (B, 16, 64)
        self.lifting = nn.Linear(input_channels, fno_width)

        # --- 3. The "Core" Model (1D FNO) ---
        # Processes the high-dimensional latent field
        # Takes (B, 16, 64) -> (B, 16, 64)
        self.fno_processor = FNO1D(
            modes=fno_modes,
            width=fno_width,  # CORRECT: Use the 64, not 3
        ).to(device)

        # --- 4. The Prediction "Head" (MLP Decoder) ---
        # Maps the processed latent field to the final 1D time series.
        # Takes (B, 16, 64) -> (B, output_dim)
        self.prediction_head = nn.Sequential(
            # CORRECT: Flatten the entire output field, don't pool
            nn.Flatten(),  # Shape: (B, 16 * 64)
            # The input to the linear layer is now latent_dim * fno_width
            nn.Linear(self.latent_dim * fno_width, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),  # Shape: (B, output_dim)
        )

    def forward(
        self, input_features: torch.Tensor, input_grid: torch.Tensor
    ) -> torch.Tensor:
        # 1. Encode variable-length input to fixed-size latent tensor
        # Shape: (B, 3, N_in) -> (B, 16, 3)
        # (OTEncoder outputs (B, latent_dim, C) by default)
        latent_1 = self.ot_encoder(input_features, input_grid)

        # 2. Lift features to FNO's processing dimension
        # Shape: (B, 16, 3) -> (B, 16, 64)
        latent_lifted = self.lifting(latent_1)

        # 3. Process the latent field with the FNO
        # FNO1D expects (B, length, channels)
        # Shape: (B, 16, 64) -> (B, 16, 64)
        latent_2 = self.fno_processor(latent_lifted)

        # 4. Map features to final prediction
        # Shape: (B, 16, 64) -> (B, 16 * 64) -> (B, output_dim)
        prediction = self.prediction_head(latent_2)

        return prediction


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch=None):
    model.train()
    total_loss = 0.0
    desc = f"Train Epoch {epoch + 1}" if epoch is not None else "Train"

    for (soil_profile, depth_grid), tf in tqdm(dataloader, desc=desc):
        soil_profile, depth_grid, tf = (
            soil_profile.to(device),
            depth_grid.to(device),
            tf.to(device),
        )

        optimizer.zero_grad()
        tf_pred = model(soil_profile, depth_grid)
        loss = criterion(tf_pred, tf)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    config = FNOLatentConfig()

    # Setup WandB
    if config.use_wandb:
        wandb.init(project=config.wandb_project, name=config.run_name)

    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # 1. Create Dataset and Dataloaders
    full_dataset = SoilProfileDataset(config)

    ## Split dataset
    train_size = int(config.train_size * len(full_dataset))
    val_size = int(config.val_size * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.seed),
    )
    ## Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    logger.info(
        f"Data loaded: {train_size} training samples, {val_size} validation samples, {test_size} test samples."
    )

    # 2. Initialize Model, Optimizer, Criterion, Scheduler
    model = SeismicSurrogateModel(config).to(config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    criterion = nn.L1Loss()
    scheduler = ReduceLROnPlateau(
        optimizer, "min", patience=config.patience, factor=0.5
    )

    logger.info(f"Model initialized: {model}")

    logger.info("Starting FNO training...")
    early_stopping_counter = 0
    best_val_loss = float("inf")

    # 3. Training Loop
    for epoch in trange(config.epochs, desc="Epochs", leave=False):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, config.device, epoch
        )
        val_loss = evaluate(model, val_loader, criterion, config.device)

        if epoch % config.log_interval == 0 or epoch == config.epochs - 1:
            logger.info(
                f"Epoch {epoch + 1}/{config.epochs} | "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

        if config.use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

        # Checkpoint the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.checkpoint_path)
            logger.info(
                f"Validation loss improved. Saved model to {config.checkpoint_path}"
            )

            early_stopping_counter = 0  # Reset counter if improvement
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= config.early_stopping_patience:
                logger.info(
                    f"No improvement in validation loss for {config.early_stopping_patience} epochs. Early stopping."
                )
                break

        scheduler.step(val_loss)

    # 4. Final Evaluation on Test Set with new metrics
    logger.info("--- Testing Final Model ---")
    model.load_state_dict(torch.load(config.checkpoint_path))
    metrics, preds, true = test_model(
        model, test_loader, config.device, scaler=full_dataset.tf_scaler
    )

    logger.info(
        f"Test Metrics -> R2: {metrics['R2']:.4f} | MAE: {metrics['MAE']:.4f} | RMSE: {metrics['RMSE']:.4f}"
    )
    if config.use_wandb:
        wandb.log(
            {
                "test_r2": metrics["R2"],
                "test_mae": metrics["MAE"],
                "test_rmse": metrics["RMSE"],
            }
        )

    # 5. Plotting results
    plot_test_predictions(preds, true, num_samples=5, save_path=config.results_path)

    if config.use_wandb:
        wandb.finish()
    logger.info("Training complete.")


def check_data_loader():
    config = FNOLatentConfig()

    # Setup WandB
    config.use_wandb = False

    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # 1. Create Dataset and Dataloaders
    full_dataset = SoilProfileDataset(config)

    ## Split dataset
    train_size = int(config.train_size * len(full_dataset))
    val_size = int(config.val_size * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.seed),
    )
    ## Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    logger.info(
        f"Data loaded: {train_size} training samples, {val_size} validation samples, {test_size} test samples."
    )

    # 2. Initialize Model, Optimizer, Criterion, Scheduler
    model = SeismicSurrogateModel(config).to(config.device)

    logger.info(f"Created train loader with {len(train_loader)} batches.")
    logger.info(f"Created val loader with {len(val_loader)} batches.")
    logger.info(f"Created test loader with {len(test_loader)} batches.")
    sample_batch = next(iter(train_loader))
    (soil_profile, depth_grid), tf = sample_batch
    logger.info(
        f"Sample batch shapes - soil_profile: {soil_profile.shape}, depth_grid: {depth_grid.shape}, tf: {tf.shape}"
    )

    logger.info(f"Model initialized: {model}")

    # Plot one sample from the dataset
    import matplotlib.pyplot as plt

    (soil_sample, depth_sample), tf_sample = full_dataset[0]
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(depth_sample.numpy(), soil_sample[0].numpy(), "b-", label="Vs")
    plt.plot(depth_sample.numpy(), soil_sample[1].numpy(), "r-", label="Vp")
    plt.plot(depth_sample.numpy(), soil_sample[2].numpy(), "g-", label="Rho")
    plt.title("Sample Soil Profile")
    plt.xlabel("Depth (m)")
    plt.ylabel("Velocity/Density")
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(soil_sample[0].numpy())
    plt.title("Sample Vs Profile")
    plt.xlabel("Depth Index")
    plt.ylabel("Vs (m/s)")
    plt.subplot(1, 3, 3)
    plt.plot(tf_sample.numpy())
    plt.title("Sample Transfer Function")
    plt.xlabel("Frequency Index")
    plt.ylabel("TF Amplitude")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
