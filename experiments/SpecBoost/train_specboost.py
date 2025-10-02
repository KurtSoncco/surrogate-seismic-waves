"""Training script for the SpecBoost framework with Fusion Towers."""

from contextlib import contextmanager
from typing import Optional

import config
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import NormalizedResidualDataset, get_residual_loaders
from specboost import ResidualFNOModel, SpecBoostModelB, TowerEncoder
from tqdm import tqdm, trange

import wandb
from wave_surrogate.logging_setup import setup_logging
from wave_surrogate.models.fno.model import (
    EncoderOperatorModel,
    OperatorDecoder,
)

logger = setup_logging()


class NRMSELoss(nn.Module):
    """
    Normalized Root Mean Square Error loss function.
    As described in the SpecB-FNO paper (Eq. 2).
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        # Calculate the L2 norm for each item in the batch
        norm_true = torch.linalg.norm(y_true, dim=1, keepdim=True)
        norm_pred_error = torch.linalg.norm(y_pred - y_true, dim=1, keepdim=True)

        # Normalize the error and compute the mean over the batch
        loss = torch.mean(norm_pred_error / (norm_true + self.eps))

        return loss


@contextmanager
def wandb_run(project: str, name: str, model: nn.Module):
    """Context manager for safe wandb initialization and cleanup."""
    run = None
    try:
        run = wandb.init(
            project=project,
            name=name,
            settings=wandb.Settings(),
        )
        wandb.watch(model, log="all")
        yield run
    except Exception as e:
        logger.error(f"Error during wandb run: {e}")
        raise
    finally:
        if run is not None:
            try:
                wandb.finish(quiet=True)
            except Exception as e:
                logger.warning(f"Error finishing wandb run: {e}")


class TrainingMetrics:
    """Helper class to track and log training metrics."""

    def __init__(self, run_name: str):
        self.run_name = run_name
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0

    def update(self, val_loss: float) -> bool:
        """Update metrics and return True if this is the best model so far."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.early_stop_counter = 0
            return True
        else:
            self.early_stop_counter += 1
            return False

    def should_stop(self) -> bool:
        """Check if early stopping criteria is met."""
        return self.early_stop_counter >= config.EARLY_STOP_PATIENCE

    def log_wandb(
        self, epoch: int, train_loss: float, val_loss: float, lr: Optional[float] = None
    ):
        """Log metrics to wandb."""
        log_dict = {
            "epoch": epoch,
            f"train_loss_{self.run_name}": train_loss,
            f"val_loss_{self.run_name}": val_loss,
        }
        if lr is not None:
            log_dict["learning_rate"] = lr
        wandb.log(log_dict)


def train_stage_one(train_loader, val_loader) -> EncoderOperatorModel:
    """Train the first model (Model A) on the original data."""
    logger.info("=== SpecBoost Stage 1: Model A ===")

    # Initialize Model A
    # encoder_a = Encoder(
    #    channels=config.ENCODER_CHANNELS_A, latent_dim=config.LATENT_DIM
    # ).to(config.DEVICE)

    encoder_a = TowerEncoder(
        channels=config.ENCODER_CHANNELS_A,
        latent_dim=config.LATENT_DIM,
        kernel_size=config.ENCODER_KERNEL_SIZE,
        pool_size=config.ENCODER_POOL_SIZE,
        use_adaptive_pool=False,
        dropout=config.DROPOUT_RATE,
    ).to(config.DEVICE)

    decoder_a = OperatorDecoder(
        latent_dim=config.LATENT_DIM,
        output_size=config.OUTPUT_SIZE,
        fno_modes=config.FNO_MODES,
        fno_width=config.FNO_WIDTH,
    ).to(config.DEVICE)

    model = EncoderOperatorModel(encoder=encoder_a, decoder=decoder_a).to(config.DEVICE)

    # Initialize lazy modules with a dummy forward pass
    dummy_input = torch.randn(1, 1, 29).to(
        config.DEVICE
    )  # Batch size 1, channels 1, length 29
    with torch.no_grad():
        _ = model(dummy_input)

    criterion = NRMSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=80, factor=0.9
    )

    metrics = TrainingMetrics("Model_A_Solo")

    with wandb_run(config.WANDB_PROJECT, "Model_A_Solo", model):
        t = trange(config.NUM_EPOCHS, desc="Training Model A")

        for epoch in t:
            # Training
            model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                inputs = inputs.to(config.DEVICE).unsqueeze(1)  # Add channel dim
                targets = targets.to(config.DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.GRAD_CLIP_NORM
                )
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)

            train_loss /= len(train_loader.dataset)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(config.DEVICE).unsqueeze(1)  # Add channel dim
                    targets = targets.to(config.DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)

            val_loss /= len(val_loader.dataset)
            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            metrics.log_wandb(epoch, train_loss, val_loss, current_lr)
            t.set_postfix(train_loss=train_loss, val_loss=val_loss, lr=current_lr)

            if metrics.update(val_loss):
                torch.save(model.state_dict(), config.MODEL_A_SAVE_PATH)

            if metrics.should_stop():
                logger.info(
                    f"Early stopping triggered for Model A at epoch {epoch + 1}"
                )
                break

    logger.info(
        f"Finished training Model A. Best validation loss: {metrics.best_val_loss:.6f}"
    )

    # Load best model
    model.load_state_dict(
        torch.load(config.MODEL_A_SAVE_PATH, map_location=config.DEVICE)
    )
    return model


def generate_predictions_for_split(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    previous_preds: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generate predictions, handling both Model A and residual models.

    For Model A: previous_preds should be None.
    For Model B: previous_preds contains cumulative predictions from earlier stages.
    """

    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )

    all_preds = []
    with torch.no_grad():
        for idx, (inputs, _) in enumerate(tqdm(loader, desc="Generating predictions")):
            inputs = inputs.to(config.DEVICE)

            if previous_preds is None:
                # Model A: standard forward pass
                outputs = model(inputs.unsqueeze(1))
            else:
                # Residual model: needs both inputs
                batch_start = idx * config.BATCH_SIZE
                batch_end = batch_start + inputs.size(0)
                prev_batch = torch.tensor(
                    previous_preds[batch_start:batch_end], dtype=torch.float32
                ).to(config.DEVICE)
                outputs = model(inputs, prev_batch)

            all_preds.append(outputs.cpu().numpy())

    return np.vstack(all_preds)


def generate_denormalized_predictions(
    model: ResidualFNOModel,
    dataset: torch.utils.data.Dataset,
    previous_preds: np.ndarray,
    residual_mean: float,
    residual_std: float,
) -> np.ndarray:
    """
    Generate predictions from a ResidualFNOModel and denormalize them.
    """
    model.eval()
    # Create a simple dataset for prediction
    pred_dataset = torch.utils.data.TensorDataset(
        torch.tensor(previous_preds, dtype=torch.float32)
    )
    loader = torch.utils.data.DataLoader(
        pred_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )

    all_preds = []
    with torch.no_grad():
        for (pred_inputs,) in tqdm(loader, desc="Generating denormalized predictions"):
            pred_inputs = pred_inputs.to(config.DEVICE)

            # Model outputs normalized residuals
            normalized_outputs = model(pred_inputs)

            # Denormalize
            outputs = normalized_outputs * residual_std + residual_mean
            all_preds.append(outputs.cpu().numpy())

    return np.vstack(all_preds)


def train_stage_two(
    train_dataset, val_dataset, train_preds_a, val_preds_a, model_b=None
) -> SpecBoostModelB:
    """
    Train the second model (Model B) using dual-tower fusion architecture.

    Model B Architecture:
    - Tower 1: Encodes Vs profile (1, 29,) → latent_dim
    - Tower 2: Encodes Model A prediction (1000,) → latent_dim
    - Fusion Head: Concatenates latents → predicts residual (1000,)

    Args:
        train_dataset: Original training dataset
        val_dataset: Original validation dataset
        train_preds_a: Predictions from Model A on training set
        val_preds_a: Predictions from Model A on validation set

    Returns:
        Trained Model B with fusion towers
    """
    logger.info("=== SpecBoost Stage 2: Model B (Fusion Towers) ===")

    # Create residual datasets
    train_loader, val_loader = get_residual_loaders(
        train_dataset, val_dataset, train_preds_a, val_preds_a, config.BATCH_SIZE
    )

    # Initialize Model B with fusion towers
    # Use config values or defaults
    vs_channels = getattr(config, "VS_ENCODER_CHANNELS", [1, 32, 64, 128])
    pred_channels = getattr(config, "PRED_ENCODER_CHANNELS", [1, 64, 128, 256])
    latent_dim = getattr(config, "FUSION_LATENT_DIM", 128)
    fusion_hidden = getattr(config, "FUSION_HIDDEN_DIMS", [256, 512])
    dropout = getattr(config, "FUSION_DROPOUT", 0.2)

    if model_b is not None:
        logger.info("Using provided Model B instance.")
        model = model_b.to(config.DEVICE)
    else:
        logger.info("Initializing new Model B instance.")
        model = SpecBoostModelB(
            vs_encoder_channels=vs_channels,
            pred_encoder_channels=pred_channels,
            latent_dim=latent_dim,
            output_size=config.OUTPUT_SIZE,
            fusion_hidden_dims=fusion_hidden,
            dropout=dropout,
        ).to(config.DEVICE)

    # Initialize lazy modules with a dummy forward pass
    dummy_vs = torch.randn(1, 29).to(
        config.DEVICE
    )  # Batch size 1, channel 1, length 29
    dummy_pred = torch.randn(1, config.OUTPUT_SIZE).to(
        config.DEVICE
    )  # Batch size 1, output size
    with torch.no_grad():
        _ = model(dummy_vs, dummy_pred)

    logger.info("Model B architecture:")
    logger.info(f"  - Vs tower channels: {vs_channels}")
    logger.info(f"  - Pred tower channels: {pred_channels}")
    logger.info(f"  - Latent dimension: {latent_dim}")
    logger.info(f"  - Fusion hidden dims: {fusion_hidden}")

    criterion = NRMSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=80, factor=0.9
    )

    metrics = TrainingMetrics("Model_B_Residual")

    with wandb_run(config.WANDB_PROJECT, "Model_B_Residual", model):
        t = trange(config.NUM_EPOCHS, desc="Training Model B")

        for epoch in t:
            # Training
            model.train()
            train_loss = 0.0
            for (vs_profiles, model_a_preds), residuals in train_loader:
                vs_profiles = vs_profiles.to(config.DEVICE)
                model_a_preds = model_a_preds.to(config.DEVICE)
                residuals = residuals.to(config.DEVICE)

                # Forward pass through fusion towers
                outputs = model(vs_profiles, model_a_preds)
                loss = criterion(outputs, residuals)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.GRAD_CLIP_NORM
                )
                optimizer.step()

                train_loss += loss.item() * vs_profiles.size(0)

            train_loss /= len(train_loader.dataset)  # type: ignore

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (vs_profiles, model_a_preds), residuals in val_loader:
                    vs_profiles = vs_profiles.to(config.DEVICE)
                    model_a_preds = model_a_preds.to(config.DEVICE)
                    residuals = residuals.to(config.DEVICE)

                    outputs = model(vs_profiles, model_a_preds)
                    loss = criterion(outputs, residuals)
                    val_loss += loss.item() * vs_profiles.size(0)

            val_loss /= len(val_loader.dataset)  # type: ignore
            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            metrics.log_wandb(epoch, train_loss, val_loss, current_lr)
            t.set_postfix(train_loss=train_loss, val_loss=val_loss, lr=current_lr)

            if metrics.update(val_loss):
                torch.save(model.state_dict(), config.MODEL_B_SAVE_PATH)

            if metrics.should_stop():
                logger.info(
                    f"Early stopping triggered for Model B at epoch {epoch + 1}"
                )
                break

    logger.info(
        f"Finished training Model B. Best validation loss: {metrics.best_val_loss:.6f}"
    )

    # Load best model
    model.load_state_dict(
        torch.load(config.MODEL_B_SAVE_PATH, map_location=config.DEVICE)
    )
    return model


def train_boosting_stage(
    stage_num: int,
    train_dataset,
    val_dataset,
    previous_stage_preds_train,
    previous_stage_preds_val,
    model_config: dict,
) -> SpecBoostModelB:
    """
    Train a single boosting stage.

    Args:
        stage_num: Current stage (1, 2, 3, ...)
        previous_stage_preds_train: Cumulative predictions from all previous stages
        previous_stage_preds_val: Validation predictions from previous stages
    """
    logger.info(f"=== SpecBoost Stage {stage_num + 1}: Model {chr(65 + stage_num)} ===")

    # Create residual datasets for this stage
    train_loader, val_loader = get_residual_loaders(
        train_dataset,
        val_dataset,
        previous_stage_preds_train,
        previous_stage_preds_val,
        config.BATCH_SIZE,
    )

    # Initialize new model for this stage
    model = SpecBoostModelB(**model_config).to(config.DEVICE)

    # Dummy forward pass
    dummy_vs = torch.randn(2, config.INPUT_SIZE).to(config.DEVICE)
    dummy_pred = torch.randn(2, config.OUTPUT_SIZE).to(config.DEVICE)
    _ = model(dummy_vs, dummy_pred)

    # Standard training loop (same as before)
    criterion = NRMSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.BOOSTING_LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=80, factor=0.9
    )

    metrics = TrainingMetrics(f"Model_{chr(65 + stage_num)}_Stage_{stage_num + 1}")

    model_save_path = (
        config.MODEL_SAVE_PATH / f"specboost_model_{chr(65 + stage_num)}.pt"
    )

    with wandb_run(
        config.WANDB_PROJECT,
        f"Model_{chr(65 + stage_num)}_Stage_{stage_num + 1}",
        model,
    ):
        t = trange(config.NUM_EPOCHS, desc=f"Training Stage {stage_num + 1}")

        for epoch in t:
            # Training
            model.train()
            train_loss = 0.0
            for (vs_profiles, cumulative_preds), residuals in train_loader:
                vs_profiles = vs_profiles.to(config.DEVICE)
                cumulative_preds = cumulative_preds.to(config.DEVICE)
                residuals = residuals.to(config.DEVICE)

                outputs = model(vs_profiles, cumulative_preds)
                loss = criterion(outputs, residuals)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.GRAD_CLIP_NORM
                )
                optimizer.step()

                train_loss += loss.item() * vs_profiles.size(0)

            train_loss /= len(train_loader.dataset)  # type: ignore

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (vs_profiles, cumulative_preds), residuals in val_loader:
                    vs_profiles = vs_profiles.to(config.DEVICE)
                    cumulative_preds = cumulative_preds.to(config.DEVICE)
                    residuals = residuals.to(config.DEVICE)

                    outputs = model(vs_profiles, cumulative_preds)
                    loss = criterion(outputs, residuals)
                    val_loss += loss.item() * vs_profiles.size(0)

            val_loss /= len(val_loader.dataset)  # type: ignore
            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            metrics.log_wandb(epoch, train_loss, val_loss, current_lr)
            t.set_postfix(train_loss=train_loss, val_loss=val_loss, lr=current_lr)

            if metrics.update(val_loss):
                torch.save(model.state_dict(), model_save_path)

            if metrics.should_stop():
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(torch.load(model_save_path, map_location=config.DEVICE))
    return model


def train_boosting_stage_normalized(
    stage_num: int,
    train_dataset,
    val_dataset,
    previous_stage_preds_train,
    previous_stage_preds_val,
    model_config: dict,
) -> tuple[ResidualFNOModel, float, float]:
    """
    Train a boosting stage with the ResidualFNOModel and residual normalization.

    Returns:
        tuple: (trained_model, residual_mean, residual_std)
    """
    logger.info(f"=== SpecBoost Stage {stage_num + 1}: Model {chr(65 + stage_num)} ===")

    # Calculate residual statistics for normalization
    train_targets = np.array(
        [train_dataset[i][1].numpy() for i in range(len(train_dataset))]
    )
    train_residuals = train_targets - previous_stage_preds_train

    residual_mean = float(train_residuals.mean())
    residual_std = float(train_residuals.std())

    logger.info("Residual statistics before normalization:")
    logger.info(f"  Mean: {residual_mean:.6f}, Std: {residual_std:.6f}")

    # Create normalized datasets
    train_dataset_norm = NormalizedResidualDataset(
        train_dataset, previous_stage_preds_train, residual_mean, residual_std
    )
    val_dataset_norm = NormalizedResidualDataset(
        val_dataset, previous_stage_preds_val, residual_mean, residual_std
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset_norm, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset_norm, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True
    )

    # Instantiate the winning ResidualFNOModel
    model = ResidualFNOModel(**model_config).to(config.DEVICE)

    # Dummy forward pass
    dummy_pred = torch.randn(1, config.OUTPUT_SIZE).to(config.DEVICE)
    with torch.no_grad():
        _ = model(dummy_pred)

    # Use MSE loss, which was validated in our experiments
    criterion = nn.L1Loss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.BOOSTING_LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        patience=10,
        factor=0.5,  # A slightly more aggressive scheduler
    )

    metrics = TrainingMetrics(f"Model_{chr(65 + stage_num)}_Stage_{stage_num + 1}")
    model_save_path = (
        config.MODEL_SAVE_PATH / f"specboost_model_{chr(65 + stage_num)}.pt"
    )

    with wandb_run(
        config.WANDB_PROJECT,
        f"Model_{chr(65 + stage_num)}_Stage_{stage_num + 1}",
        model,
    ):
        t = trange(config.NUM_EPOCHS, desc=f"Training Stage {stage_num + 1}")

        for epoch in t:
            model.train()
            train_loss = 0.0
            for pred_inputs, normalized_residuals in train_loader:
                pred_inputs = pred_inputs.to(config.DEVICE)
                normalized_residuals = normalized_residuals.to(config.DEVICE)

                outputs = model(pred_inputs)
                loss = criterion(outputs, normalized_residuals)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.GRAD_CLIP_NORM
                )
                optimizer.step()
                train_loss += loss.item() * pred_inputs.size(0)

            train_loss /= len(train_loader.dataset)  # type: ignore

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for pred_inputs, normalized_residuals in val_loader:
                    pred_inputs = pred_inputs.to(config.DEVICE)
                    normalized_residuals = normalized_residuals.to(config.DEVICE)
                    outputs = model(pred_inputs)
                    loss = criterion(outputs, normalized_residuals)
                    val_loss += loss.item() * pred_inputs.size(0)

            val_loss /= len(val_loader.dataset)  # type: ignore
            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            metrics.log_wandb(epoch, train_loss, val_loss, current_lr)
            t.set_postfix(train_loss=train_loss, val_loss=val_loss, lr=current_lr)

            if metrics.update(val_loss):
                torch.save(model.state_dict(), model_save_path)

            if metrics.should_stop():
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(torch.load(model_save_path, map_location=config.DEVICE))
    logger.info(f"Finished training. Best val loss: {metrics.best_val_loss:.6f}")

    return model, residual_mean, residual_std
