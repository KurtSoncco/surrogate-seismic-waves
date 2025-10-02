# run_experiment_pred_only.py
"""
Experiment 2: Can the residual be predicted from the previous Prediction ONLY?
"""

import config
import numpy as np
import torch
from data_loader import get_primary_loaders
from experiment_models import SimplifiedPredOnlyModel
from experiment_trainer import run_experiment_training_loop
from specboost import TowerEncoder
from torch.utils.data import DataLoader, Dataset
from train_specboost import generate_predictions_for_split

from wave_surrogate.logging_setup import setup_logging
from wave_surrogate.models.fno.model import EncoderOperatorModel, OperatorDecoder

logger = setup_logging()
logger.name = "SpecBoost_Experiment_PredOnly"


class PredOnlyResidualDataset(Dataset):
    """Dataset that provides only the prediction as input and the normalized residual as target."""

    def __init__(self, original_dataset, cumulative_preds, residual_mean, residual_std):
        self.cumulative_preds = torch.tensor(cumulative_preds, dtype=torch.float32)
        # We still need original_dataset to get the true targets for residuals
        self.original_dataset = original_dataset
        self.residual_mean = residual_mean
        self.residual_std = residual_std

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        _, original_target = self.original_dataset[idx]
        cumulative_pred = self.cumulative_preds[idx]
        residual = original_target - cumulative_pred
        normalized_residual = (residual - self.residual_mean) / (
            self.residual_std + 1e-8
        )
        # Input is just the prediction, target is the normalized residual
        return cumulative_pred, normalized_residual


def main():
    logger.info("Loading data and pre-trained Model A...")
    # This setup is copied from your main.py
    # NOTE: You might need to adjust paths if this file is in a different directory
    import pickle

    with open(config.TTF_PICKLE_PATH, "rb") as f:
        ttf_data = np.array(pickle.load(f))
    with open(config.VS_PICKLE_PATH, "rb") as f:
        vs_profiles = np.array(pickle.load(f), dtype=object)

    from wave_surrogate.models.fno.utils import f0_calc

    f0_values = np.array([f0_calc(profile) for profile in vs_profiles])
    keep_indices = np.where(f0_values < config.F0_FILTER_THRESHOLD)[0]
    vs_profiles_filtered = vs_profiles[keep_indices]
    ttf_data_filtered = ttf_data[keep_indices]

    vs_list = [arr for arr in vs_profiles_filtered]
    ttf_list = [arr for arr in ttf_data_filtered]

    _, _, _, train_dataset, val_dataset = get_primary_loaders(
        vs_list, ttf_list, config.BATCH_SIZE
    )

    # Load Model A
    encoder_a = TowerEncoder(
        channels=config.ENCODER_CHANNELS_A, latent_dim=config.LATENT_DIM
    ).to(config.DEVICE)
    decoder_a = OperatorDecoder(
        latent_dim=config.LATENT_DIM,
        output_size=config.OUTPUT_SIZE,
        fno_modes=config.FNO_MODES,
        fno_width=config.FNO_WIDTH,
    ).to(config.DEVICE)
    model_a = EncoderOperatorModel(encoder=encoder_a, decoder=decoder_a).to(
        config.DEVICE
    )
    model_a.load_state_dict(torch.load(config.MODEL_A_SAVE_PATH))

    # Generate predictions to calculate residuals
    logger.info("Generating predictions from Model A...")
    train_preds_a = generate_predictions_for_split(model_a, train_dataset)
    val_preds_a = generate_predictions_for_split(model_a, val_dataset)

    # Calculate residual stats from training set
    train_targets = np.array(
        [train_dataset[i][1].numpy() for i in range(len(train_dataset))]  # type: ignore
    )
    train_residuals = train_targets - train_preds_a
    residual_mean = float(train_residuals.mean())
    residual_std = float(train_residuals.std())

    # Create datasets and dataloaders
    train_exp_dataset = PredOnlyResidualDataset(
        train_dataset, train_preds_a, residual_mean, residual_std
    )
    val_exp_dataset = PredOnlyResidualDataset(
        val_dataset, val_preds_a, residual_mean, residual_std
    )

    train_loader = DataLoader(
        train_exp_dataset, batch_size=config.BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_exp_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )

    # Initialize the simplified model
    model = SimplifiedPredOnlyModel(
        pred_encoder_channels=config.PRED_ENCODER_CHANNELS,
        latent_dim=config.FUSION_LATENT_DIM,
        fusion_hidden_dims=config.FUSION_HIDDEN_DIMS,
        output_size=config.OUTPUT_SIZE,
        dropout=config.FUSION_DROPOUT,
    )

    # Run the training experiment
    run_experiment_training_loop(
        model,
        train_loader,
        val_loader,
        learning_rate=config.BOOSTING_LEARNING_RATE,
        experiment_name="Prediction_Only",
    )


if __name__ == "__main__":
    main()
