import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from dataloader import get_tf_dataloaders
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from test import test_model_dae  # type: ignore
from train import train_stage1, train_stage2
from utils import TrainingConfig, init_wandb

from wave_surrogate.logging_setup import setup_logging
from wave_surrogate.models.dae import DecoupledAutoencoder
from wave_surrogate.models.dae.architectures import Decoder, Encoder, WeakerDecoder

# --- Setup ---
load_dotenv()  # Load environment variables from .env file
logger = setup_logging()


def set_seed(seed: int):
    """Sets the seed for reproducibility across all relevant libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # The following two lines are often needed for full reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_data(
    config: TrainingConfig,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    MinMaxScaler,
]:
    """
    Loads the dataset, normalizes it, and returns dataloaders for train, validation, and test sets.

    Args:
        config: The training configuration object.

    Returns:
        A tuple containing the train, validation, and test DataLoader objects.
    """
    config = TrainingConfig()
    dataset_file = config.tf_data_path

    try:
        # Load the dataset
        data = pd.read_parquet(dataset_file)
        tf_profiles_list = data["model_data"].tolist()
        logger.info(
            f"Loaded and converted {len(tf_profiles_list)} profiles from {dataset_file}."
        )
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {dataset_file}")
        raise

    return get_tf_dataloaders(
        tf_profiles=tf_profiles_list,
        batch_size=config.batch_size,
        save_path=config.model_dir,
        config=config,
    )


def initialize_model(config: TrainingConfig) -> DecoupledAutoencoder:
    """
    Initializes the DecoupledAutoencoder model and its components.

    Args:
        config: The training configuration object.

    Returns:
        The initialized DecoupledAutoencoder model, moved to the correct device.
    """

    # 1. Instantiate the Encoder
    # It takes the input dimension, the list of hidden dims, and the latent dim.
    encoder = Encoder(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim_encoder,
        latent_dim=config.latent_dim,
    )

    # 2. Instantiate the main, powerful Decoder
    # It mirrors the encoder's structure but in reverse.
    decoder = Decoder(
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim_decoder,
        output_dim=config.input_dim,
    )

    # 3. Instantiate the weaker Auxiliary Decoder
    # It has the same architecture but includes dropout for regularization.
    aux_decoder = WeakerDecoder(
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim_aux_decoder,
        output_dim=config.input_dim,
        dropout_rate=config.dropout_rate,  # Dropout makes it "weaker"
    )

    # 4. Assemble the DecoupledAutoencoder
    # The DAE class will manage which decoder to use.
    model = DecoupledAutoencoder(
        encoder=encoder, decoder=decoder, aux_decoder=aux_decoder
    )

    logger.info("✅ DecoupledAutoencoder model built successfully!")
    logger.info(f"   - Encoder: {encoder}")
    logger.info(f"   - Decoder: {decoder}")
    logger.info(f"   - Aux Decoder: {aux_decoder}")

    return model.to(config.device)


def run_pipeline():
    """Main function to run the entire training and testing pipeline."""
    config = TrainingConfig()
    logger.info(f"Using device: {config.device}")
    set_seed(config.seed)
    Path(config.model_dir).mkdir(parents=True, exist_ok=True)

    # --- Data and Model ---
    train_loader, val_loader, test_loader, scaler = load_data(config)
    model = initialize_model(config)

    # --- Training ---
    run = init_wandb(config, config.run_name)
    train_stage1(model, train_loader, val_loader, config, run)

    train_stage2(model, train_loader, val_loader, config, run)
    logger.info("--- Training Complete ---")

    # --- Saving Final Model ---
    # Recommended: Save state dictionaries, not the whole model object
    model_dir = Path(config.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = model_dir / "encoder.pth"
    decoder_path = model_dir / "decoder.pth"

    torch.save(model.encoder.state_dict(), encoder_path)
    torch.save(model.decoder.state_dict(), decoder_path)
    logger.info(f"Final models saved to {model_dir}")

    # --- Testing ---
    # For a robust test, always load the saved model from disk
    logger.info("\n--- Loading persisted model for final testing ---")

    # Initialize a new model instance for loading
    test_model_instance = initialize_model(config)
    test_model_instance.encoder.load_state_dict(torch.load(encoder_path))
    test_model_instance.decoder.load_state_dict(torch.load(decoder_path))
    # Note: We don't load aux_decoder as it's typically not used for final inference

    # Testing the model
    Path(config.results_path).mkdir(parents=True, exist_ok=True)

    test_model_dae(test_model_instance, test_loader, config, scaler, run)
    logger.info("--- Testing Complete ---")
    # Finish W&B run
    run.finish() if run else None


def check_test():
    """Function to check if the test runs without executing the full pipeline."""
    config = TrainingConfig()
    logger.info(f"Using device: {config.device}")
    set_seed(config.seed)
    Path(config.model_dir).mkdir(parents=True, exist_ok=True)

    # --- Data and Model ---
    _, _, test_loader, scaler = load_data(config)
    # model = initialize_model(config)

    # --- Testing ---
    # For a robust test, always load the saved model from disk
    logger.info("\n--- Loading persisted model for final testing ---")

    # Initialize a new model instance for loading
    test_model_instance = initialize_model(config)
    encoder_path = Path(config.model_dir) / "encoder.pth"
    decoder_path = Path(config.model_dir) / "decoder.pth"

    test_model_instance.encoder.load_state_dict(torch.load(encoder_path))
    test_model_instance.decoder.load_state_dict(torch.load(decoder_path))
    # Note: We don't load aux_decoder as it's typically not used for final inference

    # Testing the model
    Path(config.results_path).mkdir(parents=True, exist_ok=True)

    test_model_dae(test_model_instance, test_loader, config, scaler, run=None)
    logger.info("--- Testing Complete ---")


if __name__ == "__main__":
    run_pipeline()
