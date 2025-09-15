# evaluate.py
"""Evaluates the trained FNO model on the test set and plots results."""

import config
import numpy as np
import torch
import torch.nn as nn
from model import Encoder, EncoderOperatorModel, OperatorDecoder
from tqdm import tqdm
from utils import plot_correlation, plot_pearson_histogram, plot_predictions

from wave_surrogate.logging_setup import setup_logging

logger = setup_logging()


def evaluate_model(test_loader, freq_data):
    """Evaluates the model and generates plots."""
    encoder = Encoder(
        channels=config.ENCODER_CHANNELS,
        latent_dim=config.LATENT_DIM,
        kernel_size=config.ENCODER_KERNEL_SIZE,
        pool_size=config.ENCODER_POOL_SIZE,
    ).to(config.DEVICE)

    decoder = OperatorDecoder(
        latent_dim=config.LATENT_DIM,
        output_size=config.OUTPUT_SIZE,
        fno_modes=config.FNO_MODES,
        fno_width=config.FNO_WIDTH,
    ).to(config.DEVICE)

    model = EncoderOperatorModel(encoder=encoder, decoder=decoder).to(config.DEVICE)

    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    model.eval()
    criterion = nn.MSELoss()

    test_loss = 0.0
    all_predictions, all_targets, all_inputs = [], [], []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_inputs.append(inputs.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    logger.info(f"Test MSE Loss: {test_loss:.6f}")

    # Concatenate results from all batches
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    all_inputs = np.vstack(all_inputs)

    # Generate plots
    plot_predictions(freq_data, all_targets, all_predictions, all_inputs)
    plot_correlation(all_targets, all_predictions)
    plot_pearson_histogram(all_targets, all_predictions)
