# evaluate.py
"""Evaluates the trained FNO model on the test set and plots results."""

import config
import numpy as np
import torch
import torch.nn as nn
from model import Encoder, EncoderOperatorModel, OperatorDecoder
from tqdm import tqdm
from utils import (
    f0_calc,
    plot_correlation,
    plot_correlation_vs_parameters,
    plot_pearson_histogram,
    plot_predictions,
)

from wave_surrogate.logging_setup import setup_logging

logger = setup_logging()


def evaluate_model(test_loader, freq_data, run=None):
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

    model.load_state_dict(
        torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    )
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

    # Concatenate results from all batches
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    all_inputs = np.vstack(all_inputs)

    ## Compute Pearson correlation coefficient
    pearson_corrs = [
        np.corrcoef(all_targets[i], all_predictions[i])[0, 1]
        for i in range(len(all_targets))
    ]
    mean_pearson = np.nanmean(pearson_corrs)

    if run is not None:
        run.log(
            {
                "test_mse_loss": test_loss,
                "test_MAE_loss": np.mean(np.abs(all_targets - all_predictions)),
                "test_R2_score": 1
                - (
                    np.sum((all_targets - all_predictions) ** 2)
                    / np.sum((all_targets - np.mean(all_targets)) ** 2)
                ),
            }
        )
        run.log(
            {
                "mean_pearson_correlation": mean_pearson,
                "median_pearson_correlation": np.nanmedian(pearson_corrs),
                "std_pearson_correlation": np.nanstd(pearson_corrs),
                "min_pearson_correlation": np.nanmin(pearson_corrs),
                "max_pearson_correlation": np.nanmax(pearson_corrs),
            }
        )
    logger.info(f"Test MSE Loss: {test_loss:.6f}")
    logger.info(f"Mean Pearson Correlation: {mean_pearson:.4f}")

    # Generate plots
    plot_predictions(
        freq_data,
        all_targets,
        all_predictions,
        all_inputs,
        correlation_array=pearson_corrs,
        save_path=config.RESULTS_SAVE_PATH,
    )
    plot_correlation(
        all_targets,
        all_predictions,
        save_path=config.RESULTS_SAVE_PATH,
    )
    plot_pearson_histogram(
        all_targets,
        all_predictions,
        save_path=config.RESULTS_SAVE_PATH,
    )

    ## Get soil parameters for correlation plots
    inputs_np = all_inputs
    num_samples = inputs_np.shape[0]
    vs_soil = np.zeros(num_samples, dtype=float)
    vs_bedrock = np.zeros(num_samples, dtype=float)
    h_soil = np.zeros(num_samples, dtype=float)

    for i in range(num_samples):
        arr = inputs_np[i]
        # If multi-channel, take first channel as Vs
        if arr.ndim > 1:
            vs = arr[0]
        else:
            vs = arr
        vs = np.nan_to_num(vs, nan=0.0)
        vs_trim = np.trim_zeros(vs, trim="b")
        if vs_trim.size == 0:
            # fallback to using full array
            vs_trim = vs

        vs_soil[i] = float(vs_trim[0]) if vs_trim.size >= 1 else 0.0
        vs_bedrock[i] = float(vs_trim[-1]) if vs_trim.size >= 1 else 0.0
        h_soil[i] = max((vs_trim.size - 1) * 5.0, 0.0)

    correlation_array = np.array(pearson_corrs)

    plot_correlation_vs_parameters(
        vs_soil,
        vs_bedrock,
        h_soil,
        correlation_array,
        save_path=config.RESULTS_SAVE_PATH,
    )


if __name__ == "__main__":
    # Example usage (requires data loaders and frequency data)
    import pickle

    from data_loader import get_data_loaders

    with open(config.TTF_PICKLE_PATH, "rb") as f:
        ttf_data = np.array(pickle.load(f))

    with open(config.VS_PICKLE_PATH, "rb") as f:
        vs_profiles = np.array(pickle.load(f))

    freq_data = np.genfromtxt(config.FREQ_PATH, delimiter=",")

    # --- Preprocessing: Filter based on f0 ---
    logger.info(f"Filtering profiles with f0 >= {config.F0_FILTER_THRESHOLD} Hz...")
    f0_values = np.array([f0_calc(profile) for profile in vs_profiles])

    # Get indices of profiles to keep
    keep_indices = np.where(f0_values < config.F0_FILTER_THRESHOLD)[0]

    # Filter the datasets
    vs_profiles_filtered = vs_profiles[keep_indices]
    ttf_data_filtered = ttf_data[keep_indices]

    logger.info(f"Kept {len(vs_profiles_filtered)} profiles after filtering.")

    _, _, test_loader = get_data_loaders(
        vs_profiles_filtered, ttf_data_filtered, config.BATCH_SIZE
    )

    evaluate_model(test_loader, freq_data)
    pass
