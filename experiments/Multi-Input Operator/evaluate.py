# evaluate.py
"""Evaluates the trained DeepONet model on the test set and plots results."""

import os

import config
import joblib
import numpy as np
import torch
import torch.nn as nn
from model import DeepONetModel, build_deeponet_components
from scipy.stats import pearsonr
from tqdm import tqdm

from wave_surrogate.logging_setup import setup_logging
from wave_surrogate.models.fno.utils import (
    plot_correlation,
    plot_pearson_histogram,
    plot_predictions,
)

logger = setup_logging()


def _safe_load_state(model: torch.nn.Module, path: str, device: str):
    """Load a saved model or state_dict safely onto device."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    loaded = torch.load(path, map_location=device)
    # If the saved object is a state_dict (mapping of param tensors)
    if isinstance(loaded, dict):
        # Many training scripts save {'state_dict': state, ...} or raw state dict
        if "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            state = loaded["state_dict"]
            model.load_state_dict(state)
        else:
            # assume loaded is the state_dict
            try:
                model.load_state_dict(loaded)
            except RuntimeError:
                # Could be a full checkpoint with different keys; try to find state_dict-like entry
                for v in ("model_state", "model", "state"):
                    if v in loaded and isinstance(loaded[v], dict):
                        model.load_state_dict(loaded[v])
                        break
                else:
                    raise
    else:
        # loaded might be a full model object (not recommended). Try to extract state_dict
        try:
            state = loaded.state_dict()
            model.load_state_dict(state)
        except Exception:
            # As a last resort, return the loaded object (caller must handle)
            return loaded
    return model


def evaluate_model(test_loader, freq_data_1d, run=None):
    """Evaluates the model and generates plots.
    The evaluation tests in RMSE, Pearson correlation, MSE, MAE and visualizes the results.

    Args:
        test_loader (DataLoader): DataLoader for the test dataset.
        freq_data_1d (np.ndarray): 1D array of frequency data for plotting
        run: Optional W&B run object for logging. Default is None.
    """

    # 1. Initialize Networks
    branch, trunk = build_deeponet_components(config)
    branch.to(config.DEVICE)
    trunk.to(config.DEVICE)

    # 2. Combine into DeepONet Model
    model = DeepONetModel(
        branch=branch, trunk=trunk, output_size=config.OUTPUT_SIZE
    ).to(config.DEVICE)

    # 2. Load weights robustly
    logger.info(f"Loading model weights from: {config.MODEL_SAVE_PATH}")
    model = _safe_load_state(model, config.MODEL_SAVE_PATH, str(config.DEVICE))
    model = model.to(config.DEVICE)

    model.eval()
    criterion = nn.MSELoss(reduction="mean")

    test_loss = 0.0
    all_predictions, all_targets, all_inputs = [], [], []

    # Data loader returns: Vs_profile, TTF_target, Frequencies
    with torch.no_grad():
        for inputs, targets, freqs in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
            freqs = freqs.to(config.DEVICE)

            # Forward pass requires both Vs profile (inputs) and Frequencies (freqs)
            outputs = model(inputs, freqs)

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

    # Scale back
    scaler = joblib.load(config.MODEL_PARAM_SAVE_PATH + "/ttf_scaler.joblib")
    all_predictions = scaler.inverse_transform(all_predictions)
    all_targets = scaler.inverse_transform(all_targets)

    # Compute correlation
    correlations = [pearsonr(t, p)[0] for t, p in zip(all_targets, all_predictions)]
    correlations = np.array(correlations)
    correlations = correlations[np.isfinite(correlations)]

    # Use torch for MAE/MSE/RMSE calculation for consistency
    preds_tensor = torch.tensor(all_predictions)
    targets_tensor = torch.tensor(all_targets)
    mse_loss = nn.MSELoss()(preds_tensor, targets_tensor).item()
    mae_loss = nn.L1Loss()(preds_tensor, targets_tensor).item()

    # Send metrics to W&B if run is provided
    if run is not None:
        run.log(
            {
                "test_loss_MSE": mse_loss,
                "test_loss_RMSE": np.sqrt(mse_loss),
                "test_loss_MAE": mae_loss,
            }
        )
        run.log(
            {
                "test_pearson_mean": np.mean(correlations),
                "test_pearson_std": np.std(correlations),
                "test_pearson_min": np.min(correlations),
                "test_pearson_max": np.max(correlations),
            }
        )
    else:
        logger.info(
            f"Pearson Correlation - Mean: {np.mean(correlations):.4f}, Std: {np.std(correlations):.4f}, Min: {np.min(correlations):.4f}, Max: {np.max(correlations):.4f}"
        )
        logger.info(
            f"Test RMSE: {np.sqrt(mse_loss):.6f}, Test MAE: {mae_loss:.6f}, Test MSE: {mse_loss:.6f}"
        )

    # Generate plots
    plot_predictions(
        freq_data_1d,
        all_targets,
        all_predictions,
        all_inputs,
        title_prefix="DeepONet ",
        save_path=config.RESULTS_SAVE_PATH,
    )
    plot_correlation(
        all_targets,
        all_predictions,
        title_prefix="DeepONet ",
        save_path=config.RESULTS_SAVE_PATH,
    )
    plot_pearson_histogram(
        all_targets,
        all_predictions,
        title_prefix="DeepONet ",
        save_path=config.RESULTS_SAVE_PATH,
    )


if __name__ == "__main__":
    # Do a simple test of the evaluation function with dummy data
    import pickle

    from data_loader import get_data_loaders

    from wave_surrogate.logging_setup import setup_logging
    from wave_surrogate.models.fno.utils import f0_calc

    logger = setup_logging()
    logger.name = "Evaluate Test"

    # --- Load Data from Pickle and CSV ---
    logger.info("Loading data from original pickle files...")
    with open(config.TTF_PICKLE_PATH, "rb") as f:
        ttf_data = np.array(pickle.load(f))
    with open(config.VS_PICKLE_PATH, "rb") as f:
        vs_profiles = np.array(pickle.load(f))
    freq_data = np.genfromtxt(config.FREQ_PATH, delimiter=",")
    if len(freq_data.shape) == 1:
        freq_data = freq_data.reshape(-1, 1)

    logger.info(f"Loaded {len(vs_profiles)} profiles before filtering.")
    # --- Preprocessing: Filter based on f0 ---
    logger.info(f"Filtering profiles with f0 >= {config.F0_FILTER_THRESHOLD} Hz...")
    f0_values = np.array([f0_calc(profile) for profile in vs_profiles])
    keep_indices = np.where(f0_values < config.F0_FILTER_THRESHOLD)[0]
    vs_profiles_filtered = vs_profiles[keep_indices]

    ttf_data_filtered = ttf_data[keep_indices]
    logger.info(f"Kept {len(vs_profiles_filtered)} profiles after filtering.")
    vs_list = [arr for arr in vs_profiles_filtered]
    ttf_list = [arr for arr in ttf_data_filtered]
    train_loader, val_loader, test_loader = get_data_loaders(
        vs_list, ttf_list, freq_data, config.BATCH_SIZE
    )
    evaluate_model(test_loader, freq_data.squeeze())  # Pass 1D freq_data for plotting
