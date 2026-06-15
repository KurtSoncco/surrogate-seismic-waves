import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

from wave_surrogate.logging_setup import setup_logging

logger = setup_logging()
sns.set_palette("colorblind")


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for (soil_profile, depth_grid), tf in tqdm(
            dataloader, desc="Validation", leave=False
        ):
            soil_profile, depth_grid, tf = (
                soil_profile.to(device),
                depth_grid.to(device),
                tf.to(device),
            )
            tf_pred = model(soil_profile, depth_grid)
            loss = criterion(tf_pred, tf)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def test_model(model, dataloader, device, scaler: Optional[Any] = None):
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for (soil_profile, depth_grid), tf in tqdm(
            dataloader, desc="Testing", leave=False
        ):
            soil_profile, depth_grid, tf = (
                soil_profile.to(device),
                depth_grid.to(device),
                tf.to(device),
            )
            tf_pred = model(soil_profile, depth_grid).cpu().numpy()

            # Inverse transform to get metrics in original scale
            if scaler:
                tf_pred = scaler.inverse_transform(tf_pred)
                tf = scaler.inverse_transform(tf.cpu().numpy())

            all_preds.append(tf_pred)
            all_true.append(tf)

    all_preds = np.concatenate(all_preds, axis=0)
    all_true = np.concatenate(all_true, axis=0)

    # Calculate metrics
    r2 = r2_score(all_true, all_preds)
    mae = mean_absolute_error(all_true, all_preds)
    rmse = np.sqrt(mean_squared_error(all_true, all_preds))

    metrics = {"R2": r2, "MAE": mae, "RMSE": rmse}

    # Return predictions for plotting
    return metrics, all_preds, all_true


def plot_test_predictions(preds, true, num_samples=5, save_path=None):
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, num_samples * 3))
    indices = np.random.choice(len(preds), num_samples, replace=False)

    for i, idx in enumerate(indices):
        ax = axes[i]
        ax.plot(true[idx], "b-", label="Ground Truth")
        ax.plot(preds[idx], "r--", label="Prediction")
        ax.set_title(f"Test Sample #{idx}")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "test_predictions.png"))
        logger.info(f"Saved prediction plot to {save_path}")
    plt.show()
