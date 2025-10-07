# utils.py
"""Utility functions for plotting and analysis."""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr

from wave_surrogate.logging_setup import setup_logging

logger = setup_logging()

sns.set_palette("colorblind")


def f0_calc(vs_profile):
    """
    Calculates the fundamental frequency (f0) from a Vs profile.
    Assumes each layer is 5m thick and the last layer is bedrock.
    f0 = Vs1 / (4 * H_soil)

    Args:
        vs_profile (np.ndarray): Array of Vs values for the profile.
    Returns:
        float: Calculated fundamental frequency f0.
    """
    # Clean the array by removing any trailing zeros/nans
    vs_cleaned = np.nan_to_num(vs_profile, nan=0.0)
    vs_cleaned = np.trim_zeros(vs_cleaned, trim="b")

    if len(vs_cleaned) <= 1:
        return np.inf  # Return a large number if profile is invalid

    vs_value_1 = vs_cleaned[0]
    # The soil profile is all but the last element (bedrock)
    h_soil = (len(vs_cleaned) - 1) * 5.0

    if h_soil == 0:
        return np.inf  # Avoid division by zero

    return vs_value_1 / (4 * h_soil)


def plot_predictions(
    freq_data,
    test_targets,
    test_predictions,
    test_inputs,
    num_plots=9,
    title_prefix="",
    save_path=None,
):
    """Plots a random selection of test predictions against ground truth."""
    random_indices = np.random.choice(len(test_predictions), num_plots, replace=False)

    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(random_indices):
        plt.subplot(3, 3, i + 1)
        plt.plot(freq_data, test_targets[idx], "b-", label="Ground Truth", alpha=0.7)
        plt.plot(freq_data, test_predictions[idx], "r-", label="Prediction", alpha=0.7)

        # --- Robust handling of test_inputs[idx] (was np.trim_zeros(...)) ---
        # Ensure we work with a numpy array
        vs_arr = np.asarray(test_inputs[idx])

        # If inputs are multi-channel (e.g., [channels, depth]), assume Vs is the first channel
        if vs_arr.ndim > 1:
            vs_arr = vs_arr[0]

        # Replace NaNs with zeros so trim_zeros can remove trailing padding
        vs_arr = np.nan_to_num(vs_arr, nan=0.0)

        # Trim trailing zeros (padding). If the entire array is zeros, fallback to zeros.
        vs_test = np.trim_zeros(vs_arr, trim="b")
        if vs_test.size == 0:
            # fallback: use the original (non-trimmed) array to avoid indexing errors
            vs_test = vs_arr

        # Safely extract first & last values
        if vs_test.size >= 2:
            vs1, vs2 = (vs_test[0], vs_test[-1])
        elif vs_test.size == 1:
            vs1, vs2 = (vs_test[0], 0.0)
        else:
            vs1, vs2 = (0.0, 0.0)

        h_soil = max((len(vs_test) - 1) * 5, 0)
        plt.title(f"Vs1={vs1:.1f}, Vs2={vs2:.1f}, h={h_soil:.0f}m")
        plt.xscale("log")
        plt.grid(True, which="both", linestyle="--")
        if i == 0:
            plt.legend()
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Transfer Function")

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.suptitle(title_prefix + "Model Predictions vs. Ground Truth", fontsize=16)

    if save_path:
        plt.savefig(os.path.join(save_path, title_prefix + "predictions.png"), dpi=300)
        logger.info(f"Saved prediction plots to {save_path}")
    else:
        plt.show()


def plot_correlation(test_targets, test_predictions, title_prefix="", save_path=None):
    """Plots a scatter plot of predictions vs. ground truth."""
    plt.figure(figsize=(8, 8))
    plt.scatter(test_targets.flatten(), test_predictions.flatten(), alpha=0.1, s=2)
    min_val = min(np.min(test_targets), np.min(test_predictions))
    max_val = max(np.max(test_targets), np.max(test_predictions))
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    plt.title(title_prefix + "Prediction vs. Ground Truth Correlation")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)
    plt.axis("equal")

    if save_path:
        plt.savefig(os.path.join(save_path, title_prefix + "correlation.png"), dpi=300)
        logger.info(f"Saved correlation plot to {save_path}")
    else:
        plt.show()


def plot_pearson_histogram(
    test_targets, test_predictions, title_prefix="", save_path=None
):
    """Calculates and plots a histogram of Pearson correlation coefficients."""
    correlations = [pearsonr(t, p)[0] for t, p in zip(test_targets, test_predictions)]
    correlations = np.array(correlations)
    correlations = correlations[np.isfinite(correlations)]

    plt.figure(figsize=(10, 6))
    plt.hist(correlations, bins=30, alpha=0.75, color="blue")
    plt.xlabel("Pearson Correlation Coefficient")
    plt.ylabel("Frequency")
    plt.title(title_prefix + "Distribution of Pearson Correlation Coefficients")
    plt.grid(True, axis="y")
    plt.axvline(
        float(np.mean(correlations)), color="red", linestyle="dashed", linewidth=1
    )
    plt.text(
        float(np.mean(correlations)) + 0.01,
        plt.ylim()[1] * 0.9,
        f"Mean: {float(np.mean(correlations)):.4f}\nMedian: {float(np.median(correlations)):.4f}\nStd: {float(np.std(correlations)):.4f}",
        color="red",
    )
    if save_path:
        plt.savefig(
            os.path.join(save_path, title_prefix + "pearson_histogram.png"), dpi=300
        )
        logger.info(f"Saved Pearson histogram to {save_path}")
    else:
        plt.show()
