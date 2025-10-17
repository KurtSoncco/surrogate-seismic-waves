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
    correlation_array,
    num_plots=6,
    title_prefix="",
    save_path=None,
):
    """Plots a random selection of test predictions against ground truth."""
    # For a 2x3 grid, ensure we don't request more plots than available
    num_plots = min(num_plots, 6)
    rng = np.random.default_rng(42)
    random_indices = rng.choice(len(test_predictions), num_plots, replace=False)

    # Define grid dimensions for clarity
    nrows, ncols = 2, 3

    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(random_indices):
        ax = plt.subplot(nrows, ncols, i + 1)
        ax.plot(
            freq_data,
            test_targets[idx],
            "-",
            label="Ground Truth",
            alpha=0.7,
            color="gray",
        )
        ax.plot(freq_data, test_predictions[idx], "r--", label="Prediction", alpha=0.7)

        # --- Robust handling of test_inputs[idx] (was np.trim_zeros(...)) ---
        vs_arr = np.asarray(test_inputs[idx])
        if vs_arr.ndim > 1:
            vs_arr = vs_arr[0]
        vs_arr = np.nan_to_num(vs_arr, nan=0.0)
        vs_test = np.trim_zeros(vs_arr, trim="b")
        if vs_test.size == 0:
            vs_test = vs_arr
        if vs_test.size >= 2:
            vs1, vs2 = (vs_test[0], vs_test[-1])
        elif vs_test.size == 1:
            vs1, vs2 = (vs_test[0], 0.0)
        else:
            vs1, vs2 = (0.0, 0.0)

        h_soil = max((len(vs_test) - 1) * 5, 0)
        ax.set_title(
            f"Vs1={vs1:.1f}, Vs2={vs2:.1f}, h={h_soil:.0f}m\n$\\rho$={correlation_array[idx]:.2f}",
            fontsize=14,
        )
        ax.set_xscale("log")
        ax.grid(True, which="both", linestyle="--")
        ax.set_xlim(freq_data.min(), freq_data.max())
        ax.set_ylim(0, 55)
        if i == 0:
            ax.legend()

        # --- MODIFICATION START ---
        # Only add labels to the outer plots to avoid repetition.

        # Add Y-axis label to plots in the first column
        if i % ncols == 0:
            ax.set_ylabel("Transfer Function", fontsize=14)

        # Add X-axis label to plots in the bottom row
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Frequency (Hz)", fontsize=14)
        # --- MODIFICATION END ---

    plt.tight_layout(rect=(0, 0, 1, 0.95))  # Adjusted rect to better fit suptitle
    plt.suptitle(title_prefix + "Model Predictions vs. Ground Truth", fontsize=20)

    if save_path:
        # Make sure the directory exists
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, title_prefix + "predictions.png"), dpi=300)
        # logger.info(f"Saved prediction plots to {save_path}")
        print(
            f"Saved prediction plots to {save_path}"
        )  # Using print as logger is not defined
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
    test_targets: np.ndarray,
    test_predictions: np.ndarray,
    title_prefix: str = "",
    save_path=None,
):
    """Calculates and plots a histogram of Pearson correlation coefficients."""

    # Calculate correlations
    correlations = [pearsonr(t, p)[0] for t, p in zip(test_targets, test_predictions)]
    correlations = np.array(correlations)
    correlations = correlations[np.isfinite(correlations)]

    plt.figure(figsize=(10, 6))

    # --- Plot the Histogram ---
    plt.hist(correlations, bins=30, alpha=0.75)
    plt.xlabel("Pearson Correlation Coefficient")
    plt.ylabel("Frequency")
    plt.title(title_prefix + "Distribution of Pearson Correlation Coefficients")
    plt.grid(True, axis="y")

    # --- Plot the Mean Line ---
    mean_corr = float(np.mean(correlations))
    median_corr = float(np.median(correlations))
    std_corr = float(np.std(correlations))

    plt.axvline(mean_corr, color="red", linestyle="dashed", linewidth=1)

    # --- Calculate and Format Statistics ---
    stats_text = (
        f"Mean: {mean_corr:.4f}\nMedian: {median_corr:.4f}\nStd: {std_corr:.4f}"
    )

    # --- Use Axes Coordinates for Boxed Text (Recommended) ---
    # Placing the text in the top-right corner of the plot area
    # x=0.98, y=0.98 are relative to the axes.
    # The 'ha' and 'va' set the alignment of the text box
    plt.gca().text(
        0.85,  # x-position: 85% from the left of the axes
        0.98,  # y-position: 98% from the bottom of the axes
        stats_text,
        transform=plt.gca().transAxes,  # Use axes coordinates
        color="red",
        fontsize=10,
        fontweight="bold",
        verticalalignment="top",  # Top of the text box is aligned with y=0.98
        horizontalalignment="right",  # Right side of the text box is aligned with x=0.98
        bbox=dict(
            boxstyle="round,pad=0.5",  # Box style (e.g., round corners)
            facecolor="wheat",  # Box background color
            alpha=0.6,  # Box transparency
        ),
    )

    if save_path:
        # Assuming 'logger' and 'os' are properly imported and configured
        plt.savefig(
            os.path.join(save_path, title_prefix + "pearson_histogram.png"), dpi=300
        )
        # logger.info(f"Saved Pearson histogram to {save_path}")
    else:
        plt.show()


def plot_correlation_vs_parameters(
    vs_soil_x,
    vs_bedrock_x,
    h_soil_x,
    correlation_factor,
    title_prefix="FNO Model",
    save_path=None,
):
    """
    Generates a three-panel figure showing correlation vs. soil parameters,
    matching the provided image style.
    """

    # Calculate the overall mean correlation for the legend
    mean_corr = np.mean(correlation_factor)

    plt.figure(figsize=(15, 5))  # Adjusted to match the aspect ratio of the image

    plot_data = [
        (
            vs_soil_x,
            r"$Vs_{1}$ [m/s]",
            r"Correlation - $Vs_{1}$ [m/s]",
        ),
        (
            vs_bedrock_x,
            r"$Vs_{2}$ [m/s]",
            r"Correlation - $Vs_{2}$ [m/s]",
        ),
        (
            h_soil_x,
            r"$h_{soil}$ [m]",
            r"Correlation - $h_{soil}$ [m]",
        ),
    ]

    for i, (x_data, x_label, plot_title) in enumerate(plot_data):
        ax = plt.subplot(1, 3, i + 1)

        # Scatter Plot
        ax.scatter(x_data, correlation_factor, s=15, color="darkorange", alpha=0.7)

        # Set Plot Limits and Labels
        ax.set_ylim(0.1, 1.05)
        ax.set_xlim(x_data.min() * 0.95, x_data.max() * 1.05)  # Dynamic x-limits

        ax.set_xlabel(x_label, fontsize=14)
        ax.set_title(plot_title, fontsize=20)

        # Only set ylabel on the first subplot
        if i == 0:
            ax.set_ylabel("Correlation coefficients", fontsize=14)

        # Grid and Ticks
        ax.grid(True, linestyle=":", alpha=0.5, which="both")
        ax.set_yticks(np.arange(0.2, 1.2, 0.2))
        ax.set_xticks(ax.get_xticks())  # Keep auto-generated major ticks

        # Mean Line and Label using Axes Coordinates (matching the image style)

        # 1. Dashed line at the mean correlation value
        ax.axhline(
            mean_corr,
            color="darkorange",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {mean_corr:.3f}",
        )

        # 2. Text in a box (using axes coordinates for consistent placement)
        # Position: 0.98 (right edge) and ~0.15 (bottom edge)
        ax.text(
            0.98,
            0.15,
            f"Mean: {mean_corr:.3f}",
            transform=ax.transAxes,  # Use axes coordinates
            color="darkorange",
            fontsize=14,
            verticalalignment="center",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="square,pad=0.3",
                facecolor="white",
                alpha=0.8,
                edgecolor="darkorange",  # Use the line color for the border
                linestyle="--",  # Use a dashed border
            ),
        )

        # Subplot letter centered below the x-label, e.g., "(a)", "(b)", "(c)"
        letter = chr(ord("a") + i)
        # y coordinate is negative in axes fraction so it's placed below the x-axis label.
        ax.text(
            0.5,
            -0.22,
            f"({letter})",
            transform=ax.transAxes,
            fontsize=14,
            ha="center",
            va="top",
            color="black",
        )

    plt.suptitle(
        f"{title_prefix} - Correlation vs. Soil Parameters", fontsize=20, y=1.0
    )
    plt.tight_layout()

    # --- Save or Show Plot ---
    if save_path:
        plt.savefig(
            os.path.join(save_path, title_prefix + "correlation_vs_parameters.png"),
            dpi=300,
        )
        logger.info(f"Saved correlation vs parameters plot to {save_path}")
    else:
        plt.show()
