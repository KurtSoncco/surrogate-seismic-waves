from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from wave_surrogate.logging_setup import setup_logging

logger = setup_logging()
sns.set_theme(style="whitegrid", palette="colorblind")
plt.set_loglevel("WARNING")


def vs_profiles(
    df: pd.DataFrame, dz: float = 5.0, max_depth: float = 150.0
) -> pd.DataFrame:
    """
    Generates Vs profiles for each model using a fully vectorized approach.

    Each profile consists of a soil layer with Vs1 and thickness H1 overlying a
    bedrock layer with Vs2. The profiles are discretized with a grid spacing of dz.

    Args:
        df (pd.DataFrame): DataFrame with columns 'H1', 'Vs1', and 'Vs2'.
        dz (float): Grid spacing in meters.
        max_depth (float): Maximum depth for the profiles in meters.

    Returns:
        pd.DataFrame: A DataFrame where each row is a Vs profile and columns
                      represent Vs at each depth. The index is preserved from the input df.
    """
    logger.info("Generating Vs profiles for all models (vectorized)...")

    # 1. Define the depth grid for the profiles. Shape: (num_depths,)
    depth_levels = np.arange(0, max_depth + dz, dz)

    # 2. Extract model parameters as NumPy arrays and reshape to column vectors.
    # This changes their shape from (num_models,) to (num_models, 1)
    # to enable broadcasting against the depth_levels array.
    h_reshaped = df["H1"].to_numpy()[:, np.newaxis]
    vs1_reshaped = df["Vs1"].to_numpy()[:, np.newaxis]
    vs2_reshaped = df["Vs2"].to_numpy()[:, np.newaxis]

    # 3. Generate all profiles at once using NumPy broadcasting.
    # The comparison `depth_levels < h_reshaped` creates a boolean mask of shape
    # (num_models, num_depths). `np.where` then uses this mask to select
    # values from the broadcasted vs1 and vs2 arrays, returning the final
    # (num_models, num_depths) result in a single, fast operation.
    profiles = np.where(depth_levels <= h_reshaped, vs1_reshaped, vs2_reshaped)

    # 4. Create the final DataFrame with descriptive column names
    column_names = [f"Depth_{d:.0f}m" for d in depth_levels]
    profile_df = pd.DataFrame(profiles, columns=column_names, index=df.index)

    logger.info("Vs profile generation complete.")
    return profile_df


def calculate_time_averaged_vs(
    profiles_df: pd.DataFrame, z: float, dz: float = 5.0
) -> np.ndarray:
    """
    Calculates the time-averaged shear-wave velocity (e.g., Vs30) for each profile.

    This is calculated as z / Σ(h_i / Vs_i), where h_i is the thickness of each layer.

    Args:
        profiles_df (pd.DataFrame): DataFrame containing Vs profiles.
        z (float): The target depth for the calculation (e.g., 30 for Vs30).
        dz (float): The uniform grid spacing of the depth profile in meters.

    Returns:
        np.ndarray: A 1D array containing the time-averaged Vs for each profile.
    """
    if z <= 0 or z % dz != 0:
        raise ValueError(f"Depth z={z} must be a positive multiple of dz={dz}.")

    # 1. Select all depth columns from the surface up to the target depth z
    relevant_depths = np.arange(0, z, dz)
    relevant_cols = [f"Depth_{d:.0f}m" for d in relevant_depths]

    # Ensure all needed columns exist
    if not set(relevant_cols).issubset(profiles_df.columns):
        raise ValueError(f"Not all required depth columns for z={z}m were found.")

    vs_layers = profiles_df[relevant_cols].to_numpy()

    # 2. Calculate the total travel time for a wave to reach depth z
    # Travel time for each layer is thickness (dz) / velocity (Vs_i)
    travel_times = dz / vs_layers
    total_travel_time = np.sum(
        travel_times, axis=1
    )  # Sum across layers for each profile

    # 3. Calculate the time-averaged velocity
    vs_z = z / total_travel_time

    return vs_z


def plot_vs_z_distributions(
    profiles_df: pd.DataFrame,
    output_dir: Path,
    depths_to_plot: list = [30, 60, 90, 120],
    dz: float = 5.0,
    name: str = "comparison",
):
    """
    Plots the distribution of time-averaged Vs at several key depths.
    """
    logger.info("Plotting distributions of time-averaged Vs...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    ax_flat = axes.flatten()

    for ax, z in zip(ax_flat, depths_to_plot):
        # Calculate the time-averaged Vs for the current depth z
        vs_z_values = calculate_time_averaged_vs(profiles_df, z, dz)

        # Plot the histogram
        sns.histplot(x=vs_z_values, ax=ax, kde=True, bins=30)

        # Add summary statistics to the plot
        mean_val = vs_z_values.mean()
        median_val = np.median(vs_z_values)
        std_val = vs_z_values.std()
        stats_text = (
            f"Mean: {mean_val:.1f} m/s\n"
            f"Median: {median_val:.1f} m/s\n"
            f"Std Dev: {std_val:.1f} m/s"
        )
        ax.text(
            0.95,
            0.95,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.6),
        )

        ax.set_title(f"Distribution of Vs{z:.0f}")
        ax.set_xlabel("Time-Averaged Vs (m/s)")

    fig.suptitle(
        f"Time-Averaged Vs Distributions ({len(profiles_df)} Profiles)", fontsize=16
    )

    # --- Save Figure ---
    output_path = output_dir / f"vs_z_distributions_{name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Vs distribution plot to {output_path}")


def plot_vs_profiles(
    df: pd.DataFrame, output_dir: Path, name: str = "HLC", dz: float = 5.0
):
    """
    Plots the Vs profiles as stair plots in a Vs vs depth graph.

    This version assumes a uniform depth discretization (dz).

    Args:
        df (pd.DataFrame): DataFrame containing the Vs profiles.
        output_dir (Path): Directory to save the plot.
        name (str): Identifier for the output file name.
        dz (float): The uniform grid spacing of the depth profile in meters.
    """
    logger.info(f"Generating plot for {len(df)} Vs profiles...")

    # --- Simplified Depth Calculation ---
    # Since dz is uniform, we can generate the depth array directly.
    # This is faster than parsing strings from column names.
    num_depths = len(df.columns)
    depth_values = np.arange(0, num_depths * dz, dz)

    # --- Vectorized Plotting ---
    # Transpose the DataFrame's values so shape is (num_depths, num_profiles).
    vs_values = df.to_numpy().T

    # --- Plotting ---
    plt.figure(figsize=(8, 10))

    # plt.step is a great choice for this type of plot.
    plt.step(vs_values, depth_values, color="#1f77b4", alpha=0.25, where="post")

    # --- Aesthetics and Configuration ---
    ax = plt.gca()
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.invert_yaxis()
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    plt.xlabel("Shear-Wave Velocity, Vs (m/s)")
    plt.ylabel("Depth (m)")
    plt.title(f"Ensemble of {len(df)} Vs Profiles")
    plt.xlim(left=0)
    plt.grid(True, linestyle="--", alpha=0.7)

    # --- Save Figure ---
    output_path = output_dir / f"vs_profiles_{name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Vs profiles plot to {output_path}")


def plot_parameter_distributions(df: pd.DataFrame, output_dir: Path, name: str = "HLC"):
    """
    Visualizes the distributions of all generated model parameters.

    Creates a figure with subplots:
    - Histograms for continuous parameters (Vs1, Vs2, H1).
    - Count plots for discrete parameters (CV, rH, aHV).

    Args:
        df (pd.DataFrame): DataFrame with model parameters.
        output_dir (Path): Directory to save the plot.
    """
    logger.info("Generating plot of parameter distributions...")

    # Define continuous and discrete parameters
    continuous_params = {"Vs1": "m/s", "Vs2": "m/s", "H1": "m"}
    discrete_params = {"CV": "unitless", "rH": "m", "aHV": "unitless"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle("Verification of Parameter Distributions", fontsize=20, y=1.03)

    # Flatten axes array for easy iteration
    ax_flat = axes.flatten()

    # Plot continuous parameters using histograms
    for i, (param, unit) in enumerate(continuous_params.items()):
        sns.histplot(data=df, x=param, kde=True, ax=ax_flat[i], bins=41)
        ax_flat[i].set_title(f"Distribution of {param}")
        ax_flat[i].set_xlabel(f"{param} [{unit}]")
        ax_flat[i].set_ylabel("Frequency")

    # Plot discrete parameters using count plots
    for i, (param, unit) in enumerate(
        discrete_params.items(), start=len(continuous_params)
    ):
        sns.countplot(data=df, x=param, ax=ax_flat[i], legend=False)
        ax_flat[i].set_title(f"Distribution of {param}")
        ax_flat[i].set_xlabel(f"{param} [{unit}]")
        ax_flat[i].set_ylabel("Count")

    # Save the figure
    output_path = output_dir / f"parameter_distributions_{name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved parameter distribution plot to {output_path}")


def plot_vs_profile_density(
    df: pd.DataFrame, output_dir: Path, name: str = "HLC", dz: float = 5.0
):
    """
    Plots the Vs profiles as a 2D density histogram, which is highly efficient
    for a large number of profiles.
    """
    logger.info(f"Generating 2D density plot for {len(df)} Vs profiles...")

    # 1. Create the depth array
    num_depths = len(df.columns)
    depth_values = np.arange(0, num_depths * dz, dz)

    # 2. Reshape the data for 2D histogramming.
    # We need two long 1D arrays: one for all Vs values and one for their
    # corresponding depths.
    all_vs_values = df.to_numpy().flatten()
    all_depth_values = np.tile(depth_values, len(df))

    # 3. Create the plot
    plt.figure(figsize=(9, 10))

    # Use hist2d for incredible performance. Bins control the resolution.
    # cmin=1 tells it to ignore empty bins, making it cleaner.
    plt.hist2d(
        x=all_vs_values,
        y=all_depth_values,
        bins=(150, num_depths),  # (vs_bins, depth_bins)
        cmap="plasma",
        cmin=1,  # Don't plot bins with zero profiles
    )
    plt.colorbar(label="Number of Profiles in Bin")

    # --- Aesthetics and Configuration ---
    ax = plt.gca()
    ax.invert_yaxis()
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    plt.xlabel("Shear-Wave Velocity, Vs (m/s)")
    plt.ylabel("Depth (m)")
    plt.title(f"Density of {len(df)} Vs Profiles")
    plt.xlim(left=0)

    # --- Save Figure ---
    output_path = output_dir / f"vs_profile_density_{name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Vs profile density plot to {output_path}")


def plot_parameter_pairplot(df: pd.DataFrame, output_dir: Path, name: str = "HLC"):
    """
    Visualizes the relationships between all generated model parameters using a pairplot.
    """
    logger.info("Generating pairplot of parameter distributions...")

    # Create the pairplot
    g = sns.pairplot(df, diag_kind="kde", corner=True)
    g.fig.suptitle("Pairplot of Model Parameters", y=1.02)

    # Save the figure
    output_path = output_dir / f"parameter_pairplot_{name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved parameter pairplot to {output_path}")
