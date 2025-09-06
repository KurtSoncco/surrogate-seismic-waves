import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns
from scipy.stats import qmc

from wave_surrogate.logging_setup import setup_logging

logger = setup_logging()
sns.set_palette("colorblind")


def generate_velocity_profiles(
    num_models: int = 1000,
    Vs_soil_range: tuple = (np.log10(100), np.log10(360)),
    n_Vs: int = 1000,
    h_range: tuple = (1, 29),
):
    """
    Generates synthetic velocity profiles.
    Assumes a one-layer soil model.
    Soil layer has uniform Vs.
    Sampling is done using Latin Hypercube Sampling (LHS) and the
    shear wave velocities are sampled in log10 space for better coverage.

    Args:
        num_models (int): Number of models to generate.
        Vs_soil_range (tuple): Log10 range for soil Vs.
        n_Vs (int): Number of samples to generate.
        h_range (tuple): Range for soil layer thickness in multiples of 5m.
    Returns:
        List of np.ndarray: Each array represents a velocity profile.
    """
    models = []
    sampler = qmc.LatinHypercube(d=1)
    lower_bound = [Vs_soil_range[0]]
    upper_bound = [Vs_soil_range[1]]
    scaled_samples = qmc.scale(sampler.random(n_Vs), lower_bound, upper_bound)

    sampler = qmc.LatinHypercube(d=1)
    h_soil_array = sampler.integers(
        l_bounds=h_range[0], u_bounds=h_range[1] + 1, n=n_Vs, endpoint=True
    )

    for i in range(num_models):
        (Vs_s,) = scaled_samples[i]
        h_soil = h_soil_array[i]
        Vs_s = 10**Vs_s
        soil_array = np.full(h_soil, Vs_s)
        models.append(soil_array)
    return models


def extract_properties(models: list, dz: float = 5.0):
    """
    Extracts properties from the generated models.
    Assumes each model is a 1D numpy array representing a soil layer.

    Args:
        models (list): List of np.ndarray, each representing a velocity profile.
        dz (float): Thickness of each soil layer in meters. Default is 5.0.
    Returns:
        Vs_soil (list): List of soil shear wave velocities.
        h (list): List of soil layer thicknesses.
    """
    Vs_soil = [np.mean(a) for a in models]
    h = [len(a) * dz for a in models]
    return Vs_soil, h


def plot_histograms(Vs_soil, h, output_dir):
    """
    Plots and saves histograms of the properties.

    The function creates histograms for Vs_soil and h,
    each with specified bin widths and axis limits. The histograms are
    saved as a single image file in the specified output directory.

    Args:
        Vs_soil (list): List of soil shear wave velocities.
        h (list): List of soil layer thicknesses.
        output_dir (str): Directory to save the plots.

    Returns:
        None
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    binwidth = 50
    ax[0].hist(
        Vs_soil,
        bins=np.arange(min(Vs_soil), max(Vs_soil) + binwidth, binwidth),
        edgecolor="black",
        linewidth=1.2,
        color="blue",
    )
    ax[0].set_xticks(np.arange(min(Vs_soil), max(Vs_soil) + binwidth, binwidth))
    ax[0].set_xlim(100, 400)
    ax[0].set_xlabel("$V_s$ [m/s]", fontsize=20)

    binwidth = 5 * 5
    ax[1].hist(
        h,
        bins=np.arange(min(h), max(h) + binwidth, binwidth),
        edgecolor="black",
        linewidth=1.2,
        color="green",
    )
    ax[1].set_xticks(np.arange(min(h), max(h) + binwidth, binwidth))
    ax[1].set_xlim(1 * 5, 29 * 5)
    ax[1].set_xlabel("h [m]", fontsize=20)

    ax[0].set_ylabel("Frequency", fontsize=20)
    for a in ax:
        a.tick_params(axis="both", labelsize=15)

    plt.subplots_adjust(wspace=0.1)
    plt.savefig(os.path.join(output_dir, "property_histograms.png"))
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    sns.histplot(Vs_soil, ax=ax[0], color="blue", binwidth=13)
    ax[0].set_title("$V_s$ [m/s]")
    sns.histplot(h, ax=ax[1], color="green", binwidth=25)
    ax[1].set_title("h [m]")
    plt.savefig(os.path.join(output_dir, "property_seaborn_histograms.png"))
    plt.close()


def plot_scatter(Vs_soil, h, output_dir):
    """
    Plots and saves scatter plots of the properties.

    Args:
        Vs_soil (list): List of soil shear wave velocities.
        h (list): List of soil layer thicknesses.
        output_dir (str): Directory to save the plots.
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(Vs_soil, h, alpha=0.5)
    ax.set_xlabel("$V_s$ [m/s]")
    ax.set_ylabel("h [m]")
    plt.savefig(os.path.join(output_dir, "property_scatter_plots.png"))
    plt.close()


if __name__ == "__main__":
    num_models = 1000
    models = generate_velocity_profiles(num_models=num_models)

    # Save models to a parquet file
    output_dir = "data/Soil"
    os.makedirs(output_dir, exist_ok=True)
    table = pa.Table.from_pandas(pd.DataFrame({"model_arrays": models}))
    pq.write_table(table, os.path.join(output_dir, "model_arrays_HLC.parquet"))
    logger.info(f"Generated and saved {num_models} models in {output_dir}")

    # Extract properties and plot
    Vs_soil, h = extract_properties(models)
    output_dir_plots = "outputs/figures/Soil"
    os.makedirs(output_dir_plots, exist_ok=True)
    plot_histograms(Vs_soil, h, output_dir_plots)
    plot_scatter(Vs_soil, h, output_dir_plots)
    logger.info(f"Saved plots in {output_dir_plots}")
