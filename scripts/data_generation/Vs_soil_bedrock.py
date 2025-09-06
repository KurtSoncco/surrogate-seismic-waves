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
    Vs_bedrock_range: tuple = (np.log10(760), np.log10(1500)),
    n_Vs: int = 1000,
    h_range: tuple = (1, 29),
):
    """
    Generates synthetic velocity profiles.
    Assumes a two-layer model: soil and bedrock.
    Soil layer has uniform Vs, bedrock layer has uniform Vs.
    Sampling is done using Latin Hypercube Sampling (LHS) and the
    shear wave velocities are sampled in log10 space for better coverage.

    Args:
        num_models (int): Number of models to generate.
        Vs_soil_range (tuple): Log10 range for soil Vs.
        Vs_bedrock_range (tuple): Log10 range for bedrock Vs.
        n_Vs (int): Number of samples to generate.
        h_range (tuple): Range for soil layer thickness in multiples of 5m.
    Returns:
        List of np.ndarray: Each array represents a velocity profile.
    """
    models = []
    sampler = qmc.LatinHypercube(d=2)
    lower_bound = [Vs_soil_range[0], Vs_bedrock_range[0]]
    upper_bound = [Vs_soil_range[1], Vs_bedrock_range[1]]
    scaled_samples = qmc.scale(sampler.random(n_Vs), lower_bound, upper_bound)

    sampler = qmc.LatinHypercube(d=1)
    h_soil_array = sampler.integers(
        l_bounds=h_range[0], u_bounds=h_range[1] + 1, n=n_Vs, endpoint=True
    )

    for i in range(num_models):
        Vs_s, Vs_b = scaled_samples[i]
        h_soil = h_soil_array[i]
        Vs_s = 10**Vs_s
        Vs_b = 10**Vs_b
        soil_array = np.full(h_soil, Vs_s)
        bedrock_array = np.array([Vs_b])
        array = np.concatenate([soil_array, bedrock_array])
        models.append(array)
    return models


def extract_properties(models: list, dz: float = 5.0):
    """
    Extracts properties from the generated models.
    Assumes each model is a 1D numpy array where:
        - All but the last element represent soil layer Vs.
        - The last element represents bedrock Vs.
        - The number of soil layers times 5m gives the soil thickness h.

    Args:
        models (list): List of np.ndarray, each representing a velocity profile.
        dz (float): Thickness of each soil layer in meters. Default is 5.0.
    Returns:
        Vs_soil (list): List of soil shear wave velocities.
        Vs_bedrock (list): List of bedrock shear wave velocities.
        h (list): List of soil layer thicknesses.
    """
    Vs_soil = [np.mean(a[:-1]) for a in models]
    Vs_bedrock = [a[-1] for a in models]
    h = [(len(a) - 1) * dz for a in models]
    return Vs_soil, Vs_bedrock, h


def plot_histograms(Vs_soil, Vs_bedrock, h, output_dir):
    """
    Plots and saves histograms of the properties.

    The function creates histograms for Vs_soil, Vs_bedrock, and h,
    each with specified bin widths and axis limits. The histograms are
    saved as a single image file in the specified output directory.

    Args:
        Vs_soil (list): List of soil shear wave velocities.
        Vs_bedrock (list): List of bedrock shear wave velocities.
        h (list): List of soil layer thicknesses.
        output_dir (str): Directory to save the plots.

    Returns:
        None
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

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
    ax[0].set_xlabel("$Vs_1$ [m/s]", fontsize=20)

    binwidth = 150
    ax[1].hist(
        Vs_bedrock,
        bins=np.arange(760, max(Vs_bedrock) + binwidth, binwidth),
        edgecolor="black",
        linewidth=1.2,
        color="orange",
    )
    ax[1].set_xticks(np.arange(760, max(Vs_bedrock) + binwidth, binwidth))
    ax[1].set_xlim(760, 1500)
    ax[1].set_xlabel("$Vs_2$ [m/s]", fontsize=20)

    binwidth = 5 * 5
    ax[2].hist(
        h,
        bins=np.arange(min(h), max(h) + binwidth, binwidth),
        edgecolor="black",
        linewidth=1.2,
        color="green",
    )
    ax[2].set_xticks(np.arange(min(h), max(h) + binwidth, binwidth))
    ax[2].set_xlim(1 * 5, 29 * 5)
    ax[2].set_xlabel("h [m]", fontsize=20)

    ax[0].set_ylabel("Frequency", fontsize=20)
    for a in ax:
        a.tick_params(axis="both", labelsize=15)

    plt.subplots_adjust(wspace=0.1)
    plt.savefig(os.path.join(output_dir, "property_histograms.png"))
    plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    sns.histplot(Vs_soil, ax=ax[0], color="blue", binwidth=13)
    ax[0].set_title("$Vs_1$ [m/s]")
    sns.histplot(Vs_bedrock, ax=ax[1], binwidth=50, color="orange")
    ax[1].set_title("$Vs_2$ [m/s]")
    sns.histplot(h, ax=ax[2], bins=25, color="green")
    ax[2].set_title("Height of the layer [m]")
    plt.savefig(os.path.join(output_dir, "property_seaborn_histograms.png"))
    plt.close()


def plot_scatter(Vs_soil: list, Vs_bedrock: list, h: list, output_dir: str):
    """
    Plots and saves scatter plots of the properties.

    The function creates scatter plots for Vs_soil, Vs_bedrock, and h,
    each against their index in the dataset. The plots are saved as a single
    image file in the specified output directory.

    Args:
        Vs_soil (list): List of soil shear wave velocities.
        Vs_bedrock (list): List of bedrock shear wave velocities.
        h (list): List of soil layer thicknesses.
        output_dir (str): Directory to save the plots.

    Returns:
        None
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].scatter(range(len(Vs_soil)), Vs_soil, color="blue")
    ax[0].set_title("Vs_soil")
    ax[1].scatter(range(len(Vs_bedrock)), Vs_bedrock, color="orange")
    ax[1].set_title("Vs_bedrock")
    ax[2].scatter(range(len(h)), h, color="green")
    ax[2].set_title("h")
    plt.savefig(os.path.join(output_dir, "property_scatter_plots.png"))
    plt.close()


def plot_frequency_analysis(Vs_soil: list, h: list, output_dir: str):
    """
    Performs frequency analysis and plots the results.

    The fundamental frequency f0 is calculated as Vs_soil / (4 * h).
    The maximum frequency fmax is estimated for different grid sizes (5m, 2.5m, 1m)
    using the formula fmax = Vs_soil / (15 * dz), where dz is the grid size.

    Args:
        Vs_soil (list): List of soil shear wave velocities.
        h (list): List of soil layer thicknesses.
        output_dir (str): Directory to save the plots.

    Returns:
        None
    """
    f0_cases = np.array(Vs_soil) / (4 * np.array(h))
    f_max_5 = np.array(Vs_soil) / (15 * 5)
    f_max_2_5 = np.array(Vs_soil) / (15 * 2.5)
    f_max_1 = np.array(Vs_soil) / (15 * 1)

    plt.figure()
    plt.hist(
        f0_cases,
        bins=200,
        edgecolor="black",
        linewidth=1.2,
        color="red",
        alpha=0.5,
        label="f0",
        density=True,
    )
    plt.legend()
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Density")
    plt.savefig(os.path.join(output_dir, "f0_histogram.png"))
    plt.close()

    plt.figure()
    plt.hist(
        f0_cases,
        bins=200,
        edgecolor="black",
        linewidth=1.2,
        color="red",
        alpha=0.5,
        label="f0",
        density=True,
    )
    plt.hist(
        f_max_5,
        bins=50,
        edgecolor="black",
        linewidth=1.2,
        color="blue",
        alpha=0.5,
        label="f_max - 5x5",
        density=True,
    )
    plt.hist(
        f_max_2_5,
        bins=50,
        edgecolor="black",
        linewidth=1.2,
        color="orange",
        alpha=0.5,
        label="f_max - 2.5x2.5",
        density=True,
    )
    plt.hist(
        f_max_1,
        bins=50,
        edgecolor="black",
        linewidth=1.2,
        color="green",
        alpha=0.5,
        label="f_max - 1x1",
        density=True,
    )
    plt.legend()
    plt.xscale("log")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Density")
    plt.savefig(os.path.join(output_dir, "f_max_comparison_histogram.png"))
    plt.close()


def save_data_parquet(models: list, output_path: str):
    """
    Saves a list of Numpy arrays to a Parquet file.

    This method preserves the variable length of each array by storing them in a single column with a list data type. It's efficient for storage and retrieval.

    Args:
        models (list): List of Numpy arrays to save.
        output_path (str): Path to save the Parquet file.

    Returns:
        None
    """
    try:
        # 1. Create a DataFrame with one column containing the arrays
        df = pd.DataFrame({"model_data": models})

        # 2. Convert the DataFrame to a PyArrow Table for optimal writing
        table = pa.Table.from_pandas(df)

        # 3. Write the Arrow Table to a Parquet file
        pq.write_table(table, output_path)
        logger.info(f"Data saved successfully to {output_path}")

    except Exception as e:
        logger.error(f"Failed to save data to Parquet: {e}")


def main():
    """
    Main function to run the data generation and analysis.
    """
    output_dir = "outputs/figures/Soil_Bedrock"
    data_dir = "data/Soil_Bedrock"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    models = generate_velocity_profiles(1000)
    Vs_soil, Vs_bedrock, h = extract_properties(models)

    plot_histograms(Vs_soil, Vs_bedrock, h, output_dir)
    plot_scatter(Vs_soil, Vs_bedrock, h, output_dir)
    plot_frequency_analysis(Vs_soil, h, output_dir)

    save_data_parquet(models, os.path.join(data_dir, "model_arrays_HLC.parquet"))


if __name__ == "__main__":
    main()
