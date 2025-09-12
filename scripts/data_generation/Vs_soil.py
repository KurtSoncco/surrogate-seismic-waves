from pathlib import Path

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
    vs_soil_range_log10: tuple[float, float] = (np.log10(100), np.log10(360)),
    h_layer_range: tuple[int, int] = (1, 29),
    dz: float = 5.0,
    seed=42,
) -> pd.DataFrame:
    """
    Generates synthetic velocity profiles using Latin Hypercube Sampling.

    This version correctly applies LHS to the 2D parameter space (Vs and h)
    to ensure optimal stratification across both dimensions simultaneously.

    Args:
        num_models (int): Number of models to generate.
        vs_soil_range_log10 (tuple): Log10 range for soil shear wave velocity (Vs).
        h_layer_range (tuple): Range for the number of soil layers.
        dz (float): Thickness of each layer in meters.

    Returns:
        pd.DataFrame: A DataFrame with columns for Vs, thickness (h), and the
                      generated velocity profile array.
    """
    # 1. Correctly sample the 2D parameter space (log10(Vs), h) with one sampler.
    rng = np.random.default_rng(seed)
    sampler = qmc.LatinHypercube(d=2, rng=rng)
    samples = sampler.random(n=num_models)

    # 2. Scale the samples to the desired parameter ranges.
    lower_bounds = [vs_soil_range_log10[0], h_layer_range[0]]
    upper_bounds = [
        vs_soil_range_log10[1],
        h_layer_range[1] + 1,
    ]  # +1 for integer range
    scaled_samples = qmc.scale(samples, lower_bounds, upper_bounds)

    # 3. Extract, transform, and prepare parameters.
    log_vs_samples = scaled_samples[:, 0]
    vs_samples = 10**log_vs_samples
    # Convert thickness samples to integers representing number of layers
    h_layers = scaled_samples[:, 1].astype(int)
    h_meters = h_layers * dz

    # 4. Generate models efficiently in a list comprehension.
    model_arrays = [np.full(h, vs) for h, vs in zip(h_layers, vs_samples)]

    # 5. Return a structured DataFrame, which is often more useful.
    return pd.DataFrame(
        {
            "vs_soil": vs_samples,
            "h": h_meters,
            "model_array": model_arrays,
        }
    )


def plot_distributions(df: pd.DataFrame, output_dir: Path):
    """
    Plots and saves histograms and a scatter plot of the model properties.

    Args:
        df (pd.DataFrame): DataFrame containing model properties ('vs_soil', 'h').
        output_dir (Path): Directory to save the plot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    fig.suptitle("Distribution of Generated Soil Properties", fontsize=16)

    # Vs histogram
    sns.histplot(data=df, x="vs_soil", ax=axes[0], kde=True)
    axes[0].set_xlabel("$V_s$ [m/s]")
    axes[0].set_title("Shear Wave Velocity Distribution")

    # Thickness (h) histogram
    sns.histplot(data=df, x="h", ax=axes[1], kde=True, color="green")
    axes[1].set_xlabel("Thickness h [m]")
    axes[1].set_title("Soil Layer Thickness Distribution")

    # Vs vs. h scatter plot
    sns.scatterplot(data=df, x="vs_soil", y="h", ax=axes[2], alpha=0.6)
    axes[2].set_xlabel("$V_s$ [m/s]")
    axes[2].set_ylabel("Thickness h [m]")
    axes[2].set_title("$V_s$ vs. Thickness")

    # Save the consolidated figure
    output_path = output_dir / "property_distributions.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved distribution plots to {output_path}")


def main():
    """Main execution script."""
    # --- Configuration ---
    NUM_MODELS = 5000
    DATA_DIR = Path("data/Soil")
    FIGURE_DIR = Path("outputs/figures/Soil")

    # --- Directory Setup ---
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # --- Data Generation ---
    profiles_df = generate_velocity_profiles(num_models=NUM_MODELS)
    logger.info(f"Generated {len(profiles_df)} velocity profiles.")

    # --- Save Models ---
    # Parquet can handle object columns (like our arrays), but it's less efficient.
    # For this use case, it's perfectly fine.
    table = pa.Table.from_pandas(profiles_df)
    output_file = DATA_DIR / "1D_1Soil_model_profiles.parquet"
    pq.write_table(table, output_file)
    logger.info(f"Saved profiles to {output_file}")

    # --- Plotting ---
    plot_distributions(profiles_df, FIGURE_DIR)


if __name__ == "__main__":
    main()
