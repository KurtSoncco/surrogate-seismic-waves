from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import qmc

from wave_surrogate.logging_setup import setup_logging

logger = setup_logging()
sns.set_palette("colorblind")


def generate_velocity_profiles(
    num_models: int = 1000,
    vs_soil_range_log10: tuple[float, float] = (np.log10(100), np.log10(360)),
    vs_bedrock_range_log10: tuple[float, float] = (np.log10(760), np.log10(1500)),
    h_layer_range: tuple[int, int] = (1, 29),
    dz: float = 5.0,
) -> pd.DataFrame:
    """
    Generates synthetic two-layer velocity profiles using correct 3D LHS.

    This function samples all three parameters (soil Vs, bedrock Vs, and thickness)
    from a single 3D Latin Hypercube to ensure proper stratification across the
    entire parameter space.

    Args:
        num_models (int): Number of models to generate.
        vs_soil_range_log10 (tuple): Log10 range for soil shear wave velocity (Vs).
        vs_bedrock_range_log10 (tuple): Log10 range for bedrock Vs.
        h_layer_range (tuple): Range for the number of soil layers.
        dz (float): Thickness of each layer in meters.

    Returns:
        pd.DataFrame: A DataFrame containing the generating parameters and the
                      resulting velocity profile array for each model.
    """
    # 1. Correctly sample the 3D parameter space with a single sampler.
    rng = np.random.default_rng(42)
    sampler = qmc.LatinHypercube(d=3, rng=rng)
    samples = sampler.random(n=num_models)

    # 2. Define the bounds and scale the samples.
    lower_bounds = [vs_soil_range_log10[0], vs_bedrock_range_log10[0], h_layer_range[0]]
    upper_bounds = [
        vs_soil_range_log10[1],
        vs_bedrock_range_log10[1],
        h_layer_range[1] + 1,
    ]
    scaled_samples = qmc.scale(samples, lower_bounds, upper_bounds)

    # 3. Extract and transform parameters with clear names.
    vs_soil = 10 ** scaled_samples[:, 0]
    vs_bedrock = 10 ** scaled_samples[:, 1]
    h_layers = scaled_samples[:, 2].astype(int)
    h_meters = h_layers * dz

    # 4. Generate the model arrays.
    model_arrays = []
    for vs_s, vs_b, h_l in zip(vs_soil, vs_bedrock, h_layers):
        soil = np.full(h_l, vs_s)
        bedrock = np.array([vs_b])
        model_arrays.append(np.concatenate([soil, bedrock]))

    # 5. Return a single, structured DataFrame. This is much more useful.
    return pd.DataFrame(
        {
            "vs_soil": vs_soil,
            "vs_bedrock": vs_bedrock,
            "h": h_meters,
            "model_array": model_arrays,
        }
    )


def plot_property_distributions(df: pd.DataFrame, output_dir: Path):
    """
    Creates a pairplot to visualize parameter distributions and correlations.

    A pairplot is superior to separate histograms and index-based scatter plots
    A pairplot is superior to separate histograms and index-based scatter plots
    because it shows histograms on the diagonal and pairwise scatter plots for
    all parameter combinations, revealing relationships between them.

    Args:
        df (pd.DataFrame): DataFrame with model parameters.
        output_dir (Path): Directory to save the plot.
    """
    logger.info("Generating property distribution pairplot...")
    plot_vars = ["vs_soil", "vs_bedrock", "h"]

    g = sns.pairplot(df[plot_vars], corner=True, diag_kind="kde")
    g.fig.suptitle(
        "Distributions and Correlations of Model Parameters", y=1.02, fontsize=16
    )

    # Add units to axis labels for clarity
    g.axes[2, 0].set_xlabel("$V_{s, soil}$ [m/s]")
    g.axes[2, 1].set_xlabel("$V_{s, bedrock}$ [m/s]")
    g.axes[0, 0].set_ylabel("$V_{s, soil}$ [m/s]")
    g.axes[1, 0].set_ylabel("$V_{s, bedrock}$ [m/s]")
    g.axes[2, 0].set_ylabel("Thickness h [m]")

    output_path = output_dir / "property_pairplot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved pairplot to {output_path}")


def plot_frequency_analysis(df: pd.DataFrame, output_dir: Path):
    """
    Performs and plots frequency analysis based on model properties.

    Args:
        df (pd.DataFrame): DataFrame containing model properties.
        output_dir (Path): Directory to save the plots.
    """
    logger.info("Generating frequency analysis plots...")
    # Perform calculations directly on DataFrame columns (vectorized and efficient)
    f0 = df["vs_soil"] / (4 * df["h"])
    f0 = f0.replace([np.inf, -np.inf], np.nan).dropna()  # Clean invalid values

    f_max_dict = {
        "5m grid": df["vs_soil"] / (15 * 5.0),
        "2.5m grid": df["vs_soil"] / (15 * 2.5),
        "1m grid": df["vs_soil"] / (15 * 1.0),
    }

    # Create a single figure with two subplots for consolidated output
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.suptitle("Frequency Analysis", fontsize=16)

    # Plot 1: Fundamental Frequency (f0) Distribution
    sns.histplot(x=f0, ax=axes[0], bins=100, kde=True, color="red", alpha=0.7)
    axes[0].set_title("Fundamental Frequency ($f_0$) Distribution")
    axes[0].set_xlabel("Frequency [Hz]")
    axes[0].set_ylabel("Density")

    # Plot 2: Comparison of f0 and f_max
    sns.histplot(
        x=f0, ax=axes[1], bins=100, kde=True, label="$f_0$", color="red", alpha=0.6
    )
    for label, f_max_data in f_max_dict.items():
        sns.histplot(
            x=f_max_data, ax=axes[1], bins=50, kde=True, label=f"$f_{{max}}$ ({label})"
        )

    axes[1].set_title("$f_0$ vs. Estimated $f_{max}$")
    axes[1].set_xlabel("Frequency [Hz] (log scale)")
    axes[1].set_ylabel("Density")
    axes[1].set_xscale("log")
    axes[1].legend()

    output_path = output_dir / "frequency_analysis.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved frequency analysis plot to {output_path}")


def main():
    """Main execution script."""
    # --- Configuration ---
    NUM_MODELS = 5000
    DATA_DIR = Path("data/Soil_Bedrock")
    FIGURE_DIR = Path("outputs/figures/Soil_Bedrock")

    # --- Directory Setup ---
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # --- Data Generation ---
    profiles_df = generate_velocity_profiles(num_models=NUM_MODELS)
    logger.info(f"Generated {len(profiles_df)} velocity profiles.")

    # --- Data Saving (Simplified) ---
    output_file = DATA_DIR / "model_profiles.parquet"
    profiles_df.to_parquet(output_file, engine="pyarrow")
    logger.info(f"Saved profiles DataFrame to {output_file}")

    # --- Plotting and Analysis ---
    plot_property_distributions(profiles_df, FIGURE_DIR)
    plot_frequency_analysis(profiles_df, FIGURE_DIR)


if __name__ == "__main__":
    main()
