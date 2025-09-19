# main_script.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import qmc

from wave_surrogate.logging_setup import setup_logging

logger = setup_logging()
sns.set_theme(style="whitegrid", palette="colorblind")
plt.set_loglevel("WARNING")


def generate_parameters(
    num_models: int = 10000, seed: int = 42, dz: float | None = None
) -> pd.DataFrame:
    """
    Generates a DataFrame of model parameters based on the document's specifications.

    This function uses Latin Hypercube Sampling (LHS) for continuous parameters
    (Vs1, Vs2, H) and random sampling for discrete stochastic parameters
    (CoV, rH, aHV) to create a comprehensive parameter set.

    Args:
        num_models (int): The total number of unique parameter sets to generate.
        seed (int): A random seed for reproducibility.

    Returns:
        pd.DataFrame: A DataFrame containing the sampled parameters for each model.
    """
    logger.info(f"Starting parameter generation for {num_models} models...")
    rng = np.random.default_rng(seed)

    # 1. Define parameter spaces as per the document
    # Continuous parameters (for LHS)
    vs1_range_log10 = (np.log10(100), np.log10(760))
    vs2_range_log10 = (np.log10(800), np.log10(1500))  # Using 800 as per doc
    h_range = (5.0, 150.0)

    # Discrete stochastic parameters
    cov_values = [0.1, 0.2, 0.3]
    rh_values = [10, 30, 50]
    ahv_values = [10, 20]

    # 2. Sample continuous parameters using Latin Hypercube Sampling (d=3)
    sampler = qmc.LatinHypercube(d=3, rng=rng)
    samples = sampler.random(n=num_models)

    # 3. Scale the LHS samples to their respective ranges
    lower_bounds = [vs1_range_log10[0], vs2_range_log10[0], h_range[0]]
    upper_bounds = [
        vs1_range_log10[1],
        vs2_range_log10[1],
        h_range[1],
    ]
    scaled_samples = qmc.scale(samples, lower_bounds, upper_bounds)

    # 4. Extract and transform continuous parameters
    df = pd.DataFrame()
    df["Vs1"] = 10 ** scaled_samples[:, 0]
    df["Vs2"] = 10 ** scaled_samples[:, 1]
    df["H"] = scaled_samples[:, 2] // dz * dz if dz else scaled_samples[:, 2]

    # 5. Sample discrete parameters randomly
    df["CoV"] = rng.choice(cov_values, size=num_models)
    df["rH"] = rng.choice(rh_values, size=num_models)
    df["aHV"] = rng.choice(ahv_values, size=num_models)

    logger.info("Parameter generation complete.")
    return df


def plot_parameter_distributions(df: pd.DataFrame, output_dir: Path):
    """
    Visualizes the distributions of all generated model parameters.

    Creates a figure with subplots:
    - Histograms for continuous parameters (Vs1, Vs2, H).
    - Count plots for discrete parameters (CoV, rH, aHV).

    Args:
        df (pd.DataFrame): DataFrame with model parameters.
        output_dir (Path): Directory to save the plot.
    """
    logger.info("Generating plot of parameter distributions...")

    # Define continuous and discrete parameters
    continuous_params = {"Vs1": "m/s", "Vs2": "m/s", "H": "m"}
    discrete_params = {"CoV": "unitless", "rH": "m", "aHV": "unitless"}

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
    output_path = output_dir / "parameter_distributions.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved parameter distribution plot to {output_path}")


def plot_parameter_pairplot(df: pd.DataFrame, output_dir: Path):
    """
    Visualizes the relationships between all generated model parameters using a pairplot.
    """
    logger.info("Generating pairplot of parameter distributions...")

    # Create the pairplot
    g = sns.pairplot(df, diag_kind="kde", corner=True)
    g.fig.suptitle("Pairplot of Model Parameters", y=1.02)

    # Save the figure
    output_path = output_dir / "parameter_pairplot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved parameter pairplot to {output_path}")


def main():
    """Main execution function."""
    # --- Configuration ---
    NUM_MODELS = 10000  # Number of unique parameter sets to generate
    DZ = 5  # Grid spacing in meters
    OUTPUT_DIR = Path("outputs/figures/2D Profiles/Soil_Bedrock")

    # --- Directory Setup ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Outputs will be saved to: {OUTPUT_DIR.resolve()}")

    # --- Data Generation ---
    params_df = generate_parameters(num_models=NUM_MODELS, dz=DZ)

    # Save parameters to a CSV file for inspection or later use
    params_df.to_csv(OUTPUT_DIR / "sampled_parameters.csv", index=False)
    logger.info(
        f"Saved sampled parameters to CSV.\nHead of the DataFrame:\n{params_df.head()}"
    )

    # --- Visualization ---
    plot_parameter_distributions(params_df, OUTPUT_DIR)
    plot_parameter_pairplot(params_df, OUTPUT_DIR)


if __name__ == "__main__":
    main()
