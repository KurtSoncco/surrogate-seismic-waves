# main_script.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import qmc
from utils import (
    plot_parameter_distributions,
    plot_parameter_pairplot,
    plot_vs_profile_density,
    plot_vs_profiles,
    plot_vs_z_distributions,
    vs_profiles,
)

from wave_surrogate.logging_setup import setup_logging

logger = setup_logging()
sns.set_theme(style="whitegrid", palette="colorblind")
plt.set_loglevel("WARNING")


def generate_parameters(
    num_models: int = 180000, seed: int = 42, dz: float | None = None
) -> pd.DataFrame:
    """
    Generates a DataFrame of model parameters based on the document's specifications.

    This function uses Latin Hypercube Sampling (LHS) for continuous parameters
    (Vs1, Vs2, H1) and random sampling for discrete stochastic parameters
    (CV, rH, aHV) to create a comprehensive parameter set.

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
    vs2_range_log10 = (np.log10(760), np.log10(1500))  # Using 760 as per doc
    h1_range = (5.0, 145.0)

    # Discrete stochastic parameters
    cv_values = [0.1, 0.2, 0.3]
    rh_values = [10, 30, 50]
    ahv_values = [10, 20]

    # 2. Sample continuous parameters using Latin Hypercube Sampling (d=3)
    sampler = qmc.LatinHypercube(d=3, rng=rng)
    samples = sampler.random(n=num_models)

    # 3. Scale the LHS samples to their respective ranges
    lower_bounds = [vs1_range_log10[0], vs2_range_log10[0], h1_range[0]]
    upper_bounds = [
        vs1_range_log10[1],
        vs2_range_log10[1],
        h1_range[1],
    ]
    scaled_samples = qmc.scale(samples, lower_bounds, upper_bounds)

    # 4. Extract and transform continuous parameters
    df = pd.DataFrame()
    df["Vs1"] = 10 ** scaled_samples[:, 0]
    df["Vs2"] = 10 ** scaled_samples[:, 1]
    df["H1"] = scaled_samples[:, 2] // dz * dz if dz else scaled_samples[:, 2]

    # 5. Sample discrete parameters randomly
    df["CV"] = rng.choice(cv_values, size=num_models)
    df["rH"] = rng.choice(rh_values, size=num_models)
    df["aHV"] = rng.choice(ahv_values, size=num_models)

    logger.info("Parameter generation complete.")
    return df


def main():
    """Main execution function."""
    # --- Configuration ---
    NUM_MODELS = 36000  # Number of unique parameter sets to generate
    DZ = 5  # Grid spacing in meters
    OUTPUT_DIR = Path("outputs/figures/2D Profiles/Soil_Bedrock")

    # --- Directory Setup ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Outputs will be saved to: {OUTPUT_DIR.resolve()}")

    # --- Data Generation ---
    params_df = generate_parameters(num_models=NUM_MODELS, dz=DZ)

    # Save parameters to a CSV file for inspection or later use
    params_df.to_csv(OUTPUT_DIR / "sampled_parameters_HLC.csv", index=False)
    logger.info(
        f"Saved sampled parameters to CSV.\nHead of the DataFrame:\n{params_df.head()}"
    )
    # --- Vs Profile Generation ---
    profiles_df = vs_profiles(params_df, dz=DZ)

    # --- Visualization ---
    plot_parameter_distributions(params_df, OUTPUT_DIR, name="HLC")
    plot_parameter_pairplot(params_df, OUTPUT_DIR, name="HLC")
    plot_vs_profiles(profiles_df, OUTPUT_DIR, name="HLC")
    plot_vs_profile_density(profiles_df, dz=DZ, output_dir=OUTPUT_DIR, name="HLC")
    plot_vs_z_distributions(profiles_df, dz=DZ, output_dir=OUTPUT_DIR, name="HLC")


if __name__ == "__main__":
    main()
