# main_script.py

from itertools import product
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm, qmc
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
    parameter_config: Dict[str, Any], seed: int = 42, dz: float | None = None
) -> pd.DataFrame:
    """
    Generates a DataFrame of model parameters using a full combinatorial approach.

    This function creates a comprehensive parameter set by taking the Cartesian
    product of all specified parameter values. Continuous parameters are sampled
    using Latin Hypercube Sampling (LHS) before being included in the product.

    Args:
        parameter_config (dict): A dictionary defining the parameter space. It
            should contain two top-level keys: 'continuous' and 'discrete'.
            - 'continuous' (Dict[str, Dict]): Keys are parameter names. Values
              are dictionaries with 'n' (int), 'range' (Tuple[float, float]),
              and an optional 'log_scale' (bool, default False).
            - 'discrete' (Dict[str, List]): Keys are parameter names and values
              are lists of the values to use.
        seed (int): A random seed for reproducibility.
        dz (float | None): Grid spacing for discretizing the 'H' parameter.
                           If None, no discretization is applied.

    Returns:
        pd.DataFrame: A DataFrame containing the full combinatorial set of model
                      parameters.
    """
    logger.info("Reading parameter space from config...")
    rng = np.random.default_rng(seed)
    sampled_params = {}

    # 1. Generate samples for each parameter dimension
    # Sample continuous parameters using Latin Hypercube Sampling
    for name, config in parameter_config.get("continuous", {}).items():
        # IMPROVEMENT: Scrambling improves the statistical properties of the sample
        sampler = qmc.LatinHypercube(d=1, scramble=True, rng=rng)

        samples_unit_scale = sampler.random(n=config["n"])

        low, high = config["range"]
        if config.get("log_scale", False):
            # For log scale, generate a log-normal distribution. This means the
            # parameter's logarithm is normally distributed.
            # We assume the provided 'range' spans 6 standard deviations
            # (mean ± 3σ), covering 99.7% of the probability mass.

            log_low, log_high = np.log10(low), np.log10(high)

            # Calculate mean and standard deviation for the normal distribution in log space
            mean_log = (log_low + log_high) / 2.0
            std_dev_log = (log_high - log_low) / 6.0

            # Use the inverse CDF (percent-point function) of the normal distribution
            # to transform the uniform LHS samples into normally distributed samples
            # in the log space.
            log_samples = norm.ppf(samples_unit_scale, loc=mean_log, scale=std_dev_log)

            # Convert samples back to the original scale
            scaled_samples = 10**log_samples
        else:
            # For linear scale, the distribution is uniform
            scaled_samples = qmc.scale(samples_unit_scale, low, high)
        sampled_params[name] = scaled_samples.flatten()

    # Use discrete parameter values directly from the improved config
    for name, config in parameter_config.get("discrete", {}).items():
        sampled_params[name] = config["values"]

    # 2. Create the Cartesian product of all parameter samples
    param_names = list(sampled_params.keys())
    param_values = list(sampled_params.values())
    df = pd.DataFrame(product(*param_values), columns=param_names)
    logger.info(f"Generated a total of {len(df)} models.")

    # 3. Apply post-processing (e.g., discretization)
    if dz is not None and "H1" in df.columns:
        logger.info(f"Discretizing 'H1' with grid spacing dz={dz}.")
        df["H1"] = (df["H1"] // dz) * dz

    df = df.convert_dtypes()

    logger.info("Parameter generation complete.")
    return df


def main():
    """Main execution function."""
    # --- Configuration ---
    DZ = 5
    OUTPUT_DIR = Path("outputs/figures/2D Profiles/Soil_Bedrock")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Outputs will be saved to: {OUTPUT_DIR.resolve()}")

    # --- Data Generation ---
    # IMPROVEMENT: Centralized config with parameter values, types, and metadata (units)
    parameter_config = {
        "continuous": {
            "Vs1": {"n": 20, "range": (100, 760), "log_scale": True, "unit": "m/s"},
            "Vs2": {"n": 10, "range": (760, 1500), "log_scale": True, "unit": "m/s"},
            "H1": {"n": 10, "range": (5, 145), "unit": "m"},
        },
        "discrete": {
            "CV": {"values": [0.1, 0.2, 0.3], "unit": "unitless"},
            "rH": {"values": [10, 30, 50], "unit": "m"},
            "aHV": {"values": [0.5, 1.0], "unit": "unitless"},
        },
    }
    params_df = generate_parameters(parameter_config=parameter_config, dz=DZ)

    # Save parameters to a CSV
    params_df.to_csv(OUTPUT_DIR / "sampled_parameters_fixed.csv", index=False)
    logger.info(
        f"Saved sampled parameters to CSV.\nHead of the DataFrame:\n{params_df.head()}"
    )
    logger.info(f"DataFrame info:\n{params_df.info()}")

    # --- Vs Profile Generation ---
    profiles_df = vs_profiles(params_df, dz=DZ)

    # --- Visualization ---
    # Pass the config to the plotting function
    plot_parameter_distributions(params_df, OUTPUT_DIR, name="Fixed")
    plot_parameter_pairplot(params_df, OUTPUT_DIR, name="Fixed")
    plot_vs_profile_density(profiles_df, OUTPUT_DIR, name="Fixed")
    plot_vs_z_distributions(profiles_df, OUTPUT_DIR, name="Fixed")
    plot_vs_profiles(profiles_df, OUTPUT_DIR, name="Fixed")


if __name__ == "__main__":
    main()
