#!/usr/bin/env python3
"""
Comprehensive ablation study for CNN encoder latent FNO model.
This script systematically tests different hyperparameter combinations to find the optimal configuration.

Based on the successful CNN encoder run with mean sample correlation of 0.64912,
we'll optimize the following hyperparameters:
- CNN architecture (channels, kernel sizes, pooling)
- Latent dimension
- FNO processor parameters
- Learning rate and training parameters
- Dropout rates
"""

import itertools
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple


def run_experiment(
    config_name: str, config_params: Dict[str, Any]
) -> Tuple[bool, Dict[str, Any]]:
    """Run a single experiment and return success status and results."""
    print(f"\n🚀 Starting ablation experiment: {config_name}")
    print("-" * 60)
    print(f"Parameters: {json.dumps(config_params, indent=2)}")
    print("-" * 60)

    try:
        # Create a temporary config file for this experiment
        config_file = f"temp_config_{config_name}.py"
        create_temp_config(config_name, config_params, config_file)

        # Run the training with wandb enabled, activating virtual environment
        result = subprocess.run(
            [
                "bash",
                "-c",
                "source /home/kurt-asus/surrogate-seismic-waves/.venv/bin/activate && python main.py train --config "
                + config_name
                + " --wandb",
            ],
            capture_output=True,
            text=True,
            timeout=3600,
        )  # 1 hour timeout

        # Clean up temp config
        if os.path.exists(config_file):
            os.remove(config_file)

        if result.returncode == 0:
            print(f"✅ {config_name} completed successfully!")
            # Parse results from output
            results = parse_experiment_results(result.stdout)
            return True, results
        else:
            print(f"❌ {config_name} failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False, {}

    except subprocess.TimeoutExpired:
        print(f"⏰ {config_name} timed out after 1 hour")
        return False, {}
    except Exception as e:
        print(f"💥 {config_name} failed with exception: {e}")
        return False, {}


def create_temp_config(config_name: str, params: Dict[str, Any], filename: str):
    """Create a temporary configuration file for the experiment."""
    config_content = f'''
# Temporary config for {config_name}
from src.configs.config import LatentFNOConfig

# Base CNN encoder config
config = LatentFNOConfig(
    encoder_type="cnn",
    encoder_config={params.get("encoder_config", {})},
    latent_dim={params.get("latent_dim", 128)},
    fno_processor_config={params.get("fno_processor_config", {})},
    learning_rate={params.get("learning_rate", 1e-3)},
    batch_size={params.get("batch_size", 128)},
    weight_decay={params.get("weight_decay", 1e-4)},
    wandb_run_name="{config_name}",
    experiment_name="{config_name}",
    description="Ablation study: {config_name}"
)
'''

    with open(filename, "w") as f:
        f.write(config_content)


def parse_experiment_results(output: str) -> Dict[str, Any]:
    """Parse experiment results from output text."""
    results = {}

    # Extract key metrics from output
    lines = output.split("\n")
    for line in lines:
        if "Test MSE:" in line:
            try:
                results["test_mse"] = float(line.split("Test MSE:")[1].strip())
            except:
                pass
        elif "Test MAE:" in line:
            try:
                results["test_mae"] = float(line.split("Test MAE:")[1].strip())
            except:
                pass
        elif "Test Correlation:" in line:
            try:
                results["test_correlation"] = float(
                    line.split("Test Correlation:")[1].strip()
                )
            except:
                pass
        elif "Best validation loss:" in line:
            try:
                results["best_val_loss"] = float(
                    line.split("Best validation loss:")[1].split("at")[0].strip()
                )
            except:
                pass

    return results


def check_environment():
    """Check the environment and print system information."""
    print("🔍 Environment Check")
    print("-" * 30)
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Working directory: {os.getcwd()}")

    # Check for CUDA/GPU availability
    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠️  No CUDA GPUs detected - will use CPU")
    except ImportError:
        print("❌ PyTorch not installed!")
        return False

    # Check for wandb
    try:
        import wandb

        print(f"Wandb version: {wandb.__version__}")
    except ImportError:
        print("❌ Wandb not installed!")
        return False

    print("✅ Environment check passed!")
    return True


def generate_ablation_experiments() -> List[Tuple[str, Dict[str, Any]]]:
    """Generate all ablation study experiments."""
    experiments = []

    # Base CNN encoder configuration (our best performer)
    base_config = {
        "encoder_config": {
            "channels": [1, 32, 64, 128],
            "kernel_sizes": [3, 3, 3],
            "pool_sizes": [2, 2, 2],
            "dropout_rate": 0.1,
            "use_adaptive_pool": True,
        },
        "latent_dim": 128,
        "fno_processor_config": {
            "fno_modes": 16,
            "fno_width": 64,
            "num_fno_layers": 3,
            "dropout_rate": 0.1,
            "use_residual": True,
        },
        "learning_rate": 1e-3,
        "batch_size": 128,
        "weight_decay": 1e-4,
    }

    # 1. CNN Architecture Ablation
    print("📋 Generating CNN Architecture Ablation Experiments...")

    # Different channel configurations
    channel_configs = [
        [1, 16, 32, 64],  # Smaller
        [1, 32, 64, 128],  # Base
        [1, 64, 128, 256],  # Larger
        [1, 32, 64, 128, 256],  # Deeper
    ]

    # Different kernel sizes
    kernel_configs = [
        [3, 3, 3],  # Base
        [5, 5, 5],  # Larger kernels
        [3, 5, 7],  # Mixed sizes
        [7, 5, 3],  # Decreasing sizes
    ]

    # Different pooling strategies
    pool_configs = [
        [2, 2, 2],  # Base
        [2, 2, 4],  # Mixed pooling
        [4, 2, 2],  # Different pattern
        [2, 4, 2],  # Another pattern
    ]

    # Generate CNN architecture experiments
    for i, (channels, kernels, pools) in enumerate(
        itertools.product(channel_configs, kernel_configs, pool_configs)
    ):
        config = base_config.copy()
        config["encoder_config"] = {
            "channels": channels,
            "kernel_sizes": kernels,
            "pool_sizes": pools,
            "dropout_rate": 0.1,
            "use_adaptive_pool": True,
        }
        experiments.append((f"cnn_arch_{i + 1:02d}", config))

    # 2. Latent Dimension Ablation
    print("📋 Generating Latent Dimension Ablation Experiments...")

    latent_dims = [64, 96, 128, 160, 192, 256, 320, 384, 512]
    for latent_dim in latent_dims:
        config = base_config.copy()
        config["latent_dim"] = latent_dim
        # Adjust FNO width proportionally
        config["fno_processor_config"]["fno_width"] = min(256, latent_dim * 2)
        experiments.append((f"latent_dim_{latent_dim}", config))

    # 3. FNO Processor Ablation
    print("📋 Generating FNO Processor Ablation Experiments...")

    fno_modes = [8, 12, 16, 20, 24, 32]
    fno_widths = [32, 48, 64, 80, 96, 128]
    fno_layers = [2, 3, 4, 5, 6]

    for modes, width, layers in itertools.product(fno_modes, fno_widths, fno_layers):
        config = base_config.copy()
        config["fno_processor_config"] = {
            "fno_modes": modes,
            "fno_width": width,
            "num_fno_layers": layers,
            "dropout_rate": 0.1,
            "use_residual": True,
        }
        experiments.append((f"fno_m{modes}_w{width}_l{layers}", config))

    # 4. Learning Rate Ablation
    print("📋 Generating Learning Rate Ablation Experiments...")

    learning_rates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    for lr in learning_rates:
        config = base_config.copy()
        config["learning_rate"] = lr
        experiments.append((f"lr_{lr:.0e}", config))

    # 5. Dropout Rate Ablation
    print("📋 Generating Dropout Rate Ablation Experiments...")

    dropout_rates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    for dropout in dropout_rates:
        config = base_config.copy()
        config["encoder_config"]["dropout_rate"] = dropout
        config["fno_processor_config"]["dropout_rate"] = dropout
        experiments.append((f"dropout_{dropout:.2f}", config))

    # 6. Batch Size Ablation
    print("📋 Generating Batch Size Ablation Experiments...")

    batch_sizes = [32, 64, 128, 256, 512]
    for batch_size in batch_sizes:
        config = base_config.copy()
        config["batch_size"] = batch_size
        experiments.append((f"batch_{batch_size}", config))

    # 7. Weight Decay Ablation
    print("📋 Generating Weight Decay Ablation Experiments...")

    weight_decays = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
    for wd in weight_decays:
        config = base_config.copy()
        config["weight_decay"] = wd
        experiments.append((f"wd_{wd:.0e}", wd))

    print(f"📊 Total experiments generated: {len(experiments)}")
    return experiments


def main():
    """Run comprehensive ablation study."""
    # Check environment first
    if not check_environment():
        print("❌ Environment check failed. Please install required dependencies.")
        return False

    # Generate all experiments
    experiments = generate_ablation_experiments()

    print("\n🎯 CNN Encoder Ablation Study")
    print("=" * 60)
    print(f"Running {len(experiments)} ablation experiments...")
    print("Each experiment will be logged to wandb project: latent_fno")
    print("Target: Optimize CNN encoder with correlation > 0.64912")
    print("=" * 60)

    results = {}
    successful = 0
    failed = 0
    best_correlation = 0.0
    best_config = None

    for i, (config_name, config_params) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Processing: {config_name}")

        # Run the experiment
        success, exp_results = run_experiment(config_name, config_params)
        results[config_name] = {
            "success": success,
            "results": exp_results,
            "config": config_params,
        }

        if success:
            successful += 1
            # Check if this is the best result
            if "test_correlation" in exp_results:
                correlation = exp_results["test_correlation"]
                if correlation > best_correlation:
                    best_correlation = correlation
                    best_config = config_name
                    print(f"🏆 NEW BEST CORRELATION: {correlation:.6f} ({config_name})")
        else:
            failed += 1

        # Brief pause between experiments
        if i < len(experiments):
            print("\n⏳ Waiting 5 seconds before next experiment...")
            time.sleep(5)

    # Print final summary
    print("\n" + "=" * 60)
    print("🎉 ABLATION STUDY COMPLETED!")
    print("=" * 60)
    print(f"✅ Successful: {successful}/{len(experiments)}")
    print(f"❌ Failed: {failed}/{len(experiments)}")
    print(f"🏆 Best correlation: {best_correlation:.6f} ({best_config})")

    # Save results to file
    results_file = "ablation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"📁 Results saved to: {results_file}")

    # Print top 10 results
    print("\n🏆 TOP 10 RESULTS:")
    print("-" * 40)

    successful_results = [
        (name, data)
        for name, data in results.items()
        if data["success"] and "test_correlation" in data["results"]
    ]
    successful_results.sort(
        key=lambda x: x[1]["results"]["test_correlation"], reverse=True
    )

    for i, (name, data) in enumerate(successful_results[:10], 1):
        corr = data["results"]["test_correlation"]
        print(f"{i:2d}. {name}: {corr:.6f}")

    print(
        "\n🌐 View all results at: https://wandb.ai/kurtwal98-university-of-california-berkeley/latent_fno"
    )

    return successful == len(experiments)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
