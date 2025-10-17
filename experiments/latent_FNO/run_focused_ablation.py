#!/usr/bin/env python3
"""
Focused ablation study for CNN encoder latent FNO model.
This script tests the most promising hyperparameter combinations first,
based on the successful CNN encoder run with correlation 0.64912.

Priority order:
1. Latent dimension (most impactful)
2. FNO processor parameters
3. Learning rate
4. CNN architecture refinements
"""

import json
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple


def run_experiment(
    config_name: str, config_params: Dict[str, Any]
) -> Tuple[bool, Dict[str, Any]]:
    """Run a single experiment and return success status and results."""
    print(f"\n🚀 Starting focused ablation: {config_name}")
    print("-" * 50)

    try:
        # Run the training with wandb enabled, activating virtual environment
        result = subprocess.run(
            [
                "bash",
                "-c",
                "source /home/kurt-asus/surrogate-seismic-waves/.venv/bin/activate && python main.py train --config cnn_encoder --wandb",
            ],
            capture_output=True,
            text=True,
            timeout=1800,
        )  # 30 min timeout

        if result.returncode == 0:
            print(f"✅ {config_name} completed successfully!")
            # Parse results from output
            results = parse_experiment_results(result.stdout)
            return True, results
        else:
            print(f"❌ {config_name} failed with return code {result.returncode}")
            return False, {}

    except subprocess.TimeoutExpired:
        print(f"⏰ {config_name} timed out after 30 minutes")
        return False, {}
    except Exception as e:
        print(f"💥 {config_name} failed with exception: {e}")
        return False, {}


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


def create_focused_experiments() -> List[Tuple[str, Dict[str, Any]]]:
    """Create focused ablation experiments targeting the most promising parameters."""
    experiments = []

    # 1. LATENT DIMENSION ABLATION (Most impactful)
    print("🎯 Phase 1: Latent Dimension Optimization")
    latent_dims = [96, 128, 160, 192, 224, 256, 320, 384]
    for latent_dim in latent_dims:
        experiments.append((f"latent_{latent_dim}", {"latent_dim": latent_dim}))

    # 2. FNO PROCESSOR OPTIMIZATION
    print("🎯 Phase 2: FNO Processor Optimization")

    # FNO modes optimization
    fno_modes = [12, 16, 20, 24, 28, 32]
    for modes in fno_modes:
        experiments.append(
            (
                f"fno_modes_{modes}",
                {
                    "fno_processor_config": {
                        "fno_modes": modes,
                        "fno_width": 64,
                        "num_fno_layers": 3,
                    }
                },
            )
        )

    # FNO width optimization
    fno_widths = [48, 64, 80, 96, 112, 128]
    for width in fno_widths:
        experiments.append(
            (
                f"fno_width_{width}",
                {
                    "fno_processor_config": {
                        "fno_modes": 16,
                        "fno_width": width,
                        "num_fno_layers": 3,
                    }
                },
            )
        )

    # FNO layers optimization
    fno_layers = [2, 3, 4, 5]
    for layers in fno_layers:
        experiments.append(
            (
                f"fno_layers_{layers}",
                {
                    "fno_processor_config": {
                        "fno_modes": 16,
                        "fno_width": 64,
                        "num_fno_layers": layers,
                    }
                },
            )
        )

    # 3. LEARNING RATE OPTIMIZATION
    print("🎯 Phase 3: Learning Rate Optimization")
    learning_rates = [5e-4, 1e-3, 2e-3, 3e-3, 5e-3]
    for lr in learning_rates:
        experiments.append((f"lr_{lr:.0e}", {"learning_rate": lr}))

    # 4. CNN ARCHITECTURE REFINEMENTS
    print("🎯 Phase 4: CNN Architecture Refinements")

    # Different channel configurations (focused on promising ones)
    channel_configs = [
        [1, 32, 64, 128],  # Base
        [1, 48, 96, 192],  # Scaled up
        [1, 24, 48, 96],  # Scaled down
        [1, 32, 64, 128, 192],  # Deeper
    ]

    for i, channels in enumerate(channel_configs):
        experiments.append(
            (
                f"cnn_channels_{i + 1}",
                {
                    "encoder_config": {
                        "channels": channels,
                        "kernel_sizes": [3, 3, 3],
                        "pool_sizes": [2, 2, 2],
                        "dropout_rate": 0.1,
                        "use_adaptive_pool": True,
                    }
                },
            )
        )

    # 5. DROPOUT OPTIMIZATION
    print("🎯 Phase 5: Dropout Rate Optimization")
    dropout_rates = [0.05, 0.1, 0.15, 0.2]
    for dropout in dropout_rates:
        experiments.append(
            (
                f"dropout_{dropout:.2f}",
                {
                    "encoder_config": {"dropout_rate": dropout},
                    "fno_processor_config": {"dropout_rate": dropout},
                },
            )
        )

    # 6. BATCH SIZE OPTIMIZATION
    print("🎯 Phase 6: Batch Size Optimization")
    batch_sizes = [64, 128, 256]
    for batch_size in batch_sizes:
        experiments.append((f"batch_{batch_size}", {"batch_size": batch_size}))

    print(f"📊 Total focused experiments: {len(experiments)}")
    return experiments


def main():
    """Run focused ablation study."""
    print("🎯 CNN Encoder Focused Ablation Study")
    print("=" * 60)
    print("Target: Optimize CNN encoder beyond correlation 0.64912")
    print("Strategy: Test most promising hyperparameters first")
    print("=" * 60)

    # Generate focused experiments
    experiments = create_focused_experiments()

    results = {}
    successful = 0
    failed = 0
    best_correlation = 0.64912  # Our baseline
    best_config = "cnn_encoder_run"  # Our baseline

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
                    print(f"   Improvement: +{correlation - 0.64912:.6f}")
        else:
            failed += 1

        # Brief pause between experiments
        if i < len(experiments):
            print("\n⏳ Waiting 3 seconds before next experiment...")
            time.sleep(3)

    # Print final summary
    print("\n" + "=" * 60)
    print("🎉 FOCUSED ABLATION STUDY COMPLETED!")
    print("=" * 60)
    print(f"✅ Successful: {successful}/{len(experiments)}")
    print(f"❌ Failed: {failed}/{len(experiments)}")
    print(f"🏆 Best correlation: {best_correlation:.6f} ({best_config})")
    print(f"📈 Improvement over baseline: +{best_correlation - 0.64912:.6f}")

    # Save results to file
    results_file = "focused_ablation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"📁 Results saved to: {results_file}")

    # Print top 10 results
    print("\n🏆 TOP 10 RESULTS:")
    print("-" * 50)

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
        improvement = corr - 0.64912
        print(f"{i:2d}. {name}: {corr:.6f} (+{improvement:.6f})")

    print(
        "\n🌐 View all results at: https://wandb.ai/kurtwal98-university-of-california-berkeley/latent_fno"
    )

    return successful == len(experiments)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
