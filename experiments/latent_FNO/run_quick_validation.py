#!/usr/bin/env python3
"""
Quick validation ablation study for CNN encoder.
This runs just a few key experiments to validate the approach before running the full ablation.
"""

import json
import subprocess
import sys
import time


def run_experiment(config_name: str) -> tuple[bool, dict]:
    """Run a single experiment and return success status and results."""
    print(f"\n🚀 Starting validation experiment: {config_name}")
    print("-" * 50)

    try:
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


def parse_experiment_results(output: str) -> dict:
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
        elif "min sample correlation:" in line.lower():
            try:
                results["min_sample_correlation"] = float(line.split(":")[1].strip())
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


def main():
    """Run quick validation ablation study."""
    print("🎯 CNN Encoder Quick Validation Ablation")
    print("=" * 50)
    print("Testing key hyperparameters to validate approach")
    print("Baseline: CNN encoder with MIN sample correlation 0.64912")
    print("=" * 50)

    # Key experiments to validate the approach
    experiments = [
        # Test different latent dimensions (most impactful)
        "high_latent_dim",  # latent_dim=256
        "low_latent_dim",  # latent_dim=64
        # Test different FNO configurations
        "sequence_fno",  # Different FNO processor
        "multiscale_fno",  # Multiscale FNO
        # Test CNN encoder again to confirm baseline
        "cnn_encoder",  # Our best performer
    ]

    results = {}
    successful = 0
    failed = 0
    best_correlation = 0.0
    best_config = None

    for i, config_name in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Processing: {config_name}")

        # Run the experiment
        success, exp_results = run_experiment(config_name)
        results[config_name] = {"success": success, "results": exp_results}

        if success:
            successful += 1
            # Check if this is the best result (prioritize min_sample_correlation)
            correlation = None
            if "min_sample_correlation" in exp_results:
                correlation = exp_results["min_sample_correlation"]
            elif "test_correlation" in exp_results:
                correlation = exp_results["test_correlation"]

            if correlation is not None and correlation > best_correlation:
                best_correlation = correlation
                best_config = config_name
                print(
                    f"🏆 NEW BEST MIN SAMPLE CORRELATION: {correlation:.6f} ({config_name})"
                )
        else:
            failed += 1

        # Brief pause between experiments
        if i < len(experiments):
            print("\n⏳ Waiting 5 seconds before next experiment...")
            time.sleep(5)

    # Print final summary
    print("\n" + "=" * 50)
    print("🎉 QUICK VALIDATION COMPLETED!")
    print("=" * 50)
    print(f"✅ Successful: {successful}/{len(experiments)}")
    print(f"❌ Failed: {failed}/{len(experiments)}")
    print(f"🏆 Best correlation: {best_correlation:.6f} ({best_config})")

    # Save results to file
    results_file = "validation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"📁 Results saved to: {results_file}")

    # Print all results
    print("\n📊 ALL RESULTS:")
    print("-" * 40)

    for name, data in results.items():
        if data["success"]:
            corr = None
            if "min_sample_correlation" in data["results"]:
                corr = data["results"]["min_sample_correlation"]
            elif "test_correlation" in data["results"]:
                corr = data["results"]["test_correlation"]

            if corr is not None:
                print(f"✅ {name}: {corr:.6f}")
            else:
                print(f"⚠️  {name}: Completed but no correlation found")
        else:
            print(f"❌ {name}: Failed")

    print(
        "\n🌐 View results at: https://wandb.ai/kurtwal98-university-of-california-berkeley/latent_fno"
    )

    # Recommend next steps
    print("\n🎯 NEXT STEPS:")
    if best_correlation > 0.64912:
        print(
            f"✅ Found improvement! Best MIN sample correlation: {best_correlation:.6f}"
        )
        print("   → Run full ablation study to find optimal hyperparameters")
        print("   → Use: python run_focused_ablation.py")
    else:
        print("⚠️  No improvement found in quick validation")
        print("   → Consider different approaches or more extensive search")
        print("   → Check if baseline CNN encoder result was reproducible")

    return successful == len(experiments)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
