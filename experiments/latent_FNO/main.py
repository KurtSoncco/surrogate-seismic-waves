# main.py
"""
Main script for running latent FNO experiments.
This script provides a command-line interface for training and evaluating models.
"""

import argparse
import os
import sys
from pathlib import Path

from src.configs.config import CONFIGS, get_config
from src.models.train import LatentFNOTrainer


def _ensure_local_src_on_path():
    test_dir = os.path.abspath(os.path.dirname(__file__))
    cur = test_dir
    for _ in range(6):  # search up to 6 levels
        candidate_src = os.path.join(cur, "src")
        if os.path.isdir(candidate_src):
            # add `cur` so `import src...` resolves to cur/src
            if cur not in sys.path:
                sys.path.insert(0, cur)
            return
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    # fallback: also try adding experiments/latent_FNO top folder as fallback
    fallback = os.path.abspath(os.path.join(test_dir, ".."))
    if fallback not in sys.path:
        sys.path.insert(0, fallback)


_ensure_local_src_on_path()


def train_experiment(config_name: str = "baseline", use_wandb: bool = False):
    """Train a model with the specified configuration."""
    print(f"Training experiment: {config_name}")

    # Get configuration
    config = get_config(config_name)

    # Override wandb settings
    config.use_wandb = use_wandb
    if use_wandb:
        config.wandb_run_name = f"{config_name}_run"

    # Create trainer
    trainer = LatentFNOTrainer(config)

    # Load data
    train_loader, val_loader, test_loader, freq_data = trainer.load_data()

    # Train model
    training_results = trainer.train(train_loader, val_loader)

    # Evaluate model
    test_results = trainer.evaluate(test_loader, freq_data)

    # Save results
    all_results = {"training_results": training_results, "test_results": test_results}
    trainer.save_results(all_results)

    # Finish logging
    trainer.finish()

    print(f"Experiment {config_name} completed successfully!")
    return all_results


def run_ablation_study():
    """Run ablation study comparing different components."""
    print("Running ablation study...")

    ablation_configs = [
        ("full_pipeline", "baseline"),
        ("no_fno", "baseline"),  # We'll modify this
        ("encoder_only", "baseline"),  # We'll modify this
    ]

    results = {}

    for study_name, base_config_name in ablation_configs:
        print(f"\nRunning ablation: {study_name}")

        config = get_config(base_config_name)
        config.experiment_name = study_name
        config.use_wandb = False  # Disable wandb for ablation

        # Create trainer
        trainer = LatentFNOTrainer(config)

        # Load data
        train_loader, val_loader, test_loader, freq_data = trainer.load_data()

        # Quick training for ablation (fewer epochs)
        original_epochs = config.num_epochs
        config.num_epochs = 50  # Reduced for quick ablation

        # Train model
        training_results = trainer.train(train_loader, val_loader)

        # Evaluate model
        test_results = trainer.evaluate(test_loader, freq_data)

        results[study_name] = {
            "training_results": training_results,
            "test_results": test_results,
        }

        print(f"Ablation {study_name} completed.")

    # Print comparison
    print("\n" + "=" * 60)
    print("ABLATION STUDY RESULTS")
    print("=" * 60)

    for study_name, result in results.items():
        test_metrics = result["test_results"]["metrics"]
        print(f"\n{study_name.upper()}:")
        print(f"  MSE: {test_metrics.get('mse', 0):.6f}")
        print(f"  MAE: {test_metrics.get('mae', 0):.6f}")
        print(f"  Correlation: {test_metrics.get('pearson_correlation', 0):.6f}")
        print(f"  R²: {test_metrics.get('r2', 0):.6f}")

    return results


def compare_configurations():
    """Compare different predefined configurations."""
    print("Comparing different configurations...")

    configs_to_compare = ["baseline", "cnn_encoder", "sequence_fno", "multiscale_fno"]

    results = {}

    for config_name in configs_to_compare:
        print(f"\nTesting configuration: {config_name}")

        config = get_config(config_name)
        config.num_epochs = 100  # Reduced for comparison
        config.use_wandb = False

        # Create trainer
        trainer = LatentFNOTrainer(config)

        # Load data
        train_loader, val_loader, test_loader, freq_data = trainer.load_data()

        # Train model
        training_results = trainer.train(train_loader, val_loader)

        # Evaluate model
        test_results = trainer.evaluate(test_loader, freq_data)

        results[config_name] = {
            "training_results": training_results,
            "test_results": test_results,
        }

    # Print comparison
    print("\n" + "=" * 80)
    print("CONFIGURATION COMPARISON RESULTS")
    print("=" * 80)
    print(
        f"{'Config':<20} {'MSE':<12} {'MAE':<12} {'Correlation':<12} {'R²':<12} {'Best Epoch':<12}"
    )
    print("-" * 80)

    for config_name, result in results.items():
        test_metrics = result["test_results"]["metrics"]
        training_results = result["training_results"]

        print(
            f"{config_name:<20} "
            f"{test_metrics.get('mse', 0):<12.6f} "
            f"{test_metrics.get('mae', 0):<12.6f} "
            f"{test_metrics.get('pearson_correlation', 0):<12.6f} "
            f"{test_metrics.get('r2', 0):<12.6f} "
            f"{training_results.get('best_epoch', 0):<12}"
        )

    return results


def run_quick_test():
    """Run a quick test to verify everything works."""
    print("Running quick functionality test...")

    # Import test functions
    sys.path.append(str(Path(__file__).parent / "tests"))
    from tests.test_comprehensive import (
        test_all_encoders,
        test_all_fno_processors,
        test_basic_functionality,
    )

    tests = [test_basic_functionality, test_all_encoders, test_all_fno_processors]

    for test_func in tests:
        try:
            test_func()
            print(f"✓ {test_func.__name__} passed")
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            return False

    print("✓ Quick test completed successfully!")
    return True


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="Latent FNO Experiment Runner")
    parser.add_argument(
        "command",
        choices=["train", "ablation", "compare", "test", "list-configs"],
        help="Command to run",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="baseline",
        help="Configuration name (for train command)",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size (overrides config)"
    )

    args = parser.parse_args()

    if args.command == "list-configs":
        print("Available configurations:")
        for config_name in CONFIGS.keys():
            print(f"  - {config_name}")
        return

    elif args.command == "test":
        success = run_quick_test()
        sys.exit(0 if success else 1)

    elif args.command == "train":
        # Override config if specified
        config = get_config(args.config)
        if args.epochs:
            config.num_epochs = args.epochs
        if args.batch_size:
            config.batch_size = args.batch_size

        results = train_experiment(args.config, args.wandb)
        print("Training completed successfully!")

    elif args.command == "ablation":
        results = run_ablation_study()
        print("Ablation study completed!")

    elif args.command == "compare":
        results = compare_configurations()
        print("Configuration comparison completed!")

    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
