# wandb_utils.py
"""
Weights & Biases utilities for experiment tracking and logging.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

import wandb


class WandbLogger:
    """
    Wrapper class for Weights & Biases logging with experiment-specific functionality.
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        entity: Optional[str] = None,
        save_dir: Optional[Path] = None,
        enabled: bool = True,
    ):
        """
        Initialize W&B logger.

        Args:
            project: W&B project name
            name: Run name (optional)
            config: Configuration dictionary
            tags: List of tags for the run
            notes: Notes for the run
            entity: W&B entity (team/username)
            save_dir: Directory to save local artifacts
            enabled: Whether to enable W&B logging
        """
        self.enabled = enabled
        self.save_dir = save_dir or Path("wandb_logs")
        self.save_dir.mkdir(exist_ok=True)

        if self.enabled:
            wandb.init(
                project=project,
                name=name,
                config=config,
                tags=tags,
                notes=notes,
                entity=entity,
                dir=str(self.save_dir),
            )

            # Log configuration
            if config:
                self.log_config(config)

    def log_config(self, config: Dict[str, Any]):
        """Log configuration to W&B."""
        if not self.enabled:
            return

        # Convert config to wandb-compatible format
        wandb_config = {}
        for key, value in config.items():
            if isinstance(value, (int, float, str, bool)):
                wandb_config[key] = value
            elif isinstance(value, (list, tuple)):
                wandb_config[key] = list(value)
            elif isinstance(value, dict):
                wandb_config[key] = json.dumps(value)
            else:
                wandb_config[key] = str(value)

        wandb.config.update(wandb_config, allow_val_change=True)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to W&B."""
        if not self.enabled:
            return

        wandb.log(metrics, step=step)

    def log_model_parameters(self, model: torch.nn.Module):
        """Log model parameters and architecture to W&B."""
        if not self.enabled:
            return

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        wandb.log(
            {
                "model/total_parameters": total_params,
                "model/trainable_parameters": trainable_params,
                "model/parameter_ratio": trainable_params / total_params
                if total_params > 0
                else 0,
            }
        )

        # Log model architecture (simplified)
        if hasattr(model, "config"):
            wandb.log({"model/config": json.dumps(model.config)})

    def log_training_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        learning_rate: float,
        additional_metrics: Optional[Dict[str, float]] = None,
    ):
        """Log training metrics for an epoch."""
        metrics = {
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "train/learning_rate": learning_rate,
        }

        if additional_metrics:
            metrics.update(additional_metrics)

        self.log_metrics(metrics, step=epoch)

    def log_evaluation_metrics(self, metrics: Dict[str, float], prefix: str = "test"):
        """Log evaluation metrics."""
        prefixed_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        self.log_metrics(prefixed_metrics)

    def log_predictions(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sample_indices: Optional[List[int]] = None,
        max_samples: int = 100,
    ):
        """Log prediction vs target plots to W&B."""
        if not self.enabled:
            return

        # Convert to numpy
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()

        # Limit number of samples
        n_samples = min(len(pred_np), max_samples)
        if sample_indices is None:
            # np.random.choice returns an ndarray; convert to a Python list of ints to satisfy the expected type
            sample_indices = np.random.choice(
                len(pred_np), n_samples, replace=False
            ).tolist()

        # Create plots
        assert isinstance(sample_indices, list), (
            "sample_indices should be a list of integers"
        )
        for i, idx in enumerate(sample_indices[:n_samples]):
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

            # Plot prediction and target
            ax1.plot(target_np[idx], label="Target", alpha=0.7)
            ax1.plot(pred_np[idx], label="Prediction", alpha=0.7)
            ax1.set_title(f"Sample {idx}: Prediction vs Target")
            ax1.legend()
            ax1.grid(True)

            # Plot residuals
            residuals = pred_np[idx] - target_np[idx]
            ax2.plot(residuals, color="red", alpha=0.7)
            ax2.set_title(f"Sample {idx}: Residuals (Pred - Target)")
            ax2.grid(True)
            ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)

            # Plot scatter plot
            ax3.scatter(target_np[idx], pred_np[idx], alpha=0.6)
            ax3.plot(
                [target_np[idx].min(), target_np[idx].max()],
                [target_np[idx].min(), target_np[idx].max()],
                "r--",
                alpha=0.8,
            )
            ax3.set_xlabel("Target")
            ax3.set_ylabel("Prediction")
            ax3.set_title(f"Sample {idx}: Prediction vs Target Scatter")
            ax3.grid(True)

            plt.tight_layout()

            # Log to wandb
            wandb.log({f"predictions/sample_{idx}": wandb.Image(fig)})
            plt.close(fig)

    def log_correlation_analysis(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        frequency_data: Optional[np.ndarray] = None,
    ):
        """Log correlation analysis plots."""
        if not self.enabled:
            return

        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()

        # Overall correlation
        overall_corr = np.corrcoef(pred_np.flatten(), target_np.flatten())[0, 1]

        # Per-sample correlations
        sample_corrs = []
        for i in range(len(pred_np)):
            corr = np.corrcoef(pred_np[i], target_np[i])[0, 1]
            sample_corrs.append(corr)

        sample_corrs = np.array(sample_corrs)

        # Log correlation statistics
        wandb.log(
            {
                "correlation/overall": overall_corr,
                "correlation/mean_per_sample": np.mean(sample_corrs),
                "correlation/std_per_sample": np.std(sample_corrs),
                "correlation/min_per_sample": np.min(sample_corrs),
                "correlation/max_per_sample": np.max(sample_corrs),
            }
        )

        # Create correlation plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Histogram of per-sample correlations
        ax1.hist(sample_corrs, bins=30, alpha=0.7, edgecolor="black")
        ax1.axvline(
            np.mean(sample_corrs),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(sample_corrs):.3f}",
        )
        ax1.set_xlabel("Correlation Coefficient")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution of Per-Sample Correlations")
        ax1.legend()
        ax1.grid(True)

        # Correlation vs sample index
        ax2.plot(sample_corrs, alpha=0.7)
        ax2.axhline(
            np.mean(sample_corrs),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(sample_corrs):.3f}",
        )
        ax2.set_xlabel("Sample Index")
        ax2.set_ylabel("Correlation Coefficient")
        ax2.set_title("Correlation vs Sample Index")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        wandb.log({"correlation/analysis": wandb.Image(fig)})
        plt.close(fig)

        # Frequency-dependent correlation if frequency data is available
        if frequency_data is not None:
            self.log_frequency_correlation(pred_np, target_np, frequency_data)

    def log_frequency_correlation(
        self, predictions: np.ndarray, targets: np.ndarray, frequencies: np.ndarray
    ):
        """Log frequency-dependent correlation analysis."""
        if not self.enabled:
            return

        # Calculate correlation at each frequency
        freq_corrs = []
        for i in range(predictions.shape[1]):
            pred_freq = predictions[:, i]
            target_freq = targets[:, i]
            corr = np.corrcoef(pred_freq, target_freq)[0, 1]
            freq_corrs.append(corr)

        freq_corrs = np.array(freq_corrs)

        # Create frequency correlation plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(frequencies, freq_corrs, "b-", linewidth=2)
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Correlation Coefficient")
        ax.set_title("Correlation vs Frequency")
        ax.grid(True)
        ax.set_yscale("log")

        plt.tight_layout()
        wandb.log({"correlation/frequency_dependent": wandb.Image(fig)})
        plt.close(fig)

    def log_model_checkpoint(
        self,
        model: torch.nn.Module,
        epoch: int,
        metrics: Dict[str, float],
        artifact_name: str = "model_checkpoint",
    ):
        """Log model checkpoint as W&B artifact."""
        if not self.enabled:
            return

        # Save model locally
        checkpoint_path = self.save_dir / f"{artifact_name}_epoch_{epoch}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
            },
            checkpoint_path,
        )

        # Create W&B artifact
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description=f"Model checkpoint at epoch {epoch}",
        )
        artifact.add_file(str(checkpoint_path))

        wandb.log_artifact(artifact)

    def log_hyperparameter_sweep(self, sweep_config: Dict[str, Any]):
        """Log hyperparameter sweep configuration."""
        if not self.enabled:
            return

        wandb.log({"sweep_config": json.dumps(sweep_config)})

    def watch_model(
        self, model: torch.nn.Module, log: str = "gradients", log_freq: int = 100
    ):
        """Watch model for gradients and parameters."""
        if not self.enabled:
            return

        wandb.watch(model, log=log, log_freq=log_freq)

    def finish(self):
        """Finish the W&B run."""
        if self.enabled:
            wandb.finish()


def create_wandb_logger(
    project: str,
    config: Dict[str, Any],
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    enabled: bool = True,
) -> WandbLogger:
    """
    Factory function to create a WandbLogger.

    Args:
        project: W&B project name
        config: Configuration dictionary
        name: Run name
        tags: List of tags
        enabled: Whether to enable W&B logging

    Returns:
        WandbLogger instance
    """
    return WandbLogger(
        project=project, name=name, config=config, tags=tags, enabled=enabled
    )


def log_experiment_summary(
    logger: WandbLogger,
    config: Dict[str, Any],
    results: Dict[str, Any],
    model_info: Dict[str, Any],
):
    """
    Log a comprehensive experiment summary.

    Args:
        logger: WandbLogger instance
        config: Experiment configuration
        results: Experiment results
        model_info: Model information
    """
    if not logger.enabled:
        return

    # Log configuration summary
    summary = {
        "experiment/name": config.get("experiment_name", "unknown"),
        "experiment/description": config.get("description", ""),
        "model/encoder_type": config.get("encoder_type", "unknown"),
        "model/decoder_type": config.get("decoder_type", "unknown"),
        "model/fno_processor_type": config.get("fno_processor_type", "unknown"),
        "model/latent_dim": config.get("latent_dim", 0),
        "training/best_epoch": results.get("best_epoch", 0),
        "training/best_val_loss": results.get("best_val_loss", float("inf")),
    }

    # Add test metrics
    if "test_metrics" in results:
        for key, value in results["test_metrics"].items():
            summary[f"test/{key}"] = value

    # Add model info
    if "total_parameters" in model_info:
        summary["model/total_parameters"] = model_info["total_parameters"]

    # Log summary
    wandb.summary.update(summary)
