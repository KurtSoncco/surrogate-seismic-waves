# train.py
"""
Enhanced training script for the latent FNO pipeline with comprehensive logging and evaluation.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..configs.config import LatentFNOConfig
from ..utils.data_utils import SeismicDataLoader
from ..utils.metrics import MetricsCalculator, ModelEvaluator
from ..utils.wandb_utils import WandbLogger, log_experiment_summary
from .pipeline import create_pipeline


class LatentFNOTrainer:
    """Enhanced trainer class for latent FNO experiments."""

    def __init__(self, config: LatentFNOConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Set random seeds for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

        # Initialize model
        self.model = create_pipeline(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            latent_dim=config.latent_dim,
            encoder_type=config.encoder_type,
            decoder_type=config.decoder_type,
            fno_processor_type=config.fno_processor_type,
            encoder_config=config.encoder_config,
            decoder_config=config.decoder_config,
            fno_processor_config=config.fno_processor_config,
        ).to(self.device)

        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=config.patience,
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # Initialize metrics calculator and evaluator
        self.metrics_calculator = MetricsCalculator(device=config.device)
        self.evaluator = ModelEvaluator(device=config.device)

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.best_epoch = 0

        # Initialize wandb logger
        self.wandb_logger = None
        if config.use_wandb:
            self.wandb_logger = WandbLogger(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=vars(config),
                tags=[config.experiment_name],
                notes=config.description,
                enabled=True,
            )

            # Watch model for gradients
            self.wandb_logger.watch_model(self.model)

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
        """Load and prepare data using the new data utilities."""
        # Create data loader
        data_loader = SeismicDataLoader(
            vs_data_path=self.config.vs_data_path,
            ttf_data_path=self.config.ttf_data_path,
            freq_data_path=self.config.freq_data_path,
            input_dim=self.config.input_dim,
            device=self.config.device,
        )

        # Load and preprocess data
        vs_data, ttf_data, freq_data = data_loader.load_raw_data()
        vs_data, ttf_data = data_loader.preprocess_data(
            filter_f0_threshold=2.0, normalize=True, remove_outliers=True
        )

        # Create data loaders
        train_loader, val_loader, test_loader = data_loader.create_data_loaders(
            train_split=self.config.train_split,
            val_split=self.config.val_split,
            test_split=self.config.test_split,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        # Store data loader for later use
        self.data_loader = data_loader

        return train_loader, val_loader, test_loader, freq_data

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch with enhanced logging."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc="Training")
        for batch_X, batch_y in pbar:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_X)
            loss = self.criterion(predictions, batch_y)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        return total_loss / num_batches

    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate the model with comprehensive metrics."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)

                total_loss += loss.item()
                num_batches += 1

                all_predictions.append(predictions.cpu())
                all_targets.append(batch_y.cpu())

        # Calculate comprehensive metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        metrics = self.metrics_calculator.calculate_all_metrics(
            all_predictions, all_targets
        )

        return total_loss / num_batches, metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """Enhanced training with comprehensive logging."""
        print(
            f"Starting training with {len(train_loader)} train batches and {len(val_loader)} val batches"
        )
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Log model parameters to wandb (without step to avoid ordering issues)
        if self.wandb_logger:
            self.wandb_logger.log_model_parameters(self.model)

        # Ensure val_metrics is always defined in case the training loop does not run
        val_metrics: Dict[str, float] = {}

        for epoch in range(self.config.num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            self.val_losses.append(val_loss)

            # Update scheduler
            self.scheduler.step(val_loss)

            # Prepare metrics for logging
            epoch_metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }

            # Add key validation metrics
            key_metrics = ["mse", "rmse", "mae", "pearson_correlation", "r2"]
            for metric in key_metrics:
                if metric in val_metrics:
                    epoch_metrics[f"val_{metric}"] = val_metrics[metric]

            # Log to wandb
            if self.wandb_logger:
                self.wandb_logger.log_training_metrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    learning_rate=self.optimizer.param_groups[0]["lr"],
                    additional_metrics={
                        f"val_{k}": v
                        for k, v in val_metrics.items()
                        if k in key_metrics
                    },
                )

            # Print progress
            if epoch % self.config.log_interval == 0:
                print(
                    f"Epoch {epoch:4d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}"
                )
                print(
                    f"  Val Metrics - MSE: {val_metrics.get('mse', 0):.6f}, "
                    f"MAE: {val_metrics.get('mae', 0):.6f}, "
                    f"Corr: {val_metrics.get('pearson_correlation', 0):.6f}"
                )

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_checkpoint(
                    self.config.model_save_dir / "best_model.pth",
                    epoch=epoch,
                    loss=val_loss,
                    metrics=val_metrics,
                )

                # Log best model to wandb
                if self.wandb_logger:
                    self.wandb_logger.log_model_checkpoint(
                        self.model, epoch, val_metrics, "best_model"
                    )

            # Save periodic checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(
                    self.config.model_save_dir / f"checkpoint_epoch_{epoch}.pth",
                    epoch=epoch,
                    loss=val_loss,
                    metrics=val_metrics,
                )

            # Early stopping
            if epoch - self.best_epoch > self.config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(
            f"Training completed. Best validation loss: {self.best_val_loss:.6f} at epoch {self.best_epoch}"
        )

        return {
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "final_val_metrics": val_metrics,
        }

    def evaluate(
        self, test_loader: DataLoader, freq_data: np.ndarray
    ) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        # Load best model
        checkpoint_path = self.config.model_save_dir / "best_model.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded best model from epoch {checkpoint['epoch']}")

        # Comprehensive evaluation
        results = self.evaluator.evaluate_model(
            model=self.model,
            dataloader=test_loader,
            frequency_data=freq_data,
            criterion=self.criterion,
        )

        # Print detailed results
        from ..utils.metrics import create_metrics_summary

        print("\n" + create_metrics_summary(results["metrics"]))

        # Log predictions and correlation analysis to wandb
        if self.wandb_logger:
            self.wandb_logger.log_evaluation_metrics(results["metrics"])
            self.wandb_logger.log_predictions(
                results["predictions"], results["targets"], max_samples=20
            )
            self.wandb_logger.log_correlation_analysis(
                results["predictions"], results["targets"], freq_data
            )

        # Save results
        results_path = self.config.results_dir / "test_results.pt"
        torch.save(
            {
                "predictions": results["predictions"],
                "targets": results["targets"],
                "metrics": results["metrics"],
            },
            results_path,
        )

        return results

    def save_checkpoint(
        self, path: Path, epoch: int, loss: float, metrics: Dict[str, float], **kwargs
    ):
        """Save model checkpoint with enhanced information."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": vars(self.config),
            "epoch": epoch,
            "loss": loss,
            "metrics": metrics,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            **kwargs,
        }

        # Add normalization parameters if available
        if hasattr(self, "data_loader") and self.data_loader.normalization_params:
            checkpoint["normalization_params"] = self.data_loader.normalization_params

        torch.save(checkpoint, path)

    def save_results(self, results: Dict[str, Any]):
        """Save comprehensive training results."""
        results_path = self.config.results_dir / "training_results.pkl"
        with open(results_path, "wb") as f:
            pickle.dump(results, f)

        # Save config
        config_path = self.config.results_dir / "config.pkl"
        with open(config_path, "wb") as f:
            pickle.dump(self.config, f)

        # Save normalization parameters
        if hasattr(self, "data_loader"):
            norm_path = self.config.results_dir / "normalization_params.pkl"
            self.data_loader.save_normalization_params(str(norm_path))

    def finish(self):
        """Finish training and logging."""
        if self.wandb_logger:
            # Log final experiment summary
            model_info = {
                "total_parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in self.model.parameters() if p.requires_grad
                ),
            }

            log_experiment_summary(
                logger=self.wandb_logger,
                config=vars(self.config),
                results={
                    "best_val_loss": self.best_val_loss,
                    "best_epoch": self.best_epoch,
                },
                model_info=model_info,
            )

            self.wandb_logger.finish()


def main():
    """Main training function."""
    # Load configuration
    config = LatentFNOConfig()

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

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
