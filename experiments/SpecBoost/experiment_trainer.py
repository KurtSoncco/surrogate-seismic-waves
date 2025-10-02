# experiment_trainer.py
"""
A dedicated, simplified training loop for the debugging experiments.
"""

import config
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

from wave_surrogate.logging_setup import setup_logging

logger = setup_logging()


def run_experiment_training_loop(
    model, train_loader, val_loader, learning_rate, experiment_name
):
    """
    Trains a simplified model and reports if learning occurred.

    Returns:
        bool: True if validation loss improved, False otherwise.
    """
    model.to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    initial_val_loss = float("inf")
    epochs_without_improvement = 0
    PATIENCE = 15  # Use a shorter patience for these quick experiments

    logger.info(f"--- Starting Experiment: {experiment_name} ---")
    logger.info(f"Optimizer LR: {learning_rate}, Patience: {PATIENCE}")

    t = trange(100, desc=f"Training {experiment_name}")  # Train for max 100 epochs
    for epoch in t:
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            targets = targets.to(config.DEVICE)

            # Handle different input formats (single or tuple)
            if isinstance(inputs, tuple):
                inputs = [i.to(config.DEVICE) for i in inputs]
                outputs = model(*inputs)
            else:
                inputs = inputs.to(config.DEVICE)
                outputs = model(inputs)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                targets = targets.to(config.DEVICE)
                if isinstance(inputs, tuple):
                    inputs = [i.to(config.DEVICE) for i in inputs]
                    outputs = model(*inputs)
                else:
                    inputs = inputs.to(config.DEVICE)
                    outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        if epoch == 0:
            initial_val_loss = val_loss

        t.set_postfix(train_loss=train_loss / len(train_loader), val_loss=val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    logger.info(f"--- Experiment {experiment_name} Finished ---")
    logger.info(f"Initial validation loss: {initial_val_loss:.6f}")
    logger.info(f"Best validation loss:    {best_val_loss:.6f}")

    # A simple check for any learning
    if best_val_loss < initial_val_loss * 0.98:  # Require at least 2% improvement
        logger.info("Conclusion: SUCCESS - The model showed evidence of learning.")
        return True
    else:
        logger.info(
            "Conclusion: FAILURE - The model did not learn a meaningful signal."
        )
        return False
