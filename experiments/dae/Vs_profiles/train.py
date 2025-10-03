import torch
import torch.nn as nn
from tqdm import tqdm
from utils import CosineWarmupScheduler, TrainingConfig, save_checkpoint

from wave_surrogate.logging_setup import setup_logging
from wave_surrogate.models.dae import DecoupledAutoencoder

logger = setup_logging()


def train_stage1(
    model: DecoupledAutoencoder,
    train_loader,
    val_loader,
    config: TrainingConfig,
    run=None,
):
    """
    Handles the training for Stage 1.
    The training optimizes both the encoder and the auxiliary decoder.
    """
    logger.info("--- Starting Stage 1 Training ---")
    optimizer = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.aux_decoder.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=config.beta_S,
    )
    criterion = nn.L1Loss()
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=config.warm_epochs,
        max_epochs=config.epochs_stage1,
        min_lr=config.min_lr,
    )

    best_val = float("inf")
    wandb = __import__("wandb") if run else None
    for epoch in range(config.epochs_stage1):
        model.train()
        train_loss = 0.0
        for images, _ in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{config.epochs_stage1}"
        ):
            images = images.to(config.device)
            optimizer.zero_grad()
            reconstructions, z = model.forward_stage1(images)
            loss = criterion(reconstructions, images)
            # L1 Regularization on latent space
            if config.l1_lambda > 0:
                l1_penalty = config.l1_lambda * torch.norm(z, 1)
                loss += l1_penalty
            # Total Variation Regularization on reconstructions
            if config.tv_lambda > 0:
                tv_penalty = config.tv_lambda * torch.sum(
                    torch.abs(reconstructions[:, 1:] - reconstructions[:, :-1])
                )
                loss += tv_penalty

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        val_loss = evaluate(model, val_loader, criterion, config, stage1=True)
        # Checkpoint if improved
        if val_loss < best_val:
            best_val = val_loss
            logger.info(
                f"Stage1: Validation improved to {val_loss:.6f}, saving checkpoint."
            )
            # Save encoder + aux_decoder
            save_path = getattr(config, "stage1_checkpoint", "stage1_best.pth")
            save_checkpoint(
                {
                    "encoder": model.encoder,
                    "aux_decoder": model.aux_decoder,
                },
                optimizer,
                save_path,
            )
            if wandb and hasattr(wandb, "log_artifact"):
                artifact = wandb.Artifact("stage1-checkpoint", type="model")
                artifact.add_file(save_path)
                wandb.log_artifact(artifact)
        # Log to wandb if available
        if wandb is not None:
            try:
                wandb.log(
                    {
                        "stage1/train_loss": train_loss / len(train_loader),
                        "stage1/val_loss": val_loss,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )
            except Exception:
                pass
        logger.info(
            f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader):.4f}, "
            f"Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        scheduler.step(epoch)


def train_stage2(
    model: DecoupledAutoencoder,
    train_loader,
    val_loader,
    config: TrainingConfig,
    run=None,
):
    """
    Handles the training for Stage 2.
    """
    logger.info("\n--- Starting Stage 2 Training ---")
    # Freeze the encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.decoder.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=config.beta_S,
    )
    criterion = nn.MSELoss()
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=config.warm_epochs,
        max_epochs=config.epochs_stage2,
        min_lr=config.min_lr,
    )

    best_val = float("inf")
    wandb = __import__("wandb") if run else None
    for epoch in range(config.epochs_stage2):
        model.train()
        train_loss = 0.0
        for images, _ in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{config.epochs_stage2}"
        ):
            images = images.to(config.device)
            optimizer.zero_grad()
            reconstructions, z = model(images)
            loss = criterion(reconstructions, images)
            # L1 Regularization on latent space
            if config.l1_lambda > 0:
                l1_penalty = config.l1_lambda * torch.norm(z, 1)
                loss += l1_penalty

            # Total Variation Regularization on reconstructions
            if config.tv_lambda > 0:
                tv_penalty = config.tv_lambda * torch.sum(
                    torch.abs(reconstructions[:, 1:] - reconstructions[:, :-1])
                )
                loss += tv_penalty

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        val_loss = evaluate(model, val_loader, criterion, config, stage1=False)
        if val_loss < best_val:
            best_val = val_loss
            logger.info(
                f"Stage2: Validation improved to {val_loss:.6f}, saving checkpoint."
            )
            save_path = getattr(config, "stage2_checkpoint", "stage2_best.pth")
            save_checkpoint(
                {
                    "decoder": model.decoder,
                },
                optimizer,
                save_path,
            )
            if wandb and hasattr(wandb, "log_artifact"):
                artifact = wandb.Artifact("stage2-checkpoint", type="model")
                artifact.add_file(save_path)
                wandb.log_artifact(artifact)
        if wandb is not None:
            try:
                wandb.log(
                    {
                        "stage2/train_loss": train_loss / len(train_loader),
                        "stage2/val_loss": val_loss,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )
            except Exception:
                pass
        logger.info(
            f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader):.4f}, "
            f"Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        scheduler.step(epoch)


def evaluate(model, dataloader, criterion, config, stage1):
    """
    Evaluates the model on a given dataset.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(config.device)
            if stage1:
                reconstructions, z = model.forward_stage1(images)
            else:
                reconstructions, z = model(images)
            loss = criterion(reconstructions, images)
            # L1 Regularization on latent space
            if config.l1_lambda > 0:
                l1_penalty = config.l1_lambda * torch.norm(z, 1)
                loss += l1_penalty
            # Total Variation Regularization on reconstructions
            if config.tv_lambda > 0:
                tv_penalty = config.tv_lambda * torch.sum(
                    torch.abs(reconstructions[:, 1:] - reconstructions[:, :-1])
                )
                loss += tv_penalty
            total_loss += loss.item()
    return total_loss / len(dataloader)
