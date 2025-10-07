import ast
import math
import os
from dataclasses import dataclass, field
from typing import List, Union

import torch


@dataclass
class TrainingConfig:
    """Configuration for the training process."""

    # Model parameters
    ## Encoder/Decoder architecture
    latent_dim: int = 400  # Dimensionality of the latent space
    hidden_dim_encoder: Union[int, List[int]] = field(
        default_factory=lambda: [1024, 512, 256, 128]
    )
    hidden_dim_decoder: Union[int, List[int]] = field(
        default_factory=lambda: [128, 256, 512, 1024]
    )

    hidden_dim_aux_decoder: Union[int, List[int]] = field(
        default_factory=lambda: [128, 256, 512, 1024]
    )

    dropout_rate: float = 0.75  # Dropout rate for the auxiliary decoder

    # Training parameters
    batch_size: int = 128
    epochs_stage1: int = 500
    epochs_stage2: int = 500
    warm_epochs: int = 150
    lr: float = 5e-3
    min_lr: float = 1e-6
    weight_decay: float = 1e-5
    beta_S: tuple = (0.6, 0.80)
    l1_lambda: float = 0  # L1 regularization weight
    tv_lambda: float = 0  # Total variation regularization weight
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    ## Data
    dataset_path: str = "./data/1D Profiles/Soil_Bedrock"
    tf_data_path: str = os.path.join(dataset_path, "TTF_data_1000.parquet")
    freq_data_path: str = os.path.join(dataset_path, "TTF_freq_1000.csv")
    input_dim: int = 1000  # Dimensionality of input data

    ## Image path
    results_path: str = "./experiments/dae/TF/images"

    ## Model saving and artifacts
    model_dir: str = "./experiments/dae/TF/models"
    encoder_path: str = "encoder.pth"
    decoder_path: str = "decoder.pth"
    # Checkpoints and artifacts
    stage1_checkpoint: str = os.path.join(model_dir, "stage1_best.pth")
    stage2_checkpoint: str = os.path.join(model_dir, "stage2_best.pth")
    reconstruction_path: str = os.path.join(results_path, "reconstruction.png")
    reconstruction_tensor_path: str = os.path.join(results_path, "reconstruction.pt")
    # W&B
    use_wandb: bool = True
    wandb_project: str | None = "dae-embeddings"
    run_name: str = "TF_dae_experiment"
    # Seed for reproducibility
    seed: int = 42


class CosineWarmupScheduler:
    """
    A learning rate scheduler with cosine annealing and a warmup phase.
    """

    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr: float = 0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.min_lr = min_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (
                self.max_epochs - self.warmup_epochs
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = self.min_lr + (self.base_lr - self.min_lr) * cosine_decay

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


def save_checkpoint(model, optimizer, path):
    """Saves model or model-state dict and optimizer state to the given path.

    model can be an object with .state_dict() or already a dict of module states
    (useful when saving multiple modules together).
    """
    if hasattr(model, "state_dict"):
        model_state = model.state_dict()
    else:
        # assume it's a mapping already
        model_state = model

    state = {"model_state": model_state}
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    torch.save(state, path)


def load_checkpoint(model, optimizer, path, map_location=None):
    """Loads model and optimizer state from path.

    If model is an object with .load_state_dict it will be loaded. If model is None
    and the checkpoint contains a dict of states, the dict will be returned.
    """
    checkpoint = torch.load(path, map_location=map_location)
    if model is not None and hasattr(model, "load_state_dict"):
        model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return model, optimizer, checkpoint.get("model_state")


# --- Metrics ---
def mse_metric(preds, targets):
    return torch.mean((preds - targets) ** 2).item()


def mae_metric(preds, targets):
    return torch.mean(torch.abs(preds - targets)).item()


def r2_metric(preds, targets):
    # R^2 = 1 - SS_res / SS_tot
    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return (1.0 - ss_res / ss_tot).item()


def total_variation_loss(output):
    # Calculate the difference between adjacent elements
    diff = output[:, 1:] - output[:, :-1]
    return torch.sum(torch.abs(diff))


# --- W&B initialization ---
def init_wandb(config, run_name: str = TrainingConfig.run_name):
    """Initialize wandb if available and requested via config.

    Returns the wandb module if initialized, otherwise None.
    """
    if not getattr(config, "use_wandb", False):
        return None

    try:
        import wandb
    except Exception:
        print(
            "wandb requested but not installed. Install with `pip install wandb` to enable logging."
        )
        return None

    # Try to pick up project and API key from environment variables typically used in this repo
    # Non-sensitive: we don't print secrets. We just set WANDB env vars if the repo .env used custom names.
    # New format: W_B_KEYS may contain a dict mapping project->api_key
    if "W_B_KEYS" in os.environ and not os.environ.get("WANDB_API_KEY"):
        try:
            parsed = (
                ast.literal_eval(os.environ["W_B_KEYS"])
                if isinstance(os.environ["W_B_KEYS"], str)
                else os.environ["W_B_KEYS"]
            )
            if isinstance(parsed, dict) and parsed:
                # choose API key for requested project or the first entry
                chosen_project = config.wandb_project or next(iter(parsed.keys()))
                api_key = parsed.get(chosen_project) or next(iter(parsed.values()))
                if api_key:
                    os.environ["WANDB_API_KEY"] = api_key
                    # set project if not set
                    if not os.environ.get("WANDB_PROJECT"):
                        os.environ["WANDB_PROJECT"] = chosen_project
        except Exception:
            # fall back to older vars if parsing fails
            pass

    if "W_B_PROJECT_ID" in os.environ and not os.environ.get("WANDB_PROJECT"):
        os.environ["WANDB_PROJECT"] = os.environ["W_B_PROJECT_ID"]
    if "W_B_API" in os.environ and not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = os.environ["W_B_API"]

    project = (
        config.wandb_project
        or os.environ.get("WANDB_PROJECT")
        or os.environ.get("W_B_PROJECT_ID")
    )

    try:
        # Increase timeout to avoid GraphQL timeouts
        os.environ["WANDB_API_TIMEOUT"] = "60"
        run = wandb.init(project=project, name=run_name, config=config)
        return run
    except Exception as e:
        print(f"Failed to initialize wandb: {e}")
        return None
