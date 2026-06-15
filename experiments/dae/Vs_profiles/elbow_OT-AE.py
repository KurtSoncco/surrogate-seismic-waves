import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from dataloader import get_material_dataloaders
from OTDecoder import OTDecoder
from OTEncoder import OTEncoder
from utils import TrainingConfig


class OT_Autoencoder(nn.Module):
    """
    An autoencoder that combines the OTEncoder and OTDecoder modules to perform
    dimensionality reduction and reconstruction of 1D features using Optimal Transport.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        ot_steps: int = 15,
        ot_lr: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        self.encoder = OTEncoder(
            latent_dim=latent_dim, ot_steps=ot_steps, ot_lr=ot_lr, device=device
        )
        self.decoder = OTDecoder(
            latent_dim=latent_dim, ot_steps=ot_steps, ot_lr=ot_lr, device=device
        )

    def forward(self, x_or_features: torch.Tensor, grid: torch.Tensor | None = None):
        """
        Encodes input features to the latent space and then decodes them back to the original space.

        Args:
            x_or_features (torch.Tensor): Combined (B, 4, N) or features (B, C, N)
            grid (torch.Tensor | None): Grid (B, N) if not passed in combined

        Returns:
            torch.Tensor: Reconstructed features of shape (B, C, N)
        """
        # Pass through encoder/decoder with combined- or separate-format compatibility
        latent_representation = self.encoder(x_or_features, grid)
        reconstructed_features = self.decoder(
            latent_representation, x_or_features if grid is None else grid
        )
        return reconstructed_features


def evaluate_reconstruction_mse(
    model: OT_Autoencoder, dataloader: torch.utils.data.DataLoader, device: str
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    mse = torch.nn.MSELoss(reduction="sum")
    for batch_inputs, _ in dataloader:
        # batch_inputs: (B, 4, L) = (grid, Vs, Vp, Rho)
        batch_inputs = batch_inputs.to(device)

        # Use combined format directly; keep grads enabled for inner OT optimization
        recon = model(batch_inputs)
        # Compare only feature channels (exclude grid)
        features = batch_inputs[:, 1:, :]
        loss = mse(recon.detach(), features)  # detach recon to avoid holding graph
        total_loss += loss.item()
        total_count += features.numel()
    return total_loss / max(total_count, 1)


def run_elbow(latent_dims: List[int], device: str, save_dir: str) -> None:
    config = TrainingConfig()
    train_loader, val_loader, test_loader, _ = get_material_dataloaders(
        dataset_path=config.materials_data_path,
        batch_size=config.batch_size,
        save_path=str(config.model_dir),
    )

    val_mses = []
    for ld in latent_dims:
        model = OT_Autoencoder(latent_dim=ld, ot_steps=15, ot_lr=0.1, device=device).to(
            device
        )
        mse_val = evaluate_reconstruction_mse(model, val_loader, device)
        val_mses.append(mse_val)
        print(f"latent_dim={ld}: val MSE={mse_val:.6f}")

    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, "elbow_ot_ae.png")
    plt.figure(figsize=(6, 4))
    plt.plot(latent_dims, val_mses, marker="o")
    plt.xlabel("Latent Dimension")
    plt.ylabel("Validation MSE (reconstruction)")
    plt.title("OT-AE Elbow Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved elbow plot to: {fig_path}")


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {DEVICE}")

    # Choose a range of latent dimensions for the elbow analysis
    latent_dims = [16, 32, 64, 128, 256, 512]

    # Save under the configured model directory
    save_dir = str(TrainingConfig().model_dir)
    run_elbow(latent_dims, DEVICE, save_dir)

    # --- Grid Search with K-fold CV ---
    # Grids (adjust as needed)
    grid_latent = [16, 32, 64, 128]
    grid_steps = [50, 100, 250]
    grid_lrs = [0.1]

    # Prepare concatenated dataset from train+val for CV
    config = TrainingConfig()
    train_loader, val_loader, _, _ = get_material_dataloaders(
        dataset_path=config.materials_data_path,
        batch_size=config.batch_size,
        save_path=str(config.model_dir),
    )
    all_batches = [x for loader in [train_loader, val_loader] for x, _ in loader]
    X = torch.cat(all_batches, dim=0)  # (N, 4, L)
    import pandas as pd
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    indices = np.arange(X.size(0))
    # indices = torch.arange(X.size(0))

    results: List[Dict[str, float]] = []
    for ld in grid_latent:
        for steps in grid_steps:
            for lr in grid_lrs:
                fold_mses: List[float] = []
                for train_idx, val_idx in kf.split(indices):
                    xb = X[val_idx].to(DEVICE)
                    model = OT_Autoencoder(
                        latent_dim=ld, ot_steps=steps, ot_lr=lr, device=DEVICE
                    ).to(DEVICE)
                    # Evaluate directly (inner OT optimizes features)
                    mse = torch.nn.MSELoss(reduction="sum")
                    total_loss = 0.0
                    total_count = 0
                    bs = 32
                    for start in range(0, xb.size(0), bs):
                        xbatch = xb[start : start + bs]
                        recon = model(xbatch)
                        feats = xbatch[:, 1:, :]
                        loss = mse(recon.detach(), feats)
                        total_loss += loss.item()
                        total_count += feats.numel()
                    fold_mses.append(total_loss / max(total_count, 1))
                avg_mse = float(sum(fold_mses) / len(fold_mses))
                results.append(
                    {
                        "latent_dim": ld,
                        "ot_steps": steps,
                        "ot_lr": lr,
                        "cv_mse": avg_mse,
                    }
                )
                print(f"CV - ld={ld}, steps={steps}, lr={lr}: MSE={avg_mse:.6f}")

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "ot_ae_grid_search.csv")
    pd.DataFrame(results).to_csv(csv_path, index=False)
    best = min(results, key=lambda r: r["cv_mse"]) if results else None
    if best is not None:
        print("Best config:", best)
