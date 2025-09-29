import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from utils import TrainingConfig, mae_metric, mse_metric, r2_metric

from wave_surrogate.logging_setup import setup_logging
from wave_surrogate.models.dae import DecoupledAutoencoder

sns.set_palette("colorblind")
logger = setup_logging()


def plot_reconstructions(real_profiles, recon_profiles, save_path, dz, num_examples=8):
    """
    Plots a comparison of real and reconstructed Vs profiles and saves the figure.

    Args:
        real_profiles (torch.Tensor): The batch of original 1D profiles.
        recon_profiles (torch.Tensor): The batch of reconstructed 1D profiles.
        save_path (str): The file path to save the resulting plot image.
        num_examples (int): The number of examples to plot from the batch.
    """
    # Ensure we don't try to plot more examples than are available
    num_examples = min(num_examples, real_profiles.shape[0])

    # Create a figure with subplots: num_examples rows, 2 columns
    fig, axes = plt.subplots(
        num_examples, 2, figsize=(8, 2.5 * num_examples), squeeze=False
    )

    # Convert tensors to numpy for plotting
    real_profiles_np = real_profiles.cpu().numpy().squeeze()
    recon_profiles_np = recon_profiles.cpu().numpy().squeeze()

    # Create a depth array based on the length of the profile
    depth_points = real_profiles_np.shape[-1]
    depth = np.arange(depth_points) * dz  # Use the provided dz value

    for i in range(num_examples):
        # --- Plot Real Profile (Left Column) ---
        ax_real = axes[i, 0]
        ax_real.plot(real_profiles_np[i], depth)
        ax_real.set_title("Real")
        ax_real.set_ylabel("Depth")
        ax_real.invert_yaxis()  # Invert the y-axis for depth
        ax_real.xaxis.tick_top()  # Move x-axis ticks to the top
        ax_real.xaxis.set_label_position("top")  # Move x-axis label to the top
        ax_real.set_xlabel("Vs")
        ax_real.grid(True, linestyle="--", alpha=0.6)

        # --- Plot Reconstructed Profile (Right Column) ---
        ax_recon = axes[i, 1]
        ax_recon.plot(recon_profiles_np[i], depth)
        ax_recon.set_title("Reconstructed")
        ax_recon.invert_yaxis()
        ax_recon.xaxis.tick_top()
        ax_recon.xaxis.set_label_position("top")
        ax_recon.set_xlabel("Vs")
        ax_recon.grid(True, linestyle="--", alpha=0.6)

        # Share y-axis with the real plot for consistent depth scaling
        ax_recon.sharey(ax_real)
        # Make y-tick labels invisible on the reconstruction plot for clarity
        plt.setp(ax_recon.get_yticklabels(), visible=False)

    plt.tight_layout(pad=2.0)
    plt.savefig(save_path)
    plt.close(fig)  # Close the figure to free up memory
    logger.info(f"Reconstruction plot saved to '{save_path}'")


def test_model_dae(
    model: DecoupledAutoencoder,
    test_loader,
    config: TrainingConfig,
    scaler: MinMaxScaler,
    run=None,
):
    """
    Tests the final trained model and saves some reconstructed images.
    """
    logger.info("\n--- Testing Model ---")
    model.to(config.device)
    model.eval()
    all_mse = []
    all_mae = []
    all_r2 = []

    first_batch_saved = False
    wandb = __import__("wandb") if run else None
    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(config.device)
            reconstructions, _ = model(images)

            # --- RESCALING STEP ---
            # Use the scaler to convert outputs back to the original Vs scale
            originals_rescaled = scaler.inverse_transform(images.cpu().numpy())
            recons_rescaled = scaler.inverse_transform(reconstructions.cpu().numpy())

            # Convert back to tensors to use your metric functions
            originals_tensor = torch.from_numpy(originals_rescaled).to(config.device)
            recons_tensor = torch.from_numpy(recons_rescaled).to(config.device)
            # --- END RESCALING ---

            all_mse.append(mse_metric(recons_tensor, originals_tensor))
            all_mae.append(mae_metric(recons_tensor, originals_tensor))
            all_r2.append(r2_metric(recons_tensor, originals_tensor))

            if not first_batch_saved:
                # Attempt to reshape if possible, otherwise save flat tensors as images
                try:
                    plot_reconstructions(
                        real_profiles=originals_tensor[:8],
                        recon_profiles=recons_tensor[:8],
                        save_path=config.reconstruction_path,
                        dz=config.new_dz,
                        num_examples=8,
                    )
                    first_batch_saved = True
                except Exception as e:
                    logger.error(f"Failed to generate reconstruction plot: {e}")
                    raise

    avg_mse = sum(all_mse) / len(all_mse) if all_mse else float("nan")
    std_mse = np.std(all_mse) if all_mse else float("nan")
    avg_mae = sum(all_mae) / len(all_mae) if all_mae else float("nan")
    std_mae = np.std(all_mae) if all_mae else float("nan")
    avg_r2 = sum(all_r2) / len(all_r2) if all_r2 else float("nan")
    std_r2 = np.std(all_r2) if all_r2 else float("nan")

    logger.info(f"Average Test MSE: {avg_mse:.6f}")
    logger.info(f"Standard Deviation Test MSE: {std_mse:.6f}")
    logger.info(f"Average Test MAE: {avg_mae:.6f}")
    logger.info(f"Standard Deviation Test MAE: {std_mae:.6f}")
    logger.info(f"Average Test R2: {avg_r2:.6f}")
    logger.info(f"Standard Deviation Test R2: {std_r2:.6f}")

    # Log to wandb if available
    if wandb is not None:
        try:
            wandb.log({"test/mse": avg_mse, "test/mae": avg_mae, "test/r2": avg_r2})
        except Exception:
            pass
