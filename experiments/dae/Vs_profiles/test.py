import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
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

    # --- Latent Representation Quality Metrics ---
    lipschitz_metrics = calculate_lipschitz_complexity(
        model, test_loader, config.device
    )
    logger.info(
        f"Encoder Lipschitz Complexity: {lipschitz_metrics['encoder_complexity']:.4f}"
    )
    logger.info(
        f"Decoder Lipschitz Complexity: {lipschitz_metrics['decoder_complexity']:.4f}"
    )

    analyze_latent_space(model, test_loader, config.device, config.results_path)

    # Log to wandb if available
    if wandb is not None:
        try:
            wandb.log(
                {
                    "test/mse": avg_mse,
                    "test/mae": avg_mae,
                    "test/r2": avg_r2,
                    "test/lipschitz_encoder": lipschitz_metrics["encoder_complexity"],
                    "test/lipschitz_decoder": lipschitz_metrics["decoder_complexity"],
                }
            )
            wandb.log(
                {
                    "test/reconstructions": wandb.Image(config.reconstruction_path),
                    "test/latent_cosine_similarity": wandb.Image(
                        os.path.join(
                            config.results_path, "latent_cosine_similarity.png"
                        )
                    ),
                    "test/latent_pca_scree_plot": wandb.Image(
                        os.path.join(config.results_path, "latent_pca_scree_plot.png")
                    ),
                }
            )
        except Exception:
            pass


def calculate_lipschitz_complexity(
    model: DecoupledAutoencoder,
    dataloader: torch.utils.data.DataLoader,
    device: str,
):
    """
    Calculate the Lipschitz complexity for the encoder and decoder.
    Based on the methodology from the paper (Appendix B.2).

    Args:
        model (DecoupledAutoencoder): The trained autoencoder model.
        dataloader (DataLoader): DataLoader for the dataset to evaluate.
        device (torch.device): The device to perform computations on.

    Returns:
        dict: A dictionary with 'encoder_complexity' and 'decoder_complexity'.
    """
    logger.info("Calculating Lipschitz Complexity...")
    model.eval()
    encoder_complexities = []
    decoder_complexities = []

    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Lipschitz Complexity"):
            # We need pairs of data, so we split each batch in half
            batch_size = images.shape[0]
            if batch_size < 2:
                continue

            n_pairs = batch_size // 2
            x1 = images[:n_pairs].to(device)
            x2 = images[n_pairs : 2 * n_pairs].to(device)

            # Get latent vectors
            z1 = model.encoder(x1)
            z2 = model.encoder(x2)

            # Get reconstructions
            recon1 = model.decoder(z1)
            recon2 = model.decoder(z2)

            # Calculate L2 norms of the differences
            # Add a small epsilon for numerical stability
            dist_x = torch.linalg.norm(
                x1.view(n_pairs, -1) - x2.view(n_pairs, -1), dim=1
            )
            dist_z = torch.linalg.norm(z1 - z2, dim=1)
            dist_recon = torch.linalg.norm(
                recon1.view(n_pairs, -1) - recon2.view(n_pairs, -1), dim=1
            )

            # Calculate complexity ratios for the batch
            #
            encoder_ratio = dist_z / (dist_x + 1e-8)
            decoder_ratio = dist_recon / (dist_z + 1e-8)

            encoder_complexities.append(torch.mean(encoder_ratio).item())
            decoder_complexities.append(torch.mean(decoder_ratio).item())

    # Return the average complexity over all batches
    avg_encoder_complexity = (
        np.mean(encoder_complexities) if encoder_complexities else float("nan")
    )
    avg_decoder_complexity = (
        np.mean(decoder_complexities) if decoder_complexities else float("nan")
    )

    return {
        "encoder_complexity": avg_encoder_complexity,
        "decoder_complexity": avg_decoder_complexity,
    }


def analyze_latent_space(model, dataloader, device, results_path):
    """
    Performs Cosine Similarity and PCA analysis on the latent space.
    Adapted from the paper's VQGAN codebook analysis (Appendix B.3).
    """
    logger.info("Analyzing Latent Space Distribution...")
    model.eval()
    all_latents = []

    # 1. Collect all latent vectors from the dataloader
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Collecting Latents"):
            images = images.to(device)
            latents = model.encoder(images)
            all_latents.append(latents.cpu())
    latents_tensor = torch.cat(all_latents, dim=0)

    # --- 2. Cosine Similarity Analysis ---
    #
    logger.info("... Computing cosine similarity.")
    # Subsample for efficiency if the dataset is large
    num_samples = min(2000, latents_tensor.shape[0])
    if num_samples < 2:
        logger.warning("Not enough samples for cosine similarity analysis.")
        return
    else:
        indices = torch.randperm(latents_tensor.shape[0])[:num_samples]
        sampled_latents = latents_tensor[indices]

        # Normalize vectors to compute cosine similarity via dot product
        normalized_latents = sampled_latents / (
            torch.linalg.norm(sampled_latents, dim=1, keepdim=True) + 1e-8
        )
        similarity_matrix = torch.matmul(normalized_latents, normalized_latents.T)

        # Correct advanced indexing: get row and col index arrays, then index separately
        triu_idx = torch.triu_indices(num_samples, num_samples, offset=1)
        pairwise_similarities = (
            similarity_matrix[triu_idx[0], triu_idx[1]].cpu().numpy()
        )

        # Ensure we have a 1-D array for seaborn/pandas
        pairwise_similarities = pairwise_similarities.ravel()

        plt.figure(figsize=(10, 6))
        sns.histplot(pairwise_similarities, bins=50, kde=True)
        plt.title("Distribution of Latent Vector Cosine Similarities")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Frequency")
        sim_save_path = os.path.join(results_path, "latent_cosine_similarity.png")
        plt.savefig(sim_save_path)
        plt.close()
        logger.info(f"Cosine similarity plot saved to '{sim_save_path}'")

    # --- 3. PCA for Dimensional Collapse Analysis ---
    #
    logger.info("... computing PCA for dimensional collapse.")
    pca = PCA()
    pca.fit(latents_tensor.numpy())

    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker=".", linestyle="--")
    plt.title("Cumulative Explained Variance of Latent Space (PCA)")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid(True)
    # Highlight the point where 95% variance is reached
    try:
        n_components_95 = np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.95)[0][
            0
        ]
        plt.axhline(y=0.95, color="r", linestyle=":", label="95% Variance")
        plt.axvline(
            x=n_components_95,
            color="g",
            linestyle=":",
            label=f"{n_components_95} Components",
        )
        plt.legend()
    except IndexError:
        pass  # Not enough variance explained

    pca_save_path = os.path.join(results_path, "latent_pca_scree_plot.png")
    plt.savefig(pca_save_path)
    plt.close()
    logger.info(f"PCA scree plot saved to '{pca_save_path}'")
