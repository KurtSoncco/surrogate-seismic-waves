# evaluate_specboost.py
"""Evaluates the trained SpecBoost model ensemble with fusion towers."""

import config
import numpy as np
import torch
from specboost import SpecBoostModelB, TowerEncoder
from tqdm import tqdm
from train_specboost import NRMSELoss

from wave_surrogate.logging_setup import setup_logging
from wave_surrogate.models.fno.model import (
    EncoderOperatorModel,
    OperatorDecoder,
)
from wave_surrogate.models.fno.utils import (
    plot_correlation,
    plot_pearson_histogram,
    plot_predictions,
)

logger = setup_logging()


def evaluate_specboost_ensemble(test_loader, freq_data):
    """
    Evaluates the SpecBoost ensemble (Model A + Model B with fusion towers).

    Architecture:
    1. Model A: Encoder-Decoder → ŷ (prediction)
    2. Model B: Dual-tower fusion → r̂ (residual prediction)
       - Tower 1: Encodes Vs profile (29,)
       - Tower 2: Encodes Model A prediction (1000,)
       - Fusion: Combines latents → residual (1000,)
    3. Final prediction: ŷ + r̂
    """

    # --- Load Model A ---
    logger.info("Loading Model A...")
    encoder_a = TowerEncoder(
        channels=config.ENCODER_CHANNELS_A,
        latent_dim=config.LATENT_DIM,
        kernel_size=config.ENCODER_KERNEL_SIZE,
        pool_size=config.ENCODER_POOL_SIZE,
        use_adaptive_pool=False,
        dropout=config.DROPOUT_RATE,
    ).to(config.DEVICE)

    decoder_a = OperatorDecoder(
        latent_dim=config.LATENT_DIM,
        output_size=config.OUTPUT_SIZE,
        fno_modes=config.FNO_MODES,
        fno_width=config.FNO_WIDTH,
    ).to(config.DEVICE)
    model_a = EncoderOperatorModel(encoder=encoder_a, decoder=decoder_a).to(
        config.DEVICE
    )
    model_a.load_state_dict(
        torch.load(config.MODEL_A_SAVE_PATH, map_location=config.DEVICE)
    )
    model_a.eval()
    logger.info("✓ Model A loaded successfully")

    # --- Load Model B (Fusion Towers) ---
    logger.info("Loading Model B (Fusion Towers)...")
    vs_channels = config.ENCODER_CHANNELS_A  # Match Model A's encoder channels
    pred_channels = getattr(config, "PRED_ENCODER_CHANNELS", [1, 64, 128, 256])
    latent_dim = config.LATENT_DIM  # Match Model A's latent dimension
    fusion_hidden = getattr(config, "FUSION_HIDDEN_DIMS", [256, 512])
    dropout = getattr(config, "FUSION_DROPOUT", 0.2)

    model_b = SpecBoostModelB(
        vs_encoder_channels=vs_channels,
        pred_encoder_channels=pred_channels,
        latent_dim=latent_dim,
        output_size=config.OUTPUT_SIZE,
        fusion_hidden_dims=fusion_hidden,
        dropout=dropout,
    ).to(config.DEVICE)

    model_b.load_state_dict(
        torch.load(config.MODEL_B_SAVE_PATH, map_location=config.DEVICE)
    )
    model_b.eval()
    logger.info("✓ Model B loaded successfully")

    # --- Evaluate Ensemble ---
    criterion = NRMSELoss()

    # Track metrics for both solo and ensemble
    test_loss_solo = 0.0  # Model A alone
    test_loss_ensemble = 0.0  # Model A + Model B

    all_predictions_solo = []
    all_predictions_ensemble = []
    all_residuals_pred = []
    all_targets = []
    all_inputs = []

    logger.info("Starting evaluation...")
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating SpecBoost"):
            inputs = inputs.to(config.DEVICE)
            targets = targets.to(config.DEVICE)

            # 1. Get prediction from Model A
            preds_a = model_a(inputs.unsqueeze(1))

            # 2. Get residual prediction from Model B (fusion towers)
            # Model B takes: (vs_profile, model_a_pred)
            residual_pred = model_b(inputs, preds_a)

            # 3. Combine for final prediction
            final_outputs = preds_a + residual_pred

            # Calculate losses
            loss_solo = criterion(preds_a, targets)
            loss_ensemble = criterion(final_outputs, targets)

            test_loss_solo += loss_solo.item() * inputs.size(0)
            test_loss_ensemble += loss_ensemble.item() * inputs.size(0)

            # Store for visualization
            all_predictions_solo.append(preds_a.cpu().numpy())
            all_predictions_ensemble.append(final_outputs.cpu().numpy())
            all_residuals_pred.append(residual_pred.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_inputs.append(inputs.cpu().numpy())

    # Calculate average losses
    test_loss_solo /= len(test_loader.dataset)
    test_loss_ensemble /= len(test_loader.dataset)

    # Calculate improvement
    improvement = ((test_loss_solo - test_loss_ensemble) / test_loss_solo) * 100

    # Log results
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"Model A (Solo) Test NMSE:      {test_loss_solo:.6f}")
    logger.info(f"SpecBoost Ensemble Test NMSE:  {test_loss_ensemble:.6f}")
    logger.info(
        f"Absolute Improvement:         {test_loss_solo - test_loss_ensemble:.6f}"
    )
    logger.info(f"Relative Improvement:         {improvement:.2f}%")
    logger.info("=" * 70 + "\n")

    # Convert to numpy arrays
    all_predictions_solo = np.vstack(all_predictions_solo)
    all_predictions_ensemble = np.vstack(all_predictions_ensemble)
    all_residuals_pred = np.vstack(all_residuals_pred)
    all_targets = np.vstack(all_targets)
    all_inputs = np.vstack(all_inputs)

    # Calculate residual statistics
    actual_residuals = all_targets - all_predictions_solo
    logger.info("Residual Statistics:")
    logger.info(
        f"  Actual residuals:    mean={actual_residuals.mean():.6f}, std={actual_residuals.std():.6f}"
    )
    logger.info(
        f"  Predicted residuals: mean={all_residuals_pred.mean():.6f}, std={all_residuals_pred.std():.6f}"
    )
    # Add this to your evaluation
    residual_correlation = np.corrcoef(
        actual_residuals.flatten(), all_residuals_pred.flatten()
    )[0, 1]
    logger.info(f"Residual prediction correlation: {residual_correlation:.4f}")

    # --- Generate Comparison Plots ---
    logger.info("\nGenerating comparison plots...")

    # Plot for Model A (Solo)
    logger.info("  → Plotting Model A (Solo) results...")
    plot_predictions(
        freq_data,
        all_targets,
        all_predictions_solo,
        all_inputs,
        title_prefix="Model_A_Solo",
    )
    plot_correlation(all_targets, all_predictions_solo, title_prefix="Model A (Solo)")
    plot_pearson_histogram(
        all_targets,
        all_predictions_solo,
        title_prefix="Model A (Solo)",
    )

    # Plot for SpecBoost Ensemble
    logger.info("  → Plotting SpecBoost Ensemble results...")
    plot_predictions(
        freq_data,
        all_targets,
        all_predictions_ensemble,
        all_inputs,
        title_prefix="SpecBoost_Ensemble",
    )
    plot_correlation(
        all_targets, all_predictions_ensemble, title_prefix="SpecBoost Ensemble"
    )
    plot_pearson_histogram(
        all_targets,
        all_predictions_ensemble,
        title_prefix="SpecBoost Ensemble",
    )

    logger.info("✓ Evaluation complete!\n")

    return {
        "solo_mse": test_loss_solo,
        "ensemble_mse": test_loss_ensemble,
        "improvement_percent": improvement,
        "predictions_solo": all_predictions_solo,
        "predictions_ensemble": all_predictions_ensemble,
        "residuals_predicted": all_residuals_pred,
        "targets": all_targets,
        "inputs": all_inputs,
    }


def evaluate_solo_model(test_loader, freq_data):
    """Evaluate Model A alone for baseline comparison."""
    logger.info("Evaluating Model A (Solo) for baseline comparison...")

    encoder_a = TowerEncoder(
        channels=config.ENCODER_CHANNELS_A,
        latent_dim=config.LATENT_DIM,
        kernel_size=config.ENCODER_KERNEL_SIZE,
        pool_size=config.ENCODER_POOL_SIZE,
        use_adaptive_pool=False,
        dropout=config.DROPOUT_RATE,
    ).to(config.DEVICE)
    decoder_a = OperatorDecoder(
        latent_dim=config.LATENT_DIM,
        output_size=config.OUTPUT_SIZE,
        fno_modes=config.FNO_MODES,
        fno_width=config.FNO_WIDTH,
    ).to(config.DEVICE)
    model_a = EncoderOperatorModel(encoder=encoder_a, decoder=decoder_a).to(
        config.DEVICE
    )
    model_a.load_state_dict(
        torch.load(config.MODEL_A_SAVE_PATH, map_location=config.DEVICE)
    )
    model_a.eval()

    criterion = NRMSELoss()
    test_loss = 0.0
    all_predictions, all_targets, all_inputs = [], [], []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating Solo Model"):
            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
            outputs = model_a(inputs.unsqueeze(1))
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_inputs.append(inputs.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    logger.info(f"Model A (Solo) Test MSE: {test_loss:.6f}")

    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    all_inputs = np.vstack(all_inputs)

    plot_predictions(freq_data, all_targets, all_predictions, all_inputs)
    plot_correlation(all_targets, all_predictions)
    plot_pearson_histogram(all_targets, all_predictions)

    return test_loss


def evaluate_multi_stage_ensemble(
    test_loader, freq_data, models, normalization_params=None
):
    """Evaluate ensemble with T stages, properly denormalizing residuals.

    Args:
        test_loader: DataLoader for the test set.
        freq_data: Frequency data for plotting.
        models: List of trained models [Model A, Model B, ...].
        normalization_params: List of (mean, std) tuples for each boosting stage. Default is None.

        Note: Model A does not require normalization parameters.
    """

    criterion = NRMSELoss()
    all_targets, all_inputs = [], []

    # Track predictions at each stage
    stage_predictions = {f"Stage_{i}": [] for i in range(len(models))}

    # Track predicted residuals at each boosting stage
    stage_residuals = {f"Stage_{i}": [] for i in range(1, len(models))}

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating Multi-Stage"):
            inputs = inputs.to(config.DEVICE)
            targets = targets.to(config.DEVICE)

            # Stage 0: Initial prediction from Model A
            cumulative_pred = models[0](inputs.unsqueeze(1))
            stage_predictions["Stage_0"].append(cumulative_pred.cpu().numpy())

            # Stages 1+: Add residual predictions from boosting models
            for stage_idx, model in enumerate(models[1:], start=1):
                # Model predicts NORMALIZED residual
                normalized_residual_pred = model(cumulative_pred)

                # Denormalize if we have normalization params
                if normalization_params is not None and stage_idx <= len(
                    normalization_params
                ):
                    residual_mean, residual_std = normalization_params[stage_idx - 1]
                    residual_pred = (
                        normalized_residual_pred * residual_std + residual_mean
                    )
                else:
                    # No normalization (fallback for old models)
                    residual_pred = normalized_residual_pred

                # Store residual for analysis
                stage_residuals[f"Stage_{stage_idx}"].append(
                    residual_pred.cpu().numpy()
                )

                # Update cumulative prediction
                cumulative_pred = cumulative_pred + config.BOOSTING_ETA * residual_pred
                stage_predictions[f"Stage_{stage_idx}"].append(
                    cumulative_pred.cpu().numpy()
                )

            all_targets.append(targets.cpu().numpy())
            all_inputs.append(inputs.cpu().numpy())

    # Convert to arrays
    all_targets = np.vstack(all_targets)
    all_inputs = np.vstack(all_inputs)

    # Calculate and log performance at each stage
    logger.info("\n" + "=" * 70)
    logger.info("MULTI-STAGE EVALUATION RESULTS")
    logger.info("=" * 70)

    stage_losses = {}
    for stage_name, preds_list in stage_predictions.items():
        preds = np.vstack(preds_list)
        preds_tensor = torch.tensor(preds, dtype=torch.float32)
        targets_tensor = torch.tensor(all_targets, dtype=torch.float32)

        loss = criterion(preds_tensor, targets_tensor).item()
        stage_losses[stage_name] = loss

        stage_num = int(stage_name.split("_")[1])
        model_letter = chr(65 + stage_num) if stage_num > 0 else "A"
        logger.info(
            f"After Model {model_letter} (Stage {stage_num + 1}): NMSE = {loss:.6f}"
        )

    # Calculate improvements
    improvement = 0.0
    if len(stage_losses) > 1:
        baseline_loss = stage_losses["Stage_0"]
        final_loss = stage_losses[f"Stage_{len(models) - 1}"]
        improvement = ((baseline_loss - final_loss) / baseline_loss) * 100

        logger.info("-" * 70)
        logger.info(
            f"Overall Improvement: {improvement:.2f}% ({baseline_loss:.6f} → {final_loss:.6f})"
        )
        logger.info("=" * 70)

    # Analyze predicted residuals
    if len(stage_residuals) > 0:
        logger.info("\nPREDICTED RESIDUAL STATISTICS")
        logger.info("=" * 70)

        # Calculate actual residuals for comparison
        baseline_preds = np.vstack(stage_predictions["Stage_0"])
        actual_residuals_stage1 = all_targets - baseline_preds

        logger.info("Actual residuals (after Model A):")
        logger.info(
            f"  Mean: {actual_residuals_stage1.mean():.6f}, Std: {actual_residuals_stage1.std():.6f}"
        )

        for stage_name, residual_list in stage_residuals.items():
            residuals = np.vstack(residual_list)
            stage_num = int(stage_name.split("_")[1])
            model_letter = chr(65 + stage_num)

            logger.info(f"\nModel {model_letter} predicted residuals:")
            logger.info(f"  Mean: {residuals.mean():.6f}, Std: {residuals.std():.6f}")

            # Calculate correlation with actual residuals
            if stage_num == 1:
                # For first stage, compare with actual residuals after Model A
                corr = np.corrcoef(
                    actual_residuals_stage1.flatten(), residuals.flatten()
                )[0, 1]
                logger.info(f"  Correlation with actual residuals: {corr:.4f}")

        logger.info("=" * 70)

    # Generate plots for final stage
    logger.info("\nGenerating visualization plots...")
    final_preds = np.vstack(stage_predictions[f"Stage_{len(models) - 1}"])

    plot_predictions(
        freq_data,
        all_targets,
        final_preds,
        all_inputs,
        title_prefix=f"SpecBoost_T{len(models) - 1}",
    )
    plot_correlation(
        all_targets, final_preds, title_prefix=f"SpecBoost T={len(models) - 1}"
    )
    plot_pearson_histogram(
        all_targets, final_preds, title_prefix=f"SpecBoost T={len(models) - 1}"
    )

    # Also plot baseline for comparison
    baseline_preds = np.vstack(stage_predictions["Stage_0"])
    plot_predictions(
        freq_data,
        all_targets,
        baseline_preds,
        all_inputs,
        title_prefix="Model_A_Baseline",
    )
    plot_correlation(all_targets, baseline_preds, title_prefix="Model A Baseline")

    logger.info("✓ Evaluation complete!\n")

    return {
        "stage_losses": stage_losses,
        "stage_predictions": {k: np.vstack(v) for k, v in stage_predictions.items()},
        "stage_residuals": {k: np.vstack(v) for k, v in stage_residuals.items()},
        "targets": all_targets,
        "inputs": all_inputs,
        "improvement_percent": improvement if len(stage_losses) > 1 else 0.0,
    }
