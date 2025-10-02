# main.py
"""Main script to run the SpecBoost training and evaluation pipeline."""

import pickle
from pathlib import Path
from typing import List

import config
import numpy as np
import torch
from data_loader import get_primary_loaders
from evaluate_specboost import (
    evaluate_multi_stage_ensemble,
    evaluate_specboost_ensemble,
)
from specboost import SpecBoostModelB, TowerEncoder
from train_specboost import (
    generate_denormalized_predictions,
    generate_predictions_for_split,
    train_boosting_stage_normalized,
    train_stage_one,
    train_stage_two,
)

from wave_surrogate.logging_setup import setup_logging
from wave_surrogate.models.fno.model import (
    EncoderOperatorModel,
    OperatorDecoder,
)
from wave_surrogate.models.fno.utils import f0_calc

logger = setup_logging()


def run_iterative_boosting_pipeline():
    """
    Executes an iterative boosting pipeline after training Model A.

    Pipeline:
    1. Load and preprocess data
    2. Train Model A (standard encoder-decoder)
    3. For each boosting stage:
       a. Generate predictions from the current ensemble
       b. Compute residuals
       c. Train a new Model B on the residuals
       d. Update the ensemble to include the new Model B
    4. Evaluate final ensemble on test set
    """

    # --- Load and Preprocess Data ---
    logger.info("=" * 70)
    logger.info("SPECBOOST PIPELINE - Loading and Preprocessing Data")
    logger.info("=" * 70)

    with open(config.TTF_PICKLE_PATH, "rb") as f:
        ttf_data = np.array(pickle.load(f))
    with open(config.VS_PICKLE_PATH, "rb") as f:
        vs_profiles = np.array(pickle.load(f), dtype=object)
    freq_data = np.genfromtxt(config.FREQ_PATH, delimiter=",")

    logger.info(
        f"Loaded {len(vs_profiles)} profiles and {len(ttf_data)} transfer functions"
    )

    # Filter based on f0 threshold
    f0_values = np.array([f0_calc(profile) for profile in vs_profiles])
    keep_indices = np.where(f0_values < config.F0_FILTER_THRESHOLD)[0]
    vs_profiles_filtered = vs_profiles[keep_indices]
    ttf_data_filtered = ttf_data[keep_indices]

    vs_list = [arr for arr in vs_profiles_filtered]
    ttf_list = [arr for arr in ttf_data_filtered]

    logger.info(
        f"After f0 filtering (< {config.F0_FILTER_THRESHOLD}): {len(vs_list)} samples"
    )
    logger.info(
        f"Input shape: (batch, {config.INPUT_SIZE}), Output shape: (batch, {config.OUTPUT_SIZE})"
    )

    # Ensure save directory exists
    Path(config.MODEL_SAVE_PATH).mkdir(parents=True, exist_ok=True)
    logger.info(f"Models will be saved to: {config.MODEL_SAVE_PATH}\n")

    # --- Get Data Splits ---
    logger.info("Creating data splits...")
    train_loader_A, val_loader_A, test_loader, train_dataset, val_dataset = (
        get_primary_loaders(vs_list, ttf_list, config.BATCH_SIZE)
    )

    train_model_A = False  # Set to True to train Model A from scratch

    if not train_model_A:
        logger.info(
            "Skipping training of Model A and just loading pre-trained weights."
        )
        logger.info(f"Loading Model A from: {config.MODEL_A_SAVE_PATH}")

        # encoder_a = Encoder(
        #    channels=config.ENCODER_CHANNELS_A, latent_dim=config.LATENT_DIM
        # ).to(config.DEVICE)

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

    else:
        logger.info(f"Training set size: {len(train_dataset)} samples")
        logger.info(f"Validation set size: {len(val_dataset)} samples")
        # --- Stage 1: Train Model A ---
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 1: Training Model A (Standard Encoder-Decoder)")
        logger.info("=" * 70)
        logger.info(
            f"Architecture: Encoder ({config.ENCODER_CHANNELS_A}) → Latent ({config.LATENT_DIM}) → FNO Decoder"
        )
        logger.info(
            f"Training for up to {config.NUM_EPOCHS} epochs with early stopping (patience={config.EARLY_STOP_PATIENCE})\n"
        )

        model_a = train_stage_one(train_loader_A, val_loader_A)

    models: List[torch.nn.Module] = []
    models = [model_a]  # List to hold all models in the ensemble

    # Get initial predictions from Model A
    train_preds_cumulative = generate_predictions_for_split(model_a, train_dataset)
    val_preds_cumulative = generate_predictions_for_split(model_a, val_dataset)

    # Calculate and log initial residual statistics
    train_targets = np.array(
        [train_dataset[i][1].numpy() for i in range(len(train_dataset))]  # type: ignore
    )
    train_residuals = train_targets - train_preds_cumulative

    logger.info("\n" + "-" * 70)
    logger.info("INITIAL RESIDUAL STATISTICS (after Model A)")
    logger.info("-" * 70)
    logger.info(f"Mean: {train_residuals.mean():.6f}")
    logger.info(f"Std:  {train_residuals.std():.6f}")
    logger.info(f"Min:  {train_residuals.min():.6f}")
    logger.info(f"Max:  {train_residuals.max():.6f}")
    logger.info("-" * 70 + "\n")

    # Start normalization parameters for each stage
    normalization_params = []

    # Iterative boosting stages
    for stage in range(config.NUM_BOOSTING_STAGES):
        logger.info("\n" + "=" * 70)
        logger.info(f"BOOSTING STAGE {stage + 2}: Training Model {chr(66 + stage)}")
        logger.info("=" * 70)

        # Model configuration (could vary per stage if desired)
        model_config = {
            "pred_encoder_channels": config.PRED_ENCODER_CHANNELS,
            "latent_dim": config.LATENT_DIM,
            "output_size": config.OUTPUT_SIZE,
            "fno_modes": config.FNO_MODES,
            "fno_width": config.FNO_WIDTH,
            "dropout": config.DROPOUT_RATE,
        }

        # Train this stage's model
        stage_model, residual_mean, residual_std = train_boosting_stage_normalized(
            stage_num=stage + 1,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            previous_stage_preds_train=train_preds_cumulative,
            previous_stage_preds_val=val_preds_cumulative,
            model_config=model_config,
        )

        models.append(stage_model)
        normalization_params.append((residual_mean, residual_std))

        # Generate DENORMALIZED predictions from this stage
        logger.info("\nGenerating denormalized predictions...")
        train_residual_preds = generate_denormalized_predictions(
            stage_model,
            train_dataset,
            train_preds_cumulative,
            residual_mean,
            residual_std,
        )
        val_residual_preds = generate_denormalized_predictions(
            stage_model, val_dataset, val_preds_cumulative, residual_mean, residual_std
        )

        # Log statistics of predicted residuals
        logger.info("\nPredicted residual statistics:")
        logger.info(f"  Mean: {train_residual_preds.mean():.6f}")
        logger.info(f"  Std:  {train_residual_preds.std():.6f}")

        # Update cumulative predictions
        train_preds_cumulative = (
            train_preds_cumulative + config.BOOSTING_ETA * train_residual_preds
        )
        val_preds_cumulative = (
            val_preds_cumulative + config.BOOSTING_ETA * val_residual_preds
        )

        # Calculate remaining residuals
        train_residuals = train_targets - train_preds_cumulative

        logger.info("\n" + "-" * 70)
        logger.info(f"REMAINING RESIDUALS (after Model {chr(66 + stage)})")
        logger.info("-" * 70)
        logger.info(f"Mean: {train_residuals.mean():.6f}")
        logger.info(f"Std:  {train_residuals.std():.6f}")
        logger.info(
            f"Cumulative train RMSE: {np.sqrt(np.mean(train_residuals**2)):.6f}"
        )
        logger.info("-" * 70 + "\n")

        # Check for improvement
        if stage > 0:
            prev_std = normalization_params[stage - 1][1]
            improvement = ((prev_std - train_residuals.std()) / prev_std) * 100
            logger.info(f"Residual std reduction: {improvement:.2f}%")

            if improvement < 1.0:
                logger.warning(
                    f"Warning: Minimal improvement ({improvement:.2f}%). "
                    "Model may not be learning effectively."
                )

    # Save normalization parameters
    norm_params_path = config.MODEL_SAVE_PATH / "normalization_params.pt"
    torch.save(normalization_params, norm_params_path)
    logger.info(f"\nSaved normalization parameters to: {norm_params_path}")

    # Evaluation with all T stages
    logger.info("\n" + "=" * 70)
    logger.info(
        "STAGE "
        + str(config.NUM_BOOSTING_STAGES + 2)
        + ": Evaluating Multi-Stage Ensemble on Test Set"
    )
    logger.info("=" * 70)

    evaluate_multi_stage_ensemble(test_loader, freq_data, models, normalization_params)


def run_specboost_pipeline():
    """
    Executes the full SpecBoost pipeline with fusion towers.

    Pipeline:
    1. Load and preprocess data
    2. Train Model A (standard encoder-decoder)
    3. Generate predictions from Model A
    4. Train Model B (dual-tower fusion) on residuals
    5. Evaluate ensemble on test set
    """

    # --- Load and Preprocess Data ---
    logger.info("=" * 70)
    logger.info("SPECBOOST PIPELINE - Loading and Preprocessing Data")
    logger.info("=" * 70)

    with open(config.TTF_PICKLE_PATH, "rb") as f:
        ttf_data = np.array(pickle.load(f))
    with open(config.VS_PICKLE_PATH, "rb") as f:
        vs_profiles = np.array(pickle.load(f), dtype=object)
    freq_data = np.genfromtxt(config.FREQ_PATH, delimiter=",")

    logger.info(
        f"Loaded {len(vs_profiles)} profiles and {len(ttf_data)} transfer functions"
    )

    # Filter based on f0 threshold
    f0_values = np.array([f0_calc(profile) for profile in vs_profiles])
    keep_indices = np.where(f0_values < config.F0_FILTER_THRESHOLD)[0]
    vs_profiles_filtered = vs_profiles[keep_indices]
    ttf_data_filtered = ttf_data[keep_indices]

    vs_list = [arr for arr in vs_profiles_filtered]
    ttf_list = [arr for arr in ttf_data_filtered]

    logger.info(
        f"After f0 filtering (< {config.F0_FILTER_THRESHOLD}): {len(vs_list)} samples"
    )
    logger.info(
        f"Input shape: (batch, {config.INPUT_SIZE}), Output shape: (batch, {config.OUTPUT_SIZE})"
    )

    # Ensure save directory exists
    Path(config.MODEL_SAVE_PATH).mkdir(parents=True, exist_ok=True)
    logger.info(f"Models will be saved to: {config.MODEL_SAVE_PATH}\n")

    # --- Get Data Splits ---
    logger.info("Creating data splits...")
    train_loader_A, val_loader_A, test_loader, train_dataset, val_dataset = (
        get_primary_loaders(vs_list, ttf_list, config.BATCH_SIZE)
    )

    train_model_A = True  # Set to True to train Model A from scratch

    if not train_model_A:
        logger.info(
            "Skipping training of Model A and just loading pre-trained weights."
        )
        logger.info(f"Loading Model A from: {config.MODEL_A_SAVE_PATH}")

        # encoder_a = Encoder(
        #    channels=config.ENCODER_CHANNELS_A, latent_dim=config.LATENT_DIM
        # ).to(config.DEVICE)

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

    else:
        logger.info(f"Training set size: {len(train_dataset)} samples")
        logger.info(f"Validation set size: {len(val_dataset)} samples")
        # --- Stage 1: Train Model A ---
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 1: Training Model A (Standard Encoder-Decoder)")
        logger.info("=" * 70)
        logger.info(
            f"Architecture: Encoder ({config.ENCODER_CHANNELS_A}) → Latent ({config.LATENT_DIM}) → FNO Decoder"
        )
        logger.info(
            f"Training for up to {config.NUM_EPOCHS} epochs with early stopping (patience={config.EARLY_STOP_PATIENCE})\n"
        )

        model_a = train_stage_one(train_loader_A, val_loader_A)

    # --- Stage 2: Generate Predictions and Train Model B ---
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 2: Generating Predictions and Training Model B (Fusion Towers)")
    logger.info("=" * 70)

    # Generate predictions from Model A for both train and val sets
    logger.info("Generating Model A predictions for training set...")
    train_preds_a = generate_predictions_for_split(model_a, train_dataset)

    logger.info("Generating Model A predictions for validation set...")
    val_preds_a = generate_predictions_for_split(model_a, val_dataset)

    logger.info(f"Train predictions shape: {train_preds_a.shape}")
    logger.info(f"Val predictions shape: {val_preds_a.shape}")

    # Calculate residual statistics
    train_targets = np.array(
        [train_dataset[i][1].numpy() for i in range(len(train_dataset))]  # type: ignore
    )
    train_residuals = train_targets - train_preds_a
    logger.info(
        f"Train residuals: mean={train_residuals.mean():.6f}, std={train_residuals.std():.6f}\n"
    )

    # Train Model B on residuals with fusion towers
    logger.info("Training Model B with dual-tower fusion architecture:")
    logger.info(
        f"  Tower 1: Vs profile ({config.INPUT_SIZE},) → latent ({config.FUSION_LATENT_DIM})"
    )
    logger.info(
        f"  Tower 2: Model A pred ({config.OUTPUT_SIZE},) → latent ({config.FUSION_LATENT_DIM})"
    )
    logger.info(
        f"  Fusion:  Combined latent ({config.FUSION_LATENT_DIM * 2}) → residual ({config.OUTPUT_SIZE})\n"
    )

    # Initialize model B
    vs_channels = config.ENCODER_CHANNELS_A  # Match Model A's encoder channels
    pred_channels = getattr(config, "PRED_ENCODER_CHANNELS", [1, 64, 128, 256])
    latent_dim = config.LATENT_DIM  # Match Model A's latent dimension
    fusion_hidden = getattr(config, "FUSION_HIDDEN_DIMS", [256, 512])
    dropout = getattr(config, "FUSION_DROPOUT", 0.2)

    # Instantiate Model B
    model_b = SpecBoostModelB(
        vs_encoder_channels=vs_channels,
        pred_encoder_channels=pred_channels,
        latent_dim=latent_dim,
        output_size=config.OUTPUT_SIZE,
        fusion_hidden_dims=fusion_hidden,
        dropout=dropout,
    ).to(config.DEVICE)

    # Dummy forward pass to initialize weights
    dummy_vs = torch.randn(2, config.INPUT_SIZE).to(config.DEVICE)
    dummy_pred = torch.randn(2, config.OUTPUT_SIZE).to(config.DEVICE)
    _ = model_b(dummy_vs, dummy_pred)

    # logger.info("Transferring Model A Encoder weights to Model B's Vs tower...")
    encoder_a_state_dict = model_a.encoder.state_dict()
    model_b.vs_tower.load_state_dict(encoder_a_state_dict)

    model_b = train_stage_two(
        train_dataset, val_dataset, train_preds_a, val_preds_a, model_b=model_b
    )

    # --- Evaluation ---
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 3: Evaluating SpecBoost Ensemble on Test Set")
    logger.info("=" * 70)
    logger.info(
        "Ensemble prediction: final_output = Model_A(x) + Model_B(x, Model_A(x))\n"
    )

    results = evaluate_specboost_ensemble(test_loader, freq_data)

    # --- Final Summary ---
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    logger.info("Model Locations:")
    logger.info(f"  Model A: {config.MODEL_A_SAVE_PATH}")
    logger.info(f"  Model B: {config.MODEL_B_SAVE_PATH}")
    logger.info("\nPerformance Summary:")
    logger.info(f"  Solo Model A MSE:      {results['solo_mse']:.6f}")
    logger.info(f"  Ensemble MSE:          {results['ensemble_mse']:.6f}")
    logger.info(f"  Relative Improvement:  {results['improvement_percent']:.2f}%")
    logger.info("=" * 70 + "\n")


if __name__ == "__main__":
    run_iterative_boosting_pipeline()
