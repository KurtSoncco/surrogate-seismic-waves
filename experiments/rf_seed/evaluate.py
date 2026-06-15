# evaluate.py
"""Evaluation: metrics, plots, save to disk, and log to wandb."""

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import wandb

import config


def _compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute regression and correlation metrics."""
    mse = float(np.mean((targets - predictions) ** 2))
    mae = float(np.mean(np.abs(targets - predictions)))
    rmse = float(np.sqrt(mse))
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = float(1 - (ss_res / (ss_tot + 1e-12)))

    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    pearson_overall = float(np.corrcoef(pred_flat, target_flat)[0, 1])
    if np.isnan(pearson_overall):
        pearson_overall = 0.0

    sample_corrs = []
    for i in range(len(predictions)):
        c = np.corrcoef(predictions[i], targets[i])[0, 1]
        if np.isfinite(c):
            sample_corrs.append(c)
    sample_corrs = np.array(sample_corrs) if sample_corrs else np.array([0.0])

    return {
        "test_mse": mse,
        "test_mae": mae,
        "test_rmse": rmse,
        "test_r2": r2,
        "test_pearson_overall": pearson_overall,
        "test_pearson_mean": float(np.mean(sample_corrs)),
        "test_pearson_median": float(np.median(sample_corrs)),
        "test_pearson_std": float(np.std(sample_corrs)),
        "test_pearson_min": float(np.min(sample_corrs)),
        "test_pearson_max": float(np.max(sample_corrs)),
    }


def _plot_predictions_grid(
    predictions: np.ndarray,
    targets: np.ndarray,
    freq_data: Optional[np.ndarray],
    save_path: Path,
    run: Optional[Any],
) -> None:
    """2x3 grid: col0 = 2 worst (by Pearson), col1 = 2 median, col2 = 2 best. Saves as predictions_summary.png."""
    n = len(predictions)
    corrs = np.array(
        [
            np.corrcoef(predictions[i], targets[i])[0, 1]
            if np.isfinite(np.corrcoef(predictions[i], targets[i])[0, 1])
            else -np.inf
            for i in range(n)
        ]
    )
    order = np.argsort(corrs)  # worst to best
    worst_two = order[: min(2, n)]
    mid = max(0, n // 2 - 1)
    median_two = order[mid : mid + 2] if n >= 2 else order[:1]
    best_two = order[-min(2, n) :]
    # Column-major: col0 = worst, col1 = median, col2 = best (2 rows each)
    parts = [worst_two, median_two, best_two]
    indices = []
    for part in parts:
        indices.extend(part[:2])
    indices = np.array(indices[:6])

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10))
    x_axis = (
        freq_data
        if freq_data is not None and len(freq_data) == targets.shape[1]
        else np.arange(targets.shape[1])
    )

    n_plots = min(6, len(indices))
    for k in range(n_plots):
        row, col = k % 2, k // 2
        idx = indices[k]
        ax = axes[row, col]
        ax.plot(x_axis, targets[idx], "-", label="Target", alpha=0.7, color="gray")
        ax.plot(x_axis, predictions[idx], "r--", label="Prediction", alpha=0.7)
        corr = corrs[idx] if np.isfinite(corrs[idx]) else 0.0
        title_suffix = (
            " (worst)" if col == 0 else " (median)" if col == 1 else " (best)"
        )
        ax.set_title(f"Sample {idx}, $\\rho$={corr:.3f}{title_suffix}")
        ax.grid(True)
        if freq_data is not None:
            ax.set_xscale("log")
        if k == 0:
            ax.legend()
        if row == 1:
            ax.set_xlabel("Frequency (Hz)" if freq_data is not None else "Index")
        if col == 0:
            ax.set_ylabel("Transfer function")
    for k in range(n_plots, 6):
        row, col = k % 2, k // 2
        axes[row, col].set_visible(False)
    plt.tight_layout()
    out_path = save_path / "predictions_summary.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    if run is not None:
        wandb.log({"eval/predictions_summary": wandb.Image(str(out_path))})


def _plot_correlation_scatter(
    predictions: np.ndarray, targets: np.ndarray, save_path: Path, run: Optional[Any]
) -> None:
    """Scatter: all targets vs all predictions."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(targets.flatten(), predictions.flatten(), alpha=0.1, s=2)
    mn = min(np.min(targets), np.min(predictions))
    mx = max(np.max(targets), np.max(predictions))
    ax.plot([mn, mx], [mn, mx], "r--")
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")
    ax.set_title("Prediction vs Ground Truth Correlation")
    ax.grid(True)
    plt.tight_layout()
    p = save_path / "correlation.png"
    plt.savefig(p, dpi=150)
    plt.close()
    if run is not None:
        wandb.log({"eval/correlation": wandb.Image(str(p))})


def _plot_pearson_histogram(
    predictions: np.ndarray, targets: np.ndarray, save_path: Path, run: Optional[Any]
) -> None:
    """Histogram of per-sample Pearson correlations."""
    corrs = []
    for i in range(len(predictions)):
        c = np.corrcoef(predictions[i], targets[i])[0, 1]
        if np.isfinite(c):
            corrs.append(c)
    corrs = np.array(corrs) if corrs else np.array([0.0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.hist(corrs, bins=30, alpha=0.75, edgecolor="black")
    mean_c = np.mean(corrs)
    median_c = np.median(corrs)
    std_c = np.std(corrs)
    ax1.axvline(mean_c, color="red", linestyle="--", label=f"Mean: {mean_c:.3f}")
    ax1.axvline(median_c, color="green", linestyle=":", label=f"Median: {median_c:.3f}")
    ax1.set_xlabel("Pearson correlation")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"Per-sample Pearson (std={std_c:.3f})")
    ax1.legend()
    ax1.grid(True, axis="y")

    ax2.plot(corrs, alpha=0.7)
    ax2.axhline(mean_c, color="red", linestyle="--", label=f"Mean: {mean_c:.3f}")
    ax2.set_xlabel("Sample index")
    ax2.set_ylabel("Pearson correlation")
    ax2.set_title("Correlation vs sample index")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    p = save_path / "pearson_histogram.png"
    plt.savefig(p, dpi=150)
    plt.close()
    if run is not None:
        wandb.log({"eval/pearson_histogram": wandb.Image(str(p))})


def _plot_frequency_correlation(
    predictions: np.ndarray,
    targets: np.ndarray,
    freq_data: np.ndarray,
    save_path: Path,
    run: Optional[Any],
) -> None:
    """Correlation at each frequency bin."""
    n_freq = predictions.shape[1]
    freq_corrs = []
    for i in range(n_freq):
        c = np.corrcoef(predictions[:, i], targets[:, i])[0, 1]
        freq_corrs.append(c if np.isfinite(c) else 0.0)
    freq_corrs = np.array(freq_corrs)
    if len(freq_data) != n_freq:
        freq_data = np.arange(n_freq)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(freq_data, freq_corrs, "b-", linewidth=2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Correlation coefficient")
    ax.set_title("Correlation vs frequency")
    ax.grid(True)
    if np.all(freq_data > 0):
        ax.set_xscale("log")
    plt.tight_layout()
    p = save_path / "correlation_vs_frequency.png"
    plt.savefig(p, dpi=150)
    plt.close()
    if run is not None:
        wandb.log({"eval/correlation_vs_frequency": wandb.Image(str(p))})


def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    freq_data: Optional[np.ndarray] = None,
    save_dir: Optional[Path] = None,
    run: Optional[Any] = None,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Run model on test set, compute metrics, save plots, log to wandb.

    Args:
        model: Trained model (already in eval mode and on correct device).
        test_loader: Test data loader.
        freq_data: Optional frequency axis for plots (length must match output size).
        save_dir: Directory to save figures (default config.RESULTS_SAVE_DIR).
        run: Optional wandb run (if None, skips wandb logging).
        seed: Unused; kept for API compatibility.

    Returns:
        Dictionary of test metrics.
    """
    save_dir = save_dir or config.RESULTS_SAVE_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    predictions = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    metrics = _compute_metrics(predictions, targets)

    _plot_predictions_grid(predictions, targets, freq_data, save_dir, run)
    _plot_correlation_scatter(predictions, targets, save_dir, run)
    _plot_pearson_histogram(predictions, targets, save_dir, run)
    if freq_data is not None and len(freq_data) == predictions.shape[1]:
        _plot_frequency_correlation(predictions, targets, freq_data, save_dir, run)

    if run is not None:
        wandb.log({k: v for k, v in metrics.items()})
        wandb.summary.update(
            {
                "test/mse": metrics["test_mse"],
                "test/mae": metrics["test_mae"],
                "test/r2": metrics["test_r2"],
                "test/pearson_mean": metrics["test_pearson_mean"],
                "test/pearson_median": metrics["test_pearson_median"],
                "test/pearson_std": metrics["test_pearson_std"],
            }
        )

    return metrics
