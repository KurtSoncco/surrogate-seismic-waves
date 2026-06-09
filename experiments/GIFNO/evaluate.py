# evaluate.py
"""Evaluation: masked metrics and spatial TF heatmaps."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import wandb

import config


def _masked_flat(pred: np.ndarray, target: np.ndarray, mask: np.ndarray):
    """Extract values at recorder positions only."""
    pred_list, target_list = [], []
    for i in range(len(mask)):
        idx = np.where(mask[i] > 0.5)[0]
        if len(idx):
            pred_list.append(pred[i, idx, :])
            target_list.append(target[i, idx, :])
    if not pred_list:
        return np.array([]), np.array([])
    return np.vstack([p.ravel() for p in pred_list]), np.vstack(
        [t.ravel() for t in target_list]
    )


def _compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
) -> Dict[str, float]:
    pred_flat, target_flat = _masked_flat(predictions, targets, masks)
    if pred_flat.size == 0:
        return {"test_mse": 0.0, "test_mae": 0.0, "test_rel_l2": 0.0}

    mse = float(np.mean((target_flat - pred_flat) ** 2))
    mae = float(np.mean(np.abs(target_flat - pred_flat)))
    rel_l2 = float(
        np.linalg.norm(target_flat - pred_flat) / (np.linalg.norm(target_flat) + 1e-12)
    )
    pearson = float(np.corrcoef(pred_flat, target_flat)[0, 1])
    if not np.isfinite(pearson):
        pearson = 0.0

    return {
        "test_mse": mse,
        "test_mae": mae,
        "test_rel_l2": rel_l2,
        "test_pearson": pearson,
    }


def _plot_tf_heatmap(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    freq: np.ndarray,
    save_path: Path,
    title: str,
) -> None:
    """Plot target vs prediction TF fields at recorder positions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    recorder_idx = np.where(mask > 0.5)[0]
    if len(recorder_idx) == 0:
        plt.close(fig)
        return

    t_rec = target[recorder_idx, :]
    p_rec = pred[recorder_idx, :]

    im0 = axes[0].imshow(
        np.log10(t_rec + 1e-12),
        aspect="auto",
        origin="lower",
        extent=[freq[0], freq[-1], 0, len(recorder_idx)],
        cmap="viridis",
    )
    axes[0].set_title("Target |TF| (log10)")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Recorder index")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        np.log10(p_rec + 1e-12),
        aspect="auto",
        origin="lower",
        extent=[freq[0], freq[-1], 0, len(recorder_idx)],
        cmap="viridis",
    )
    axes[1].set_title("Prediction |TF| (log10)")
    axes[1].set_xlabel("Frequency (Hz)")
    plt.colorbar(im1, ax=axes[1])

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_lateral_tf_curves(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    freq: np.ndarray,
    save_path: Path,
    n_curves: int = 3,
) -> None:
    """Semilogx TF curves at a few recorder x-positions."""
    recorder_idx = np.where(mask > 0.5)[0]
    if len(recorder_idx) == 0:
        return
    picks = recorder_idx[np.linspace(0, len(recorder_idx) - 1, n_curves, dtype=int)]

    fig, ax = plt.subplots(figsize=(10, 6))
    for x_idx in picks:
        ax.semilogx(freq, target[x_idx], "-", alpha=0.5, label=f"target x={x_idx}")
        ax.semilogx(freq, pred[x_idx], "--", alpha=0.8, label=f"pred x={x_idx}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|TF|")
    ax.set_title("TF curves at recorder positions")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    freq_data: Optional[np.ndarray] = None,
    save_dir: Optional[Path] = None,
    run: Optional[Any] = None,
    seed: int = 42,
) -> Dict[str, float]:
    save_dir = save_dir or config.RESULTS_SAVE_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    freq = freq_data if freq_data is not None else np.load(config.TF_FREQ_PATH)

    device = next(model.parameters()).device
    model.eval()
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_masks: List[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets, masks in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())
            all_masks.append(masks.numpy())

    predictions = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    metrics = _compute_metrics(predictions, targets, masks)

    n_plot = min(3, len(predictions))
    for i in range(n_plot):
        _plot_tf_heatmap(
            predictions[i],
            targets[i],
            masks[i],
            freq,
            save_dir / f"tf_heatmap_sample_{i}.png",
            title=f"Sample {i}",
        )
        _plot_lateral_tf_curves(
            predictions[i],
            targets[i],
            masks[i],
            freq,
            save_dir / f"tf_curves_sample_{i}.png",
        )

    if run is not None:
        wandb.log({k: v for k, v in metrics.items()})
        for i in range(n_plot):
            wandb.log(
                {
                    f"eval/tf_heatmap_{i}": wandb.Image(
                        str(save_dir / f"tf_heatmap_sample_{i}.png")
                    )
                }
            )

    return metrics
