# evaluate.py
"""Evaluation: masked metrics, spatial TF heatmaps, and W&B diagnostics."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import wandb

import config


def _safe_log10(arr: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """log10(|x|) with a positive floor to avoid invalid-value warnings."""
    return np.log10(np.maximum(np.abs(arr), floor))


def _pearson_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two 1-D arrays; returns NaN if undefined."""
    if a.size < 2:
        return float("nan")
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _recorder_x_indices(mask: np.ndarray) -> np.ndarray:
    """Sorted grid x-indices where recorders are active."""
    return np.where(mask > 0.5)[0]


def _central_recorder_x(mask: np.ndarray) -> int:
    """Grid x-index of the central lateral recorder."""
    recorder_idx = _recorder_x_indices(mask)
    if len(recorder_idx) == 0:
        return config.NX // 2
    center = config.NX // 2
    return int(recorder_idx[np.argmin(np.abs(recorder_idx - center))])


def _masked_flat(pred: np.ndarray, target: np.ndarray, mask: np.ndarray):
    """Extract values at recorder positions only."""
    pred_list, target_list = [], []
    for i in range(len(mask)):
        idx = _recorder_x_indices(mask[i])
        if len(idx):
            pred_list.append(pred[i, idx, :])
            target_list.append(target[i, idx, :])
    if not pred_list:
        return np.array([]), np.array([])
    return np.concatenate([p.ravel() for p in pred_list]), np.concatenate(
        [t.ravel() for t in target_list]
    )


def _compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
) -> Dict[str, float]:
    pred_flat, target_flat = _masked_flat(predictions, targets, masks)
    if pred_flat.size == 0:
        return {
            "test_mse": 0.0,
            "test_mae": 0.0,
            "test_rel_l2": 0.0,
            "test_pearson": 0.0,
        }

    mse = float(np.mean((target_flat - pred_flat) ** 2))
    mae = float(np.mean(np.abs(target_flat - pred_flat)))
    rel_l2 = float(
        np.linalg.norm(target_flat - pred_flat) / (np.linalg.norm(target_flat) + 1e-12)
    )
    pearson = _pearson_1d(pred_flat, target_flat)
    if not np.isfinite(pearson):
        pearson = 0.0

    return {
        "test_mse": mse,
        "test_mae": mae,
        "test_rel_l2": rel_l2,
        "test_pearson": pearson,
    }


def _pearson_by_recorder(
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-sample Pearson r at each recorder x-index (correlation along frequency).

    Returns
    -------
    recorder_x : (R,) grid x-indices
    pearson    : (R, N) matrix — one r per recorder per test sample
    """
    recorder_x = _recorder_x_indices(masks[0])
    n_rec = len(recorder_x)
    n_samples = len(predictions)
    pearson = np.full((n_rec, n_samples), np.nan, dtype=np.float64)

    for s in range(n_samples):
        for r, x_idx in enumerate(recorder_x):
            pearson[r, s] = _pearson_1d(
                np.abs(predictions[s, x_idx, :]),
                np.abs(targets[s, x_idx, :]),
            )

    return recorder_x, pearson


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
    recorder_idx = _recorder_x_indices(mask)
    if len(recorder_idx) == 0:
        plt.close(fig)
        return

    t_rec = np.abs(target[recorder_idx, :])
    p_rec = np.abs(pred[recorder_idx, :])

    im0 = axes[0].imshow(
        _safe_log10(t_rec),
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
        _safe_log10(p_rec),
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


def _plot_central_tf_curve(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    freq: np.ndarray,
    save_path: Path,
    sample_label: str,
) -> None:
    """|TF| vs frequency at the central lateral recorder — pred vs target."""
    x_idx = _central_recorder_x(mask)
    t_line = np.abs(target[x_idx, :])
    p_line = np.abs(pred[x_idx, :])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogx(freq, t_line, "-", linewidth=2, label=f"target (x={x_idx})")
    ax.semilogx(freq, p_line, "--", linewidth=2, label=f"pred (x={x_idx})")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|TF|")
    ax.set_title(f"Central recorder |TF| — {sample_label}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_pearson_boxplot(
    recorder_x: np.ndarray,
    pearson: np.ndarray,
    save_path: Path,
) -> None:
    """Box plot of per-sample Pearson r grouped by lateral recorder index."""
    valid = pearson[~np.all(np.isnan(pearson), axis=1)]
    valid_x = recorder_x[~np.all(np.isnan(pearson), axis=1)]
    if valid.size == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    positions = np.arange(len(valid_x))
    bp = ax.boxplot(
        [valid[i, ~np.isnan(valid[i])] for i in range(len(valid_x))],
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels([str(x) for x in valid_x], rotation=45, ha="right")
    ax.set_xlabel("Recorder grid x-index")
    ax.set_ylabel("Pearson r (|TF| pred vs |TF| target, along frequency)")
    ax.set_title("Per-recorder correlation across test samples")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _select_random_indices(n_total: int, n_pick: int, seed: int) -> np.ndarray:
    """Pick up to n_pick distinct sample indices (all if n_total < n_pick)."""
    n_pick = min(n_pick, n_total)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=n_pick, replace=False))


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
    recorder_x, pearson_mat = _pearson_by_recorder(predictions, targets, masks)

    per_rec_mean = np.nanmean(pearson_mat, axis=1)
    if np.any(np.isfinite(per_rec_mean)):
        metrics["test_pearson_central"] = float(
            per_rec_mean[np.argmin(np.abs(recorder_x - config.NX // 2))]
        )
        metrics["test_pearson_recorder_mean"] = float(np.nanmean(per_rec_mean))

    heatmap_idx = _select_random_indices(
        len(predictions), config.EVAL_N_HEATMAPS, seed
    )
    central_idx = _select_random_indices(
        len(predictions), config.EVAL_N_CENTRAL_CURVES, seed + 1
    )

    for rank, i in enumerate(heatmap_idx):
        _plot_tf_heatmap(
            predictions[i],
            targets[i],
            masks[i],
            freq,
            save_dir / f"tf_heatmap_sample_{i}.png",
            title=f"Sample {i} (random pick {rank + 1}/{len(heatmap_idx)})",
        )

    for rank, i in enumerate(central_idx):
        _plot_central_tf_curve(
            predictions[i],
            targets[i],
            masks[i],
            freq,
            save_dir / f"central_tf_sample_{i}.png",
            sample_label=f"sample {i} (pick {rank + 1}/{len(central_idx)})",
        )

    pearson_plot_path = save_dir / "pearson_by_recorder_boxplot.png"
    _plot_pearson_boxplot(recorder_x, pearson_mat, pearson_plot_path)

    if run is not None:
        log_payload: Dict[str, Any] = {k: v for k, v in metrics.items()}

        for rank, i in enumerate(heatmap_idx):
            log_payload[f"eval/tf_heatmap_{rank}"] = wandb.Image(
                str(save_dir / f"tf_heatmap_sample_{i}.png"),
                caption=f"Sample index {i}",
            )

        for rank, i in enumerate(central_idx):
            log_payload[f"eval/central_tf_{rank}"] = wandb.Image(
                str(save_dir / f"central_tf_sample_{i}.png"),
                caption=f"Central recorder, sample index {i}",
            )

        if pearson_plot_path.exists():
            log_payload["eval/pearson_by_recorder"] = wandb.Image(
                str(pearson_plot_path),
                caption="Pearson r per recorder (box = test-sample distribution)",
            )

        wandb.log(log_payload)

    return metrics
