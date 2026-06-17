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
from metrics import (
    aggregate_test_metrics,
    pearson_1d,
    recorder_x_indices_from_mask,
    valley_mask_from_log_target,
)


def _safe_log10(arr: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """log10(|x|) with a positive floor to avoid invalid-value warnings."""
    return np.log10(np.maximum(np.abs(arr), floor))


def _central_recorder_x(mask: np.ndarray) -> int:
    """Grid x-index of the central lateral recorder."""
    recorder_idx = recorder_x_indices_from_mask(mask)
    if len(recorder_idx) == 0:
        return config.NX // 2
    center = config.NX // 2
    return int(recorder_idx[np.argmin(np.abs(recorder_idx - center))])


def _pearson_by_recorder(
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-recorder Pearson r across test samples. Returns (recorder_x, (R, N))."""
    recorder_x = recorder_x_indices_from_mask(masks[0])
    n_rec = len(recorder_x)
    n_samples = len(predictions)
    pearson = np.full((n_rec, n_samples), np.nan, dtype=np.float64)

    for s in range(n_samples):
        for r, x_idx in enumerate(recorder_x):
            pearson[r, s] = pearson_1d(
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
    recorder_idx = recorder_x_indices_from_mask(mask)
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
    *,
    mark_valleys: bool = False,
) -> None:
    """|TF| vs frequency at the central lateral recorder — pred vs target."""
    x_idx = _central_recorder_x(mask)
    t_line = np.abs(target[x_idx, :])
    p_line = np.abs(pred[x_idx, :])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.loglog(freq, t_line, "-", linewidth=2, label=f"target (x={x_idx})")
    ax.loglog(freq, p_line, "--", linewidth=2, label=f"pred (x={x_idx})")
    if mark_valleys:
        vmask = valley_mask_from_log_target(target[x_idx : x_idx + 1, :])[0]
        if np.any(vmask):
            ax.scatter(
                freq[vmask],
                t_line[vmask],
                c="crimson",
                s=18,
                zorder=5,
                label="valley bins (target)",
            )
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


def _plot_metric_ecdf(
    values: Dict[str, np.ndarray],
    save_path: Path,
    title: str,
) -> None:
    """ECDF of per-sample metrics."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, v in values.items():
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        xs = np.sort(v)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, label=label)
    ax.set_xlabel("Metric value")
    ax.set_ylabel("ECDF")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_rel_l2_vs_pearson(
    rel_l2: np.ndarray,
    pearson: np.ndarray,
    save_path: Path,
) -> None:
    """Scatter per-sample relative L2 vs Pearson r."""
    mask = np.isfinite(rel_l2) & np.isfinite(pearson)
    if not np.any(mask):
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(rel_l2[mask], pearson[mask], alpha=0.35, s=12, edgecolors="none")
    ax.set_xlabel("Per-sample relative L2")
    ax.set_ylabel("Per-sample Pearson r")
    ax.set_title("Tail outliers (bottom-left = bad)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_per_sample_metric_boxplot(
    per_sample: Dict[str, np.ndarray],
    save_path: Path,
) -> None:
    """Box plot comparing per-sample metric distributions."""
    labels = []
    data = []
    for key in ("rel_l2", "pearson", "linf", "linf_valley", "linf_peak", "h1_freq"):
        v = per_sample.get(key)
        if v is None:
            continue
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        labels.append(key)
        data.append(v)

    if not data:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)
    ax.set_title("Per-sample metric distributions (test set)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _select_random_indices(n_total: int, n_pick: int, seed: int) -> np.ndarray:
    """Pick up to n_pick distinct sample indices (all if n_total < n_pick)."""
    n_pick = min(n_pick, n_total)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=n_pick, replace=False))


def _select_worst_indices(
    rel_l2: np.ndarray,
    pearson: np.ndarray,
    n_pick: int,
) -> np.ndarray:
    """Pick samples with highest rel_l2 and lowest pearson (combined rank)."""
    n_pick = min(n_pick, len(rel_l2))
    rel_rank = np.argsort(np.nan_to_num(rel_l2, nan=-1.0))
    pear_rank = np.argsort(
        np.nan_to_num(pearson, nan=2.0)
    )  # ascending: low pearson first
    combined = np.zeros(len(rel_l2))
    for rank, idx in enumerate(rel_rank):
        combined[idx] += rank
    for rank, idx in enumerate(pear_rank):
        combined[idx] += rank
    worst = np.argsort(-combined)[:n_pick]
    return np.sort(worst)


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

    metrics, per_sample = aggregate_test_metrics(predictions, targets, masks, freq)
    recorder_x, pearson_mat = _pearson_by_recorder(predictions, targets, masks)

    per_rec_mean = np.nanmean(pearson_mat, axis=1)
    if np.any(np.isfinite(per_rec_mean)):
        metrics["test_pearson_central"] = float(
            per_rec_mean[np.argmin(np.abs(recorder_x - config.NX // 2))]
        )
        metrics["test_pearson_recorder_mean"] = float(np.nanmean(per_rec_mean))

    heatmap_idx = _select_random_indices(len(predictions), config.EVAL_N_HEATMAPS, seed)
    central_idx = _select_random_indices(
        len(predictions), config.EVAL_N_CENTRAL_CURVES, seed + 1
    )
    worst_idx = _select_worst_indices(
        per_sample["rel_l2"],
        per_sample["pearson"],
        config.EVAL_N_WORST_SAMPLES,
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

    for rank, i in enumerate(worst_idx):
        _plot_central_tf_curve(
            predictions[i],
            targets[i],
            masks[i],
            freq,
            save_dir / f"worst_central_tf_sample_{i}.png",
            sample_label=(
                f"sample {i} (worst {rank + 1}/{len(worst_idx)}, "
                f"rel_l2={per_sample['rel_l2'][i]:.3f}, "
                f"r={per_sample['pearson'][i]:.3f}, "
                f"linf_valley={per_sample['linf_valley'][i]:.3f})"
            ),
            mark_valleys=True,
        )

    pearson_plot_path = save_dir / "pearson_by_recorder_boxplot.png"
    _plot_pearson_boxplot(recorder_x, pearson_mat, pearson_plot_path)

    _plot_metric_ecdf(
        {
            "rel_l2": per_sample["rel_l2"],
            "pearson": per_sample["pearson"],
            "linf_valley": per_sample["linf_valley"],
            "linf_peak": per_sample["linf_peak"],
        },
        save_dir / "per_sample_linf_split_ecdf.png",
        "Per-sample linf valley vs peak ECDF",
    )

    ecdf_path = save_dir / "per_sample_ecdf.png"
    _plot_metric_ecdf(
        {"rel_l2": per_sample["rel_l2"], "pearson": per_sample["pearson"]},
        ecdf_path,
        "Per-sample rel L2 and Pearson ECDF",
    )

    scatter_path = save_dir / "rel_l2_vs_pearson.png"
    _plot_rel_l2_vs_pearson(per_sample["rel_l2"], per_sample["pearson"], scatter_path)

    sample_box_path = save_dir / "per_sample_metrics_boxplot.png"
    _plot_per_sample_metric_boxplot(per_sample, sample_box_path)

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

        for rank, i in enumerate(worst_idx):
            log_payload[f"eval/worst_central_tf_{rank}"] = wandb.Image(
                str(save_dir / f"worst_central_tf_sample_{i}.png"),
                caption=(
                    f"Worst sample {i}: rel_l2={per_sample['rel_l2'][i]:.3f}, "
                    f"pearson={per_sample['pearson'][i]:.3f}"
                ),
            )

        for plot_key, plot_path, caption in [
            ("pearson_by_recorder", pearson_plot_path, "Pearson r per recorder"),
            ("per_sample_ecdf", ecdf_path, "Per-sample ECDF"),
            (
                "per_sample_linf_split_ecdf",
                save_dir / "per_sample_linf_split_ecdf.png",
                "linf valley vs peak ECDF",
            ),
            ("rel_l2_vs_pearson", scatter_path, "rel L2 vs Pearson scatter"),
            (
                "per_sample_metrics_boxplot",
                sample_box_path,
                "Per-sample metric boxplot",
            ),
        ]:
            if plot_path.exists():
                log_payload[f"eval/{plot_key}"] = wandb.Image(
                    str(plot_path), caption=caption
                )

        wandb.log(log_payload)

    return metrics
