# evaluate.py
"""Evaluation: masked metrics, spatial TF heatmaps, and W&B diagnostics."""

import json
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
    """Target / prediction / L1-error TF fields at recorder positions.

    Heatmaps are only meaningful alongside the error field, so the absolute
    error |target - pred| is always shown as a third panel.
    """
    recorder_idx = recorder_x_indices_from_mask(mask)
    if len(recorder_idx) == 0:
        return

    t_rec = np.abs(target[recorder_idx, :])
    p_rec = np.abs(pred[recorder_idx, :])
    err_rec = np.abs(t_rec - p_rec)

    extent = [freq[0], freq[-1], 0, len(recorder_idx)]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)

    vmin = min(_safe_log10(t_rec).min(), _safe_log10(p_rec).min())
    vmax = max(_safe_log10(t_rec).max(), _safe_log10(p_rec).max())

    im0 = axes[0].imshow(
        _safe_log10(t_rec),
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title("Target |TF| (log10)")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Recorder index")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        _safe_log10(p_rec),
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title("Prediction |TF| (log10)")
    axes[1].set_xlabel("Frequency (Hz)")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(
        err_rec,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="magma",
    )
    axes[2].set_title("L1 error |target - pred|")
    axes[2].set_xlabel("Frequency (Hz)")
    plt.colorbar(im2, ax=axes[2])

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


def _recorder_abs_stack(
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """(N, R, F) |TF| at recorder columns for predictions and targets."""
    recorder_idx = recorder_x_indices_from_mask(masks[0])
    p = np.abs(predictions[:, recorder_idx, :])
    t = np.abs(targets[:, recorder_idx, :])
    return p, t


def _plot_error_by_frequency(
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    freq: np.ndarray,
    save_path: Path,
) -> None:
    """Per-frequency |error| and relative error (median + p10-p90 band)."""
    p, t = _recorder_abs_stack(predictions, targets, masks)
    abs_err = np.abs(p - t).reshape(-1, len(freq))  # (N*R, F)
    rel_err = abs_err / (t.reshape(-1, len(freq)) + 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, data, label in (
        (axes[0], abs_err, "absolute |error|"),
        (axes[1], rel_err, "relative |error| / |target|"),
    ):
        med = np.nanmedian(data, axis=0)
        p10 = np.nanpercentile(data, 10, axis=0)
        p90 = np.nanpercentile(data, 90, axis=0)
        ax.fill_between(freq, p10, p90, alpha=0.25, color="steelblue", label="p10-p90")
        ax.plot(freq, med, color="navy", linewidth=2, label="median")
        ax.set_xscale("log")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(label)
        ax.set_title(f"Per-frequency {label}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle("Error vs frequency (pooled over recorders & test samples)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_error_freq_hist2d(
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    freq: np.ndarray,
    save_path: Path,
) -> None:
    """2D histogram of (frequency, |error|) over all recorder/sample bins."""
    p, t = _recorder_abs_stack(predictions, targets, masks)
    abs_err = np.abs(p - t)  # (N, R, F)
    n_rep = abs_err.shape[0] * abs_err.shape[1]
    freq_grid = np.tile(freq[None, :], (n_rep, 1)).reshape(-1)
    err_flat = abs_err.reshape(-1)
    good = np.isfinite(err_flat) & (err_flat > 0)
    if not np.any(good):
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    h = ax.hist2d(
        np.log10(np.maximum(freq_grid[good], 1e-6)),
        np.log10(err_flat[good]),
        bins=(80, 80),
        cmap="inferno",
    )
    plt.colorbar(h[3], ax=ax, label="count")
    ax.set_xlabel("log10 frequency (Hz)")
    ax.set_ylabel("log10 |error|")
    ax.set_title("Per-frequency error histogram (2D density)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_metric_histograms(
    per_sample: Dict[str, np.ndarray],
    save_path: Path,
) -> None:
    """Grid of per-sample metric histograms."""
    keys = [
        "rel_l2",
        "rel_l2_band_mid",
        "rel_l2_band_high",
        "pearson",
        "peak_amp_err",
        "peak_freq_err",
        "logspec_rel_l2",
        "linf_peak",
    ]
    keys = [k for k in keys if k in per_sample]
    if not keys:
        return
    ncol = 4
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax, key in zip(axes, keys):
        v = per_sample[key]
        v = v[np.isfinite(v)]
        if v.size == 0:
            ax.set_visible(False)
            continue
        ax.hist(v, bins=30, color="steelblue", alpha=0.8)
        ax.axvline(np.median(v), color="crimson", linestyle="--", linewidth=1)
        ax.set_title(key)
        ax.grid(True, alpha=0.3)
    for ax in axes[len(keys) :]:
        ax.set_visible(False)
    fig.suptitle("Per-sample metric distributions")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# Stratification variables: name -> (manifest column, human label)
_STRAT_SPECS: List[Tuple[str, str, str]] = [
    ("CoV", "CoV", "coefficient of variation"),
    ("H", "H_discretized", "soil thickness H"),
    ("rH", "rH", "correlation length rH"),
]
_STRAT_METRICS: Tuple[str, ...] = (
    "rel_l2",
    "rel_l2_band_mid",
    "rel_l2_band_high",
    "pearson",
    "peak_amp_err",
    "logspec_rel_l2",
)


def _extract_test_metadata(test_loader: Any) -> Dict[str, np.ndarray]:
    """Per-test-sample stratifier values aligned to evaluation order.

    Returns {} if the loader's dataset does not expose manifest rows. Only
    columns present in the manifest are returned; missing/non-finite columns
    (e.g. rH on an un-backfilled manifest) are dropped with a note.
    """
    from torch.utils.data import Subset

    ds = getattr(test_loader, "dataset", None)
    rows: Optional[List[dict]] = None
    if isinstance(ds, Subset) and hasattr(ds.dataset, "manifest_rows"):
        rows = [ds.dataset.manifest_rows[i] for i in ds.indices]
    elif hasattr(ds, "manifest_rows"):
        rows = list(ds.manifest_rows)
    if not rows:
        return {}

    def col(key: str) -> np.ndarray:
        out = np.full(len(rows), np.nan, dtype=np.float64)
        for i, r in enumerate(rows):
            try:
                out[i] = float(r.get(key, "nan"))
            except (TypeError, ValueError):
                pass
        return out

    meta: Dict[str, np.ndarray] = {}
    for name, column, _ in _STRAT_SPECS:
        vals = col(column)
        if (
            np.sum(np.isfinite(vals)) >= 2
            and np.unique(vals[np.isfinite(vals)]).size > 1
        ):
            meta[name] = vals
        else:
            print(f"[evaluate] stratifier '{name}' ({column}) unavailable — skipping")
    return meta


def _stratified_summary(
    per_sample: Dict[str, np.ndarray],
    strat_values: np.ndarray,
    strat_name: str,
    n_bins: int,
) -> Tuple[Dict[str, float], List[dict]]:
    """Metric summaries within quantile bins of a stratifier.

    Returns a flat dict (for W&B/json) and a per-bin table (list of dicts).
    """
    finite = np.isfinite(strat_values)
    sv = strat_values[finite]
    if sv.size < n_bins:
        return {}, []
    edges = np.unique(np.quantile(sv, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 3:
        return {}, []
    bin_idx = np.clip(np.digitize(strat_values, edges[1:-1]), 0, len(edges) - 2)

    flat: Dict[str, float] = {}
    table: List[dict] = []
    for b in range(len(edges) - 1):
        sel = finite & (bin_idx == b)
        row = {
            "stratifier": strat_name,
            "bin": b,
            "lo": float(edges[b]),
            "hi": float(edges[b + 1]),
            "count": int(np.sum(sel)),
        }
        for m in _STRAT_METRICS:
            if m not in per_sample:
                continue
            v = per_sample[m][sel]
            v = v[np.isfinite(v)]
            med = float(np.median(v)) if v.size else float("nan")
            row[f"{m}_median"] = med
            flat[f"test_strat_{strat_name}_q{b}_{m}_median"] = med
        table.append(row)
    return flat, table


def _plot_metrics_by_stratifier(
    per_sample: Dict[str, np.ndarray],
    strat_values: np.ndarray,
    strat_name: str,
    label: str,
    save_path: Path,
    n_bins: int,
) -> List[dict]:
    """Plot median metrics across quantile bins of a stratifier; return table."""
    _, table = _stratified_summary(per_sample, strat_values, strat_name, n_bins)
    if not table:
        return []
    centers = [0.5 * (r["lo"] + r["hi"]) for r in table]
    metrics = [m for m in _STRAT_METRICS if m in per_sample]
    ncol = 3
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax, m in zip(axes, metrics):
        ys = [r.get(f"{m}_median", np.nan) for r in table]
        ax.plot(centers, ys, "o-", color="navy")
        ax.set_xlabel(label)
        ax.set_ylabel(f"{m} (median)")
        ax.set_title(m)
        ax.grid(True, alpha=0.3)
    for ax in axes[len(metrics) :]:
        ax.set_visible(False)
    counts = ", ".join(f"q{r['bin']}:n={r['count']}" for r in table)
    fig.suptitle(f"Metrics by {label} quantiles ({counts})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return table


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

    err_freq_path = save_dir / "error_by_frequency.png"
    _plot_error_by_frequency(predictions, targets, masks, freq, err_freq_path)

    err_hist_path = save_dir / "error_freq_hist2d.png"
    _plot_error_freq_hist2d(predictions, targets, masks, freq, err_hist_path)

    metric_hist_path = save_dir / "metric_histograms.png"
    _plot_metric_histograms(per_sample, metric_hist_path)

    # Stratified breakdowns: CoV, H, rH quantile bins.
    n_bins = int(getattr(config, "EVAL_STRAT_BINS", 4))
    meta = _extract_test_metadata(test_loader)
    strat_plot_paths: Dict[str, Path] = {}
    strat_tables: Dict[str, List[dict]] = {}
    for name, _column, label in _STRAT_SPECS:
        if name not in meta:
            continue
        flat, _ = _stratified_summary(per_sample, meta[name], name, n_bins)
        metrics.update(flat)
        strat_path = save_dir / f"metrics_by_{name}_quantiles.png"
        table = _plot_metrics_by_stratifier(
            per_sample, meta[name], name, label, strat_path, n_bins
        )
        if table:
            strat_plot_paths[name] = strat_path
            strat_tables[name] = table

    # Dump per-sample arrays + metadata for cross-variant aggregation.
    npz_payload: Dict[str, np.ndarray] = {
        k: np.asarray(v) for k, v in per_sample.items()
    }
    for name, vals in meta.items():
        npz_payload[f"meta_{name}"] = vals
    np.savez(save_dir / "per_sample_metrics.npz", **npz_payload)
    if strat_tables:
        (save_dir / "stratified_metrics.json").write_text(
            json.dumps(strat_tables, indent=2)
        )

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
            ("error_by_frequency", err_freq_path, "Per-frequency error bands"),
            ("error_freq_hist2d", err_hist_path, "Per-frequency error histogram"),
            ("metric_histograms", metric_hist_path, "Per-sample metric histograms"),
        ]:
            if plot_path.exists():
                log_payload[f"eval/{plot_key}"] = wandb.Image(
                    str(plot_path), caption=caption
                )

        for name, strat_path in strat_plot_paths.items():
            if strat_path.exists():
                log_payload[f"eval/metrics_by_{name}"] = wandb.Image(
                    str(strat_path), caption=f"Metrics by {name} quantiles"
                )

        for name, table in strat_tables.items():
            columns = list(table[0].keys())
            wb_table = wandb.Table(columns=columns)
            for row in table:
                wb_table.add_data(*[row.get(c) for c in columns])
            log_payload[f"eval/strat_table_{name}"] = wb_table

        wandb.log(log_payload)

    return metrics
