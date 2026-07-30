"""GIFNO-style |TF|(f) comparison panels for W&B (center + edge recorders)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

import config
from model import GatedDeltaModel


def _rel_l2(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _freq_vector(n_freq: int) -> np.ndarray:
    if config.TF_FREQ_PATH.is_file():
        return np.load(config.TF_FREQ_PATH).astype(np.float64)
    return np.logspace(-1.0, 1.0, n_freq)


@torch.no_grad()
def collect_predictions(
    model: GatedDeltaModel,
    loader: DataLoader,
    device: torch.device,
    *,
    max_samples: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pred, target, mask, center_rel_l2 for up to ``max_samples``."""
    model.eval()
    preds, targs, masks, rels = [], [], [], []
    rec = config.recorder_x_indices()
    center = int(rec[len(rec) // 2])
    for batch in loader:
        x, haskell, target, mask, cov, dip, physics = [t.to(device) for t in batch]
        pred = model(x, haskell, cov, dip, physics=physics)
        for b in range(pred.shape[0]):
            p = pred[b].detach().cpu().numpy()
            t = target[b].detach().cpu().numpy()
            m = mask[b].detach().cpu().numpy()
            preds.append(p)
            targs.append(t)
            masks.append(m)
            rels.append(_rel_l2(p[center], t[center]))
            if len(preds) >= max_samples:
                return (
                    np.stack(preds),
                    np.stack(targs),
                    np.stack(masks),
                    np.asarray(rels, dtype=np.float64),
                )
    if not preds:
        empty = np.zeros((0,), dtype=np.float64)
        return empty, empty, empty, empty
    return (
        np.stack(preds),
        np.stack(targs),
        np.stack(masks),
        np.asarray(rels, dtype=np.float64),
    )


def plot_center_edge_overlay(
    pred: np.ndarray,
    target: np.ndarray,
    freq: np.ndarray,
    *,
    title: str,
    save_path: Path | None = None,
) -> Path:
    """3-panel loglog |TF|(f): left edge, center, right edge."""
    rec = config.recorder_x_indices()
    left, center, right = int(rec[0]), int(rec[len(rec) // 2]), int(rec[-1])
    panels = [
        ("left edge", left),
        ("center", center),
        ("right edge", right),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (name, xi) in zip(axes, panels):
        t = np.abs(target[xi])
        p = np.abs(pred[xi])
        ax.loglog(freq, t, "-", lw=2, label="target")
        ax.loglog(freq, p, "--", lw=2, label="pred")
        r = _rel_l2(p, t)
        pear = _pearson(p, t)
        ax.set_title(f"{name} x={xi}\nrelL2={r:.3f}  r={pear:.3f}")
        ax.set_xlabel("Frequency (Hz)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("|TF|")
    fig.suptitle(title)
    fig.tight_layout()
    out = save_path or (config.RESULTS_SAVE_DIR / "tf_overlay_tmp.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def log_tf_comparison_plots(
    model: GatedDeltaModel,
    loader: DataLoader,
    device: torch.device,
    *,
    tag: str = "test",
    n_random: int | None = None,
    n_worst: int | None = None,
    max_collect: int = 64,
) -> dict[str, float]:
    """Build center/edge overlays and log them to the active W&B run."""
    n_random = n_random if n_random is not None else getattr(config, "EVAL_N_TF_CURVES", 4)
    n_worst = n_worst if n_worst is not None else getattr(config, "EVAL_N_WORST_TF", 3)

    preds, targs, _masks, rels = collect_predictions(
        model, loader, device, max_samples=max_collect
    )
    if preds.size == 0:
        print("[TH-FNO] no samples for TF plots", flush=True)
        return {}

    freq = _freq_vector(preds.shape[-1])
    out_dir = config.RESULTS_SAVE_DIR / "tf_plots" / f"{config.PREDICT_MODE}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(config.SEED)
    n = len(rels)
    rand_idx = rng.choice(n, size=min(n_random, n), replace=False).tolist()
    worst_idx = np.argsort(-rels)[: min(n_worst, n)].tolist()

    payload: dict = {}
    for rank, i in enumerate(rand_idx):
        path = plot_center_edge_overlay(
            preds[i],
            targs[i],
            freq,
            title=f"{tag} random #{rank + 1} (idx={i}, relL2_c={rels[i]:.3f})",
            save_path=out_dir / f"random_{rank}_sample_{i}.png",
        )
        payload[f"eval/{tag}_tf_center_edge_random_{rank}"] = wandb.Image(str(path))

    for rank, i in enumerate(worst_idx):
        path = plot_center_edge_overlay(
            preds[i],
            targs[i],
            freq,
            title=f"{tag} worst #{rank + 1} (idx={i}, relL2_c={rels[i]:.3f})",
            save_path=out_dir / f"worst_{rank}_sample_{i}.png",
        )
        payload[f"eval/{tag}_tf_center_edge_worst_{rank}"] = wandb.Image(str(path))

    if wandb.run is not None:
        wandb.log(payload)
        print(f"[TH-FNO] logged {len(payload)} TF overlay panels to W&B → {out_dir}", flush=True)
    else:
        print(f"[TH-FNO] W&B inactive; wrote TF overlays to {out_dir}", flush=True)

    return {
        "plot_n": float(n),
        "plot_rel_l2_center_mean": float(np.mean(rels)),
        "plot_rel_l2_center_worst": float(rels[worst_idx[0]]) if worst_idx else float("nan"),
    }
