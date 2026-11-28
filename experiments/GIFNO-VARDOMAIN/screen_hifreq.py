#!/usr/bin/env python3
"""Frequency-window autoencoder and the frozen depth-column comparison.

The frequency model is the shared shallow decoder of Kim and Choi
(arXiv:2611.17773). Each label bin belongs to one window. The decoder output
connects a bin only to the hidden units of its neighboring bins. Windows from
the training curves are pooled, and the holdout curves are scored after the
windows are stacked back into a full 0.1–10 Hz curve.

The depth score loads the frozen column autoencoder and compares it with the
equal-weight masked row collapse on the same holdout columns.

    python screen_hifreq.py --cache ... --out .../hifreq --depth-checkpoint .../depth_ae.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import Cache
from model import ColumnDepthAutoencoder, MaskedDepthCollapse, SparseWindowAutoencoder, fit_log_pod
from pretrain_ae import (
    _refuse_shipped,
    _strip_batch,
    column_splits,
    pod_holdout_mae,
    select_curves,
    split_rows,
)
from train import _curves

WINDOW = 50
N_LOC = 3
KNN = 5
BLOCK = 4
STRIDE = 1
BANDS = ((0.1, 0.4), (0.4, 1.0), (1.0, 2.5), (2.5, 5.0), (5.0, 10.0))


def tile_windows(curves: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """(N, F) -> (N, F/window, window). Windows do not overlap."""
    count, n_freq = curves.shape
    if n_freq % window:
        raise ValueError(f"frequency count {n_freq} is not divisible by window {window}")
    return np.asarray(curves, dtype=np.float64).reshape(count, n_freq // window, window)


def stack_windows(windows: np.ndarray) -> np.ndarray:
    count, n_windows, width = windows.shape
    return windows.reshape(count, n_windows * width)


def band_means(freq: np.ndarray, series: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for lo, hi in BANDS:
        chosen = (freq >= lo) & (freq <= hi)
        key = f"{lo:g}-{hi:g}"
        out[key] = {name: float(values[chosen].mean()) for name, values in series.items()}
    return out


def relative_frobenius(recon: np.ndarray, target: np.ndarray) -> float:
    scale = float(np.linalg.norm(target))
    if scale == 0.0:
        return 0.0
    return float(np.linalg.norm(recon - target) / scale)


def train_windows(
    train: np.ndarray,
    hold: np.ndarray,
    *,
    epochs: int,
    batch: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> SparseWindowAutoencoder:
    """Fit one autoencoder on every training window. ``train`` and ``hold`` are (N, F)."""
    torch.manual_seed(seed)
    model = SparseWindowAutoencoder(WINDOW, N_LOC, knn=KNN, block=BLOCK, stride=STRIDE).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    pooled = torch.as_tensor(tile_windows(train).reshape(-1, WINDOW), dtype=torch.float32)
    hold_w = torch.as_tensor(tile_windows(hold).reshape(-1, WINDOW), dtype=torch.float32)
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(pooled.shape[0])
        total = 0.0
        seen = 0
        for start in range(0, order.size, batch):
            idx = order[start : start + batch]
            batch_x = pooled[idx].to(device)
            loss = (model(batch_x) - batch_x).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * batch_x.shape[0]
            seen += batch_x.shape[0]
        model.eval()
        hold_mse = _batched_mse(model, hold_w, device, batch)
        print(f"epoch {epoch + 1}/{epochs} train {total / max(seen, 1):.6f} hold {hold_mse:.6f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def _batched_mse(model: SparseWindowAutoencoder, windows: torch.Tensor, device: torch.device, batch: int) -> float:
    total = 0.0
    seen = 0
    for start in range(0, windows.shape[0], batch):
        chunk = windows[start : start + batch].to(device)
        total += float((model(chunk) - chunk).square().sum())
        seen += chunk.numel()
    return total / max(seen, 1)


@torch.no_grad()
def reconstruct_curves(model: SparseWindowAutoencoder, curves: np.ndarray, device: torch.device, batch: int) -> np.ndarray:
    windows = tile_windows(curves)
    count, n_windows, width = windows.shape
    flat = torch.as_tensor(windows.reshape(-1, width), dtype=torch.float32)
    pieces = []
    for start in range(0, flat.shape[0], batch):
        pieces.append(model(flat[start : start + batch].to(device)).cpu().numpy())
    pred = np.concatenate(pieces, axis=0).reshape(count, n_windows, width)
    return stack_windows(pred)


def pod_reconstruction(train: np.ndarray, hold: np.ndarray, rank: int) -> np.ndarray:
    modes, mean = fit_log_pod(train, min(int(rank), train.shape[0] - 1, train.shape[1]))
    coeff = (hold - mean) @ modes.T
    return np.asarray(mean + coeff @ modes, dtype=np.float64)


def linear_broadcast(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Broadcast the equal-weight masked mean to every row. One weight per depth row, all ones."""
    collapse = MaskedDepthCollapse(values.shape[1], values.shape[2]).to(values.device)
    with torch.no_grad():
        collapse.weight.fill_(1.0)
    surface = collapse(values, mask)
    return surface.expand_as(values)


def _row_sums(err: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum of absolute error and the masked channel count at each depth row."""
    weight = mask[:, None]
    summed = (err * weight).sum(dim=(0, 1, 3))
    count = mask.sum(dim=(0, 2)) * err.shape[1]
    return summed, count


@torch.no_grad()
def score_depth_columns(
    cache: Cache,
    pairs: list[tuple[int, int]],
    checkpoint: Path,
    device: torch.device,
    batch: int,
    code_dim: int = 2,
) -> dict:
    model = ColumnDepthAutoencoder(2, code_dim=code_dim).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    ae_sum: list[float] = []
    lin_sum: list[float] = []
    counts: list[float] = []
    ae_total = 0.0
    lin_total = 0.0
    weight = 0.0
    grouped: dict[int, list[int]] = {}
    for flat, col in pairs:
        grouped.setdefault(int(flat), []).append(int(col))
    for flat, cols in grouped.items():
        packed = _strip_batch(cache, flat, cols)
        if packed is None:
            continue
        values, mask, depth = packed
        for start in range(0, values.shape[-1], batch):
            v = values[..., start : start + batch].to(device)
            m = mask[..., start : start + batch].to(device)
            d = depth[..., start : start + batch].to(device)
            _surface, recon = model(v, m, d)
            linear = linear_broadcast(v, m)
            ae_err = (recon - v).abs()
            lin_err = (linear - v).abs()
            cell = m[:, None]
            ae_total += float((ae_err * cell).sum())
            lin_total += float((lin_err * cell).sum())
            weight += float(cell.sum() * recon.shape[1])
            ae_row, row_count = _row_sums(ae_err, m)
            lin_row, _same = _row_sums(lin_err, m)
            _extend(ae_sum, lin_sum, counts, ae_row, lin_row, row_count)
    scale = max(weight, 1.0)
    return {
        "holdout_masked_l1": ae_total / scale,
        "linear_masked_l1": lin_total / scale,
        "row_index": list(range(len(counts))),
        "row_count": counts,
        "row_mae_ae": [s / c if c else 0.0 for s, c in zip(ae_sum, counts)],
        "row_mae_linear": [s / c if c else 0.0 for s, c in zip(lin_sum, counts)],
    }


def _extend(
    ae_sum: list[float],
    lin_sum: list[float],
    counts: list[float],
    ae_row: torch.Tensor,
    lin_row: torch.Tensor,
    row_count: torch.Tensor,
) -> None:
    height = int(ae_row.shape[0])
    while len(counts) < height:
        ae_sum.append(0.0)
        lin_sum.append(0.0)
        counts.append(0.0)
    ae_cpu = ae_row.detach().cpu().tolist()
    lin_cpu = lin_row.detach().cpu().tolist()
    count_cpu = row_count.detach().cpu().tolist()
    for i in range(height):
        ae_sum[i] += float(ae_cpu[i])
        lin_sum[i] += float(lin_cpu[i])
        counts[i] += float(count_cpu[i])


def screen(
    curves: np.ndarray,
    freq: np.ndarray,
    *,
    epochs: int,
    batch: int,
    lr: float,
    seed: int,
    device: torch.device,
    out: Path,
    depth: dict | None = None,
) -> dict:
    train_idx, hold_idx = split_rows(curves.shape[0], seed)
    train, hold = curves[train_idx], curves[hold_idx]
    print(f"curves train {train.shape[0]} hold {hold.shape[0]} freq {freq.size}", flush=True)
    model = train_windows(train, hold, epochs=epochs, batch=batch, lr=lr, seed=seed, device=device)
    torch.save(model.state_dict(), out / "sparse_window_ae.pt")
    recon = reconstruct_curves(model, hold, device, batch)
    pod60 = pod_reconstruction(train, hold, 60)
    pod64 = pod_reconstruction(train, hold, 64)
    mae_ae = np.abs(recon - hold).mean(axis=0)
    mae60 = np.abs(pod60 - hold).mean(axis=0)
    mae64 = pod_holdout_mae(train, hold, 64)
    series = {"ae": mae_ae, "pod60": mae60, "pod64": mae64}
    report = {
        "label_freq_hz": np.asarray(freq, dtype=np.float64).tolist(),
        "window": WINDOW,
        "n_loc": N_LOC,
        "n_windows": int(freq.size // WINDOW),
        "knn": KNN,
        "block": BLOCK,
        "stride": STRIDE,
        "n_train_curves": int(train.shape[0]),
        "n_holdout_curves": int(hold.shape[0]),
        "spectral": {
            "holdout_mae": mae_ae.tolist(),
            "pod60_holdout_mae": mae60.tolist(),
            "pod64_holdout_mae": mae64.tolist(),
            "mean_0p1_10": float(mae_ae.mean()),
            "pod60_mean_0p1_10": float(mae60.mean()),
            "pod64_mean_0p1_10": float(mae64.mean()),
            "bands": band_means(freq, series),
            "frobenius_ae": relative_frobenius(recon, hold),
            "frobenius_pod60": relative_frobenius(pod60, hold),
            "frobenius_pod64": relative_frobenius(pod64, hold),
        },
    }
    if depth is not None:
        report["depth"] = depth
    (out / "report.json").write_text(json.dumps(report))
    bands = report["spectral"]["bands"]
    print(
        f"mean ae {report['spectral']['mean_0p1_10']:.4f} "
        f"pod60 {report['spectral']['pod60_mean_0p1_10']:.4f} "
        f"pod64 {report['spectral']['pod64_mean_0p1_10']:.4f}",
        flush=True,
    )
    for key, row in bands.items():
        print(f"  {key} ae {row['ae']:.4f} pod60 {row['pod60']:.4f} pod64 {row['pod64']:.4f}", flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--depth-checkpoint", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    _refuse_shipped(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = Cache(args.cache)
    curves, _groups = _curves(cache, "residual")
    curves = select_curves(curves, 0, args.seed)
    freq = np.asarray(cache.freq, dtype=np.float64)
    _train_cols, hold_cols = column_splits(cache, 0, args.seed)
    print(f"depth holdout columns {len(hold_cols)} device {device}", flush=True)
    depth = score_depth_columns(cache, hold_cols, args.depth_checkpoint, device, batch=256)
    print(
        f"depth masked L1 {depth['holdout_masked_l1']:.4f} linear {depth['linear_masked_l1']:.4f}",
        flush=True,
    )
    screen(
        curves, freq, epochs=args.epochs, batch=args.batch, lr=args.lr, seed=args.seed,
        device=device, out=args.out, depth=depth,
    )


if __name__ == "__main__":
    main()
