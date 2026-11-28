#!/usr/bin/env python3
"""Offline self-supervised fits for the depth and spectral autoencoders.

The depth model rebuilds log Vs and log zeta on soil rows. The spectral model
rebuilds residual curves log|TF| - log|TF_1D|. A POD basis of the same rank is
fit on the same training rows. Error is the holdout mean absolute residual at
each label frequency (0.1–10 Hz on the shipped cache). The spectral decoder is
also queried from 0.001 to 50 Hz; those values are predictions, not errors.

    python pretrain_ae.py --cache ../../data/vardomain/cache --out /tmp/ae_sub --limit-curves 2048 --limit-columns 512
    python pretrain_ae.py --cache ... --out /tmp/ae_full --limit-curves 0 --limit-columns 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import Cache
from model import ColumnDepthAutoencoder, CurveEncoder, SpectralDecoder, fit_log_pod
from train import POD_CURVES_MAX, _assert_isolated, _curves

VS_REF = 400.0
ZETA_REF = 0.02
DEPTH_REF_M = 150.0
WIDE_F_MIN = 0.001
WIDE_F_MAX = 50.0
WIDE_N = 64
HOLDOUT = 0.1


def wide_grid() -> np.ndarray:
    return np.geomspace(WIDE_F_MIN, WIDE_F_MAX, WIDE_N).astype(np.float64)


def split_rows(n: int, seed: int, holdout: float = HOLDOUT) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError(f"need at least 2 rows to hold out, got {n}")
    perm = np.random.default_rng(seed).permutation(n)
    n_hold = max(1, int(round(holdout * n)))
    n_hold = min(n_hold, n - 1)
    return perm[n_hold:], perm[:n_hold]


def pod_holdout_mae(train: np.ndarray, hold: np.ndarray, k: int) -> np.ndarray:
    rank = min(int(k), train.shape[0] - 1, train.shape[1])
    modes, mean = fit_log_pod(train, rank)
    coeff = (hold - mean) @ modes.T
    recon = mean + coeff @ modes
    return np.abs(recon - hold).mean(axis=0)


def train_spectral(
    train: np.ndarray,
    freq: np.ndarray,
    k: int,
    *,
    epochs: int,
    batch: int,
    lr: float,
    hidden: int,
    seed: int,
    device: torch.device,
) -> tuple[CurveEncoder, SpectralDecoder]:
    torch.manual_seed(seed)
    encoder = CurveEncoder(train.shape[1], k, hidden).to(device)
    decoder = SpectralDecoder(k, hidden).to(device)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    data = torch.as_tensor(train, dtype=torch.float32, device=device)
    freq_t = torch.as_tensor(freq, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    encoder.train()
    decoder.train()
    for _ in range(epochs):
        order = rng.permutation(data.shape[0])
        for start in range(0, data.shape[0], batch):
            idx = order[start : start + batch]
            if idx.size == 0:
                continue
            batch_x = data[idx]
            code = encoder(batch_x)
            pred = decoder(code[:, None, :], freq_t).squeeze(1)
            loss = (pred - batch_x).abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    encoder.eval()
    decoder.eval()
    return encoder, decoder


@torch.no_grad()
def spectral_holdout_mae(
    encoder: CurveEncoder, decoder: SpectralDecoder, hold: np.ndarray, freq: np.ndarray, device: torch.device, batch: int,
) -> np.ndarray:
    freq_t = torch.as_tensor(freq, dtype=torch.float32, device=device)
    total = np.zeros(hold.shape[1], dtype=np.float64)
    seen = 0
    for start in range(0, hold.shape[0], batch):
        chunk = torch.as_tensor(hold[start : start + batch], dtype=torch.float32, device=device)
        pred = decoder(encoder(chunk)[:, None, :], freq_t).squeeze(1)
        total += (pred - chunk).abs().sum(dim=0).cpu().numpy()
        seen += chunk.shape[0]
    return total / max(seen, 1)


@torch.no_grad()
def wide_prediction(
    encoder: CurveEncoder, decoder: SpectralDecoder, hold: np.ndarray, wide: np.ndarray, device: torch.device, batch: int,
) -> np.ndarray:
    """Mean decoded curve on ``wide``. This is a shape, not an error."""
    wide_t = torch.as_tensor(wide, dtype=torch.float32, device=device)
    total = np.zeros(wide.shape[0], dtype=np.float64)
    seen = 0
    for start in range(0, hold.shape[0], batch):
        chunk = torch.as_tensor(hold[start : start + batch], dtype=torch.float32, device=device)
        pred = decoder(encoder(chunk)[:, None, :], wide_t).squeeze(1)
        total += pred.sum(dim=0).cpu().numpy()
        seen += chunk.shape[0]
    return total / max(seen, 1)


def _pack_columns(columns: list[tuple[np.ndarray, np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height = max(vs.shape[0] for vs, _ in columns)
    width = len(columns)
    values = np.zeros((1, 2, height, width), dtype=np.float32)
    mask = np.zeros((1, height, width), dtype=np.float32)
    depth = np.zeros((1, 1, height, width), dtype=np.float32)
    for i, (vs, zeta) in enumerate(columns):
        n = vs.shape[0]
        values[0, 0, :n, i] = np.log(np.maximum(vs, 1e-6) / VS_REF)
        values[0, 1, :n, i] = np.log(np.maximum(zeta, 1e-8) / ZETA_REF)
        mask[0, :n, i] = 1.0
        depth[0, 0, :n, i] = np.arange(n, dtype=np.float32) / DEPTH_REF_M
    return torch.from_numpy(values), torch.from_numpy(mask), torch.from_numpy(depth)


def depth_masked_l1(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask[:, None]
    err = (recon - target).abs() * weight
    return err.sum() / (weight.sum() * recon.shape[1]).clamp_min(1.0)


def train_depth(
    columns: list[tuple[np.ndarray, np.ndarray]],
    *,
    epochs: int,
    batch: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> ColumnDepthAutoencoder:
    torch.manual_seed(seed)
    model = ColumnDepthAutoencoder(2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(columns))
        for start in range(0, len(columns), batch):
            pick = [columns[int(i)] for i in order[start : start + batch]]
            if not pick:
                continue
            values, mask, depth = _pack_columns(pick)
            values, mask, depth = values.to(device), mask.to(device), depth.to(device)
            _surface, recon = model(values, mask, depth)
            loss = depth_masked_l1(recon, values, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def depth_holdout_l1(
    model: ColumnDepthAutoencoder, columns: list[tuple[np.ndarray, np.ndarray]], device: torch.device, batch: int,
) -> float:
    total = 0.0
    weight = 0.0
    for start in range(0, len(columns), batch):
        values, mask, depth = _pack_columns(columns[start : start + batch])
        values, mask, depth = values.to(device), mask.to(device), depth.to(device)
        _surface, recon = model(values, mask, depth)
        cell = mask.sum() * recon.shape[1]
        total += float((recon - values).abs().mul(mask[:, None]).sum())
        weight += float(cell)
    return total / max(weight, 1.0)


def _group_columns(pairs: list[tuple[int, int]]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = {}
    for flat, col in pairs:
        grouped.setdefault(int(flat), []).append(int(col))
    return grouped


def column_splits(cache: Cache, limit: int, seed: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Train and holdout (run, column) pairs. The holdout is not fit."""
    rng = np.random.default_rng(seed)
    train: list[tuple[int, int]] = []
    hold: list[tuple[int, int]] = []
    for flat in cache.runs("train", ("iid",)):
        camp, j = cache.campaign(int(flat))
        width = int(camp.vs.shape[-1])
        if width < 2:
            train.append((int(flat), 0))
            continue
        order = rng.permutation(width)
        n_hold = min(width - 1, max(1, int(round(HOLDOUT * width))))
        hold.extend((int(flat), int(c)) for c in order[:n_hold])
        train.extend((int(flat), int(c)) for c in order[n_hold:])
    if not train:
        raise RuntimeError("no training IID columns in the cache")
    if limit > 0:
        train = [train[int(i)] for i in rng.choice(len(train), size=min(limit, len(train)), replace=False)]
        hold = [hold[int(i)] for i in rng.choice(len(hold), size=min(max(1, limit // 10), len(hold)), replace=False)]
    return train, hold


def _strip_batch(cache: Cache, flat: int, cols: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Pack selected soil columns of one strip. Rows below n_active are absent."""
    camp, j = cache.campaign(flat)
    n_active = int(camp.n_active[j])
    if n_active < 1 or not cols:
        return None
    pick = np.asarray(cols, dtype=np.int64)
    vs = np.asarray(camp.vs[j, :n_active][:, pick], dtype=np.float64)
    zeta = np.asarray(camp.zeta[j, :n_active][:, pick], dtype=np.float64)
    height, width = vs.shape
    values = np.empty((1, 2, height, width), dtype=np.float32)
    values[0, 0] = np.log(np.maximum(vs, 1e-6) / VS_REF)
    values[0, 1] = np.log(np.maximum(zeta, 1e-8) / ZETA_REF)
    mask = np.ones((1, height, width), dtype=np.float32)
    depth = np.empty((1, 1, height, width), dtype=np.float32)
    depth[0, 0] = (np.arange(height, dtype=np.float32) / DEPTH_REF_M)[:, None]
    return torch.from_numpy(values), torch.from_numpy(mask), torch.from_numpy(depth)


def train_depth_indexed(
    cache: Cache,
    pairs: list[tuple[int, int]],
    *,
    epochs: int,
    batch: int,
    lr: float,
    seed: int,
    device: torch.device,
    code_dim: int = 2,
    hold_pairs: list[tuple[int, int]] | None = None,
) -> ColumnDepthAutoencoder:
    """One strip is read once per visit and packed as a single tensor."""
    torch.manual_seed(seed)
    model = ColumnDepthAutoencoder(2, code_dim=code_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    grouped = _group_columns(pairs)
    keys = np.asarray(list(grouped), dtype=np.int64)
    best_hold = float("inf")
    best_state: dict | None = None
    best_epoch = 0
    model.train()
    for epoch in range(epochs):
        for flat in keys[rng.permutation(keys.size)]:
            packed = _strip_batch(cache, int(flat), grouped[int(flat)])
            if packed is None:
                continue
            values, mask, depth = packed
            for start in range(0, values.shape[-1], batch):
                v = values[..., start : start + batch].to(device)
                m = mask[..., start : start + batch].to(device)
                d = depth[..., start : start + batch].to(device)
                _surface, recon = model(v, m, d)
                loss = depth_masked_l1(recon, v, m)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        message = f"depth epoch {epoch + 1}/{epochs}"
        if hold_pairs and (epoch + 1) % 5 == 0:
            model.eval()
            hold_l1 = depth_indexed_l1(model, cache, hold_pairs, device, batch)
            if hold_l1 < best_hold:
                best_hold = hold_l1
                best_epoch = epoch + 1
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            model.train()
            message += f" hold {hold_l1:.4f}"
        print(message, flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"kept epoch {best_epoch} hold {best_hold:.4f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def depth_indexed_l1(
    model: ColumnDepthAutoencoder, cache: Cache, pairs: list[tuple[int, int]], device: torch.device, batch: int,
) -> float:
    total = 0.0
    weight = 0.0
    for flat, cols in _group_columns(pairs).items():
        packed = _strip_batch(cache, flat, cols)
        if packed is None:
            continue
        values, mask, depth = packed
        for start in range(0, values.shape[-1], batch):
            v = values[..., start : start + batch].to(device)
            m = mask[..., start : start + batch].to(device)
            d = depth[..., start : start + batch].to(device)
            _surface, recon = model(v, m, d)
            total += float((recon - v).abs().mul(m[:, None]).sum())
            weight += float(m.sum() * recon.shape[1])
    return total / max(weight, 1.0)


def select_curves(curves: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if curves.shape[0] > POD_CURVES_MAX:
        take = np.random.default_rng(seed).choice(curves.shape[0], POD_CURVES_MAX, replace=False)
        curves = curves[take]
    if limit > 0 and curves.shape[0] > limit:
        take = np.random.default_rng(seed + 1).choice(curves.shape[0], limit, replace=False)
        curves = curves[take]
    return np.asarray(curves, dtype=np.float64)


def fit_report(
    curves: np.ndarray,
    freq: np.ndarray,
    train_columns: list[tuple[np.ndarray, np.ndarray]],
    hold_columns: list[tuple[np.ndarray, np.ndarray]],
    *,
    ranks: tuple[int, ...],
    epochs: int,
    batch: int,
    lr: float,
    seed: int,
    device: torch.device,
    out: Path,
) -> dict:
    train_idx, hold_idx = split_rows(curves.shape[0], seed)
    train, hold = curves[train_idx], curves[hold_idx]
    wide = wide_grid()
    spectral: dict[str, dict] = {}
    for k in ranks:
        mae_pod = pod_holdout_mae(train, hold, k)
        encoder, decoder = train_spectral(
            train, freq, k, epochs=epochs, batch=batch, lr=lr, hidden=256, seed=seed + k, device=device,
        )
        mae_ae = spectral_holdout_mae(encoder, decoder, hold, freq, device, batch)
        pred = wide_prediction(encoder, decoder, hold, wide, device, batch)
        torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "k": k}, out / f"spectral_k{k}.pt")
        spectral[str(k)] = {
            "holdout_mae": mae_ae.tolist(),
            "pod_holdout_mae": mae_pod.tolist(),
            "mean_0p1_10": float(mae_ae.mean()),
            "pod_mean_0p1_10": float(mae_pod.mean()),
            "ae_at_or_below_pod": bool(float(mae_ae.mean()) <= float(mae_pod.mean())),
            "wide_pred_mean": pred.tolist(),
        }
        print(
            f"k={k} holdout mae {spectral[str(k)]['mean_0p1_10']:.4f} "
            f"pod {spectral[str(k)]['pod_mean_0p1_10']:.4f}",
            flush=True,
        )
    depth_model = train_depth(train_columns, epochs=epochs, batch=batch, lr=lr, seed=seed, device=device)
    depth_l1 = depth_holdout_l1(depth_model, hold_columns, device, batch)
    torch.save(depth_model.state_dict(), out / "depth_ae.pt")
    print(f"depth holdout masked L1 {depth_l1:.4f}", flush=True)
    report = {
        "label_freq_hz": np.asarray(freq, dtype=np.float64).tolist(),
        "wide_freq_hz": wide.tolist(),
        "n_train_curves": int(train.shape[0]),
        "n_holdout_curves": int(hold.shape[0]),
        "n_train_columns": len(train_columns),
        "n_holdout_columns": len(hold_columns),
        "spectral": spectral,
        "depth": {"holdout_masked_l1": depth_l1},
    }
    (out / "report.json").write_text(json.dumps(report))
    return report


def _refuse_shipped(out: Path) -> None:
    _assert_isolated(out)
    text = str(out.resolve())
    for token in ("/nonlinear", "/early_width", "/coord_reseed"):
        if token in text:
            raise SystemExit(f"refusing to write into {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit-curves", type=int, default=2048, help="0 keeps the 80,000 curve cap")
    ap.add_argument("--limit-columns", type=int, default=512, help="0 uses every training IID column")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ranks", type=int, nargs="+", default=[64, 32])
    args = ap.parse_args()
    _refuse_shipped(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = Cache(args.cache)
    curves, _groups = _curves(cache, "residual")
    curves = select_curves(curves, args.limit_curves, args.seed)
    freq = np.asarray(cache.freq, dtype=np.float64)
    train_idx, hold_idx = split_rows(curves.shape[0], args.seed)
    train, hold = curves[train_idx], curves[hold_idx]
    wide = wide_grid()
    print(
        f"curves train {train.shape[0]} hold {hold.shape[0]} freq {freq.size} device {device}",
        flush=True,
    )
    spectral: dict[str, dict] = {}
    for k in args.ranks:
        mae_pod = pod_holdout_mae(train, hold, k)
        encoder, decoder = train_spectral(
            train, freq, k, epochs=args.epochs, batch=args.batch, lr=args.lr, hidden=256, seed=args.seed + k, device=device,
        )
        mae_ae = spectral_holdout_mae(encoder, decoder, hold, freq, device, args.batch)
        pred = wide_prediction(encoder, decoder, hold, wide, device, args.batch)
        torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "k": int(k)}, args.out / f"spectral_k{k}.pt")
        spectral[str(k)] = {
            "holdout_mae": mae_ae.tolist(),
            "pod_holdout_mae": mae_pod.tolist(),
            "mean_0p1_10": float(mae_ae.mean()),
            "pod_mean_0p1_10": float(mae_pod.mean()),
            "ae_at_or_below_pod": bool(float(mae_ae.mean()) <= float(mae_pod.mean())),
            "wide_pred_mean": pred.tolist(),
        }
        print(
            f"k={k} holdout mae {spectral[str(k)]['mean_0p1_10']:.4f} pod {spectral[str(k)]['pod_mean_0p1_10']:.4f}",
            flush=True,
        )
    col_train, col_hold = column_splits(cache, args.limit_columns, args.seed)
    print(f"columns train {len(col_train)} hold {len(col_hold)}", flush=True)
    depth_model = train_depth_indexed(
        cache, col_train, epochs=args.epochs, batch=args.batch, lr=args.lr, seed=args.seed, device=device,
    )
    depth_l1 = depth_indexed_l1(depth_model, cache, col_hold, device, args.batch)
    torch.save(depth_model.state_dict(), args.out / "depth_ae.pt")
    print(f"depth holdout masked L1 {depth_l1:.4f}", flush=True)
    report = {
        "label_freq_hz": freq.tolist(),
        "wide_freq_hz": wide.tolist(),
        "n_train_curves": int(train.shape[0]),
        "n_holdout_curves": int(hold.shape[0]),
        "n_train_columns": len(col_train),
        "n_holdout_columns": len(col_hold),
        "spectral": spectral,
        "depth": {"holdout_masked_l1": depth_l1},
    }
    (args.out / "report.json").write_text(json.dumps(report))


if __name__ == "__main__":
    main()
