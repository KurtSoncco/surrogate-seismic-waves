#!/usr/bin/env python3
"""Stability of the frequency-window and soil-column autoencoders.

The holdout is the seed-0 split already used for the full-cache reports.
What changes is the initial seed and one hyperparameter at a time for the
frequency model, and the code width crossed with the learning rate for the
column model. A run keeps the epoch with the best holdout score.

Frequency is the shared window decoder. Its total code length is
20 windows times n_loc, and each run is compared with POD of that rank.
The column model is trained on columns shuffled across strips, in batches,
so its absolute error is not the strip-wise 0.247 figure. Comparisons inside
this sweep share that trainer.

    python stability_ae.py --cache ... --out .../ae_pretrain/stability
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from data import Cache
from model import ColumnDepthAutoencoder, SparseWindowAutoencoder
from pretrain_ae import (
    DEPTH_REF_M,
    _refuse_shipped,
    _strip_batch,
    column_splits,
    select_curves,
    split_rows,
)
from screen_hifreq import (
    BANDS,
    WINDOW,
    band_means,
    pod_reconstruction,
    relative_frobenius,
    tile_windows,
)
from train import _curves

SPLIT_SEED = 0
EPOCHS = 40
SEEDS = (0, 1, 2)
FREQ_BATCH = 4096
DEPTH_BATCH = 8192
DEPTH_EDGES = ((0, 10), (10, 30), (30, 60), (60, 110), (100, 110))

# Frequency factors move one at a time off this reference, except knn and block,
# which both set the output mask and are crossed.
FREQ_REFERENCE = {"n_loc": 3, "knn": 5, "block": 4, "lr": 1e-3}
FREQ_N_LOC = (2, 4)
FREQ_LR = (3e-4, 3e-3)
FREQ_KNN = (3, 5, 9)
FREQ_BLOCK = (2, 4, 8)

DEPTH_CODE = (2, 8, 16, 32, 64)
DEPTH_LR = (3e-4, 1e-3, 3e-3)

FREQ_METRICS = (
    "mae",
    "frobenius",
    "p95_bin",
    "band_0.1-0.4",
    "band_0.4-1",
    "band_1-2.5",
    "band_2.5-5",
    "band_5-10",
    "gap_mae",
    "gap_high",
)
DEPTH_METRICS = (
    "masked_l1",
    "vs_l1",
    "zeta_l1",
    "band_0_10",
    "band_10_30",
    "band_30_60",
    "band_60_110",
    "band_100_110",
    "worst_populated_row",
)


def frequency_configs() -> list[dict]:
    """Reference, learning rate, code length, and the knn-by-block mask grid."""
    seen: dict[tuple, dict] = {}

    def add(**factors: float) -> None:
        row = dict(FREQ_REFERENCE)
        row.update(factors)
        key = (row["n_loc"], row["knn"], row["block"], row["lr"])
        seen[key] = row

    add()
    for n_loc in FREQ_N_LOC:
        add(n_loc=n_loc)
    for lr in FREQ_LR:
        add(lr=lr)
    for knn in FREQ_KNN:
        for block in FREQ_BLOCK:
            add(knn=knn, block=block)
    return list(seen.values())


def depth_configs() -> list[dict]:
    return [{"code_dim": code, "lr": lr} for code in DEPTH_CODE for lr in DEPTH_LR]


def _band_key(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g}"


def spectral_metrics(per_bin: np.ndarray, frobenius: float, freq: np.ndarray, pod: dict) -> dict:
    bands = band_means(freq, {"ae": per_bin})
    out = {
        "mae": float(per_bin.mean()),
        "frobenius": float(frobenius),
        "p95_bin": float(np.quantile(per_bin, 0.95)),
        "pod_mae": float(pod["mae"]),
        "pod_high": float(pod["bands"][_band_key(5.0, 10.0)]),
    }
    for lo, hi in BANDS:
        key = _band_key(lo, hi)
        out[f"band_{key}"] = float(bands[key]["ae"])
    out["gap_mae"] = out["mae"] - out["pod_mae"]
    out["gap_high"] = out["band_5-10"] - out["pod_high"]
    return out


def pod_table(train: np.ndarray, hold: np.ndarray, freq: np.ndarray, ranks: tuple[int, ...]) -> dict[int, dict]:
    table = {}
    for rank in ranks:
        recon = pod_reconstruction(train, hold, rank)
        per_bin = np.abs(recon - hold).mean(axis=0)
        bands = band_means(freq, {"pod": per_bin})
        table[int(rank)] = {
            "mae": float(per_bin.mean()),
            "frobenius": relative_frobenius(recon, hold),
            "bands": {key: float(row["pod"]) for key, row in bands.items()},
        }
    return table


def fit_frequency(
    train: np.ndarray,
    hold: np.ndarray,
    freq: np.ndarray,
    pod_by_rank: dict[int, dict],
    factors: dict,
    init_seed: int,
    device: torch.device,
) -> dict:
    window = WINDOW
    n_loc = int(factors["n_loc"])
    torch.manual_seed(init_seed)
    model = SparseWindowAutoencoder(
        window, n_loc, knn=int(factors["knn"]), block=int(factors["block"]), stride=1,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(factors["lr"]))
    train_w = torch.as_tensor(tile_windows(train, window).reshape(-1, window), dtype=torch.float32, device=device)
    hold_w = torch.as_tensor(tile_windows(hold, window), dtype=torch.float32, device=device)
    rng = np.random.default_rng(init_seed)
    best = float("inf")
    best_epoch = 0
    best_metrics: dict | None = None
    best_state: dict | None = None
    last_metrics: dict | None = None
    rank = int(freq.size // window) * n_loc
    pod = pod_by_rank[rank]
    for epoch in range(EPOCHS):
        model.train()
        order = rng.permutation(train_w.shape[0])
        failed = False
        for start in range(0, order.size, FREQ_BATCH):
            batch = train_w[order[start : start + FREQ_BATCH]]
            loss = (model(batch) - batch).square().mean()
            if not torch.isfinite(loss):
                failed = True
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if failed:
            if best_metrics is None:
                return {**factors, "seed": init_seed, "rank": rank, "failed": True, "epoch": epoch + 1}
            break
        metrics = _frequency_holdout(model, hold_w, freq, pod)
        last_metrics = metrics
        if metrics["mae"] < best:
            best = metrics["mae"]
            best_epoch = epoch + 1
            best_metrics = metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert best_metrics is not None and last_metrics is not None and best_state is not None
    model.load_state_dict(best_state)
    return {
        **factors,
        "seed": init_seed,
        "rank": rank,
        "failed": False,
        "diverged": failed,
        "best_epoch": best_epoch,
        "last_mae": last_metrics["mae"],
        **best_metrics,
    }


@torch.no_grad()
def _frequency_holdout(
    model: SparseWindowAutoencoder,
    hold_w: torch.Tensor,
    freq: np.ndarray,
    pod: dict,
) -> dict:
    count, n_windows, width = hold_w.shape
    flat = hold_w.reshape(-1, width)
    pieces = []
    for start in range(0, flat.shape[0], FREQ_BATCH):
        pieces.append(model(flat[start : start + FREQ_BATCH]))
    pred = torch.cat(pieces, dim=0).reshape(count, n_windows, width)
    err = (pred - hold_w).abs()
    per_bin = err.mean(dim=0).reshape(-1).detach().cpu().numpy()
    numer = torch.linalg.vector_norm(pred - hold_w)
    denom = torch.linalg.vector_norm(hold_w).clamp_min(1e-12)
    frobenius = float(numer / denom)
    return spectral_metrics(per_bin, frobenius, freq, pod)


def preload_columns(cache: Cache, pairs: list[tuple[int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack soil columns as (N, 2, H) with zeros past n_active. Those rows stay masked."""
    grouped: dict[int, list[int]] = {}
    for flat, col in pairs:
        grouped.setdefault(int(flat), []).append(int(col))
    blocks: list[torch.Tensor] = []
    heights: list[np.ndarray] = []
    for flat, cols in grouped.items():
        packed = _strip_batch(cache, flat, cols)
        if packed is None:
            continue
        values, _mask, _depth = packed
        height = int(values.shape[2])
        blocks.append(values[0].permute(2, 0, 1).contiguous())
        heights.append(np.full(values.shape[-1], height, dtype=np.int64))
    if not blocks:
        raise RuntimeError("no soil columns to preload")
    height = max(int(block.shape[-1]) for block in blocks)
    count = sum(int(block.shape[0]) for block in blocks)
    values = torch.zeros(count, 2, height, dtype=torch.float32)
    cursor = 0
    for block in blocks:
        width = int(block.shape[0])
        values[cursor : cursor + width, :, : block.shape[-1]] = block
        cursor += width
    return values, torch.from_numpy(np.concatenate(heights))


def _depth_batch(
    values: torch.Tensor, n_active: torch.Tensor, index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    picked = values.index_select(0, index).unsqueeze(-1)
    active = n_active.index_select(0, index)
    rows = torch.arange(values.shape[-1], device=values.device)
    mask = (rows.view(1, -1) < active.view(-1, 1)).to(dtype=picked.dtype).unsqueeze(-1)
    depth = (rows.to(dtype=picked.dtype) / DEPTH_REF_M).view(1, 1, -1, 1).expand(picked.shape[0], 1, -1, 1)
    return picked, mask, depth


def fit_depth(
    train_v: torch.Tensor,
    train_n: torch.Tensor,
    hold_v: torch.Tensor,
    hold_n: torch.Tensor,
    factors: dict,
    init_seed: int,
) -> dict:
    device = train_v.device
    torch.manual_seed(init_seed)
    model = ColumnDepthAutoencoder(2, code_dim=int(factors["code_dim"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(factors["lr"]))
    rng = np.random.default_rng(init_seed)
    count = train_v.shape[0]
    best = float("inf")
    best_epoch = 0
    best_metrics: dict | None = None
    best_state: dict | None = None
    last_metrics: dict | None = None
    for epoch in range(EPOCHS):
        model.train()
        order = torch.as_tensor(rng.permutation(count), device=device, dtype=torch.long)
        failed = False
        for start in range(0, count, DEPTH_BATCH):
            values, mask, depth = _depth_batch(train_v, train_n, order[start : start + DEPTH_BATCH])
            _surface, recon = model(values, mask, depth)
            loss = (recon - values).abs().mul(mask[:, None]).sum() / mask.sum().clamp_min(1.0).mul(recon.shape[1])
            if not torch.isfinite(loss):
                failed = True
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if failed:
            if best_metrics is None:
                return {**factors, "seed": init_seed, "failed": True, "epoch": epoch + 1}
            break
        metrics = depth_holdout(model, hold_v, hold_n)
        last_metrics = metrics
        if metrics["masked_l1"] < best:
            best = metrics["masked_l1"]
            best_epoch = epoch + 1
            best_metrics = {key: value for key, value in metrics.items() if key != "row_mae"}
            best_metrics["row_mae"] = metrics["row_mae"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert best_metrics is not None and last_metrics is not None and best_state is not None
    return {
        **factors,
        "seed": init_seed,
        "failed": False,
        "diverged": failed,
        "best_epoch": best_epoch,
        "last_masked_l1": last_metrics["masked_l1"],
        **best_metrics,
    }


@torch.no_grad()
def depth_holdout(model: ColumnDepthAutoencoder, values: torch.Tensor, n_active: torch.Tensor) -> dict:
    height = int(values.shape[-1])
    ae_row = torch.zeros(height, device=values.device)
    vs_sum = 0.0
    zeta_sum = 0.0
    lin_sum = 0.0
    count = torch.zeros(height, device=values.device)
    weight = 0.0
    for start in range(0, values.shape[0], DEPTH_BATCH):
        index = torch.arange(start, min(start + DEPTH_BATCH, values.shape[0]), device=values.device)
        batch, mask, depth = _depth_batch(values, n_active, index)
        _surface, recon = model(batch, mask, depth)
        linear = _equal_weight(batch, mask)
        cell = mask[:, None]
        err = (recon - batch).abs()
        ae_row += (err * cell).sum(dim=(0, 1, 3))
        count += mask.sum(dim=(0, 2))
        vs_sum += float((err[:, 0] * mask).sum())
        zeta_sum += float((err[:, 1] * mask).sum())
        lin_sum += float(((linear - batch).abs() * cell).sum())
        weight += float(mask.sum())
    scale = max(weight, 1.0)
    row_mae = (ae_row / count.clamp_min(1.0).mul(2)).detach().cpu().tolist()
    row_count = count.detach().cpu().tolist()
    bands = {}
    for lo, hi in DEPTH_EDGES:
        num = 0.0
        den = 0.0
        for i in range(lo, min(hi, height)):
            num += row_mae[i] * row_count[i] * 2.0
            den += row_count[i] * 2.0
        bands[f"band_{lo}_{hi}"] = num / den if den else 0.0
    populated = [
        row_mae[i]
        for i, n in enumerate(row_count)
        if n >= 0.05 * row_count[0]
    ]
    return {
        "masked_l1": float(ae_row.sum() / (scale * 2.0)),
        "vs_l1": vs_sum / scale,
        "zeta_l1": zeta_sum / scale,
        "linear_l1": lin_sum / (scale * 2.0),
        "worst_populated_row": max(populated) if populated else 0.0,
        "row_mae": row_mae,
        "row_count": [float(n) for n in row_count],
        **bands,
    }


def _equal_weight(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Broadcast the equal-weight masked mean. Same smear as a collapse of all ones."""
    numer = (values * mask[:, None]).sum(dim=2, keepdim=True)
    denom = mask.sum(dim=1).clamp_min(1.0).view(-1, 1, 1, 1)
    return numer / denom


def _finite_runs(runs: list[dict]) -> list[dict]:
    return [row for row in runs if not row.get("failed")]


def summarize(runs: list[dict], factors: tuple[str, ...], metrics: tuple[str, ...]) -> dict:
    """Mean and seed-to-seed spread. Lower is better for every metric here."""
    groups: dict[tuple, list[dict]] = {}
    for row in _finite_runs(runs):
        groups.setdefault(tuple(row[name] for name in factors), []).append(row)
    configs = []
    for key, rows in groups.items():
        stats = {}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
            stats[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
            }
        configs.append({
            "factors": {name: key[i] for i, name in enumerate(factors)},
            "n": len(rows),
            "seeds": [int(row["seed"]) for row in rows],
            "best_epoch_mean": float(np.mean([row["best_epoch"] for row in rows])),
            "metrics": stats,
        })
    winners = {}
    for metric in metrics:
        ranked = sorted(configs, key=lambda row: row["metrics"][metric]["mean"])
        if not ranked:
            continue
        winner = ranked[0]
        entry = {
            "factors": winner["factors"],
            "mean": winner["metrics"][metric]["mean"],
            "std": winner["metrics"][metric]["std"],
            "separated": False,
        }
        if len(ranked) > 1:
            runner = ranked[1]
            entry["runner_up"] = runner["factors"]
            entry["runner_mean"] = runner["metrics"][metric]["mean"]
            entry["separated"] = winner["metrics"][metric]["max"] < runner["metrics"][metric]["min"]
        winners[metric] = entry
    ranks = _mean_ranks(groups, factors, metrics)
    return {"configs": configs, "winners": winners, "mean_rank": ranks}


def _mean_ranks(
    groups: dict[tuple, list[dict]], factors: tuple[str, ...], metrics: tuple[str, ...],
) -> dict[str, list[dict]]:
    """Average place of each config across seeds. Place 1 is the lowest error."""
    seeds = sorted({int(row["seed"]) for rows in groups.values() for row in rows})
    out: dict[str, list[dict]] = {}
    for metric in metrics:
        places: dict[tuple, list[float]] = {key: [] for key in groups}
        for seed in seeds:
            scored = []
            for key, rows in groups.items():
                match = [row for row in rows if int(row["seed"]) == seed]
                if match:
                    scored.append((key, float(match[0][metric])))
            scored.sort(key=lambda item: item[1])
            for place, (key, _value) in enumerate(scored, start=1):
                places[key].append(place)
        order = sorted(places, key=lambda key: float(np.mean(places[key])) if places[key] else 1e9)
        out[metric] = [
            {
                "factors": {name: key[i] for i, name in enumerate(factors)},
                "mean_rank": float(np.mean(places[key])),
                "rank_std": float(np.std(places[key], ddof=1)) if len(places[key]) > 1 else 0.0,
            }
            for key in order
        ]
    return out


def _append(path: Path, row: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def _finished(runs: list[dict], factors: dict, seed: int) -> bool:
    return any(
        (not row.get("failed")) and row.get("seed") == seed and all(row.get(key) == value for key, value in factors.items())
        for row in runs
    )


def _read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _run_frequency(curves: np.ndarray, freq: np.ndarray, device: torch.device, log_path: Path) -> list[dict]:
    train_idx, hold_idx = split_rows(curves.shape[0], SPLIT_SEED)
    train, hold = curves[train_idx], curves[hold_idx]
    ranks = tuple(sorted({(freq.size // WINDOW) * int(row["n_loc"]) for row in frequency_configs()}))
    print(f"frequency curves train {train.shape[0]} hold {hold.shape[0]} pod ranks {ranks}", flush=True)
    pod_by_rank = pod_table(train, hold, freq, ranks)
    runs = _read_log(log_path)
    for factors in frequency_configs():
        for seed in SEEDS:
            if _finished(runs, factors, seed):
                continue
            started = time.perf_counter()
            row = fit_frequency(train, hold, freq, pod_by_rank, factors, seed, device)
            row["seconds"] = time.perf_counter() - started
            runs.append(row)
            _append(log_path, row)
            if row.get("failed"):
                print(f"frequency FAILED {factors} seed {seed} epoch {row['epoch']}", flush=True)
            else:
                print(
                    f"frequency n_loc {row['n_loc']} knn {row['knn']} block {row['block']} "
                    f"lr {row['lr']} seed {seed} mae {row['mae']:.4f} high {row['band_5-10']:.4f} "
                    f"gap {row['gap_mae']:+.4f} epoch {row['best_epoch']} ({row['seconds']:.0f}s)",
                    flush=True,
                )
    return runs


def _run_depth(cache: Cache, device: torch.device, log_path: Path) -> list[dict]:
    train_pairs, hold_pairs = column_splits(cache, 0, SPLIT_SEED)
    print(f"preloading columns train {len(train_pairs)} hold {len(hold_pairs)}", flush=True)
    started = time.perf_counter()
    train_v, train_n = preload_columns(cache, train_pairs)
    hold_v, hold_n = preload_columns(cache, hold_pairs)
    train_v, train_n = train_v.to(device), train_n.to(device)
    hold_v, hold_n = hold_v.to(device), hold_n.to(device)
    print(
        f"preloaded train {tuple(train_v.shape)} hold {tuple(hold_v.shape)} in {time.perf_counter() - started:.0f}s",
        flush=True,
    )
    runs = _read_log(log_path)
    for factors in depth_configs():
        for seed in SEEDS:
            if _finished(runs, factors, seed):
                continue
            started = time.perf_counter()
            row = fit_depth(train_v, train_n, hold_v, hold_n, factors, seed)
            row["seconds"] = time.perf_counter() - started
            runs.append(row)
            _append(log_path, row)
            if row.get("failed"):
                print(f"depth FAILED {factors} seed {seed} epoch {row['epoch']}", flush=True)
            else:
                print(
                    f"depth code {row['code_dim']} lr {row['lr']} seed {seed} "
                    f"l1 {row['masked_l1']:.4f} vs {row['vs_l1']:.4f} zeta {row['zeta_l1']:.4f} "
                    f"tail {row['band_100_110']:.4f} epoch {row['best_epoch']} ({row['seconds']:.0f}s)",
                    flush=True,
                )
    return runs


def build_report(freq_runs: list[dict], depth_runs: list[dict]) -> dict:
    return {
        "split_seed": SPLIT_SEED,
        "epochs": EPOCHS,
        "seeds": list(SEEDS),
        "frequency": {
            "reference": FREQ_REFERENCE,
            "window": WINDOW,
            "held_fixed": ["stride 1", "encoder hidden (64, 32)", f"batch {FREQ_BATCH}", "AdamW defaults"],
            "runs": [{key: value for key, value in row.items() if key not in ("row_mae", "row_count")} for row in freq_runs],
            "summary": summarize(freq_runs, ("n_loc", "knn", "block", "lr"), FREQ_METRICS),
        },
        "depth": {
            "held_fixed": [
                f"batch {DEPTH_BATCH} columns shuffled across strips",
                "AdamW defaults",
                "40 epochs, best holdout masked L1",
            ],
            "runs": [
                {key: value for key, value in row.items() if key != "row_count"}
                for row in depth_runs
            ],
            "summary": summarize(depth_runs, ("code_dim", "lr"), DEPTH_METRICS),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    _refuse_shipped(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = Cache(args.cache)
    freq_log = args.out / "frequency.jsonl"
    depth_log = args.out / "depth.jsonl"
    print(f"device {device}", flush=True)
    depth_runs = _run_depth(cache, device, depth_log)
    curves, _groups = _curves(cache, "residual")
    curves = select_curves(curves, 0, SPLIT_SEED)
    freq = np.asarray(cache.freq, dtype=np.float64)
    freq_runs = _run_frequency(curves, freq, device, freq_log)
    report = build_report(freq_runs, depth_runs)
    (args.out / "report.json").write_text(json.dumps(report))
    freq_winner = report["frequency"]["summary"]["winners"].get("mae", {})
    depth_winner = report["depth"]["summary"]["winners"].get("masked_l1", {})
    print(f"frequency mae winner {freq_winner}", flush=True)
    print(f"depth l1 winner {depth_winner}", flush=True)


if __name__ == "__main__":
    main()
