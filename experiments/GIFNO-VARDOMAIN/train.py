#!/usr/bin/env python3
"""Train one metric LOGLO member.

The network outputs the residual g, or log|TF| when --target direct. The column
1D prior is added outside the network. Stage 0 is the 500 m runs. Stage 1 draws
IID windows from 500 m to 1500 m. Stage 2 replays IID and adds the manifold
interiors, whose Rayleigh pair stays (f1, 3 f1). Half of the manifold slots
are the near-1D sections, a repeated column. Checkpoints stay under checkpoints/vardomain/.

    python train.py --stage 0 --target residual --seed 0 --cache ../../data/vardomain/cache
    python train.py --stage 2 --init ../../checkpoints/vardomain/metric_s0 --cache ...
    python train.py --smoke --cache ...

``--select-harness-sqrt`` is off by default. It does not change ``--loss`` or
the selector that produced P128n128.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import (
    STAGE_UNIFIED, KIND_RUN, Cache, WindowSet, _loader_threads, eval_records, manifold_folds,
    manifold_pair_group_ids, training_records,
)
from encoding import CONTEXT_M, ENCODINGS
from evaluate import evaluate, seed_variance
from model import (
    coefficient_scale,
    POD_MODES,
    extend_pod_modes,
    fit_weighted_pod,
    spectral_weight,
    load_vardomain_state,
    model_from_config,
    sample_grid,
)
from sampling import NEAR_1D_SHARE, QuotaBatchSampler, batches_per_epoch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "checkpoints" / "vardomain"
BLOCKED_OUT = ("GIFNO-FDO-XT-LOGLO-POD", "DeepONet-Residual/checkpoints", "tier2_pod64_full7680")
POD_CURVES_MAX = 80_000
POINTWISE_FLOOR = 0.90
AGREE_FLOOR = 0.0136
# Manifold validation is flat near 0.05. A new checkpoint has to move the
# 3-epoch mean by more than the seed-to-seed scatter.
SELECT_WINDOW = 3
SELECT_DELTA = 1e-3
COSINE_EPOCHS = 60
PROBE_M = 32


def _parse_stage(text: str) -> int:
    if str(text) == "unified":
        return STAGE_UNIFIED
    if str(text) in {"0", "1", "2"}:
        return int(text)
    raise argparse.ArgumentTypeError("stage must be 0, 1, 2, or unified")


def _assert_isolated(out: Path) -> None:
    text = str(out.resolve())
    for token in BLOCKED_OUT:
        if token in text:
            raise SystemExit(f"refusing to write into a shipped model path: {out}")


def _curves(cache: Cache, target: str, limit_runs: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Training-split IID curves of the POD target, and a group id per station."""
    idx = cache.runs("train", ("iid",))
    if limit_runs is not None and idx.size > limit_runs:
        idx = idx[:limit_runs]
    blocks, groups = [], []
    for flat in idx:
        camp, j = cache.campaign(flat)
        log_tf = np.asarray(camp.log_tf[j], dtype=np.float64)
        prior = np.asarray(camp.log_tf1d[j], dtype=np.float64)
        block = log_tf - prior if target == "residual" else log_tf
        blocks.append(block)
        stations = np.arange(block.shape[0], dtype=np.int64)
        groups.append(int(camp.group[j]) * 1000 + stations)
    if not blocks:
        raise RuntimeError("no IID training runs in the cache")
    return np.concatenate(blocks), np.concatenate(groups)


def balanced_curves(
    cache: Cache, target: str, per_family: int, seed: int,
    exclude_group_ids: set[str] | None = None, iid_groups: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """IID and manifold curves for one POD basis. Dipping and three-layer stay out."""
    rng = np.random.default_rng(seed)
    blocks, groups = [], []
    gid = 0
    for family in ("iid",):
        idx = np.asarray(cache.runs("train", (family,)))
        if idx.size == 0:
            continue
        take = idx if idx.size <= per_family else rng.choice(idx, per_family, replace=False)
        for flat in take:
            camp, j = cache.campaign(int(flat))
            if iid_groups is not None and int(camp.group[j]) not in iid_groups:
                continue
            log_tf = np.asarray(camp.log_tf[j], dtype=np.float64)
            prior = np.asarray(camp.log_tf1d[j], dtype=np.float64)
            block = log_tf - prior if target == "residual" else log_tf
            step = max(int(block.shape[0] // 4), 1)
            block = block[::step]
            blocks.append(block)
            groups.append(np.full(block.shape[0], gid, dtype=np.int64))
            gid += 1
    sections = section_curves(cache, target, exclude_group_ids)
    if sections.size:
        blocks.append(sections)
        groups.append(np.arange(sections.shape[0], dtype=np.int64) + gid * 1000)
    if not blocks:
        raise RuntimeError("unified POD found no training curves")
    return np.concatenate(blocks), np.concatenate(groups)


def retained_fraction(
    curves: np.ndarray, groups: np.ndarray, modes: np.ndarray, mean: np.ndarray, freq: np.ndarray,
    weight: np.ndarray | None = None,
) -> float:
    """Share of within-seed variance above 0.4 Hz kept by the POD."""
    band = (freq >= 0.4) & (freq <= 10.0)
    g = curves - mean
    if weight is None:
        resid = g - (g @ modes.T) @ modes
    else:
        y = g * weight
        resid = g - ((y @ modes.T) @ modes) / weight
    num = den = 0.0
    for gid in np.unique(groups):
        idx = np.flatnonzero(groups == gid)
        if idx.size < 2:
            continue
        d = g[idx][:, band]
        d = d - d.mean(axis=0)
        r = resid[idx][:, band]
        r = r - r.mean(axis=0)
        den += float((d * d).sum())
        num += float((r * r).sum())
    return 1.0 if den == 0 else 1.0 - num / den


def section_curves(cache: Cache, target: str, exclude_group_ids: set[str] | None = None) -> np.ndarray:
    rows = []
    for j in cache.section_indices("train"):
        sec = cache.sections[j]
        if exclude_group_ids and sec.group_id in exclude_group_ids:
            continue
        block = sec.log_tf - sec.log_tf1d if target == "residual" else sec.log_tf
        # training stations only: every 20 m, inside the margin
        x = sec.x_m.astype(np.float64)
        keep = (x >= 5.0) & (x <= sec.lx - 5.0) & np.isclose(np.mod(x, 20.0), 0.0, atol=0.2)
        if keep.any():
            rows.append(np.asarray(block[keep], dtype=np.float64))
    if not rows:
        return np.zeros((0, cache.freq.size))
    curves = np.concatenate(rows)
    if curves.shape[0] > POD_CURVES_MAX:
        curves = curves[np.random.default_rng(0).choice(curves.shape[0], POD_CURVES_MAX, replace=False)]
    return curves


def weighted_l1(
    pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, freq: torch.Tensor, width: torch.Tensor,
    freq_weight: torch.Tensor | None = None,
    point_weight: torch.Tensor | None = None,
    low_band_scale: float = 2.0,
) -> torch.Tensor:
    """L1 on the log curve. Windows of 1000 m and wider count 0.1–0.4 Hz twice.

    ``freq_weight`` is w(f) = sqrt(f / 1 Hz) for a basis fit with that weight.
    ``point_weight`` is a per-station, per-frequency factor such as a vRBA weight.
    Validation still reports the unweighted log-L1.
    """
    low = (freq >= 0.1) & (freq <= 0.4)
    boost = torch.where(width >= 1000, torch.full_like(width, float(low_band_scale)), torch.ones_like(width))
    w = valid.to(pred.dtype)[:, :, None] * torch.ones(freq.shape[0], device=pred.device)[None, None, :]
    w = w * torch.where(low.view(1, 1, -1), boost[:, None, None], torch.ones((), device=pred.device))
    if freq_weight is not None:
        w = w * freq_weight.to(dtype=pred.dtype).view(1, 1, -1)
    if point_weight is not None:
        w = w * point_weight.to(dtype=pred.dtype)
    err = (pred - target).abs()
    return (err * w).sum() / w.sum().clamp_min(1.0)


def pair_l1(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, info: torch.Tensor) -> torch.Tensor:
    """L1 of seed-to-seed differences, for rows that share family, group, width and start."""
    keys: dict[tuple, list[int]] = {}
    info_c = info.detach().cpu().numpy()
    for i in range(info_c.shape[0]):
        keys.setdefault((int(info_c[i, 4]), int(info_c[i, 5]), int(info_c[i, 2]), int(info_c[i, 3])), []).append(i)
    total = pred.new_zeros(())
    n = 0
    for idxs in keys.values():
        if len(idxs) != 2:
            continue
        a, b = idxs
        mask = valid[a] & valid[b]
        if int(mask.sum()) == 0:
            continue
        total = total + (pred[a, mask] - pred[b, mask] - (target[a, mask] - target[b, mask])).abs().mean()
        n += 1
    if n == 0:
        return pred.new_zeros(())
    return total / n


def _family_mean(res: dict, families: tuple[str, ...]) -> float:
    parts = []
    for fam in families:
        value = family_log_l1(res, fam)
        if value is not None:
            parts.append(value)
    return float(np.mean(parts)) if parts else float("inf")


def accept_improvement(smooth: float, best: float, *, delta: float) -> bool:
    """True when the smoothed score beats the saved one by more than ``delta``."""
    return bool(np.isfinite(smooth) and (best - smooth) >= delta - 1e-12)


def selection_score(res: dict, stage: int) -> float:
    """Score used to keep best_model.pt and to stop early.

    Unified training selects on the manifold validation split only. IID,
    dipping, three-layer, and the blind manifold are logged and do not
    enter this score. The full frequency band is used. The low band is reported
    beside it. ``--select-harness-sqrt`` does not change this function.
    """
    if stage == STAGE_UNIFIED:
        return _family_mean(res, ("manifold",))
    bins = res.get("bins", {})
    if stage >= 2:
        keys = [k for k in bins if k.startswith("manifold_w")]
    else:
        keys = [k for k in bins if k.startswith("iid_w")]
    if not keys:
        keys = list(bins)
    vals = [bins[k]["log_l1"] for k in keys]
    return float(np.mean(vals)) if vals else float("inf")


def _selector_families(stage: int, protocol: str) -> tuple[str, ...]:
    """Families that enter the checkpoint score. Blind bins are not in this list."""

    if protocol in {"p128", "audit"}:
        return ("iid", "manifold")
    if stage == STAGE_UNIFIED or stage >= 2:
        return ("manifold",)
    return ("iid",)


def _bin_mean(res: dict, family: str, key: str) -> float | None:
    vals = [
        m[key]
        for name, m in res.get("bins", {}).items()
        if name.startswith(family + "_w") and key in m
    ]
    return float(np.mean(vals)) if vals else None


def harness_sqrt_selection_score(res: dict, stage: int, *, protocol: str) -> float:
    """Opt-in score. Same families as the active selector, harness √f overall log-L1.

    The value is ``log_l1_overall_sqrt`` from ``evaluate.station_metrics``.
    That number is the harness overall band with ``model.spectral_weight``.
    Blind keys do not start with ``manifold_w`` or ``iid_w``.
    """

    parts = []
    for family in _selector_families(stage, protocol):
        value = _bin_mean(res, family, "log_l1_overall_sqrt")
        if value is not None:
            parts.append(value)
    return float(np.mean(parts)) if parts else float("inf")


def epoch_selection_score(res: dict, stage: int, *, protocol: str, harness_sqrt: bool) -> float:
    """Checkpoint score for one epoch.

    ``harness_sqrt`` is true only when ``--select-harness-sqrt`` is passed.
    Otherwise protocol ``p128`` keeps the unweighted IID and manifold mean,
    and every other protocol keeps ``selection_score``.
    """

    if harness_sqrt:
        return harness_sqrt_selection_score(res, stage, protocol=protocol)
    if protocol in {"p128", "audit"}:
        return _family_mean(res, ("iid", "manifold"))
    return selection_score(res, stage)


def moving_mean(values: list[float], window: int) -> float:
    tail = [v for v in values[-window:] if np.isfinite(v)]
    return float(np.mean(tail)) if tail else float("inf")


def _objective(args, model, pred, target, prior, valid, info, freq, freq_weight, vrba_weights, epoch):
    """The selected loss. Depth reconstruction is added for the column autoencoder."""
    from spectral_loss import hybrid_loss, linear_tf, rel_l1, spectral_binned_loss
    from vrba import grad_norm_balance, weighted_residual

    residual = args.target == "residual"
    width = info[:, 2].to(pred.dtype)
    if args.loss == "log_l1":
        loss = weighted_l1(
            pred, target, valid, freq, width, freq_weight,
            low_band_scale=float(getattr(args, "low_band_scale", 2.0)),
        )
    elif args.loss == "rel_l1":
        loss = rel_l1(linear_tf(pred, prior, residual=residual), linear_tf(target, prior, residual=residual), valid)
    elif args.loss == "hybrid_rel_log":
        loss, _parts = hybrid_loss(
            pred, target, prior, valid, freq, width, freq_weight, residual=residual,
            alpha=args.alpha_rel, beta=args.beta_modal, spectral=args.spectral_weight_loss, log_l1_fn=weighted_l1,
        )
    elif args.loss == "vrba_log_l1":
        lam = vrba_weights(pred - target, valid, epoch)
        loss = weighted_l1(pred, target, valid, freq, width, freq_weight, point_weight=lam)
    else:
        data = pred - target
        lam = vrba_weights(data, valid, epoch)
        weighted = weighted_residual(data, valid, lam)
        _hybrid, parts = hybrid_loss(
            pred, target, prior, valid, freq, width, freq_weight, residual=residual,
            alpha=args.alpha_rel, beta=args.beta_modal, spectral=args.spectral_weight_loss, log_l1_fn=weighted_l1,
        )
        bundle = {"vrba": weighted, "rel": parts["rel_l1"], "modal": parts["modal"], "spectral": parts["spectral"]}
        if args.pair_weight > 0:
            bundle["pair"] = pair_l1(pred, target, valid, info)
        loss = grad_norm_balance(bundle, [p for p in model.parameters() if p.requires_grad])
    if args.loss != "vrba_hybrid" and args.pair_weight > 0:
        loss = loss + args.pair_weight * pair_l1(pred, target, valid, info)
    if args.depth_ae and args.recon_weight > 0:
        loss = loss + args.recon_weight * model.reconstruction_l1()
    return loss


def _log_wandb(wb, epoch: int, row: list, res: dict, out: Path, smooth: float | None = None) -> None:
    if wb is None:
        return
    payload = {
        "epoch": epoch, "train_loss": row[1], "val_score": row[2], "iid_log_l1": row[3],
        "val_low": row[11], "val_mid": row[12], "val_high": row[13], "agreement": row[8],
    }
    if smooth is not None and np.isfinite(smooth):
        payload["select_smooth"] = smooth
    logged = (
        "iid", "ood_dipping", "ood_three_layer", "manifold",
        "manifold_near", "manifold_lateral", "manifold_blind",
        "manifold_blind_near", "manifold_blind_lateral",
    )
    for fam in logged:
        block = res.get(f"{fam}_mean")
        if not block:
            continue
        for key in ("log_l1", "log_l1_low", "log_l1_mid", "rel_l1", "pearson", "pearson_log", "spearman", "ccc_log", "m1_amp_log", "m2_amp_log", "m1_rel_lin"):
            if key in block and np.isfinite(block[key]):
                payload[f"{fam}/{key}"] = block[key]
    wb.log(payload)
    if epoch % 10 == 0:
        png = out / f"epoch_{epoch:04d}_metrics.png"
        if png.is_file():
            wb.log({"metrics_png": wb.Image(str(png))})


def family_log_l1(res: dict, family: str) -> float | None:
    vals = [m["log_l1"] for k, m in res.get("bins", {}).items() if k.startswith(family + "_w")]
    return float(np.mean(vals)) if vals else None


def iid_log_l1(res: dict) -> float | None:
    vals = [m["log_l1"] for k, m in res.get("bins", {}).items() if k.startswith("iid_w")]
    return float(np.mean(vals)) if vals else None


def curvature_penalty(model, coeff_grid, latent, query, valid, width) -> torch.Tensor:
    """Mean square second difference of g on a 32 m run, at columns that are not stations."""
    span = (width.to(coeff_grid.device).float() - PROBE_M).clamp(min=0)
    start = CONTEXT_M + torch.rand(coeff_grid.shape[0], device=coeff_grid.device) * span
    cols = start[:, None] + torch.arange(PROBE_M, device=coeff_grid.device, dtype=torch.float32)
    g = model.readout(coeff_grid, latent, cols)
    d2 = g[:, 2:] - 2 * g[:, 1:-1] + g[:, :-2]
    centers = cols[:, 1:-1]
    near = (centers[:, :, None] - query.float()[:, None, :]).abs() < 0.5
    near = near & valid.to(coeff_grid.device)[:, None, :]
    off = ~near.any(-1)
    weight = off.to(g.dtype)[:, :, None]
    return (d2.square() * weight).sum() / weight.sum().clamp_min(1.0)


@torch.no_grad()
def median_cancellation(model, cache: Cache, encoding: str, device, stage: int) -> float:
    """Median gross/|net| of the POD sum at one validation station."""
    records = eval_records(cache, "val", iid_limit=1, positions=1, families=("iid",))
    if not records:
        return float("nan")
    window = WindowSet(cache, encoding, records[:1], stage, seed=0, mirror=False)
    x, query, _tf, _prior, valid, _info, _xa = window[0]
    coeff_grid, _latent, _recon = model.encode(x.unsqueeze(0).to(device))
    if coeff_grid is None:
        return float("nan")
    coeff = sample_grid(coeff_grid, query.unsqueeze(0).to(device))
    idx = np.flatnonzero(valid.numpy())
    if idx.size == 0:
        return float("nan")
    parts = coeff[0, int(idx[idx.size // 2])].float().cpu()[:, None] * model.pod_modes.float().cpu()
    net = parts.sum(0)
    gross = parts.abs().sum(0)
    return float((gross / net.abs().clamp_min(1e-8)).median())


def _basis_floor(
    curves: np.ndarray, modes: np.ndarray, mean: np.ndarray, weight: np.ndarray, freq: np.ndarray,
) -> tuple[float, dict]:
    """Mean absolute reconstruction error of the POD on the curves it was fit to."""
    centered = np.asarray(curves, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    coeff = (centered * np.asarray(weight, dtype=np.float64)) @ np.asarray(modes, dtype=np.float64).T
    recon = np.asarray(mean, dtype=np.float64) + coeff @ np.asarray(modes, dtype=np.float64) / np.asarray(weight, dtype=np.float64)
    err = np.abs(recon - np.asarray(curves, dtype=np.float64))
    bands = {}
    for name, lo, hi in (("low", 0.1, 0.4), ("mid", 0.4, 2.5), ("high", 2.5, 10.0), ("band_5_10", 5.0, 10.0)):
        mask = (freq >= lo) & (freq <= hi)
        bands[name] = float(err[:, mask].mean()) if mask.any() else None
    return float(err.mean()), bands


def _band_val(res: dict) -> tuple[float, float, float]:
    """IID validation log-L1 in the low, mid, and high bands."""
    bins = res.get("bins", {})
    keys = [key for key in bins if key.startswith("iid_w")]

    def mean(name: str) -> float:
        vals = [bins[key][name] for key in keys if name in bins[key]]
        return float(np.mean(vals)) if vals else float("nan")

    return mean("log_l1_low"), mean("log_l1_mid"), mean("log_l1_high")


def load_frozen_basis(path: Path) -> dict:
    """Load POD, PCA, and coefficient scale. Never returns network weights."""
    path = Path(path)
    modes = np.load(path / "pod_modes.npy")
    mean = np.load(path / "pod_mean.npy")
    weight_path = path / "pod_weight.npy"
    weight = np.load(weight_path) if weight_path.is_file() else np.ones(modes.shape[1], np.float32)
    scale_path = path / "pod_coeff_scale.npy"
    scale = np.load(scale_path) if scale_path.is_file() else None
    if scale is None and (path / "best_model.pt").is_file():
        blob = torch.load(path / "best_model.pt", map_location="cpu", weights_only=False)
        tensor = (blob.get("state_dict") or {}).get("pod_coeff_scale")
        if tensor is not None:
            scale = np.asarray(tensor.detach().cpu().numpy(), dtype=np.float32)
    pca = None
    if (path / "pca_mean.npy").is_file() and (path / "pca_components.npy").is_file():
        pca = (np.load(path / "pca_mean.npy"), np.load(path / "pca_components.npy"))
    return {"modes": modes, "mean": mean, "weight": np.asarray(weight, dtype=np.float32), "scale": scale, "pca": pca}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoding", choices=ENCODINGS, default="metric")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--stage", type=_parse_stage, default=0, help="0, 1, 2, or unified")
    ap.add_argument("--target", choices=("residual", "direct"), default="residual")
    ap.add_argument("--pair-weight", type=float, default=0.0)
    ap.add_argument(
        "--near-1d-share", type=float, default=None,
        help="stage 2: fraction of manifold slots drawn from repeated-column sections (default 0.5)",
    )
    ap.add_argument("--init", type=Path, default=None)
    ap.add_argument("--iid-baseline", type=float, default=None, help="stage 2 stops if IID val L1 exceeds (1 + --iid-rise) times this")
    ap.add_argument("--iid-rise", type=float, default=0.10, help="allowed fractional rise of IID val L1 above the stage-1 baseline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iid-group-frac", type=float, default=1.0, help="fraction of IID geology groups kept in training")
    ap.add_argument("--manifold-group-frac", type=float, default=1.0, help="fraction of manifold groups kept; every width keeps at least one section")
    ap.add_argument("--iid-seeds-per-group", type=int, default=None, help="RF seeds kept inside each kept IID group")
    ap.add_argument("--fixed-batches", type=int, default=None, help="optimizer steps per epoch, independent of the subset size")
    ap.add_argument("--fixed-basis-dir", type=Path, default=None, help="reuse this run's POD, column PCA, and coefficient scale")
    ap.add_argument("--protocol", choices=("unified", "p128", "audit"), default="unified", help="audit selects the pure sqrt(f) overall log-L1; p128 selects on IID+manifold")
    ap.add_argument("--subset-seed", type=int, default=None, help="geology-subset permutation; defaults to --seed")
    ap.add_argument("--model-seed", type=int, default=None, help="parameter initialization; defaults to --seed")
    ap.add_argument("--sampler-seed", type=int, default=None, help="window and batch order; defaults to --seed")
    ap.add_argument("--skip-test", action="store_true", help="do not evaluate the test split")
    ap.add_argument("--train-splits", default="train", help="comma-separated splits; test, blind, and external are refused")
    ap.add_argument("--no-early-stop", action="store_true", help="train a fixed epoch count and keep the last epoch")
    ap.add_argument("--low-band-scale", type=float, default=None, help="extra weight on 0.1-0.4 Hz for windows of at least 1000 m; audit uses 1")
    ap.add_argument("--representation", choices=("pca", "dual_pod", "coupled_pod"), default="pca")
    ap.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=None, help="default on except for --smoke")
    ap.add_argument("--fast-val", action="store_true", help="log-L1 only during the epoch, and the bf16 pass only when the score would be saved")
    ap.add_argument("--cv-fold", type=int, default=None, help="held-out fold of the 22 paired manifold groups")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--cv-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None, help="stop after this many epochs without a better selection score")
    ap.add_argument("--lr-patience", type=int, default=13, help="ReduceLROnPlateau patience, independent of early stopping")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--pod-modes", type=int, default=POD_MODES)
    ap.add_argument("--narrow", type=int, default=32)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--modes-x", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--jitter", action=argparse.BooleanOptionalAction, default=None, help="shift each training window by a random sub-station offset")
    ap.add_argument("--curvature", type=float, default=0.0, help="weight on the off-station second difference of g")
    ap.add_argument("--whiten", action=argparse.BooleanOptionalAction, default=None, help="predict POD coefficients divided by their training standard deviation")
    ap.add_argument("--hfp-kernel", type=int, default=4)
    ap.add_argument("--hfp-stride", type=int, default=4)
    ap.add_argument("--local-min-wavelength", type=float, default=0.0, help="drop local-patch modes shorter than this many metres")
    ap.add_argument("--depth-pca", action="store_true", help="frozen PCA of the soil column in place of the learned depth collapse")
    ap.add_argument("--pca-rank", type=int, default=16, help="components kept by the frozen column PCA")
    ap.add_argument("--depth-ae", action="store_true", help="mask-gated column autoencoder in place of the linear depth collapse")
    ap.add_argument("--recon-weight", type=float, default=0.1, help="weight on the masked depth reconstruction")
    ap.add_argument("--readout", choices=("pod", "nonlinear"), default="pod")
    ap.add_argument("--loss", choices=("log_l1", "rel_l1", "hybrid_rel_log", "vrba_log_l1", "vrba_hybrid"), default="log_l1")
    ap.add_argument(
        "--select-harness-sqrt",
        action="store_true",
        help=(
            "Opt-in checkpoint score: harness overall 0.1-10 Hz absolute g log-L1 "
            "with the POD sqrt(f) weight, on the same families as the active selector. "
            "Default off. Unified stays on unweighted manifold validation log-L1. "
            "Protocol p128 stays on the unweighted IID and manifold mean."
        ),
    )
    ap.add_argument("--vrba-potential", choices=("quadratic", "exponential"), default="quadratic")
    ap.add_argument("--translation-equivariant", action="store_true")
    ap.add_argument("--pca-joint", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--deltaphi", action="store_true")
    ap.add_argument("--contact-features", action="store_true", help="add whitened bedrock-contact channels at the lift")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="vardomain-v2")
    ap.add_argument("--wandb-mode", choices=("online", "offline"), default="offline")
    ap.add_argument("--alpha-rel", type=float, default=1.0)
    ap.add_argument("--beta-modal", type=float, default=1.0)
    ap.add_argument("--spectral-weight-loss", type=float, default=0.5)
    return ap


def _iid_subset(cache: Cache, frac: float, seeds_per_group: int | None, seed: int) -> set[int] | None:
    if frac >= 1.0 and seeds_per_group is None:
        return None
    grouped: dict[int, list[int]] = {}
    for flat in cache.runs("train", ("iid",)):
        camp, j = cache.campaign(int(flat))
        grouped.setdefault(int(camp.group[j]), []).append(int(flat))
    from data import choose_iid_flats

    return choose_iid_flats(grouped, frac, seeds_per_group, seed)


def main() -> None:
    args = build_parser().parse_args()
    model_seed = args.seed if args.model_seed is None else int(args.model_seed)
    subset_seed = args.seed if args.subset_seed is None else int(args.subset_seed)
    sampler_seed = args.seed if args.sampler_seed is None else int(args.sampler_seed)
    if args.protocol == "audit":
        args.select_harness_sqrt = True
        if args.low_band_scale is None:
            args.low_band_scale = 1.0
    if args.low_band_scale is None:
        args.low_band_scale = 2.0
    if args.mirror is None:
        args.mirror = not args.smoke
    if args.stage == STAGE_UNIFIED:
        if args.jitter is None:
            args.jitter = True
        if args.whiten is None:
            args.whiten = True
        if args.local_min_wavelength == 0.0:
            args.local_min_wavelength = 10.0
        if args.pca_joint is None:
            args.pca_joint = True
    else:
        args.jitter = bool(args.jitter)
        args.whiten = bool(args.whiten)
        args.pca_joint = bool(args.pca_joint)
    if args.patience is None:
        args.patience = 15 if args.stage == STAGE_UNIFIED else 40

    if args.representation != "pca":
        args.depth_pca = True
    if args.depth_pca and (args.depth_ae or args.readout != "pod" or args.target != "residual"):
        raise SystemExit("frozen PCA-16 is the depth code of a residual POD model")
    stage_name = "unified" if args.stage == STAGE_UNIFIED else str(args.stage)
    out = args.out or DEFAULT_ROOT / f"stage{stage_name}_{args.target}_s{args.seed}"
    _assert_isolated(out)
    if (out / "best_model.pt").exists() and not args.overwrite:
        raise SystemExit(f"{out}/best_model.pt exists; pass --overwrite or a new --out")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = not args.no_amp and device.type == "cuda"
    epochs = 2 if args.smoke else (args.epochs if args.epochs is not None else {0: 200, 1: 80, 2: 120, STAGE_UNIFIED: 120}[args.stage])

    cache = Cache(args.cache)
    holdout_ids: list[str] = []
    if args.cv_fold is not None:
        folds = manifold_folds(manifold_pair_group_ids(cache), args.cv_folds, args.cv_seed)
        if not 0 <= int(args.cv_fold) < len(folds):
            raise SystemExit(f"--cv-fold {args.cv_fold} is outside 0..{len(folds) - 1}")
        holdout_ids = list(folds[int(args.cv_fold)])
        print(f"manifold CV fold {args.cv_fold}: holding out {holdout_ids}", flush=True)
    init_cfg = {}
    init_ck = None
    frozen = None
    fixed_basis = args.fixed_basis_dir
    held_out = set(holdout_ids)
    iid_keep = _iid_subset(cache, args.iid_group_frac, args.iid_seeds_per_group, subset_seed)
    if fixed_basis is not None and args.init is not None:
        raise SystemExit("--fixed-basis-dir and --init load two different bases")
    if args.init:
        init_ck = torch.load(args.init / "best_model.pt", map_location="cpu", weights_only=False)
        init_cfg = init_ck["config"]
        modes = np.load(args.init / "pod_modes.npy")
        mean = np.load(args.init / "pod_mean.npy")
    elif fixed_basis is not None:
        frozen = load_frozen_basis(fixed_basis)
        modes = frozen["modes"]
        mean = frozen["mean"]
        curves = groups = None
        freq_weight = spectral_weight(cache.freq)
        fraction_path = fixed_basis / "pod_fraction.json"
        fraction = json.loads(fraction_path.read_text()).get("above_0.4_hz", 1.0) if fraction_path.is_file() else 1.0
        basis_floor, basis_bands = 0.0, {}
        print(f"frozen preprocessing from {fixed_basis} ({modes.shape[0]} modes); network weights were not loaded", flush=True)
    else:
        if args.stage == STAGE_UNIFIED:
            curves, groups = balanced_curves(
                cache, args.target, 8 if args.smoke else 512, subset_seed,
                exclude_group_ids=held_out, iid_groups=None if iid_keep is None else set(
                    int(cache.campaign(int(flat))[0].group[cache.campaign(int(flat))[1]]) for flat in iid_keep
                ),
            )
        else:
            curves, groups = _curves(cache, args.target, limit_runs=64 if args.smoke else None)
        if curves.shape[0] > POD_CURVES_MAX:
            take = np.random.default_rng(subset_seed).choice(curves.shape[0], POD_CURVES_MAX, replace=False)
            curves, groups = curves[take], groups[take]
        freq_weight = spectral_weight(cache.freq)
        modes, mean = fit_weighted_pod(curves, min(args.pod_modes, curves.shape[0] - 1), freq_weight)
        fraction = retained_fraction(curves, groups, modes, mean, cache.freq, freq_weight)
        print(f"POD keeps {fraction:.3f} of within-seed variance above 0.4 Hz", flush=True)
        basis_floor, basis_bands = _basis_floor(curves, modes, mean, freq_weight, cache.freq)
        print(
            f"POD basis floor {basis_floor:.4f} "
            f"(2.5-10 Hz {basis_bands['high']:.4f}, 5-10 Hz {basis_bands['band_5_10']:.4f})",
            flush=True,
        )
    else_pointwise = bool(init_cfg.get("pointwise", False))
    if args.init is None and fixed_basis is None:
        else_pointwise = fraction < POINTWISE_FLOOR
        (out / "pod_fraction.json").write_text(json.dumps({
            "above_0.4_hz": fraction,
            "pointwise": else_pointwise,
            "basis_floor": basis_floor,
            "basis_bands": basis_bands,
            "spectral_weight": "sqrt_f",
        }))
    if args.readout == "nonlinear" or args.representation != "pca":
        else_pointwise = False
    if args.init is None and fixed_basis is None:
        pod_weight = freq_weight.astype(np.float32)
    else:
        weight_path = (args.init or fixed_basis) / "pod_weight.npy"
        pod_weight = np.load(weight_path) if weight_path.is_file() else np.ones(modes.shape[1], np.float32)
    np.save(out / "pod_modes.npy", modes)
    np.save(out / "pod_mean.npy", mean)
    np.save(out / "pod_weight.npy", pod_weight)
    pca_mean = pca_components = None
    input_weight = None
    coupled = None
    if args.depth_pca:
        if args.representation == "pca":
            if args.init is None and fixed_basis is None:
                from depth_probe import fit_column_pca
                pca = fit_column_pca(
                    cache, int(args.pca_rank), joint=bool(args.pca_joint), column_stride=20 if args.smoke else 10,
                    exclude_section_groups=held_out,
                    iid_groups=None if iid_keep is None else set(
                        int(cache.campaign(int(flat))[0].group[cache.campaign(int(flat))[1]]) for flat in iid_keep
                    ),
                )
                pca_mean = np.asarray(pca["mean"], dtype=np.float32)
                pca_components = np.asarray(pca["components"], dtype=np.float32)
                print(
                    f"frozen PCA-{int(pca_components.shape[0])} on {pca['runs']} runs"
                    f" and {pca.get('sections', 0)} sections, height {pca['height']}, "
                    f"explained {pca['explained']:.3f}",
                    flush=True,
                )
            elif frozen is not None and frozen["pca"] is not None:
                pca_mean = np.asarray(frozen["pca"][0], dtype=np.float32)
                pca_components = np.asarray(frozen["pca"][1], dtype=np.float32)
            else:
                pca_src = args.init or fixed_basis
                pca_mean = np.load(pca_src / "pca_mean.npy")
                pca_components = np.load(pca_src / "pca_components.npy")
        else:
            from representation import fit_representation_from_cache

            fitted = fit_representation_from_cache(
                cache, args.representation, rank_cap=max(int(args.pca_rank), 64),
                modes=modes, mean=mean, weight=pod_weight, freq=cache.freq,
                exclude_section_groups=held_out, iid_flats=iid_keep, smoke=bool(args.smoke), seed=subset_seed,
            )
            pca_mean = fitted["mean"]
            pca_components = fitted["components"]
            input_weight = fitted["weight"]
            coupled = fitted.get("coupled")
            (out / "representation.json").write_text(json.dumps(fitted["report"], indent=2))
            print(
                f"{args.representation} input rank {pca_components.shape[0]} "
                f"reconstruction {fitted['report']['input_reconstruction']:.4f}",
                flush=True,
            )
        np.save(out / "pca_mean.npy", pca_mean)
        np.save(out / "pca_components.npy", pca_components)
        if input_weight is not None:
            np.save(out / "input_pod_weight.npy", input_weight)
    contact_mean = contact_std = None
    if args.contact_features:
        from model import fit_contact_whitening

        contact_src = fixed_basis if fixed_basis is not None and (fixed_basis / "contact_mean.npy").is_file() else None
        if contact_src is not None:
            contact_mean = np.load(contact_src / "contact_mean.npy")
            contact_std = np.load(contact_src / "contact_std.npy")
        else:
            contact_mean, contact_std = fit_contact_whitening(cache, column_stride=20 if args.smoke else 10)
        np.save(out / "contact_mean.npy", contact_mean)
        np.save(out / "contact_std.npy", contact_std)
        print(
            f"contact whitening mean {np.array2string(contact_mean, precision=3)} "
            f"std {np.array2string(contact_std, precision=3)}",
            flush=True,
        )

    cfg = {
        "encoding": args.encoding, "seed": args.seed, "stage": args.stage, "target": args.target,
        "pair_weight": args.pair_weight, "modes_x": args.modes_x or (96 if args.encoding == "metric" else 32),
        "narrow": args.narrow if not args.smoke else 8, "latent": args.latent if not args.smoke else 16,
        "layers": args.layers if not args.smoke else 1, "pod_modes": int(modes.shape[0]),
        "pointwise": else_pointwise, "lr": args.lr, "wd": args.wd, "batch": args.batch, "amp": amp,
        "cache": str(args.cache.resolve()), "init": str(args.init) if args.init else None,
        "iid_rise": args.iid_rise,
        "near_1d_share": 0.0 if args.stage < 2 else (NEAR_1D_SHARE if args.near_1d_share is None else args.near_1d_share),
        "query_dtype": "float32", "jitter": bool(args.jitter), "curvature": float(args.curvature),
        "hfp_kernel": int(args.hfp_kernel), "hfp_stride": int(args.hfp_stride),
        "local_min_wavelength": float(args.local_min_wavelength), "whiten": bool(args.whiten),
        "agree_floor": AGREE_FLOOR,
        "depth_ae": bool(args.depth_ae), "recon_weight": float(args.recon_weight), "readout": args.readout,
        "depth_pca": bool(args.depth_pca), "pca_rank": int(pca_components.shape[0]) if pca_components is not None else 0,
        "pca_joint": bool(args.pca_joint),
        "translation_equivariant": bool(args.translation_equivariant),
        "contact_features": bool(args.contact_features),
        "contact_mean": None if contact_mean is None else [float(v) for v in contact_mean],
        "contact_std": None if contact_std is None else [float(v) for v in contact_std],
        "loss": args.loss, "select_harness_sqrt": bool(args.select_harness_sqrt),
        "vrba_potential": args.vrba_potential, "deltaphi": bool(args.deltaphi),
        "spectral_weight": None if np.allclose(pod_weight, 1.0) else "sqrt_f",
        "patience": int(args.patience), "lr_patience": int(args.lr_patience),
        "select_families": (
            "iid,manifold:harness_overall_sqrt_f" if args.protocol == "audit"
            else "harness_overall_sqrt_f" if args.select_harness_sqrt
            else "iid,manifold" if args.protocol == "p128"
            else "manifold_val" if args.stage == STAGE_UNIFIED else "legacy"
        ),
        "select_window": 1 if args.protocol in {"p128", "audit"} else (SELECT_WINDOW if args.stage == STAGE_UNIFIED else 1),
        "select_delta": 0.0 if args.protocol in {"p128", "audit"} else (SELECT_DELTA if args.stage == STAGE_UNIFIED else 0.0),
        "lr_schedule": "plateau" if args.protocol in {"p128", "audit"} else ("cosine" if args.stage == STAGE_UNIFIED else "plateau"),
        "cosine_epochs": 0 if args.protocol in {"p128", "audit"} else (COSINE_EPOCHS if args.stage == STAGE_UNIFIED else 0),
        "low_band_scale": float(args.low_band_scale),
        "representation": args.representation,
        "model_seed": int(model_seed),
        "subset_seed": int(subset_seed),
        "sampler_seed": int(sampler_seed),
        "train_splits": str(args.train_splits),
        "no_early_stop": bool(args.no_early_stop),
        "mirror": bool(args.mirror),
        "reported_seed": 0,
        "train_families": "iid,manifold,cov0" if args.stage == STAGE_UNIFIED else "legacy",
        "protocol": args.protocol,
        "iid_group_frac": float(args.iid_group_frac),
        "manifold_group_frac": float(args.manifold_group_frac),
        "iid_seeds_per_group": None if args.iid_seeds_per_group is None else int(args.iid_seeds_per_group),
        "fixed_batches": None if args.fixed_batches is None else int(args.fixed_batches),
        "fixed_basis_dir": str(fixed_basis) if fixed_basis else None,
        "cv_fold": None if args.cv_fold is None else int(args.cv_fold),
        "cv_folds": int(args.cv_folds),
        "manifold_holdout": sorted(holdout_ids),
    }
    model = model_from_config(cfg, modes, mean, pca_mean, pca_components, input_weight=input_weight).to(device)
    if coupled is not None:
        model.attach_coupled(coupled["beta"], coupled["x_mean"], coupled["y_mean"])
    with torch.no_grad():
        model.pod_weight.copy_(torch.as_tensor(pod_weight, dtype=torch.float32, device=device))
    if args.whiten and args.readout == "pod" and frozen is not None and frozen["scale"] is not None:
        scale = np.asarray(frozen["scale"], dtype=np.float32)
        with torch.no_grad():
            model.pod_coeff_scale.copy_(torch.as_tensor(scale, device=device))
    elif args.whiten and args.init is None and args.readout == "pod" and curves is not None:
        scale = coefficient_scale(curves, modes, mean, None if cfg["spectral_weight"] is None else freq_weight)
        with torch.no_grad():
            model.pod_coeff_scale.copy_(torch.as_tensor(scale, device=device))
    if init_ck is not None:
        load_vardomain_state(model, init_ck["state_dict"])
        if args.stage >= 2 and args.stage != STAGE_UNIFIED and not args.smoke and args.readout == "pod":
            extra_src = section_curves(cache, args.target)
            basis_weight = None if cfg["spectral_weight"] is None else np.asarray(pod_weight, dtype=np.float64)
            extended = extend_pod_modes(modes, mean, extra_src, n_extra=16, weight=basis_weight) if extra_src.size else modes
            if extended.shape[0] > modes.shape[0]:
                n_old = int(modes.shape[0])
                model.grow_modes(torch.as_tensor(extended[n_old:], dtype=torch.float32))
                if args.whiten and extra_src.size:
                    scale_new = coefficient_scale(extra_src, extended[n_old:], mean, basis_weight)
                    with torch.no_grad():
                        model.pod_coeff_scale[n_old:] = torch.as_tensor(scale_new, device=device)
                modes = extended
                np.save(out / "pod_modes.npy", modes)
                cfg["pod_modes"] = int(modes.shape[0])
                print(f"appended {extended.shape[0] - init_cfg.get('pod_modes', modes.shape[0])} POD modes", flush=True)
    if model.readout_kind == "nonlinear":
        model.set_frequencies(torch.as_tensor(cache.freq, dtype=torch.float32, device=device))
    np.save(out / "pod_coeff_scale.npy", model.pod_coeff_scale.detach().cpu().numpy())
    cfg["n_params"] = int(sum(p.numel() for p in model.parameters()))
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg), flush=True)

    records = []
    for split_name in [part.strip() for part in str(args.train_splits).split(",") if part.strip()]:
        records.extend(training_records(
            cache, args.stage, split_name, iid_group_frac=args.iid_group_frac,
            iid_seeds_per_group=args.iid_seeds_per_group, seed=subset_seed,
            manifold_group_frac=args.manifold_group_frac,
            manifold_exclude=set(holdout_ids) or None,
            include_pair_val=bool(holdout_ids) and split_name == "train",
        ))
    if args.smoke:
        runs = [r for r in records if r["kind"] == "run"][:24]
        sections = [r for r in records if r["kind"] == "section"][:8]
        records = runs + sections
        draws: dict[int, int] = {}
        for rec in records:
            if rec.get("pair"):
                draws[int(rec["draw"])] = draws.get(int(rec["draw"]), 0) + 1
        for rec in records:
            if rec.get("pair") and draws.get(int(rec["draw"]), 0) < 2:
                rec["pair"] = False
    n_batches = 4 if args.smoke else batches_per_epoch(records, args.batch, args.stage, fixed=args.fixed_batches)
    ds = WindowSet(cache, args.encoding, records, args.stage, sampler_seed, mirror=bool(args.mirror), jitter=args.jitter)
    sampler = QuotaBatchSampler(
        records, args.batch, args.stage, n_batches, sampler_seed, use_pairs=args.pair_weight > 0,
        near_1d_share=cfg["near_1d_share"],
    )
    freq_t = torch.as_tensor(cache.freq, dtype=torch.float32, device=device)
    loss_weight = freq_t.clamp_min(1e-6).sqrt() if cfg["spectral_weight"] else None
    phi_index = None
    if args.deltaphi:
        from deltaphi import index_from_cache

        phi_index = index_from_cache(cache)
        model._phi_index = phi_index
    vrba_weights = None
    if args.loss in ("vrba_hybrid", "vrba_log_l1"):
        from vrba import VRBAWeights

        vrba_weights = VRBAWeights(int(freq_t.shape[0]), potential=args.vrba_potential).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    wb = None
    if args.wandb:
        import wandb

        wb = wandb.init(project=args.wandb_project, mode=args.wandb_mode, config=cfg, dir=str(out))
    use_p128 = args.protocol in {"p128", "audit"}
    if args.stage == STAGE_UNIFIED and not use_p128:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=COSINE_EPOCHS, eta_min=args.lr / 100.0)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(5, args.lr_patience))
    best, best_epoch = float("inf"), -1
    select_hist: list[float] = []
    hist_path = out / "history.csv"
    with hist_path.open("w", newline="") as fh:
        csv.writer(fh).writerow([
            "epoch", "train_log_l1", "val_score", "iid_log_l1", "lr", "seconds", "peak_mem_gb",
            "curvature", "agreement", "cancellation", "val_bf16",
            "val_low", "val_mid", "val_high",
        ])

    workers = 0 if args.smoke else args.workers
    loader_kwargs = {"batch_sampler": sampler, "num_workers": workers, "pin_memory": device.type == "cuda"}
    if workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["worker_init_fn"] = _loader_threads
    loader = DataLoader(ds, **loader_kwargs)
    for epoch in range(epochs):
        ds.set_epoch(epoch)
        sampler.set_epoch(epoch)
        model.train()
        t0, total, n, curv_total = time.time(), 0.0, 0, 0.0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for x, query, log_tf, prior, valid, info, _x_abs in loader:
            x = x.to(device, non_blocking=True)
            query = query.to(device, non_blocking=True)
            log_tf = log_tf.to(device, non_blocking=True)
            prior = prior.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            info = info.to(device, non_blocking=True)
            target = log_tf - prior if args.target == "residual" else log_tf
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                coeff_grid, latent, _recon = model.encode(x)
                pred = model.readout(coeff_grid, latent, query)
            pred = pred.float()
            if phi_index is not None and getattr(model, "deltaphi", None) is not None:
                from deltaphi import retrieve_batch

                g_k, sim = retrieve_batch(phi_index, x, query, pred, info, train=True)
                pred = g_k + model.deltaphi(sample_grid(latent, query), g_k, sim)
            loss = _objective(
                args, model, pred, target, prior, valid, info, freq_t, loss_weight, vrba_weights, epoch,
            )
            curv = pred.new_zeros(())
            if args.curvature > 0:
                curv = curvature_penalty(model, coeff_grid, latent, query, valid, info[:, 2])
                loss = loss + args.curvature * curv
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * x.shape[0]
            curv_total += float(curv.detach()) * x.shape[0]
            n += x.shape[0]
        if args.no_early_stop:
            payload = {
                "state_dict": model.state_dict(), "config": cfg, "epoch": epoch,
                "val_score": None, "selection": "fixed_epoch",
            }
            torch.save(payload, out / "last.pt")
            with hist_path.open("a", newline="") as fh:
                csv.writer(fh).writerow([
                    epoch, total / max(n, 1), "", "", opt.param_groups[0]["lr"],
                    round(time.time() - t0, 1), "", "", "", "", "", "", "", "",
                ])
            print(f"epoch {epoch:4d} train {total / max(n, 1):.4f} fixed-epoch", flush=True)
            continue
        if args.stage == STAGE_UNIFIED:
            val_families = ("iid", "ood_dipping", "ood_three_layer")
            val_limit = 4 if args.smoke else None
        elif args.stage >= 2:
            val_families = ("iid",)
            val_limit = 4 if args.smoke else 32
        else:
            val_families = ("iid", "ood_dipping", "ood_three_layer")
            val_limit = 4 if args.smoke else 32
        extra_sections = ("blind",) if args.stage == STAGE_UNIFIED else ()
        val_workers = 0 if args.smoke else workers
        res = evaluate(
            model, cache, args.encoding, "val", device, batch=args.batch, workers=val_workers,
            iid_limit=val_limit, positions=1, amp=False, target=args.target, families=val_families,
            stage=args.stage, extra_section_splits=extra_sections,
            section_group_ids=set(holdout_ids) or None, core=bool(args.fast_val),
            sqrt_overall=bool(args.select_harness_sqrt),
        )
        score = epoch_selection_score(
            res, args.stage, protocol=args.protocol, harness_sqrt=bool(args.select_harness_sqrt),
        )
        if use_p128:
            would_save = np.isfinite(score) and best - score >= 1e-12
        else:
            window = SELECT_WINDOW if args.stage == STAGE_UNIFIED else 1
            delta = SELECT_DELTA if args.stage == STAGE_UNIFIED else 0.0
            would_save = accept_improvement(moving_mean(select_hist + [float(score)], window), best, delta=delta)
        if amp and (not args.fast_val or would_save):
            res_bf = evaluate(
                model, cache, args.encoding, "val", device, batch=args.batch, workers=val_workers,
                iid_limit=val_limit, positions=1, amp=True, target=args.target, families=val_families,
                stage=args.stage, section_group_ids=set(holdout_ids) or None, core=bool(args.fast_val),
                sqrt_overall=bool(args.select_harness_sqrt),
            )
            score_bf = epoch_selection_score(
                res_bf, args.stage, protocol=args.protocol, harness_sqrt=bool(args.select_harness_sqrt),
            )
        else:
            score_bf = score
        agree = abs(score - score_bf) if np.isfinite(score) and np.isfinite(score_bf) else float("inf")
        iid_l1 = iid_log_l1(res)
        cancel = median_cancellation(model, cache, args.encoding, device, args.stage)
        iid_limit = None if args.iid_baseline is None else (1.0 + args.iid_rise) * args.iid_baseline
        iid_exceeded = (
            args.stage >= 2 and iid_limit is not None and iid_l1 is not None and iid_l1 > iid_limit
        )
        if args.stage == STAGE_UNIFIED and not use_p128:
            sched.step()
        else:
            stepped = _family_mean(res, ("iid", "ood_dipping", "ood_three_layer", "manifold")) if use_p128 else score
            sched.step(stepped if np.isfinite(stepped) else 1.0)
        peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        val_low, val_mid, val_high = _band_val(res)
        row = [
            epoch, total / max(n, 1), score, iid_l1, opt.param_groups[0]["lr"], round(time.time() - t0, 1), round(peak, 2),
            curv_total / max(n, 1), agree, cancel, score_bf, val_low, val_mid, val_high,
        ]
        with hist_path.open("a", newline="") as fh:
            csv.writer(fh).writerow(row)
        dip_l1 = family_log_l1(res, "ood_dipping")
        three_l1 = family_log_l1(res, "ood_three_layer")
        man_l1 = family_log_l1(res, "manifold")
        near_l1 = family_log_l1(res, "manifold_near")
        lateral_l1 = family_log_l1(res, "manifold_lateral")
        blind_l1 = family_log_l1(res, "manifold_blind")
        man_block = res.get("manifold_mean") or {}
        man_low = man_block.get("log_l1_low")
        man_mid = man_block.get("log_l1_mid")
        m1_rel = man_block.get("m1_rel_lin")
        select_hist.append(float(score))
        window = 1 if use_p128 else (SELECT_WINDOW if args.stage == STAGE_UNIFIED else 1)
        delta = 0.0 if use_p128 else (SELECT_DELTA if args.stage == STAGE_UNIFIED else 0.0)
        smooth = score if use_p128 else moving_mean(select_hist, window)

        def _fmt(value: float | None) -> str:
            return "nan" if value is None or not np.isfinite(value) else f"{value:.4f}"

        print(
            f"epoch {epoch:4d} train {row[1]:.4f} val {score:.4f} smooth {smooth:.4f} "
            f"low {val_low:.4f} mid {val_mid:.4f} high {val_high:.4f} "
            f"bf16 {score_bf:.4f} agree {agree:.4f} iid {iid_l1} man {man_l1} "
            f"man_low {_fmt(man_low)} man_mid {_fmt(man_mid)} m1_rel {_fmt(m1_rel)} "
            f"near {_fmt(near_l1)} lateral {_fmt(lateral_l1)} blind {_fmt(blind_l1)} "
            f"dip {dip_l1} three {three_l1} lr {row[4]:.2e} {row[5]}s",
            flush=True,
        )
        if holdout_ids:
            with (out / "cv_epochs.jsonl").open("a") as fh:
                fh.write(json.dumps({
                    "epoch": epoch, "selection": score, "iid": iid_l1, "manifold": man_l1,
                    "lateral": lateral_l1, "holdout": holdout_ids,
                }) + "\n")
        if iid_exceeded:
            print(
                f"IID val L1 {iid_l1:.4f} rose more than {100 * args.iid_rise:.0f}% above {args.iid_baseline:.4f}; stopping stage 2",
                flush=True,
            )
            break
        # IID, dipping, three-layer, and the blind manifold stay in the log.
        if use_p128:
            improved = np.isfinite(score) and agree < AGREE_FLOOR and best - score >= 1e-12
        else:
            improved = accept_improvement(smooth, best, delta=delta) and agree < AGREE_FLOOR
        payload = {
            "state_dict": model.state_dict(), "config": cfg, "epoch": epoch,
            "val_score": score, "select_smooth": smooth, "iid_log_l1": iid_l1,
        }
        torch.save(payload, out / "last.pt")
        _log_wandb(wb, epoch, row, res, out, smooth)
        if improved:
            best, best_epoch = smooth, epoch
            torch.save({**payload, "val": res}, out / "best_model.pt")
        elif best_epoch >= 0 and epoch - best_epoch >= args.patience:
            print(f"early stop at epoch {epoch} (best {best:.4f} at {best_epoch})", flush=True)
            break

    if args.no_early_stop and (out / "last.pt").is_file():
        blob = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        torch.save(blob, out / "best_model.pt")
        best_epoch = int(blob["epoch"])
    if not (out / "best_model.pt").is_file():
        torch.save({"state_dict": model.state_dict(), "config": cfg, "epoch": 0, "val": {}}, out / "best_model.pt")
    ck = torch.load(out / "best_model.pt", map_location=device, weights_only=False)
    load_vardomain_state(model, ck["state_dict"])
    if args.skip_test:
        test = {"skipped": True, "reason": "development run does not open the test split"}
    else:
        test = evaluate(
            model, cache, args.encoding, "test", device, batch=args.batch, workers=0,
            iid_limit=4 if args.smoke else None, positions=1 if args.smoke else 3, amp=False,
            target=args.target, families=("iid", "ood_dipping", "ood_three_layer"), stage=args.stage,
            extra_section_splits=("blind",) if args.stage == STAGE_UNIFIED else (),
        )
    if not args.skip_test and not args.smoke and args.stage == 0:
        test["seed"] = seed_variance(model, cache, device, target=args.target, split="test", amp=False)
    test["best_epoch"], test["best_val"] = best_epoch, best
    (out / "eval_test.json").write_text(json.dumps(test, indent=2))
    print(json.dumps({k: v for k, v in test.items() if k != "bins"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
