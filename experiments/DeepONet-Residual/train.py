"""Train / evaluate signed-residual DeepONet."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import config
import numpy as np
import torch
from model import BranchMode, FieldEncoderKind, build_model
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm, trange
from wandb_util import finish_wandb, init_wandb, log_wandb, summary_wandb

from data import (
    ResidualDeepONetDataset,
    TargetName,
    TrunkSet,
    iid_resample_sampler,
    make_splits,
    stoch_dim,
)


def _build_signed_cache(*args, **kwargs):
    from residual_signed import build_signed_cache

    return build_signed_cache(*args, **kwargs)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _stack_key(ds, key: str) -> torch.Tensor:
    return torch.stack([ds._cache[i][key] for i in range(len(ds))], dim=0)


def fit_norms(train_ds) -> dict[str, torch.Tensor]:
    """Z-score stats from the training split only."""
    stoch = _stack_key(train_ds, "stoch")
    trunk = _stack_key(train_ds, "trunk_y").reshape(
        -1, _stack_key(train_ds, "trunk_y").shape[-1]
    )
    target = _stack_key(train_ds, "target").reshape(-1)
    return {
        "stoch_mean": stoch.mean(0),
        "stoch_std": stoch.std(0).clamp_min(1e-6),
        "trunk_mean": trunk.mean(0),
        "trunk_std": trunk.std(0).clamp_min(1e-6),
        "target_mean": target.mean(),
        "target_std": target.std().clamp_min(1e-6),
    }


def apply_norms(ds, stats: dict[str, torch.Tensor]) -> None:
    for i in range(len(ds)):
        item = ds._cache[i]
        item["stoch"] = (item["stoch"] - stats["stoch_mean"]) / stats["stoch_std"]
        item["trunk_y"] = (item["trunk_y"] - stats["trunk_mean"]) / stats["trunk_std"]
        item["target_raw"] = item["target"].clone()
        item["target"] = (item["target"] - stats["target_mean"]) / stats["target_std"]


def fit_and_apply_norms(train_ds, *other_ds) -> dict[str, torch.Tensor]:
    """Z-score stoch, trunk, and target using the training split only."""
    stats = fit_norms(train_ds)
    apply_norms(train_ds, stats)
    for ds in other_ds:
        apply_norms(ds, stats)
    return stats


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mode: BranchMode,
    stats: dict[str, torch.Tensor],
    *,
    n_rec: int,
    n_freq: int,
) -> dict[str, float]:
    """Primary metrics are on signed R (what TF_1D cannot explain).

    TF reconstruction R² is reported only as a secondary diagnostic, together with
    the lazy TF_1D-only baseline so a near-zero residual cannot look successful.
    """
    model.eval()
    y_all, pred_all, tf2d_all, tf1d_all, tf_hat_all = [], [], [], [], []
    loss_sum, n_batches = 0.0, 0
    crit = nn.SmoothL1Loss(beta=config.SMOOTH_L1_BETA)
    t_mean = stats["target_mean"].to(device)
    t_std = stats["target_std"].to(device)
    for batch in tqdm(loader, desc="eval", leave=False):
        fields = batch["fields"].to(device)
        stoch = batch["stoch"].to(device)
        trunk_y = batch["trunk_y"].to(device)
        target_n = batch["target"].to(device)
        target = batch["target_raw"].to(device)
        tf1d = batch["tf1d"].to(device)
        tf2d = batch["tf2d"].to(device)
        pred_n = _forward(model, fields, stoch, trunk_y, mode)
        loss_sum += float(crit(pred_n, target_n).item())
        n_batches += 1
        pred = pred_n * t_std + t_mean
        y_all.append(target.cpu().numpy().ravel())
        pred_all.append(pred.cpu().numpy().ravel())
        tf1d_all.append(tf1d.cpu().numpy().ravel())
        tf2d_all.append(tf2d.cpu().numpy().ravel())
        tf_hat_all.append((tf1d + pred).cpu().numpy().ravel())
    y = np.concatenate(y_all)
    p = np.concatenate(pred_all)
    tf1d = np.concatenate(tf1d_all)
    tf2d = np.concatenate(tf2d_all)
    tf_hat = np.concatenate(tf_hat_all)

    # Lazy baselines: predict R̂=0 (TF̂ = TF_1D only)
    zero = np.zeros_like(y)
    r2_tf_1d_only = _r2(tf2d, tf1d)
    r2_tf = _r2(tf2d, tf_hat)

    return {
        # --- primary: residual learning ---
        "smooth_l1": loss_sum / max(n_batches, 1),
        "r2_R": _r2(y, p),
        "rel_l2_R": _rel_l2(y, p),
        "pearson_R": _pearson(y, p),
        "pearson_R_freq": _pearson_across_freq(y, p, n_rec=n_rec, n_freq=n_freq),
        "r2_R_zero": _r2(y, zero),  # always ~0 if mean≈0; sanity
        "smooth_l1_R_raw": float(
            np.mean(np.where(np.abs(y) < 1.0, 0.5 * y**2, np.abs(y) - 0.5))
        ),
        "smooth_l1_R_pred": float(
            np.mean(
                np.where(
                    np.abs(y - p) < 1.0,
                    0.5 * (y - p) ** 2,
                    np.abs(y - p) - 0.5,
                )
            )
        ),
        # --- secondary: TF recon (must beat TF_1D-only) ---
        "r2_TF": r2_tf,
        "rel_l2_TF": _rel_l2(tf2d, tf_hat),
        "rel_l2_TF_1d_only": _rel_l2(tf2d, tf1d),
        "r2_TF_1d_only": r2_tf_1d_only,
        "delta_r2_TF": r2_tf - r2_tf_1d_only,
        "pearson_TF_freq": _pearson_across_freq(
            tf2d, tf_hat, n_rec=n_rec, n_freq=n_freq
        ),
        "pearson_TF_1d_only_freq": _pearson_across_freq(
            tf2d, tf1d, n_rec=n_rec, n_freq=n_freq
        ),
    }


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _rel_l2(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.linalg.norm(y - p) / max(np.linalg.norm(y), 1e-12))


def _pearson(y: np.ndarray, p: np.ndarray) -> float:
    y = y.astype(np.float64).ravel()
    p = p.astype(np.float64).ravel()
    if y.size < 2 or y.std() < 1e-12 or p.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(y, p)[0, 1])


def _pearson_across_freq(
    y: np.ndarray,
    p: np.ndarray,
    *,
    n_rec: int,
    n_freq: int,
) -> float:
    """Mean Pearson correlation of spectra along frequency for each (sample, recorder)."""
    y = y.astype(np.float64).ravel()
    p = p.astype(np.float64).ravel()
    q = n_rec * n_freq
    if y.size % q != 0:
        # fallback: global pearson if layout unexpected
        return _pearson(y, p)
    n_s = y.size // q
    Y = y.reshape(n_s, n_rec, n_freq)
    P = p.reshape(n_s, n_rec, n_freq)
    cors: list[float] = []
    for i in range(n_s):
        for r in range(n_rec):
            a, b = Y[i, r], P[i, r]
            if a.std() < 1e-12 or b.std() < 1e-12:
                continue
            cors.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(cors)) if cors else 0.0


def _forward(
    model: nn.Module,
    fields: torch.Tensor,
    stoch: torch.Tensor,
    trunk_y: torch.Tensor,
    mode: BranchMode,
) -> torch.Tensor:
    if mode == "stoch_only":
        return model(None, stoch, trunk_y)
    if mode == "fields_only":
        return model(fields, None, trunk_y)
    return model(fields, stoch, trunk_y)


def _arch_kwargs(
    *,
    residual_fno: bool,
    n_rec: int,
    fno_width: int,
    fno_n_modes: tuple[int, int],
    fno_n_layers: int,
    n_gno_layers: int,
    fno_kind: str = "vanilla",
) -> dict[str, Any]:
    return {
        "residual_fno": bool(residual_fno),
        "n_rec": int(n_rec),
        "fno_width": int(fno_width),
        "fno_n_modes": tuple(fno_n_modes),
        "fno_n_layers": int(fno_n_layers),
        "n_gno_layers": int(n_gno_layers),
        "fno_kind": str(fno_kind),
    }


def _peak_band_mask(freq_s: np.ndarray, n_rec: int) -> torch.Tensor | None:
    lo, hi = config.PEAK_BAND_HZ
    band = (np.asarray(freq_s) >= lo) & (np.asarray(freq_s) <= hi)
    if not np.any(band):
        return None
    mask = np.broadcast_to(band[None, :], (n_rec, len(freq_s))).reshape(-1)
    return torch.from_numpy(np.ascontiguousarray(mask))


def _batch_rel_l2(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    if mask is not None:
        m = mask.to(pred.device)
        pred = pred[:, m]
        target = target[:, m]
    diff = torch.linalg.vector_norm(pred - target, dim=-1)
    den = torch.linalg.vector_norm(target, dim=-1).clamp_min(1e-12)
    return (diff / den).mean()


def _flatten_domain_metrics(
    per_domain: dict[str, dict[str, float]], prefix: str = "test"
) -> dict[str, float]:
    out: dict[str, float] = {}
    for dname, mets in per_domain.items():
        for key, val in mets.items():
            if isinstance(val, (int, float)):
                out[f"{prefix}/{dname}/{key}"] = float(val)
    return out


def train_from_datasets(
    *,
    train_ds,
    val_ds,
    extra_tests: dict[str, Any],
    target: TargetName,
    branch_mode: BranchMode,
    trunk_set: TrunkSet,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    run_name: str,
    patience: int | None = None,
    no_early_stop: bool = False,
    field_encoder: FieldEncoderKind = "conv",
    n_freq_train: int = config.N_FREQ_TRAIN,
    n_freq_eval: int = config.N_FREQ_EVAL,
    init_ckpt: Path | None = None,
    serial_tf1d: bool = False,
    use_wandb: bool = True,
    iid_frac: float | None = None,
    aux_tf_rel_l2: float = 0.0,
    aux_peak_band: float = 0.0,
    residual_fno: bool = False,
    fno_width: int = config.FNO_WIDTH,
    fno_n_modes: tuple[int, int] = config.FNO_N_MODES,
    fno_n_layers: int = config.FNO_N_LAYERS,
    n_gno_layers: int = config.GNO_N_LAYERS,
    fno_kind: str = "vanilla",
    mix_tag: str | None = None,
    lr_sched_factor: float = config.LR_SCHED_FACTOR,
    lr_sched_patience: int = config.LR_SCHED_PATIENCE,
    lr_sched_min: float = config.LR_SCHED_MIN,
    use_lr_sched: bool = True,
) -> dict[str, Any]:
    """Train on already-built datasets; evaluate extra_tests with train-split norms."""
    from torch.utils.data import DataLoader as _DL

    device = _device()
    stats = fit_and_apply_norms(train_ds, val_ds, *extra_tests.values())
    sampler: WeightedRandomSampler | None = None
    if iid_frac is not None and hasattr(train_ds, "domain_names_per_item"):
        sampler = iid_resample_sampler(train_ds, float(iid_frac))
    train_loader = _DL(
        train_ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
    )
    val_loader = _DL(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    n_rec = train_ds.n_rec
    n_freq = len(train_ds.f_idx) if hasattr(train_ds, "f_idx") else n_freq_train
    trunk_dim = _trunk_dim(train_ds)
    arch_kw = _arch_kwargs(
        residual_fno=residual_fno,
        n_rec=n_rec,
        fno_width=fno_width,
        fno_n_modes=fno_n_modes,
        fno_n_layers=fno_n_layers,
        n_gno_layers=n_gno_layers,
        fno_kind=fno_kind,
    )
    model = build_model(
        branch_mode,
        field_channels=config.FIELD_CHANNELS,
        stoch_dim=stoch_dim(),
        trunk_dim=trunk_dim,
        latent_dim=config.LATENT_DIM,
        field_hidden=config.FIELD_HIDDEN,
        branch_hidden=config.BRANCH_HIDDEN,
        trunk_hidden=config.TRUNK_HIDDEN,
        trunk_layers=config.TRUNK_LAYERS,
        field_encoder=field_encoder,
        **arch_kw,
    ).to(device)
    if init_ckpt is not None:
        blob = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        print(f"[train] loaded weights from {init_ckpt}", flush=True)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=config.ADAMW_BETAS,
        weight_decay=config.WEIGHT_DECAY,
    )
    sched: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
    if use_lr_sched:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=float(lr_sched_factor),
            patience=int(lr_sched_patience),
            min_lr=float(lr_sched_min),
            threshold=1e-8,
        )
    crit = nn.SmoothL1Loss(beta=config.SMOOTH_L1_BETA)
    t_mean = stats["target_mean"].to(device)
    t_std = stats["target_std"].to(device)
    freq_s = getattr(train_ds, "freq_s", None)
    peak_mask = (
        _peak_band_mask(np.asarray(freq_s), n_rec) if freq_s is not None else None
    )
    wandb_run = init_wandb(
        run_name,
        {
            "encoder": field_encoder,
            "serial_tf1d": serial_tf1d,
            "n_freq_train": n_freq_train,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience if patience is not None else config.PATIENCE,
            "lr_sched": bool(use_lr_sched),
            "lr_sched_factor": float(lr_sched_factor),
            "lr_sched_patience": int(lr_sched_patience),
            "lr_sched_min": float(lr_sched_min),
            "val_monitor": "smooth_l1",
            "seed": seed,
            "iid_frac": iid_frac,
            "aux_tf_rel_l2": aux_tf_rel_l2,
            "aux_peak_band": aux_peak_band,
            "residual_fno": residual_fno,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "host": os.environ.get("WANDB_HOST", "laptop"),
            "mix": mix_tag,
            "n_freq": n_freq_train,
            **arch_kw,
        },
        enabled=use_wandb,
    )
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    patience_budget = int(patience if patience is not None else config.PATIENCE)
    patience_left = patience_budget
    history: list[dict] = []
    ckpt_path = config.CHECKPOINT_DIR / f"{run_name}.pt"
    t0 = time.time()
    epoch_bar = trange(1, epochs + 1, desc=run_name)
    for epoch in epoch_bar:
        model.train()
        tr_loss, tr_n = 0.0, 0
        tr_aux, tr_peak = 0.0, 0.0
        for batch in tqdm(train_loader, desc="train", leave=False):
            fields = batch["fields"].to(device)
            stoch = batch["stoch"].to(device)
            trunk_y = batch["trunk_y"].to(device)
            target_t = batch["target"].to(device)
            pred = _forward(model, fields, stoch, trunk_y, branch_mode)
            loss = crit(pred, target_t)
            if aux_tf_rel_l2 > 0 or aux_peak_band > 0:
                pred_raw = pred * t_std + t_mean
                tf1d = batch["tf1d"].to(device)
                tf2d = batch["tf2d"].to(device)
                tf_hat = tf1d + pred_raw
                if aux_tf_rel_l2 > 0:
                    aux = _batch_rel_l2(tf_hat, tf2d, None)
                    loss = loss + float(aux_tf_rel_l2) * aux
                    tr_aux += float(aux.item()) * target_t.shape[0]
                if aux_peak_band > 0 and peak_mask is not None:
                    peak = _batch_rel_l2(tf_hat, tf2d, peak_mask)
                    loss = loss + float(aux_peak_band) * peak
                    tr_peak += float(peak.item()) * target_t.shape[0]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * target_t.shape[0]
            tr_n += target_t.shape[0]
        val_m = evaluate(
            model,
            val_loader,
            device,
            branch_mode,
            stats,
            n_rec=n_rec,
            n_freq=n_freq,
        )
        train_sL1 = tr_loss / max(tr_n, 1)
        row = {
            "epoch": epoch,
            "train_smooth_l1": train_sL1,
            **{f"val_{k}": v for k, v in val_m.items()},
        }
        if aux_tf_rel_l2 > 0:
            row["train_tf_rel_l2"] = tr_aux / max(tr_n, 1)
        if aux_peak_band > 0:
            row["train_peak_rel_l2"] = tr_peak / max(tr_n, 1)
        history.append(row)
        improved = val_m["smooth_l1"] < best_val - 1e-8
        if improved:
            best_val = val_m["smooth_l1"]
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            patience_left = patience_budget
        elif not no_early_stop:
            patience_left -= 1
        epoch_bar.set_postfix(
            sL1=f"{train_sL1:.3e}",
            r2_R=f"{val_m['r2_R']:.3f}",
            lr=f"{opt.param_groups[0]['lr']:.1e}",
        )
        payload = {
            "epoch": epoch,
            "train/smooth_l1": train_sL1,
            "lr": opt.param_groups[0]["lr"],
            "val/smooth_l1": val_m["smooth_l1"],
            "val/r2_R": val_m["r2_R"],
            "val/pearson_R_freq": val_m["pearson_R_freq"],
            "val/delta_r2_TF": val_m["delta_r2_TF"],
            "val/rel_l2_TF": val_m["rel_l2_TF"],
            "val/rel_l2_TF_1d_only": val_m["rel_l2_TF_1d_only"],
            "val/best_smooth_l1": best_val,
            "val/best_epoch": best_epoch,
        }
        if "train_tf_rel_l2" in row:
            payload["train/tf_rel_l2"] = row["train_tf_rel_l2"]
        if "train_peak_rel_l2" in row:
            payload["train/peak_rel_l2"] = row["train_peak_rel_l2"]
        log_wandb(wandb_run, payload, step=epoch)
        if sched is not None:
            prev_lr = float(opt.param_groups[0]["lr"])
            sched.step(val_m["smooth_l1"])
            new_lr = float(opt.param_groups[0]["lr"])
            if new_lr < prev_lr * 0.999:
                print(
                    f"[{run_name}] ReduceLROnPlateau {prev_lr:.2e} → {new_lr:.2e} "
                    f"at epoch {epoch} (monitor=val/smooth_l1)",
                    flush=True,
                )
        if not no_early_stop and not improved and patience_left <= 0:
            print(
                f"[{run_name}] early stop at epoch {epoch} "
                f"(best val smooth_l1={best_val:.4e} @ {best_epoch})",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    per_domain: dict[str, Any] = {}
    for dname, ds in extra_tests.items():
        loader = _DL(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        per_domain[dname] = evaluate(
            model,
            loader,
            device,
            branch_mode,
            stats,
            n_rec=ds.n_rec,
            n_freq=len(ds.f_idx),
        )
        print(
            f"[{run_name}] test[{dname}] r2_R={per_domain[dname]['r2_R']:.3f} "
            f"Δr2_TF={per_domain[dname]['delta_r2_TF']:+.3f} "
            f"rel_l2_TF={per_domain[dname]['rel_l2_TF']:.3f}",
            flush=True,
        )
    flat = _flatten_domain_metrics(per_domain)
    log_wandb(wandb_run, flat)
    summary_wandb(
        wandb_run,
        {
            "best_val_smooth_l1": best_val,
            "best_epoch": best_epoch,
            "epochs_ran": len(history),
            **flat,
        },
    )

    stats_cpu = {k: v.detach().cpu() for k, v in stats.items()}
    torch.save(
        {
            "model": best_state or model.state_dict(),
            "target": target,
            "branch_mode": branch_mode,
            "trunk_set": trunk_set,
            "field_encoder": field_encoder,
            "n_freq_train": int(n_freq_train),
            "n_freq_eval": int(n_freq_eval),
            "serial_tf1d": bool(serial_tf1d),
            "iid_frac": iid_frac,
            "aux_tf_rel_l2": float(aux_tf_rel_l2),
            "aux_peak_band": float(aux_peak_band),
            "stats": stats_cpu,
            "test_by_domain": per_domain,
            "history": history,
            "best_epoch": int(best_epoch),
            **arch_kw,
        },
        ckpt_path,
    )
    result = {
        "name": run_name,
        "target": target,
        "branch_mode": branch_mode,
        "trunk_set": trunk_set,
        "field_encoder": field_encoder,
        "serial_tf1d": bool(serial_tf1d),
        "n_freq_train": int(n_freq_train),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "epochs_ran": len(history),
        "seconds": time.time() - t0,
        "checkpoint": str(ckpt_path),
        "init_ckpt": str(init_ckpt) if init_ckpt else None,
        "test_by_domain": per_domain,
        "best_val_smooth_l1": best_val,
        "best_epoch": int(best_epoch),
        "seed": seed,
        "iid_frac": iid_frac,
        "aux_tf_rel_l2": float(aux_tf_rel_l2),
        "aux_peak_band": float(aux_peak_band),
        **arch_kw,
    }
    out_json = config.RESULTS_DIR / "arch_train" / f"{run_name}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, default=str))
    print(f"[{run_name}] wrote {out_json}", flush=True)
    finish_wandb(wandb_run)
    return result


def _trunk_dim(ds) -> int:
    return int(ds._cache[0]["trunk_y"].shape[-1])


def train_one(
    *,
    cache_tag: str,
    target: TargetName,
    branch_mode: BranchMode,
    trunk_set: TrunkSet,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    run_name: str | None = None,
    patience: int | None = None,
    no_early_stop: bool = False,
    field_encoder: FieldEncoderKind = "conv",
    n_freq_train: int = config.N_FREQ_TRAIN,
    n_freq_eval: int = config.N_FREQ_EVAL,
    serial_tf1d: bool = False,
    use_wandb: bool = True,
    lr_sched_factor: float = config.LR_SCHED_FACTOR,
    lr_sched_patience: int = config.LR_SCHED_PATIENCE,
    lr_sched_min: float = config.LR_SCHED_MIN,
    use_lr_sched: bool = True,
) -> dict[str, Any]:
    cache_dir = _build_signed_cache(cache_tag)
    meta = np.load(cache_dir / "meta.npz", allow_pickle=True)
    n = len(meta["sample_idx"])
    splits = make_splits(n, seed=seed)
    enc_tag = "" if field_encoder == "conv" else f"_{field_encoder}"
    name = run_name or f"{branch_mode}{enc_tag}_{trunk_set}_{target}_{cache_tag}"
    train_ds = ResidualDeepONetDataset(
        cache_dir,
        splits.train,
        target=target,
        trunk_set=trunk_set,
        n_freq=n_freq_train,
        serial_tf1d=serial_tf1d,
    )
    val_ds = ResidualDeepONetDataset(
        cache_dir,
        splits.val,
        target=target,
        trunk_set=trunk_set,
        n_freq=n_freq_train,
        serial_tf1d=serial_tf1d,
    )
    extra: dict[str, Any] = {
        "test": ResidualDeepONetDataset(
            cache_dir,
            splits.test,
            target=target,
            trunk_set=trunk_set,
            n_freq=n_freq_train,
            serial_tf1d=serial_tf1d,
        )
    }
    if int(n_freq_eval) != int(n_freq_train):
        extra["test_full"] = ResidualDeepONetDataset(
            cache_dir,
            splits.test,
            target=target,
            trunk_set=trunk_set,
            n_freq=n_freq_eval,
            serial_tf1d=serial_tf1d,
        )
    result = train_from_datasets(
        train_ds=train_ds,
        val_ds=val_ds,
        extra_tests=extra,
        target=target,
        branch_mode=branch_mode,
        trunk_set=trunk_set,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        run_name=name,
        patience=patience,
        no_early_stop=no_early_stop,
        field_encoder=field_encoder,
        n_freq_train=n_freq_train,
        n_freq_eval=n_freq_eval,
        serial_tf1d=serial_tf1d,
        use_wandb=use_wandb,
        lr_sched_factor=lr_sched_factor,
        lr_sched_patience=lr_sched_patience,
        lr_sched_min=lr_sched_min,
        use_lr_sched=use_lr_sched,
    )
    result["cache_tag"] = cache_tag
    result["n_samples"] = n
    result["n_test"] = len(splits.test)
    by = result.get("test_by_domain", {})
    result["test_train_freq"] = by.get("test")
    result["test"] = by.get("test_full", by.get("test"))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-tag", default="n1000_seed42")
    p.add_argument(
        "--target", choices=["R_col", "R_nom"], default=config.DEFAULT_TARGET
    )
    p.add_argument(
        "--branch-mode",
        choices=["single", "multi", "stoch_only", "fields_only"],
        default="single",
    )
    p.add_argument(
        "--trunk-set",
        choices=["fstar", "fstar_fourier", "xL", "full"],
        default="full",
    )
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--patience", type=int, default=config.PATIENCE)
    p.add_argument(
        "--lr-sched",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ReduceLROnPlateau on val SmoothL1; restore best-val weights at the end.",
    )
    p.add_argument("--lr-sched-factor", type=float, default=config.LR_SCHED_FACTOR)
    p.add_argument("--lr-sched-patience", type=int, default=config.LR_SCHED_PATIENCE)
    p.add_argument("--lr-sched-min", type=float, default=config.LR_SCHED_MIN)
    p.add_argument("--no-early-stop", action="store_true")
    p.add_argument(
        "--field-encoder",
        choices=["conv", "resunet", "gno", "attn", "gat"],
        default=config.DEFAULT_FIELD_ENCODER,
    )
    p.add_argument(
        "--serial-tf1d",
        action=argparse.BooleanOptionalAction,
        default=config.DEFAULT_SERIAL_TF1D,
        help="Condition R-hat on log(TF_1D) in the trunk (shipped serial operator).",
    )
    p.add_argument(
        "--n-freq-train",
        type=int,
        default=config.N_FREQ_TRAIN,
        help="Log-spaced trunk queries during training (eval is always full grid).",
    )
    p.add_argument(
        "--n-freq-eval",
        type=int,
        default=config.N_FREQ_EVAL,
        help="Frequency bins for reported test metrics (default: full 1000).",
    )
    p.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=config.WANDB_DEFAULT,
        help="Log train/val/test metrics to Weights & Biases.",
    )
    args = p.parse_args()
    train_one(
        cache_tag=args.cache_tag,
        target=args.target,  # type: ignore[arg-type]
        branch_mode=args.branch_mode,  # type: ignore[arg-type]
        trunk_set=args.trunk_set,  # type: ignore[arg-type]
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        patience=args.patience,
        no_early_stop=args.no_early_stop,
        field_encoder=args.field_encoder,  # type: ignore[arg-type]
        n_freq_train=args.n_freq_train,
        n_freq_eval=args.n_freq_eval,
        serial_tf1d=args.serial_tf1d,
        use_wandb=args.wandb,
        lr_sched_factor=args.lr_sched_factor,
        lr_sched_patience=args.lr_sched_patience,
        lr_sched_min=args.lr_sched_min,
        use_lr_sched=args.lr_sched,
    )


if __name__ == "__main__":
    main()
