"""Train / evaluate signed-residual DeepONet."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from data import (
    ResidualDeepONetDataset,
    TargetName,
    TrunkSet,
    make_splits,
    stoch_dim,
    trunk_feature_names,
)
from model import BranchMode, FieldEncoderKind, build_model


def _build_signed_cache(*args, **kwargs):
    from residual_signed import build_signed_cache

    return build_signed_cache(*args, **kwargs)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _stack_key(ds, key: str) -> torch.Tensor:
    return torch.stack([ds._cache[i][key] for i in range(len(ds))], dim=0)


def fit_and_apply_norms(train_ds, *other_ds) -> Dict[str, torch.Tensor]:
    """Z-score stoch, trunk, and target using the training split only."""
    stoch = _stack_key(train_ds, "stoch")
    trunk = _stack_key(train_ds, "trunk_y").reshape(-1, _stack_key(train_ds, "trunk_y").shape[-1])
    target = _stack_key(train_ds, "target").reshape(-1)

    stats = {
        "stoch_mean": stoch.mean(0),
        "stoch_std": stoch.std(0).clamp_min(1e-6),
        "trunk_mean": trunk.mean(0),
        "trunk_std": trunk.std(0).clamp_min(1e-6),
        "target_mean": target.mean(),
        "target_std": target.std().clamp_min(1e-6),
    }

    def _apply(ds) -> None:
        for i in range(len(ds)):
            item = ds._cache[i]
            item["stoch"] = (item["stoch"] - stats["stoch_mean"]) / stats["stoch_std"]
            item["trunk_y"] = (item["trunk_y"] - stats["trunk_mean"]) / stats["trunk_std"]
            item["target_raw"] = item["target"].clone()
            item["target"] = (item["target"] - stats["target_mean"]) / stats["target_std"]

    _apply(train_ds)
    for ds in other_ds:
        _apply(ds)
    return stats


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mode: BranchMode,
    stats: Dict[str, torch.Tensor],
    *,
    n_rec: int,
    n_freq: int,
) -> Dict[str, float]:
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
    for batch in loader:
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
        "smooth_l1_R_raw": float(np.mean(np.where(np.abs(y) < 1.0, 0.5 * y**2, np.abs(y) - 0.5))),
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
) -> Dict[str, Any]:
    device = _device()
    cache_dir = _build_signed_cache(cache_tag)
    meta = np.load(cache_dir / "meta.npz", allow_pickle=True)
    n = len(meta["sample_idx"])
    splits = make_splits(n, seed=seed)
    from torch.utils.data import DataLoader as _DL

    train_ds = ResidualDeepONetDataset(
        cache_dir, splits.train, target=target, trunk_set=trunk_set
    )
    val_ds = ResidualDeepONetDataset(
        cache_dir, splits.val, target=target, trunk_set=trunk_set
    )
    test_ds = ResidualDeepONetDataset(
        cache_dir, splits.test, target=target, trunk_set=trunk_set
    )
    stats = fit_and_apply_norms(train_ds, val_ds, test_ds)
    train_loader = _DL(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = _DL(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = _DL(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    n_rec = train_ds.n_rec
    n_freq = len(train_ds.f_idx)

    trunk_dim = len(trunk_feature_names(trunk_set))
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
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=config.ADAMW_BETAS,
        weight_decay=config.WEIGHT_DECAY,
    )
    crit = nn.SmoothL1Loss(beta=config.SMOOTH_L1_BETA)
    best_val = float("inf")
    best_state = None
    patience_budget = int(patience if patience is not None else config.PATIENCE)
    patience_left = patience_budget
    history: list[dict] = []

    enc_tag = "" if field_encoder == "conv" else f"_{field_encoder}"
    name = run_name or f"{branch_mode}{enc_tag}_{trunk_set}_{target}_{cache_tag}"
    ckpt_path = config.CHECKPOINT_DIR / f"{name}.pt"

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, tr_n = 0.0, 0
        for batch in train_loader:
            fields = batch["fields"].to(device)
            stoch = batch["stoch"].to(device)
            trunk_y = batch["trunk_y"].to(device)
            target_t = batch["target"].to(device)
            pred = _forward(model, fields, stoch, trunk_y, branch_mode)
            loss = crit(pred, target_t)
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
        row = {
            "epoch": epoch,
            "train_smooth_l1": tr_loss / max(tr_n, 1),
            **{f"val_{k}": v for k, v in val_m.items()},
        }
        history.append(row)
        print(
            f"[{name}] epoch {epoch}/{epochs}  "
            f"train_sL1={row['train_smooth_l1']:.4e}  "
            f"val_r2_R={val_m['r2_R']:.3f}  "
            f"val_pearson_R_freq={val_m['pearson_R_freq']:.3f}  "
            f"val_Δr2_TF={val_m['delta_r2_TF']:.4f}",
            flush=True,
        )
        if val_m["smooth_l1"] < best_val - 1e-8:
            best_val = val_m["smooth_l1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience_budget
        elif not no_early_stop:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[{name}] early stop at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_m = evaluate(
        model,
        test_loader,
        device,
        branch_mode,
        stats,
        n_rec=n_rec,
        n_freq=n_freq,
    )
    torch.save(
        {
            "model": best_state or model.state_dict(),
            "target": target,
            "branch_mode": branch_mode,
            "trunk_set": trunk_set,
            "cache_tag": cache_tag,
            "field_encoder": field_encoder,
            "loss": "SmoothL1Loss",
            "optimizer": "AdamW",
            "adamw_betas": list(config.ADAMW_BETAS),
            "smooth_l1_beta": config.SMOOTH_L1_BETA,
            "test": test_m,
            "history": history,
        },
        ckpt_path,
    )
    result = {
        "name": name,
        "target": target,
        "branch_mode": branch_mode,
        "trunk_set": trunk_set,
        "field_encoder": field_encoder,
        "cache_tag": cache_tag,
        "loss": "SmoothL1Loss",
        "optimizer": "AdamW",
        "adamw_betas": list(config.ADAMW_BETAS),
        "n_samples": n,
        "n_train": len(splits.train),
        "n_val": len(splits.val),
        "n_test": len(splits.test),
        "epochs_ran": len(history),
        "seconds": time.time() - t0,
        "checkpoint": str(ckpt_path),
        "test": test_m,
        "best_val_smooth_l1": best_val,
    }
    out_json = config.RESULTS_DIR / f"{name}.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"[{name}] test {test_m} → {out_json}", flush=True)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-tag", default="n100_seed42")
    p.add_argument("--target", choices=["R_col", "R_nom"], default="R_col")
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
    p.add_argument("--no-early-stop", action="store_true")
    p.add_argument(
        "--field-encoder",
        choices=["conv", "resunet"],
        default="conv",
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
    )


if __name__ == "__main__":
    main()
