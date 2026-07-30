"""Train TH-FNO on GIFNO corpus — direct |TF|(x, log f), SmoothL1, W&B."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

import config
from losses_th import THFNOLoss
from model import GatedDeltaModel, create_model


def _curve_rel_l2(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(pred - target) / (np.linalg.norm(target) + 1e-12))


def _percentile_stats(vals: list[float], prefix: str) -> dict[str, float]:
    if not vals:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_p10": float("nan"),
            f"{prefix}_p90": float("nan"),
        }
    a = np.asarray(vals, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(a)),
        f"{prefix}_median": float(np.median(a)),
        f"{prefix}_p10": float(np.percentile(a, 10)),
        f"{prefix}_p90": float(np.percentile(a, 90)),
    }


@torch.no_grad()
def evaluate_loader(
    model: GatedDeltaModel, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    """Center/edge means plus per-curve median/p10/p90 (Session N+1 §F)."""
    model.eval()
    rels, pears = [], []
    edge_rels = []
    all_curve_rels: list[float] = []
    rec = config.recorder_x_indices()
    edge = [int(rec[0]), int(rec[-1])]
    center = int(rec[len(rec) // 2])
    for batch in loader:
        x, haskell, target, mask, cov, dip, physics = [t.to(device) for t in batch]
        pred = model(x, haskell, cov, dip, physics=physics)
        for b in range(pred.shape[0]):
            # All supervised recorders → per-curve distribution
            cols = torch.where(mask[b] > 0.5)[0].detach().cpu().numpy()
            for c in cols:
                p = pred[b, int(c)].detach().cpu().numpy()
                t = target[b, int(c)].detach().cpu().numpy()
                all_curve_rels.append(_curve_rel_l2(p, t))
            for c in [center, *edge]:
                p = pred[b, c].detach().cpu().numpy()
                t = target[b, c].detach().cpu().numpy()
                r = _curve_rel_l2(p, t)
                if c == center:
                    rels.append(r)
                    if np.std(p) > 1e-15 and np.std(t) > 1e-15:
                        pears.append(float(np.corrcoef(p, t)[0, 1]))
                else:
                    edge_rels.append(r)
    out = {
        "rel_l2_center_mean": float(np.mean(rels)) if rels else float("nan"),
        "pearson_center_mean": float(np.mean(pears)) if pears else float("nan"),
        "rel_l2_edge_mean": float(np.mean(edge_rels)) if edge_rels else float("nan"),
        "n": float(len(rels)),
        "n_curves": float(len(all_curve_rels)),
    }
    out.update(_percentile_stats(all_curve_rels, "rel_l2_curve"))
    # Alias expected by B1 gate reporting
    out["rel_c"] = out["rel_l2_center_mean"]
    return out


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    num_epochs: int | None = None,
    save_path: Path | None = None,
) -> GatedDeltaModel:
    device = config.DEVICE
    model = create_model().to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        amsgrad=config.AMSGRAD,
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5)
    loss_fn = THFNOLoss().to(device)
    epochs = num_epochs or config.NUM_EPOCHS
    save_path = Path(save_path or config.MODEL_SAVE_PATH)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project=config.WANDB_PROJECT,
        name=config.WANDB_RUN_NAME,
        config={
            "predict_mode": config.PREDICT_MODE,
            "amplitude_domain": config.AMPLITUDE_DOMAIN,
            "loss_smooth_l1_w": config.LOSS_SMOOTH_L1_WEIGHT,
            "loss_peak_w": config.LOSS_PEAK_WEIGHT,
            "loss_spec_w": config.LOSS_SPEC_WEIGHT,
            "loss_term_norm": config.LOSS_TERM_NORM,
            "loss_term_norm_momentum": config.LOSS_TERM_NORM_MOMENTUM,
            "smooth_l1_beta": config.SMOOTH_L1_BETA,
            "nz_max": config.NZ_MAX,
            "n_freq": config.N_FREQ,
            "n_lateral": config.N_LATERAL,
            "latent_channels": config.LATENT_CHANNELS,
            "deeponet_dim": config.DEEPONET_LATENT_DIM,
            "num_fno_layers": config.NUM_FNO_LAYERS,
            "use_fourier": config.USE_FOURIER_FEATURES,
            "residual_mode": config.RESIDUAL_MODE,
            "log_delta_c": float(getattr(config, "LOG_DELTA_C", config.LOG_DELTA_CLAMP)),
            "zero_init_residual_head": bool(
                getattr(config, "ZERO_INIT_RESIDUAL_HEAD", True)
            ),
            "trend_freq_scale": config.TREND_FREQ_SCALE,
            "seed": config.SEED,
            "lr": config.LEARNING_RATE,
            "batch_size": config.BATCH_SIZE,
            "num_epochs": epochs,
            "n_train": len(train_loader.dataset),
            "n_val": len(val_loader.dataset),
            "tf_log_eps": config.TF_LOG_EPS,
            "ckpt": str(save_path),
        },
        tags=[
            config.PREDICT_MODE,
            "ab_check" if config.PREDICT_MODE == "residual" else "primary",
            "robust_c123",
        ],
    )
    print(f"[TH-FNO] W&B run: {run.url}", flush=True)

    best = float("inf")
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        part_keys = (
            "loss_smooth_l1",
            "loss_peak",
            "loss_spec",
            "loss_smooth_l1_raw",
            "loss_peak_raw",
            "loss_spec_raw",
        )
        part_acc = {k: [] for k in part_keys}
        for batch in train_loader:
            x, haskell, target, mask, cov, dip, physics = [t.to(device) for t in batch]
            pred = model(x, haskell, cov, dip, physics=physics)
            loss, parts = loss_fn(pred, target, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            opt.step()
            losses.append(float(loss.detach()))
            for k in part_keys:
                if k in parts:
                    part_acc[k].append(parts[k])
        val = evaluate_loader(model, val_loader, device)
        metric = val["rel_l2_center_mean"]
        sched.step(metric)
        train_loss = float(np.mean(losses))
        payload = {
            "epoch": epoch,
            "train_loss": train_loss,
            "loss_smooth_l1": float(np.mean(part_acc["loss_smooth_l1"])),
            "loss_peak": float(np.mean(part_acc["loss_peak"])),
            "loss_spec": float(np.mean(part_acc["loss_spec"])),
            "loss_smooth_l1_raw": float(np.mean(part_acc["loss_smooth_l1_raw"] or [0.0])),
            "loss_peak_raw": float(np.mean(part_acc["loss_peak_raw"] or [0.0])),
            "loss_spec_raw": float(np.mean(part_acc["loss_spec_raw"] or [0.0])),
            "val_rel_l2_center": metric,
            "val_pearson_center": val["pearson_center_mean"],
            "val_rel_l2_edge": val["rel_l2_edge_mean"],
            "val_rel_l2_curve_median": val.get("rel_l2_curve_median", float("nan")),
            "val_rel_l2_curve_p10": val.get("rel_l2_curve_p10", float("nan")),
            "val_rel_l2_curve_p90": val.get("rel_l2_curve_p90", float("nan")),
            "learning_rate": opt.param_groups[0]["lr"],
            "epoch_sec": time.time() - t0,
        }
        wandb.log(payload)
        print(
            f"[TH-FNO epoch {epoch:03d}] loss={train_loss:.4f} "
            f"sL1={payload['loss_smooth_l1']:.4f} "
            f"peak={payload['loss_peak']:.4f} "
            f"spec={payload['loss_spec']:.4f} "
            f"(raw sL1={payload['loss_smooth_l1_raw']:.3f} "
            f"peak={payload['loss_peak_raw']:.3f} "
            f"spec={payload['loss_spec_raw']:.3f}) "
            f"val_rel_c={metric:.4f} val_pear_c={val['pearson_center_mean']:.4f} "
            f"val_rel_edge={val['rel_l2_edge_mean']:.4f} "
            f"curve_med={payload['val_rel_l2_curve_median']:.4f} "
            f"({payload['epoch_sec']:.1f}s)",
            flush=True,
        )
        if metric < best - 1e-4:
            best = metric
            stale = 0
            torch.save(model.state_dict(), save_path)
            wandb.run.summary["best_val_rel_l2_center"] = best
        else:
            stale += 1
            if stale >= config.EARLY_STOP_PATIENCE:
                print(f"Early stop at epoch {epoch}")
                break

    if save_path.is_file():
        model.load_state_dict(torch.load(save_path, map_location=device))
    # Leave run open so main() can attach TF comparison images before finish.
    return model
