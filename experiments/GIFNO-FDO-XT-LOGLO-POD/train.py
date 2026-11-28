# train.py
"""Training loop for GIFNO-FDO-XT-LOGLO-POD."""

import math
import time

import config
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange

import wandb

from curriculum import BandCurriculumController
from losses import band_curriculum_weights, build_training_loss
from metrics import compute_val_tail_metrics_torch
from model import create_model


class DivergenceError(RuntimeError):
    """Non-finite or exploding loss; remaining epochs would waste the GPU."""


def _band_curriculum_enabled(criterion: torch.nn.Module) -> bool:
    """Curriculum needs the composite loss (band reweighting hooks)."""
    return getattr(config, "BAND_CURRICULUM", False) and hasattr(
        criterion, "set_band_weights"
    )


def band_balanced_val_metric(val_tail: dict) -> float:
    """Mean of per-band val rel-L2 (low/mid/high), equal weight.

    Unlike neutral val_loss this is not energy-weighted, so high-band gains are
    not drowned by the larger low/mid bands -- the point of band-aware selection.
    Returns NaN if the band metrics are unavailable.
    """
    keys = (
        "val_rel_l2_band_low_mean",
        "val_rel_l2_band_mid_mean",
        "val_rel_l2_band_high_mean",
    )
    vals = [val_tail[k] for k in keys if k in val_tail]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def build_optimizer(model: torch.nn.Module) -> optim.Optimizer:
    opt_name = config.OPTIMIZER.lower()
    kwargs = {
        "lr": config.LEARNING_RATE,
        "betas": (config.ADAM_BETA1, config.ADAM_BETA2),
        "eps": config.ADAM_EPS,
        "weight_decay": config.WEIGHT_DECAY,
        "amsgrad": config.AMSGRAD,
    }
    # NOTE: fused=True is unsupported here (model has complex64 spectral params).
    if opt_name == "adamw":
        return optim.AdamW(model.parameters(), **kwargs)
    if opt_name == "adam":
        return optim.Adam(model.parameters(), **kwargs)
    raise ValueError(f"Unsupported OPTIMIZER={config.OPTIMIZER!r}")


def _enable_tf32() -> None:
    """Enable TF32 matmul/conv, using the PyTorch >=2.9 API when available.

    The old ``allow_tf32`` flags are deprecated (and warn) on newer torch; fall
    back to them only when the new ``fp32_precision`` knobs are absent.
    """
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _use_amp() -> bool:
    return config.USE_AMP and config.DEVICE.type == "cuda"


def diverge_bound() -> float:
    return float(getattr(config, "DIVERGE_LOSS_MAX", 100.0))


def check_loss_finite(value: float, *, epoch: int, where: str) -> None:
    """Raise DivergenceError so the pipeline does not keep stepping NaNs."""
    bound = diverge_bound()
    exploded = bound > 0.0 and math.isfinite(value) and value > bound
    if math.isfinite(value) and not exploded:
        return
    kind = "non-finite" if not math.isfinite(value) else f"{value:.4g} > {bound:g}"
    raise DivergenceError(f"[diverge] {where} epoch {epoch}: loss {kind}")


def _reinit_pod_branch_last_linear(model: torch.nn.Module) -> None:
    """Reset the coefficient layer. It belongs to the basis, not the encoder."""
    head = model.head
    last = head.branch[-1]
    n_modes = int(head.n_modes)
    if not isinstance(last, torch.nn.Linear) or int(last.out_features) != n_modes:
        raise RuntimeError(
            "POD branch last layer must be Linear -> "
            f"{n_modes} modes, got {type(last).__name__} "
            f"out={getattr(last, 'out_features', None)}"
        )
    last.reset_parameters()
    print(f"[init] Reinitialized head.branch[-1] for {n_modes} POD modes", flush=True)


def _zero_init_pod_branch_last(model: torch.nn.Module) -> None:
    """Start log-POD at exp(mean): random Kaiming coeffs overflow through exp."""
    last = model.head.branch[-1]
    torch.nn.init.zeros_(last.weight)
    torch.nn.init.zeros_(last.bias)
    print("[init] Zero-initialized head.branch[-1] (recon starts at exp(POD mean))", flush=True)


def window_consistency_loss(
    outputs: torch.Tensor,
    recorder_x: torch.Tensor,
    pair_offset: int,
    *,
    dense: bool = True,
    masks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean squared log difference on the physical overlap of a shifted pair.

    ``outputs`` is (2B, nx, F) with pairs interleaved (a0, b0, a1, b1, ...).
    Window b starts ``pair_offset`` stations (15 m) east of a. Shared layout
    compares overlapping columns; the 21-station fallback is for legacy
    per-recorder heads that zero the rest of the grid.

    When ``masks`` is (2B, nx), only columns labeled in *both* windows
    contribute. Unlabeled dense columns cannot explode the objective.
    """
    nx = outputs.shape[-2]
    n_freq = outputs.shape[-1]
    out = outputs.view(-1, 2, nx, n_freq)
    if dense:
        shift = int(round(pair_offset * config.LATERAL_SPACING_M / config.DX))
        if shift <= 0 or shift >= nx:
            return outputs.new_zeros(())
        a = out[:, 0, shift:, :]
        b = out[:, 1, : nx - shift, :]
        weight = None
        if masks is not None:
            m = masks.view(-1, 2, nx)
            weight = m[:, 0, shift:] * m[:, 1, : nx - shift]
    else:
        a = out[:, 0].index_select(1, recorder_x[pair_offset:])
        b = out[:, 1].index_select(1, recorder_x[: recorder_x.numel() - pair_offset])
        weight = None
        if masks is not None:
            m = masks.view(-1, 2, nx)
            weight = m[:, 0].index_select(1, recorder_x[pair_offset:]) * m[
                :, 1
            ].index_select(1, recorder_x[: recorder_x.numel() - pair_offset])
    log_a = torch.log(a.float().clamp(1e-6, 1e6))
    log_b = torch.log(b.float().clamp(1e-6, 1e6))
    err = (log_a - log_b).pow(2)
    if weight is None:
        return err.mean()
    w = weight.float()
    wsum = w.sum()
    if float(wsum) == 0:
        return outputs.new_zeros(())
    return (err * w.unsqueeze(-1)).sum() / (wsum * err.shape[-1])


def masked_log_pearson(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    freq: torch.Tensor,
    band: tuple[float, float] = (0.1, 2.5),
) -> torch.Tensor:
    """Per-sample mean Pearson of log amplitude on labeled columns in ``band``.

    ``pred`` / ``target`` are (B, nx, F); ``mask`` is (B, nx). Returns (B,).
    Residual batches are Pearson of the ratio; outcrop batches of |TF|.
    """
    lo, hi = band
    fmask = (freq >= lo) & (freq <= hi)
    log_p = torch.log(pred[..., fmask].float().clamp_min(1e-6))
    log_t = torch.log(target[..., fmask].float().clamp_min(1e-6))
    pc = log_p - log_p.mean(dim=-1, keepdim=True)
    tc = log_t - log_t.mean(dim=-1, keepdim=True)
    num = (pc * tc).sum(dim=-1)
    den = torch.sqrt((pc * pc).sum(dim=-1) * (tc * tc).sum(dim=-1) + 1e-12)
    r = num / den
    w = mask.float() * torch.isfinite(r).float()
    nlab = w.sum(dim=1)
    out = (r.nan_to_num(0.0) * w).sum(dim=1) / nlab.clamp_min(1.0)
    return torch.where(nlab > 0, out, out.new_full(out.shape, float("nan")))


def train_model(train_loader, val_loader):
    # Free A100 speedups: TF32 matmuls/convs + cuDNN autotune (fixed input sizes).
    if config.DEVICE.type == "cuda":
        _enable_tf32()
        torch.backends.cudnn.benchmark = True

    model = create_model().to(config.DEVICE)

    init_ckpt = getattr(config, "INIT_CHECKPOINT", "") or ""
    if init_ckpt:
        from pathlib import Path

        ckpt_path = Path(init_ckpt)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"INIT_CHECKPOINT not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"[init] Loaded weights from {ckpt_path} "
            f"(missing={len(missing)} unexpected={len(unexpected)})"
        )
        if missing:
            print(f"[init] missing keys (first 8): {missing[:8]}")
        if getattr(config, "REINIT_POD_BRANCH", False):
            _reinit_pod_branch_last_linear(model)
    else:
        _zero_init_pod_branch_last(model)

    if config.TORCH_COMPILE and hasattr(torch, "compile"):
        model = torch.compile(model)

    dummy = torch.randn(1, config.IN_CHANNELS, config.NZ_MAX, config.NX).to(
        config.DEVICE
    )
    assert model(dummy).shape == (1, config.NX, config.N_FREQ)

    amp_enabled = _use_amp()
    # bf16 keeps fp32 dynamic range, so no GradScaler is needed and it is more
    # accuracy-safe than fp16 for this FFT-heavy model.
    amp_dtype = torch.bfloat16
    autocast = torch.autocast(
        device_type=config.DEVICE.type, dtype=amp_dtype, enabled=amp_enabled
    )
    use_scaler = amp_enabled and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    non_blocking = amp_enabled
    if amp_enabled:
        print(f"[amp] enabled with dtype={amp_dtype}, grad_scaler={use_scaler}")

    criterion = build_training_loss()
    optimizer = build_optimizer(model)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        patience=config.LR_SCHED_PATIENCE,
        factor=config.LR_SCHED_FACTOR,
    )

    run = wandb.init(
        project=config.WANDB_PROJECT,
        name=config.WANDB_RUN_NAME,
        config={k: v for k, v in vars(config).items() if k.isupper()},
    )

    best_metric = float("inf")
    early_stop_counter = 0
    freq = np.load(config.TF_FREQ_PATH)
    band_curriculum = _band_curriculum_enabled(criterion)
    selection_metric = getattr(config, "SELECTION_METRIC", "val_loss")
    curriculum_mode = getattr(config, "BAND_CURRICULUM_MODE", "time")
    convergence = band_curriculum and curriculum_mode == "convergence"

    controller = None
    if convergence:
        controller = BandCurriculumController(
            n_bands=3,
            floor=config.BAND_CURRICULUM_FLOOR,
            patience=config.BAND_CURRICULUM_PHASE_PATIENCE,
            min_delta=config.BAND_CURRICULUM_MIN_DELTA,
            ramp_epochs=config.BAND_CURRICULUM_RAMP_EPOCHS,
        )
        # Convergence mode advances on the band-balanced metric, so force it.
        selection_metric = "band_balanced"

    # The curriculum controller and band-aware selection need band metrics every
    # epoch; otherwise fall back to the cheap VAL_TAIL_EVERY cadence.
    val_tail_every = int(getattr(config, "VAL_TAIL_EVERY", 1))
    if band_curriculum:
        val_tail_every = 1

    consistency_weight = float(getattr(config, "WINDOW_CONSISTENCY_WEIGHT", 0.0))
    pair_offset = int(getattr(config, "WINDOW_PAIR_OFFSET", 2))
    recorder_x = torch.as_tensor(config.recorder_x_indices(), device=config.DEVICE)
    freq_t = torch.as_tensor(freq, device=config.DEVICE, dtype=torch.float32)
    trusted_band = (0.1, 2.5)

    t = trange(config.NUM_EPOCHS, desc="Training")
    for epoch in t:
        band_w = (1.0, 1.0, 1.0)
        if convergence:
            band_w = controller.current_weights()
            criterion.set_band_balanced_weights(*band_w)
        elif band_curriculum:
            band_w = band_curriculum_weights(
                epoch,
                config.NUM_EPOCHS,
                floor=config.BAND_CURRICULUM_FLOOR,
                mid_start=config.BAND_CURRICULUM_MID_START,
                high_start=config.BAND_CURRICULUM_HIGH_START,
                ramp=config.BAND_CURRICULUM_RAMP,
            )
            criterion.set_band_weights(*band_w)

        t_train_start = time.time()
        model.train()
        train_loss = 0.0
        n_train = 0
        for inputs, targets, masks in train_loader:
            inputs = inputs.to(config.DEVICE, non_blocking=non_blocking)
            targets = targets.to(config.DEVICE, non_blocking=non_blocking)
            masks = masks.to(config.DEVICE, non_blocking=non_blocking)
            paired = inputs.ndim == 5
            if paired:
                inputs, targets, masks = (t.flatten(0, 1) for t in (inputs, targets, masks))

            optimizer.zero_grad(set_to_none=True)
            with autocast:
                outputs = model(inputs)
                loss = criterion(outputs, targets, masks)
                if paired and consistency_weight > 0:
                    loss = loss + consistency_weight * window_consistency_loss(
                        outputs,
                        recorder_x,
                        pair_offset,
                        dense=bool(getattr(model.head, "shared", False)),
                        masks=masks,
                    )
            if epoch == 0 and n_train == 0:
                pred_f = outputs.detach().float()
                print(
                    f"[init] batch0 pred min={float(pred_f.min()):.4g} "
                    f"mean={float(pred_f.mean()):.4g} max={float(pred_f.max()):.4g} "
                    f"loss={float(loss.detach().float().item()):.4g}",
                    flush=True,
                )
            check_loss_finite(float(loss.detach().float().item()), epoch=epoch, where="train batch")
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.GRAD_CLIP_NORM
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.GRAD_CLIP_NORM
                )
                optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            n_train += inputs.size(0)

        train_loss /= max(n_train, 1)
        check_loss_finite(train_loss, epoch=epoch, where="train")
        t_train = time.time() - t_train_start

        # Validation uses a fixed-composition objective so the LR scheduler
        # (on val_loss) is not chasing the moving curriculum weights.
        if convergence:
            criterion.set_band_balanced_weights(None)
        elif band_curriculum:
            criterion.set_band_weights(None)

        t_val_start = time.time()
        model.eval()
        val_loss = 0.0
        n_val = 0
        pearson_parts: list[torch.Tensor] = []
        with torch.no_grad():
            for inputs, targets, masks in val_loader:
                inputs = inputs.to(config.DEVICE, non_blocking=non_blocking)
                targets = targets.to(config.DEVICE, non_blocking=non_blocking)
                masks = masks.to(config.DEVICE, non_blocking=non_blocking)
                with autocast:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets, masks)
                check_loss_finite(float(loss.detach().float().item()), epoch=epoch, where="val batch")
                val_loss += loss.item() * inputs.size(0)
                n_val += inputs.size(0)
                pearson_parts.append(
                    masked_log_pearson(outputs, targets, masks, freq_t, trusted_band).detach()
                )

        val_loss /= max(n_val, 1)
        check_loss_finite(val_loss, epoch=epoch, where="val")
        if pearson_parts:
            pcat = torch.cat(pearson_parts)
            pcat = pcat[torch.isfinite(pcat)]
            val_pearson = float(pcat.median().item()) if pcat.numel() else float("nan")
        else:
            val_pearson = float("nan")
        t_val = time.time() - t_val_start
        # LR scheduler always tracks the neutral objective.
        scheduler.step(val_loss)

        t_tail_start = time.time()
        is_last_epoch = epoch == config.NUM_EPOCHS - 1
        if val_tail_every <= 1 or epoch % val_tail_every == 0 or is_last_epoch:
            val_tail = compute_val_tail_metrics_torch(
                model, val_loader, config.DEVICE, freq
            )
        else:
            val_tail = {}
        t_tail = time.time() - t_tail_start

        if epoch < 3 or epoch % val_tail_every == 0:
            print(
                f"[timing] epoch {epoch}: train={t_train:.1f}s val={t_val:.1f}s "
                f"val_tail={t_tail:.1f}s",
                flush=True,
            )

        band_bal_val = band_balanced_val_metric(val_tail)
        if selection_metric == "band_balanced" and np.isfinite(band_bal_val):
            monitored = band_bal_val
        elif selection_metric == "pearson_trusted" and np.isfinite(val_pearson):
            monitored = -val_pearson
        else:
            monitored = val_loss

        log_payload = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_pearson_trusted": val_pearson,
            "band_bal_val": band_bal_val,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **val_tail,
        }
        if band_curriculum:
            log_payload["w_band_low"] = band_w[0]
            log_payload["w_band_mid"] = band_w[1]
            log_payload["w_band_high"] = band_w[2]
        if controller is not None:
            log_payload["curriculum_phase"] = controller.phase
        wandb.log(log_payload)
        t.set_postfix(train_loss=train_loss, val_loss=val_loss, val_P=val_pearson)

        if monitored < best_metric:
            best_metric = monitored
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Advance the convergence curriculum and warm-restart on a phase change.
        in_final_phase = True
        if controller is not None:
            advanced = controller.step(band_bal_val)
            in_final_phase = controller.is_final_phase
            if advanced:
                # A new phase gets a fresh early-stop budget.
                early_stop_counter = 0
                if config.BAND_CURRICULUM_LR_RESTART:
                    if config.BAND_CURRICULUM_RESET_OPT_STATE:
                        optimizer = build_optimizer(model)
                    restart_lr = (
                        config.LEARNING_RATE * config.BAND_CURRICULUM_LR_RESTART_SCALE
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = restart_lr
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        "min",
                        patience=config.LR_SCHED_PATIENCE,
                        factor=config.LR_SCHED_FACTOR,
                    )

        # Early stop only once the curriculum has reached its final phase.
        if in_final_phase and early_stop_counter >= config.EARLY_STOP_PATIENCE:
            break

    return run
