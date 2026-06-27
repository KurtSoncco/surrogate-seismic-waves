# train.py
"""Training loop for GIFNO-FDO-XT-LOGLO-POD."""

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


def _use_amp() -> bool:
    return config.USE_AMP and config.DEVICE.type == "cuda"


def train_model(train_loader, val_loader):
    # Free A100 speedups: TF32 matmuls/convs + cuDNN autotune (fixed input sizes).
    if config.DEVICE.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    model = create_model().to(config.DEVICE)

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

            optimizer.zero_grad(set_to_none=True)
            with autocast:
                outputs = model(inputs)
                loss = criterion(outputs, targets, masks)
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
        with torch.no_grad():
            for inputs, targets, masks in val_loader:
                inputs = inputs.to(config.DEVICE, non_blocking=non_blocking)
                targets = targets.to(config.DEVICE, non_blocking=non_blocking)
                masks = masks.to(config.DEVICE, non_blocking=non_blocking)
                with autocast:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets, masks)
                val_loss += loss.item() * inputs.size(0)
                n_val += inputs.size(0)

        val_loss /= max(n_val, 1)
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
        else:
            monitored = val_loss

        log_payload = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
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
        t.set_postfix(train_loss=train_loss, val_loss=val_loss)

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
