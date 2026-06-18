# transfer.py
"""Load XT pretrained weights and phased fine-tuning helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import torch
import torch.nn as nn

import config

# Keys shared between XT and LG (FNO path + DeepONet head).
_XT_PREFIXES = ("lift.", "fno.", "head.")


def load_xt_pretrained(
    model: nn.Module,
    checkpoint: str | Path,
) -> Dict[str, torch.Tensor]:
    """
    Load compatible weights from a GIFNO-FDO-XT checkpoint.

    Returns anchor tensors for L2-SP on loaded parameters.
    """
    ckpt = Path(checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Pretrain checkpoint not found: {ckpt}")

    src = torch.load(ckpt, map_location="cpu", weights_only=True)
    dst = model.state_dict()
    to_load: Dict[str, torch.Tensor] = {}
    for key, tensor in src.items():
        if not any(key.startswith(p) for p in _XT_PREFIXES):
            continue
        if key not in dst:
            continue
        if dst[key].shape != tensor.shape:
            raise ValueError(
                f"Shape mismatch for {key}: LG {tuple(dst[key].shape)} "
                f"vs XT {tuple(tensor.shape)}"
            )
        to_load[key] = tensor

    if not to_load:
        raise RuntimeError(f"No compatible XT keys loaded from {ckpt}")

    missing = set(dst) - set(to_load)
    model.load_state_dict(to_load, strict=False)
    print(
        f"[transfer] Loaded {len(to_load)} tensors from {ckpt.name}; "
        f"{len(missing)} LG-only keys remain at init"
    )
    return {k: v.detach().clone() for k, v in to_load.items()}


def l2sp_penalty(
    model: nn.Module,
    anchor: Dict[str, torch.Tensor],
    weight: float,
) -> torch.Tensor:
    """L2-SP: penalize deviation from pretrained anchor weights."""
    if weight <= 0 or not anchor:
        return torch.zeros((), device=next(model.parameters()).device)
    loss = torch.zeros((), device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if name in anchor:
            loss = loss + torch.sum((param - anchor[name].to(param.device)) ** 2)
    return weight * loss


def _set_requires(module: nn.Module | None, requires: bool) -> None:
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = requires


def apply_train_phase(model: nn.Module, phase: int) -> list[str]:
    """
    Freeze/unfreeze modules for transfer phase.

    Phase 1: train lift_unet, unet, fusion
    Phase 2: + head
    Phase 3: all (FNO uses lower LR via optimizer param groups)
    """
    from model import GIFNOLGModel  # noqa: F401, F841

    trainable: list[str] = []
    freeze_all = [
        model.lift,
        model.fno,
        model.lift_unet,
        model.unet,
        model.fusion,
        model.head,
    ]
    for m in freeze_all:
        _set_requires(m, False)

    if phase <= 1:
        for name, m in (
            ("lift_unet", model.lift_unet),
            ("unet", model.unet),
            ("fusion", model.fusion),
        ):
            _set_requires(m, True)
            trainable.append(name)
    elif phase == 2:
        for name, m in (
            ("lift_unet", model.lift_unet),
            ("unet", model.unet),
            ("fusion", model.fusion),
            ("head", model.head),
        ):
            _set_requires(m, True)
            trainable.append(name)
    else:
        for name, m in (
            ("lift", model.lift),
            ("fno", model.fno),
            ("lift_unet", model.lift_unet),
            ("unet", model.unet),
            ("fusion", model.fusion),
            ("head", model.head),
        ):
            _set_requires(m, True)
            trainable.append(name)

    print(f"[transfer] TRAIN_PHASE={phase} trainable: {', '.join(trainable)}")
    return trainable


def build_phase_optimizer(
    model: nn.Module,
    phase: int,
) -> torch.optim.Optimizer:
    """Adam with phase-appropriate learning rates."""
    if phase >= 3:
        fno_params: list[nn.Parameter] = []
        other_params: list[nn.Parameter] = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("lift.") or name.startswith("fno."):
                fno_params.append(p)
            else:
                other_params.append(p)
        groups = [
            {"params": fno_params, "lr": config.PHASE3_FNO_LR},
            {"params": other_params, "lr": config.PHASE3_OTHER_LR},
        ]
    else:
        groups = [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": config.phase_learning_rate(phase),
            }
        ]
    kwargs = {
        "betas": (config.ADAM_BETA1, config.ADAM_BETA2),
        "eps": config.ADAM_EPS,
        "weight_decay": config.WEIGHT_DECAY,
        "amsgrad": config.AMSGRAD,
    }
    if config.OPTIMIZER.lower() == "adamw":
        return torch.optim.AdamW(groups, **kwargs)
    return torch.optim.Adam(groups, **kwargs)


def trainable_parameter_names(model: nn.Module) -> Iterable[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad]
