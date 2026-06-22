# transfer.py
"""Load frozen XT base weights for band-targeted boost training."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

_BASE_PREFIXES = ("lift.", "fno.", "head.")


def load_xt_base(
    model: nn.Module,
    checkpoint: str | Path,
) -> None:
    """Load XT checkpoint into model.base and freeze base parameters."""
    ckpt = Path(checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Pretrain checkpoint not found: {ckpt}")

    src = torch.load(ckpt, map_location="cpu", weights_only=True)
    base_sd = model.base.state_dict()
    to_load: dict[str, torch.Tensor] = {}
    for key, tensor in src.items():
        if not any(key.startswith(p) for p in _BASE_PREFIXES):
            continue
        if key not in base_sd:
            continue
        if base_sd[key].shape != tensor.shape:
            raise ValueError(
                f"Shape mismatch for base.{key}: "
                f"{tuple(base_sd[key].shape)} vs XT {tuple(tensor.shape)}"
            )
        to_load[key] = tensor

    if not to_load:
        raise RuntimeError(f"No compatible XT keys in {ckpt}")

    model.base.load_state_dict(to_load, strict=False)
    print(
        f"[transfer] Loaded {len(to_load)} base tensors from {ckpt.name}; "
        f"residual head remains at init"
    )

    for param in model.base.parameters():
        param.requires_grad = False
    model.base.eval()


def freeze_base(model: nn.Module) -> None:
    for param in model.base.parameters():
        param.requires_grad = False
    model.base.eval()
