"""Best-effort Weights & Biases logging (never abort training)."""

from __future__ import annotations

import os
from typing import Any

import config

_TAG_KEYS = ("mix", "encoder", "fno_kind", "host")


def _run_tags(cfg: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    for key in _TAG_KEYS:
        val = cfg.get(key)
        if val is None or val == "":
            continue
        tags.append(str(val))
    return tags


def _configure_epoch_metrics(wandb_mod: Any) -> None:
    wandb_mod.define_metric("epoch")
    wandb_mod.define_metric("train/*", step_metric="epoch")
    wandb_mod.define_metric("val/*", step_metric="epoch")
    wandb_mod.define_metric("lr", step_metric="epoch")


def init_wandb(
    run_name: str,
    cfg: dict[str, Any],
    *,
    enabled: bool = True,
) -> Any:
    if not enabled:
        return None
    try:
        import wandb
    except Exception as exc:
        print(f"[wandb] import failed ({exc}); continuing offline", flush=True)
        return None
    project = os.environ.get("WANDB_PROJECT", config.WANDB_PROJECT)
    kwargs: dict[str, Any] = {}
    if os.environ.get("WANDB_API_KEY") and not os.environ.get("WANDB_MODE"):
        kwargs["mode"] = "online"
    elif not os.environ.get("WANDB_API_KEY") and not os.environ.get("WANDB_MODE"):
        kwargs["mode"] = "offline"
    cfg = dict(cfg)
    cfg.setdefault("host", os.environ.get("WANDB_HOST", "laptop"))
    tags = _run_tags(cfg)
    group = str(cfg["mix"]) if cfg.get("mix") else None
    try:
        run = wandb.init(
            project=project,
            name=run_name,
            config=cfg,
            tags=tags,
            group=group,
            **kwargs,
        )
    except Exception as exc:
        print(f"[wandb] init failed ({exc}); continuing without wandb", flush=True)
        return None
    try:
        _configure_epoch_metrics(wandb)
    except Exception as exc:
        print(f"[wandb] define_metric failed ({exc})", flush=True)
    return run


def log_wandb(run: Any, payload: dict[str, Any], step: int | None = None) -> None:
    if run is None:
        return
    try:
        if step is None:
            run.log(payload)
        else:
            run.log(payload, step=step)
    except Exception as exc:
        print(f"[wandb] log failed ({exc})", flush=True)


def summary_wandb(run: Any, payload: dict[str, Any]) -> None:
    if run is None:
        return
    try:
        for key, val in payload.items():
            run.summary[key] = val
    except Exception as exc:
        print(f"[wandb] summary failed ({exc})", flush=True)


def finish_wandb(run: Any) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass
