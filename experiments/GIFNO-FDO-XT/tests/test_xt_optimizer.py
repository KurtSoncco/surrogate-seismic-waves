"""Optimizer and sweep tests for GIFNO-FDO-XT."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
import torch.optim as optim


def _reload_config():
    for name in ("config", "train"):
        sys.modules.pop(name, None)
    return importlib.import_module("config")


def test_build_optimizer_amsgrad_default():
    cfg = _reload_config()
    from train import build_optimizer

    model = torch.nn.Linear(4, 2)
    opt = build_optimizer(model)
    assert isinstance(opt, optim.Adam)
    assert opt.defaults["amsgrad"] is True
    assert cfg.LOSS_P == 1


def test_sweep_variants_xt_loads_three():
    from sweep_launch import load_variants

    path = Path(__file__).resolve().parents[1] / "sweep_variants_xt.tsv"
    variants = load_variants(path)
    assert len(variants) == 3
    meters = next(v for v in variants if v.name == "xt_p1_amsgrad_wide_meters")
    assert meters.overrides["X_COORD_MODE"] == "meters"
