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


def test_sweep_variants_xt_loads_six():
    from sweep_launch import load_variants

    path = Path(__file__).resolve().parents[1] / "sweep_variants_xt.tsv"
    variants = load_variants(path)
    assert len(variants) == 7
    wide = next(v for v in variants if v.name == "xt_lat128_d128")
    assert wide.overrides["LATENT_CHANNELS"] == "128"
    assert wide.overrides["DEEPONET_LATENT_DIM"] == "128"
    bandcurr = next(v for v in variants if v.name == "xt_lat128_d128_bandcurr")
    assert bandcurr.overrides["BAND_CURRICULUM"] == "true"
    assert bandcurr.overrides["SELECTION_METRIC"] == "band_balanced"
    assert bandcurr.overrides["EARLY_STOP_PATIENCE"] == "140"
