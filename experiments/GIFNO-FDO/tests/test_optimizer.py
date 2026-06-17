"""Optimizer config and factory tests for GIFNO-FDO."""

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


def test_build_optimizer_adam_defaults():
    cfg = _reload_config()
    from train import build_optimizer

    model = torch.nn.Linear(4, 2)
    opt = build_optimizer(model)
    assert isinstance(opt, optim.Adam)
    assert opt.defaults["lr"] == cfg.LEARNING_RATE
    assert opt.defaults["betas"] == (cfg.ADAM_BETA1, cfg.ADAM_BETA2)
    assert opt.defaults["eps"] == cfg.ADAM_EPS
    assert opt.defaults["amsgrad"] is False


def test_build_optimizer_adamw_from_env(monkeypatch):
    monkeypatch.setenv("GIFNO_OPTIMIZER", "adamw")
    monkeypatch.setenv("GIFNO_WEIGHT_DECAY", "0.01")
    monkeypatch.setenv("GIFNO_AMSGRAD", "true")
    _reload_config()
    from train import build_optimizer

    model = torch.nn.Linear(4, 2)
    opt = build_optimizer(model)
    assert isinstance(opt, optim.AdamW)
    assert opt.defaults["weight_decay"] == 0.01
    assert opt.defaults["amsgrad"] is True


def test_loss_p_env_override(monkeypatch):
    monkeypatch.setenv("GIFNO_LOSS_P", "1")
    cfg = _reload_config()
    assert cfg.LOSS_P == 1


def test_sweep_variants_opt_loads_twelve():
    from sweep_launch import load_variants

    path = Path(__file__).resolve().parents[1] / "sweep_variants_opt.tsv"
    variants = load_variants(path)
    assert len(variants) == 12
    p1 = next(v for v in variants if v.name == "wide_h1_p1_adamw")
    assert p1.overrides["LOSS_P"] == "1"
    assert p1.overrides["OPTIMIZER"] == "adamw"
