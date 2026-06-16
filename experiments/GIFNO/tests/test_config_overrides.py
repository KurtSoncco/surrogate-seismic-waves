# tests/test_config_overrides.py
"""Tests for GIFNO config env overrides."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


GIFNO_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("GIFNO_DATA_ROOT", str(GIFNO_DIR / "dummy_data"))
if str(GIFNO_DIR) not in sys.path:
    sys.path.insert(0, str(GIFNO_DIR))


def _reload_config(**env: str):
    for key, val in env.items():
        os.environ[key] = val
    if "config" in sys.modules:
        del sys.modules["config"]
    import config  # noqa: F401

    return importlib.import_module("config")


def test_parse_fno_modes():
    import config

    assert config._parse_gifno_env_value("FNO_MODES", "48,48") == (48, 48)


def test_parse_hard_mining_false():
    import config

    assert config._parse_gifno_env_value("HARD_MINING", "false") is False
    assert config._parse_gifno_env_value("HARD_MINING", "true") is True


def test_env_override_loss_h1(monkeypatch):
    monkeypatch.setenv("GIFNO_LOSS_H1_WEIGHT", "0.25")
    cfg = _reload_config()
    assert cfg.LOSS_H1_WEIGHT == 0.25


def test_env_override_latent_channels(monkeypatch):
    monkeypatch.setenv("GIFNO_LATENT_CHANNELS", "96")
    cfg = _reload_config()
    assert cfg.LATENT_CHANNELS == 96


def test_env_override_wandb_run_name(monkeypatch):
    monkeypatch.setenv("WANDB_RUN_NAME", "sweep_test_screen")
    cfg = _reload_config()
    assert cfg.WANDB_RUN_NAME == "sweep_test_screen"


def test_sweep_run_tag_and_paths():
    from sweep_launch import SweepVariant, build_export_env, sweep_run_tag

    assert sweep_run_tag(screen=True, limit=4000) == "n4000"
    assert sweep_run_tag(screen=False, limit=None) == "full"

    tf = Path("/data/transfer_function")
    env = build_export_env(
        SweepVariant("latent_wide", {"LATENT_CHANNELS": "96"}),
        tf,
        screen=True,
        limit=4000,
    )
    assert env["WANDB_RUN_NAME"] == "sweep_latent_wide_n4000"
    assert env["GIFNO_MODEL_DIR"].endswith("models/sweep/n4000/latent_wide")
    assert env["GIFNO_LATENT_CHANNELS"] == "96"


def test_env_override_learning_rate(monkeypatch):
    monkeypatch.setenv("GIFNO_LEARNING_RATE", "0.005")
    cfg = _reload_config()
    assert cfg.LEARNING_RATE == 0.005


def test_sweep_variants_r2_loads_nine():
    from sweep_launch import load_variants

    path = GIFNO_DIR / "sweep_variants_r2.tsv"
    variants = load_variants(path)
    assert len(variants) == 9
    combo = next(v for v in variants if v.name == "lw_no_mine_fno_lr5e3")
    assert combo.overrides["LATENT_CHANNELS"] == "96"
    assert combo.overrides["HARD_MINING"] == "false"
    assert combo.overrides["FNO_MODES"] == "48,48"
    assert combo.overrides["LEARNING_RATE"] == "0.005"


def test_sweep_variants_tsv_loads_six():
    from sweep_launch import load_variants

    path = GIFNO_DIR / "sweep_variants.tsv"
    variants = load_variants(path)
    assert len(variants) == 6
    names = {v.name for v in variants}
    assert names == {
        "baseline",
        "h1_strong",
        "fno_wide",
        "latent_wide",
        "freq_loss",
        "no_mining",
    }
    h1 = next(v for v in variants if v.name == "h1_strong")
    assert h1.overrides["LOSS_H1_WEIGHT"] == "0.25"
