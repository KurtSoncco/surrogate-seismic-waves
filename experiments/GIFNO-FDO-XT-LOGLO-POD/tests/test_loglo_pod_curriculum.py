"""Tests for the Tier 2 band curriculum wiring (GIFNO-FDO-XT-LOGLO-POD)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import config
from curriculum import BandCurriculumController
from losses import MaskedCompositeLoss
from train import band_balanced_val_metric


def _make_loss(n_freq: int = 128, band_balanced_weight: float = 0.0):
    freq = np.logspace(-1, 1, n_freq).astype(np.float32)
    return MaskedCompositeLoss(
        rel_weight=1.0,
        h1_weight=0.0,
        p=1,
        band_balanced_weight=band_balanced_weight,
        freq=freq,
    )


def _manual_band_balanced(loss, pred, target, weights):
    rels = []
    used = []
    for slc, w in zip(loss._bb_slices, weights):
        if w == 0.0:
            continue
        p = pred[..., slc].reshape(pred.shape[0], -1)
        t = target[..., slc].reshape(target.shape[0], -1)
        rels.append(
            w
            * torch.linalg.norm(p - t, dim=1)
            / torch.linalg.norm(t, dim=1).clamp_min(1e-8)
        )
        used.append(w)
    return sum(rels) / sum(used)


def test_curriculum_defaults_off_and_neutral():
    # Baseline config: curriculum off and the band-balanced term disabled, so the
    # loss is bit-for-bit the existing LOGLO-POD objective.
    assert config.BAND_CURRICULUM is False
    assert config.BAND_CURRICULUM_MODE == "time"
    assert config.LOSS_BAND_BALANCED_WEIGHT == 0.0
    assert config.SELECTION_METRIC == "val_loss"
    loss = _make_loss()
    assert loss.band_balanced_weight == 0.0


def test_config_schedule_defaults_retuned():
    assert config.BAND_CURRICULUM_MID_START == 0.20
    assert config.BAND_CURRICULUM_HIGH_START == 0.50
    assert config.BAND_CURRICULUM_RAMP == 0.20


def test_band_balanced_equals_mean_of_band_rel_l2():
    loss = _make_loss(128)
    torch.manual_seed(1)
    pred = torch.randn(3, 5, 128)
    target = torch.randn(3, 5, 128)
    bb = loss._band_balanced_loss(pred, target)
    manual = _manual_band_balanced(loss, pred, target, (1.0, 1.0, 1.0))
    assert torch.allclose(bb, manual, atol=1e-6)


def test_band_balanced_zero_weight_excludes_band():
    loss = _make_loss(128)
    torch.manual_seed(2)
    pred = torch.randn(2, 5, 128)
    target = torch.randn(2, 5, 128)
    loss.set_band_balanced_weights(1.0, 1.0, 0.0)
    bb = loss._band_balanced_loss(pred, target)
    manual = _manual_band_balanced(loss, pred, target, (1.0, 1.0, 0.0))
    assert torch.allclose(bb, manual, atol=1e-6)


def test_band_balanced_val_metric_means_bands():
    val_tail = {
        "val_rel_l2_band_low_mean": 0.10,
        "val_rel_l2_band_mid_mean": 0.20,
        "val_rel_l2_band_high_mean": 0.60,
        "val_rel_l2_mean": 0.12,  # ignored: energy-weighted, not band-balanced
    }
    assert np.isclose(band_balanced_val_metric(val_tail), 0.30)


def test_band_balanced_val_metric_nan_when_missing():
    assert np.isnan(band_balanced_val_metric({"val_rel_l2_mean": 0.1}))


def test_selection_metric_parse_and_validate():
    assert config._parse_env_value("SELECTION_METRIC", "band_balanced") == (
        "band_balanced"
    )
    assert config._parse_env_value("SELECTION_METRIC", "VAL_LOSS") == "val_loss"
    try:
        config._parse_env_value("SELECTION_METRIC", "bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid SELECTION_METRIC")


def test_band_curriculum_mode_parse_and_validate():
    assert config._parse_env_value("BAND_CURRICULUM_MODE", "convergence") == (
        "convergence"
    )
    try:
        config._parse_env_value("BAND_CURRICULUM_MODE", "bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid BAND_CURRICULUM_MODE")


def test_tier2_overrides_parse():
    assert config._parse_env_value("LR_SCHED_PATIENCE", "40") == 40
    assert config._parse_env_value("LR_SCHED_FACTOR", "0.5") == 0.5
    assert config._parse_env_value("LOSS_BAND_BALANCED_WEIGHT", "0.5") == 0.5
    assert config._parse_env_value("BAND_CURRICULUM", "true") is True
    assert config._parse_env_value("BAND_CURRICULUM_LR_RESTART", "true") is True
    assert config._parse_env_value("BAND_CURRICULUM_RESET_OPT_STATE", "true") is True
    assert config._parse_env_value("BAND_CURRICULUM_PHASE_PATIENCE", "30") == 30


def test_controller_advances_on_plateau_and_warm_restart_signal():
    ctrl = BandCurriculumController(
        n_bands=3, floor=0.25, patience=3, min_delta=1e-4, ramp_epochs=2
    )
    # Phase 0 fully activates the low band; mid/high held at the floor.
    assert ctrl.current_weights() == (1.0, 0.25, 0.25)
    # A plateau (no improvement) advances after patience+1 steps (first sets base).
    advanced = [ctrl.step(1.0) for _ in range(4)]
    assert advanced == [False, False, False, True]
    assert ctrl.phase == 1
    assert not ctrl.is_final_phase


def test_sweep_variants_loglo_pod_loads_bandcurr_cl():
    from sweep_launch import load_variants

    path = Path(__file__).resolve().parents[1] / "sweep_variants_loglo_pod.tsv"
    variants = load_variants(path)
    assert len(variants) == 10
    cl = next(v for v in variants if v.name == "loglo_pod_bandcurr_cl")
    assert cl.overrides["BAND_CURRICULUM"] == "true"
    assert cl.overrides["BAND_CURRICULUM_MODE"] == "convergence"
    assert cl.overrides["LOSS_BAND_BALANCED_WEIGHT"] == "0.5"
    assert cl.overrides["LOSS_RADIAL_WEIGHT"] == "0.25"
    assert cl.overrides["EARLY_STOP_PATIENCE"] == "140"
