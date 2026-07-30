# config.py
"""TH-FNO: H_1D(trend) + gated shallow FNO/DeepONet residual (AGENTS.md)."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

EXPERIMENT_DIR = Path(__file__).resolve().parent
_GIFNO_DIR = EXPERIMENT_DIR.parent / "GIFNO"
_XT_DIR = EXPERIMENT_DIR.parent / "GIFNO-FDO-XT"

_xt_spec = importlib.util.spec_from_file_location(
    "gifno_fdo_xt_config_for_thfno", _XT_DIR / "config.py"
)
if _xt_spec is None or _xt_spec.loader is None:
    raise ImportError(f"Cannot load XT config from {_XT_DIR / 'config.py'}")
_xt_config = importlib.util.module_from_spec(_xt_spec)
if str(_XT_DIR) not in sys.path:
    sys.path.insert(0, str(_XT_DIR))
_xt_spec.loader.exec_module(_xt_config)

DATA_ROOT = _xt_config.DATA_ROOT
H5_DIR = _xt_config.H5_DIR
TF_RESULTS_DIR = _xt_config.TF_RESULTS_DIR
TF_PER_SAMPLE_PATH = _xt_config.TF_PER_SAMPLE_PATH
TF_FREQ_PATH = _xt_config.TF_FREQ_PATH
MANIFEST_PATH = _xt_config.MANIFEST_PATH

MODEL_SAVE_DIR = Path(
    os.environ.get("THFNO_MODEL_DIR", EXPERIMENT_DIR / "checkpoints" / "th_fno")
)
RESULTS_SAVE_DIR = Path(
    os.environ.get("THFNO_RESULTS_DIR", EXPERIMENT_DIR / "results")
)
MODEL_SAVE_PATH = MODEL_SAVE_DIR / "best_model.pt"
DIAGNOSTICS_DIR = RESULTS_SAVE_DIR / "diagnostics"
for d in (MODEL_SAVE_DIR, RESULTS_SAVE_DIR, DIAGNOSTICS_DIR):
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

# Grid / recorders — central variability strip of the full (nz, 1500) mesh
# Train IID: NX=500 = columns [BC_WIDTH : BC_WIDTH+NX] = [500:1000].
# Alternate NX (wider/narrower central crop) = strip-extent OOD probe (AGENTS §0.1).
NZ_MAX: int = _xt_config.NZ_MAX
NX: int = _xt_config.NX
NX_FULL: int = _xt_config.NX_FULL
BC_WIDTH: int = _xt_config.BC_WIDTH
LX_VARIABILITY: int = _xt_config.LX_VARIABILITY
X_SLICE_START: int = _xt_config.X_SLICE_START
X_SLICE_END: int = _xt_config.X_SLICE_END
DX: float = _xt_config.DX
DZ: float = _xt_config.DZ
N_LATERAL: int = _xt_config.N_LATERAL
NODES_EACH_SIDE: int = _xt_config.NODES_EACH_SIDE
LATERAL_SPACING_M: float = _xt_config.LATERAL_SPACING_M
N_FREQ: int = _xt_config.N_FREQ
NORMALIZE_VS_SURFACE: bool = True
NORMALIZE_ZETA: bool = True
VS_NORM_EPS: float = 1e-6
ZETA_NORM_EPS: float = 1e-12

# Context: Vs, zeta, x, z, dip, impedance_grad
IN_CHANNELS: int = 6
# Shallow FNO (AGENTS §2) — field encoder on (NZ_MAX, NX=500) strip
LATENT_CHANNELS: int = 48
FNO_MODES: Tuple[int, int] = (16, 16)
NUM_FNO_LAYERS: int = 2
DEEPONET_LATENT_DIM: int = 64
BRANCH_MODE: str = "surface"
TRUNK_HIDDEN: int = 128
TRUNK_LAYERS: int = 3
X_COORD_MODE: str = "normalized"
OUTPUT_ACTIVATION: str = "none"
USE_FOURIER_FEATURES: bool = True
FOURIER_FREQS: int = 8
# Physics latents to head (interim without KL): CoV, rH, aHV (+ optional seed emb)
PHYSICS_LATENT_DIM: int = 3
USE_PHYSICS_HEAD: bool = True
# Direct |TF|(x, log f) — D2 FAIL pivot (AGENTS §2 / §4).
# residual stack remains for ablations if D2 is revisited.
PREDICT_MODE: str = os.environ.get("THFNO_PREDICT_MODE", "direct")  # direct | residual
RESIDUAL_MODE: str = os.environ.get(
    "THFNO_RESIDUAL_MODE", "log_mult"
)  # additive | log_mult (only if PREDICT_MODE=residual)
# Soft bound on g·Δ for log_mult: Δ_eff = C * tanh(g·Δ / C) ⇒ residual ∈ [1/3, 3]×.
# Hard clamp of ±5 allowed exp≈e^5 blow-ups (A/B #1). Override via THFNO_LOG_DELTA_C.
LOG_DELTA_C: float = float(
    os.environ.get("THFNO_LOG_DELTA_C", str(float(np.log(3.0))))
)
# Backward-compat alias used by older call sites / checkpoints configs
LOG_DELTA_CLAMP: float = LOG_DELTA_C
# Zero-init residual DeepONet branch final layer so exp(g·Δ)=1 at step 0.
ZERO_INIT_RESIDUAL_HEAD: bool = os.environ.get(
    "THFNO_ZERO_INIT_RESIDUAL", "1"
).strip().lower() not in ("0", "false", "no")
WANDB_RUN_NAME: str = os.environ.get("THFNO_WANDB_RUN_NAME", "th_fno_direct")

GATE_COV_REF: float = 0.1
GATE_DIP_REF: float = 0.05
DEFAULT_RHO: float = 2000.0
DEFAULT_XI_TREND: float = 0.05
TF_LOG_EPS: float = 1e-3  # amplitude floor for ln|TF|
# Resonance calibration: median(f_truth / f_H1D_uncalibrated). H_eff = H / scale.
# 1.0 = off. Override via THFNO_TREND_FREQ_SCALE or diagnostics/calibrate_*.
# KEEP: GIFNO median ≈ 0.938 (Session N+1).
TREND_FREQ_SCALE: float = float(os.environ.get("THFNO_TREND_FREQ_SCALE", "0.938"))

# Loss (AGENTS §3) — single amplitude domain
# Prefer raw |TF| SmoothL1 (linear). Log is available via AMPLITUDE_DOMAIN=log.
AMPLITUDE_DOMAIN: str = os.environ.get("THFNO_AMPLITUDE_DOMAIN", "linear")  # linear | log
LOSS_SMOOTH_L1_WEIGHT: float = 1.0
SMOOTH_L1_BETA: float = 1.0
LOSS_PEAK_WEIGHT: float = 0.25
LOSS_SPEC_WEIGHT: float = 0.05
LOSS_REL_WEIGHT: float = 0.0  # reported only / optional aux
# Per-term running-magnitude normalization BEFORE λ (Session N+1 C3).
# Without this, ∂/∂logf is ~95% of loss magnitude and base/peak barely train.
LOSS_TERM_NORM: bool = os.environ.get(
    "THFNO_LOSS_TERM_NORM", "1"
).strip().lower() not in ("0", "false", "no")
LOSS_TERM_NORM_MOMENTUM: float = float(
    os.environ.get("THFNO_LOSS_TERM_NORM_MOMENTUM", "0.99")
)
LOSS_TERM_NORM_EPS: float = 1e-6

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
NUM_EPOCHS: int = 200
BATCH_SIZE: int = 8
TRAIN_SPLIT: float = 0.7
VAL_SPLIT: float = 0.15
TEST_SPLIT: float = 0.15
SEED: int = 42
EARLY_STOP_PATIENCE: int = 40
GRAD_CLIP_NORM: float = 1.0
NUM_WORKERS: int = 0
AMSGRAD: bool = True
EVAL_N_TF_CURVES: int = 4
EVAL_N_WORST_TF: int = 3
EVAL_TF_COLLECT: int = 64
WANDB_PROJECT: str = os.environ.get("WANDB_PROJECT", "th_fno")

RV_ROOT = Path(
    os.environ.get(
        "RV_ROOT", Path.home() / "seiskit" / "comparison" / "Response_Variability"
    )
)
RV_H5_DIR = Path(os.environ.get("RV_H5_DIR", RV_ROOT / "results" / "h5"))
CAPABILITY_ROOT = Path(
    os.environ.get(
        "SEISKIT_NO_EXPERIMENTS",
        Path.home() / "seiskit" / "neural-operator" / "experiments",
    )
)

# Verified shape constants (AGENTS §8) — update if live assert fails
SHAPE_N_LATERAL: int = 21
SHAPE_N_FREQ: int = 1000
SHAPE_NX_STRIP: int = 500  # train IID central variability window
SHAPE_NZ_MAX: int = 128
SHAPE_NX_FULL: int = 1500  # full OpenSees mesh before crop
SHAPE_RV_SOBOL: int = 64
SHAPE_RV_RF_SEEDS: int = 40


def central_strip_slice(nx_strip: int | None = None) -> slice:
    """Columns of the full (nz, NX_FULL) mesh used as the model strip.

    Default is the train IID central-500 window. Pass another ``nx_strip``
    (still centered) for strip-extent OOD evals — see AGENTS §0.1 / §5.
    """
    w = int(nx_strip if nx_strip is not None else NX)
    if w > NX_FULL:
        raise ValueError(f"nx_strip={w} exceeds NX_FULL={NX_FULL}")
    start = (NX_FULL - w) // 2
    return slice(start, start + w)


def recorder_x_indices(
    nx: int = NX,
    nodes_each_side: int = NODES_EACH_SIDE,
    spacing_m: float = LATERAL_SPACING_M,
    dx: float = DX,
) -> np.ndarray:
    return _xt_config.recorder_x_indices(
        nx=nx, nodes_each_side=nodes_each_side, spacing_m=spacing_m, dx=dx
    )


def recorder_x_trunk_coords(recorder_x, nx: int = NX, mode: str = X_COORD_MODE):
    return _xt_config.recorder_x_trunk_coords(recorder_x, nx=nx, mode=mode)


def domain_half_width_m(nx: int = NX, dx: float = DX) -> float:
    return _xt_config.domain_half_width_m(nx=nx, dx=dx)


def setup_import_paths() -> None:
    for p in (str(_XT_DIR), str(_GIFNO_DIR), str(EXPERIMENT_DIR)):
        while p in sys.path:
            sys.path.remove(p)
    sys.path[:0] = [str(EXPERIMENT_DIR), str(_GIFNO_DIR), str(_XT_DIR)]
    exp = EXPERIMENT_DIR.resolve()
    for name in (
        "model",
        "train",
        "rv_dataset",
        "haskell_baseline",
        "context_features",
        "eval_rv",
        "gifno_dataset",
        "losses_th",
        "tf_plots",
    ):
        mod = sys.modules.get(name)
        if mod is None or not getattr(mod, "__file__", None):
            continue
        f = Path(mod.__file__).resolve()
        if exp not in f.parents:
            sys.modules.pop(name, None)
