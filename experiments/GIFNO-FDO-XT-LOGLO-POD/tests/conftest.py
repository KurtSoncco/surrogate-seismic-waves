# tests/conftest.py
"""Pytest setup for GIFNO-FDO-XT-LOGLO-POD."""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_EXP_DIR = Path(__file__).resolve().parents[1]
_GIFNO_DIR = _EXP_DIR.parent / "GIFNO"
_DUMMY = _GIFNO_DIR / "dummy_data"

_SHARED_MODULES = (
    "config",
    "data_loader",
    "losses",
    "metrics",
    "evaluate",
    "model",
    "train",
    "spectral_layers",
    "xt_readout",
    "pod_readout",
    "radial_spectral_loss",
)

os.environ.setdefault("GIFNO_DATA_ROOT", str(_DUMMY))


def _load_config_module(config_path: Path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config from {config_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["config"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_dummy_pod(config_mod) -> None:
    rec = config_mod.recorder_x_indices()
    n_freq = config_mod.N_FREQ
    k = config_mod.POD_NUM_MODES
    modes = np.random.randn(len(rec), k, n_freq).astype(np.float32)
    mean = np.random.randn(len(rec), n_freq).astype(np.float32)
    config_mod.POD_MODES_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(config_mod.POD_MODES_PATH, modes)
    np.save(config_mod.POD_MEAN_PATH, mean)


def _activate_imports() -> None:
    for name in _SHARED_MODULES:
        sys.modules.pop(name, None)
    for path in (str(_EXP_DIR), str(_GIFNO_DIR)):
        while path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, str(_EXP_DIR))
    sys.path.insert(1, str(_GIFNO_DIR))
    config = _load_config_module(_EXP_DIR / "config.py")
    config.setup_import_paths()
    _write_dummy_pod(config)
    model_spec = importlib.util.spec_from_file_location("model", _EXP_DIR / "model.py")
    if model_spec is None or model_spec.loader is None:
        raise ImportError(f"Cannot load model from {_EXP_DIR}")
    model_mod = importlib.util.module_from_spec(model_spec)
    sys.modules["model"] = model_mod
    model_spec.loader.exec_module(model_mod)


_activate_imports()


@pytest.fixture(autouse=True)
def _isolate_imports():
    _activate_imports()
    yield
