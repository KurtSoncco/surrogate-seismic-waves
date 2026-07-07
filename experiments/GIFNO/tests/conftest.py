"""Pytest fixtures for GIFNO — use local dummy data, never Box mounts."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_GIFNO_DIR = Path(__file__).resolve().parents[1]
_DUMMY_DATA_ROOT = _GIFNO_DIR / "dummy_data"

_SHARED_MODULES = (
    "config",
    "sweep_launch",
    "data_loader",
    "losses",
    "metrics",
    "evaluate",
    "model",
    "train",
)

os.environ.setdefault("GIFNO_DATA_ROOT", str(_DUMMY_DATA_ROOT))


def _load_config_module(config_path: Path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config from {config_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["config"] = mod
    spec.loader.exec_module(mod)
    return mod


def _activate_gifno_imports() -> None:
    for name in _SHARED_MODULES:
        sys.modules.pop(name, None)
    while str(_GIFNO_DIR) in sys.path:
        sys.path.remove(str(_GIFNO_DIR))
    sys.path.insert(0, str(_GIFNO_DIR))
    _load_config_module(_GIFNO_DIR / "config.py")


_activate_gifno_imports()


@pytest.fixture(autouse=True)
def _isolate_gifno_imports():
    _activate_gifno_imports()
    yield
