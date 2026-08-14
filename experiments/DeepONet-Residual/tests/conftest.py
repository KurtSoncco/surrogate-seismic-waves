"""Pytest setup for DeepONet-Residual."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_EXP_DIR = Path(__file__).resolve().parents[1]
_RES_DIR = _EXP_DIR.parent / "Residual"
_SHARED = (
    "config",
    "select_indices",
    "eval_ood",
    "data",
    "model",
    "train",
    "residual_signed",
    "haskell_baseline",
    "residual_target",
    "features",
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _activate() -> None:
    for name in _SHARED:
        sys.modules.pop(name, None)
    for path in (str(_EXP_DIR), str(_RES_DIR)):
        while path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, str(_EXP_DIR))
    sys.path.insert(1, str(_RES_DIR))
    _load("config", _EXP_DIR / "config.py")


_activate()


@pytest.fixture(autouse=True)
def _isolate_imports():
    _activate()
    yield
