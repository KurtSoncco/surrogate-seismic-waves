# tests/conftest.py
"""Pytest setup for GIFNO-FDO-LG."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_LG_DIR = Path(__file__).resolve().parents[1]
_GIFNO_DIR = _LG_DIR.parent / "GIFNO"
_XT_DIR = _LG_DIR.parent / "GIFNO-FDO-XT"
_DUMMY = _GIFNO_DIR / "dummy_data"

_SHARED_MODULES = (
    "config",
    "sweep_launch",
    "data_loader",
    "losses",
    "metrics",
    "evaluate",
    "model",
    "train",
    "transfer",
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


def _activate_lg_imports() -> None:
    for name in _SHARED_MODULES:
        sys.modules.pop(name, None)
    for path in (str(_LG_DIR), str(_XT_DIR), str(_GIFNO_DIR)):
        while path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, str(_LG_DIR))
    sys.path.insert(1, str(_GIFNO_DIR))
    config = _load_config_module(_LG_DIR / "config.py")
    config.setup_import_paths()


_activate_lg_imports()


@pytest.fixture(autouse=True)
def _isolate_lg_imports():
    _activate_lg_imports()
    yield
