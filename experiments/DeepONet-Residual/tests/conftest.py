"""Pytest setup for DeepONet-Residual (dummy GIFNO_DATA_ROOT, experiment on path)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_EXP_DIR = Path(__file__).resolve().parents[1]
_DUMMY = _EXP_DIR / "dummy_data"
_DUMMY.mkdir(parents=True, exist_ok=True)

os.environ["GIFNO_DATA_ROOT"] = str(_DUMMY)
# Drop Box OOD env so tests hit dummy_data/ood_*
os.environ.pop("GIFNO_OOD_DIPPING", None)
os.environ.pop("GIFNO_OOD_THREE_LAYER", None)


def _put_experiment_first() -> None:
    while str(_EXP_DIR) in sys.path:
        sys.path.remove(str(_EXP_DIR))
    sys.path.insert(0, str(_EXP_DIR))


_put_experiment_first()
import config as _DN_CONFIG  # noqa: E402
import data as _DN_DATA  # noqa: E402
import features as _DN_FEATURES  # noqa: E402
import haskell_baseline as _DN_HASKELL  # noqa: E402
import model as _DN_MODEL  # noqa: E402
import residual_target as _DN_RESIDUAL_TARGET  # noqa: E402
import train as _DN_TRAIN  # noqa: E402

_BOUND = {
    "config": _DN_CONFIG,
    "data": _DN_DATA,
    "features": _DN_FEATURES,
    "haskell_baseline": _DN_HASKELL,
    "model": _DN_MODEL,
    "residual_target": _DN_RESIDUAL_TARGET,
    "train": _DN_TRAIN,
}


@pytest.fixture(autouse=True)
def _restore_deeponet_config():
    """Keep this package's modules after sibling experiment conftests overwrite them."""
    _put_experiment_first()
    sys.modules.update(_BOUND)
    yield
