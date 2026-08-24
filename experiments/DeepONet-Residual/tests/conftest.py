"""Pytest setup for DeepONet-Residual (dummy GIFNO_DATA_ROOT, experiment on path)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_EXP_DIR = Path(__file__).resolve().parents[1]
_RES_DIR = _EXP_DIR.parent / "Residual"
_DUMMY = _EXP_DIR / "dummy_data"
_DUMMY.mkdir(parents=True, exist_ok=True)

os.environ["GIFNO_DATA_ROOT"] = str(_DUMMY)
# Drop Box OOD env so tests hit dummy_data/ood_*
os.environ.pop("GIFNO_OOD_DIPPING", None)
os.environ.pop("GIFNO_OOD_THREE_LAYER", None)


def _put_experiment_first() -> None:
    for path in (str(_EXP_DIR), str(_RES_DIR)):
        while path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, str(_EXP_DIR))
    sys.path.insert(1, str(_RES_DIR))


_put_experiment_first()
import config as _DN_CONFIG  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_deeponet_config():
    """Keep this package's config after sibling experiment conftests overwrite it."""
    _put_experiment_first()
    sys.modules["config"] = _DN_CONFIG
    yield
