# tests/conftest.py
"""Pytest setup for GIFNO-FDO."""

import os
import sys
from pathlib import Path

_FDO_DIR = Path(__file__).resolve().parents[1]
_GIFNO_DIR = _FDO_DIR.parent / "GIFNO"
_DUMMY = _GIFNO_DIR / "dummy_data"

os.environ.setdefault("GIFNO_DATA_ROOT", str(_DUMMY))
sys.path.insert(0, str(_FDO_DIR))
sys.path.insert(1, str(_GIFNO_DIR))

import config  # noqa: E402, F401

config.setup_import_paths()
