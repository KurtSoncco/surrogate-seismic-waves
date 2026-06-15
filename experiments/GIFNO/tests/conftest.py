"""Pytest fixtures for GIFNO — use local dummy data, never Box mounts."""

from __future__ import annotations

import os
from pathlib import Path

_DUMMY_DATA_ROOT = Path(__file__).resolve().parents[1] / "dummy_data"
os.environ.setdefault("GIFNO_DATA_ROOT", str(_DUMMY_DATA_ROOT))
