"""Pytest bootstrap for TH-FNO."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
_GIFNO = _EXP.parent / "GIFNO"
_XT = _EXP.parent / "GIFNO-FDO-XT"

for name in ("config", "model", "train", "haskell_baseline", "context_features", "losses_th"):
    sys.modules.pop(name, None)
for p in (str(_XT), str(_GIFNO), str(_EXP)):
    while p in sys.path:
        sys.path.remove(p)
sys.path[:0] = [str(_EXP), str(_GIFNO), str(_XT)]

_spec = importlib.util.spec_from_file_location("config", _EXP / "config.py")
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
sys.modules["config"] = _mod
_spec.loader.exec_module(_mod)
