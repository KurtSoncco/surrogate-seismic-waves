"""Repository paths for the vardomain paper analyses."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
PAPER = SCRIPTS.parent
VARDOMAIN = PAPER.parent
REPO = VARDOMAIN.parents[1]
CASE7A = REPO / "experiments" / "case7a"
CACHE = REPO / "data" / "vardomain" / "cache"
MANIFEST = VARDOMAIN / "sections" / "manifold150.csv"
LABELS = REPO / "data" / "vardomain" / "manifold_labels" / "stride5"
STAGE1 = REPO / "checkpoints" / "vardomain" / "early_width" / "stage1"
STAGE2 = REPO / "checkpoints" / "vardomain" / "early_width" / "stage2_rise50"
STAGE0_RESIDUAL = REPO / "checkpoints" / "vardomain" / "stage0_residual_s0"
STAGE0_DIRECT = REPO / "checkpoints" / "vardomain" / "stage0_direct_s0"
FIG = VARDOMAIN / "figures"
DUMP = PAPER / "dump"


def current_pass(frame) -> str:
    """The reported model. The four-checkpoint mean replaces stage 2 when it has been scored."""
    if "pass" in frame.columns and (frame["pass"].astype(str) == "ensemble").any():
        return "ensemble"
    return "stage2"
FLOOR_JSON = PAPER / "04_where_correction_hurts" / "consistency_floor" / "results.json"


def add_paths() -> tuple[Path, Path, Path]:
    """Put the analysis scripts, the vardomain package, and Case 7a on ``sys.path``."""
    for path in (SCRIPTS, VARDOMAIN, CASE7A, REPO):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return REPO, VARDOMAIN, PAPER
