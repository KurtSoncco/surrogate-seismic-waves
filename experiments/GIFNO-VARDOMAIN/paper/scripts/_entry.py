"""Put the paper scripts on the path. Each leaf calls this before other local imports."""

from __future__ import annotations

import sys
from pathlib import Path


def enter(start: Path | None = None):
    here = Path(start or __file__).resolve()
    for parent in here.parents:
        if (parent / "scripts" / "paths.py").is_file() and parent.name == "paper":
            scripts = parent / "scripts"
            if str(scripts) not in sys.path:
                sys.path.insert(0, str(scripts))
            break
    else:
        raise RuntimeError("paper/scripts/paths.py not found")
    from paths import add_paths

    return add_paths()
