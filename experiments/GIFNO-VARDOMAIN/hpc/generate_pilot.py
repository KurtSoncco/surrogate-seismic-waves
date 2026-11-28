#!/usr/bin/env python3
"""Write the Phase 4 pilot CSV next to this directory. Does not submit a job.

    python generate_pilot.py
    python generate_pilot.py --out pilot_g4.csv --timing-out pilot_record_timing.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parents[2] / "case7a"
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

from pilot_manifest import main  # noqa: E402

if __name__ == "__main__":
    main()
