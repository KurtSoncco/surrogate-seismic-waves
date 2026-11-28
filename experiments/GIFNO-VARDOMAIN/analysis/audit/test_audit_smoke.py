"""Import every Phase 1 audit script and check the harness reductions."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_census  # noqa: F401
import audit_coverage  # noqa: F401
import audit_errors  # noqa: F401
import audit_floors  # noqa: F401
import audit_representation  # noqa: F401
import audit_splits  # noqa: F401
import common


def test_self_check() -> None:
    common.self_check()


if __name__ == "__main__":
    test_self_check()
    print("smoke import ok")
