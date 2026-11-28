"""Render the SDEE results figures into ``paper/figures``."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fig01_training  # noqa: E402
import fig02_residual  # noqa: E402
import fig03_metrics  # noqa: E402
import fig04_sota  # noqa: E402
import fig05_iid  # noqa: E402
import fig06_ood  # noqa: E402
import fig07_leftover  # noqa: E402
import fig_graphical_abstract  # noqa: E402


def main() -> None:
    fig01_training.main()
    fig02_residual.main()
    fig03_metrics.main()
    fig04_sota.main()
    print("fig05", fig05_iid.main())
    print("fig06", fig06_ood.main())
    fig07_leftover.main()
    fig_graphical_abstract.main()


if __name__ == "__main__":
    main()
