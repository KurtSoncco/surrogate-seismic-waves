#!/usr/bin/env python3
"""The learning-curve figure is not redrawn from the old history.

``appendix/learning_curve/history.csv`` is the pre-harness curve. It lives in
``superseded/`` and is not a frozen source. Losses from that file are not quoted.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "Not redrawn. appendix/learning_curve/history.csv is in superseded/ "
        "and is the old definition. There is no frozen history to plot."
    )


if __name__ == "__main__":
    main()
