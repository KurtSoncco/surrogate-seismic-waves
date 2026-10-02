"""Corner IS split tagging (no H5 / checkpoint)."""

from __future__ import annotations

import pandas as pd
import pytest

from ood_io import corpus_root, default_ood_roots
from response_variability.eval_corner_is import split_corner_summaries


def test_corner_is_is_an_ood_root():
    roots = default_ood_roots()
    assert "corner_is" in roots
    assert corpus_root("corner") == corpus_root("corner_is")


def test_split_corner_summaries_separates_held_out():
    df = pd.DataFrame(
        {
            "method": ["GINO"] * 5,
            "pearson": [0.9, 0.8, 0.7, 0.6, 0.5],
            "held_out": [0, 0, 0, 1, 1],
        }
    )
    slices = split_corner_summaries(df)
    assert len(slices["corner_train"]) == 3
    assert len(slices["corner_held"]) == 2
    with pytest.raises(KeyError):
        split_corner_summaries(df.drop(columns=["held_out"]))
