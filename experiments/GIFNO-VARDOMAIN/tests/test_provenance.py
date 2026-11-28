"""Checkpoint stamps stay stable and the test scorer refuses a missing checkpoint."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from provenance import stamp_matches, weight_sha256  # noqa: E402


def test_weight_hash_sorts_keys_and_includes_complex_buffers():
    first = {
        "b": torch.zeros(2, dtype=torch.float32),
        "a": torch.tensor([1.0, 2.0]),
        "c": torch.ones(1, 2, dtype=torch.complex64),
    }
    second = {"c": first["c"].clone(), "a": first["a"].clone(), "b": first["b"].clone()}
    assert weight_sha256(first) == weight_sha256(second)
    second["a"] = second["a"] + 1
    assert weight_sha256(first) != weight_sha256(second)


def test_discarded_or_different_dtype_does_not_match():
    expected = {
        "checkpoint": "/ckpt",
        "weight_sha256": "abc",
        "autocast": True,
        "inference_dtype": "bfloat16",
    }
    assert stamp_matches(dict(expected), expected)
    assert not stamp_matches({**expected, "discarded": True}, expected)
    assert not stamp_matches({**expected, "inference_dtype": "float32", "autocast": False}, expected)


def test_score_test_methods_requires_checkpoint():
    script = ROOT / "score_test_methods.py"
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert done.returncode != 0
    assert "checkpoint" in done.stderr.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
