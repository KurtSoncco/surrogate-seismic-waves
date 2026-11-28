"""Proposed gates for the compatibility-accuracy revision.

Written before any score in ``analysis/``. Each number is a PI proposal, not a
threshold fitted to a result. Phase 1.4 replaces the noise-floor placeholder
with ``max(FE floor, 1D-convention uncertainty)``.

``paper/scripts/metrics.py`` defines ``FLOOR_LOG_L1 = 0.014``. That constant is
not the noise floor. Nothing in this package reads it.
"""

from __future__ import annotations

# Phase 1.7 cites the training rule: bf16 and float32 agree below this
# validation log-L1 (``train.AGREE_FLOOR``). It is the placeholder until the
# measured floor exists. It is not ``metrics.FLOOR_LOG_L1``.
NOISE_FLOOR_PLACEHOLDER_LOG_L1 = 0.0136
NOISE_FLOOR_DEFINITION = "max(FE floor, 1D-convention uncertainty)"

# Phase 2.4 consistency proposals.
MIRROR_L1_FRACTION = 0.1
MONOTONE_SIGN_AGREEMENT = 0.80

# Phase 4, gate G4. Masters for TOST of the paired width difference.
TOST_DELTA_LOG_L1 = 0.02
TOST_POWER = 0.80
TOST_ALPHA = 0.05

# Phase 6.4. Stop if harm exceeds this share in any family, or a consistency
# test fails this many epochs. Accept only with at least this many seeds,
# an overall validation improvement larger than the bootstrap half-width,
# and no family worse by more than ``TOST_DELTA_LOG_L1``.
HARM_RATE_STOP = 0.05
CONSISTENCY_FAIL_EPOCHS = 3
MIN_TRAINING_SEEDS = 3

# Cluster-bootstrap tail probability. Matches ``bootstrap_ci`` and G4.
BOOTSTRAP_ALPHA = 0.05

# Gate G2. A positive Paper 2 headline has to beat baseline (b) on 0.4–10 Hz.
# The 0.1–0.4 Hz band is stated separately and is not part of that claim.
G2_POSITIVE_BANDS = ("mid", "high")


def noise_floor_log_l1() -> float:
    """Floor used for skill suppression and the dtype freeze.

    This returns the placeholder. Phase 1.4 is what replaces it with a
    measurement. Callers must not substitute ``metrics.FLOOR_LOG_L1``.
    """

    return float(NOISE_FLOOR_PLACEHOLDER_LOG_L1)
