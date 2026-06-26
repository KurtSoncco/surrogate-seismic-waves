# curriculum.py
"""Closed-loop, convergence-gated frequency-band curriculum controller.

Phases activate bands one at a time (low -> low+mid -> low+mid+high). The
controller advances to the next phase only when the monitored metric (a
band-balanced val metric) plateaus, so the curriculum is driven by optimization
progress rather than a fixed epoch schedule. On each advance the caller is
expected to warm-restart the optimizer/LR (the controller only signals it).
"""

from __future__ import annotations

from typing import Tuple


class BandCurriculumController:
    """Convergence-gated phase controller for band weights.

    Phase ``p`` fully activates bands ``0..p-1`` (weight 1), ramps the newly
    activated band ``p`` from ``floor`` to 1 over ``ramp_epochs``, and holds the
    not-yet-activated bands at ``floor`` (band 0 is active from the start). The
    controller advances when the monitored metric fails to improve by
    ``min_delta`` for ``patience`` epochs.
    """

    def __init__(
        self,
        n_bands: int = 3,
        floor: float = 0.25,
        patience: int = 30,
        min_delta: float = 1e-4,
        ramp_epochs: int = 10,
    ):
        self.n_bands = n_bands
        self.floor = floor
        self.patience = patience
        self.min_delta = min_delta
        self.ramp_epochs = max(1, ramp_epochs)
        self.phase = 0
        self.best = float("inf")
        self.wait = 0
        self.epochs_in_phase = 0

    @property
    def is_final_phase(self) -> bool:
        return self.phase >= self.n_bands - 1

    def current_weights(self) -> Tuple[float, ...]:
        """Per-band weights for the current phase/epoch (band 0 = low)."""
        weights = []
        for b in range(self.n_bands):
            if b < self.phase:
                weights.append(1.0)
            elif b == self.phase:
                if self.phase == 0:
                    weights.append(1.0)
                else:
                    frac = min(1.0, self.epochs_in_phase / self.ramp_epochs)
                    weights.append(self.floor + (1.0 - self.floor) * frac)
            else:
                weights.append(self.floor)
        return tuple(weights)

    def step(self, metric: float) -> bool:
        """Advance the epoch counter and the phase if the metric plateaued.

        Returns True iff the phase advanced this call (the caller should then
        warm-restart the optimizer/LR). No advance happens in the final phase.
        """
        self.epochs_in_phase += 1
        if self.is_final_phase:
            return False
        if metric < self.best - self.min_delta:
            self.best = metric
            self.wait = 0
        else:
            self.wait += 1
        if self.wait >= self.patience:
            self.phase += 1
            self.best = float("inf")
            self.wait = 0
            self.epochs_in_phase = 0
            return True
        return False
