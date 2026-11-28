# pod_readout.py
"""POD-DeepONet readout: branch predicts POD coefficients, trunk is fixed basis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class RecorderPODDeepONetHeadXT(nn.Module):
    """POD-DeepONet readout on frequency.

    ``shared`` layout (modes ``(K,F)``, mean ``(F,)``): the branch runs at every
    column. The 21 stations are collocation points in the loss, not the output
    grid. ``TF[b,x,f] = mean[f] + sum_k coeff[b,x,k] * mode[k,f]``.

    ``per_recorder`` layout (legacy modes ``(R,K,F)``, mean ``(R,F)``): branch
    and basis stay at the labeled slots; other x are zero.
    """

    def __init__(
        self,
        branch_in_channels: int,
        nx: int,
        n_freq: int,
        recorder_x: np.ndarray,
        pod_modes: np.ndarray,
        pod_mean: np.ndarray,
        branch_hidden: int | None = None,
        branch_mode: str = "surface",
        output_space: str = "linear",
        log_clamp: float = 20.0,
    ):
        super().__init__()
        if output_space not in ("linear", "log"):
            raise ValueError(f"output_space must be linear or log, got {output_space!r}")
        self.nx = nx
        self.n_freq = n_freq
        self.branch_mode = branch_mode
        self.output_space = output_space
        self.log_clamp = float(log_clamp)
        self.shared = pod_modes.ndim == 2
        self.n_modes = int(pod_modes.shape[0] if self.shared else pod_modes.shape[1])
        if self.shared:
            if pod_mean.ndim != 1 or pod_modes.shape[-1] != n_freq:
                raise ValueError(
                    f"shared POD expects modes (K,F) and mean (F,), got {pod_modes.shape} {pod_mean.shape}"
                )
        elif pod_modes.ndim != 3 or pod_mean.ndim != 2:
            raise ValueError(
                f"per-recorder POD expects modes (R,K,F) and mean (R,F), got {pod_modes.shape} {pod_mean.shape}"
            )
        self.register_buffer(
            "_recorder_x",
            torch.as_tensor(recorder_x, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_pod_modes",
            torch.as_tensor(pod_modes, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_pod_mean",
            torch.as_tensor(pod_mean, dtype=torch.float32),
            persistent=False,
        )

        hidden = branch_hidden or max(branch_in_channels, self.n_modes * 2)
        self.branch = nn.Sequential(
            nn.Linear(branch_in_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.n_modes),
        )

    def _select_x(self, field: torch.Tensor, columns: torch.Tensor) -> torch.Tensor:
        if self.branch_mode == "depth":
            return field.index_select(3, columns).mean(dim=2).permute(0, 2, 1)
        return field[:, :, 0, :].index_select(2, columns).permute(0, 2, 1)

    def _branch_features(
        self, x_global: torch.Tensor, x_local: torch.Tensor, columns: torch.Tensor | None = None
    ) -> torch.Tensor:
        if columns is None:
            columns = (
                torch.arange(self.nx, device=x_global.device)
                if self.shared
                else self._recorder_x
            )
        return torch.cat(
            [self._select_x(x_global, columns), self._select_x(x_local, columns)], dim=-1
        )

    def _scatter_recorders(self, tf_rec: torch.Tensor) -> torch.Tensor:
        out = tf_rec.new_zeros(tf_rec.shape[0], self.nx, self.n_freq)
        out.index_copy_(1, self._recorder_x, tf_rec)
        return out

    def _reconstruct(self, coeffs: torch.Tensor) -> torch.Tensor:
        if self.shared:
            tf = self._pod_mean + torch.einsum("bxk,kf->bxf", coeffs, self._pod_modes)
        else:
            tf = self._pod_mean.unsqueeze(0) + torch.einsum(
                "brk,rkf->brf", coeffs, self._pod_modes
            )
        if self.output_space == "log":
            # Unclamped exp overflows to Inf when a batch of coefficients blows
            # up; the rest of the epoch then trains on NaNs. Physical |TF| and
            # TF/TF_1D both live well inside e^±20.
            return torch.exp(tf.clamp(-self.log_clamp, self.log_clamp))
        return tf

    def forward(self, x_global: torch.Tensor, x_local: torch.Tensor) -> torch.Tensor:
        feats = self._branch_features(x_global, x_local)
        tf = self._reconstruct(self.branch(feats))
        return tf if self.shared else self._scatter_recorders(tf)

    def predict_recorders(
        self, x_global: torch.Tensor, x_local: torch.Tensor
    ) -> torch.Tensor:
        feats = self._branch_features(x_global, x_local, self._recorder_x)
        return self._reconstruct(self.branch(feats))


def load_pod_basis(
    pod_modes_path: Path, pod_mean_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    modes = np.load(pod_modes_path)
    mean = np.load(pod_mean_path)
    if modes.ndim == 2:
        if mean.ndim != 1:
            raise ValueError(f"shared pod_mean must be (F,), got {mean.shape}")
        return modes.astype(np.float32), mean.astype(np.float32)
    if modes.ndim != 3:
        raise ValueError(f"pod_modes must be (K,F) or (R,K,F), got {modes.shape}")
    if mean.ndim != 2:
        raise ValueError(f"per-recorder pod_mean must be (R,F), got {mean.shape}")
    return modes.astype(np.float32), mean.astype(np.float32)
