# pod_readout.py
"""POD-DeepONet readout: branch predicts POD coefficients, trunk is fixed basis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class RecorderPODDeepONetHeadXT(nn.Module):
    """
    Branch from concat(global, local) latent at recorders; fixed POD basis on frequency.

    TF_rec[b,r,f] = mean[r,f] + sum_k coeff[b,r,k] * mode[r,k,f]
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
    ):
        super().__init__()
        self.nx = nx
        self.n_freq = n_freq
        self.branch_mode = branch_mode
        self.n_modes = int(pod_modes.shape[1])
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

    def _branch_features(
        self, x_global: torch.Tensor, x_local: torch.Tensor
    ) -> torch.Tensor:
        """Concat global+local recorder features -> (B, R, 2C)."""
        if self.branch_mode == "depth":
            g = x_global.index_select(3, self._recorder_x).mean(dim=2).permute(0, 2, 1)
            local = (
                x_local.index_select(3, self._recorder_x).mean(dim=2).permute(0, 2, 1)
            )
        else:
            g = x_global[:, :, 0, :].index_select(2, self._recorder_x).permute(0, 2, 1)
            local = (
                x_local[:, :, 0, :].index_select(2, self._recorder_x).permute(0, 2, 1)
            )
        return torch.cat([g, local], dim=-1)

    def _scatter_recorders(self, tf_rec: torch.Tensor) -> torch.Tensor:
        out = tf_rec.new_zeros(tf_rec.shape[0], self.nx, self.n_freq)
        out.index_copy_(1, self._recorder_x, tf_rec)
        return out

    def forward(self, x_global: torch.Tensor, x_local: torch.Tensor) -> torch.Tensor:
        feats = self._branch_features(x_global, x_local)
        coeffs = self.branch(feats)
        tf_rec = self._pod_mean.unsqueeze(0) + torch.einsum(
            "brk,rkf->brf", coeffs, self._pod_modes
        )
        return self._scatter_recorders(tf_rec)

    def predict_recorders(
        self, x_global: torch.Tensor, x_local: torch.Tensor
    ) -> torch.Tensor:
        feats = self._branch_features(x_global, x_local)
        coeffs = self.branch(feats)
        return self._pod_mean.unsqueeze(0) + torch.einsum(
            "brk,rkf->brf", coeffs, self._pod_modes
        )


def load_pod_basis(
    pod_modes_path: Path, pod_mean_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    modes = np.load(pod_modes_path)
    mean = np.load(pod_mean_path)
    if modes.ndim != 3:
        raise ValueError(f"pod_modes must be (R,K,F), got {modes.shape}")
    if mean.ndim != 2:
        raise ValueError(f"pod_mean must be (R,F), got {mean.shape}")
    return modes.astype(np.float32), mean.astype(np.float32)
